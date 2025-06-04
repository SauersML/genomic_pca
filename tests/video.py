#!/usr/bin/env python3
import os
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# --- Configuration ---
BASE_SWEEP_OUTPUT_DIR = Path("eigensnp_sweeps_output")
SWEEP_SUMMARY_FILE = BASE_SWEEP_OUTPUT_DIR / "sweeps_summary.tsv"
PLOT_SCRIPT_PATH = Path("plot.py") # plot.py is in the same directory

ANIMATIONS_OUTPUT_DIR = BASE_SWEEP_OUTPUT_DIR / "animations_from_sweeps_parallel"
TEMP_FRAMES_DIR_ROOT = ANIMATIONS_OUTPUT_DIR / "_temp_frames_parallel"

PARAMS_TO_ANIMATE = [
    "eigensnp_components_per_block",
    "eigensnp_local_power_iter",
    "eigensnp_min_maf",
    "eigensnp_refine_passes",
]
FPS = 1
NUM_PARALLEL_PLOT_JOBS = max(1, cpu_count())

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, text=True, errors='ignore')
        logging.info("ffmpeg found.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("ffmpeg not found or not executable. MP4 generation will fail.")
        logging.error("Please install ffmpeg and ensure it's in your system PATH.")
        return False

def run_command_robust(command_list, working_dir=None, command_description="Command"):
    logging.debug(f"Executing: {' '.join(command_list)} in CWD: {working_dir or os.getcwd()}")
    try:
        process = subprocess.run(command_list, capture_output=True, text=True, check=True, cwd=working_dir, errors='ignore')
        if process.stdout:
            logging.debug(f"{command_description} STDOUT:\n{process.stdout.strip()}")
        if process.stderr:
            logging.debug(f"{command_description} STDERR:\n{process.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"{command_description} failed: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout: logging.error(f"STDOUT:\n{e.stdout.strip()}")
        if e.stderr: logging.error(f"STDERR:\n{e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error(f"Executable not found: {command_list[0]}")
        return False
    except Exception as e_gen:
        logging.error(f"An unexpected error occurred running {command_list[0]}: {e_gen}")
        return False


def to_numeric_sortable(value_str):
    try:
        return float(value_str)
    except ValueError:
        logging.warning(f"Could not convert '{value_str}' to float for sorting, keeping as string.")
        return str(value_str)

def generate_single_frame(task_info):
    run_id = task_info['run_id']
    swept_value = task_info['swept_value']
    run_output_dir_str = task_info['run_output_dir_str']
    target_frame_file = Path(task_info['target_frame_file']) # Convert back to Path
    param_short_name = task_info['param_short_name']
    frame_idx_display = task_info['frame_idx_display']
    total_frames_for_param = task_info['total_frames_for_param']

    run_output_dir = Path(run_output_dir_str)

    logging.info(f"Generating frame {frame_idx_display}/{total_frames_for_param} for {param_short_name}={swept_value} (Run ID: {run_id})")

    if not run_output_dir.is_dir():
        logging.warning(f"Run output directory not found: {run_output_dir}. Skipping frame for {run_id}.")
        return None

    expected_plot_output = run_output_dir / "pca.png"
    if expected_plot_output.exists():
        try:
            expected_plot_output.unlink()
        except OSError as e_rm:
            logging.warning(f"Could not remove pre-existing plot {expected_plot_output}: {e_rm}")

    # Ensure PLOT_SCRIPT_PATH and run_output_dir are absolute for robustness with subprocess
    plot_py_cmd = ["python", str(PLOT_SCRIPT_PATH.resolve()), str(run_output_dir.resolve())]

    if not run_command_robust(plot_py_cmd, command_description=f"plot.py for {run_id}"):
        logging.error(f"Failed to generate plot for {run_id}. Skipping frame.")
        return None

    if not expected_plot_output.exists():
        logging.error(f"Plot script ran but output {expected_plot_output} not found for {run_id}. Skipping frame.")
        return None

    try:
        target_frame_file.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        shutil.move(str(expected_plot_output), str(target_frame_file))
        logging.debug(f"Frame saved to {target_frame_file} for {run_id}")
        return str(target_frame_file)
    except Exception as e_mv:
        logging.error(f"Failed to move plot {expected_plot_output} to {target_frame_file} for {run_id}: {e_mv}")
        return None

def main():
    if not SWEEP_SUMMARY_FILE.exists():
        logging.error(f"Sweep summary file not found: {SWEEP_SUMMARY_FILE}")
        return
    if not PLOT_SCRIPT_PATH.exists():
        logging.error(f"Plotting script not found: {PLOT_SCRIPT_PATH}")
        return

    ffmpeg_available = check_ffmpeg()

    ANIMATIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_FRAMES_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    logging.info(f"Animations will be saved to: {ANIMATIONS_OUTPUT_DIR.resolve()}")
    logging.info(f"Temporary frames will be stored in: {TEMP_FRAMES_DIR_ROOT.resolve()}")
    logging.info(f"Using {NUM_PARALLEL_PLOT_JOBS} parallel processes for frame generation.")

    try:
        df_summary = pd.read_csv(SWEEP_SUMMARY_FILE, sep='\t', na_filter=False)
        logging.info(f"Loaded sweep summary with {len(df_summary)} records.")
    except Exception as e:
        logging.error(f"Failed to load sweep summary file '{SWEEP_SUMMARY_FILE}': {e}")
        return

    for param_internal_name in PARAMS_TO_ANIMATE:
        param_short_name = param_internal_name.replace("eigensnp_", "")
        logging.info(f"\n--- Processing animation for parameter: {param_short_name} ---")

        if "swept_param_name" not in df_summary.columns or "swept_param_value" not in df_summary.columns:
            logging.error(f"Missing 'swept_param_name' or 'swept_param_value' in {SWEEP_SUMMARY_FILE}")
            continue
        df_param_sweep = df_summary[df_summary["swept_param_name"] == param_internal_name].copy()

        if df_param_sweep.empty:
            logging.warning(f"No runs found for swept parameter: {param_internal_name}. Skipping.")
            continue

        df_param_sweep["sortable_value"] = df_param_sweep["swept_param_value"].apply(to_numeric_sortable)
        df_param_sweep.sort_values(by="sortable_value", inplace=True)

        temp_param_frames_dir = TEMP_FRAMES_DIR_ROOT / f"{param_short_name}_frames"
        temp_param_frames_dir.mkdir(exist_ok=True)

        tasks = []
        for idx, row in enumerate(df_param_sweep.itertuples(index=False)):
            # Pass paths as strings to worker for better pickling/compatibility
            tasks.append({
                'run_id': row.run_id,
                'run_output_dir_str': str(Path(row.output_dir)), # Ensure it's a string
                'swept_value': row.swept_param_value,
                'target_frame_file': str(temp_param_frames_dir / f"frame_{idx:04d}.png"),
                'param_short_name': param_short_name,
                'frame_idx_display': idx + 1,
                'total_frames_for_param': len(df_param_sweep)
            })

        generated_frame_paths = []
        if tasks:
            with Pool(processes=NUM_PARALLEL_PLOT_JOBS) as pool:
                # Using imap_unordered for progress bar, results will be filtered for None
                results_iterator = pool.imap_unordered(generate_single_frame, tasks)
                for result in tqdm(results_iterator, total=len(tasks), desc=f"Generating frames for {param_short_name}"):
                    if result:
                        generated_frame_paths.append(result)
        
        # Sort frame paths numerically by frame index in filename (though tasks were ordered, imap_unordered may not preserve it)
        # This is critical if filenames were not pre-determined based on sorted order.
        # Since target_frame_file names are pre-determined based on sorted order, simple list is fine if all succeed.
        # However, if some frames fail, we need to ensure the existing frames are correctly ordered.
        generated_frame_paths.sort()


        if len(generated_frame_paths) < 2:
            logging.warning(f"Less than 2 frames generated for {param_short_name}. Skipping MP4 creation.")
            if len(generated_frame_paths) == 1:
                 final_image_path = ANIMATIONS_OUTPUT_DIR / f"{param_short_name}_single_valid_frame.png"
                 shutil.copy(generated_frame_paths[0], final_image_path)
                 logging.info(f"Copied single available valid frame to {final_image_path}")
            # Clean up potentially non-empty temp_param_frames_dir if no MP4 is made
            if temp_param_frames_dir.exists():
                try:
                    shutil.rmtree(temp_param_frames_dir)
                except OSError as e_clean:
                    logging.warning(f"Could not clean up temp frames dir {temp_param_frames_dir} after failed animation: {e_clean}")
            continue

        if not ffmpeg_available:
            logging.error(f"ffmpeg not available. Cannot create MP4 for {param_short_name}. Frames are in {temp_param_frames_dir}")
            continue

        mp4_output_path = ANIMATIONS_OUTPUT_DIR / f"animation_sweep_{param_short_name}.mp4"
        frame_pattern_abs = str(temp_param_frames_dir.resolve() / "frame_%04d.png")

        ffmpeg_cmd = [
            "ffmpeg",
            "-r", str(FPS),
            "-i", frame_pattern_abs,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", # Ensure even dimensions
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            str(mp4_output_path.resolve())
        ]

        logging.info(f"Compiling MP4: {mp4_output_path.name} ({len(generated_frame_paths)} frames)")
        if run_command_robust(ffmpeg_cmd, command_description=f"ffmpeg for {param_short_name}"):
            logging.info(f"Successfully created animation: {mp4_output_path}")
            # Clean up frames after successful MP4 creation
            if temp_param_frames_dir.exists():
                try:
                    shutil.rmtree(temp_param_frames_dir)
                    logging.debug(f"Cleaned up temporary frames directory: {temp_param_frames_dir}")
                except OSError as e_clean:
                    logging.warning(f"Could not clean up temp frames dir {temp_param_frames_dir}: {e_clean}")
        else:
            logging.error(f"Failed to create MP4 for {param_short_name}.")
            logging.info(f"Frames for {param_short_name} are available at: {temp_param_frames_dir}")
            # Do not delete frames if ffmpeg failed

    logging.info("\n--- Animation generation process complete. ---")
    if TEMP_FRAMES_DIR_ROOT.exists() and not any(TEMP_FRAMES_DIR_ROOT.iterdir()):
        try:
            TEMP_FRAMES_DIR_ROOT.rmdir()
        except OSError:
            pass

if __name__ == "__main__":
    main()
