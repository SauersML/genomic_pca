import subprocess
import os
import time
import csv # For writing TSV
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil

# --- Configuration ---
RUST_EXECUTABLE = Path("~/genomic_pca/target/release/genomic_pca").expanduser().resolve()
BASE_OUTPUT_DIR = Path("./eigensnp_sweeps_output").resolve() # Resolve path

# Input files (data files in current dir, LD block file specified)
BED_FILE = Path("chr22_hg38_plink1.bed").resolve()
LD_BLOCK_FILE = Path("pyrho_EAS_LD_blocks.bed").resolve()

# Number of parallel EIGENSNP processes to run
NUM_PARALLEL_RUST_JOBS = max(1, cpu_count() // 4)
THREADS_PER_RUST_JOB = 2 # Each Rust job will use this many threads

# --- Helper Functions ---

def get_default_params():
    """Returns a dictionary of default parameters for an EIGENSNP run."""
    return {
        # Fixed for all runs as per request
        "eigensnp_k_global": 10,
        "eigensnp_collect_diagnostics": False,

        # Defaults from CliArgs or EigenSNPCoreAlgorithmConfig
        "eigensnp_min_call_rate": 0.98,
        "eigensnp_min_maf": 0.01,
        "eigensnp_max_hwe_p": 1e-6,
        "eigensnp_components_per_block": 7,
        "eigensnp_subset_factor": 0.075,
        "eigensnp_min_subset_size": 10000,
        "eigensnp_max_subset_size": 40000,
        "eigensnp_global_oversampling": 10,
        "eigensnp_global_power_iter": 2,
        "eigensnp_local_oversampling": 10,
        "eigensnp_local_power_iter": 2,
        "eigensnp_seed": 2025,
        "eigensnp_snp_strip_size": 2000,
        "eigensnp_refine_passes": 1,
        "eigensnp_sample_keep_file": None,
        "threads": THREADS_PER_RUST_JOB,
        "log_level": "Warn", # Reduce log verbosity for batch runs unless debugging
    }

def generate_run_configs():
    """Generates a list of configurations, sweeping one parameter at a time."""
    all_configs = []
    default_params = get_default_params()
    run_counter = 0

    # Define sweep values (moderate number of values for reasonable total runtime)
    sweeps = {
      "eigensnp_min_maf": [0.001, 0.03, 0.06, 0.25],  # Exploring lower, intermediate, and slightly higher, all distinct
      "eigensnp_max_hwe_p": [5e-8, 5e-5, 1e-4, 0.001, 0.01, 0.1], # More points, avoiding exact original values
      "eigensnp_components_per_block": [2, 8, 18, 22, 50], # Shifted and expanded, distinct from originals
      "eigensnp_local_oversampling": [1, 8, 15], # Exploring lower values, distinct from originals
      "eigensnp_local_power_iter": [1, 3, 5, 10],
      "eigensnp_snp_strip_size": [3000, 7500, 12000, 30000],
      "eigensnp_refine_passes": [4, 6, 8, 12],
    }

    # Base run with all defaults
    run_counter += 1
    base_config = default_params.copy()
    base_config["run_id"] = f"run_{run_counter:003d}_base_defaults"
    base_config["output_dir"] = BASE_OUTPUT_DIR / base_config["run_id"]
    base_config["output_prefix"] = base_config["output_dir"] / "eigensnp_results"
    base_config["swept_param_name"] = "N/A (Base Defaults)"
    base_config["swept_param_value"] = "N/A"
    all_configs.append(base_config)

    for param_name, sweep_values in sweeps.items():
        for value in sweep_values:
            if value == default_params.get(param_name) and param_name != "eigensnp_min_maf": # always run default MAF explicitly
                continue # Skip if sweep value is the same as default, to avoid redundancy

            run_counter += 1
            config = default_params.copy()
            config[param_name] = value

            # Adjust min/max subset size if sweep forces an invalid state
            if param_name == "eigensnp_min_subset_size":
                config["eigensnp_max_subset_size"] = max(default_params["eigensnp_max_subset_size"], value)
            elif param_name == "eigensnp_max_subset_size":
                config["eigensnp_min_subset_size"] = min(default_params["eigensnp_min_subset_size"], value)

            # Create a descriptive run ID
            # Sanitize value for run_id if it's float like 1e-5
            value_str = str(value)
            if isinstance(value, float) and ("e-" in value_str or "E-" in value_str):
                value_str = f"{value:.0e}".replace("-0", "-") # e.g., 1e-05 -> 1e-5
            
            config["run_id"] = f"run_{run_counter:003d}_{param_name.replace('eigensnp_', '')}_{value_str}"
            config["output_dir"] = BASE_OUTPUT_DIR / config["run_id"]
            config["output_prefix"] = config["output_dir"] / "eigensnp_results"
            config["swept_param_name"] = param_name
            config["swept_param_value"] = value
            all_configs.append(config)
    
    print(f"Generated {len(all_configs)} run configurations.")
    return all_configs

def execute_single_eigensnp_run(config_tuple):
    """
    Executes a single EIGENSNP run based on the given config.
    config_tuple is (config_dict, RUST_EXECUTABLE_PATH, BED_FILE_PATH, LD_BLOCK_FILE_PATH)
    """
    config, rust_exec_path, bed_file_path, ld_block_file_path = config_tuple
    
    os.makedirs(config["output_dir"], exist_ok=True)

    cmd_list = [
        str(rust_exec_path),
        "--eigensnp",
        "--out", str(config["output_prefix"]),
        "--bed-file", str(bed_file_path),
        "--ld-block-file", str(ld_block_file_path),
        "--threads", str(config["threads"]),
        "--log-level", str(config["log_level"]),
    ]

    for key, value in config.items():
        if key.startswith("eigensnp_") and value is not None:
            cli_arg_name = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value: # Add flag only if true
                    cmd_list.append(cli_arg_name)
            else:
                cmd_list.extend([cli_arg_name, str(value)])
    
    # Ensure --eigensnp-collect-diagnostics is NOT present if set to False
    if not config.get("eigensnp_collect_diagnostics", False):
        try:
            # Repeatedly remove if it somehow got there multiple times
            while "--eigensnp-collect-diagnostics" in cmd_list:
                cmd_list.remove("--eigensnp-collect-diagnostics")
        except ValueError:
            pass # Not present, which is correct

    run_log_stdout = config["output_dir"] / "run.stdout.log"
    run_log_stderr = config["output_dir"] / "run.stderr.log"
    
    start_time = time.time()
    success = False
    error_message = ""
    return_code = -1

    try:
        with open(run_log_stdout, 'wb') as f_out, open(run_log_stderr, 'wb') as f_err: # Open in binary for Popen
            process = subprocess.Popen(cmd_list, stdout=f_out, stderr=f_err)
            process.wait() # Wait for the process to complete
            return_code = process.returncode
            if return_code == 0:
                success = True
            else:
                error_message = f"Process exited with code {return_code}"
    except FileNotFoundError:
        error_message = f"Executable not found: {cmd_list[0]}"
        success = False
    except Exception as e:
        error_message = f"Subprocess execution error: {str(e)}"
        success = False
    
    end_time = time.time()
    duration_seconds = end_time - start_time

    return {
        "run_id": config["run_id"],
        "success": success,
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "swept_param_name": config["swept_param_name"],
        "swept_param_value": str(config["swept_param_value"]), # Ensure string for TSV
        "error_message": error_message,
        "output_dir": str(config["output_dir"]),
        "command": " ".join(cmd_list)
    }

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Python script started. Base output directory: {BASE_OUTPUT_DIR}")
    print(f"Rust executable: {RUST_EXECUTABLE}")
    print(f"BED file: {BED_FILE}")
    print(f"LD Block file: {LD_BLOCK_FILE}")

    if not RUST_EXECUTABLE.exists():
        print(f"Error: Rust executable not found at '{RUST_EXECUTABLE}'")
        exit(1)
    if not BED_FILE.exists():
        print(f"Error: BED file not found at '{BED_FILE}'")
        exit(1)
    if not LD_BLOCK_FILE.exists():
        print(f"Error: LD Block file not found at '{LD_BLOCK_FILE}'")
        exit(1)

    if BASE_OUTPUT_DIR.exists():
        print(f"Warning: Base output directory '{BASE_OUTPUT_DIR}' already exists.")
        # Optionally, add logic to prompt for removal or backup
    else:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        print(f"Created base output directory: {BASE_OUTPUT_DIR}")


    run_configs = generate_run_configs()
    
    tasks_for_pool = [(cfg, RUST_EXECUTABLE, BED_FILE, LD_BLOCK_FILE) for cfg in run_configs]

    print(f"\nStarting {len(run_configs)} EIGENSNP runs using up to {NUM_PARALLEL_RUST_JOBS} parallel Rust processes.")
    print(f"Each Rust process will be configured to use {THREADS_PER_RUST_JOB} threads.")

    results_data = []
    with Pool(processes=NUM_PARALLEL_RUST_JOBS) as pool:
        for result in tqdm(pool.imap_unordered(execute_single_eigensnp_run, tasks_for_pool), total=len(run_configs), desc="EIGENSNP Runs"):
            results_data.append(result)

    # --- Summarize Results ---
    print("\n--- EIGENSNP Sweeps Summary ---")
    
    # Sort results by run_id for consistent reporting if desired
    results_data.sort(key=lambda r: r["run_id"])

    # Define column order for TSV and console output
    tsv_columns = ["run_id", "swept_param_name", "swept_param_value", 
                   "duration_seconds", "success", "return_code", 
                   "error_message", "output_dir", "command"]

    summary_file_path = BASE_OUTPUT_DIR / "sweeps_summary.tsv"
    try:
        with open(summary_file_path, 'w', newline='', encoding='utf-8') as f_tsv:
            writer = csv.DictWriter(f_tsv, fieldnames=tsv_columns, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for res_dict in results_data:
                # Ensure all keys are present, provide default if not (though they should be)
                row_to_write = {key: res_dict.get(key, "") for key in tsv_columns}
                writer.writerow(row_to_write)
        print(f"Detailed summary saved to TSV: {summary_file_path}")
    except IOError as e:
        print(f"Error saving summary TSV: {e}")
    except Exception as e_gen:
        print(f"A general error occurred while writing TSV: {e_gen}")


    print(f"\n{'Run ID':<55} | {'Swept Parameter':<30} | {'Value':<15} | {'Duration (s)':<15} | Status")
    print("-" * 150)

    successful_runs_count = 0
    failed_runs_list = []

    for res in results_data:
        status_msg = "SUCCESS" if res["success"] else f"FAILED (Code: {res['return_code']})"
        if res["success"]:
            successful_runs_count +=1
        else:
            failed_runs_list.append(res)
        
        swept_param_display = res["swept_param_name"].replace("eigensnp_", "")
        
        # Corrected f-string for duration
        print(f"{res['run_id']:<55} | {swept_param_display:<30} | {str(res['swept_param_value']):<15} | {res['duration_seconds']:<15.2f} | {status_msg}")

    print("-" * 150)
    print(f"\nTotal runs attempted: {len(results_data)}")
    print(f"Successful runs: {successful_runs_count}")
    print(f"Failed runs: {len(failed_runs_list)}")

    if failed_runs_list:
        print("\nDetails for failed runs (check *.stdout.log and *.stderr.log in their respective output_dir):")
        for fres in failed_runs_list:
            print(f"  - ID: {fres['run_id']}")
            print(f"    Output Dir: {fres['output_dir']}")
            print(f"    Error: {fres['error_message']}")
            print(f"    Command: {fres['command']}")

    print("\nSweep analysis complete.")
