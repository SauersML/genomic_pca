import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import importlib.util
import os
import re
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import math

# --- Configuration ---
BASE_SWEEP_OUTPUT_DIR = Path("eigensnp_sweeps_output").resolve()
METRICS_SCRIPT_PY_PATH = Path("metrics.py").resolve()
SAMPLE_FILE_METRICS = Path("igsr_samples.tsv").resolve()
PLOTS_OUTPUT_DIR = BASE_SWEEP_OUTPUT_DIR / "analysis_plots"

# --- Exact PCA Reference Configuration ---
EXACT_PCA_DIR = Path("chr22_hg38_safe_pca").resolve()
EXACT_PCA_TSV_NAME = "py.pca.tsv" # The specific PCA file in EXACT_PCA_DIR
EXACT_PCA_RUN_ID = "exact_pca_reference"
EXACT_PCA_DISPLAY_NAME = "Exact PCA Reference" # For legends and dataframes

# --- Metrics Configuration ---

# --- Metrics Configuration ---
# This tells metrics.py how many PCs to use from the .pca.tsv files
# Should match the --eigensnp-k-global used for the runs.
METRICS_PCS_TO_USE = 10
METRICS_MC_SAMPLES_KDE = 4000 # Default for metrics.py

METRIC_COLUMNS_TO_ANALYZE = [
    "Mutual_information_nats",
    "Mean_multivariate_Jensen_Shannon_divergence_nats",
    "Average_silhouette",
    "Mean_contrastive_violation",
    "HDBSCAN_adjusted_mutual_information",
]

METRIC_PROPERTIES = {
    "Mutual_information_nats": {"higher_is_better": True, "name": "Mutual Information (nats)"},
    "Mean_multivariate_Jensen_Shannon_divergence_nats": {"higher_is_better": True, "name": "Mean JSD (nats)"},
    "Average_silhouette": {"higher_is_better": True, "name": "Avg. Silhouette"},
    "Mean_contrastive_violation": {"higher_is_better": False, "name": "Mean Contrastive Violation"},
    "HDBSCAN_adjusted_mutual_information": {"higher_is_better": True, "name": "HDBSCAN AMI"},
}

# --- Parallelism Configuration ---
NUM_METRIC_CALC_WORKERS = max(1, cpu_count() // 2)

# --- Helper: Load metrics.py ---
try:
    spec = importlib.util.spec_from_file_location("metrics_module", METRICS_SCRIPT_PY_PATH)
    metrics_module = importlib.util.module_from_spec(spec)
    sys.modules["metrics_module"] = metrics_module
    spec.loader.exec_module(metrics_module)
    print(f"Successfully loaded metrics module from: {METRICS_SCRIPT_PY_PATH}")
except Exception as e:
    print(f"CRITICAL Error loading metrics module from {METRICS_SCRIPT_PY_PATH}: {e}")
    print("Please ensure metrics.py is present, executable, and all its dependencies are installed.")
    sys.exit(1)

# --- Helper: EIGENSNP Default Parameter Parsing ---
def parse_param_from_command(command_str, param_name_cli):
    # Regex to find param and its value, handling potential trailing options
    match = re.search(f"{param_name_cli}\\s+([^\\s]+)", command_str)
    if match:
        val_str = match.group(1)
        try: return float(val_str)
        except ValueError:
            try: return int(val_str)
            except ValueError:
                if val_str.lower() == 'true': return True
                if val_str.lower() == 'false': return False
                return val_str # as string if not numeric/bool
    return None

def get_eigensnp_default_params_from_base_run_command(base_run_command_str):
    """
    Infers default parameters by parsing the command string of the base run.
    These are the effective defaults used for the 'base_defaults' run.
    """
    # Start with a set of known/expected EIGENSNP parameters from the CLI
    # The values will be filled by parsing the command
    expected_params_internal_names = [
        "eigensnp_k_global", "eigensnp_min_call_rate", "eigensnp_min_maf",
        "eigensnp_max_hwe_p", "eigensnp_components_per_block", "eigensnp_subset_factor",
        "eigensnp_min_subset_size", "eigensnp_max_subset_size", "eigensnp_global_oversampling",
        "eigensnp_global_power_iter", "eigensnp_local_oversampling", "eigensnp_local_power_iter",
        "eigensnp_seed", "eigensnp_snp_strip_size", "eigensnp_refine_passes", "threads"
    ]
    
    defaults = {}
    for key_internal in expected_params_internal_names:
        cli_arg_name = f"--{key_internal.replace('_', '-')}"
        parsed_val = parse_param_from_command(base_run_command_str, cli_arg_name)
        if parsed_val is not None:
            defaults[key_internal] = parsed_val
        # Special handling for boolean flags that might not have a value
        elif key_internal == "eigensnp_collect_diagnostics":
             if cli_arg_name in base_run_command_str:
                 defaults[key_internal] = True
             else:
                 defaults[key_internal] = False # Assume false if flag not present
    
    # Ensure METRICS_PCS_TO_USE is aligned if k_global was parsed
    if 'eigensnp_k_global' in defaults and defaults['eigensnp_k_global'] != METRICS_PCS_TO_USE:
        print(f"Warning: Parsed eigensnp_k_global from base run ({defaults['eigensnp_k_global']}) "
              f"differs from METRICS_PCS_TO_USE ({METRICS_PCS_TO_USE}). "
              f"Using {METRICS_PCS_TO_USE} for metrics script.")
    defaults['eigensnp_k_global'] = METRICS_PCS_TO_USE # Enforce this for metrics script call

    return defaults

# --- Helper: Percent Better Calculation ---
def calculate_percent_better(current_val, baseline_val, higher_is_better):
    if pd.isna(current_val) or pd.isna(baseline_val):
        return np.nan

    if baseline_val == 0:
        if current_val == 0: percent_diff = 0.0
        else: percent_diff = np.sign(current_val) * 200.0 # Cap for change from zero
    else:
        percent_diff = ((current_val - baseline_val) / abs(baseline_val)) * 100.0

    return percent_diff * (-1 if not higher_is_better else 1)

# --- Worker for Parallel Metric Calculation ---
def process_single_run_metrics_worker(task_args):
    run_info, base_output_dir_of_sweeps_str = task_args # Unpack
    base_output_dir_of_sweeps = Path(base_output_dir_of_sweeps_str)

    run_id = run_info["run_id"]
    run_output_dir = Path(run_info["output_dir"])

    # Determine PCA file path
    if "pca_file_path_override" in run_info:
        pca_file = Path(run_info["pca_file_path_override"])
    else:
        # output_dir in sweeps_summary.tsv is the absolute path to the run's folder for sweep runs
        pca_file = run_output_dir / "eigensnp_results.eigensnp.pca.tsv"
    
    # Use a unique name for the metrics output for this specific call, stored in the run's output_dir
    metrics_output_temp_file = run_output_dir / f"{pca_file.stem}_metrics_calc_temp.tsv"
    
    run_metrics_results = []

    if pca_file.exists():
        try:
            # Call the imported function from metrics.py
            metrics_module.compute_metrics_for_superpopulations(
                pca_file_path=str(pca_file),
                sample_file_path=str(SAMPLE_FILE_METRICS),
                number_of_pcs=METRICS_PCS_TO_USE,
                monte_carlo_samples=METRICS_MC_SAMPLES_KDE,
                output_tsv_path=str(metrics_output_temp_file)
            )
            if metrics_output_temp_file.exists():
                df_metrics_current_run = pd.read_csv(metrics_output_temp_file, sep='\t')
                for _, metrics_row in df_metrics_current_run.iterrows():
                    data_point = {
                        "run_id": run_id,
                        "swept_param_name": run_info["swept_param_name"],
                        "swept_param_value_str": str(run_info["swept_param_value"]), # it's string
                        "superpopulation": metrics_row["Superpopulation"],
                        "command": run_info.get("command", "")
                    }
                    for m_col in METRIC_COLUMNS_TO_ANALYZE:
                        data_point[m_col] = metrics_row.get(m_col, np.nan)
                    run_metrics_results.append(data_point)
                
                try: # Attempt to clean up the temporary metrics file
                    os.remove(metrics_output_temp_file)
                except OSError as e_rm:
                    print(f"Warning: Could not remove temporary metrics file {metrics_output_temp_file}: {e_rm}")
            else:
                pass
        except Exception as e:
            pass
    return run_metrics_results


# --- Main Analysis Logic ---
def main():
    # File/Dir Checks
    for p in [BASE_SWEEP_OUTPUT_DIR, METRICS_SCRIPT_PY_PATH, SAMPLE_FILE_METRICS]:
        if not p.exists():
            print(f"CRITICAL Error: Required path not found: {p}")
            sys.exit(1)

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    print(f"Analysis plots will be saved to: {PLOTS_OUTPUT_DIR}")

    # 1. Load Main Sweep Summary
    sweep_summary_file = BASE_SWEEP_OUTPUT_DIR / "sweeps_summary.tsv"
    if not sweep_summary_file.exists():
        print(f"CRITICAL Error: Main sweep summary file not found: {sweep_summary_file}")
        sys.exit(1)
    df_sweep_summary = pd.read_csv(sweep_summary_file, sep='\t', na_filter=False) # Keep "N/A" as string
    print(f"Loaded sweep summary with {len(df_sweep_summary)} total run records.")

    # 2. Collect Metrics in Parallel
    all_run_metrics_data_flat = []
    tasks_for_pool = []
    
    # Identify runs that potentially have PCA files from sweeps_summary.tsv
    for _, run_info_series in df_sweep_summary.iterrows():
        run_info_dict = run_info_series.to_dict()
        # The worker function will do the final check for PCA file existence.
        tasks_for_pool.append((run_info_dict, str(BASE_SWEEP_OUTPUT_DIR))) # Second arg (base_output_dir_of_sweeps_str) is legacy, worker uses run_info["output_dir"]

    # Add the Exact PCA reference task
    exact_pca_task_info = {
        "run_id": EXACT_PCA_RUN_ID,
        "output_dir": str(EXACT_PCA_DIR), # For its temp metrics file
        "pca_file_path_override": str(EXACT_PCA_DIR / EXACT_PCA_TSV_NAME),
        "swept_param_name": EXACT_PCA_DISPLAY_NAME,
        "swept_param_value": "Reference", # This will be the value in swept_param_value_str
        "command": "Exact PCA Reference Data"
    }
    os.makedirs(EXACT_PCA_DIR, exist_ok=True)
    if not (EXACT_PCA_DIR / EXACT_PCA_TSV_NAME).exists():
        print(f"WARNING: Exact PCA file not found at {EXACT_PCA_DIR / EXACT_PCA_TSV_NAME}. Exact PCA Reference will be missing.")
    else:
        tasks_for_pool.append((exact_pca_task_info, str(EXACT_PCA_DIR)))

    print(f"Starting parallel metric calculation for {len(tasks_for_pool)} total runs (including sweeps and exact PCA if applicable) using {NUM_METRIC_CALC_WORKERS} workers...")
    
    with Pool(processes=NUM_METRIC_CALC_WORKERS) as pool:
        # Using imap_unordered for potentially faster aggregation if tasks complete out of order
        # Wrap with tqdm for progress bar
        results_iterator = pool.imap_unordered(process_single_run_metrics_worker, tasks_for_pool)
        for single_run_results_list in tqdm(results_iterator, total=len(tasks_for_pool), desc="Calculating Metrics"):
            if single_run_results_list: # If list is not empty (i.e., metrics were calculated)
                all_run_metrics_data_flat.extend(single_run_results_list)
    
    if not all_run_metrics_data_flat:
        print("CRITICAL Error: No metrics data could be collected. Exiting.")
        sys.exit(1)

    df_all_metrics = pd.DataFrame(all_run_metrics_data_flat)
    print(f"Collected metrics for {df_all_metrics['run_id'].nunique()} runs across {df_all_metrics['superpopulation'].nunique()} unique superpopulations.")

    def to_numeric_robust(val):
        if isinstance(val, (int, float)): return val
        if pd.isna(val) or val == "N/A" or not isinstance(val, str): return np.nan
        try: return float(val)
        except ValueError: return np.nan
    df_all_metrics["swept_param_value_numeric"] = df_all_metrics["swept_param_value_str"].apply(to_numeric_robust)

    # 3. Identify Baseline Metrics and Default EIGENSNP Params
    base_run_command_series = df_all_metrics[df_all_metrics["swept_param_name"].str.contains("Base Defaults", na=False)]["command"]
    if base_run_command_series.empty or pd.isna(base_run_command_series.iloc[0]) or base_run_command_series.iloc[0] == "":
        print("CRITICAL Error: Command string for 'Base Defaults' run not found or empty. Cannot establish EIGENSNP defaults.")
        sys.exit(1)
    eigensnp_defaults = get_eigensnp_default_params_from_base_run_command(base_run_command_series.iloc[0])
    print(f"Using EIGENSNP default parameters derived from base run: {eigensnp_defaults}")

    df_baseline_metrics_all_fields = df_all_metrics[df_all_metrics["swept_param_name"].str.contains("Base Defaults", na=False)]
    if df_baseline_metrics_all_fields.empty:
         print("CRITICAL Error: 'Base Defaults' run data not found in collected metrics. Cannot establish metric baselines.")
         sys.exit(1)

    df_baseline_metrics_pivot = df_baseline_metrics_all_fields.pivot_table(
        index="superpopulation", values=METRIC_COLUMNS_TO_ANALYZE
    ).rename(columns={m: f"baseline_{m}" for m in METRIC_COLUMNS_TO_ANALYZE})

    # 4. Plotting - Per Swept Parameter Metric Summary
    
    # Prepare Exact PCA metrics for plotting (will contain its 'percent_better' values)
    # This is done after all percent_better columns are calculated in df_all_metrics
    df_all_metrics_with_baselines = pd.merge(df_all_metrics, df_baseline_metrics_pivot, on="superpopulation", how="left")
    for metric_key_for_pct in METRIC_COLUMNS_TO_ANALYZE: # Calculate percent better for all, including exact PCA
        metric_props_for_pct = METRIC_PROPERTIES.get(metric_key_for_pct, {"higher_is_better": True})
        df_all_metrics_with_baselines[f"percent_better_{metric_key_for_pct}"] = df_all_metrics_with_baselines.apply(
            lambda row: calculate_percent_better(row.get(metric_key_for_pct), row.get(f"baseline_{metric_key_for_pct}"), metric_props_for_pct["higher_is_better"]), axis=1
        )
    df_exact_pca_metrics = df_all_metrics_with_baselines[df_all_metrics_with_baselines["swept_param_name"] == EXACT_PCA_DISPLAY_NAME].copy()


    unique_swept_eigensnp_params = sorted([
        p for p in df_all_metrics["swept_param_name"].unique() 
        if not pd.isna(p) and "Base Defaults" not in p and p != EXACT_PCA_DISPLAY_NAME
    ])
    print(f"\nStarting stacked metric plotting for {len(unique_swept_eigensnp_params)} swept EIGENSNP parameters...")

    for eigensnp_param_name in unique_swept_eigensnp_params:
        swept_param_name_cleaned_for_filename = eigensnp_param_name.replace("eigensnp_", "").replace(".", "_dot_")
        print(f"  Generating stacked plot for swept parameter: {eigensnp_param_name}")

        df_current_sweep = df_all_metrics[df_all_metrics["swept_param_name"] == eigensnp_param_name].copy()
        default_value_for_this_param = eigensnp_defaults.get(eigensnp_param_name)
        
        base_run_to_add = df_baseline_metrics_all_fields.copy()
        if default_value_for_this_param is not None:
            base_run_to_add["swept_param_value_numeric"] = float(default_value_for_this_param)
            base_run_to_add["swept_param_name"] = eigensnp_param_name # Mark as part of this sweep for grouping
            
            is_default_already_swept = default_value_for_this_param in df_current_sweep["swept_param_value_numeric"].unique()
            df_plot_data_for_sweep = pd.concat([df_current_sweep, base_run_to_add if not is_default_already_swept else pd.DataFrame()], ignore_index=True)
        else:
            df_plot_data_for_sweep = df_current_sweep
        
        df_plot_data_for_sweep = pd.merge(df_plot_data_for_sweep, df_baseline_metrics_pivot, on="superpopulation", how="left")

        num_metrics_to_plot = len(METRIC_COLUMNS_TO_ANALYZE)
        fig_param_metrics, axes_param_metrics = plt.subplots(num_metrics_to_plot, 2, 
                                                             figsize=(20, 6 * num_metrics_to_plot), 
                                                             squeeze=False) # Ensure axes is always 2D array

        for metric_idx, metric_key in enumerate(METRIC_COLUMNS_TO_ANALYZE):
            ax_left = axes_param_metrics[metric_idx, 0]
            ax_right = axes_param_metrics[metric_idx, 1]
            metric_props = METRIC_PROPERTIES.get(metric_key, {"higher_is_better": True, "name": metric_key})
    
            df_current_sweep_with_pct_better = df_all_metrics_with_baselines[df_all_metrics_with_baselines["swept_param_name"] == eigensnp_param_name].copy()
            
            base_run_to_add_with_pct_better = df_all_metrics_with_baselines[
                df_all_metrics_with_baselines["swept_param_name"].str.contains("Base Defaults", na=False)
            ].copy()

            if default_value_for_this_param is not None:
                base_run_to_add_with_pct_better["swept_param_value_numeric"] = float(default_value_for_this_param)
                base_run_to_add_with_pct_better["swept_param_name"] = eigensnp_param_name
                is_default_already_swept_in_current = default_value_for_this_param in df_current_sweep_with_pct_better["swept_param_value_numeric"].unique()
                df_plot_data_for_sweep_final = pd.concat([df_current_sweep_with_pct_better, base_run_to_add_with_pct_better if not is_default_already_swept_in_current else pd.DataFrame()], ignore_index=True)
            else:
                df_plot_data_for_sweep_final = df_current_sweep_with_pct_better

            df_plot_ready = df_plot_data_for_sweep_final.dropna(subset=[f"percent_better_{metric_key}", "swept_param_value_numeric"])


            if df_plot_ready.empty and df_exact_pca_metrics[pd.notna(df_exact_pca_metrics[f"percent_better_{metric_key})"])].empty : # Check both swept and exact
                ax_left.text(0.5, 0.5, "No data for this metric/sweep", ha='center', va='center', transform=ax_left.transAxes)
                ax_right.text(0.5, 0.5, "No data for this metric/sweep", ha='center', va='center', transform=ax_right.transAxes)
                ax_left.set_title(f"{metric_props['name']}\nPer Superpopulation")
                ax_right.set_title(f"{metric_props['name']}\nAggregated (Mean/Median)")
                # Still need common settings for axes if no data, like labels
                for ax_common in [ax_left, ax_right]: # Apply common settings even if no data
                    ax_common.grid(True, alpha=0.5)
                    if metric_idx == num_metrics_to_plot - 1:
                         ax_common.set_xlabel(eigensnp_param_name.replace("eigensnp_", ""))
                continue

            # Left Subplot: Plot swept data first
            ax_left.set_ylabel("Improvement vs Baseline (%)")
            ax_left.set_title(f"{metric_props['name']}\nPer Superpopulation")
            if not df_plot_ready.empty:
                for superpop_label in sorted(df_plot_ready["superpopulation"].unique()):
                    df_superpop_plot = df_plot_ready[df_plot_ready["superpopulation"] == superpop_label].sort_values("swept_param_value_numeric")
                    ax_left.plot(df_superpop_plot["swept_param_value_numeric"], df_superpop_plot[f"percent_better_{metric_key}"], marker='o', linestyle='-', label=superpop_label, markersize=5, alpha=0.8)
            
            # Determine Y-limits for ax_left based on swept data
            y_min_swept_left, y_max_swept_left = (np.nan, np.nan)
            if not df_plot_ready.empty:
                all_y_values_swept_left = df_plot_ready[f"percent_better_{metric_key}"].dropna().values
                if len(all_y_values_swept_left) > 0:
                    y_min_swept_left = np.nanmin(all_y_values_swept_left)
                    y_max_swept_left = np.nanmax(all_y_values_swept_left)
            
            if pd.notna(y_min_swept_left) and pd.notna(y_max_swept_left):
                y_range_swept_left = y_max_swept_left - y_min_swept_left
                padding_left = y_range_swept_left * 0.05 if y_range_swept_left > 1e-9 else 1.0 
                final_y_min_left = y_min_swept_left - padding_left
                final_y_max_left = y_max_swept_left + padding_left
                ax_left.set_ylim(final_y_min_left, final_y_max_left)
            else: # Fallback if no swept data or only NaNs
                ax_left.set_ylim(-100, 100) # Default reasonable range if no data to scale from

            y_plot_min_left, y_plot_max_left = ax_left.get_ylim() # Get final limits for capping Exact PCA

            # Left Subplot: Plot Exact PCA reference lines
            relevant_exact_pca_for_metric = df_exact_pca_metrics[pd.notna(df_exact_pca_metrics[f"percent_better_{metric_key}"])].copy()
            exact_pca_legend_added_ax_left = False
            if not relevant_exact_pca_for_metric.empty:
                for _, exact_row in relevant_exact_pca_for_metric.iterrows():
                    exact_val = exact_row[f"percent_better_{metric_key}"]
                    plot_y = exact_val
                    marker_char = None
                    
                    if exact_val > y_plot_max_left:
                        plot_y = y_plot_max_left
                        marker_char = '^'
                    elif exact_val < y_plot_min_left:
                        plot_y = y_plot_min_left
                        marker_char = 'v'
                    
                    label_for_exact_line = "_nolegend_"
                    if not exact_pca_legend_added_ax_left:
                        label_for_exact_line = EXACT_PCA_DISPLAY_NAME
                        exact_pca_legend_added_ax_left = True
                    
                    ax_left.axhline(y=plot_y, color='magenta', linestyle='-.', linewidth=1.5, alpha=0.9, label=label_for_exact_line)
                    if marker_char:
                        ax_left.plot(ax_left.get_xlim()[1] * 0.98, plot_y, marker=marker_char, color='magenta', markersize=7, clip_on=False, linestyle='None')


            # Right Subplot: Plot swept data aggregates first
            ax_right.set_title(f"{metric_props['name']}\nAggregated (Mean/Median)")
            df_agg = pd.DataFrame() # ensure df_agg is defined
            if not df_plot_ready.empty:
                df_agg = df_plot_ready.groupby("swept_param_value_numeric")[f"percent_better_{metric_key}"].agg(['mean', 'median']).reset_index().sort_values("swept_param_value_numeric")
                if not df_agg.empty:
                    ax_right.plot(df_agg["swept_param_value_numeric"], df_agg["mean"], marker='s', linestyle='--', label="Mean (Swept)")
                    ax_right.plot(df_agg["swept_param_value_numeric"], df_agg["median"], marker='^', linestyle=':', label="Median (Swept)")

            # Determine Y-limits for ax_right based on swept aggregate data
            y_min_swept_right, y_max_swept_right = (np.nan, np.nan)
            if not df_agg.empty:
                all_y_values_swept_right = pd.concat([df_agg['mean'], df_agg['median']]).dropna().values
                if len(all_y_values_swept_right) > 0:
                    y_min_swept_right = np.nanmin(all_y_values_swept_right)
                    y_max_swept_right = np.nanmax(all_y_values_swept_right)

            if pd.notna(y_min_swept_right) and pd.notna(y_max_swept_right):
                y_range_swept_right = y_max_swept_right - y_min_swept_right
                padding_right = y_range_swept_right * 0.05 if y_range_swept_right > 1e-9 else 1.0
                final_y_min_right = y_min_swept_right - padding_right
                final_y_max_right = y_max_swept_right + padding_right
                ax_right.set_ylim(final_y_min_right, final_y_max_right)
            elif not df_plot_ready.empty: # If df_agg was empty but df_plot_ready was not, try to use ax_left's limits
                 ax_right.set_ylim(y_plot_min_left, y_plot_max_left)
            else: # Fallback
                ax_right.set_ylim(-100, 100)
            
            y_plot_min_right, y_plot_max_right = ax_right.get_ylim()

            # Right Subplot: Plot Exact PCA aggregate lines
            if not relevant_exact_pca_for_metric.empty:
                exact_mean_val = relevant_exact_pca_for_metric[f"percent_better_{metric_key}"].mean()
                if pd.notna(exact_mean_val):
                    plot_y_exact_mean = exact_mean_val
                    marker_char_exact_mean = None
                    if exact_mean_val > y_plot_max_right:
                        plot_y_exact_mean = y_plot_max_right
                        marker_char_exact_mean = '^'
                    elif exact_mean_val < y_plot_min_right:
                        plot_y_exact_mean = y_plot_min_right
                        marker_char_exact_mean = 'v'
                    ax_right.axhline(y=plot_y_exact_mean, color='purple', linestyle='-.', linewidth=1.5, label=f"{EXACT_PCA_DISPLAY_NAME} (Mean)")
                    if marker_char_exact_mean:
                        ax_right.plot(ax_right.get_xlim()[1] * 0.98, plot_y_exact_mean, marker=marker_char_exact_mean, color='purple', markersize=7, clip_on=False, linestyle='None')

                exact_median_val = relevant_exact_pca_for_metric[f"percent_better_{metric_key}"].median()
                if pd.notna(exact_median_val):
                    plot_y_exact_median = exact_median_val
                    marker_char_exact_median = None
                    if exact_median_val > y_plot_max_right:
                        plot_y_exact_median = y_plot_max_right
                        marker_char_exact_median = '^'
                    elif exact_median_val < y_plot_min_right:
                        plot_y_exact_median = y_plot_min_right
                        marker_char_exact_median = 'v'
                    ax_right.axhline(y=plot_y_exact_median, color='darkviolet', linestyle=':', linewidth=1.5, label=f"{EXACT_PCA_DISPLAY_NAME} (Median)")
                    if marker_char_exact_median:
                        ax_right.plot(ax_right.get_xlim()[1] * 0.98, plot_y_exact_median, marker=marker_char_exact_median, color='darkviolet', markersize=7, clip_on=False, linestyle='None')
            
            # Common settings for this metric's row of subplots
            for ax_common in [ax_left, ax_right]:
                ax_common.grid(True, alpha=0.5)
                if eigensnp_param_name in ["eigensnp_min_maf", "eigensnp_max_hwe_p", "eigensnp_subset_factor"] and \
                   not df_plot_ready["swept_param_value_numeric"].empty and \
                   df_plot_ready["swept_param_value_numeric"].min() > 0:
                    try: ax_common.set_xscale('log')
                    except ValueError: ax_common.set_xscale('linear')
                
                if default_value_for_this_param is not None:
                    ax_common.axvline(float(default_value_for_this_param), color='grey', linestyle='--', linewidth=1.2, label=f"Default ({default_value_for_this_param:.2g})" if ax_common == ax_left and metric_idx ==0 else None) # Only label once
                
                if metric_idx == num_metrics_to_plot - 1: # Only x-label for bottom row
                     ax_common.set_xlabel(eigensnp_param_name.replace("eigensnp_", ""))
            
            if not ax_left.get_legend_handles_labels()[0]: # If no lines were plotted
                 ax_left.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_left.transAxes)
            else:
                 ax_left.legend(title="Superpopulation", loc="best", fontsize='small')

            if not ax_right.get_legend_handles_labels()[0]:
                 ax_right.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_right.transAxes)
            else:
                 ax_right.legend(loc="best", fontsize='small')


        fig_param_metrics.suptitle(f"Impact of Sweeping '{eigensnp_param_name.replace('eigensnp_', '')}' on Population Metrics", fontsize=18, y=0.999)
        plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust for suptitle
        plot_filepath = PLOTS_OUTPUT_DIR / f"metrics_summary__{swept_param_name_cleaned_for_filename}.png"
        plt.savefig(plot_filepath)
        print(f"    Saved stacked metrics plot: {plot_filepath}")
        plt.close(fig_param_metrics)

    # 5. Plotting - Runtimes Summary (Mega-Plot)
    print("\nGenerating runtime mega-plot...")
    num_runtime_subplots = len(unique_swept_eigensnp_params)
    if num_runtime_subplots == 0:
        print("No swept parameters found for runtime plot. Skipping.")
    else:
        # Determine layout for runtime plot (e.g., 3 or 4 columns)
        runtime_plot_cols = min(4, num_runtime_subplots)
        runtime_plot_rows = math.ceil(num_runtime_subplots / runtime_plot_cols)
        fig_runtimes, axes_runtimes = plt.subplots(runtime_plot_rows, runtime_plot_cols, 
                                                   figsize=(5 * runtime_plot_cols, 4 * runtime_plot_rows),
                                                   squeeze=False)
        axes_runtimes_flat = axes_runtimes.flatten()

        base_run_duration = df_sweep_summary[df_sweep_summary["swept_param_name"].str.contains("Base Defaults", na=False)]["duration_seconds"].iloc[0]

        for idx, eigensnp_param_name in enumerate(unique_swept_eigensnp_params):
            ax_rt = axes_runtimes_flat[idx]
            df_runtime_sweep = df_sweep_summary[df_sweep_summary["swept_param_name"] == eigensnp_param_name].copy()
            df_runtime_sweep["swept_param_value_numeric"] = df_runtime_sweep["swept_param_value"].apply(to_numeric_robust)
            df_runtime_sweep = df_runtime_sweep.dropna(subset=["swept_param_value_numeric", "duration_seconds"]).sort_values("swept_param_value_numeric")

            if not df_runtime_sweep.empty:
                ax_rt.plot(df_runtime_sweep["swept_param_value_numeric"], df_runtime_sweep["duration_seconds"], marker='o', linestyle='-')
            
            default_val_for_param = eigensnp_defaults.get(eigensnp_param_name)
            if default_val_for_param is not None and pd.notna(base_run_duration):
                # Plot default point if not already part of the sweep
                if float(default_val_for_param) not in df_runtime_sweep["swept_param_value_numeric"].values:
                    ax_rt.plot(float(default_val_for_param), base_run_duration, marker='*', color='red', markersize=10, linestyle='None', label=f"Default ({default_val_for_param:.2g})")
                ax_rt.axvline(float(default_val_for_param), color='grey', linestyle='--', linewidth=1)
            
            ax_rt.set_title(eigensnp_param_name.replace("eigensnp_", ""), fontsize='medium')
            ax_rt.set_ylabel("Runtime (s)", fontsize='small')
            ax_rt.set_xlabel("Parameter Value", fontsize='small')
            ax_rt.grid(True, alpha=0.5)
            ax_rt.tick_params(axis='x', labelsize='x-small')
            ax_rt.tick_params(axis='y', labelsize='x-small')

            if eigensnp_param_name in ["eigensnp_min_maf", "eigensnp_max_hwe_p", "eigensnp_subset_factor"] and \
               not df_runtime_sweep.empty and df_runtime_sweep["swept_param_value_numeric"].min() > 0:
                try: ax_rt.set_xscale('log')
                except ValueError: ax_rt.set_xscale('linear')
            if ax_rt.has_data() and default_val_for_param is not None : ax_rt.legend(fontsize='xx-small')

        # Hide any unused subplots
        for i in range(num_runtime_subplots, len(axes_runtimes_flat)):
            fig_runtimes.delaxes(axes_runtimes_flat[i])

        fig_runtimes.suptitle("EIGENSNP Runtime vs. Swept Parameters", fontsize=20, y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        runtime_plot_filepath = PLOTS_OUTPUT_DIR / "runtime_summary_mega_plot.png"
        plt.savefig(runtime_plot_filepath)
        print(f"  Saved runtime mega-plot: {runtime_plot_filepath}")
        plt.close(fig_runtimes)

    print("\nAnalysis and plotting complete.")

if __name__ == "__main__":
    main()
