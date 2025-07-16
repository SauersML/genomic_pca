import os
import sys
import logging
import subprocess
import shutil
import zipfile
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bed_reader import open_bed
from numpy.linalg import qr
from scipy.linalg import subspace_angles, norm
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

CWD = Path.cwd()
if not (CWD / "Cargo.toml").exists() and (CWD.parent / "Cargo.toml").exists():
    logging.warning(f"Cargo.toml not found in {CWD}. Moving to parent directory {CWD.parent}.")
    CWD = CWD.parent

EIGENSNP_EXECUTABLE = CWD / "target" / "release" / "genomic_pca"
CARGO_EXECUTABLE = "cargo"

# --- Input Data (Relative to project root) ---
RAW_DATA_PREFIX = CWD / "data" / "chr22_hg38_plink1"
SAMPLE_INFO_FILE = CWD / "data" / "igsr_samples.tsv"

# --- Output Directories (Created by the script) ---
MAIN_OUTPUT_DIR = CWD / "local_pca_oos_analysis_output"
LOCAL_PCA_OUTPUT_DIR = MAIN_OUTPUT_DIR / "local_pca_outputs"
# This single prefix will be used for the one efficient run
PCA_RUN_PREFIX = LOCAL_PCA_OUTPUT_DIR / "run_prefix"
# This is the path to the ground truth loadings from the single run
GT_LOADINGS_FILE = PCA_RUN_PREFIX.with_suffix(".eigensnp.loadings.tsv")


# --- Parameters ---
K_COMPONENTS = 10
WINDOW_SIZE = 30_000
TEST_SET_FRACTION = 0.5
RANDOM_STATE = 2025
SNP_CHUNK_SIZE = 5000 # Process this many SNPs at a time for memory efficiency

# --- Helper Functions ---
def run_command(cmd, work_dir, description):
    logging.info(f"--- Running: {description} ---")
    logging.info(f"CMD: {' '.join(map(str, cmd))}")
    try:
        result = subprocess.run(cmd, cwd=work_dir, check=True, text=True, capture_output=True)
        if result.stdout: logging.info(f"STDOUT:\n{result.stdout}")
        if result.stderr: logging.warning(f"STDERR:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: {description} failed. STDERR:\n{e.stderr}\nSTDOUT:\n{e.stdout}"); return False

def setup_genomic_pca():
    if EIGENSNP_EXECUTABLE.exists():
        logging.info(f"Found 'genomic_pca' executable: {EIGENSNP_EXECUTABLE}"); return True
    logging.warning("'genomic_pca' not found. Attempting to build from source.")
    if not (CWD / "Cargo.toml").exists():
        logging.error(f"Cannot build: 'Cargo.toml' not found in project directory '{CWD}'."); return False
    logging.info("Building 'genomic_pca' with Cargo... (This may take several minutes)")
    cmd_build = [CARGO_EXECUTABLE, "build", "--release", "--features", "openblas-faer"]
    if not run_command(cmd_build, CWD, "Build genomic_pca"): return False
    if EIGENSNP_EXECUTABLE.exists():
        logging.info("Successfully built 'genomic_pca' executable."); return True
    else:
        logging.error("Build process finished, but executable not found."); return False

def prepare_input_data():
    logging.info("Checking for input data files...")
    all_present = True
    for ext in [".bed", ".bim", ".fam"]:
        target = RAW_DATA_PREFIX.with_suffix(ext)
        if not target.exists() and (zip_path := RAW_DATA_PREFIX.with_suffix(f"{ext}.zip")).exists():
            logging.info(f"Unzipping {zip_path}..."); zipfile.ZipFile(zip_path, 'r').extractall(target.parent)
        if not target.exists():
            logging.error(f"Required file not found: {target}"); all_present = False
    if not SAMPLE_INFO_FILE.exists():
        logging.error(f"Sample info file not found: {SAMPLE_INFO_FILE}"); all_present = False
    return all_present

# --- Phase 1: Create Sample Lists and Windows ---
def create_sample_lists(fam_df, sample_info_df):
    logging.info("--- Creating Train/Test Sample ID Lists ---")
    merged_df = pd.merge(fam_df, sample_info_df, on='IID', how='left').fillna('Unknown')
    train_indices, test_indices = train_test_split(
        np.arange(len(fam_df)), test_size=TEST_SET_FRACTION, random_state=RANDOM_STATE,
        stratify=merged_df['Superpopulation code'])
    train_df, test_df = fam_df.iloc[train_indices], fam_df.iloc[test_indices]
    logging.info(f"Splitting into {len(train_df)} train samples and {len(test_df)} test samples.")
    train_ids_path = MAIN_OUTPUT_DIR / "train_ids.txt"
    train_df['IID'].to_csv(train_ids_path, header=False, index=False)
    return train_ids_path, train_indices, test_indices

def create_window_file(bim_df, window_size):
    logging.info(f"--- Creating {window_size/1000:.0f}kb window file ---")
    chrom_str = str(bim_df['chrom'].iloc[0])
    max_pos = bim_df['pos'].max()
    windows = [{'chrom': chrom_str, 'start': s, 'end': s + window_size - 1} for s in range(1, max_pos, window_size)]
    window_df = pd.DataFrame(windows)
    window_file_path = MAIN_OUTPUT_DIR / f"windows_{window_size/1000:.0f}kb.bed"
    window_df.to_csv(window_file_path, sep=' ', index=False, header=False)
    logging.info(f"Created {len(window_df)} windows in {window_file_path}"); return window_file_path

# --- Phase 2: PCA Computation ---
def run_combined_pca(train_ids_path, window_file, k):
    logging.info("--- Running Combined PCA: Global (Ground Truth) and Local Models in One Go ---")
    LOCAL_PCA_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    logging.info(f"Clearing previous results from {LOCAL_PCA_OUTPUT_DIR}")
    for f in LOCAL_PCA_OUTPUT_DIR.glob("*"):
        if f.is_file(): f.unlink()
        elif f.is_dir(): shutil.rmtree(f)

    cmd = [
        str(EIGENSNP_EXECUTABLE), "--eigensnp",
        "--bed-file", str(RAW_DATA_PREFIX.with_suffix(".bed")),
        "--ld-block-file", str(window_file),
        "--out", str(PCA_RUN_PREFIX),
        "--eigensnp-k-global", str(k),
        "--save-local-pcs", str(LOCAL_PCA_OUTPUT_DIR),
        "--eigensnp-components-per-block", str(k),
        "--eigensnp-sample-keep-file", str(train_ids_path),
        "--no-filter"
    ]
    return run_command(cmd, CWD, "Run Combined Global and Local PCA")

def compute_train_means_chunked(train_indices):
    logging.info("--- Computing Training Set SNP Means (Memory-Efficient Chunks) ---")
    with open_bed(str(RAW_DATA_PREFIX.with_suffix(".bed"))) as bed:
        n_snps = bed.sid_count
        snp_sums = np.zeros(n_snps, dtype=np.float64)
        snp_counts = np.zeros(n_snps, dtype=np.int32)
        for i in tqdm(range(0, n_snps, SNP_CHUNK_SIZE), desc="Calculating SNP Means"):
            start, end = i, min(i + SNP_CHUNK_SIZE, n_snps)
            geno_chunk = bed.read(index=np.s_[train_indices, start:end], dtype=np.float32)
            snp_sums[start:end] = np.nansum(geno_chunk, axis=0)
            snp_counts[start:end] = np.sum(~np.isnan(geno_chunk), axis=0)
        snp_counts[snp_counts == 0] = 1
        return snp_sums / snp_counts

def project_test_set_python_chunked(test_indices, gt_loadings_df, train_means, full_bim_df):
    logging.info("--- Projecting Test Set onto GT Loadings (Memory-Efficient Chunks) ---")
    with open_bed(str(RAW_DATA_PREFIX.with_suffix(".bed"))) as bed:
        merged = pd.merge(full_bim_df.reset_index(), gt_loadings_df, left_on='sid', right_on='VariantID', how='inner')
        snp_indices_in_bed = merged['index'].values
        loading_cols = [f'PC{i+1}_loading' for i in range(K_COMPONENTS)]
        loadings_matrix = merged[loading_cols].values
        train_means_subset = train_means[snp_indices_in_bed]
        final_projections = np.zeros((len(test_indices), K_COMPONENTS))
        for i in tqdm(range(0, len(snp_indices_in_bed), SNP_CHUNK_SIZE), desc="Projecting Test Set"):
            start, end = i, min(i + SNP_CHUNK_SIZE, len(snp_indices_in_bed))
            snp_chunk_indices = snp_indices_in_bed[start:end]
            loadings_chunk = loadings_matrix[start:end, :]
            means_chunk = train_means_subset[start:end]
            X_test_chunk = bed.read(index=np.s_[test_indices, snp_chunk_indices], dtype=np.float32)
            X_test_chunk -= means_chunk
            final_projections += np.nan_to_num(X_test_chunk) @ loadings_chunk
    return final_projections

# --- Phase 3 & 4: The Core Analysis and Plotting ---
def analyze_local_pca_results(train_means, gt_loadings_df, gt_projections_test, test_indices, full_bim_df, window_file):
    logging.info("--- Evaluating Local PCAs Out-of-Sample (FIXED: Iterating over existing files) ---")
    try:
        windows_df = pd.read_csv(window_file, sep=r'\s+', header=None, names=['chrom', 'start', 'end'])
    except Exception as e:
        logging.error(f"Failed to load window file {window_file}: {e}", exc_info=True); return pd.DataFrame()

    full_bim_with_bed_index = full_bim_df.reset_index().rename(columns={'index': 'bed_index'})
    results = []
    gt_loading_cols = [f'PC{j+1}_loading' for j in range(K_COMPONENTS)]
    gt_dists_raw = pdist(gt_projections_test, 'euclidean')
    gt_dists_z = (gt_dists_raw - gt_dists_raw.mean()) / gt_dists_raw.std() if gt_dists_raw.std() > 1e-9 else np.zeros_like(gt_dists_raw)

    # **FIX**: Iterate over the block files that were actually created, not what we expect to exist.
    existing_block_files = sorted(LOCAL_PCA_OUTPUT_DIR.glob("block_*.local_loadings.tsv"),
                                  key=lambda f: int(re.search(r'block_(\d+)', f.name).group(1)))
    if not existing_block_files:
        logging.error("No block loading files found in the output directory. Cannot perform analysis.")
        return pd.DataFrame()
    logging.info(f"Found {len(existing_block_files)} local PCA loading files to analyze.")

    with open_bed(str(RAW_DATA_PREFIX.with_suffix(".bed"))) as bed:
        for local_loadings_path in tqdm(existing_block_files, desc="Analyzing Windows"):
            try:
                # **FIX**: Parse the block index directly from the filename.
                match = re.search(r'block_(\d+)', local_loadings_path.name)
                if not match: continue
                block_index = int(match.group(1))
                window = windows_df.iloc[block_index]
            except (IndexError, ValueError) as e:
                logging.warning(f"Could not parse block index from {local_loadings_path.name} or find in window file: {e}. Skipping."); continue

            snps_in_block = full_bim_with_bed_index[
                (full_bim_with_bed_index['pos'] >= window['start']) &
                (full_bim_with_bed_index['pos'] <= window['end'])
            ]
            if snps_in_block.empty: continue

            try:
                local_loadings = pd.read_csv(local_loadings_path, sep="\t", header=None, dtype=np.float32).values
            except pd.errors.EmptyDataError:
                logging.warning(f"[Block {block_index}] Loadings file is empty. Skipping."); continue

            if len(snps_in_block) != local_loadings.shape[0]:
                logging.warning(f"[Block {block_index}] SNP count mismatch! Inferred from BIM: {len(snps_in_block)}, "
                                f"Rows in loadings file: {local_loadings.shape[0]}. Tool likely filtered SNPs. Skipping block for safety."); continue

            snp_indices = snps_in_block['bed_index'].values.astype(int)
            X_test_block = bed.read(index=np.s_[test_indices, snp_indices], dtype="float32")
            X_test_block -= train_means[snp_indices]
            local_projections = np.nan_to_num(X_test_block) @ local_loadings

            local_dists_raw = pdist(local_projections, 'euclidean')
            if local_dists_raw.std() < 1e-9:
                logging.warning(f"[Block {block_index}] Local projections have zero variance. Skipping."); continue
            local_dists_z = (local_dists_raw - local_dists_raw.mean()) / local_dists_raw.std()
            dist_mse = np.mean((gt_dists_z - local_dists_z)**2)

            gt_loadings_block = gt_loadings_df[gt_loadings_df.VariantID.isin(snps_in_block['sid'])]
            gt_loadings_matrix = gt_loadings_block[gt_loading_cols].values
            if gt_loadings_matrix.shape[0] != local_loadings.shape[0]: continue

            q_gt, _ = qr(gt_loadings_matrix)
            q_local, _ = qr(local_loadings)
            principal_angles = subspace_angles(q_gt, q_local)
            subspace_dist = norm(np.sin(principal_angles))

            results.append({"block_index": block_index, "chrom": window["chrom"], "start": window["start"], "end": window["end"],
                "num_snps": len(snps_in_block), "dist_mse": dist_mse, "subspace_dist": subspace_dist})
    return pd.DataFrame(results)

def create_plots(results_df, gt_projections_test, test_indices, train_means, full_fam_df, sample_info_df, full_bim_df):
    logging.info("--- Creating final plots ---")
    if results_df.empty: logging.warning("Results DataFrame is empty, cannot create plots."); return

    best_blocks = results_df.sort_values("dist_mse", ascending=True).head(3)
    logging.info(f"Plotting ground truth and the best {len(best_blocks)} blocks based on Distance MSE.")
    test_fam = full_fam_df.iloc[test_indices]
    plot_df = pd.merge(test_fam, sample_info_df, on='IID', how='left').fillna('Unknown')
    full_bim_with_bed_index = full_bim_df.reset_index().rename(columns={'index': 'bed_index'})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 16)); axes = axes.flatten()
    sns.scatterplot(x=gt_projections_test[:, 0], y=gt_projections_test[:, 1], hue=plot_df['Superpopulation code'], alpha=0.8, s=20, ax=axes[0], palette="viridis")
    axes[0].set_title("Ground Truth PCA (Projected Test Set)", fontsize=14, weight='bold')
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(title="Superpop")

    with open_bed(str(RAW_DATA_PREFIX.with_suffix(".bed"))) as bed:
        for i, (idx, block_info) in enumerate(best_blocks.iterrows()):
            ax = axes[i + 1]
            block_id = int(block_info['block_index'])
            snps_in_block = full_bim_with_bed_index[
                (full_bim_with_bed_index['pos'] >= block_info['start']) &
                (full_bim_with_bed_index['pos'] <= block_info['end'])
            ]
            if snps_in_block.empty:
                ax.text(0.5, 0.5, f'Error: Data for block {block_id} not found.', ha='center'); ax.set_title(f"Best Block #{i+1}: Error"); continue

            local_loadings = pd.read_csv(LOCAL_PCA_OUTPUT_DIR / f"block_{block_id}.local_loadings.tsv", sep="\t", header=None, dtype=np.float32).values
            snp_indices = snps_in_block['bed_index'].values
            X_test_block = bed.read(index=np.s_[test_indices, snp_indices], dtype="float32")
            X_test_block -= train_means[snp_indices]
            local_projections = np.nan_to_num(X_test_block) @ local_loadings

            sns.scatterplot(x=local_projections[:, 0], y=local_projections[:, 1], hue=plot_df['Superpopulation code'], alpha=0.8, s=20, ax=ax, palette="viridis", legend=False)
            title = (f"Best Block #{i+1}: chr{block_info['chrom']}:{int(block_info['start'])}-{int(block_info['end'])}\n"
                     f"Dist MSE: {block_info['dist_mse']:.4f} | Subspace Dist: {block_info['subspace_dist']:.4f}")
            ax.set_title(title, fontsize=12); ax.set_xlabel("Local PC1"); ax.set_ylabel("Local PC2")

    for i in range(len(best_blocks) + 1, 4): fig.delaxes(axes[i])
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.suptitle("Out-of-Sample Local PCA Evaluation", fontsize=20, weight='bold')
    fig.savefig(MAIN_OUTPUT_DIR / "local_pca_oos_summary.png", dpi=300)
    logging.info(f"Summary plot saved to {MAIN_OUTPUT_DIR / 'local_pca_oos_summary.png'}")

# --- Main Orchestration ---
def main():
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not all([setup_genomic_pca(), prepare_input_data()]): sys.exit(1)

    logging.info("--- Pre-loading core data files to reduce I/O ---")
    full_fam_df = pd.read_csv(str(RAW_DATA_PREFIX.with_suffix(".fam")), sep=r'\s+', header=None, names=["FID", "IID", "FATHER", "MOTHER", "SEX", "PHENO"], dtype={'IID': str})
    full_bim_df = pd.read_csv(str(RAW_DATA_PREFIX.with_suffix(".bim")), sep='\t', header=None, names=['chrom', 'sid', 'cm', 'pos', 'a1', 'a2'])
    sample_info_df = pd.read_csv(SAMPLE_INFO_FILE, sep='\t', usecols=['Sample name', 'Superpopulation code'], dtype={'Sample name': str}).rename(columns={'Sample name': 'IID'})

    logging.info("Starting analysis pipeline with out-of-sample evaluation.")
    train_ids_path, train_indices, test_indices = create_sample_lists(full_fam_df, sample_info_df)
    window_file = create_window_file(full_bim_df, WINDOW_SIZE)

    # **FIX**: Perform one efficient run for both global and local PCAs.
    if not run_combined_pca(train_ids_path, window_file, K_COMPONENTS):
        logging.error("Halting due to combined PCA generation failure."); sys.exit(1)

    # **FIX**: Load the ground truth loadings from the single run's output.
    try:
        gt_loadings_df = pd.read_csv(GT_LOADINGS_FILE, sep=r'\s+')
        logging.info(f"Successfully loaded Ground Truth loadings from {GT_LOADINGS_FILE}")
    except Exception as e:
        logging.error(f"Failed to read ground truth loadings from {GT_LOADINGS_FILE}: {e}", exc_info=True); sys.exit(1)

    train_means = compute_train_means_chunked(train_indices)
    gt_projections_test = project_test_set_python_chunked(test_indices, gt_loadings_df, train_means, full_bim_df)

    analysis_results_df = analyze_local_pca_results(train_means, gt_loadings_df, gt_projections_test, test_indices, full_bim_df, window_file)
    if analysis_results_df is None or analysis_results_df.empty:
        logging.error("Analysis did not produce any valid results. Halting before plotting."); sys.exit(1)

    analysis_results_df.to_csv(MAIN_OUTPUT_DIR / "local_pca_metrics.csv", index=False)
    logging.info(f"Full metrics saved to {MAIN_OUTPUT_DIR / 'local_pca_metrics.csv'}")

    print("\n" + "="*25 + " Analysis Report " + "="*25)
    print(f"Total windows successfully analyzed: {len(analysis_results_df)}")
    print("\n--- Top 5 Blocks by Pairwise Distance MSE (Lower is Better) ---")
    print(analysis_results_df.sort_values('dist_mse').head(5).to_string(index=False))
    print("\n--- Top 5 Blocks by Subspace Distance (Lower is Better) ---")
    print(analysis_results_df.sort_values('subspace_dist').head(5).to_string(index=False))

    create_plots(analysis_results_df, gt_projections_test, test_indices, train_means, full_fam_df, sample_info_df, full_bim_df)

    print("\n" + "="*25 + " Analysis Complete " + "="*25)

if __name__ == "__main__":
    main()
