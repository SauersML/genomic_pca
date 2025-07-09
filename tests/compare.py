import os
import sys
import subprocess
import time
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from bed_reader import open_bed
from numpy.linalg import eigh
from scipy.stats import chi2 as chi2_dist
from scipy.spatial.distance import pdist
from scipy.linalg import subspace_angles
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# --- Configuration ---

# Paths and Executables
CWD = Path.cwd()
EIGENSNP_PROJECT_DIR = CWD.parent
EIGENSNP_EXECUTABLE = EIGENSNP_PROJECT_DIR / "target" / "release" / "genomic_pca"
PCAONE_SOURCE_DIR = CWD / "PCAone_source"
PCAONE_EXECUTABLE = PCAONE_SOURCE_DIR / "PCAone"
PCAONE_GIT_URL = "https://github.com/Zilong-Li/PCAone.git"

# Input Data
RAW_DATA_PREFIX = CWD.parent / "data" / "chr22_hg38_plink1"
# This will be the new, fully filtered dataset used by all tools
QC_DATA_PREFIX = CWD / "comparison_outputs" / "chr22_subset50.qc"
SAMPLE_INFO_FILE = CWD / "igsr_samples.tsv"
LD_BLOCK_FILE = CWD / "pyrho_EAS_LD_blocks.bed"


# Output Directories
MAIN_OUTPUT_DIR = CWD / "comparison_outputs"
REF_OUTPUT_DIR = MAIN_OUTPUT_DIR / "ref_pca"
EIGENSNP_OUTPUT_DIR = MAIN_OUTPUT_DIR / "eigensnp_pca"
PCAONE_OUTPUT_DIR = MAIN_OUTPUT_DIR / "pcaone_pca"

# PCA & QC Parameters
K_COMPONENTS = 10
CPU_COUNT = os.cpu_count()
QC_MIN_CALL_RATE = 0.98
QC_MIN_MAF = 0.01
QC_MAX_HWE_P = 1e-6

# --- Phase 1: Setup and Tool Preparation ---

def run_command(cmd, work_dir, description):
    """A robust helper to run external commands, capturing output."""
    print(f"--- Running: {description} ---")
    print(f"CMD: {' '.join(map(str, cmd))}")
    print(f"DIR: {work_dir}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            check=True,
            text=True,
            capture_output=True
        )
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr, file=sys.stderr)
        
        duration = time.time() - start_time
        print(f"--- Success: {description} finished in {duration:.2f}s ---")
        return True
    except FileNotFoundError:
        print(f"ERROR: Command not found: {cmd[0]}", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed with exit code {e.returncode}.", file=sys.stderr)
        print(f"STDOUT:\n{e.stdout}", file=sys.stderr)
        print(f"STDERR:\n{e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during '{description}': {e}", file=sys.stderr)
        return False

def prepare_input_data(prefix_path):
    """Checks for genetic data files (.bed, .bim, .fam) and unzips them if necessary."""
    print("Checking for input data files...")
    extensions = [".bed", ".bim", ".fam"]
    all_present = all((prefix_path.with_suffix(ext)).exists() for ext in extensions)

    if all_present:
        print("Input data files found.")
        return True

    print("One or more input files missing. Checking for zip archives...")
    try:
        for ext in extensions:
            zip_path = prefix_path.with_suffix(f"{ext}.zip")
            target_path = prefix_path.with_suffix(ext)
            if not target_path.exists() and zip_path.exists():
                print(f"Unzipping {zip_path} to {target_path.parent}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_path.parent)
            elif not target_path.exists():
                print(f"ERROR: Cannot find {target_path} or {zip_path}", file=sys.stderr)
                return False
        return True
    except Exception as e:
        print(f"ERROR: Failed to prepare input data: {e}", file=sys.stderr)
        return False

def _hwe_pval(a_aa, a_ab, a_bb):
    """Helper to calculate Hardy-Weinberg equilibrium p-value."""
    n = a_aa + a_ab + a_bb
    if n == 0: return 1.0
    p = (2 * a_aa + a_ab) / (2 * n)
    if p == 0 or p == 1: return 1.0
    q = 1.0 - p
    exp = np.array([n * p * p, 2 * n * p * q, n * q * q])
    obs = np.array([a_aa, a_ab, a_bb])
    if (exp == 0).any(): return 0.0
    chi2 = ((obs - exp)**2 / exp).sum()
    return 1.0 - chi2_dist.cdf(chi2, 1)

def create_qc_filtered_dataset(raw_prefix, qc_prefix):
    """
    Filters a genetic dataset for polymorphic sites, call rate, MAF, and HWE.
    This creates a single, definitive dataset that all PCA tools will use.
    """
    print("\n--- Pre-processing: Creating a single QC-filtered dataset ---")
    try:
        print("Calculating SNP QC metrics using bed-reader...")
        bed = open_bed(f"{raw_prefix}.bed", count_A1=False)
        n_snps = bed.sid_count
        
        chunk_size = 5000
        keep_indices = []
        for i in tqdm(range(0, n_snps, chunk_size), desc="SNP QC Filtering"):
            g_chunk = bed.read(index=np.s_[:, i:min(i + chunk_size, n_snps)], dtype='float32')
            
            # Call Rate
            call_rate = np.nanmean(~np.isnan(g_chunk), axis=0)
            
            # MAF
            allele_freq = np.nanmean(g_chunk, axis=0) / 2.0
            maf = np.minimum(allele_freq, 1 - allele_freq)
            
            # HWE
            h0, h1, h2 = np.nansum(g_chunk == 0, axis=0), np.nansum(g_chunk == 1, axis=0), np.nansum(g_chunk == 2, axis=0)
            hwe_p = np.fromiter((_hwe_pval(aa, ab, bb) for aa, ab, bb in zip(h0, h1, h2)), dtype=float, count=g_chunk.shape[1])
            
            # Combine filters
            ok = (call_rate >= QC_MIN_CALL_RATE) & (maf >= QC_MIN_MAF) & (hwe_p > QC_MAX_HWE_P)
            
            chunk_indices = np.arange(i, i + g_chunk.shape[1])
            qc_passed_in_chunk = chunk_indices[ok]
            keep_indices.extend(qc_passed_in_chunk)

        print(f"Identified {len(keep_indices)} QC-passing SNPs out of {n_snps}.")
        if not keep_indices:
             raise ValueError("No SNPs passed the QC filters. Cannot proceed.")

        print("Writing filtered dataset...")
        bim_df = pd.read_csv(f"{raw_prefix}.bim", sep='\t', header=None)
        bim_df.iloc[keep_indices].to_csv(f"{qc_prefix}.bim", sep='\t', header=False, index=False)
        shutil.copy(f"{raw_prefix}.fam", f"{qc_prefix}.fam")
        
        n_samples = bed.iid_count
        bytes_per_snp = (n_samples + 3) // 4
        
        with open(f"{raw_prefix}.bed", "rb") as f_in, open(f"{qc_prefix}.bed", "wb") as f_out:
            f_out.write(f_in.read(3)) # Copy .bed header
            
            current_keep_idx_ptr = 0
            for i in tqdm(range(n_snps), desc="Writing filtered .bed"):
                snp_data = f_in.read(bytes_per_snp)
                if current_keep_idx_ptr < len(keep_indices) and i == keep_indices[current_keep_idx_ptr]:
                    f_out.write(snp_data)
                    current_keep_idx_ptr += 1
        
        print("--- Successfully created QC'd dataset ---")
        return True

    except Exception as e:
        print(f"ERROR: Dataset QC filtering failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False

def prepare_sample_info():
    """Checks if the local sample info file exists."""
    if SAMPLE_INFO_FILE.exists():
        print("Sample info file found locally.")
        return True
    
    print(f"ERROR: Sample info file not found at {SAMPLE_INFO_FILE}", file=sys.stderr)
    print("Please ensure 'igsr_samples.tsv' is in the current directory.", file=sys.stderr)
    return False

def setup_pcaone():
    """Clones and builds PCAone if the executable is not found."""
    if PCAONE_EXECUTABLE.exists():
        print("PCAone executable found. Skipping build.")
        return True

    print("PCAone executable not found. Attempting to clone and build...")
    if not PCAONE_SOURCE_DIR.exists():
        if not run_command(["git", "clone", PCAONE_GIT_URL, str(PCAONE_SOURCE_DIR)], CWD, "Git Clone PCAone"):
            return False
    
    print("Building PCAone...")
    if not run_command(["make", "-j", str(CPU_COUNT)], PCAONE_SOURCE_DIR, "Build PCAone"):
        return False
        
    if not PCAONE_EXECUTABLE.exists():
        print("ERROR: PCAone build succeeded, but executable not found.", file=sys.stderr)
        return False
        
    return True

def setup_eigensnp():
    """Checks for the eigensnp executable and builds it if missing."""
    if EIGENSNP_EXECUTABLE.exists():
        print("eigensnp executable found.")
        return True

    print(f"eigensnp executable not found. Attempting to build...", file=sys.stderr)
    if not (EIGENSNP_PROJECT_DIR / "Cargo.toml").exists():
        print(f"ERROR: Cargo.toml not found in {EIGENSNP_PROJECT_DIR}. Cannot build.", file=sys.stderr)
        return False

    if not run_command(["cargo", "build", "--release", "--features", "openblas-faer"], EIGENSNP_PROJECT_DIR, "Build eigensnp."):
        return False

    if not EIGENSNP_EXECUTABLE.exists():
        print("ERROR: 'cargo build' ran, but the executable is still not at the expected path.", file=sys.stderr)
        return False

    print("eigensnp successfully built.")
    return True

# --- Phase 2: PCA Execution ---

def run_reference_pca():
    """
    Runs a full, exact PCA on the pre-filtered QC'd dataset.
    No internal filtering is needed here.
    """
    print("\n--- Running Reference (Exact) PCA ---")
    REF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        bed = open_bed(f"{QC_DATA_PREFIX}.bed", count_A1=False)
        n_samples, n_variants = bed.iid_count, bed.sid_count
        fam = pd.read_csv(f"{QC_DATA_PREFIX}.fam", sep=r"\s+", header=None, usecols=[1], names=["SampleID"])

        # Read the entire QC'd genotype matrix
        X = bed.read(dtype='float32')
        
        # Center and impute mean for missing values
        col_means = np.nanmean(X, axis=0)
        X -= col_means
        X = np.nan_to_num(X, copy=False)

        # Build the GRM
        print("Building GRM...")
        gram = X @ X.T
        gram /= n_variants
        
        # Eigendecomposition
        print("Performing eigendecomposition...")
        evals_all, evecs_all = eigh(gram)
        
        idx = np.argsort(evals_all)[::-1]
        pcs = evecs_all[:, idx][:, :K_COMPONENTS]
        
        pc_cols = [f"PC{i+1}" for i in range(K_COMPONENTS)]
        df_pcs = pd.DataFrame(pcs, columns=pc_cols)
        df_pcs.insert(0, "SampleID", fam.SampleID)
        
        output_path = REF_OUTPUT_DIR / "ref_pca.tsv"
        df_pcs.to_csv(output_path, sep='\t', index=False, float_format="%.6g")

        duration = time.time() - start_time
        print(f"--- Success: Reference PCA finished in {duration:.2f}s ---")
        return {"tool": "Reference", "runtime": duration, "scores_path": output_path, "success": True}

    except Exception as e:
        print(f"ERROR: Reference PCA failed: {e}", file=sys.stderr)
        return {"tool": "Reference", "runtime": -1, "scores_path": None, "success": False}

def run_eigensnp():
    """
    Runs the eigensnp tool on the QC'd dataset.
    Internal QC filters are disabled to ensure a fair comparison.
    """
    print("\n--- Running eigensnp ---")
    EIGENSNP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = EIGENSNP_OUTPUT_DIR / "eigensnp_results"
    
    # NOTE: We disable internal filtering by setting thresholds to non-restrictive values.
    # This is CRUCIAL for a fair comparison, as QC is now done in the pre-processing step.
    cmd = [
        str(EIGENSNP_EXECUTABLE),
        "--eigensnp",
        "--bed-file", f"{QC_DATA_PREFIX}.bed",
        "--out", str(output_prefix),
        "--eigensnp-k-global", str(K_COMPONENTS),
        "--threads", str(CPU_COUNT),
        "--log-level", "Warn",
        "--ld-block-file", str(LD_BLOCK_FILE),
        # Disable internal QC filters
        "--no-filter"
    ]
    
    start_time = time.time()
    success = run_command(cmd, CWD, "eigensnp Execution")
    duration = time.time() - start_time
    
    scores_path = output_prefix.with_suffix(".eigensnp.pca.tsv")
    if success and not scores_path.exists():
        print(f"ERROR: eigensnp ran but output file not found at {scores_path}", file=sys.stderr)
        success = False

    return {"tool": "eigensnp", "runtime": duration, "scores_path": scores_path if success else None, "success": success}

def run_pcaone():
    """
    Runs the PCAone tool on the pre-filtered QC'd dataset.
    This is now a fair comparison.
    """
    print("\n--- Running PCAone ---")
    PCAONE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = PCAONE_OUTPUT_DIR / "pcaone_results"
    
    cmd = [
        str(PCAONE_EXECUTABLE),
        "-b", str(QC_DATA_PREFIX),   # QC-filtered PLINK prefix
        "-k", str(K_COMPONENTS),     # number of PCs
        "-d", "2",                   # winSVD algorithm
        "-p", "80",                  # ↑ power iterations → ↑ accuracy
        "-o", str(output_prefix),
        "-n", str(CPU_COUNT),        # threads
    ]

    
    start_time = time.time()
    success = run_command(cmd, CWD, "PCAone Execution")
    duration = time.time() - start_time
    
    scores_path = output_prefix.with_suffix(".eigvecs")
    if success and not scores_path.exists():
        print(f"ERROR: PCAone ran but output file not found at {scores_path}", file=sys.stderr)
        success = False

    return {"tool": "PCAone", "runtime": duration, "scores_path": scores_path if success else None, "success": success}

# --- Phase 3: Metric Calculation ---

def load_and_standardize_scores(filepath, tool_name, sample_order):
    """Loads PCA scores and standardizes the format."""
    print(f"Loading scores for {tool_name} from {filepath}...")
    if tool_name == "PCAone":
        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        # PCAone might produce more than K components, so truncate
        df = df.iloc[:, :K_COMPONENTS]
        df.columns = [f"PC{i+1}" for i in range(df.shape[1])]
        df.insert(0, "SampleID", sample_order)
    else:
        df = pd.read_csv(filepath, sep='\t')
        df['SampleID'] = df['SampleID'].astype(str)
        df = df.set_index('SampleID').loc[sample_order].reset_index()
    
    # Ensure exactly K components are used for comparison
    pc_cols = [f"PC{i+1}" for i in range(K_COMPONENTS)]
    if len(df.columns) - 1 < K_COMPONENTS:
        print(f"WARNING: {tool_name} produced fewer than {K_COMPONENTS} PCs. Comparison will use {len(df.columns) - 1} PCs.", file=sys.stderr)
        pc_cols = [col for col in df.columns if col.startswith('PC')]
    
    return df[['SampleID'] + pc_cols]


def calculate_logreg_accuracy(scores_df, sample_info_df):
    """
    Calculates the median normalized balanced accuracy for a given set of PC scores.
    This is an absolute measure of the utility of the PCs for classification.
    """
    print(f"Calculating Logistic Regression Accuracy for {scores_df.attrs.get('tool_name', 'Unknown Tool')}...")
    df = scores_df.merge(sample_info_df, on="SampleID")
    pc_cols = [col for col in scores_df.columns if col.startswith('PC')]
    
    normalized_accuracies = []
    
    for superpop in df['Superpopulation code'].unique():
        if pd.isna(superpop): continue
            
        df_super = df[df['Superpopulation code'] == superpop]
        subpop_counts = df_super['Population code'].value_counts()
        valid_subpops = subpop_counts[subpop_counts >= 2]
        
        if len(valid_subpops) < 2:
            # print(f"Skipping superpop '{superpop}': Not enough sub-populations with >=2 samples.")
            continue
            
        df_super_filt = df_super[df_super['Population code'].isin(valid_subpops.index)]
        X, y = df_super_filt[pc_cols].values, df_super_filt['Population code'].values
        X_scaled = StandardScaler().fit_transform(X)
        
        n_classes = len(valid_subpops)
        chance_level = 1.0 / n_classes
        
        base_logreg = LogisticRegressionCV(
            Cs=10, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            penalty='l2', scoring='balanced_accuracy',
            max_iter=1000, random_state=42
        )
        
        logreg = OneVsRestClassifier(base_logreg, n_jobs=-1)
        
        try:
            logreg.fit(X_scaled, y)
            y_pred = logreg.predict(X_scaled)
            balanced_acc = balanced_accuracy_score(y, y_pred)
            
            normalized_acc = (balanced_acc - chance_level) / (1.0 - chance_level) if chance_level < 1 else 0.0
            normalized_accuracies.append(normalized_acc)
        except Exception as e:
            print(f"LogReg for superpop '{superpop}' failed: {e}")

    return np.median(normalized_accuracies) if normalized_accuracies else np.nan

def calculate_distance_mse(approx_scores_df, ref_scores_df):
    """Calculates MSE between Z-normalized pairwise distance matrices."""
    print(f"Calculating Pairwise Distance MSE vs Reference for {approx_scores_df.attrs.get('tool_name', 'Unknown Tool')}...")
    pc_cols = [col for col in ref_scores_df.columns if col.startswith('PC')]
    
    # Ensure both dataframes have the same columns for comparison
    approx_pc_cols = [col for col in approx_scores_df.columns if col.startswith('PC')]
    common_cols = list(set(pc_cols) & set(approx_pc_cols))
    
    ref_mat, approx_mat = ref_scores_df[common_cols].values, approx_scores_df[common_cols].values
    
    ref_dists = pdist(ref_mat, 'euclidean')
    approx_dists = pdist(approx_mat, 'euclidean')
    
    ref_dists_norm = (ref_dists - np.mean(ref_dists)) / np.std(ref_dists)
    approx_dists_norm = (approx_dists - np.mean(approx_dists)) / np.std(approx_dists)
    
    return np.mean((ref_dists_norm - approx_dists_norm)**2)

def calculate_subspace_distance(approx_scores_df, ref_scores_df):
    """Calculates the sine of the largest principal angle between PC subspaces."""
    print(f"Calculating Subspace Distance vs Reference for {approx_scores_df.attrs.get('tool_name', 'Unknown Tool')}...")
    pc_cols = [col for col in ref_scores_df.columns if col.startswith('PC')]

    # Ensure both dataframes have the same columns for comparison
    approx_pc_cols = [col for col in approx_scores_df.columns if col.startswith('PC')]
    common_cols = list(set(pc_cols) & set(approx_pc_cols))

    q_ref, _ = np.linalg.qr(ref_scores_df[common_cols].values)
    q_approx, _ = np.linalg.qr(approx_scores_df[common_cols].values)
    angles = subspace_angles(q_ref, q_approx)
    return np.sin(np.max(angles))

# --- Phase 4: Main Orchestration and Reporting ---

def main():
    """Main function to run the entire comparison suite."""
    print("====== PCA Comparison Suite Start ======")
    
    # 1. Setup
    MAIN_OUTPUT_DIR.mkdir(exist_ok=True)
    if not setup_eigensnp() or not setup_pcaone() or not prepare_input_data(RAW_DATA_PREFIX) or not prepare_sample_info():
        sys.exit("Halting due to setup failure.")

    # 2. Pre-processing
    if not create_qc_filtered_dataset(RAW_DATA_PREFIX, QC_DATA_PREFIX):
        sys.exit("Halting due to dataset QC filtering failure.")

    # 3. Execution
    ref_result = run_reference_pca()
    if not ref_result["success"]:
        sys.exit("Reference PCA failed. Cannot proceed with comparison.")
        
    eigensnp_result = run_eigensnp()
    pcaone_result = run_pcaone()
    
    all_results = [ref_result, eigensnp_result, pcaone_result]
    
    # 4. Data Loading
    print("\n--- Loading and Standardizing All PCA Scores ---")
    try:
        sample_info_df = pd.read_csv(SAMPLE_INFO_FILE, sep='\t', dtype={'Sample name': str})
        sample_info_df = sample_info_df.rename(columns={'Sample name': 'SampleID'})
        
        ref_fam = pd.read_csv(f"{QC_DATA_PREFIX}.fam", sep=r'\s+', header=None, usecols=[1], names=["SampleID"], dtype=str)
        canonical_sample_order = ref_fam.SampleID.tolist()

        scores_dfs = {}
        for res in all_results:
            if res["success"]:
                df = load_and_standardize_scores(res["scores_path"], res["tool"], canonical_sample_order)
                df.attrs['tool_name'] = res["tool"] # Attach tool name for logging
                scores_dfs[res["tool"]] = df
            
    except Exception as e:
        print(f"ERROR: FATAL ERROR during data loading: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Metric Calculation
    print("\n--- Calculating Comparison Metrics ---")
    report_data = []
    ref_scores = scores_dfs.get("Reference")

    for tool_result in all_results:
        tool_name = tool_result["tool"]
        if not tool_result["success"]:
            report_data.append({"Tool": tool_name, "Runtime (s)": "FAILED", "LogReg Accuracy (Median Norm.)": np.nan, "Pairwise Distance MSE": np.nan, "Subspace Distance": np.nan})
            continue

        current_scores = scores_dfs.get(tool_name)
        if current_scores is None:
            print(f"WARNING: Scores for {tool_name} could not be loaded. Skipping metrics.", file=sys.stderr)
            report_data.append({"Tool": tool_name, "Runtime (s)": tool_result['runtime'], "LogReg Accuracy (Median Norm.)": np.nan, "Pairwise Distance MSE": np.nan, "Subspace Distance": np.nan})
            continue
        
        metrics = {"Tool": tool_name, "Runtime (s)": tool_result['runtime']}

        # Absolute metric: Calculated for ALL tools.
        metrics["LogReg Accuracy (Median Norm.)"] = calculate_logreg_accuracy(current_scores, sample_info_df)

        # Comparative metrics: Calculated vs Reference. For Reference itself, this is 0.
        if tool_name == "Reference":
            metrics["Pairwise Distance MSE"] = 0.0
            metrics["Subspace Distance"] = 0.0
        else:
            metrics["Pairwise Distance MSE"] = calculate_distance_mse(current_scores, ref_scores)
            metrics["Subspace Distance"] = calculate_subspace_distance(current_scores, ref_scores)
            
        report_data.append(metrics)

    # 6. Reporting
    print("\n\n" + "="*20 + " Final Comparison Report " + "="*20)
    report_df = pd.DataFrame(report_data).set_index("Tool")
    
    # Reorder rows for clarity
    tool_order = ["Reference", "eigensnp", "PCAone"]
    report_df = report_df.reindex([t for t in tool_order if t in report_df.index])

    for col in report_df.columns:
        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
        if "Accuracy" in col or "MSE" in col or "Distance" in col:
            report_df[col] = report_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    report_df["Runtime (s)"] = report_df["Runtime (s)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "FAILED")

    print(report_df.to_string())
    print("\n" + "="*25 + " PCA Comparison Suite Finished " + "="*25)

if __name__ == "__main__":
    main()
