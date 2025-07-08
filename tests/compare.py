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
RAW_DATA_PREFIX = CWD.parent / "data" / "chr22_subset50"
# This will be the new, filtered dataset used by all tools
QC_DATA_PREFIX = CWD / "comparison_outputs" / "chr22_subset50.qc"
SAMPLE_INFO_FILE = CWD / "igsr_samples.tsv"
LD_BLOCK_FILE = CWD / "pyrho_EAS_LD_blocks.bed"


# Output Directories
MAIN_OUTPUT_DIR = CWD / "comparison_outputs"
REF_OUTPUT_DIR = MAIN_OUTPUT_DIR / "ref_pca"
EIGENSNP_OUTPUT_DIR = MAIN_OUTPUT_DIR / "eigensnp_pca"
PCAONE_OUTPUT_DIR = MAIN_OUTPUT_DIR / "pcaone_pca"

# PCA Parameters
K_COMPONENTS = 10
CPU_COUNT = os.cpu_count()

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

def filter_monomorphic_sites(raw_prefix, qc_prefix):
    """
    Filters a genetic dataset to remove monomorphic sites (MAF=0), using
    bed-reader for efficient MAF calculation.
    """
    print("\n--- Pre-processing: Filtering for polymorphic sites ---")
    try:
        print("Calculating MAF using bed-reader...")
        bed = open_bed(f"{raw_prefix}.bed", count_A1=False)
        n_snps = bed.sid_count
        
        chunk_size = 10000 
        keep_indices = []
        for i in tqdm(range(0, n_snps, chunk_size), desc="Calculating MAF"):
            g_chunk = bed.read(index=np.s_[:, i:min(i + chunk_size, n_snps)], dtype='float32')
            allele_freq = np.nanmean(g_chunk, axis=0) / 2.0
            maf = np.minimum(allele_freq, 1 - allele_freq)
            chunk_indices = np.arange(i, i + g_chunk.shape[1])
            polymorphic_in_chunk = chunk_indices[maf > 0]
            keep_indices.extend(polymorphic_in_chunk)

        print(f"Identified {len(keep_indices)} polymorphic SNPs out of {n_snps}.")

        print("Writing filtered dataset...")
        bim_df = pd.read_csv(f"{raw_prefix}.bim", sep='\t', header=None)
        bim_df.iloc[keep_indices].to_csv(f"{qc_prefix}.bim", sep='\t', header=False, index=False)
        shutil.copy(f"{raw_prefix}.fam", f"{qc_prefix}.fam")
        
        n_samples = bed.iid_count
        bytes_per_snp = (n_samples + 3) // 4
        
        with open(f"{raw_prefix}.bed", "rb") as f_in, open(f"{qc_prefix}.bed", "wb") as f_out:
            f_out.write(f_in.read(3))
            
            current_keep_idx_ptr = 0
            for i in tqdm(range(n_snps), desc="Writing filtered .bed"):
                snp_data = f_in.read(bytes_per_snp)
                if current_keep_idx_ptr < len(keep_indices) and i == keep_indices[current_keep_idx_ptr]:
                    f_out.write(snp_data)
                    current_keep_idx_ptr += 1
        
        print("--- Successfully created QC'd dataset ---")
        return True

    except Exception as e:
        print(f"ERROR: MAF filtering failed: {e}", file=sys.stderr)
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

    if not run_command(["cargo", "build", "--release"], EIGENSNP_PROJECT_DIR, "Build eigensnp (cargo build --release)"):
        return False

    if not EIGENSNP_EXECUTABLE.exists():
        print("ERROR: 'cargo build' ran, but the executable is still not at the expected path.", file=sys.stderr)
        return False

    print("eigensnp successfully built.")
    return True

# --- Phase 2: PCA Execution ---

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

def run_reference_pca():
    """Runs a full, exact PCA by building the GRM on the QC'd dataset."""
    print("\n--- Running Reference (Exact) PCA ---")
    REF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        bed = open_bed(f"{QC_DATA_PREFIX}.bed", count_A1=False)
        n_samples, n_variants = bed.iid_count, bed.sid_count
        fam = pd.read_csv(f"{QC_DATA_PREFIX}.fam", sep=r"\s+", header=None, usecols=[1], names=["SampleID"])

        gram = np.zeros((n_samples, n_samples), dtype=np.float64)
        kept_variants = 0
        
        min_call_rate, min_maf, max_hwe_p, min_var_eps = 0.98, 0.01, 1e-6, 1e-9
        chunk_size = 2000

        for i in tqdm(range(0, n_variants, chunk_size), desc="Building GRM"):
            X = bed.read(index=np.s_[:, i:min(i + chunk_size, n_variants)], dtype='float32', order='C')

            call_rate = np.nanmean(~np.isnan(X), axis=0)
            maf = np.nanmean(X, axis=0) / 2.0
            maf = np.where(maf > 0.5, 1 - maf, maf)
            h0, h1, h2 = np.nansum(X == 0, axis=0), np.nansum(X == 1, axis=0), np.nansum(X == 2, axis=0)
            hwe_p = np.fromiter((_hwe_pval(aa, ab, bb) for aa, ab, bb in zip(h0, h1, h2)), dtype=float, count=X.shape[1])
            var = np.nanvar(X, axis=0, ddof=1)
            ok = (call_rate >= min_call_rate) & (maf >= min_maf) & (hwe_p > max_hwe_p) & (var > min_var_eps)

            if ok.any():
                X_good = X[:, ok]
                X_good -= np.nanmean(X_good, axis=0)
                X_good = np.nan_to_num(X_good, copy=False)
                gram += X_good @ X_good.T
                kept_variants += X_good.shape[1]

        if kept_variants == 0: raise ValueError("No variants passed QC for reference PCA.")
        
        gram /= kept_variants
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
    """Runs the eigensnp tool on the QC'd dataset."""
    print("\n--- Running eigensnp ---")
    EIGENSNP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = EIGENSNP_OUTPUT_DIR / "eigensnp_results"
    
    cmd = [
        str(EIGENSNP_EXECUTABLE),
        "--eigensnp",
        "--bed-file", f"{QC_DATA_PREFIX}.bed",
        "--out", str(output_prefix),
        "--eigensnp-k-global", str(K_COMPONENTS),
        "--threads", str(CPU_COUNT),
        "--log-level", "Warn",
        "--eigensnp-min-call-rate", "0.98",
        "--eigensnp-min-maf", "0.01",
        "--eigensnp-max-hwe-p", "1e-6",
        "--ld-block-file", str(LD_BLOCK_FILE)
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
    """Runs the PCAone tool on the QC'd dataset."""
    print("\n--- Running PCAone ---")
    PCAONE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = PCAONE_OUTPUT_DIR / "pcaone_results"
    
    cmd = [
        str(PCAONE_EXECUTABLE),
        "-b", str(QC_DATA_PREFIX),
        "-k", str(K_COMPONENTS),
        "-o", str(output_prefix),
        "-n", str(CPU_COUNT),
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
        df.columns = [f"PC{i+1}" for i in range(df.shape[1])]
        df.insert(0, "SampleID", sample_order)
    else:
        df = pd.read_csv(filepath, sep='\t')
        df['SampleID'] = df['SampleID'].astype(str)
        df = df.set_index('SampleID').loc[sample_order].reset_index()
    
    pc_cols_to_keep = ['SampleID'] + [f"PC{i+1}" for i in range(K_COMPONENTS)]
    return df[pc_cols_to_keep]

def calculate_logreg_accuracy(approx_scores_df, ref_scores_df, sample_info_df):
    """Calculates the median normalized balanced accuracy using OneVsRestClassifier."""
    print("Calculating Logistic Regression Accuracy...")
    df = approx_scores_df.merge(sample_info_df, on="SampleID")
    pc_cols = [f"PC{i+1}" for i in range(K_COMPONENTS)]
    
    normalized_accuracies = []
    
    for superpop in df['Superpopulation code'].unique():
        if pd.isna(superpop): continue
            
        df_super = df[df['Superpopulation code'] == superpop]
        subpop_counts = df_super['Population code'].value_counts()
        valid_subpops = subpop_counts[subpop_counts >= 2]
        
        if len(valid_subpops) < 2:
            print(f"Skipping superpop '{superpop}': Not enough sub-populations with >=2 samples.")
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
    print("Calculating Pairwise Distance MSE...")
    pc_cols = [f"PC{i+1}" for i in range(K_COMPONENTS)]
    ref_mat, approx_mat = ref_scores_df[pc_cols].values, approx_scores_df[pc_cols].values
    
    ref_dists = pdist(ref_mat, 'euclidean')
    approx_dists = pdist(approx_mat, 'euclidean')
    
    ref_dists_norm = (ref_dists - np.mean(ref_dists)) / np.std(ref_dists)
    approx_dists_norm = (approx_dists - np.mean(approx_dists)) / np.std(approx_dists)
    
    return np.mean((ref_dists_norm - approx_dists_norm)**2)

def calculate_subspace_distance(approx_scores_df, ref_scores_df):
    """Calculates the sine of the largest principal angle between PC subspaces."""
    print("Calculating Subspace Distance...")
    pc_cols = [f"PC{i+1}" for i in range(K_COMPONENTS)]
    q_ref, _ = np.linalg.qr(ref_scores_df[pc_cols].values)
    q_approx, _ = np.linalg.qr(approx_scores_df[pc_cols].values)
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
    if not filter_monomorphic_sites(RAW_DATA_PREFIX, QC_DATA_PREFIX):
        sys.exit("Halting due to MAF filtering failure.")

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

        scores_dfs = {
            res["tool"]: load_and_standardize_scores(res["scores_path"], res["tool"], canonical_sample_order)
            for res in all_results if res["success"]
        }
    except Exception as e:
        print(f"ERROR: FATAL ERROR during data loading: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 5. Metric Calculation
    print("\n--- Calculating Comparison Metrics ---")
    report_data = [{"Tool": "Reference", "Runtime (s)": ref_result['runtime'], "LogReg Accuracy (Median Norm.)": 1.0, "Pairwise Distance MSE": 0.0, "Subspace Distance": 0.0}]
    
    ref_scores = scores_dfs.get("Reference")
    
    for tool_result in [eigensnp_result, pcaone_result]:
        tool_name = tool_result["tool"]
        if not tool_result["success"]:
            report_data.append({"Tool": tool_name, "Runtime (s)": "FAILED", "LogReg Accuracy (Median Norm.)": np.nan, "Pairwise Distance MSE": np.nan, "Subspace Distance": np.nan})
            continue

        approx_scores = scores_dfs.get(tool_name)
        metrics = {"Tool": tool_name, "Runtime (s)": tool_result['runtime']}
        metrics["LogReg Accuracy (Median Norm.)"] = calculate_logreg_accuracy(approx_scores, ref_scores, sample_info_df)
        metrics["Pairwise Distance MSE"] = calculate_distance_mse(approx_scores, ref_scores)
        metrics["Subspace Distance"] = calculate_subspace_distance(approx_scores, ref_scores)
        report_data.append(metrics)

    # 6. Reporting
    print("\n\n" + "="*20 + " Final Comparison Report " + "="*20)
    report_df = pd.DataFrame(report_data).set_index("Tool")
    
    for col in report_df.columns:
        report_df[col] = pd.to_numeric(report_df[col], errors='coerce')
        if "Accuracy" in col or "MSE" in col or "Distance" in col:
            report_df[col] = report_df[col].apply(lambda x: f"{x:.6f}" if pd.notna(x) else "N/A")
    report_df["Runtime (s)"] = report_df["Runtime (s)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "FAILED")

    print(report_df.to_string())
    print("\n" + "="*25 + " PCA Comparison Suite Finished " + "="*25)

if __name__ == "__main__":
    main()
