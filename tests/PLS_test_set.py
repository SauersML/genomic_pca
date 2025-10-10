import os
import io
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
# Keep bed_reader available in the environment (as requested).
from bed_reader import open_bed  # noqa: F401
import pgenlib
from scipy.linalg import eigh
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# ============================================================
#                       CONFIG / CONSTANTS
# ============================================================
OUT_DIR  = Path("fast_pca_transfer").resolve()
DATA_DIR = OUT_DIR / "data"

CHUNK_SNPS = 5000
K_PCS      = 15
K_LVS      = 15
PLS_TARGET = "Population"

URLS = {
    "igsr_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/igsr_samples.tsv",
    "whitelist_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
}

# ============================================================
#                    STAGE / PRINT HELPERS
# ============================================================
class StageTracker:
    def __init__(self, stage_names):
        self.stage_names = stage_names
        self.total = len(stage_names)
        self.idx = 0
    def start(self, name=None):
        self.idx += 1
        name = name or (self.stage_names[self.idx - 1] if self.idx - 1 < len(self.stage_names) else "Unnamed")
        print(f"\n=== [Stage {self.idx}/{self.total}] {name} ===", flush=True)
    def note(self, msg):
        print(f"  → {msg}", flush=True)

# ============================================================
#                          IO HELPERS
# ============================================================
def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request, zipfile  # local scope; no conditional imports
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = r.getheader("Content-Length")
        total = int(total) if total is not None else None
        with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=f"Downloading {dest.name}", leave=False, dynamic_ncols=True) as pbar:
            while True:
                chunk = r.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

# ============================================================
#                          READERS (PLINK 2)
# ============================================================
def read_psam(prefix: Path) -> pd.DataFrame:
    psam_path = prefix.with_suffix(".psam")
    df = pd.read_csv(psam_path, sep=r"\s+", dtype=str)
    if "#FID" in df.columns: df = df.rename(columns={"#FID": "FID"})
    if "#IID" in df.columns: df = df.rename(columns={"#IID": "IID"})
    df["IID"] = df["IID"].astype(str).str.strip()
    return df

def read_pvar(prefix: Path) -> pd.DataFrame:
    pvar_path = prefix.with_suffix(".pvar")
    with open(pvar_path, "r") as f:
        lines = [ln for ln in f if not ln.startswith("##")]
    pvar = pd.read_csv(io.StringIO("".join(lines)), sep=r"\s+", dtype=str)
    if "#CHROM" in pvar.columns:
        pvar = pvar.rename(columns={"#CHROM": "CHROM"})
    chrom_str = (
        pvar["CHROM"].astype(str).str.strip()
        .str.replace(r"^chr", "", regex=True)
        .str.upper()
    )
    chrom_num = pd.to_numeric(chrom_str, errors="coerce")
    pos_num   = pd.to_numeric(pvar["POS"].astype(str).str.strip(), errors="coerce")
    out = pvar.copy()
    out["chrom_norm"] = chrom_num.astype("Int64")
    out["pos"] = pos_num.astype("Int64")
    return out

def read_igsr(igsr_path: Path) -> pd.DataFrame:
    df = pd.read_csv(igsr_path, sep="\t", dtype=str)
    df["SampleID"] = df["Sample name"].astype(str).str.strip()
    df["Population"] = df["Population code"].astype(str).str.strip()
    df["Superpopulation"] = df["Superpopulation code"].astype(str).str.strip()
    df["PopulationNameLong"] = df.get("Population name", df["Population"])
    return df[["SampleID","Population","PopulationNameLong","Superpopulation"]]

def read_whitelist(tsv_path: Path) -> pd.DataFrame:
    wl_raw = pd.read_csv(tsv_path, sep="\t", usecols=["CHROM","POS"], dtype=str)
    chrom_str = (
        wl_raw["CHROM"].str.strip()
        .str.replace(r"^chr", "", regex=True)
        .str.upper()
    )
    chrom_num = pd.to_numeric(chrom_str, errors="coerce")
    pos_num   = pd.to_numeric(wl_raw["POS"].str.strip(), errors="raise")
    wl = pd.DataFrame({
        "chrom_norm": chrom_num.astype("Int64"),
        "pos": pos_num.astype("Int64"),
    })
    wl = wl[(wl["chrom_norm"] >= 1) & (wl["chrom_norm"] <= 22)]
    return wl[["chrom_norm","pos"]]

# ============================================================
#                 SNP SELECTION / SHARED KEEP MASK
# ============================================================
def compute_keep_mask(pvar: pd.DataFrame, wl: pd.DataFrame) -> np.ndarray:
    print("  • Computing whitelist ∩ PVAR (restricted to chr22) …", flush=True)
    n_snps = len(pvar)
    wl_22 = wl[wl["chrom_norm"] == 22]
    pvar_idxed = pvar.reset_index().rename(columns={"index":"sidx"})
    allowed = pvar_idxed.merge(wl_22[["chrom_norm","pos"]], on=["chrom_norm","pos"], how="inner")
    keep_indices = allowed["sidx"].to_numpy(dtype=int)
    keep_mask = np.zeros(n_snps, dtype=bool)
    keep_mask[keep_indices] = True
    print(f"    └─ PVAR SNPs: {len(pvar):,} | Whitelist chr22: {len(wl_22):,} | Kept: {keep_mask.sum():,}", flush=True)
    return keep_mask

# ============================================================
#                PGEN ACCESS + MEANS (TRAIN) + STREAMING
# ============================================================
def _open_pgen(prefix: Path):
    pgen_path_bytes = os.fsencode(str(prefix.with_suffix(".pgen")))
    return pgenlib.PgenReader(pgen_path_bytes)

def _counts_and_samples(prefix: Path):
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    pgen.close()
    return n_samples, n_snps

def compute_train_means(prefix: Path, keep_mask: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """
    Streaming computation of per-SNP means over TRAIN samples only (kept SNPs).
    Returns a vector mu of length M_kept.
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    assert train_mask.shape[0] == n_samples

    m_kept = int(keep_mask.sum())
    sum_vec = np.zeros(m_kept, dtype=np.float64)
    cnt_vec = np.zeros(m_kept, dtype=np.int64)

    scanned = tqdm(total=n_snps, desc="Means: scanning variants", unit="SNP", leave=False, dynamic_ncols=True)
    kp = 0  # running pointer across kept SNPs
    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]
        scanned.update(end - start)
        if not submask.any():
            continue

        width = end - start
        buf = np.empty((n_samples, width), dtype=np.int8, order="C")
        pgen.read_range(start, end, buf, allele_idx=1, sample_maj=1)

        X = buf.astype(np.float32, copy=False)[train_mask, :][:, submask]
        X[X < 0] = np.nan
        valid = ~np.isnan(X)
        sum_vec[kp:kp+X.shape[1]] += np.nansum(X, axis=0)
        cnt_vec[kp:kp+X.shape[1]] += valid.sum(axis=0)
        kp += X.shape[1]

    scanned.close()
    pgen.close()

    cnt_safe = np.maximum(cnt_vec, 1)
    mu = (sum_vec / cnt_safe).astype(np.float32)
    return mu

def build_grm_train(prefix: Path, keep_mask: np.ndarray, train_mask: np.ndarray, mu_kept: np.ndarray):
    """
    Build GRM over TRAIN samples using precomputed mu_kept (means across TRAIN).
    Returns (GRAM, kept_total, m_kept).
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    idx_train = np.where(train_mask)[0]
    n_train = idx_train.size

    gram = np.zeros((n_train, n_train), dtype=np.float64)
    kept_total = 0
    m_kept = int(keep_mask.sum())

    scanned = tqdm(total=n_snps, desc="GRM: scanning variants", unit="SNP", leave=False, dynamic_ncols=True)
    accumulated = tqdm(total=m_kept, desc="GRM: accumulating kept SNPs", unit="SNP", leave=False, dynamic_ncols=True)

    kp = 0
    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]
        scanned.update(end - start)
        if not submask.any():
            continue

        width = end - start
        buf = np.empty((n_samples, width), dtype=np.int8, order="C")
        pgen.read_range(start, end, buf, allele_idx=1, sample_maj=1)

        X = buf.astype(np.float32, copy=False)[idx_train, :][:, submask]
        X[X < 0] = np.nan
        mu_chunk = mu_kept[kp:kp+X.shape[1]]
        X -= mu_chunk
        np.nan_to_num(X, copy=False)
        gram += X @ X.T
        kept_total += X.shape[1]
        accumulated.update(X.shape[1])
        kp += X.shape[1]

    scanned.close()
    accumulated.close()
    pgen.close()

    gram /= max(kept_total, 1)
    return gram, kept_total, m_kept

def project_pcs(prefix: Path, keep_mask: np.ndarray, mu_kept: np.ndarray,
                train_mask: np.ndarray, proj_mask: np.ndarray,
                V: np.ndarray, evals: np.ndarray, k: int):
    """
    Project PROJ samples onto TRAIN PCs computed from GRM.
    Uses cross-similarity g = (1/m) X_proj X_train^T with the same training means.
    Returns (pcs_proj: n_proj × k).
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    idx_train = np.where(train_mask)[0]
    idx_proj  = np.where(proj_mask)[0]
    n_train, n_proj = idx_train.size, idx_proj.size
    m_kept = int(keep_mask.sum())

    cross = np.zeros((n_proj, n_train), dtype=np.float64)

    scanned = tqdm(total=n_snps, desc="PCA project: scanning variants", unit="SNP", leave=False, dynamic_ncols=True)
    kp = 0
    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]
        scanned.update(end - start)
        if not submask.any():
            continue

        width = end - start
        buf = np.empty((n_samples, width), dtype=np.int8, order="C")
        pgen.read_range(start, end, buf, allele_idx=1, sample_maj=1)

        Xt = buf.astype(np.float32, copy=False)[idx_train, :][:, submask]
        Xp = buf.astype(np.float32, copy=False)[idx_proj,  :][:, submask]
        Xt[Xt < 0] = np.nan
        Xp[Xp < 0] = np.nan
        mu_chunk = mu_kept[kp:kp+Xt.shape[1]]
        Xt -= mu_chunk
        Xp -= mu_chunk
        np.nan_to_num(Xt, copy=False)
        np.nan_to_num(Xp, copy=False)
        cross += Xp @ Xt.T
        kp += Xt.shape[1]

    scanned.close()
    pgen.close()

    cross /= max(m_kept, 1)
    k_eff = min(k, V.shape[1], np.count_nonzero(evals > 0))
    denom = np.sqrt(np.maximum(evals[:k_eff], 1e-12))
    pcs_proj = cross @ (V[:, :k_eff] / denom)
    return pcs_proj

def build_X_for(prefix: Path, keep_mask: np.ndarray, mu_kept: np.ndarray, sample_mask: np.ndarray):
    """
    Materialize centered X (samples × kept SNPs) for the specified samples using training means.
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    idx = np.where(sample_mask)[0]

    cols = []
    scanned = tqdm(total=n_snps, desc="X build: scanning variants", unit="SNP", leave=False, dynamic_ncols=True)
    kp = 0
    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]
        scanned.update(end - start)
        if not submask.any():
            continue

        width = end - start
        buf = np.empty((n_samples, width), dtype=np.int8, order="C")
        pgen.read_range(start, end, buf, allele_idx=1, sample_maj=1)

        X = buf.astype(np.float32, copy=False)[idx, :][:, submask]
        X[X < 0] = np.nan
        mu_chunk = mu_kept[kp:kp+X.shape[1]]
        X -= mu_chunk
        np.nan_to_num(X, copy=False)
        cols.append(X)
        kp += X.shape[1]

    scanned.close()
    pgen.close()
    if len(cols) == 0:
        return np.zeros((idx.size, 0), dtype="float32")
    X_all = np.concatenate(cols, axis=1).astype("float32", copy=False)
    return X_all

def pca_from_grm(gram: np.ndarray, k: int):
    print("  • Eigen-decomposing GRM (eigh) → PCs …", flush=True)
    evals_all, evecs_all = eigh(gram)
    k_eff = min(k, gram.shape[0] - 1) if gram.shape[0] > 1 else 0
    if k_eff <= 0:
        return np.zeros((gram.shape[0], 0), dtype=np.float32), np.array([], dtype=np.float64)
    evals = evals_all[-k_eff:][::-1]
    evecs = evecs_all[:, -k_eff:][:, ::-1]
    pcs   = evecs * np.sqrt(evals)
    return pcs, evals

def encode_one_hot(series: pd.Series):
    classes = series.fillna("UNK").astype(str)
    Y = pd.get_dummies(classes, drop_first=False)
    return Y.to_numpy(dtype=np.float32), list(Y.columns)

def fit_pls_da(X_train: np.ndarray, y_series: pd.Series, max_lv: int):
    Y, class_names = encode_one_hot(y_series)
    k_eff = max(1, min(max_lv, X_train.shape[0] - 1, Y.shape[1]))
    pls = PLSRegression(n_components=k_eff, scale=False)
    pls.fit(X_train, Y)
    LV_train = pls.x_scores_
    return pls, LV_train, class_names

# ============================================================
#                            PLOTTING
# ============================================================
SUPERPOP_COLORS = {
    "AFR": "#e74c3c", "EUR": "#1f77b4", "EAS": "#9467bd",
    "SAS": "#2ca02c", "AMR": "#ff7f0e", "OTH": "#7f7f7f", "": "#7f7f7f"
}

def pop_color_map(series: pd.Series):
    uniq = sorted(series.fillna("").unique().tolist())
    cmap = plt.get_cmap("tab20")
    return {p: cmap(i % 20) for i, p in enumerate(uniq)}

def make_plots_2x2(
    pca_df: pd.DataFrame,
    pca_umap_df: pd.DataFrame,
    pls_df: pd.DataFrame,
    pls_umap_df: pd.DataFrame,
    evals_train: np.ndarray,
    out_png: Path
):
    from matplotlib.lines import Line2D
    all_pops = pd.concat(
        [pca_df["Population"], pca_umap_df["Population"], pls_df["Population"], pls_umap_df["Population"]],
        ignore_index=True
    )
    pop_colors = pop_color_map(all_pops)
    sp_colors  = SUPERPOP_COLORS

    def scatter(ax, x, y, pop_series, sp_series, title, xlabel, ylabel, with_legends: bool):
        fc = pop_series.map(pop_colors).tolist()
        ec = sp_series.map(lambda s: sp_colors.get(s, "#7f7f7f")).tolist()
        ax.scatter(x, y, s=18, alpha=0.85, c=fc, edgecolors=ec, linewidths=0.6, rasterized=True)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        if with_legends:
            sp_handles = [
                Line2D([0],[0], marker='o', linestyle='', markerfacecolor='white',
                       markeredgecolor=color, markeredgewidth=1.2, label=sp)
                for sp, color in sp_colors.items()
            ]
            leg_sp = ax.legend(handles=sp_handles, title="Superpopulation (edgecolor)",
                               frameon=False, fontsize=8, loc="best")
            ax.add_artist(leg_sp)
            unique_pops = list(pop_colors.keys())
            max_lab = 20
            pop_handles = [
                Line2D([0],[0], marker='o', linestyle='',
                       markerfacecolor=pop_colors[p], markeredgecolor='k', label=p)
                for p in unique_pops[:max_lab]
            ]
            ax.legend(handles=pop_handles,
                      title=("Population (facecolor)" if len(unique_pops) <= max_lab
                             else f"Population (facecolor; first {max_lab})"),
                      frameon=False, fontsize=7, loc="lower left")

    total_var = float(np.sum(evals_train)) if len(evals_train) else 1.0
    pct = [(100.0 * ev / total_var) for ev in evals_train[:2]] if len(evals_train) >= 2 else [0.0, 0.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_pca, ax_pca_umap, ax_pls, ax_pls_umap = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    scatter(
        ax_pca,
        pca_df["PC1"], pca_df["PC2"],
        pca_df["Population"], pca_df["Superpopulation"],
        f"PCA: PC1 vs PC2",
        f"PC1 ({pct[0]:.2f}% var)", f"PC2 ({pct[1]:.2f}% var)",
        with_legends=False
    )

    scatter(
        ax_pca_umap,
        pca_umap_df["UMAP1"], pca_umap_df["UMAP2"],
        pca_umap_df["Population"], pca_umap_df["Superpopulation"],
        f"UMAP on PCA 1..{K_PCS}",
        "UMAP1", "UMAP2",
        with_legends=True
    )

    scatter(
        ax_pls,
        pls_df["LV1"], pls_df["LV2"],
        pls_df["Population"], pls_df["Superpopulation"],
        "PLS-DA: LV1 vs LV2",
        "LV1", "LV2",
        with_legends=False
    )

    scatter(
        ax_pls_umap,
        pls_umap_df["UMAP1"], pls_umap_df["UMAP2"],
        pls_umap_df["Population"], pls_umap_df["Superpopulation"],
        f"UMAP on PLS 1..{min(K_LVS, pls_df.filter(like='LV').shape[1])}",
        "UMAP1", "UMAP2",
        with_legends=False
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"[plot] saved {out_png}")

def fit_umap_2d(X):
    n_samples = X.shape[0]
    n_neighbors = max(2, min(15, n_samples - 1))  # robust when projecting small cohorts
    reducer = UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, random_state=42, low_memory=True)
    return reducer.fit_transform(X)

# ============================================================
#                            MAIN (EXPERIMENTS)
# ============================================================
def main():
    stages = [
        "Initialize output/data directories",
        "Download inputs (TSV metadata/whitelist)",
        "Read metadata (PSAM/PVAR/IGSR/whitelist) + join",
        "Compute SNP keep mask (whitelist ∩ PVAR@chr22)",
        "Experiment A: Train on non-SAS, project SAS",
        "Experiment B: Train on 80%, project 20%"
    ]
    ST = StageTracker(stages)

    # ---- 1) init dirs
    ST.start()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_DIR:  {OUT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")

    # ---- 2) download inputs
    ST.start()
    igsr_tsv = DATA_DIR / "igsr_samples.tsv"
    white_tsv = DATA_DIR / "GSAv2_hg38.tsv"
    if not igsr_tsv.exists():
        ST.note(f"Fetching IGSR sample metadata → {igsr_tsv.name}")
        download(URLS["igsr_tsv"], igsr_tsv)
    if not white_tsv.exists():
        ST.note(f"Fetching array whitelist (hg38) → {white_tsv.name}")
        download(URLS["whitelist_tsv"], white_tsv)

    prefix = Path("hg38_chr22").resolve()

    # ---- 3) read metadata + join
    ST.start()
    ST.note("Reading PSAM …")
    psam = read_psam(prefix)
    ST.note("Reading PVAR …")
    pvar = read_pvar(prefix)
    ST.note("Reading IGSR …")
    igsr = read_igsr(igsr_tsv)
    ST.note("Reading whitelist …")
    wl = read_whitelist(white_tsv)

    # Join IGSR onto PSAM by sample ID (IID)
    ann = psam.rename(columns={"IID":"SampleID"}).merge(igsr, on="SampleID", how="left")
    ann["Population"] = ann["Population"].fillna("UNK")
    ann["Superpopulation"] = ann["Superpopulation"].fillna("OTH")

    # ---- 4) keep mask
    ST.start()
    keep_mask = compute_keep_mask(pvar, wl)
    n_samples, n_snps = _counts_and_samples(prefix)
    m_kept = int(keep_mask.sum())
    print(f"Samples: {n_samples} | SNPs kept: {m_kept}")

    # ========================================================
    #         Experiment A: Train on non-SAS, project SAS
    # ========================================================
    ST.start()
    out_A = OUT_DIR / "exp_A_nonSAS_train_SAS_project"
    out_A.mkdir(parents=True, exist_ok=True)

    is_SAS = (ann["Superpopulation"].values == "SAS")
    train_mask_A = ~is_SAS
    proj_mask_A  =  is_SAS

    print(f"[A] Train (non-SAS): {train_mask_A.sum()}  |  Project (SAS): {proj_mask_A.sum()}")

    # Means (train only) → GRM → PCs (train)
    ST.note("[A] Computing training means (non-SAS) …")
    mu_A = compute_train_means(prefix, keep_mask, train_mask_A)
    ST.note("[A] Building GRM (non-SAS) …")
    gram_A, kept_total_A, _ = build_grm_train(prefix, keep_mask, train_mask_A, mu_A)
    ST.note("[A] Eigendecomposition → PCs (train) …")
    pcs_train_A, evals_A = pca_from_grm(gram_A, K_PCS)

    # Project SAS onto those PCs
    ST.note("[A] Projecting SAS onto non-SAS PCs …")
    pcs_proj_A = project_pcs(prefix, keep_mask, mu_A, train_mask_A, proj_mask_A, pcs_train_A / np.sqrt(np.maximum(evals_A, 1e-12)) if pcs_train_A.size else pcs_train_A, evals_A, K_PCS)
    # Above: V = pcs_train / sqrt(λ); but pca_from_grm returned scores = V*sqrt(λ).
    # Recover eigenvectors V (train) as pcs_train / sqrt(λ).
    V_A = pcs_train_A / np.sqrt(np.maximum(evals_A, 1e-12)) if pcs_train_A.size else pcs_train_A
    pcs_proj_A = project_pcs(prefix, keep_mask, mu_A, train_mask_A, proj_mask_A, V_A, evals_A, K_PCS)

    pc_cols_A = [f"PC{i+1}" for i in range(pcs_proj_A.shape[1])]
    sas_ids = ann.loc[proj_mask_A, "SampleID"].values
    df_pca_A = pd.DataFrame(pcs_proj_A, columns=pc_cols_A)
    df_pca_A.insert(0, "SampleID", sas_ids)
    df_pca_A = df_pca_A.merge(ann[["SampleID","Population","Superpopulation"]], on="SampleID", how="left")

    # PLS: fit on non-SAS, transform SAS
    ST.note("[A] Building X (train=non-SAS) …")
    X_train_A = build_X_for(prefix, keep_mask, mu_A, train_mask_A)
    y_train_A = ann.loc[train_mask_A, PLS_TARGET].reset_index(drop=True)
    ST.note("[A] Fitting PLS-DA (train=non-SAS) …")
    pls_A, LV_train_A, class_names_A = fit_pls_da(X_train_A, y_train_A, K_LVS)
    ST.note("[A] Building X (project=SAS) …")
    X_proj_A = build_X_for(prefix, keep_mask, mu_A, proj_mask_A)
    ST.note("[A] Transforming SAS with trained PLS-DA …")
    LV_proj_A = pls_A.transform(X_proj_A)
    lv_cols_A = [f"LV{i+1}" for i in range(LV_proj_A.shape[1])]
    df_pls_A = pd.DataFrame(LV_proj_A, columns=lv_cols_A)
    df_pls_A.insert(0, "SampleID", sas_ids)
    df_pls_A = df_pls_A.merge(ann[["SampleID","Population","Superpopulation"]], on="SampleID", how="left")

    # UMAP on projected points only (PCA & PLS)
    umap_pca_A = fit_umap_2d(df_pca_A[pc_cols_A[:min(K_PCS, len(pc_cols_A))]].to_numpy()) if pc_cols_A else np.zeros((len(df_pca_A), 2))
    umap_pls_A = fit_umap_2d(df_pls_A[lv_cols_A[:min(K_LVS, len(lv_cols_A))]].to_numpy()) if lv_cols_A else np.zeros((len(df_pls_A), 2))

    df_umap_pca_A = df_pca_A[["SampleID","Population","Superpopulation"]].copy()
    df_umap_pca_A["UMAP1"] = umap_pca_A[:,0] if umap_pca_A.size else []
    df_umap_pca_A["UMAP2"] = umap_pca_A[:,1] if umap_pca_A.size else []

    df_umap_pls_A = df_pls_A[["SampleID","Population","Superpopulation"]].copy()
    df_umap_pls_A["UMAP1"] = umap_pls_A[:,0] if umap_pls_A.size else []
    df_umap_pls_A["UMAP2"] = umap_pls_A[:,1] if umap_pls_A.size else []

    # Write and plot (SAS only)
    (out_A / "pca_projected_SAS.tsv").write_text(
        df_pca_A[["SampleID"] + pc_cols_A].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    (out_A / "pls_projected_SAS.tsv").write_text(
        df_pls_A[["SampleID"] + lv_cols_A].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evals_A))], "Eigenvalue": evals_A}).to_csv(
        out_A / "eigenvalues_train_nonSAS.tsv", sep="\t", index=False, float_format="%.6g"
    )

    make_plots_2x2(
        pca_df=df_pca_A[["SampleID","Population","Superpopulation","PC1","PC2"]] if pcs_proj_A.shape[1] >= 2 else
                df_pca_A.assign(PC1=0.0, PC2=0.0)[["SampleID","Population","Superpopulation","PC1","PC2"]],
        pca_umap_df=df_umap_pca_A,
        pls_df=df_pls_A[["SampleID","Population","Superpopulation","LV1","LV2"]] if LV_proj_A.shape[1] >= 2 else
               df_pls_A.assign(LV1=0.0, LV2=0.0)[["SampleID","Population","Superpopulation","LV1","LV2"]],
        pls_umap_df=df_umap_pls_A,
        evals_train=evals_A,
        out_png=out_A / "plots_transfer_SAS_only_2x2.png"
    )

    # ========================================================
    #         Experiment B: Train on 80%, project 20%
    # ========================================================
    ST.start()
    out_B = OUT_DIR / "exp_B_split_80_20"
    out_B.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    idx_all = np.arange(len(ann))
    rng.shuffle(idx_all)
    split = int(0.8 * len(idx_all))
    idx_train_B = np.sort(idx_all[:split])
    idx_proj_B  = np.sort(idx_all[split:])
    train_mask_B = np.zeros(len(ann), dtype=bool); train_mask_B[idx_train_B] = True
    proj_mask_B  = np.zeros(len(ann), dtype=bool); proj_mask_B[idx_proj_B]  = True

    print(f"[B] Train (80%): {train_mask_B.sum()}  |  Project (20%): {proj_mask_B.sum()}")

    # Means → GRM → PCs (train on 80%)
    ST.note("[B] Computing training means (80%) …")
    mu_B = compute_train_means(prefix, keep_mask, train_mask_B)
    ST.note("[B] Building GRM (80%) …")
    gram_B, kept_total_B, _ = build_grm_train(prefix, keep_mask, train_mask_B, mu_B)
    ST.note("[B] Eigendecomposition → PCs (train 80%) …")
    pcs_train_B, evals_B = pca_from_grm(gram_B, K_PCS)
    V_B = pcs_train_B / np.sqrt(np.maximum(evals_B, 1e-12)) if pcs_train_B.size else pcs_train_B

    # Project 20%
    ST.note("[B] Projecting holdout (20%) onto 80% PCs …")
    pcs_proj_B = project_pcs(prefix, keep_mask, mu_B, train_mask_B, proj_mask_B, V_B, evals_B, K_PCS)
    pc_cols_B = [f"PC{i+1}" for i in range(pcs_proj_B.shape[1])]
    ids_B = ann.loc[proj_mask_B, "SampleID"].values
    df_pca_B = pd.DataFrame(pcs_proj_B, columns=pc_cols_B)
    df_pca_B.insert(0, "SampleID", ids_B)
    df_pca_B = df_pca_B.merge(ann[["SampleID","Population","Superpopulation"]], on="SampleID", how="left")

    # PLS: fit on 80%, transform 20%
    ST.note("[B] Building X (train=80%) …")
    X_train_B = build_X_for(prefix, keep_mask, mu_B, train_mask_B)
    y_train_B = ann.loc[train_mask_B, PLS_TARGET].reset_index(drop=True)
    ST.note("[B] Fitting PLS-DA (train=80%) …")
    pls_B, LV_train_B, class_names_B = fit_pls_da(X_train_B, y_train_B, K_LVS)
    ST.note("[B] Building X (project=20%) …")
    X_proj_B = build_X_for(prefix, keep_mask, mu_B, proj_mask_B)
    ST.note("[B] Transforming holdout (20%) with trained PLS-DA …")
    LV_proj_B = pls_B.transform(X_proj_B)
    lv_cols_B = [f"LV{i+1}" for i in range(LV_proj_B.shape[1])]
    df_pls_B = pd.DataFrame(LV_proj_B, columns=lv_cols_B)
    df_pls_B.insert(0, "SampleID", ids_B)
    df_pls_B = df_pls_B.merge(ann[["SampleID","Population","Superpopulation"]], on="SampleID", how="left")

    # UMAP on projected points only (PCA & PLS)
    umap_pca_B = fit_umap_2d(df_pca_B[pc_cols_B[:min(K_PCS, len(pc_cols_B))]].to_numpy()) if pc_cols_B else np.zeros((len(df_pca_B), 2))
    umap_pls_B = fit_umap_2d(df_pls_B[lv_cols_B[:min(K_LVS, len(lv_cols_B))]].to_numpy()) if lv_cols_B else np.zeros((len(df_pls_B), 2))

    df_umap_pca_B = df_pca_B[["SampleID","Population","Superpopulation"]].copy()
    df_umap_pca_B["UMAP1"] = umap_pca_B[:,0] if umap_pca_B.size else []
    df_umap_pca_B["UMAP2"] = umap_pca_B[:,1] if umap_pca_B.size else []

    df_umap_pls_B = df_pls_B[["SampleID","Population","Superpopulation"]].copy()
    df_umap_pls_B["UMAP1"] = umap_pls_B[:,0] if umap_pls_B.size else []
    df_umap_pls_B["UMAP2"] = umap_pls_B[:,1] if umap_pls_B.size else []

    # Write and plot (holdout only)
    (out_B / "pca_projected_holdout.tsv").write_text(
        df_pca_B[["SampleID"] + pc_cols_B].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    (out_B / "pls_projected_holdout.tsv").write_text(
        df_pls_B[["SampleID"] + lv_cols_B].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evals_B))], "Eigenvalue": evals_B}).to_csv(
        out_B / "eigenvalues_train_80.tsv", sep="\t", index=False, float_format="%.6g"
    )

    make_plots_2x2(
        pca_df=df_pca_B[["SampleID","Population","Superpopulation","PC1","PC2"]] if pcs_proj_B.shape[1] >= 2 else
                df_pca_B.assign(PC1=0.0, PC2=0.0)[["SampleID","Population","Superpopulation","PC1","PC2"]],
        pca_umap_df=df_umap_pca_B,
        pls_df=df_pls_B[["SampleID","Population","Superpopulation","LV1","LV2"]] if LV_proj_B.shape[1] >= 2 else
               df_pls_B.assign(LV1=0.0, LV2=0.0)[["SampleID","Population","Superpopulation","LV1","LV2"]],
        pls_umap_df=df_umap_pls_B,
        evals_train=evals_B,
        out_png=out_B / "plots_holdout20_only_2x2.png"
    )

    # Short epilogue
    print("\n=== Summary ===")
    print(f"[A] Files in: {out_A}")
    print(f"[B] Files in: {out_B}")

if __name__ == "__main__":
    main()
