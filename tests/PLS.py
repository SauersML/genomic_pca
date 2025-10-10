import os
import sys
import zipfile
import urllib.request
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
from tqdm import tqdm
from bed_reader import open_bed
from scipy.linalg import eigh
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression

# ============================================================
#                       CONFIG / CONSTANTS
# ============================================================
OUT_DIR  = Path("fast_pca_out").resolve()
DATA_DIR = OUT_DIR / "data"

CHUNK_SNPS = 5000     # streaming chunk size (variants per read)
K_PCS      = 15       # number of PCs to compute; also drives UMAP(PCA)
K_LVS      = 15       # max number of PLS latent variables; also drives UMAP(PLS)
PLS_TARGET = "Population"   # class label used to supervise PLS ("Population" or "Superpopulation")

URLS = {
    "igsr_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/igsr_samples.tsv",
    "whitelist_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
}

# ============================================================
#                    STAGE / PRINT HELPERS
# ============================================================
class StageTracker:
    """Tiny helper to print stage banners."""
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
    """
    Stream a URL to disk with a byte-aware progress bar.
    Shows % complete and bytes transferred.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = r.getheader("Content-Length")
        total = int(total) if total is not None else None
        desc = f"Downloading {dest.name}"
        with tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=desc, leave=False, dynamic_ncols=True
        ) as pbar:
            while True:
                chunk = r.read(1024 * 64)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))

def unzip(zip_path: Path, out_dir: Path):
    """
    Extract a ZIP with progress over total uncompressed bytes.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()
        total_bytes = sum(i.file_size for i in infos)
        with tqdm(
            total=total_bytes, unit="B", unit_scale=True, unit_divisor=1024,
            desc=f"Unzipping {zip_path.name}", leave=False, dynamic_ncols=True
        ) as pbar:
            for info in infos:
                zf.extract(info, out_dir)
                pbar.update(info.file_size)

# ============================================================
#                          READERS
# ============================================================
def read_fam(prefix: Path) -> pd.DataFrame:
    fam = pd.read_csv(
        prefix.with_suffix(".fam"),
        sep=r"\s+",
        header=None,
        names=["FID","IID","PAT","MAT","SEX","PHENO"],
        dtype=str
    )
    fam["IID"] = fam["IID"].astype(str).str.strip()
    return fam

def read_bim(prefix: Path) -> pd.DataFrame:
    bim = pd.read_csv(
        prefix.with_suffix(".bim"),
        sep=r"\s+",
        header=None,
        names=["chrom","sid","cm","pos","a1","a2"]
    )
    bim["chrom_norm"] = pd.to_numeric(bim["chrom"], errors="coerce").astype("Int64")
    bim["pos"] = pd.to_numeric(bim["pos"], errors="coerce").astype("Int64")
    return bim

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
    chrom_num = pd.to_numeric(chrom_str, errors="coerce")  # non-numeric (X/Y/MT) → NaN
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
def compute_keep_mask(prefix: Path, bim: pd.DataFrame, wl: pd.DataFrame) -> np.ndarray:
    """
    Return a boolean mask over all SNPs in the BED indicating those in whitelist∩BIM (here chr22).
    Prints a concise report of overlap cardinality.
    """
    print("  • Computing whitelist ∩ BIM (restricted to chr22) …", flush=True)
    bed = open_bed(str(prefix.with_suffix(".bed")), count_A1=False)
    n_snps = bed.sid_count

    wl_22 = wl[wl["chrom_norm"] == 22]
    bim_idxed = bim.reset_index().rename(columns={"index":"sidx"})
    allowed = bim_idxed.merge(wl_22[["chrom_norm","pos"]], on=["chrom_norm","pos"], how="inner")
    keep_indices = allowed["sidx"].to_numpy(dtype=int)

    keep_mask = np.zeros(n_snps, dtype=bool)
    keep_mask[keep_indices] = True

    print(f"    └─ BIM SNPs: {len(bim):,} | Whitelist chr22: {len(wl_22):,} | Kept: {keep_mask.sum():,}", flush=True)
    return keep_mask

# ============================================================
#                PCA: GRM CONSTRUCTION (STREAMED)
# ============================================================
def build_grm_from_mask(prefix: Path, keep_mask: np.ndarray):
    """
    Stream the genotype matrix and accumulate the GRM using only kept SNPs.
    Dual progress bars:
      • Variants scanned (all SNPs)
      • Kept SNPs accumulated (effective work)
    """
    bed = open_bed(str(prefix.with_suffix(".bed")), count_A1=False)
    n_samples, n_snps = bed.iid_count, bed.sid_count

    gram = np.zeros((n_samples, n_samples), dtype=np.float64)
    kept_total = 0

    total_kept_target = int(keep_mask.sum())
    scanned = tqdm(total=n_snps, desc="Streaming variants (scan)", unit="SNP",
                   leave=False, dynamic_ncols=True)
    accumulated = tqdm(total=total_kept_target, desc="Accumulating kept SNPs", unit="SNP",
                       leave=False, dynamic_ncols=True)

    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]

        # update "scanned" bar regardless (we swept these SNPs)
        scanned.update(end - start)

        if not submask.any():
            continue

        X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")  # (n_samples, width)
        X = X[:, submask]
        means = np.nanmean(X, axis=0)
        X -= means
        np.nan_to_num(X, copy=False)  # NaNs→0 post-centering
        gram += X @ X.T
        kept_total += X.shape[1]
        accumulated.update(X.shape[1])

    scanned.close()
    accumulated.close()

    gram /= max(kept_total, 1)
    return gram, kept_total

def pca_from_grm(gram: np.ndarray, k: int):
    """Eigen-decompose the GRM and return top-k PC scores and eigenvalues."""
    print("  • Eigen-decomposing GRM (eigh) → PCs …", flush=True)
    evals_all, evecs_all = eigh(gram)  # ascending eigenvalues
    k_eff = min(k, gram.shape[0] - 1)
    evals = evals_all[-k_eff:][::-1]
    evecs = evecs_all[:, -k_eff:][:, ::-1]
    pcs   = evecs * np.sqrt(evals)     # scores = V * sqrt(Λ)
    return pcs, evals

# ============================================================
#                 PLS-DA: BUILD X AND FIT SUPERVISED
# ============================================================
def build_X_from_mask(prefix: Path, keep_mask: np.ndarray):
    """
    Materialize the centered genotype matrix X over kept SNPs.
    Telemetry mirrors GRM construction (scanned vs kept SNPs).
    """
    bed = open_bed(str(prefix.with_suffix(".bed")), count_A1=False)
    n_samples, n_snps = bed.iid_count, bed.sid_count

    cols = []
    kept_total = 0

    total_kept_target = int(keep_mask.sum())
    scanned = tqdm(total=n_snps, desc="Streaming variants (X scan)", unit="SNP",
                   leave=False, dynamic_ncols=True)
    accumulated = tqdm(total=total_kept_target, desc="Building X (kept SNPs)", unit="SNP",
                       leave=False, dynamic_ncols=True)

    for start in range(0, n_snps, CHUNK_SNPS):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]

        scanned.update(end - start)

        if not submask.any():
            continue

        X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")  # (n_samples, width)
        X = X[:, submask]
        means = np.nanmean(X, axis=0)
        X -= means
        np.nan_to_num(X, copy=False)
        cols.append(X)  # horizontally concatenate later
        kept_total += X.shape[1]
        accumulated.update(X.shape[1])

    scanned.close()
    accumulated.close()

    if len(cols) == 0:
        return np.zeros((bed.iid_count, 0), dtype="float32"), 0

    X_all = np.concatenate(cols, axis=1).astype("float32", copy=False)
    return X_all, kept_total

def encode_one_hot(series: pd.Series):
    """One-hot encode labels (including 'UNK' if present); returns Y (n×K) and class order."""
    classes = series.fillna("UNK").astype(str)
    Y = pd.get_dummies(classes, drop_first=False)
    return Y.to_numpy(dtype=np.float32), list(Y.columns)

def fit_pls_da(X: np.ndarray, Y: np.ndarray, max_lv: int):
    """
    Fit PLS with one-hot Y for classification (PLS-DA).
    Returns the PLS model and X scores (LVs).
    """
    n, p = X.shape
    k_eff = max(1, min(max_lv, n - 1, Y.shape[1]))  # ≤ classes, ≤ n−1
    print(f"  • Fitting PLS-DA with up to {k_eff} latent variables …", flush=True)
    pls = PLSRegression(n_components=k_eff, scale=False)  # X already centered; Y is 0/1
    pls.fit(X, Y)
    LV = pls.x_scores_  # (n × k_eff)
    return pls, LV

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
    evals: np.ndarray,
    out_png: Path
):
    from matplotlib.lines import Line2D

    # Harmonize color maps across all panels
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
            # Superpopulation legend (edgecolor)
            sp_handles = [
                Line2D([0],[0], marker='o', linestyle='', markerfacecolor='white',
                       markeredgecolor=color, markeredgewidth=1.2, label=sp)
                for sp, color in sp_colors.items()
            ]
            leg_sp = ax.legend(handles=sp_handles, title="Superpopulation (edgecolor)",
                               frameon=False, fontsize=8, loc="best")
            ax.add_artist(leg_sp)
            # Population legend (facecolor), capped
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

    # Compute % variance for PC axes (for labels)
    total_var = float(np.sum(evals)) if len(evals) else 1.0
    pct = [(100.0 * ev / total_var) for ev in evals[:2]] if len(evals) >= 2 else [0.0, 0.0]

    # 2×2 figure: PCA (PC1–PC2), UMAP(PCA), PLS (LV1–LV2), UMAP(PLS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax_pca, ax_pca_umap, ax_pls, ax_pls_umap = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    scatter(
        ax_pca,
        pca_df["PC1"], pca_df["PC2"],
        pca_df["Population"], pca_df["Superpopulation"],
        f"PCA: PC1 vs PC2",
        f"PC1 ({pct[0]:.2f}% var)" if len(pct) > 0 else "PC1",
        f"PC2 ({pct[1]:.2f}% var)" if len(pct) > 1 else "PC2",
        with_legends=False
    )

    scatter(
        ax_pca_umap,
        pca_umap_df["UMAP1"], pca_umap_df["UMAP2"],
        pca_umap_df["Population"], pca_umap_df["Superpopulation"],
        f"UMAP on PCA 1..{K_PCS}",
        "UMAP1", "UMAP2",
        with_legends=True  # place legends once, here
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

# ============================================================
#                            MAIN
# ============================================================
def main():
    stage_names = [
        "Initialize output/data directories",
        "Download inputs (TSV metadata/whitelist)",
        "Read metadata (FAM/BIM/IGSR/whitelist)",
        "Compute SNP keep mask (whitelist ∩ BIM@chr22)",
        "PCA pipeline: build GRM (streamed) and decompose",
        "Write PCA outputs",
        "PLS-DA pipeline: build X (streamed) and fit",
        "Write PLS outputs",
        "UMAP embeddings on PCs and LVs",
        "Unified plotting (2×2 panel)",
        "Console summary"
    ]
    ST = StageTracker(stage_names)

    # ---- 1) init dirs
    ST.start()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ST.note(f"OUT_DIR:  {OUT_DIR}")
    ST.note(f"DATA_DIR: {DATA_DIR}")

    # ---- 2) download inputs
    ST.start()
    igsr_tsv = DATA_DIR / "igsr_samples.tsv"
    white_tsv = DATA_DIR / "GSAv2_hg38.tsv"

    ST.note(f"Fetching IGSR sample metadata → {igsr_tsv.name}")
    download(URLS["igsr_tsv"], igsr_tsv)
    ST.note(f"Fetching array whitelist (hg38) → {white_tsv.name}")
    download(URLS["whitelist_tsv"], white_tsv)

    prefix = "/home/user/agents/data/fast_pca_out/data/hg38_chr22"

    # ---- 3) read metadata
    ST.start()
    ST.note("Reading FAM …")
    fam = read_fam(prefix)
    ST.note("Reading BIM …")
    bim = read_bim(prefix)
    ST.note("Reading IGSR metadata …")
    igsr = read_igsr(igsr_tsv)
    ST.note("Reading whitelist …")
    wl  = read_whitelist(white_tsv)

    # ---- 4) shared SNP mask
    ST.start()
    keep_mask = compute_keep_mask(prefix, bim, wl)

    # ---- 5) PCA: GRM + eigen-decomp
    ST.start()
    ST.note("Building GRM from kept SNPs (streamed) …")
    gram, kept_total = build_grm_from_mask(prefix, keep_mask)
    ST.note("Decomposing GRM → top PCs …")
    pcs, evals = pca_from_grm(gram, K_PCS)

    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pcs_df = pd.DataFrame(pcs, columns=pc_cols)
    pcs_df.insert(0, "SampleID", fam["IID"].values)

    # annotate with populations/superpops
    ann_pca = pcs_df.merge(igsr, left_on="SampleID", right_on="SampleID", how="left")
    ann_pca["Population"] = ann_pca["Population"].fillna("UNK")
    ann_pca["Superpopulation"] = ann_pca["Superpopulation"].fillna("OTH")

    # ---- 6) write PCA outputs
    ST.start()
    ST.note("Writing PCA scores and eigenvalues …")
    (OUT_DIR / "pca.tsv").write_text(
        ann_pca[["SampleID"] + pc_cols].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    pd.DataFrame({"PC": pc_cols[:len(evals)], "Eigenvalue": evals}).to_csv(
        OUT_DIR / "eigenvalues.tsv", sep="\t", index=False, float_format="%.6g"
    )

    # ---- 7) PLS-DA pipeline
    ST.start()
    ST.note("Materializing X on kept SNPs (streamed) …")
    X, kept_total_pls = build_X_from_mask(prefix, keep_mask)
    if kept_total_pls != kept_total:
        print(f"[warn] kept_total differs between GRM and X paths: {kept_total} vs {kept_total_pls}")

    # make PLS target labels using the same annotation table
    ann_labels = ann_pca[["SampleID","Population","Superpopulation"]].copy()
    if PLS_TARGET not in ann_labels.columns:
        raise ValueError(f"PLS_TARGET '{PLS_TARGET}' not in annotation columns")
    Y, class_names = encode_one_hot(ann_labels[PLS_TARGET])

    ST.note("Fitting PLS-DA …")
    pls_model, LV = fit_pls_da(X, Y, max_lv=K_LVS)
    lv_cols = [f"LV{i+1}" for i in range(LV.shape[1])]
    pls_df = pd.DataFrame(LV, columns=lv_cols)
    pls_df.insert(0, "SampleID", ann_labels["SampleID"].values)

    ann_pls = pls_df.merge(ann_labels, on="SampleID", how="left")  # preserve Population & Superpopulation

    # ---- 8) write PLS outputs
    ST.start()
    ST.note("Writing PLS-DA scores …")
    (OUT_DIR / "pls.tsv").write_text(
        ann_pls[["SampleID"] + lv_cols].to_csv(sep="\t", index=False, float_format="%.6g")
    )

    # ---- 9) UMAP on PCs and LVs
    ST.start()
    ST.note("Fitting UMAP(2D) on PCA scores …")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, low_memory=True)

    X_pca_umap = ann_pca[pc_cols[:min(K_PCS, len(pc_cols))]].to_numpy()
    emb_pca = reducer.fit_transform(X_pca_umap)
    umap_pca_df = ann_pca[["SampleID","Population","Superpopulation"]].copy()
    umap_pca_df["UMAP1"] = emb_pca[:, 0]
    umap_pca_df["UMAP2"] = emb_pca[:, 1]

    ST.note("Fitting UMAP(2D) on PLS LVs …")
    X_pls_umap = ann_pls[lv_cols[:min(K_LVS, len(lv_cols))]].to_numpy()
    emb_pls = reducer.fit_transform(X_pls_umap)
    umap_pls_df = ann_pls[["SampleID","Population","Superpopulation"]].copy()
    umap_pls_df["UMAP1"] = emb_pls[:, 0]
    umap_pls_df["UMAP2"] = emb_pls[:, 1]

    # ---- 10) unified 2×2 figure
    ST.start()
    ST.note("Rendering composite figure …")
    make_plots_2x2(
        pca_df=ann_pca[["SampleID","Population","Superpopulation","PC1","PC2"]],
        pca_umap_df=umap_pca_df,
        pls_df=ann_pls[["SampleID","Population","Superpopulation","LV1","LV2"]],
        pls_umap_df=umap_pls_df,
        evals=evals,
        out_png=OUT_DIR / "plots_pca_pls_2x2.png"
    )

    # ---- 11) console summary
    ST.start()
    total_var = float(np.sum(evals)) if len(evals) else 1.0
    pct1 = 100.0 * (evals[0] / total_var) if len(evals) >= 1 else 0.0
    pct2 = 100.0 * (evals[1] / total_var) if len(evals) >= 2 else 0.0

    print(f"Samples: {len(ann_pca)}")
    print(f"SNPs used (whitelist ∩ BIM@chr22): {kept_total}")
    print(f"PCA  PC1: {pct1:.2f}%   PC2: {pct2:.2f}% of GRM variance")
    print(f"PLS-DA components (LVs): {LV.shape[1]}  |  classes ({PLS_TARGET}): {len(class_names)}")
    print("Outputs:")
    print(f"  - {OUT_DIR/'pca.tsv'}")
    print(f"  - {OUT_DIR/'eigenvalues.tsv'}")
    print(f"  - {OUT_DIR/'pls.tsv'}")
    print(f"  - {OUT_DIR/'plots_pca_pls_2x2.png'}")

if __name__ == "__main__":
    main()
