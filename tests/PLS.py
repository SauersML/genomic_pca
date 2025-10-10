import os
import sys
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from bed_reader import open_bed
from scipy.linalg import eigh
from umap import UMAP
import matplotlib.pyplot as plt

# ------------------ Fixed paths/URLs ------------------
OUT_DIR  = Path("fast_pca_out").resolve()
DATA_DIR = OUT_DIR / "data"
CHUNK_SNPS = 5000   # streaming chunk size (variants per read)
K_PCS = 15          # compute top 15 PCs and use them for UMAP

URLS = {
    "bed_zip": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bed.zip",
    "bim_zip": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bim.zip",
    "fam_zip": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.fam.zip",
    "igsr_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/igsr_samples.tsv",
    "whitelist_tsv": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
}

# ------------------ IO helpers (no conditionals) ------------------
def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())

def unzip(zip_path: Path, out_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

# ------------------ Data readers ------------------
def read_fam(prefix: Path) -> pd.DataFrame:
    fam = pd.read_csv(prefix.with_suffix(".fam"), sep=r"\s+", header=None,
                      names=["FID","IID","PAT","MAT","SEX","PHENO"], dtype=str)
    fam["IID"] = fam["IID"].astype(str).str.strip()   # <- fixed: .str.strip()
    return fam

def read_bim(prefix: Path) -> pd.DataFrame:
    bim = pd.read_csv(prefix.with_suffix(".bim"), sep=r"\s+", header=None,
                      names=["chrom","sid","cm","pos","a1","a2"])
    bim["chrom_norm"] = pd.to_numeric(bim["chrom"], errors="coerce").astype("Int64")
    return bim

def read_igsr(igsr_path: Path) -> pd.DataFrame:
    df = pd.read_csv(igsr_path, sep="\t", dtype=str)
    df["SampleID"] = df["Sample name"].astype(str).str.strip()
    df["Population"] = df["Population code"].astype(str).str.strip()
    df["Superpopulation"] = df["Superpopulation code"].astype(str).str.strip()
    df["PopulationNameLong"] = df.get("Population name", df["Population"])
    return df[["SampleID","Population","PopulationNameLong","Superpopulation"]]

def read_whitelist(tsv_path: Path) -> pd.DataFrame:
    # Read as strings; normalize chromosome labels; keep autosomes only (1..22)
    wl_raw = pd.read_csv(tsv_path, sep="\t", usecols=["CHROM", "POS"], dtype=str)

    chrom_str = (
        wl_raw["CHROM"]
        .str.strip()
        .str.replace(r"^chr", "", regex=True)
        .str.upper()
    )
    chrom_num = pd.to_numeric(chrom_str, errors="coerce")  # non-numeric (X/Y/MT) → NaN

    pos_num = pd.to_numeric(wl_raw["POS"].str.strip(), errors="raise")

    wl = pd.DataFrame({
        "chrom_norm": chrom_num.astype("Int64"),
        "pos": pos_num.astype("Int64"),
    })

    wl = wl[(wl["chrom_norm"] >= 1) & (wl["chrom_norm"] <= 22)]
    return wl[["chrom_norm", "pos"]]

# ------------------ GRM from whitelisted SNPs ------------------
def build_grm_whitelist(prefix: Path, bim: pd.DataFrame, wl: pd.DataFrame):
    bed = open_bed(str(prefix.with_suffix(".bed")), count_A1=False)
    n_samples, n_snps = bed.iid_count, bed.sid_count

    # Intersect whitelist with BIM (this dataset is chr22, so restrict whitelist to chrom 22)
    wl_22 = wl[wl["chrom_norm"] == 22]
    bim_idxed = bim.reset_index().rename(columns={"index":"sidx"})
    allowed = bim_idxed.merge(wl_22[["chrom_norm","pos"]], on=["chrom_norm","pos"], how="inner")
    keep_indices = allowed["sidx"].to_numpy(dtype=int)

    keep_mask = np.zeros(n_snps, dtype=bool)
    keep_mask[keep_indices] = True

    gram = np.zeros((n_samples, n_samples), dtype=np.float64)
    kept_total = 0

    for start in tqdm(range(0, n_snps, CHUNK_SNPS), desc="Streaming variants"):
        end = min(start + CHUNK_SNPS, n_snps)
        submask = keep_mask[start:end]
        if submask.any():
            X = bed.read(index=np.s_[:, start:end], dtype="float32", order="C")  # (n_samples, width)
            X = X[:, submask]                         # keep only whitelisted SNPs
            means = np.nanmean(X, axis=0)
            X = X - means                             # mean-center per SNP
            np.nan_to_num(X, copy=False)              # NaNs→0 after centering
            gram += X @ X.T
            kept_total += X.shape[1]

    gram /= kept_total
    return gram, kept_total

# ------------------ PCA + UMAP ------------------
def pca_from_grm(gram: np.ndarray, k: int):
    evals_all, evecs_all = eigh(gram)    # ascending eigenvalues
    k_eff = min(k, gram.shape[0] - 1)
    evals = evals_all[-k_eff:][::-1]
    evecs = evecs_all[:, -k_eff:][:, ::-1]
    pcs = evecs * np.sqrt(evals)         # scores = V * sqrt(Λ)
    return pcs, evals

# ------------------ Plotting ------------------
SUPERPOP_COLORS = {
    "AFR": "#e74c3c", "EUR": "#1f77b4", "EAS": "#9467bd",
    "SAS": "#2ca02c", "AMR": "#ff7f0e", "OTH": "#7f7f7f", "": "#7f7f7f"
}

def pop_color_map(series: pd.Series):
    uniq = sorted(series.fillna("").unique().tolist())
    cmap = plt.get_cmap("tab20")
    return {p: cmap(i % 20) for i, p in enumerate(uniq)}

def make_plots(pcs_df: pd.DataFrame, umap_df: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Build a population colormap across both datasets, and use fixed superpop colors
    all_pops = pd.concat([pcs_df["Population"], umap_df["Population"]], ignore_index=True)
    pop_colors = _pop_cmap(all_pops)  # facecolor by Population
    sp_colors = SUPERPOP_COLORS       # edgecolor by Superpopulation

    def scatter(ax, x, y, pop_series, sp_series, title, xlabel, ylabel):
        fc = pop_series.map(pop_colors).tolist()
        ec = sp_series.map(lambda s: sp_colors.get(s, "#7f7f7f")).tolist()
        ax.scatter(x, y, s=18, alpha=0.85, c=fc, edgecolors=ec, linewidths=0.6, rasterized=True)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

        # Legend for Superpopulations (edgecolor)
        sp_handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor='white', markeredgecolor=color,
                   markeredgewidth=1.2, label=sp)
            for sp, color in sp_colors.items()
        ]
        leg_sp = ax.legend(handles=sp_handles, title="Superpopulation (edgecolor)",
                           frameon=False, fontsize=8, loc="best")
        ax.add_artist(leg_sp)

        # Legend for Populations (facecolor) — cap to avoid clutter
        unique_pops = list(pop_colors.keys())
        max_lab = 20
        pop_handles = [
            Line2D([0], [0], marker='o', linestyle='',
                   markerfacecolor=pop_colors[p], markeredgecolor='k',
                   label=p)
            for p in unique_pops[:max_lab]
        ]
        ax.legend(handles=pop_handles,
                  title=("Population (facecolor)" if len(unique_pops) <= max_lab else
                         f"Population (facecolor; first {max_lab})"),
                  frameon=False, fontsize=7, loc="lower left")

    # Two panels: PCA (PC1 vs PC2) and UMAP
    fig, (ax_pca, ax_umap) = plt.subplots(1, 2, figsize=(14, 6))

    scatter(ax_pca,
            pcs_df["PC1"], pcs_df["PC2"],
            pcs_df["Population"], pcs_df["Superpopulation"],
            "PC1 vs PC2 (facecolor=Population, edgecolor=Superpopulation)",
            "PC1", "PC2")

    scatter(ax_umap,
            umap_df["UMAP1"], umap_df["UMAP2"],
            umap_df["Population"], umap_df["Superpopulation"],
            "UMAP (PC1..PC15) (facecolor=Population, edgecolor=Superpopulation)",
            "UMAP1", "UMAP2")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"[plot] saved {out_png}")

# ------------------ Main ------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    bed_zip = DATA_DIR / "chr22_subset50.bed.zip"
    bim_zip = DATA_DIR / "chr22_subset50.bim.zip"
    fam_zip = DATA_DIR / "chr22_subset50.fam.zip"
    igsr_tsv = DATA_DIR / "igsr_samples.tsv"
    white_tsv = DATA_DIR / "GSAv2_hg38.tsv"

    download(URLS["bed_zip"], bed_zip)
    download(URLS["bim_zip"], bim_zip)
    download(URLS["fam_zip"], fam_zip)
    download(URLS["igsr_tsv"], igsr_tsv)
    download(URLS["whitelist_tsv"], white_tsv)

    unzip(bed_zip, DATA_DIR)
    unzip(bim_zip, DATA_DIR)
    unzip(fam_zip, DATA_DIR)

    prefix = DATA_DIR / "chr22_subset50"

    fam = read_fam(prefix)
    bim = read_bim(prefix)
    igsr = read_igsr(igsr_tsv)
    wl  = read_whitelist(white_tsv)

    gram, kept_total = build_grm_whitelist(prefix, bim, wl)
    pcs, evals = pca_from_grm(gram, K_PCS)

    pc_cols = [f"PC{i+1}" for i in range(pcs.shape[1])]
    pcs_df = pd.DataFrame(pcs, columns=pc_cols)
    pcs_df.insert(0, "SampleID", fam["IID"].values)

    ann = pcs_df.merge(igsr, on="SampleID", how="left")
    ann["Population"] = ann["Population"].fillna("UNK")
    ann["Superpopulation"] = ann["Superpopulation"].fillna("OTH")

    # write outputs
    (OUT_DIR / "pca.tsv").write_text(
        ann[["SampleID"] + pc_cols].to_csv(sep="\t", index=False, float_format="%.6g")
    )
    pd.DataFrame({"PC": pc_cols[:len(evals)], "Eigenvalue": evals}).to_csv(
        OUT_DIR / "eigenvalues.tsv", sep="\t", index=False, float_format="%.6g"
    )

    # UMAP on first 15 PCs
    X = ann[pc_cols[:K_PCS]].to_numpy()
    reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42, low_memory=True)
    emb = reducer.fit_transform(X)
    umap_df = ann[["SampleID","Population","Superpopulation"]].copy()
    umap_df["UMAP1"] = emb[:, 0]
    umap_df["UMAP2"] = emb[:, 1]

    make_plots(
        pcs_df=ann[["SampleID","Population","Superpopulation","PC1","PC2"]],
        umap_df=umap_df,
        out_png=OUT_DIR / "plots_pca_umap.png"
    )

    total_var = float(np.sum(evals))
    pct1 = 100.0 * evals[0] / total_var
    pct2 = 100.0 * evals[1] / total_var
    print("===== SUMMARY =====")
    print(f"Samples: {len(ann)}")
    print(f"SNPs used (whitelist ∩ BIM@chr22): {kept_total}")
    print(f"PC1: {pct1:.2f}%   PC2: {pct2:.2f}% of GRM variance")
    print(f"Outputs: {OUT_DIR/'pca.tsv'}, {OUT_DIR/'eigenvalues.tsv'}, {OUT_DIR/'plots_pca_umap.png'}")

if __name__ == "__main__":
    main()
