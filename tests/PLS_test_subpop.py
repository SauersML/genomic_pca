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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix
)
from sklearn.cross_decomposition import PLSRegression

# ============================================================
#                       CONFIG / CONSTANTS
# ============================================================
OUT_DIR  = Path("fast_pca_ood_subpops").resolve()
DATA_DIR = OUT_DIR / "data"

CHUNK_SNPS = 5000
K_PCS      = 15
K_LVS      = 15
PLS_TARGET = "Population"   # LVs supervised on Population (subpop)
EXCLUDE_IN_METRICS = {"OTH", "UNK", "", "nan"}  # omit these from metrics only
HOLDOUT_FRAC = 0.20
RNG_SEED = 42

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
    import urllib.request
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
#                        KEEP MASK (chr22 ∩ whitelist)
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
#                        PGEN ACCESS / STREAMING
# ============================================================
def _open_pgen(prefix: Path):
    pgen_path_bytes = os.fsencode(str(prefix.with_suffix(".pgen")))
    return pgenlib.PgenReader(pgen_path_bytes)

def _counts(prefix: Path):
    p = _open_pgen(prefix)
    ns = p.get_raw_sample_ct()
    nv = p.get_variant_ct()
    p.close()
    return ns, nv

def compute_train_means(prefix: Path, keep_mask: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """
    Per-SNP means over TRAIN samples only (kept SNPs), streamed.
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    assert train_mask.shape[0] == n_samples

    m_kept = int(keep_mask.sum())
    sum_vec = np.zeros(m_kept, dtype=np.float64)
    cnt_vec = np.zeros(m_kept, dtype=np.int64)

    scanned = tqdm(total=n_snps, desc="Means: scanning variants", unit="SNP", leave=False, dynamic_ncols=True)
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
        X = buf.astype(np.float32, copy=False)[train_mask, :][:, submask]
        X[X < 0] = np.nan
        valid = ~np.isnan(X)
        sum_vec[kp:kp+X.shape[1]] += np.nansum(X, axis=0)
        cnt_vec[kp:kp+X.shape[1]] += valid.sum(axis=0)
        kp += X.shape[1]

    scanned.close()
    pgen.close()

    cnt_safe = np.maximum(cnt_vec, 1)
    return (sum_vec / cnt_safe).astype(np.float32)

def build_grm_train(prefix: Path, keep_mask: np.ndarray, train_mask: np.ndarray, mu_kept: np.ndarray):
    """
    Build GRM over TRAIN samples using precomputed training means.
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
    return gram, kept_total

def project_pcs(prefix: Path, keep_mask: np.ndarray, mu_kept: np.ndarray,
                train_mask: np.ndarray, proj_mask: np.ndarray,
                V: np.ndarray, evals: np.ndarray, k: int):
    """
    Project PROJ samples onto TRAIN PCs via cross-similarity.
    """
    pgen = _open_pgen(prefix)
    n_samples = pgen.get_raw_sample_ct()
    n_snps = pgen.get_variant_ct()
    idx_train = np.where(train_mask)[0]
    idx_proj  = np.where(proj_mask)[0]
    n_train, n_proj = idx_train.size, idx_proj.size

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

    cross /= max(int(keep_mask.sum()), 1)
    k_eff = min(k, V.shape[1], np.count_nonzero(evals > 0))
    denom = np.sqrt(np.maximum(evals[:k_eff], 1e-12))
    return cross @ (V[:, :k_eff] / denom)

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
    return np.concatenate(cols, axis=1).astype("float32", copy=False)

def pca_from_grm(gram: np.ndarray, k: int):
    print("  • Eigen-decomposing GRM (eigh) → PCs …", flush=True)
    evals_all, evecs_all = eigh(gram)
    k_eff = min(k, gram.shape[0] - 1) if gram.shape[0] > 1 else 0
    if k_eff <= 0:
        return np.zeros((gram.shape[0], 0), dtype=np.float32), np.array([], dtype=np.float64), np.zeros((gram.shape[0], 0), dtype=np.float32)
    evals = evals_all[-k_eff:][::-1]
    evecs = evecs_all[:, -k_eff:][:, ::-1]
    pcs   = evecs * np.sqrt(evals)
    V = pcs / np.sqrt(np.maximum(evals, 1e-12))  # eigenvectors in sample space
    return pcs, evals, V

def fit_pls_da(X_train: np.ndarray, y_series: pd.Series, max_lv: int):
    Y = pd.get_dummies(y_series.fillna("UNK").astype(str), drop_first=False).to_numpy(dtype=np.float32)
    k_eff = max(1, min(max_lv, X_train.shape[0] - 1, Y.shape[1]))
    pls = PLSRegression(n_components=k_eff, scale=False)
    pls.fit(X_train, Y)
    LV_train = pls.x_scores_
    return pls, LV_train

# ============================================================
#                       EVALUATION HELPERS
# ============================================================
def _renorm_rows(A: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    s = np.maximum(s, eps)
    return A / s

def _align_proba_union(y_true: np.ndarray, proba: np.ndarray, model_classes: np.ndarray, exclude: set):
    """
    Build an evaluation class set as the UNION of (present y_true excluding EXCLUDE) and model classes.
    Return (y_eval, proba_eval, eval_classes). New columns for unseen true classes are zeros.
    """
    keep_mask = ~pd.Series(y_true).isin(exclude).to_numpy()
    y_eval = y_true[keep_mask]
    if y_eval.size == 0:
        return y_eval, np.zeros((0, 0)), []

    present = sorted(set(y_eval))
    eval_classes = sorted(set(c for c in present) | set(c for c in model_classes if c not in exclude))
    # Map model probabilities into the union matrix
    col_index = {c: i for i, c in enumerate(eval_classes)}
    proba_eval = np.zeros((keep_mask.sum(), len(eval_classes)), dtype=float)
    for j, c in enumerate(model_classes):
        if c in EXCLUDE_IN_METRICS:  # drop excluded model classes entirely
            continue
        if c in col_index:
            proba_eval[:, col_index[c]] = proba[keep_mask, j]
    return y_eval, proba_eval, eval_classes

def _align_proba_intersection(y_true: np.ndarray, proba: np.ndarray, model_classes: np.ndarray, exclude: set):
    """
    INTERSECTION of (present y_true excluding EXCLUDE) and model_classes (excluding EXCLUDE).
    Rows are renormalized to sum to 1.
    """
    keep_mask = ~pd.Series(y_true).isin(exclude).to_numpy()
    y_eval = y_true[keep_mask]
    if y_eval.size == 0:
        return y_eval, np.zeros((0, 0)), []

    present = sorted(set(y_eval))
    eval_classes = [c for c in present if (c in model_classes) and (c not in exclude)]
    if len(eval_classes) < 2:
        return y_eval, np.zeros((len(y_eval), 0)), eval_classes

    idx = [np.where(model_classes == c)[0][0] for c in eval_classes]
    proba_eval = proba[keep_mask][:, idx]
    proba_eval = _renorm_rows(proba_eval)
    return y_eval, proba_eval, eval_classes

def _brier(y_true, proba, classes):
    # Multiclass Brier across the provided class list
    Y = pd.get_dummies(pd.Categorical(y_true, categories=classes)).to_numpy(dtype=float)
    diff = proba - Y
    return float(np.mean(np.sum(diff * diff, axis=1)))

def _eval_and_print(y_true, proba, model_classes, title, mode="intersection"):
    """
    mode:
      - "intersection": used for SUPERPOP (closed-set). Renormalizes proba after class filtering.
      - "union"       : used for SUBPOP held-out (open-set). Adds zero-prob columns for unseen true classes.
    """
    if mode == "intersection":
        y_eval, proba_eval, eval_classes = _align_proba_intersection(y_true, proba, model_classes, EXCLUDE_IN_METRICS)
    else:
        y_eval, proba_eval, eval_classes = _align_proba_union(y_true, proba, model_classes, EXCLUDE_IN_METRICS)

    print(f"\n--- {title} ---")
    if y_eval.size == 0 or len(eval_classes) == 0 or proba_eval.shape[1] == 0:
        print("Not enough evaluable data after exclusions; metrics undefined.")
        return

    # Predictions from eval proba
    y_pred = np.array(eval_classes)[np.argmax(proba_eval, axis=1)]

    # Metrics
    acc = accuracy_score(y_eval, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_eval, y_pred, labels=eval_classes, average="macro", zero_division=0
    )
    # Clip probabilities for log-loss stability
    proba_clip = np.clip(proba_eval, 1e-15, 1 - 1e-15)
    try:
        ll_overall = log_loss(y_eval, proba_clip, labels=eval_classes)
    except ValueError:
        ll_overall = float("nan")
    try:
        auc_macro = roc_auc_score(y_eval, proba_eval, multi_class="ovr", average="macro", labels=eval_classes)
    except ValueError:
        auc_macro = float("nan")
    brier = _brier(y_eval, proba_eval, eval_classes)

    cm = confusion_matrix(y_eval, y_pred, labels=eval_classes, normalize="true")

    print(f"n_test (after exclusion): {len(y_eval)} | classes: {', '.join(eval_classes)}")
    print(f"Accuracy           : {acc:.6f}")
    print(f"LogLoss (overall)  : {ll_overall:.6f}")
    print(f"Precision (macro)  : {prec:.6f}")
    print(f"Recall (macro)     : {rec:.6f}")
    print(f"F1 (macro)         : {f1:.6f}")
    print(f"AUROC (macro, OvR) : {auc_macro:.6f}")
    print(f"Brier score        : {brier:.6f}")
    header = "           " + "  ".join([f"{c:>5}" for c in eval_classes])
    print("Confusion matrix (rows=true, cols=pred, normalized by true):")
    print(header)
    for i, c in enumerate(eval_classes):
        row = "  ".join([f"{v:5.2f}" for v in cm[i]])
        print(f"{c:>10}  {row}")

# ============================================================
#                    SPLIT: HOLD OUT 20% SUBPOPS
# ============================================================
def split_holdout_subpops(ann: pd.DataFrame, holdout_frac=0.20, seed=42):
    """
    Choose ~20% of distinct Population labels as held-out SUBPOPS.
    All samples of those subpops go to TEST; the rest to TRAIN.
    """
    rng = np.random.RandomState(seed)
    pops = sorted([p for p in ann["Population"].astype(str).unique() if p not in EXCLUDE_IN_METRICS])
    n_hold = max(1, int(round(holdout_frac * len(pops))))
    held_out = sorted(rng.choice(pops, size=n_hold, replace=False).tolist())

    is_holdout = ann["Population"].astype(str).isin(held_out).to_numpy()
    train_mask = ~is_holdout
    test_mask  =  is_holdout

    return train_mask, test_mask, held_out

# ============================================================
#                            MAIN
# ============================================================
def main():
    stages = [
        "Initialize output/data directories",
        "Fetch IGSR + whitelist (if needed)",
        "Read PSAM/PVAR/IGSR/whitelist; annotate samples",
        "Compute SNP keep mask (whitelist ∩ PVAR@chr22)",
        "Hold out 20% of subpopulations (Population labels)",
        "PCA: means→GRM→eigen (train) and projection (test)",
        "PLS-DA: fit on train (target=Population) and transform (test)",
        "Logistic (no reg) on PCs/LVs → evaluate on HELD-OUT SUBPOPS only (Superpop & Subpop)"
    ]
    ST = StageTracker(stages)

    # 1) init
    ST.start()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    prefix = Path("hg38_chr22").resolve()
    print(f"OUT_DIR: {OUT_DIR}")

    # 2) fetch aux data
    ST.start()
    igsr_tsv = DATA_DIR / "igsr_samples.tsv"
    white_tsv = DATA_DIR / "GSAv2_hg38.tsv"
    if not igsr_tsv.exists():
        ST.note(f"Fetching IGSR → {igsr_tsv.name}")
        download(URLS["igsr_tsv"], igsr_tsv)
    if not white_tsv.exists():
        ST.note(f"Fetching whitelist → {white_tsv.name}")
        download(URLS["whitelist_tsv"], white_tsv)

    # 3) read meta
    ST.start()
    ST.note("Reading PSAM …")
    psam = read_psam(prefix)
    ST.note("Reading PVAR …")
    pvar = read_pvar(prefix)
    ST.note("Reading IGSR …")
    igsr = read_igsr(igsr_tsv)
    ST.note("Reading whitelist …")
    wl = read_whitelist(white_tsv)

    ann = psam.rename(columns={"IID":"SampleID"}).merge(igsr, on="SampleID", how="left")
    ann["Population"] = ann["Population"].fillna("UNK")
    ann["Superpopulation"] = ann["Superpopulation"].fillna("UNK")

    # 4) keep mask
    ST.start()
    keep_mask = compute_keep_mask(pvar, wl)
    n_samples, n_snps = _counts(prefix)
    print(f"Samples: {n_samples} | SNPs kept: {int(keep_mask.sum())}")

    # 5) hold out 20% subpops
    ST.start()
    train_mask, test_mask, held_out_pops = split_holdout_subpops(ann, HOLDOUT_FRAC, RNG_SEED)
    print(f"Held-out subpopulations ({len(held_out_pops)}): {', '.join(held_out_pops)}")
    print(f"Train n={train_mask.sum()} | Test n={test_mask.sum()}")

    # 6) PCA pipeline
    ST.start()
    ST.note("Computing training means …")
    mu = compute_train_means(prefix, keep_mask, train_mask)
    ST.note("Building GRM (train) …")
    gram, kept_total = build_grm_train(prefix, keep_mask, train_mask, mu)
    ST.note("Eigendecomposition …")
    pcs_train, evals, V = pca_from_grm(gram, K_PCS)
    ST.note("Projecting held-out test → PCs …")
    pcs_test = project_pcs(prefix, keep_mask, mu, train_mask, test_mask, V, evals, K_PCS)

    # 7) PLS-DA pipeline (LVs based on Population)
    ST.start()
    ST.note("Building X (train) …")
    X_train = build_X_for(prefix, keep_mask, mu, train_mask)
    y_pop_train = ann.loc[train_mask, PLS_TARGET].reset_index(drop=True)
    ST.note(f"Fitting PLS-DA (target={PLS_TARGET}) …")
    pls, LV_train = fit_pls_da(X_train, y_pop_train, K_LVS)
    ST.note("Building X (held-out test) …")
    X_test = build_X_for(prefix, keep_mask, mu, test_mask)
    ST.note("Transforming held-out test → LVs …")
    LV_test = pls.transform(X_test)

    # 8) Logistic (no regularization) and evaluation on HELD-OUT subpops only
    ST.start()
    y_super_train = ann.loc[train_mask, "Superpopulation"].astype(str).values
    y_super_test  = ann.loc[test_mask,  "Superpopulation"].astype(str).values
    y_pop_train   = ann.loc[train_mask, "Population"].astype(str).values
    y_pop_test    = ann.loc[test_mask,  "Population"].astype(str).values

    def fit_model(Xtr, ytr):
        k_use = min(15, Xtr.shape[1])
        clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
        clf.fit(Xtr[:, :k_use], ytr)
        return clf, k_use

    def predict_proba(clf, Xte, k_use):
        return clf.predict_proba(Xte[:, :k_use]), clf.classes_

    # SUPERPOP (closed-set; intersection eval on held-out subpop samples)
    print("\n=== EVALUATION ON HELD-OUT SUBPOPS: SUPERPOPULATION ===")
    clf_sp_pcs, k_sp_pcs = fit_model(pcs_train, y_super_train)
    proba_sp_pcs, classes_sp_pcs = predict_proba(clf_sp_pcs, pcs_test, k_sp_pcs)
    _eval_and_print(y_super_test, proba_sp_pcs, classes_sp_pcs, f"Superpopulation | PCs (k={k_sp_pcs}) | test n={len(y_super_test)}", mode="intersection")

    clf_sp_lvs, k_sp_lvs = fit_model(LV_train, y_super_train)
    proba_sp_lvs, classes_sp_lvs = predict_proba(clf_sp_lvs, LV_test, k_sp_lvs)
    _eval_and_print(y_super_test, proba_sp_lvs, classes_sp_lvs, f"Superpopulation | PLS LVs (k={k_sp_lvs}) | test n={len(y_super_test)}", mode="intersection")

    # SUBPOP (open-set; union eval because test labels were never seen during training)
    print("\n=== EVALUATION ON HELD-OUT SUBPOPS: POPULATION (SUBPOP) ===")
    clf_pp_pcs, k_pp_pcs = fit_model(pcs_train, y_pop_train)
    proba_pp_pcs, classes_pp_pcs = predict_proba(clf_pp_pcs, pcs_test, k_pp_pcs)
    _eval_and_print(y_pop_test, proba_pp_pcs, classes_pp_pcs, f"Population | PCs (k={k_pp_pcs}) | test n={len(y_pop_test)}", mode="union")

    clf_pp_lvs, k_pp_lvs = fit_model(LV_train, y_pop_train)
    proba_pp_lvs, classes_pp_lvs = predict_proba(clf_pp_lvs, LV_test, k_pp_lvs)
    _eval_and_print(y_pop_test, proba_pp_lvs, classes_pp_lvs, f"Population | PLS LVs (k={k_pp_lvs}) | test n={len(y_pop_test)}", mode="union")

    print("\nDone. All metrics were computed **on held-out subpopulations only**; superpop uses closed-set (intersection) evaluation, subpop uses open-set (union) evaluation with zero-prob columns for unseen classes.")
    
if __name__ == "__main__":
    main()
