"""
Stream-PCA for PLINK BED

QC order:
  1. call-rate   ≥ --min-call-rate
  2. MAF         ≥ --min-maf
  3. HWE p-value > --max-hwe-p      (skip if 1)
  4. variance    > --min-variance-epsilon
"""

from pathlib import Path
import argparse, logging, sys, time

import numpy as np
import pandas as pd
from bed_reader import open_bed
from numpy.linalg import eigh                # full, exact eigen-solver
from scipy.stats import chi2 as chi2_dist     # HWE test

# ────────────────────────── CLI ──────────────────────────
def cli():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--bed-prefix',  required=True,
                   help='Path prefix of PLINK files (.bed/.bim/.fam)')
    p.add_argument('--out-dir',     default='pca_results',
                   help='Folder to write pca.tsv & py.eigenvalues.tsv')
    p.add_argument('--k-components', type=int, default=10)

    # QC thresholds (same names & meanings as Rust code)
    p.add_argument('--min-call-rate',        type=float, default=0.98)
    p.add_argument('--min-maf',              type=float, default=0.01)
    p.add_argument('--max-hwe-p',            type=float, default=1e-6)
    p.add_argument('--min-variance-epsilon', type=float, default=1e-9)

    p.add_argument('--variant-chunk', type=int, default=2000,
                   help='Variants per streamed batch')

    return p.parse_args()


# ──────────────────────── logging ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("stream_pca")


# ────────────── Hardy-Weinberg χ² p-value ───────────────
def hwe_pval(a_aa, a_ab, a_bb) -> float:
    """Return HWE χ² p (1 d.o.f.)."""
    n = a_aa + a_ab + a_bb
    if n == 0:
        return 1.0
    p = (2 * a_aa + a_ab) / (2 * n)
    q = 1.0 - p
    exp = np.array([n * p * p, 2 * n * p * q, n * q * q])
    obs = np.array([a_aa, a_ab, a_bb])
    if (exp == 0).any():
        return 0.0
    chi2 = ((obs - exp) ** 2 / exp).sum()
    return 1.0 - chi2_dist.cdf(chi2, 1)


# ────────────────────────── main ─────────────────────────
def main() -> None:
    args = cli()
    t0 = time.time()

    bed = open_bed(f"{args.bed_prefix}.bed", count_A1=False)
    n_samples, n_variants = bed.iid_count, bed.sid_count
    log.info(f"Samples = {n_samples:,}    Variants = {n_variants:,}")

    fam = pd.read_csv(f"{args.bed_prefix}.fam", sep=r"\s+", header=None,
                      names=["fid", "iid", "pat", "mat", "sex", "pheno"])

    gram   = np.zeros((n_samples, n_samples), dtype=np.float64)
    kept   = 0
    chunk  = args.variant_chunk

    for start in range(0, n_variants, chunk):
        end = min(start + chunk, n_variants)
        X   = bed.read(index=np.s_[:, start:end], dtype='float32', order='C')

        # QC metrics
        call_rate = np.nanmean(~np.isnan(X), axis=0)
        maf       = np.nanmean(X, axis=0) / 2.0
        maf       = np.where(maf > 0.5, 1 - maf, maf)

        h0 = np.nansum(X == 0, axis=0)
        h1 = np.nansum(X == 1, axis=0)
        h2 = np.nansum(X == 2, axis=0)
        hwe = np.fromiter((hwe_pval(a, b, c) for a, b, c in zip(h0, h1, h2)),
                          dtype=np.float32, count=X.shape[1])

        var = np.nanvar(X, axis=0, ddof=1)

        ok  = (call_rate >= args.min_call_rate) & \
              (maf       >= args.min_maf)       & (maf > 1e-7) & \
              (hwe       >  args.max_hwe_p)     & \
              (var       >  args.min_variance_epsilon)

        if ok.any():
            X_good  = X[:, ok]
            X_good -= np.nanmean(X_good, axis=0)
            X_good  = np.nan_to_num(X_good, copy=False)
            gram   += X_good @ X_good.T
            kept   += X_good.shape[1]

        pct = end / n_variants * 100
        log.info(f"{pct:6.2f}% | Variants {start:,}–{end-1:,} processed "
                 f"(kept {kept:,})")

    if kept == 0:
        log.error("No variants passed QC.")
        sys.exit(1)

    gram /= kept
    data_loading_time = time.time() - t0
    log.info(f"GRM built from {kept:,} variants in {data_loading_time/60:.1f} min")

    # ═══════════════════════════════════════════════════════════════
    # TIME THE PCA STEP ITSELF (eigendecomposition)
    # ═══════════════════════════════════════════════════════════════
    log.info("Starting PCA eigendecomposition...")
    pca_start = time.time()
    
    # Exact eigen-decomposition → top-k PCs
    evals_all, evecs_all = eigh(gram)   # ascending
    
    pca_time = time.time() - pca_start
    log.info(f"PCA eigendecomposition completed in {pca_time:.3f} seconds ({pca_time/60:.2f} minutes)")
    
    k      = min(args.k_components, n_samples - 1)
    evals  = evals_all[-k:][::-1]
    evecs  = evecs_all[:, -k:][:, ::-1]
    pcs    = evecs * np.sqrt(evals)

    # ─── write outputs ───
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pc_cols = [f"PC{i+1}" for i in range(k)]
    pd.DataFrame(pcs, columns=pc_cols) \
        .assign(SampleID=fam.iid.values) \
        .to_csv(out_dir / "pca.tsv", sep='\t', index=False, float_format="%.6g")

    pd.DataFrame({"PC": pc_cols, "Eigenvalue": evals}) \
        .to_csv(out_dir / "py.eigenvalues.tsv", sep='\t',
                index=False, float_format="%.6g")

    # ─── summary ───
    total_time = time.time() - t0
    print("\n===== PCA COMPLETE =====")
    print(f"  QC-passed variants : {kept:,} / {n_variants:,}")
    for i, ev in enumerate(evals, 1):
        print(f"  PC{i:<2} eigenvalue : {ev:.6g}")
    print(f"\nTiming breakdown:")
    print(f"  Data loading + GRM : {data_loading_time:.1f} seconds ({data_loading_time/60:.2f} minutes)")
    print(f"  PCA eigendecomp    : {pca_time:.1f} seconds ({pca_time/60:.2f} minutes)")
    print(f"  Total wall-time    : {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nResults written to directory: {out_dir}\n"
          f"  • pca.tsv          (scores × {k})\n"
          f"  • py.eigenvalues.tsv  (λ₁…λ{k})")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log.warning("Interrupted by user.")
