import argparse
import logging
import time
import sys

import numpy as np
import pandas as pd
import sgkit as sg
import xarray as xr
from dask.distributed import Client, LocalCluster

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sgkit_standard_pca")

# sgkit.io.plink is directly available if sgkit[plink] is installed
from sgkit.io import plink as sgkit_plink_io


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform standard PCA on PLINK BED files using sgkit, "
        "applying QC filters. Defaults for QC are set to match EigenSNP-Rust's --eigensnp mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--bed-prefix", required=True, help="Prefix for PLINK .bed, .bim, .fam files."
    )
    parser.add_argument(
        "--sample-keep-file", type=str, default=None, help="Optional: File listing sample IIDs to keep."
    )
    parser.add_argument(
        "--min-call-rate", type=float, default=0.98, help="Minimum SNP call rate."
    )
    parser.add_argument(
        "--min-maf", type=float, default=0.01, help="Minimum MAF (based on A1)."
    )
    parser.add_argument(
        "--max-hwe-p", type=float, default=1e-6, help="Max HWE p-value (filter if p <= threshold)."
    )
    parser.add_argument(
        "--min-variance-epsilon", type=float, default=1e-9, help="Minimum genotype dosage variance."
    )
    parser.add_argument(
        "--k-components", type=int, default=10, help="Number of PCs."
    )
    parser.add_argument(
        "--out-prefix", required=True, help="Prefix for output files."
    )
    parser.add_argument(
        "--threads", type=int, default=4, help="Number of Dask workers."
    )
    parser.add_argument(
        "--memory-limit-gb", type=float, default=8.0, help="Memory limit per Dask worker (GB)."
    )
    parser.add_argument(
        "--variant-chunk-size", type=int, default=5000, # Reduced for potentially better stability
        help="Chunk size for variants."
    )
    parser.add_argument(
        "--sample-chunk-size", type=int, default=500,   # Reduced for potentially better stability
        help="Chunk size for samples."
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    total_time_start = time.time()
    logger.info(f"Starting sgkit standard PCA: {args}")

    logger.info(f"Dask: {args.threads} workers, {args.memory_limit_gb}GB/worker memory limit.")
    with LocalCluster(
        n_workers=args.threads,
        threads_per_worker=1,
        memory_limit=f"{args.memory_limit_gb}GB",
    ) as cluster, Client(cluster) as client:
        logger.info(f"Dask client: {client}, dashboard: {client.dashboard_link}")

        logger.info(f"Loading PLINK: {args.bed_prefix}")
        ds = sgkit_plink_io.read_plink(
            bed_path=f"{args.bed_prefix}.bed",
            bim_path=f"{args.bed_prefix}.bim",
            fam_path=f"{args.bed_prefix}.fam",
            # count_a1=True, # Removed due to error in sgkit 0.10.0; default is True
            chunks={"variants": args.variant_chunk_size, "samples": args.sample_chunk_size},
        )
        logger.info(f"Initial dataset: {ds.dims}")

        ds_filtered_samples = ds
        if args.sample_keep_file:
            logger.info(f"Filtering samples from: {args.sample_keep_file}")
            with open(args.sample_keep_file, "r") as f:
                kept_iids = {line.strip() for line in f if line.strip()}
            logger.info(f"Keeping {len(kept_iids)} samples.")
            sample_ids_arr = ds.sample_id.astype(str).compute()
            keep_mask = np.array([sid in kept_iids for sid in sample_ids_arr])
            ds_filtered_samples = ds.sel(samples=keep_mask)
            logger.info(f"Samples after filter: {ds_filtered_samples.dims['samples']}")
            if ds_filtered_samples.dims['samples'] == 0:
                logger.error("No samples left. Exiting."); sys.exit(1)
        else:
            logger.info("No sample keep file. Using all samples.")

        logger.info("Starting Variant QC...")

        # Step 1: Pre-calculate variant_allele_count as sg.variant_stats might use it
        logger.info("Calculating variant_allele_count using sg.count_variant_alleles...")
        # sg.count_variant_alleles uses sg.count_call_alleles internally
        ds_with_var_counts = sg.count_variant_alleles(ds_filtered_samples, merge=True)
        # ds_with_var_counts now has 'variant_allele_count' and implicitly 'call_allele_count'

        # Step 2: Calculate other variant statistics (including call rate)
        logger.info("Calculating main variant statistics (call_rate etc.) using sg.variant_stats...")
        # Now call sg.variant_stats; it should find and use the pre-computed 'variant_allele_count'
        ds_stats = sg.variant_stats(ds_with_var_counts, merge=True)
        
        # 2a. Call Rate filter
        logger.info(f"Applying call rate filter (>= {args.min_call_rate})...")
        call_rate_mask = ds_stats.variant_call_rate >= args.min_call_rate
        logger.info(f"Variants passing call rate: {call_rate_mask.sum().compute().item()}")

        # 2b. MAF filter (using variant_allele_count from ds_stats)
        logger.info(f"Applying MAF filter (>= {args.min_maf})...")
        count_A1 = ds_stats.variant_allele_count[:, 0]
        count_A2 = ds_stats.variant_allele_count[:, 1]
        total_alleles = count_A1 + count_A2
        p_A1 = xr.where(total_alleles > 0, count_A1 / total_alleles, 0.0)
        maf_A1 = xr.where(p_A1 > 0.5, 1.0 - p_A1, p_A1)
        maf_mask = maf_A1 >= args.min_maf
        poly_mask = maf_A1 > 1e-7 # Ensure not strictly monomorphic
        logger.info(f"Variants passing MAF & Polymorphic: {(maf_mask & poly_mask).sum().compute().item()}")

        combined_mask1 = call_rate_mask & maf_mask & poly_mask
        ds_qc1 = ds_stats.sel(variants=combined_mask1)
        logger.info(f"Variants after Call Rate & MAF QC: {ds_qc1.dims['variants']}")
        if ds_qc1.dims['variants'] == 0: logger.error("No variants after CR/MAF. Exiting."); sys.exit(1)

        # Step 3: HWE Test
        ds_for_hwe = ds_qc1 # This dataset contains variant_allele_count needed by HWE test
        hwe_pass_mask = xr.DataArray(np.ones(ds_for_hwe.dims['variants'], dtype=bool), dims="variants", coords={'variants': ds_for_hwe.variants})
        if args.max_hwe_p < 1.0:
            logger.info(f"Calculating HWE (filter if p <= {args.max_hwe_p})...")
            ds_hwe = sg.hardy_weinberg_test(ds_for_hwe, merge=True)
            hwe_pass_mask = ds_hwe.variant_hwe_p_value > args.max_hwe_p
            logger.info(f"Variants passing HWE: {hwe_pass_mask.sum().compute().item()}")
            ds_after_hwe = ds_hwe.sel(variants=hwe_pass_mask)
        else:
            logger.info("HWE filtering disabled.")
            ds_after_hwe = ds_for_hwe
        
        logger.info(f"Variants after HWE filter: {ds_after_hwe.dims['variants']}")
        if ds_after_hwe.dims['variants'] == 0: logger.error("No variants after HWE. Exiting."); sys.exit(1)

        # Step 4: Variance Filter (manual)
        logger.info(f"Calculating genotype variance (> {args.min_variance_epsilon})...")
        # Convert call_genotype (-1 for missing) to float with NaNs
        gt_float_nan = ds_after_hwe.call_genotype.astype(float).where(ds_after_hwe.call_genotype != -1, np.nan)
        snp_std_dev = gt_float_nan.std(dim="samples", skipna=True, ddof=1) # ddof=1 for sample variance
        variance_final_mask = (snp_std_dev**2) > args.min_variance_epsilon
        
        ds_pca_input_prep = ds_after_hwe.sel(variants=variance_final_mask)
        logger.info(f"Variants for PCA (after variance): {ds_pca_input_prep.dims['variants']}")
        if ds_pca_input_prep.dims['variants'] == 0: logger.error("No variants after var filter. Exiting."); sys.exit(1)

        # Prepare the specific variable for PCA: 'call_genotype' converted to float with NaNs
        # sg.stats.pca.pca with PattersonScaler expects allele counts (0,1,2).
        # Missing values must be NaN for PattersonScaler to impute them (typically to 2p).
        ds_pca_input_prep["pca_genotype_input"] = ds_pca_input_prep.call_genotype.astype(float).where(
            ds_pca_input_prep.call_genotype != -1, np.nan
        )
        
        # Rechunk for 'tsqr' algorithm (default for sg.stats.pca.pca)
        # Needs to be chunked only along variants, not samples.
        ds_pca_input_prep["pca_genotype_input"] = ds_pca_input_prep.pca_genotype_input.chunk(
            {"variants": args.variant_chunk_size, "samples": -1} # -1 means single chunk for samples
        )
        logger.info(f"Rechunked pca_genotype_input for PCA: {ds_pca_input_prep.pca_genotype_input.chunks}")

        ds_pca_ready = ds_pca_input_prep
        N_final_samples = ds_pca_ready.dims["samples"]
        D_final_snps = ds_pca_ready.dims["variants"]

        if N_final_samples < 2: logger.error(f"Need >=2 samples, got {N_final_samples}. Exiting."); sys.exit(1)
        if args.k_components <= 0: logger.error("k_components <= 0. Exiting."); sys.exit(1)
        effective_max_k = min(N_final_samples - 1, D_final_snps)
        if effective_max_k <= 0: logger.error(f"Eff. max k = {effective_max_k}. Exiting."); sys.exit(1)
        if args.k_components > effective_max_k:
            logger.warning(f"k_components ({args.k_components}) > max ({effective_max_k}). Using k={effective_max_k}.")
            args.k_components = effective_max_k
        
        logger.info(f"PCA: {N_final_samples} samples, {D_final_snps} SNPs, {args.k_components} components...")
        # Using 'pca_genotype_input' which is A1 dosage (0,1,2) with NaNs for missing
        sample_pca_scores_da, pca_model_ds = sg.stats.pca.pca(
            ds_pca_ready,
            n_components=args.k_components,
            variable="pca_genotype_input", # Use our prepared variable
            scaler="standard", # PattersonScaler is applied to 'pca_genotype_input'
        )
        logger.info("PCA complete.")

        logger.info("Extracting PCA results...")
        sample_pcs_np = sample_pca_scores_da.compute().to_numpy()
        snp_loadings_np = pca_model_ds.components.compute().to_numpy() 
        eigenvalues_np = pca_model_ds.explained_variance.compute().to_numpy()

        logger.info(f"Writing results to prefix: {args.out_prefix}")
        pcs_output_file = f"{args.out_prefix}.pca.tsv"
        sample_ids_np = ds_pca_ready.sample_id.astype(str).compute().to_numpy()
        pc_cols = [f"PC{i+1}" for i in range(args.k_components)]
        pcs_df = pd.DataFrame(sample_pcs_np, columns=pc_cols)
        pcs_df.insert(0, "SampleID", sample_ids_np)
        pcs_df.to_csv(pcs_output_file, sep="\t", index=False, float_format="%.6g")
        logger.info(f"PCs written to: {pcs_output_file}")

        eigenvalues_output_file = f"{args.out_prefix}.eigenvalues.tsv"
        eigenvalues_df = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(args.k_components)], "Eigenvalue": eigenvalues_np
        })
        eigenvalues_df.to_csv(eigenvalues_output_file, sep="\t", index=False, float_format="%.6g")
        logger.info(f"Eigenvalues written to: {eigenvalues_output_file}")

        loadings_output_file = f"{args.out_prefix}.loadings.tsv"
        variant_ids_np = ds_pca_ready.variant_id.astype(str).compute().to_numpy()
        variant_chrom_np = ds_pca_ready.variant_chromosome.astype(str).compute().to_numpy()
        variant_pos_np = ds_pca_ready.variant_position.compute().to_numpy()
        loading_cols = [f"PC{i+1}_loading" for i in range(args.k_components)]
        loadings_df = pd.DataFrame(snp_loadings_np.T, columns=loading_cols) 
        loadings_df.insert(0, "VariantID", variant_ids_np)
        loadings_df.insert(1, "Chrom", variant_chrom_np)
        loadings_df.insert(2, "Pos", variant_pos_np)
        loadings_df.to_csv(loadings_output_file, sep="\t", index=False, float_format="%.6g")
        logger.info(f"Loadings written to: {loadings_output_file}")

    total_time_end = time.time()
    logger.info(f"Script finished in {total_time_end - total_time_start:.2f}s.")

if __name__ == "__main__":
    main()
