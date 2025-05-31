// main.rs

mod prepare;
mod vcf;

// --- External Crate Imports ---
use anyhow::{anyhow, Error, Result};
use clap::Parser;
use efficient_pca::PCA as EfficientPcaModel;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use ndarray::{Array2};
use noodles_vcf::{
    Header as VcfHeader,
};
use num_cpus;
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

// --- Main Function ---
fn main() -> Result<(), Error> {
    let total_time_start = Instant::now();
    let cli_args = cli::CliArgs::parse();

    // Initialize logger
    let log_level = cli_args
        .log_level
        .parse::<log::LevelFilter>()
        .unwrap_or_else(|_| {
            eprintln!(
                "Warning: Invalid log level '{}' provided. Defaulting to Info.",
                cli_args.log_level
            );
            log::LevelFilter::Info
        });
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_micros()
        .init();

    info!("Starting genomic_pca with args: {:?}", cli_args);

    // Configure Rayon thread pool
    let num_threads = cli_args.threads.unwrap_or_else(num_cpus::get);
    info!("Using {} threads for parallel operations.", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    // --- 1. Discover VCF files and Read Header from First VCF & Prepare Shared Info ---
    info!(
        "Discovering VCF files in directory: {}",
        cli_args.vcf_dir.display()
    );
    let mut vcf_files: Vec<PathBuf> = fs::read_dir(&cli_args.vcf_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && (path.extension().map_or(false, |ext| {
                    ext == "vcf" || ext == "gz"
                }))
                && (path
                    .file_name()
                    .map_or(false, |name| name.to_string_lossy().contains(".vcf")))
        })
        .collect();

    if vcf_files.is_empty() {
        return Err(anyhow!(
            "No VCF files (ending in .vcf or .vcf.gz) found in directory: {}",
            cli_args.vcf_dir.display()
        ));
    }
    vcf_files.sort();
    info!(
        "Found {} VCF file(s). Processing order (first 5): {:?}",
        vcf_files.len(),
        vcf_files.iter().take(5).collect::<Vec<_>>()
    );

    let first_vcf_path = &vcf_files[0];
    info!("Reading header from first VCF: {}", first_vcf_path.display());
    let mut first_reader = noodles_vcf::reader::Builder::default().build_from_path(first_vcf_path)?;
    let header_template = Arc::new(first_reader.read_header()?); // Used for sample name consistency
    let samples_info = Arc::new(vcf_processing::SamplesHeaderInfo::from_header(
        &header_template,
        first_vcf_path,
    )?);
    info!(
        "Established sample set from {}: {} samples. All other VCFs must match this set and order.",
        first_vcf_path.display(),
        samples_info.sample_count
    );
    debug!(
        "Sample names (first 5): {:?}",
        samples_info.sample_names.iter().take(5).collect::<Vec<_>>()
    );

    // --- 2. Parallel VCF Processing ---
    info!(
        "Processing {} VCF file(s) in parallel...",
        vcf_files.len()
    );
    let pb_vcf_style = ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} VCFs ({percent}%) ETA: {eta}",
        )
        .map_err(|e| anyhow!("Failed to create progress bar style: {}", e))?
        .progress_chars("=> ");
    let pb_vcf = ProgressBar::new(vcf_files.len() as u64).with_style(pb_vcf_style);

    let per_chromosome_data_results: Vec<Result<Option<Vec<vcf_processing::VariantGenotypeData>>>> =
        vcf_files
            .par_iter()
            .map(|vcf_path| {
                let result = vcf_processing::process_single_vcf(
                    vcf_path,
                    samples_info.clone(), // Arc clone
                    &cli_args,
                    first_vcf_path,
                );
                pb_vcf.inc(1);
                result
            })
            .collect();
    pb_vcf.finish_with_message("VCF processing complete.");

    let mut all_good_chromosome_data: Vec<Vec<vcf_processing::VariantGenotypeData>> = Vec::new();
    let mut processing_errors: Vec<Error> = Vec::new();

    for (i, result_chunk) in per_chromosome_data_results.into_iter().enumerate() {
        match result_chunk {
            Ok(Some(data)) if !data.is_empty() => all_good_chromosome_data.push(data),
            Ok(Some(_)) | Ok(None) => {
                debug!(
                    "No variants passed filters for VCF: {}",
                    vcf_files[i].display()
                );
            }
            Err(e) => processing_errors
                .push(anyhow!("Error processing VCF file {}: {}", vcf_files[i].display(), e)),
        }
    }

    if !processing_errors.is_empty() {
        for err in processing_errors {
            error!("{}", err);
        }
        return Err(anyhow!(
            "Failed to process one or more VCF files. See errors above."
        ));
    }

    if all_good_chromosome_data.is_empty() {
        return Err(anyhow!(
            "No variants passed filters across all VCF files. Cannot proceed with PCA."
        ));
    }

    // --- 3. Aggregate Data & Build Matrix ---
    info!(
        "Aggregating variant data from {} processed VCF file(s)...",
        all_good_chromosome_data.len()
    );
    let (variant_ids, chromosomes, positions, numerical_genotypes_variant_major) =
        matrix_ops::aggregate_chromosome_data(all_good_chromosome_data);

    let num_total_variants = variant_ids.len();
    info!(
        "Aggregated {} variants in total across all VCFs.",
        num_total_variants
    );

    if num_total_variants == 0 {
        return Err(anyhow!(
            "No variants available for PCA after aggregation."
        ));
    }

    info!(
        "Building genotype matrix ({} samples x {} variants)...",
        samples_info.sample_count, num_total_variants
    );
    let genotype_matrix = matrix_ops::build_matrix(
        numerical_genotypes_variant_major,
        samples_info.sample_count,
    )?;

    // --- 4. Run PCA ---
    info!("Running PCA...");
    let (pca_model, transformed_pcs, pc_variances) =
        pca_runner::run_genomic_pca(genotype_matrix, &cli_args)?;
    info!(
        "PCA computation complete. Resulted in {} principal components.",
        transformed_pcs.ncols()
    );

    // --- 5. Write Outputs ---
    let output_prefix_path = PathBuf::from(&cli_args.output_prefix);
    if let Some(parent) = output_prefix_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                anyhow!("Failed to create output directory {}: {}", parent.display(), e)
            })?;
            info!("Created output directory: {}", parent.display());
        }
    }
    info!(
        "Writing results to files with prefix '{}'...",
        cli_args.output_prefix
    );

    output_writer::write_principal_components(
        &cli_args.output_prefix,
        &samples_info.sample_names,
        &transformed_pcs,
    )?;
    output_writer::write_eigenvalues(&cli_args.output_prefix, &pc_variances)?;

    // efficient-pca 0.1.3 does not expose the rotation matrix directly after rfit.
    // Loadings cannot be written.
    warn!("Rotation matrix (loadings) is not available with the current version of efficient-pca after rfit. Loadings output will be skipped.");
    // The entire if/else block for pca_model.rotation() has been removed.

    info!(
        "genomic_pca finished successfully in {:.2?}.",
        total_time_start.elapsed()
    );
    Ok(())
}

// --- Module Implementations ---

mod cli {
    use std::path::PathBuf;
    use clap::Parser; // For the derive macro to find Parser

    #[derive(Parser, Debug)]
    #[command(author, version, about = "Genomic PCA Tool from VCF files.", long_about = None, propagate_version = true)]
    pub(crate) struct CliArgs {
        #[arg(short = 'd', long = "vcf-dir", required = true)]
        pub(crate) vcf_dir: PathBuf,

        #[arg(short, long = "out", required = true)]
        pub(crate) output_prefix: String,

        #[arg(short = 'k', long, required = true)]
        pub(crate) components: usize,

        #[arg(long, default_value_t = 0.01)]
        pub(crate) maf: f64,

        #[arg(long)]
        pub(crate) rfit_seed: Option<u64>,

        #[arg(short = 't', long)]
        pub(crate) threads: Option<usize>,

        #[arg(long, default_value = "Info")]
        pub(crate) log_level: String,
    }
}

mod pca_runner {
    use super::{anyhow, warn, info, Result, Error, Array2, EfficientPcaModel, cli};

    pub(crate) fn run_genomic_pca(
        genotype_matrix: Array2<f64>, // This matrix is consumed by rfit
        cli_args: &cli::CliArgs,
    ) -> Result<(EfficientPcaModel, Array2<f64>, Vec<f64>), Error> {
        let mut pca_model = EfficientPcaModel::new();
        let mut k_requested_components = cli_args.components;

        // --- 1. Validate Input Parameters ---
        if k_requested_components == 0 {
            return Err(anyhow!("Number of components (-k) must be > 0."));
        }

        let num_samples = genotype_matrix.nrows();
        let num_features = genotype_matrix.ncols();

        if num_samples < 2 {
            return Err(anyhow!("PCA requires at least 2 samples, found {}.", num_samples));
        }
        if num_features == 0 {
            // This check is important as efficient_pca::rfit itself also errors on 0 features.
            return Err(anyhow!("PCA requires at least 1 variant (feature), found 0."));
        }

        // Adjust k_requested_components if it exceeds maximum possible rank
        let max_possible_k = num_samples.min(num_features);
        if k_requested_components > max_possible_k {
            warn!(
                "Requested k={} components exceeds max possible for data ({} samples x {} features), which is {}. Adjusting to {}.",
                k_requested_components, num_samples, num_features, max_possible_k, max_possible_k
            );
            k_requested_components = max_possible_k;
        }

        // If, after adjustment, k is 0 (e.g., if max_possible_k was 0, though num_features=0 is caught above), error out.
        if k_requested_components == 0 {
            // This could happen if max_possible_k is 0 because n_samples or n_features is 0,
            // although specific checks for num_features == 0 and num_samples < 2 exist.
            // This check ensures k_requested_components for rfit is non-zero.
            return Err(anyhow!(
                "Effective number of components to request (k_requested_components) is 0 for matrix ({} samples x {} features). Cannot proceed.",
                num_samples, num_features
            ));
        }

        // --- 2. Run PCA using efficient_pca::rfit ---
        // `efficient_pca::PCA::rfit` now consumes the genotype_matrix and
        // returns the principal component scores directly. It also populates the pca_model.
        let n_oversamples = 10; // Standard oversampling parameter for randomized SVD
        let seed = cli_args.rfit_seed;
        // `tolerance_rfit` is set to None, meaning rfit uses its default behavior.
        let tolerance_rfit = None;

        let data_for_transform = genotype_matrix.clone(); // Clone for transform call

        info!(
            "Running efficient_pca rfit: k_requested={}, n_oversamples={}, seed={:?}, rfit_tolerance=None",
            k_requested_components, n_oversamples, seed
        );
        
        pca_model // Modified rfit call
            .rfit(
                genotype_matrix, // Consumed here
                k_requested_components,
                n_oversamples,
                seed,
                tolerance_rfit,
            )
            .map_err(|e| anyhow!("PCA computation with rfit failed: {}", e.to_string()))?;

        // Now, call transform to get the principal component scores
        let transformed_pcs = pca_model.transform(data_for_transform)
            .map_err(|e| anyhow!("PCA transformation failed after rfit: {}", e.to_string()))?;
        
        // `rfit` populates self.rotation and self.explained_variance within pca_model.
        // The number of columns in transformed_pcs is the actual number of components kept by rfit.
        let num_components_kept = transformed_pcs.ncols();

        // --- 3. Handle Results and Variances ---
        if num_components_kept == 0 && k_requested_components > 0 {
            // This warning is useful if rfit decided to keep 0 components due to data properties or internal logic,
            // even if components were requested.
            warn!(
                "PCA model resulted in 0 components being kept by rfit, despite requesting {} (adjusted from {}). This can occur with low-rank data or strict tolerance if used.",
                num_components_kept, cli_args.components
            );
            // transformed_pcs from rfit will be an N x 0 matrix in this case.
            // pca_model.explained_variance() should also yield an empty or zeroed vector.
        } else if num_components_kept < k_requested_components {
            // Log if the number of components rfit decided to keep is less than what was requested (after adjustments).
            info!(
                "PCA model effectively computed {} components (requested/capped at {}).",
                num_components_kept, k_requested_components
            );
        }
        // The explicit pca_model.transform call on the original genotype_matrix is no longer needed.

        // Get principal component variances (eigenvalues) from the PCA model.
        // efficient-pca 0.1.3 does not expose explained_variance directly after rfit.
        // We will return an empty Vec and log a warning.
        warn!("Explained variance data is not available with the current version of efficient-pca after rfit. Eigenvalues output will be empty.");
        let pc_variances: Vec<f64> = Vec::new();

        // The function returns the fitted model, the PC scores for the training data, and their variances.
        Ok((pca_model, transformed_pcs, pc_variances))
    }
}

mod output_writer {
    use super::{anyhow, info, warn, Result, Array2, File, BufWriter, Write};

    fn create_output_file(prefix: &str, suffix: &str) -> Result<BufWriter<File>> {
        let filename = format!("{}.{}", prefix, suffix);
        File::create(&filename)
            .map(BufWriter::new)
            .map_err(|e| anyhow!("Failed to create output file {}: {}", filename, e))
    }

    pub(crate) fn write_principal_components(
        output_prefix: &str,
        sample_names: &[String],
        transformed_pcs: &Array2<f64>,
    ) -> Result<()> {
        if transformed_pcs.ncols() == 0 {
            info!("No principal components to write.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "pca.tsv")?;
        info!("Writing principal components to {}.pca.tsv", output_prefix);

        write!(writer, "SampleID")?;
        for i in 1..=transformed_pcs.ncols() {
            write!(writer, "\tPC{}", i)?;
        }
        writeln!(writer)?;

        for (sample_idx, sample_name) in sample_names.iter().enumerate() {
            write!(writer, "{}", sample_name)?;
            if sample_idx < transformed_pcs.nrows() {
                for pc_idx in 0..transformed_pcs.ncols() {
                    write!(writer, "\t{:.6}", transformed_pcs[[sample_idx, pc_idx]])?;
                }
            } else {
                warn!(
                    "Sample index {} out of bounds for PCs ({} rows). Writing NA.",
                    sample_idx, transformed_pcs.nrows()
                );
                for _ in 0..transformed_pcs.ncols() {
                    write!(writer, "\tNA")?;
                }
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    pub(crate) fn write_eigenvalues(
        output_prefix: &str,
        pc_variances: &[f64],
    ) -> Result<()> {
        if pc_variances.is_empty() {
            info!("No eigenvalues to write.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "eigenvalues.tsv")?;
        info!("Writing eigenvalues to {}.eigenvalues.tsv", output_prefix);

        writeln!(writer, "PC\tEigenvalue")?;
        for (i, variance) in pc_variances.iter().enumerate() {
            writeln!(writer, "{}\t{:.6}", i + 1, variance)?;
        }
        Ok(())
    }

    pub(crate) fn write_loadings(
        output_prefix: &str,
        variant_ids: &[String],
        chromosomes: &[String],
        positions: &[u64],
        rotation_matrix: &Array2<f64>,
    ) -> Result<()> {
        if rotation_matrix.ncols() == 0 {
            info!("No loadings to write (0 components).");
            return Ok(());
        }
        if variant_ids.is_empty() {
            info!("No variants for loadings.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "loadings.tsv")?;
        info!("Writing variant loadings to {}.loadings.tsv", output_prefix);

        write!(writer, "VariantID\tChrom\tPos")?;
        for i in 1..=rotation_matrix.ncols() {
            write!(writer, "\tPC{}_loading", i)?;
        }
        writeln!(writer)?;

        for variant_idx in 0..variant_ids.len() {
            if variant_idx >= chromosomes.len()
                || variant_idx >= positions.len()
                || variant_idx >= rotation_matrix.nrows()
            {
                warn!("Index out of bounds for loadings (variant {}). Skipping.", variant_idx);
                continue;
            }
            write!(
                writer,
                "{}\t{}\t{}",
                variant_ids[variant_idx], chromosomes[variant_idx], positions[variant_idx]
            )?;
            for pc_idx in 0..rotation_matrix.ncols() {
                write!(writer, "\t{:.6}", rotation_matrix[[variant_idx, pc_idx]])?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}
