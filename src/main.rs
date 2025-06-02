// main.rs

// --- Local Module Declarations ---
mod vcf;          // 
mod pca_runner;   // For the VCF workflow using EfficientPcaModel
mod output_writer;// For writing results from both workflows

// --- External Crate Imports ---
use anyhow::{anyhow, Error, Result};
use clap::Parser;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use ndarray::{Array1, Array2, s as ndarray_s}; // Aliased 's'
use noodles_vcf::{self as noodles_vcf_crate, Header as VcfHeader}; // Aliased noodles_vcf
use num_cpus;
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    sync::Arc,
    time::Instant,
};

// --- Imports for Local Project Crates/Modules (Workflows) ---
use crate::cli::CliArgs;
// VCF Workflow specific
use crate::vcf::vcf_processing::SamplesHeaderInfo;
use crate::matrix_ops::{aggregate_chromosome_data, build_matrix};
use crate::pca_runner::run_genomic_pca;
// EigenSNP-Rust Data Preparation (local, from src/prepare.rs)
use crate::prepare::{MicroarrayDataPreparer, MicroarrayDataPreparerConfig};


// --- Imports for External `efficient_pca` Crate ---
use efficient_pca::PCA as EfficientPcaModel; // For the VCF workflow

// For EigenSNP-Rust Core Algorithm & Interface Types from external crate
use efficient_pca::eigensnp::{
    EigenSNPCoreAlgorithm, EigenSNPCoreAlgorithmConfig, EigenSNPCoreOutput,
    LdBlockSpecification,       // Used by MicroarrayDataPreparer and EigenSNPCoreAlgorithm
    PcaReadyGenotypeAccessor,   // Trait implemented by MicroarrayGenotypeAccessor in src/prepare.rs
    // PcaSnpId, QcSampleId,    // These are used by types above and in src/prepare.rs
    // ThreadSafeStdError,      // Used in PcaReadyGenotypeAccessor definition in src/prepare.rs
};

// Conditional import for EigenSNP diagnostics handling
#[cfg(feature = "eigensnp-diagnostics")]
use efficient_pca::diagnostics::FullPcaRunDetailedDiagnostics;
#[cfg(feature = "eigensnp-diagnostics")]
use serde_json;

// --- Main Function ---
fn main() -> Result<(), Error> {
    let total_time_start = Instant::now();
    let cli_args = CliArgs::parse();

    // Initialize logger
    let log_level = cli_args
        .log_level
        .parse::<log::LevelFilter>()
        .unwrap_or_else(|_| {
            eprintln!( /* logging setup */);
            log::LevelFilter::Info
        });
    env_logger::Builder::new()
        .filter_level(log_level)
        .format_timestamp_micros()
        .init();

    info!("Starting genomic_pca with CLI args: {:?}", cli_args);

    // Configure Rayon thread pool
    let num_threads = cli_args.threads.unwrap_or_else(num_cpus::get);
    info!("Using {} threads for parallel operations.", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()?;

    // --- Workflow Dispatch ---
    if cli_args.eigensnp {
        info!("EigenSNP-Rust workflow selected (using BED/LD input).");
        run_eigensnp_rust_workflow(&cli_args)?;
    } else {
        info!("Default VCF processing workflow selected.");
        // Validate required arguments for VCF workflow
        if cli_args.vcf_dir.is_none() {
            return Err(anyhow!("--vcf-dir is required for the default VCF workflow."));
        }
        if cli_args.components.is_none() {
             return Err(anyhow!("-k/--components is required for the default VCF workflow."));
        }
        run_vcf_workflow(&cli_args)?;
    }

    info!(
        "genomic_pca (mode: {}) finished successfully in {:.2?}.",
        if cli_args.eigensnp { "EigenSNP-Rust" } else { "VCF" },
        total_time_start.elapsed()
    );
    Ok(())
}

// --- VCF Workflow Function ---
fn run_vcf_workflow(cli_args: &CliArgs) -> Result<(), Error> {
    // Unwrap validated VCF workflow specific arguments
    let vcf_dir = cli_args.vcf_dir.as_ref().expect("--vcf-dir is required for VCF workflow and should be validated by now");
    // Note: cli_args.components is unwrapped inside pca_runner::run_genomic_pca

    info!("Discovering VCF files in directory: {}", vcf_dir.display());
    let mut vcf_files: Vec<PathBuf> = fs::read_dir(vcf_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && (path.extension().map_or(false, |ext| ext == "vcf" || ext == "gz"))
                && (path.file_name().map_or(false, |name| name.to_string_lossy().contains(".vcf")))
        })
        .collect();

    if vcf_files.is_empty() {
        return Err(anyhow!("No VCF files (ending in .vcf or .vcf.gz) found in directory: {}", vcf_dir.display()));
    }
    vcf_files.sort(); // consistent processing order
    info!("Found {} VCF file(s). Processing order (first 5 shown): {:?}", vcf_files.len(), vcf_files.iter().take(5).collect::<Vec<_>>());

    let first_vcf_path = &vcf_files[0];
    info!("Reading header from first VCF file: {}", first_vcf_path.display());
    let mut first_reader = noodles_vcf_crate::io::reader::Builder::default().build_from_path(first_vcf_path)?;
    let header_template = Arc::new(first_reader.read_header()?);
    let samples_info = Arc::new(SamplesHeaderInfo::from_header(&header_template, first_vcf_path)?);
    info!("Established sample set from {}: {} samples. All other VCFs must match this set and order.", first_vcf_path.display(), samples_info.sample_count);
    debug!("Sample names (first 5): {:?}", samples_info.sample_names.iter().take(5).collect::<Vec<_>>());

    info!("Processing {} VCF file(s) in parallel...", vcf_files.len());
    let pb_vcf_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} VCFs ({percent}%) ETA: {eta}")
        .map_err(|e| anyhow!("Failed to create progress bar style: {}", e))?
        .progress_chars("=> ");
    let pb_vcf = ProgressBar::new(vcf_files.len() as u64).with_style(pb_vcf_style.clone()); // Clone for thread safety if needed by ProgressBar internals

    let per_chromosome_data_results: Vec<Result<Option<Vec<vcf::vcf_processing::VariantGenotypeData>>>> = vcf_files
        .par_iter()
        .map(|vcf_path| {
            // cli_args is passed to process_single_vcf if it needs it for MAF etc.
            let result = crate::vcf::vcf_processing::process_single_vcf(vcf_path, samples_info.clone(), cli_args, first_vcf_path);
            pb_vcf.inc(1);
            result
        })
        .collect();
    pb_vcf.finish_with_message("VCF processing complete.");

    let mut all_good_chromosome_data: Vec<Vec<vcf::vcf_processing::VariantGenotypeData>> = Vec::new();
    let mut processing_errors: Vec<Error> = Vec::new();
    for (i, result_chunk) in per_chromosome_data_results.into_iter().enumerate() {
        match result_chunk {
            Ok(Some(data)) if !data.is_empty() => all_good_chromosome_data.push(data),
            Ok(Some(_)) | Ok(None) => debug!("No variants passed filters for VCF: {}", vcf_files[i].display()),
            Err(e) => processing_errors.push(anyhow!("Error processing VCF file {}: {}", vcf_files[i].display(), e)),
        }
    }

    if !processing_errors.is_empty() {
        for err in processing_errors { error!("{}", err); }
        return Err(anyhow!("Failed to process one or more VCF files. See errors above."));
    }
    if all_good_chromosome_data.is_empty() {
        return Err(anyhow!("No variants passed filters across all VCF files. Cannot proceed with PCA."));
    }

    info!("Aggregating variant data from {} processed VCF file segments...", all_good_chromosome_data.len());
    let (variant_ids, chromosomes, positions, numerical_genotypes_variant_major) = aggregate_chromosome_data(all_good_chromosome_data);
    let num_total_variants = variant_ids.len();
    info!("Aggregated {} variants in total across all VCFs.", num_total_variants);
    if num_total_variants == 0 { return Err(anyhow!("No variants available for PCA after aggregation.")); }

    info!("Building genotype matrix ({} samples x {} variants) for VCF workflow...", samples_info.sample_count, num_total_variants);
    let genotype_matrix = build_matrix(numerical_genotypes_variant_major, samples_info.sample_count)?;

    info!("Running PCA using efficient-pca library...");
    let (_pca_model, transformed_pcs, pc_variances) = run_genomic_pca(genotype_matrix, cli_args)?;
    info!("VCF PCA computation complete. Resulted in {} principal components.", transformed_pcs.ncols());

    let output_prefix_path = PathBuf::from(&cli_args.output_prefix);
    if let Some(parent) = output_prefix_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| anyhow!("Failed to create output directory {}: {}", parent.display(), e))?;
            info!("Created output directory: {}", parent.display());
        }
    }
    info!("Writing VCF PCA results to files with prefix '{}'...", cli_args.output_prefix);

    output_writer::write_principal_components_f64(&cli_args.output_prefix, &samples_info.sample_names, &transformed_pcs)?;
    output_writer::write_eigenvalues(&cli_args.output_prefix, &pc_variances)?;
    warn!("Loadings output for VCF-based PCA is currently skipped (not directly available from efficient-pca rfit).");

    Ok(())
}

// --- EigenSNP-Rust Workflow Function ---
fn run_eigensnp_rust_workflow(cli_args: &CliArgs) -> Result<(), Error> {
    info!("Initializing EigenSNP-Rust PCA workflow.");

    // 1. Create Configurations from cli_args
    let bed_file_path_str = cli_args.bed_file.as_ref()
        .ok_or_else(|| anyhow!("--bed-file is required when --eigensnp is used"))?
        .to_string_lossy().into_owned();
    let ld_block_file_path_str = cli_args.ld_block_file.as_ref()
        .ok_or_else(|| anyhow!("--ld-block-file is required when --eigensnp is used"))?
        .to_string_lossy().into_owned();

    let prep_config = MicroarrayDataPreparerConfig {
        bed_file_path: bed_file_path_str,
        ld_block_file_path: ld_block_file_path_str,
        sample_ids_to_keep_file_path: cli_args.eigensnp_sample_keep_file.as_ref().map(|p| p.to_string_lossy().into_owned()),
        min_snp_call_rate_threshold: cli_args.eigensnp_min_call_rate.unwrap_or(0.98),
        min_snp_maf_threshold: cli_args.eigensnp_min_maf.unwrap_or(0.01),
        max_snp_hwe_p_value_threshold: cli_args.eigensnp_max_hwe_p.unwrap_or(1e-6),
    };

    let algo_config = EigenSNPCoreAlgorithmConfig {
        target_num_global_pcs: cli_args.eigensnp_k_global.unwrap_or(10),
        components_per_ld_block: cli_args.eigensnp_components_per_block.unwrap_or(7),
        subset_factor_for_local_basis_learning: cli_args.eigensnp_subset_factor.unwrap_or(0.075),
        min_subset_size_for_local_basis_learning: cli_args.eigensnp_min_subset_size.unwrap_or(10_000),
        max_subset_size_for_local_basis_learning: cli_args.eigensnp_max_subset_size.unwrap_or(40_000),
        global_pca_sketch_oversampling: cli_args.eigensnp_global_oversampling.unwrap_or(10),
        global_pca_num_power_iterations: cli_args.eigensnp_global_power_iter.unwrap_or(2),
        local_rsvd_sketch_oversampling: cli_args.eigensnp_local_oversampling.unwrap_or(10),
        local_rsvd_num_power_iterations: cli_args.eigensnp_local_power_iter.unwrap_or(2),
        random_seed: cli_args.eigensnp_seed.unwrap_or(2025),
        snp_processing_strip_size: cli_args.eigensnp_snp_strip_size.unwrap_or(2000),
        refine_pass_count: cli_args.eigensnp_refine_passes.unwrap_or(1),
        collect_diagnostics: cli_args.eigensnp_collect_diagnostics,
        #[cfg(feature = "eigensnp-diagnostics")]
        diagnostic_block_list_id_to_trace: None,
    };

    // 2. Prepare Data using MicroarrayDataPreparer (from crate::prepare)
    info!("Initializing MicroarrayDataPreparer (from src/prepare.rs)...");
    let preparer = MicroarrayDataPreparer::try_new(prep_config)
        .map_err(|e| anyhow!("Failed to initialize MicroarrayDataPreparer from src/prepare.rs: {}", e))?;

    info!("Preparing data for EigenSNP-Rust (using src/prepare.rs)...");
    // prepare_data_for_eigen_snp returns types from efficient_pca::eigensnp if src/prepare.rs was modified correctly
    let (genotype_accessor, ld_block_specifications, num_qc_samples, num_pca_snps) =
        preparer.prepare_data_for_eigen_snp()
            .map_err(|e| anyhow!("Data preparation (src/prepare.rs) for EigenSNP-Rust failed: {}", e))?;
    
    info!("EigenSNP-Rust Data prepared: {} QC'd samples, {} PCA SNPs for analysis.", num_qc_samples, num_pca_snps);

    if num_qc_samples == 0 || num_pca_snps == 0 {
        warn!("No samples or SNPs available for EigenSNP-Rust PCA after preparation. EigenSNP-Rust workflow will not proceed further.");
        return Ok(());
    }

    // 3. Run EigenSNP-Rust PCA using external efficient_pca::eigensnp module
    info!("Initializing EigenSNPCoreAlgorithm (from efficient_pca crate)...");
    let algorithm = EigenSNPCoreAlgorithm::new(algo_config);

    info!("Computing EigenSNP-Rust PCA...");
    // compute_pca expects &impl PcaReadyGenotypeAccessor and &[LdBlockSpecification]
    // where these types are from efficient_pca::eigensnp
    let (pca_output, diagnostics_option) = algorithm.compute_pca(&genotype_accessor, &ld_block_specifications)
        .map_err(|e| anyhow!("EigenSNP-Rust PCA computation (efficient_pca crate) failed: {}", e))?;

    // 4. Write Outputs
    let output_prefix_path = PathBuf::from(&cli_args.output_prefix);
     if let Some(parent) = output_prefix_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| anyhow!("Failed to create output directory {}: {}", parent.display(), e))?;
            info!("Created output directory: {}", parent.display());
        }
    }
    info!("Writing EigenSNP-Rust PCA results to files with prefix '{}'...", cli_args.output_prefix);

    // Retrieve QC'd sample names using the accessor for original_indices_of_qc_samples
    let ordered_qc_sample_names = get_eigensnp_ordered_qc_sample_names(
        &preparer,
        genotype_accessor.original_indices_of_qc_samples().as_slice(), // 
    )?;
    output_writer::write_principal_components_f32(
        &cli_args.output_prefix,
        &ordered_qc_sample_names,
        &pca_output.final_sample_principal_component_scores,
    )?;

    output_writer::write_eigenvalues(
        &cli_args.output_prefix,
        &pca_output.final_principal_component_eigenvalues.to_vec(), // Convert Array1<f64> to Vec<f64> for slice
    )?;

    // Retrieve PCA SNP metadata using the accessor for original_indices_of_pca_snps
    let (pca_snp_ids, pca_snp_chroms, pca_snp_pos) = get_eigensnp_ordered_pca_snp_metadata(
        &preparer,
        genotype_accessor.original_indices_of_pca_snps().as_slice(),
    )?;
    output_writer::write_loadings_f32(
        &cli_args.output_prefix,
        &pca_snp_ids,
        &pca_snp_chroms,
        &pca_snp_pos,
        &pca_output.final_snp_principal_component_loadings,
    )?;
    
    // Conditional diagnostics output
    #[cfg(feature = "enable-eigensnp-diagnostics")]
    if cli_args.eigensnp_collect_diagnostics {
        // The type of diagnostics_option here is `Option<efficient_pca::diagnostics::FullPcaRunDetailedDiagnostics>`
        // if the feature `enable-eigensnp-diagnostics` is active for the `efficient_pca` crate.
        // If not, it's `()`.
        if let Some(diag_data) = diagnostics_option {
            let diag_filename = format!("{}.eigensnp_diagnostics.json", cli_args.output_prefix);
            info!("Writing EigenSNP-Rust diagnostics to {}...", diag_filename);
            match serde_json::to_string_pretty(&diag_data) {
                Ok(json_string) => {
                    if let Err(e_write) = std::fs::write(&diag_filename, json_string) {
                        warn!("Failed to write EigenSNP diagnostics to {}: {}", diag_filename, e_write);
                    }
                }
                Err(e_json) => warn!("Failed to serialize EigenSNP diagnostics to JSON: {}", e_json),
            }
        } else {
             info!("EigenSNP-Rust diagnostics collection was enabled via CLI, but no diagnostic data structure was returned from PCA computation (this might happen if the 'enable-eigensnp-diagnostics' feature is not active in the 'efficient_pca' library itself).");
        }
    }
    #[cfg(not(feature = "enable-eigensnp-diagnostics"))]
    { let _ = diagnostics_option; /* Mark as used if diagnostics feature is off */ }

    Ok(())
}

// --- Helper for EigenSNP-Rust Ordered QC'd Sample Names ---
fn get_eigensnp_ordered_qc_sample_names(
    preparer: &MicroarrayDataPreparer,      // From crate::prepare
    qc_sample_original_indices: &[isize], // From MicroarrayGenotypeAccessor in crate::prepare
) -> Result<Vec<String>, Error> {
    let initial_fam_ids = preparer.initial_sample_ids_from_fam_arc();
    let mut ordered_names = Vec::with_capacity(qc_sample_original_indices.len());
    for &original_idx_isize in qc_sample_original_indices {
        let original_idx = original_idx_isize as usize; // Convert isize to usize for indexing
        if original_idx < initial_fam_ids.len() {
            ordered_names.push(initial_fam_ids[original_idx].clone());
        } else {
            return Err(anyhow!(
                "Original sample index {} out of bounds for initial FAM ID array (len {}).",
                original_idx,
                initial_fam_ids.len()
            ));
        }
    }
    Ok(ordered_names)
}

// --- Helper for EigenSNP-Rust Ordered PCA SNP Metadata ---
fn get_eigensnp_ordered_pca_snp_metadata(
    preparer: &MicroarrayDataPreparer,      // From crate::prepare
    original_indices_pca_snps: &[usize],  // From MicroarrayGenotypeAccessor in crate::prepare
) -> Result<(Vec<String>, Vec<String>, Vec<u64>), Error> {
    let mut snp_ids = Vec::with_capacity(original_indices_pca_snps.len());
    let mut snp_chroms = Vec::with_capacity(original_indices_pca_snps.len());
    let mut snp_pos = Vec::with_capacity(original_indices_pca_snps.len());

    let initial_sids = preparer.initial_bim_sids_arc();
    let initial_chroms = preparer.initial_bim_chromosomes_arc();
    let initial_positions = preparer.initial_bim_bp_positions_arc();

    for &original_idx in original_indices_pca_snps {
        if original_idx >= initial_sids.len() || original_idx >= initial_chroms.len() || original_idx >= initial_positions.len() {
            return Err(anyhow!(
                "Original SNP index {} out of bounds for BIM metadata arrays (SIDs len {}, Chroms len {}, Pos len {}).",
                original_idx, initial_sids.len(), initial_chroms.len(), initial_positions.len()
            ));
        }
        snp_ids.push(initial_sids[original_idx].clone());
        snp_chroms.push(initial_chroms[original_idx].clone());
        snp_pos.push(initial_positions[original_idx] as u64); // Cast i32 to u64 as per existing output_writer
    }
    Ok((snp_ids, snp_chroms, snp_pos))
}


// --- Module Implementations ---

mod cli {
    use std::path::PathBuf;
    use clap::Parser;

    #[derive(Parser, Debug)]
    #[command(author, version, about = "Genomic PCA Tool from VCF or BED/LD-block files.", long_about = None, propagate_version = true)]
    pub(crate) struct CliArgs {
        // --- Common Arguments ---
        #[arg(short, long = "out", required = true, help = "Output file prefix.")]
        pub(crate) output_prefix: String,

        #[arg(short = 't', long, help = "Number of threads for parallel operations (default: all available CPUs).")]
        pub(crate) threads: Option<usize>,

        #[arg(long, default_value = "Info", help = "Logging level (e.g., Off, Error, Warn, Info, Debug, Trace).")]
        pub(crate) log_level: String,

        // --- VCF Workflow Specific Arguments ---
        #[arg(short = 'd', long = "vcf-dir", help = "Directory containing VCF files (required if not using --eigensnp).", required_unless_present("eigensnp"))]
        pub(crate) vcf_dir: Option<PathBuf>,

        #[arg(short = 'k', long, help = "Number of principal components to compute (for VCF workflow).", required_unless_present("eigensnp"))]
        pub(crate) components: Option<usize>,

        #[arg(long, help = "Minimum Minor Allele Frequency (MAF) for VCF variant filtering (default: 0.01 for VCF workflow).", default_value_if("vcf_dir", Some("*"), Some("0.01")))]
        pub(crate) maf: Option<f64>, // Default applies if vcf_dir is used.

        #[arg(long, help = "Seed for randomized SVD in efficient-pca (for VCF workflow).")]
        pub(crate) rfit_seed: Option<u64>,

        // --- EigenSNP-Rust Workflow Flag ---
        #[arg(long, help = "Run PCA using the EigenSNP-Rust algorithm (requires BED & LD block files).")]
        pub(crate) eigensnp: bool,

        // --- EigenSNP-Rust Workflow Specific Arguments ---
        #[arg(long, help="Path to the BED file (required if --eigensnp is used).", required_if_eq("eigensnp", "true"))]
        pub(crate) bed_file: Option<PathBuf>,

        #[arg(long, help="Path to the LD block definition file (required if --eigensnp is used).", required_if_eq("eigensnp", "true"))]
        pub(crate) ld_block_file: Option<PathBuf>,

        #[arg(long, help="Optional: Path to a file listing sample IDs to keep (for --eigensnp).")]
        pub(crate) eigensnp_sample_keep_file: Option<PathBuf>,

        #[arg(long, help="Min SNP call rate for EigenSNP-Rust QC (default: 0.98 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("0.98")))]
        pub(crate) eigensnp_min_call_rate: Option<f64>,

        #[arg(long, help="Min SNP MAF for EigenSNP-Rust QC (default: 0.01 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("0.01")))]
        pub(crate) eigensnp_min_maf: Option<f64>,

        #[arg(long, help="Max SNP HWE p-value for EigenSNP-Rust QC (1.0 to disable; default: 1e-6 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("1e-6")))]
        pub(crate) eigensnp_max_hwe_p: Option<f64>,

        #[arg(long, help="Target number of global PCs for EigenSNP-Rust (default: 10 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("10")))]
        pub(crate) eigensnp_k_global: Option<usize>,

        #[arg(long, help="Number of local components per LD block for EigenSNP-Rust (default: 7 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("7")))]
        pub(crate) eigensnp_components_per_block: Option<usize>,

        #[arg(long, help="Subset factor for local basis learning in EigenSNP-Rust (default: 0.075 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("0.075")))]
        pub(crate) eigensnp_subset_factor: Option<f64>,

        #[arg(long, help="Min subset size for local basis learning in EigenSNP-Rust (default: 10000 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("10000")))]
        pub(crate) eigensnp_min_subset_size: Option<usize>,

        #[arg(long, help="Max subset size for local basis learning in EigenSNP-Rust (default: 40000 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("40000")))]
        pub(crate) eigensnp_max_subset_size: Option<usize>,

        #[arg(long, help="Global PCA sketch oversampling for EigenSNP-Rust (default: 10 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("10")))]
        pub(crate) eigensnp_global_oversampling: Option<usize>,

        #[arg(long, help="Global PCA power iterations for EigenSNP-Rust (default: 2 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("2")))]
        pub(crate) eigensnp_global_power_iter: Option<usize>,

        #[arg(long, help="Local RSVD sketch oversampling for EigenSNP-Rust (default: 10 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("10")))]
        pub(crate) eigensnp_local_oversampling: Option<usize>,

        #[arg(long, help="Local RSVD power iterations for EigenSNP-Rust (default: 2 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("2")))]
        pub(crate) eigensnp_local_power_iter: Option<usize>,

        #[arg(long, help="Random seed for EigenSNP-Rust (default: 2025 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("2025")))]
        pub(crate) eigensnp_seed: Option<u64>,

        #[arg(long, help="SNP processing strip size for EigenSNP-Rust (default: 2000 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("2000")))]
        pub(crate) eigensnp_snp_strip_size: Option<usize>,

        #[arg(long, help="Number of refinement passes for EigenSNP-Rust (default: 1 when --eigensnp is active).", default_value_if("eigensnp", "true", Some("1")))]
        pub(crate) eigensnp_refine_passes: Option<usize>,

        #[arg(long, help="Enable detailed diagnostics collection for EigenSNP-Rust (if library feature 'enable-eigensnp-diagnostics' is active).")]
        pub(crate) eigensnp_collect_diagnostics: bool, // Defaults to false
    }
}

mod pca_runner { // For VCF workflow
    use super::{anyhow, warn, info, Result, Error, Array2, EfficientPcaModel, CliArgs};

    pub(crate) fn run_genomic_pca(
        genotype_matrix: Array2<f64>,
        cli_args: &CliArgs,
    ) -> Result<(EfficientPcaModel, Array2<f64>, Vec<f64>), Error> {
        let mut pca_model = EfficientPcaModel::new();
        // This unwrap is safe because `components` is `required_unless_present("eigensnp")`
        // and this function is only called when `eigensnp` is false.
        let mut k_requested_components = cli_args.components.unwrap();

        if k_requested_components == 0 {
            return Err(anyhow!("Number of components (-k) must be > 0."));
        }

        let num_samples = genotype_matrix.nrows();
        let num_features = genotype_matrix.ncols();

        if num_samples < 2 {
            return Err(anyhow!("PCA requires at least 2 samples, found {}.", num_samples));
        }
        if num_features == 0 {
            return Err(anyhow!("PCA requires at least 1 variant (feature), found 0."));
        }

        let max_possible_k = num_samples.min(num_features);
        if k_requested_components > max_possible_k {
            warn!(
                "Requested k={} components exceeds max possible for data ({} samples x {} features), which is {}. Adjusting to {}.",
                k_requested_components, num_samples, num_features, max_possible_k, max_possible_k
            );
            k_requested_components = max_possible_k;
        }
        if k_requested_components == 0 {
             return Err(anyhow!(
                "Effective number of components to request (k_requested_components) is 0 for matrix ({} samples x {} features). Cannot proceed.",
                num_samples, num_features
            ));
        }

        let n_oversamples = 10; // A common default for randomized SVD
        let seed = cli_args.rfit_seed; // Pass through seed if provided
        let tolerance_rfit = None; // Use efficient-pca's default tolerance
        
        let data_for_transform = genotype_matrix.clone(); // `rfit` consumes the matrix

        info!(
            "Running efficient_pca rfit: k_requested={}, n_oversamples={}, seed={:?}, rfit_tolerance=None",
            k_requested_components, n_oversamples, seed
        );
        
        // The `rfit` method performs the PCA and stores loadings and explained variance internally.
        pca_model
            .rfit(
                genotype_matrix, // Consumed here
                k_requested_components,
                n_oversamples,
                seed,
                tolerance_rfit,
            )
            .map_err(|e| anyhow!("PCA computation with efficient-pca rfit failed: {}", e.to_string()))?;
            
        // `transform` applies the learned PCA to project data into the PC space.
        let transformed_pcs = pca_model.transform(data_for_transform)
            .map_err(|e| anyhow!("PCA transformation with efficient-pca failed after rfit: {}", e.to_string()))?;
        
        let num_components_kept = transformed_pcs.ncols();

        if num_components_kept == 0 && k_requested_components > 0 {
            warn!(
                "PCA model (efficient-pca) resulted in 0 components being kept by rfit, despite requesting {} (adjusted from initial {}).",
                 num_components_kept, cli_args.components.unwrap_or_default() // Use initial for logging
            );
        } else if num_components_kept < k_requested_components {
            info!(
                "PCA model (efficient-pca) effectively computed {} components (requested/capped at {}).",
                num_components_kept, k_requested_components
            );
        }

        let pc_variances: Vec<f64> = Vec::new(); // ?

        Ok((pca_model, transformed_pcs, pc_variances))
    }
}

mod output_writer {
    use super::{anyhow, info, warn, Result, Array2, File, BufWriter, Write};
    // Array1 might be needed if directly writing from ndarray::Array1
    use ndarray::Array1; 

    // Helper to create output files.
    fn create_output_file(prefix: &str, suffix: &str) -> Result<BufWriter<File>> {
        let filename = format!("{}.{}", prefix, suffix);
        File::create(&filename)
            .map(BufWriter::new)
            .map_err(|e| anyhow!("Failed to create output file '{}': {}", filename, e))
    }

    // Writes principal components (sample scores) for f64 data (VCF workflow).
    pub(crate) fn write_principal_components_f64(
        output_prefix: &str,
        sample_names: &[String],
        transformed_pcs: &Array2<f64>,
    ) -> Result<()> {
        if transformed_pcs.ncols() == 0 {
            info!("No f64 principal components (sample scores) to write.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "vcf.pca.tsv")?; // Distinguish from EigenSNP output
        info!("Writing f64 principal components (VCF workflow) to {}.vcf.pca.tsv", output_prefix);

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
                // This case should ideally not happen if sample_names and transformed_pcs correspond correctly.
                warn!("Sample index {} out of bounds for f64 PCs ({} rows). Writing NA for remaining PCs for this sample.", sample_idx, transformed_pcs.nrows());
                 for _ in 0..transformed_pcs.ncols() { write!(writer, "\tNA")?; }
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    // Writes principal components (sample scores) for f32 data (EigenSNP-Rust workflow).
    pub(crate) fn write_principal_components_f32(
        output_prefix: &str,
        sample_names: &[String],
        transformed_pcs_f32: &Array2<f32>,
    ) -> Result<()> {
        if transformed_pcs_f32.ncols() == 0 {
            info!("No f32 principal components (sample scores) to write.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "eigensnp.pca.tsv")?;
        info!("Writing f32 principal components (EigenSNP-Rust workflow) to {}.eigensnp.pca.tsv", output_prefix);

        write!(writer, "SampleID")?;
        for i in 1..=transformed_pcs_f32.ncols() {
            write!(writer, "\tPC{}", i)?;
        }
        writeln!(writer)?;

        for (sample_idx, sample_name) in sample_names.iter().enumerate() {
            write!(writer, "{}", sample_name)?;
            if sample_idx < transformed_pcs_f32.nrows() {
                for pc_idx in 0..transformed_pcs_f32.ncols() {
                    write!(writer, "\t{:.6}", transformed_pcs_f32[[sample_idx, pc_idx]])?;
                }
            } else {
                warn!("Sample index {} out of bounds for f32 PCs ({} rows). Writing NA for remaining PCs for this sample.", sample_idx, transformed_pcs_f32.nrows());
                 for _ in 0..transformed_pcs_f32.ncols() { write!(writer, "\tNA")?; }
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    // Writes eigenvalues (common for both workflows if data is &[f64]).
    pub(crate) fn write_eigenvalues(
        output_prefix: &str,
        pc_variances: &[f64],
    ) -> Result<()> {
        if pc_variances.is_empty() {
            info!("No eigenvalues to write.");
            // Create an empty file with header for consistency if desired, or just return.
            let mut writer = create_output_file(output_prefix, "eigenvalues.tsv")?;
            writeln!(writer, "PC\tEigenvalue")?; // Write header even if empty
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

    // Writes SNP loadings for f32 data (EigenSNP-Rust workflow).
    pub(crate) fn write_loadings_f32(
        output_prefix: &str,
        variant_ids: &[String],
        chromosomes: &[String],
        positions: &[u64],       // 
        loadings_matrix_f32: &Array2<f32>,
    ) -> Result<()> {
        if loadings_matrix_f32.ncols() == 0 {
            info!("No f32 SNP loadings to write (0 components).");
            return Ok(());
        }
        if variant_ids.is_empty() {
            info!("No variants for f32 SNP loadings.");
            // Create an empty file with header for consistency if desired.
            let mut writer = create_output_file(output_prefix, "eigensnp.loadings.tsv")?;
            write!(writer, "VariantID\tChrom\tPos")?;
            for i in 1..=loadings_matrix_f32.ncols() { write!(writer, "\tPC{}_loading", i)?; }
            writeln!(writer)?;
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "eigensnp.loadings.tsv")?;
        info!("Writing f32 SNP loadings (EigenSNP-Rust workflow) to {}.eigensnp.loadings.tsv", output_prefix);

        write!(writer, "VariantID\tChrom\tPos")?;
        for i in 1..=loadings_matrix_f32.ncols() {
            write!(writer, "\tPC{}_loading", i)?;
        }
        writeln!(writer)?;

        // variant metadata arrays have same length as loadings matrix rows
        if !(variant_ids.len() == loadings_matrix_f32.nrows() && 
             chromosomes.len() == loadings_matrix_f32.nrows() &&
             positions.len() == loadings_matrix_f32.nrows()) {
            return Err(anyhow!(
                "Mismatch in lengths of variant metadata and loadings matrix rows. VariantIDs: {}, Chroms: {}, Pos: {}, LoadingsRows: {}",
                variant_ids.len(), chromosomes.len(), positions.len(), loadings_matrix_f32.nrows()
            ));
        }

        for variant_idx in 0..loadings_matrix_f32.nrows() {
            // Bounds already checked by the assertion above for the primary arrays.
            write!(
                writer,
                "{}\t{}\t{}",
                variant_ids[variant_idx], chromosomes[variant_idx], positions[variant_idx]
            )?;
            for pc_idx in 0..loadings_matrix_f32.ncols() {
                write!(writer, "\t{:.6}", loadings_matrix_f32[[variant_idx, pc_idx]])?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }
}
