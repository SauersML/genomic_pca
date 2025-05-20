// main.rs

// --- External Crate Imports ---
use anyhow::{anyhow, Error, Result};
use clap::Parser;
use efficient_pca::PCA as EfficientPcaModel;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, info, warn, error};
use ndarray::{Array2, Axis};
use noodles_vcf::{
    self as vcf,
    variant::io::Read as VcfIoRead,
    variant::record::samples::keys::Key as VcfSampleKey,
    Header as VcfHeader,
};
use rayon::prelude::*;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

// --- Project Modules (defined within this file) ---
mod cli;
mod vcf_processing;
mod matrix_ops;
mod pca_runner;
mod output_writer;

// --- Main Function ---
fn main() -> Result<(), Error> {
    let total_time_start = Instant::now();
    let cli_args = cli::CliArgs::parse();

    // Initialize logger
    let log_level = cli_args.log_level.parse::<log::LevelFilter>()
        .unwrap_or_else(|_| {
            eprintln!("Warning: Invalid log level '{}' provided. Defaulting to Info.", cli_args.log_level);
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

    // --- 1. Read Header from First VCF & Prepare Shared Info ---
    if cli_args.vcfs.is_empty() {
        return Err(anyhow!("No VCF files provided. Use --vcfs option."));
    }
    info!("Reading header from first VCF: {}", cli_args.vcfs[0].display());
    let mut first_reader = vcf::io::reader::Builder::default()
        .build_from_path(&cli_args.vcfs[0])?;
    let header_template = Arc::new(first_reader.read_header()?); // Arc for sharing
    let samples_info = Arc::new(vcf_processing::SamplesHeaderInfo::from_header(&header_template, &cli_args.vcfs[0])?);
    info!(
        "Established sample set from {}: {} samples. All other VCFs must match this set and order.",
        cli_args.vcfs[0].display(),
        samples_info.sample_count
    );
    debug!("Sample names (first few): {:?}", samples_info.sample_names.iter().take(5).collect::<Vec<_>>());


    // --- 2. Parallel VCF Processing ---
    info!("Processing {} VCF file(s) in parallel...", cli_args.vcfs.len());
    let pb_vcf_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} VCFs ({percent}%) ETA: {eta}")
        .map_err(|e| anyhow!("Failed to create progress bar style: {}", e))?
        .progress_chars("=> "); // Using different progress chars
    let pb_vcf = ProgressBar::new(cli_args.vcfs.len() as u64).with_style(pb_vcf_style);

    let per_chromosome_data_results: Vec<Result<Option<Vec<vcf_processing::VariantGenotypeData>>>> = cli_args
        .vcfs
        .par_iter()
        .map(|vcf_path| {
            let result = vcf_processing::process_single_vcf(vcf_path, samples_info.clone(), &cli_args);
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
                 debug!("No variants passed filters for VCF: {}", cli_args.vcfs[i].display());
            }
            Err(e) => processing_errors.push(
                anyhow!("Error processing VCF file {}: {}", cli_args.vcfs[i].display(), e)
            ),
        }
    }

    if !processing_errors.is_empty() {
        for err in processing_errors {
            error!("{}", err);
        }
        return Err(anyhow!("Failed to process one or more VCF files. See errors above."));
    }
    
    if all_good_chromosome_data.is_empty() {
         return Err(anyhow!("No variants passed filters across all VCF files. Cannot proceed with PCA."));
    }

    // --- 3. Aggregate Data & Build Matrix ---
    info!("Aggregating variant data from {} processed chromosome file(s)...", all_good_chromosome_data.len());
    let (variant_ids, chromosomes, positions, numerical_genotypes_variant_major) =
        matrix_ops::aggregate_chromosome_data(all_good_chromosome_data);
    
    let num_total_variants = variant_ids.len();
    info!(
        "Aggregated {} variants in total across all chromosomes.",
        num_total_variants
    );

    if num_total_variants == 0 {
        return Err(anyhow!("No variants available for PCA after aggregation."));
    }

    info!("Building genotype matrix ({} samples x {} variants)...", samples_info.sample_count, num_total_variants);
    let genotype_matrix = matrix_ops::build_matrix(
        numerical_genotypes_variant_major,
        samples_info.sample_count,
    )?;

    // --- 4. Run PCA ---
    info!("Running PCA...");
    // Pass genotype_matrix by value as it won't be needed in its original state in main after this.
    // run_genomic_pca will clone it internally as needed.
    let (pca_model, transformed_pcs, pc_variances) =
        pca_runner::run_genomic_pca(genotype_matrix, &cli_args)?;
    info!("PCA computation complete. Resulted in {} principal components.", transformed_pcs.ncols());

    // --- 5. Write Outputs ---
    let output_prefix_path = PathBuf::from(&cli_args.output_prefix);
    if let Some(parent) = output_prefix_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() { // Check if parent is not root/empty
            std::fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create output directory {}: {}", parent.display(), e))?;
            info!("Created output directory: {}", parent.display());
        }
    }
    info!("Writing results to files with prefix '{}'...", cli_args.output_prefix);

    output_writer::write_principal_components(
        &cli_args.output_prefix,
        &samples_info.sample_names,
        &transformed_pcs,
    )?;
    output_writer::write_eigenvalues(&cli_args.output_prefix, &pc_variances)?;

    if let Some(rotation) = pca_model.rotation() { // Use accessor from efficient_pca
        if rotation.ncols() > 0 { // Only write if there are actual loading vectors
            output_writer::write_loadings(
                &cli_args.output_prefix,
                &variant_ids,
                &chromosomes,
                &positions,
                rotation,
            )?;
        } else {
            info!("PCA model has 0 components in rotation matrix, skipping loadings output.");
        }
    } else {
        warn!("Rotation matrix not available in PCA model, skipping loadings output.");
    }

    info!("genomic_pca finished successfully in {:.2?}.", total_time_start.elapsed());
    Ok(())
}


// --- Module Implementations ---

mod cli {
    use super::*; // To bring PathBuf into scope for clap derive

    #[derive(Parser, Debug)]
    #[clap(author, version, about = "Genomic PCA Tool from VCF files.", long_about = None, propagate_version = true)]
    pub(crate) struct CliArgs {
        /// Input VCF/VCF.gz file paths (one per chromosome).
        #[clap(short, long = "vcfs", required = true, num_args = 1.., value_delimiter = ',')]
        pub(crate) vcfs: Vec<PathBuf>,

        /// Prefix for output files (e.g., "my_analysis/pca_results").
        #[clap(short, long = "out", required = true)]
        pub(crate) output_prefix: String,

        /// Number of principal components to compute.
        #[clap(short = 'k', long, required = true)]
        pub(crate) components: usize,

        /// Minimum Minor Allele Frequency (MAF) threshold for variants.
        #[clap(long, default_value_t = 0.01)]
        pub(crate) maf: f64,

        /// Seed for randomized PCA (rfit) for reproducible results.
        #[clap(long)]
        pub(crate) rfit_seed: Option<u64>,

        /// Number of threads for parallel operations. Defaults to available physical cores.
        #[clap(short = 't', long)]
        pub(crate) threads: Option<usize>,

        /// Logging verbosity [Options: Error, Warn, Info, Debug, Trace].
        #[clap(long, default_value = "Info")]
        pub(crate) log_level: String,
    }
}

mod vcf_processing {
    use super::*;

    #[derive(Debug)]
    pub(crate) struct SamplesHeaderInfo {
        pub(crate) sample_names: Vec<String>,
        pub(crate) sample_count: usize,
        // No longer storing gt_format_key_index here, will be determined per file
    }

    impl SamplesHeaderInfo {
        pub(crate) fn from_header(header: &VcfHeader, filepath: &Path) -> Result<Self> {
            let sample_names: Vec<String> = header.sample_names().map(String::from).collect();
            let sample_count = sample_names.len();
            if sample_count == 0 {
                return Err(anyhow!("VCF header from {} contains no samples.", filepath.display()));
            }
            // GT key presence is checked per file in process_single_vcf
            Ok(Self {
                sample_names,
                sample_count,
            })
        }
    }

    #[derive(Debug)]
    pub(crate) struct VariantGenotypeData {
        pub(crate) id: String,
        pub(crate) chromosome: String,
        pub(crate) position: u64,
        pub(crate) numerical_genotypes: Vec<u8>, // 0,1,2. No 255 here.
    }

    /// Parses a diploid genotype string (e.g., "0/0", "0|1", "1/1") into an optional sum of alternate alleles.
    /// Returns `Some(u8)` (0, 1, or 2) for valid bi-allelic diploid genotypes using 0 and 1.
    /// Returns `None` for missing (`./.`), haploid, non-diploid, or if alleles are not '0' or '1'.
    #[inline(always)]
    fn parse_gt_to_option_u8(gt_string: &str) -> Option<u8> {
        let bytes = gt_string.as_bytes();
        
        if bytes.len() != 3 { return None; } // Must be "A/B" or "A|B" form
        if bytes[1] != b'/' && bytes[1] != b'|' { return None; } // Separator check

        let allele1 = match bytes[0] {
            b'0' => 0u8,
            b'1' => 1u8,
            b'.' => return None, // Missing allele part
            _ => return None,    // Invalid character for allele
        };
        let allele2 = match bytes[2] {
            b'0' => 0u8,
            b'1' => 1u8,
            b'.' => return None, // Missing allele part
            _ => return None,    // Invalid character for allele
        };
        
        // This function assumes alleles are '0' or '1'.
        // It does not handle cases like '2' for a second ALT allele.
        // These are implicitly treated as "unparsable" and lead to None.
        Some(allele1 + allele2)
    }

    pub(crate) fn process_single_vcf(
        vcf_path: &Path,
        canonical_samples_info: Arc<SamplesHeaderInfo>, // Used for validation and consistent sample count
        cli_args: &cli::CliArgs,
    ) -> Result<Option<Vec<VariantGenotypeData>>> {
        debug!("Processing VCF: {}", vcf_path.display());
        let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
        let current_header = reader.read_header()?;

        // --- Validate current VCF's samples against the canonical list from the first VCF ---
        if current_header.sample_names().len() != canonical_samples_info.sample_count ||
           current_header.sample_names().iter().zip(canonical_samples_info.sample_names.iter()).any(|(s1,s2)| s1 != s2) {
            return Err(anyhow!(
                "Sample mismatch in VCF {}: Expected {} samples starting with '{:?}', but found {} samples starting with '{:?}'. All VCFs must have identical sample sets in the same order as the first VCF ({}).",
                vcf_path.display(),
                canonical_samples_info.sample_count, canonical_samples_info.sample_names.iter().take(3).collect::<Vec<_>>(),
                current_header.sample_names().len(), current_header.sample_names().iter().take(3).collect::<Vec<_>>(),
                cli_args.vcfs[0].display()
            ));
        }

        // Get GT key index from the current file's header
        let current_gt_key_idx = current_header
            .formats()
            .get_index_of(&VcfSampleKey::Genotype)
            .ok_or_else(|| anyhow!("GT key not found in FORMAT header for VCF {}", vcf_path.display()))?;

        let mut chromosome_variants_data = Vec::new();
        let mut record_buffer = vcf::Record::default(); // Re-use buffer for efficiency

        // Use current_header for parsing records from this file
        while reader.read_record(&current_header, &mut record_buffer)? != 0 {
            let record = &record_buffer;

            // --- Filter for bi-allelic SNPs only ---
            let ref_bases = record.reference_bases();
            let alt_bases = record.alternate_bases();
            if ref_bases.as_ref().len() != 1 || alt_bases.len() != 1 || alt_bases.as_ref()[0].as_ref().len() != 1 {
                debug!("Variant at {}:{} is not a bi-allelic SNP, skipping.", record.chromosome(), record.position().map_or(0, |p| p.get()));
                continue;
            }
            
            // --- Genotype Extraction & "Drop if ANY Missing" Rule ---
            let mut temp_genotypes_for_variant: Vec<u8> = Vec::with_capacity(canonical_samples_info.sample_count);
            let mut has_any_missing_or_unparsable_gt = false;

            let record_samples = record.samples()?;

            for sample_idx in 0..canonical_samples_info.sample_count {
                if let Some(sample_fields) = record_samples.get_index(sample_idx) {
                    if let Some(Ok(Some(vcf::variant::record::samples::series::Value::String(gt_string)))) = sample_fields.get_index(current_gt_key_idx) {
                        if let Some(gt_val) = parse_gt_to_option_u8(gt_string) {
                            temp_genotypes_for_variant.push(gt_val);
                        } else {
                            has_any_missing_or_unparsable_gt = true;
                            break; 
                        }
                    } else {
                        has_any_missing_or_unparsable_gt = true; 
                        break;
                    }
                } else {
                    warn!("Could not get sample fields for sample index {} at {}:{}. This should not happen if headers match.", sample_idx, record.chromosome(), record.position().map_or(0, |p| p.get()));
                    has_any_missing_or_unparsable_gt = true;
                    break;
                }
            }

            if has_any_missing_or_unparsable_gt {
                debug!(
                    "Variant at {}:{} dropped due to missing, unparsable, or non-standard genotype in at least one sample.",
                    record.chromosome(), record.position().map_or(0, |p| p.get())
                );
                continue;
            }

            // --- MAF Filtering ---
            let allele_sum: u32 = temp_genotypes_for_variant.iter().map(|&g| g as u32).sum();
            let num_alleles_total: u32 = (canonical_samples_info.sample_count * 2) as u32; // Diploid assumed

            if num_alleles_total == 0 { // Should be caught by sample_count > 0 check earlier
                debug!("Variant at {}:{} dropped due to zero total alleles (no samples).", record.chromosome(), record.position().map_or(0, |p| p.get()));
                continue;
            }

            let alt_allele_freq = allele_sum as f64 / num_alleles_total as f64;
            let maf = alt_allele_freq.min(1.0 - alt_allele_freq); // Handles case where alt is major

            let maf_threshold = cli_args.maf; // Use directly from CliArgs
            if maf < maf_threshold {
                debug!(
                    "Variant at {}:{} (MAF={:.4}) dropped due to MAF < threshold ({:.4}).",
                    record.chromosome(), record.position().map_or(0, |p| p.get()), maf, maf_threshold
                );
                continue;
            }
            
            // --- Store Variant Data ---
            let chrom_str = record.chromosome().to_string();
            let pos_val = record.position().map_or(0, |p| p.get() as u64); // noodles Position is 1-based
            let ref_allele_str = ref_bases.to_string(); // Already confirmed to be single char
            let alt_allele_str = alt_bases.as_ref()[0].to_string(); // Already confirmed bi-allelic, SNP

            let variant_id = format!("{}:{}:{}:{}", chrom_str, pos_val, ref_allele_str, alt_allele_str);

            chromosome_variants_data.push(VariantGenotypeData {
                id: variant_id,
                chromosome: chrom_str,
                position: pos_val,
                numerical_genotypes: temp_genotypes_for_variant, // Contains no missing values (255)
            });
        }

        if chromosome_variants_data.is_empty() {
            Ok(None)
        } else {
            Ok(Some(chromosome_variants_data))
        }
    }
}

mod matrix_ops {
    use super::*;

    pub(crate) fn aggregate_chromosome_data(
        per_chromosome_data: Vec<Vec<VariantGenotypeData>>, // Input is already Vec<Vec<...>>
    ) -> (Vec<String>, Vec<String>, Vec<u64>, Vec<Vec<u8>>) {
        let mut all_variant_ids = Vec::new();
        let mut all_chromosomes = Vec::new();
        let mut all_positions = Vec::new();
        let mut all_numerical_genotypes_variant_major = Vec::new();

        for chrom_data_vec in per_chromosome_data {
            for variant_data in chrom_data_vec {
                all_variant_ids.push(variant_data.id);
                all_chromosomes.push(variant_data.chromosome);
                all_positions.push(variant_data.position);
                all_numerical_genotypes_variant_major.push(variant_data.numerical_genotypes);
            }
        }
        (all_variant_ids, all_chromosomes, all_positions, all_numerical_genotypes_variant_major)
    }

    pub(crate) fn build_matrix(
        variant_genotypes_major: Vec<Vec<u8>>, 
        num_samples: usize,
    ) -> Result<Array2<f64>> {
        let num_variants = variant_genotypes_major.len();
        if num_variants == 0 {
            return Err(anyhow!("No variants available to build matrix after aggregation."));
        }
        if num_samples == 0 { // Should be caught earlier by SamplesHeaderInfo
            return Err(anyhow!("No samples available to build matrix (num_samples is 0)."));
        }

        // Create matrix as (samples x variants)
        // Initialize with zeros, or consider uninit for slight perf gain if careful
        let mut final_matrix = Array2::<f64>::zeros((num_samples, num_variants));

        for (variant_idx, genotypes_for_one_variant) in variant_genotypes_major.iter().enumerate() {
            if genotypes_for_one_variant.len() != num_samples {
                // This should ideally not happen if parsing and aggregation are correct
                return Err(anyhow!(
                    "Internal error: Genotype count mismatch for variant at index {}: expected {}, found {}.",
                    variant_idx, num_samples, genotypes_for_one_variant.len()
                ));
            }
            for sample_idx in 0..num_samples {
                // Genotypes are already 0, 1, or 2 (u8). No NaNs expected due to "drop if any missing" rule.
                final_matrix[[sample_idx, variant_idx]] = genotypes_for_one_variant[sample_idx] as f64;
            }
        }
        Ok(final_matrix)
    }
}

mod pca_runner {
    use super::*;

    pub(crate) fn run_genomic_pca(
        genotype_matrix: Array2<f64>, // Consumes the original (numerically encoded, complete) matrix
        cli_args: &cli::CliArgs,
    ) -> Result<(EfficientPcaModel, Array2<f64>, Vec<f64>), Error> {
        
        let mut pca_model = EfficientPcaModel::new();

        let mut k_actual = cli_args.components;
        if k_actual == 0 {
            return Err(anyhow!("Number of components (-k) must be greater than 0."));
        }
        
        let num_samples = genotype_matrix.nrows();
        let num_features = genotype_matrix.ncols(); // variants
        
        if num_samples < 2 { // efficient_pca requires at least 2 samples
            return Err(anyhow!("PCA requires at least 2 samples, found {}.", num_samples));
        }
        if num_features == 0 { // efficient_pca requires features
            return Err(anyhow!("PCA requires at least 1 variant (feature), found 0."));
        }


        let max_possible_k = num_samples.min(num_features);

        if k_actual > max_possible_k {
            warn!(
                "Requested components k={} exceeds max possible ({}) based on matrix dimensions ({} samples x {} variants). Adjusting k to {}.",
                k_actual, max_possible_k, num_samples, num_features, max_possible_k
            );
            k_actual = max_possible_k;
        }
        
        // After adjustment, if k_actual is still 0 (e.g., if max_possible_k was 0 and k_actual was capped to it)
        // this check catches it. This ensures k_actual >= 1 for rfit.
        if k_actual == 0 {
             return Err(anyhow!(
                "Effective number of components to compute is 0. Matrix dimensions ({} samples x {} variants) are too small.",
                num_samples, num_features
            ));
        }

        let n_oversamples = 10; 
        let seed = cli_args.rfit_seed;
        let tolerance_rfit = None; 

        info!(
            "Running efficient_pca rfit with k={}, oversamples={}, seed={:?}, tolerance=None",
            k_actual, n_oversamples, seed
        );

        // `rfit` modifies `matrix_for_rfit` in place (centers/scales).
        let mut matrix_for_rfit = genotype_matrix.clone(); 
        pca_model.rfit(
            matrix_for_rfit, // This matrix will be modified by rfit
            k_actual,        // Use the (potentially capped) k_actual
            n_oversamples,
            seed,
            tolerance_rfit,
        )?;
        
        let computed_k_in_model = pca_model.rotation().map_or(0, |r| r.ncols());
        if computed_k_in_model == 0 && k_actual > 0 {
            warn!("PCA model (rfit) resulted in 0 components despite requesting {}. This can happen with data that has no resolvable variance after internal checks.", k_actual);
            // Return empty results gracefully
            let transformed_pcs_empty = Array2::<f64>::zeros((num_samples, 0));
            let pc_variances_empty = Vec::new();
            return Ok((pca_model, transformed_pcs_empty, pc_variances_empty));
        } else if computed_k_in_model < k_actual {
            info!("PCA model computed {} components (requested/capped at {}).", computed_k_in_model, k_actual);
        }


        // `transform` applies the learned mean/scale to the input matrix.
        // Pass the original (numerically encoded, complete) genotype_matrix.
        // efficient_pca::transform takes `mut x` and modifies it. So pass a clone if original is needed.
        // Here, genotype_matrix passed to run_genomic_pca is consumed, so we can clone it for transform.
        let transformed_pcs = pca_model.transform(genotype_matrix.clone())?; // Pass a clone of original numeric matrix

        let pc_variances: Vec<f64> = transformed_pcs
            .axis_iter(Axis(1)) 
            .map(|pc_column| pc_column.var(0.0)) // ndarray's var(0.0) is variance with N in denominator
            .collect();

        Ok((pca_model, transformed_pcs, pc_variances))
    }
}

mod output_writer {
    use super::*;

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
            info!("No principal components to write (0 components computed).");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "pca.tsv")?;
        info!("Writing principal components to {}.pca.tsv", output_prefix);

        // Header
        write!(writer, "SampleID")?;
        for i in 1..=transformed_pcs.ncols() {
            write!(writer, "\tPC{}", i)?;
        }
        writeln!(writer)?;

        // Data
        for (sample_idx, sample_name) in sample_names.iter().enumerate() {
            write!(writer, "{}", sample_name)?;
            if sample_idx < transformed_pcs.nrows() { // sample_idx is within bounds
                for pc_idx in 0..transformed_pcs.ncols() {
                    write!(writer, "\t{:.6}", transformed_pcs[[sample_idx, pc_idx]])?;
                }
            } else {
                // This case should ideally not be reached if sample_names and transformed_pcs are consistent
                warn!("Sample index {} out of bounds for transformed PCs ({} rows). Writing fewer PC rows than samples.", sample_idx, transformed_pcs.nrows());
                for _ in 0..transformed_pcs.ncols() { // Write empty tabs to maintain column structure
                    write!(writer, "\tNA")?;
                }
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    pub(crate) fn write_eigenvalues(
        output_prefix: &str,
        pc_variances: &[f64], // These are the eigenvalues
    ) -> Result<()> {
        if pc_variances.is_empty() {
            info!("No eigenvalues (PC variances) to write.");
            return Ok(());
        }
        let mut writer = create_output_file(output_prefix, "eigenvalues.tsv")?;
        info!("Writing eigenvalues (PC variances) to {}.eigenvalues.tsv", output_prefix);

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
        rotation_matrix: &Array2<f64>, // n_features (variants) x k_components
    ) -> Result<()> {
        if rotation_matrix.ncols() == 0 { // k_components
            info!("No loadings to write (0 components in rotation matrix).");
            return Ok(());
        }
        if variant_ids.is_empty() { // n_features
             info!("No variants (features) for which to write loadings.");
            return Ok(());
        }

        let mut writer = create_output_file(output_prefix, "loadings.tsv")?;
        info!("Writing variant loadings to {}.loadings.tsv", output_prefix);
        
        // Header
        write!(writer, "VariantID\tChrom\tPos")?;
        for i in 1..=rotation_matrix.ncols() { 
            write!(writer, "\tPC{}_loading", i)?;
        }
        writeln!(writer)?;

        // Data
        // rotation_matrix rows correspond to variants, columns to PCs
        for variant_idx in 0..variant_ids.len() { 
            // Check bounds to be safe, though lengths should match if logic is correct
            if variant_idx >= chromosomes.len() || variant_idx >= positions.len() || variant_idx >= rotation_matrix.nrows() {
                warn!("Index out of bounds when writing loadings for variant_idx {}. Skipping.", variant_idx);
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
