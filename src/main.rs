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
    variant::record::samples::keys::Key as VcfSampleKey,
    variant::Record as VcfRecordExt, // To access trait methods like chromosome(), position() etc.
    Header as VcfHeader,
};
use rayon::prelude::*;
use std::{
    fs::{self, File}, // fs for read_dir
    io::{BufWriter, Write},
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

    // --- 1. Discover VCF files and Read Header from First VCF & Prepare Shared Info ---
    info!("Discovering VCF files in directory: {}", cli_args.vcf_dir.display());
    let mut vcf_files: Vec<PathBuf> = fs::read_dir(&cli_args.vcf_dir)?
        .filter_map(Result::ok) // Ignore read errors for individual entries
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() &&
            (path.extension().map_or(false, |ext| ext == "vcf" || ext == "gz")) // check for .vcf or .vcf.gz
        })
        .collect();

    if vcf_files.is_empty() {
        return Err(anyhow!("No VCF files (.vcf, .vcf.gz) found in directory: {}", cli_args.vcf_dir.display()));
    }
    vcf_files.sort(); // Ensure consistent order for selecting the first VCF
    info!("Found {} VCF file(s). Processing order: {:?}", vcf_files.len(), vcf_files.iter().take(5).collect::<Vec<_>>());


    let first_vcf_path = &vcf_files[0];
    info!("Reading header from first VCF: {}", first_vcf_path.display());
    let mut first_reader = vcf::io::reader::Builder::default()
        .build_from_path(first_vcf_path)?;
    let header_template = Arc::new(first_reader.read_header()?);
    let samples_info = Arc::new(vcf_processing::SamplesHeaderInfo::from_header(&header_template, first_vcf_path)?);
    info!(
        "Established sample set from {}: {} samples. All other VCFs must match this set and order.",
        first_vcf_path.display(),
        samples_info.sample_count
    );
    debug!("Sample names (first few): {:?}", samples_info.sample_names.iter().take(5).collect::<Vec<_>>());


    // --- 2. Parallel VCF Processing ---
    info!("Processing {} VCF file(s) in parallel...", vcf_files.len());
    let pb_vcf_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} VCFs ({percent}%) ETA: {eta}")
        .map_err(|e| anyhow!("Failed to create progress bar style: {}", e))?
        .progress_chars("=> ");
    let pb_vcf = ProgressBar::new(vcf_files.len() as u64).with_style(pb_vcf_style);

    let per_chromosome_data_results: Vec<Result<Option<Vec<vcf_processing::VariantGenotypeData>>>> = vcf_files // Use the discovered and sorted list
        .par_iter()
        .map(|vcf_path| {
            // Pass first_vcf_path (as &Path) to process_single_vcf for context in error messages if needed.
            let result = vcf_processing::process_single_vcf(vcf_path, samples_info.clone(), &cli_args, first_vcf_path);
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
                 debug!("No variants passed filters for VCF: {}", vcf_files[i].display());
            }
            Err(e) => processing_errors.push(
                anyhow!("Error processing VCF file {}: {}", vcf_files[i].display(), e)
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
    let (pca_model, transformed_pcs, pc_variances) =
        pca_runner::run_genomic_pca(genotype_matrix, &cli_args)?;
    info!("PCA computation complete. Resulted in {} principal components.", transformed_pcs.ncols());

    // --- 5. Write Outputs ---
    let output_prefix_path = PathBuf::from(&cli_args.output_prefix);
    if let Some(parent) = output_prefix_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
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

    if let Some(rotation) = pca_model.rotation() {
        if rotation.ncols() > 0 { 
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
    use super::*; 

    #[derive(Parser, Debug)]
    #[clap(author, version, about = "Genomic PCA Tool from VCF files.", long_about = None, propagate_version = true)]
    pub(crate) struct CliArgs {
        /// Directory containing input VCF/VCF.gz files.
        #[clap(short = 'd', long = "vcf-dir", required = true)]
        pub(crate) vcf_dir: PathBuf,

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
    use noodles_vcf::variant::record::Samples as VcfSamplesTrait; // For trait methods
    use noodles_vcf::variant::record::samples::Series as VcfSeriesTrait; // For trait methods

    #[derive(Debug)]
    pub(crate) struct SamplesHeaderInfo {
        pub(crate) sample_names: Vec<String>,
        pub(crate) sample_count: usize,
    }

    impl SamplesHeaderInfo {
        pub(crate) fn from_header(header: &VcfHeader, filepath: &Path) -> Result<Self> {
            let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
            let sample_count = sample_names.len();
            if sample_count == 0 {
                return Err(anyhow!("VCF header from {} contains no samples.", filepath.display()));
            }
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
        pub(crate) numerical_genotypes: Vec<u8>, 
    }

    #[inline(always)]
    fn parse_gt_to_option_u8(gt_string: &str) -> Option<u8> {
        let bytes = gt_string.as_bytes();
        
        if bytes.len() != 3 { return None; } 
        if bytes[1] != b'/' && bytes[1] != b'|' { return None; }

        let allele1 = match bytes[0] {
            b'0' => 0u8,
            b'1' => 1u8,
            b'.' => return None, 
            _ => return None,   
        };
        let allele2 = match bytes[2] {
            b'0' => 0u8,
            b'1' => 1u8,
            b'.' => return None, 
            _ => return None,   
        };
        
        Some(allele1 + allele2)
    }

    pub(crate) fn process_single_vcf(
        vcf_path: &Path,
        canonical_samples_info: Arc<SamplesHeaderInfo>,
        cli_args: &cli::CliArgs,
        first_vcf_path_for_error_msg: &Path, // Used for more informative error messages
    ) -> Result<Option<Vec<VariantGenotypeData>>> {
        debug!("Processing VCF: {}", vcf_path.display());
        let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
        let current_header = reader.read_header()?;

        if current_header.sample_names().len() != canonical_samples_info.sample_count ||
           current_header.sample_names().iter().zip(canonical_samples_info.sample_names.iter()).any(|(s1,s2)| s1 != s2) {
            return Err(anyhow!(
                "Sample mismatch in VCF {}: Expected {} samples starting with '{:?}', but found {} samples starting with '{:?}'. All VCFs must have identical sample sets in the same order as the first VCF ({}).",
                vcf_path.display(),
                canonical_samples_info.sample_count, canonical_samples_info.sample_names.iter().take(3).collect::<Vec<_>>(),
                current_header.sample_names().len(), current_header.sample_names().iter().take(3).collect::<Vec<_>>(),
                first_vcf_path_for_error_msg.display()
            ));
        }

        // Get the GT key string from the FORMAT fields in the current header.
        // `VcfSampleKey::Genotype` is a `noodles_vcf::variant::record::samples::keys::Key`.
        let gt_format_key_str = VcfSampleKey::Genotype.as_ref(); // This gives "GT"

        // Check if "GT" is actually defined in the header's FORMAT fields.
        // While `get_index_of` was used before, we need the key string for `samples_obj.get()`.
        // And it's good practice to ensure "GT" is defined if we expect to use it.
        if !current_header.formats().contains_key(gt_format_key_str) {
             return Err(anyhow!("GT key (FORMAT={}) not found in FORMAT header for VCF {}", gt_format_key_str, vcf_path.display()));
        }

        let mut chromosome_variants_data = Vec::new();
        let mut record_buffer = vcf::Record::default(); 

        while reader.read_record(&current_header, &mut record_buffer)? != 0 {
            let record = &record_buffer; // record is &vcf::Record

            // --- Filter for bi-allelic SNPs only ---
            let ref_bases_str = record.reference_bases(); // Is &str
            let alt_bases_obj = record.alternate_bases(); // Is AlternateBases<'_>

            if ref_bases_str.len() != 1 ||                  // Ref allele is 1 char
               alt_bases_obj.len() != 1 ||                 // Exactly one alt allele
               alt_bases_obj.as_ref().len() != 1 {         // That alt allele is 1 char
                debug!("Variant at {}:{} (REF:{}, ALT:{}) is not a bi-allelic SNP, skipping.", 
                    record.chromosome(), 
                    record.position().map_or(0, |p| p.get()),
                    ref_bases_str,
                    alt_bases_obj.as_ref()
                );
                continue;
            }
            
            // --- Genotype Extraction & "Drop if ANY Missing" Rule ---
            let mut temp_genotypes_for_variant: Vec<u8> = Vec::with_capacity(canonical_samples_info.sample_count);
            let mut has_any_missing_or_unparsable_gt = false;
            let mut processed_sample_count = 0;

            let samples_obj = record.samples()?; // Result<Box<dyn VcfSamplesTrait + '_>>

            // Get the series for the "GT" key
            match samples_obj.get(gt_format_key_str)? { // Pass the key string "GT"
                Some(gt_series) => { // gt_series is Box<dyn VcfSeriesTrait + '_>
                    for value_option_result in gt_series.iter() {
                        processed_sample_count += 1;
                        match value_option_result {
                            Ok(Some(vcf::variant::record::samples::series::Value::String(gt_string_cow))) => {
                                if let Some(gt_val) = parse_gt_to_option_u8(gt_string_cow.as_ref()) {
                                    temp_genotypes_for_variant.push(gt_val);
                                } else {
                                    has_any_missing_or_unparsable_gt = true;
                                    break; 
                                }
                            }
                            Ok(Some(_)) | Ok(None) => { // GT is not a String, or is missing (represented by None value in series)
                                has_any_missing_or_unparsable_gt = true;
                                break;
                            }
                            Err(e) => {
                                warn!(
                                    "Error parsing a genotype value for variant at {}:{} in VCF {}: {}",
                                    record.chromosome(), record.position().map_or(0, |p| p.get()),
                                    vcf_path.display(), e
                                );
                                has_any_missing_or_unparsable_gt = true;
                                break;
                            }
                        }
                    }
                }
                None => { // "GT" key not found in this particular record's samples (e.g. no samples have GT)
                    debug!(
                        "Variant at {}:{} in VCF {} is missing the GT series for all samples.",
                        record.chromosome(), record.position().map_or(0, |p| p.get()),
                        vcf_path.display()
                    );
                    has_any_missing_or_unparsable_gt = true;
                }
            }
            
            // Final check after attempting to process all samples for this variant
            if has_any_missing_or_unparsable_gt || temp_genotypes_for_variant.len() != canonical_samples_info.sample_count {
                 if processed_sample_count != canonical_samples_info.sample_count && !has_any_missing_or_unparsable_gt {
                    // This case implies the GT series had fewer values than expected samples,
                    // which might indicate an issue with the VCF or noodles parsing consistency.
                    warn!(
                        "Variant at {}:{} in VCF {}: GT series yielded {} values, but expected {} samples. Dropping variant.",
                        record.chromosome(), record.position().map_or(0, |p| p.get()),
                        vcf_path.display(),
                        temp_genotypes_for_variant.len(), // or processed_sample_count if it's more accurate
                        canonical_samples_info.sample_count
                    );
                }
                debug!(
                    "Variant at {}:{} in VCF {} dropped due to missing, unparsable, or incomplete genotype data across samples (collected {} valid GTs, needed {}).",
                    record.chromosome(), record.position().map_or(0, |p| p.get()),
                    vcf_path.display(),
                    temp_genotypes_for_variant.len(),
                    canonical_samples_info.sample_count
                );
                continue;
            }

            // --- MAF Filtering ---
            let allele_sum: u32 = temp_genotypes_for_variant.iter().map(|&g| g as u32).sum();
            let num_alleles_total: u32 = (canonical_samples_info.sample_count * 2) as u32; 

            if num_alleles_total == 0 { 
                debug!("Variant at {}:{} dropped due to zero total alleles (no samples).", record.chromosome(), record.position().map_or(0, |p| p.get()));
                continue;
            }

            let alt_allele_freq = allele_sum as f64 / num_alleles_total as f64;
            let maf = alt_allele_freq.min(1.0 - alt_allele_freq); 

            let maf_threshold = cli_args.maf; 
            if maf < maf_threshold {
                debug!(
                    "Variant at {}:{} (MAF={:.4}) dropped due to MAF < threshold ({:.4}).",
                    record.chromosome(), record.position().map_or(0, |p| p.get()), maf, maf_threshold
                );
                continue;
            }
            
            // --- Store Variant Data ---
            let chrom_str = record.chromosome().to_string();
            let pos_val = record.position().map_or(0, |p| p.get() as u64); 
            // let ref_allele_str = ref_bases_str.to_string(); // Already a &str
            // Corrected: alt_bases_obj.as_ref() is already the string of the single alt allele
            let alt_allele_str = alt_bases_obj.as_ref().to_string(); 

            let variant_id = format!("{}:{}:{}:{}", chrom_str, pos_val, ref_bases_str, alt_allele_str);

            chromosome_variants_data.push(VariantGenotypeData {
                id: variant_id,
                chromosome: chrom_str,
                position: pos_val,
                numerical_genotypes: temp_genotypes_for_variant,
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
        per_chromosome_data: Vec<Vec<VariantGenotypeData>>, 
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
        if num_samples == 0 { 
            return Err(anyhow!("No samples available to build matrix (num_samples is 0)."));
        }

        let mut final_matrix = Array2::<f64>::zeros((num_samples, num_variants));

        for (variant_idx, genotypes_for_one_variant) in variant_genotypes_major.iter().enumerate() {
            if genotypes_for_one_variant.len() != num_samples {
                return Err(anyhow!(
                    "Internal error: Genotype count mismatch for variant at index {}: expected {}, found {}.",
                    variant_idx, num_samples, genotypes_for_one_variant.len()
                ));
            }
            for sample_idx in 0..num_samples {
                final_matrix[[sample_idx, variant_idx]] = genotypes_for_one_variant[sample_idx] as f64;
            }
        }
        Ok(final_matrix)
    }
}

mod pca_runner {
    use super::*;

    pub(crate) fn run_genomic_pca(
        genotype_matrix: Array2<f64>, 
        cli_args: &cli::CliArgs,
    ) -> Result<(EfficientPcaModel, Array2<f64>, Vec<f64>), Error> {
        
        let mut pca_model = EfficientPcaModel::new();

        let mut k_actual = cli_args.components;
        if k_actual == 0 {
            return Err(anyhow!("Number of components (-k) must be greater than 0."));
        }
        
        let num_samples = genotype_matrix.nrows();
        let num_features = genotype_matrix.ncols(); // variants
        
        if num_samples < 2 { 
            return Err(anyhow!("PCA requires at least 2 samples, found {}.", num_samples));
        }
        if num_features == 0 { 
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

        let mut matrix_for_rfit = genotype_matrix.clone(); 
        pca_model.rfit(
            matrix_for_rfit, 
            k_actual,       
            n_oversamples,
            seed,
            tolerance_rfit,
        )?;
        
        let computed_k_in_model = pca_model.rotation().map_or(0, |r| r.ncols());
        if computed_k_in_model == 0 && k_actual > 0 {
            warn!("PCA model (rfit) resulted in 0 components despite requesting {}. This can happen with data that has no resolvable variance after internal checks.", k_actual);
            let transformed_pcs_empty = Array2::<f64>::zeros((num_samples, 0));
            let pc_variances_empty = Vec::new();
            return Ok((pca_model, transformed_pcs_empty, pc_variances_empty));
        } else if computed_k_in_model < k_actual {
            info!("PCA model computed {} components (requested/capped at {}).", computed_k_in_model, k_actual);
        }

        let transformed_pcs = pca_model.transform(genotype_matrix.clone())?; 

        let pc_variances: Vec<f64> = transformed_pcs
            .axis_iter(Axis(1)) 
            .map(|pc_column| pc_column.var(0.0)) 
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
                warn!("Sample index {} out of bounds for transformed PCs ({} rows). Writing fewer PC rows than samples.", sample_idx, transformed_pcs.nrows());
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
        rotation_matrix: &Array2<f64>, 
    ) -> Result<()> {
        if rotation_matrix.ncols() == 0 { 
            info!("No loadings to write (0 components in rotation matrix).");
            return Ok(());
        }
        if variant_ids.is_empty() { 
             info!("No variants (features) for which to write loadings.");
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
