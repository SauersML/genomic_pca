// main.rs

// --- External Crate Imports ---
use anyhow::{anyhow, Error, Result};
use clap::Parser;
use efficient_pca::PCA as EfficientPcaModel;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use ndarray::{Array2, Axis};
use noodles_vcf::{
    self as vcf,
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
    let mut first_reader = vcf::io::reader::Builder::default().build_from_path(first_vcf_path)?;
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

mod vcf_processing {
    use super::{anyhow, debug, warn, Result, Path, Arc, VcfHeader, cli};
    use noodles_vcf::{
        self as vcf,
        variant::record::{
            samples::series::value::genotype::Phasing, AlternateBases as _,
        },
        // noodles_vcf::record::samples::Series is concrete, Series trait for dyn dispatch
        variant::record::samples::Series as VcfSeriesTrait,
    };

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
                return Err(anyhow!(
                    "VCF header from {} contains no samples.",
                    filepath.display()
                ));
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
            b'0' => 0u8, b'1' => 1u8, b'.' => return None, _ => return None,
        };
        let allele2 = match bytes[2] {
            b'0' => 0u8, b'1' => 1u8, b'.' => return None, _ => return None,
        };
        Some(allele1 + allele2)
    }

    pub(crate) fn process_single_vcf(
        vcf_path: &Path,
        canonical_samples_info: Arc<SamplesHeaderInfo>,
        cli_args: &cli::CliArgs, // Mark as unused if not used, or remove. Currently it's used for cli_args.maf
        first_vcf_path_for_error_msg: &Path,
    ) -> Result<Option<Vec<VariantGenotypeData>>> {
        debug!("Processing VCF: {}", vcf_path.display());
        let mut reader = vcf::io::reader::Builder::default().build_from_path(vcf_path)?;
        let current_header = reader.read_header()?;

        if current_header.sample_names().len() != canonical_samples_info.sample_count
            || current_header
                .sample_names()
                .iter()
                .zip(canonical_samples_info.sample_names.iter())
                .any(|(s1, s2)| s1 != s2)
        {
            return Err(anyhow!(
                "Sample mismatch in VCF {}: Expected {} samples (names like '{:?}'), but found {} samples (names like '{:?}'). All VCFs must match the sample set of the first VCF ({}).",
                vcf_path.display(),
                canonical_samples_info.sample_count, canonical_samples_info.sample_names.iter().take(3).collect::<Vec<_>>(),
                current_header.sample_names().len(), current_header.sample_names().iter().take(3).collect::<Vec<_>>(),
                first_vcf_path_for_error_msg.display()
            ));
        }
        
        let gt_key_str = vcf::variant::record::samples::keys::key::GENOTYPE.as_ref();

        if !current_header.formats().contains_key(gt_key_str) {
            return Err(anyhow!(
                "GT key (FORMAT={}) not found in FORMAT header for VCF {}",
                gt_key_str, vcf_path.display()
            ));
        }

        let mut chromosome_variants_data = Vec::new();
        let mut record_buffer = vcf::Record::default();

        while reader.read_record(&mut record_buffer)? != 0 {
            let record = &record_buffer;

            let ref_bases_str = record.reference_bases();
            let alt_bases_obj = record.alternate_bases();

            if ref_bases_str.len() != 1          // Reference is not a single base
                || alt_bases_obj.len() != 1      // Not exactly one alternate allele listed
                || alt_bases_obj.as_ref().first().map_or(true, |allele_struct| allele_struct.as_ref().len() != 1) // The string of the first (and only) alt allele is not length 1
            {
                debug!(
                    "Variant at {}:{} (REF:{}, ALT:{}) is not a bi-allelic SNP (single base REF, single base ALT), skipping.",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                    ref_bases_str,
                    alt_bases_obj.as_ref()
                );
                continue;
            }

            let mut temp_genotypes_for_variant: Vec<u8> =
                Vec::with_capacity(canonical_samples_info.sample_count);
            let mut has_any_missing_or_unparsable_gt = false;
            
            let samples_obj = record.samples();

            match samples_obj.select(gt_key_str) {
                Some(gt_series_struct) => {
                    for (sample_idx, value_option_result) in gt_series_struct.iter(&current_header).enumerate() {
                        if sample_idx >= canonical_samples_info.sample_count {
                            warn!("More GT values in series than expected samples for variant at {}:{}. VCF: {}. Truncating.",
                                record.reference_sequence_name(),
                                record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                vcf_path.display());
                            has_any_missing_or_unparsable_gt = true;
                            break;
                        }
                        match value_option_result {
                            Ok(Some(vcf::variant::record::samples::series::Value::String(gt_string_cow_val))) => {
                                let gt_str_slice = gt_string_cow_val.as_ref(); // gt_string_cow_val is Cow<'a, str>
                                if let Some(gt_val) = parse_gt_to_option_u8(gt_str_slice) {
                                    temp_genotypes_for_variant.push(gt_val);
                                } else {
                                    debug!(
                                        "Variant at {}:{}: GT field (String type, assumed Cow) for sample {} was unparsable ('{}'), skipping processing for this variant.",
                                        record.reference_sequence_name(),
                                        record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                        sample_idx,
                                        gt_str_slice
                                    );
                                    has_any_missing_or_unparsable_gt = true;
                                    break;
                                }
                            }
                            Ok(Some(vcf::variant::record::samples::series::Value::Genotype(boxed_gt))) => {
                                let genotype_trait_object = &*boxed_gt;
                                let mut allele_parts: Vec<String> = Vec::with_capacity(2);
                                let mut overall_phasing = Phasing::Unphased; 
                                let mut iteration_failed = false;
                                let mut collected_alleles = 0;

                                for result_item in genotype_trait_object.iter() {
                                    if collected_alleles >= 2 { // Process at most 2 alleles for diploid
                                        break;
                                    }
                                    match result_item {
                                        Ok((opt_allele_idx, current_phasing)) => {
                                            let allele_str = match opt_allele_idx {
                                                Some(0) => "0".to_string(),
                                                Some(1) => "1".to_string(),
                                                Some(idx) => idx.to_string(), // Other indices will make parse_gt_to_option_u8 fail
                                                None => ".".to_string(),    // Missing allele
                                            };
                                            allele_parts.push(allele_str);
                                            if current_phasing == Phasing::Phased {
                                                overall_phasing = Phasing::Phased;
                                            }
                                            collected_alleles += 1;
                                        }
                                        Err(_) => {
                                            debug!(
                                                "Variant at {}:{}: Error iterating alleles for GT field of sample {}, skipping.",
                                                record.reference_sequence_name(),
                                                record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                                sample_idx
                                            );
                                            iteration_failed = true;
                                            break; 
                                        }
                                    }
                                }

                                if iteration_failed {
                                    has_any_missing_or_unparsable_gt = true;
                                    break; 
                                }

                                let reconstructed_gt_string = if allele_parts.len() == 2 {
                                    let phasing_char = if overall_phasing == Phasing::Phased { '|' } else { '/' };
                                    format!("{}{}{}", allele_parts[0], phasing_char, allele_parts[1])
                                } else if allele_parts.len() == 1 {
                                     allele_parts[0].clone() // e.g. "0" or ".", will be rejected by parse_gt_to_option_u8 length check
                                }
                                 else { // 0 or >2 parts
                                    ".".to_string() // Default to ensure parse_gt_to_option_u8 fails
                                };
                                
                                if let Some(gt_val) = parse_gt_to_option_u8(&reconstructed_gt_string) {
                                    temp_genotypes_for_variant.push(gt_val);
                                } else {
                                    debug!(
                                        "Variant at {}:{}: Reconstructed GT string ('{}') for sample {} from Value::Genotype was unparsable by parse_gt_to_option_u8.",
                                        record.reference_sequence_name(),
                                        record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                        reconstructed_gt_string,
                                        sample_idx
                                    );
                                    has_any_missing_or_unparsable_gt = true;
                                    break;
                                }
                            }
                            Ok(Some(other_type)) => {
                                debug!("Variant at {}:{}: GT field for sample {} is an unexpected VCF Value type (type: {:?}), skipping processing for this variant.",
                                    record.reference_sequence_name(),
                                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    sample_idx,
                                    other_type);
                                has_any_missing_or_unparsable_gt = true;
                                break;
                            }
                            Ok(None) => { 
                                debug!(
                                    "Variant at {}:{}: GT field for sample {} is missing (None value from series iterator), skipping processing for this variant.",
                                    record.reference_sequence_name(),
                                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    sample_idx
                                );
                                has_any_missing_or_unparsable_gt = true;
                                break;
                            }
                            Err(e) => { 
                                warn!(
                                    "Error parsing a genotype value for variant at {}:{} in VCF {}: {}. Skipping processing for this variant.",
                                    record.reference_sequence_name(),
                                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    vcf_path.display(), e
                                );
                                has_any_missing_or_unparsable_gt = true;
                                break;
                            }
                        }
                    }
                }
                None => { 
                    debug!(
                        "Variant at {}:{}: GT series not found in VCF {}.",
                        record.reference_sequence_name(),
                        record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                        vcf_path.display()
                    );
                    has_any_missing_or_unparsable_gt = true;
                }
            }

            if has_any_missing_or_unparsable_gt
                || temp_genotypes_for_variant.len() != canonical_samples_info.sample_count
            {
                debug!(
                    "Variant at {}:{}:{} in VCF {} dropped (missing/unparsable/incomplete GTs: {}/{} processed).",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                    record.reference_bases(), 
                    vcf_path.display(),
                    temp_genotypes_for_variant.len(), // Number of GTs successfully parsed for this variant
                    canonical_samples_info.sample_count // Expected number of samples
                );
                continue;
            }

            let allele_sum: u32 = temp_genotypes_for_variant.iter().map(|&g| g as u32).sum();
            let num_alleles_total: u32 = (canonical_samples_info.sample_count * 2) as u32;

            if num_alleles_total == 0 { 
                debug!(
                    "Variant at {}:{}: No alleles to calculate MAF (num_alleles_total is 0). Skipping.",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64))
                );
                continue;
            }

            let alt_allele_freq = allele_sum as f64 / num_alleles_total as f64;
            let maf = alt_allele_freq.min(1.0 - alt_allele_freq);
            let maf_threshold = cli_args.maf;

            if maf < maf_threshold {
                debug!(
                    "Variant at {}:{}:{} (MAF={:.4}) below threshold ({:.4}). Skipping.",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                    record.reference_bases(),
                    maf, maf_threshold
                );
                continue;
            }
            
            // Get the first alternate allele string for the ID
            // alt_bases_obj here is AlternateBases which can be iterated.
            // We've already filtered for bi-allelic SNPs where alt_bases_obj.len() == 1
            // and the first alt allele is a single base/string.
            let alt_allele_str = alt_bases_obj.as_ref() // This is &[alternate_bases::Allele]
                .first() // This is Option<&alternate_bases::Allele>
                .map(|allele_struct| allele_struct.as_ref().to_string()) // allele_struct.as_ref() is &str
                .unwrap_or_else(|| {
                    // This case should ideally not be reached if the bi-allelic SNP check is correct
                    // and alt_bases_obj.len() == 1.
                    warn!("Could not determine alt_allele_str for variant ID construction, defaulting to N. This may indicate an issue with bi-allelic SNP filtering or AlternateBases structure.");
                    String::from("N")
                });
            let chrom_str = record.reference_sequence_name().to_string();
            let pos_val = record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64));
            
            let variant_id =
                format!("{}:{}:{}:{}", chrom_str, pos_val, ref_bases_str, alt_allele_str);

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
    use super::{anyhow, Result, Array2};
    use super::vcf_processing::VariantGenotypeData;

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
        (
            all_variant_ids,
            all_chromosomes,
            all_positions,
            all_numerical_genotypes_variant_major,
        )
    }

    pub(crate) fn build_matrix(
        variant_genotypes_major: Vec<Vec<u8>>,
        num_samples: usize,
    ) -> Result<Array2<f64>> {
        let num_variants = variant_genotypes_major.len();
        if num_variants == 0 {
            return Err(anyhow!("No variants available to build matrix."));
        }
        if num_samples == 0 {
            return Err(anyhow!("No samples available to build matrix."));
        }

        let mut final_matrix = Array2::<f64>::zeros((num_samples, num_variants));
        for (variant_idx, genotypes_for_one_variant) in
            variant_genotypes_major.iter().enumerate()
        {
            if genotypes_for_one_variant.len() != num_samples {
                return Err(anyhow!(
                    "Genotype count mismatch for variant index {}: expected {}, found {}.",
                    variant_idx, num_samples, genotypes_for_one_variant.len()
                ));
            }
            for sample_idx in 0..num_samples {
                final_matrix[[sample_idx, variant_idx]] =
                    genotypes_for_one_variant[sample_idx] as f64;
            }
        }
        Ok(final_matrix)
    }
}

mod pca_runner {
    use super::{anyhow, warn, info, Result, Error, Array2, Axis, EfficientPcaModel, cli};

    pub(crate) fn run_genomic_pca(
        genotype_matrix: Array2<f64>,
        cli_args: &cli::CliArgs,
    ) -> Result<(EfficientPcaModel, Array2<f64>, Vec<f64>), Error> {
        let mut pca_model = EfficientPcaModel::new();
        let mut k_actual = cli_args.components;

        if k_actual == 0 {
            return Err(anyhow!("Number of components (-k) must be > 0."));
        }

        let num_samples = genotype_matrix.nrows();
        let num_features = genotype_matrix.ncols();

        if num_samples < 2 {
            return Err(anyhow!("PCA requires at least 2 samples, found {}.", num_samples));
        }
        if num_features == 0 {
            return Err(anyhow!("PCA requires at least 1 variant, found 0."));
        }

        let max_possible_k = num_samples.min(num_features);
        if k_actual > max_possible_k {
            warn!(
                "Requested k={} exceeds max possible ({}), adjusting to {}.",
                k_actual, max_possible_k, max_possible_k
            );
            k_actual = max_possible_k;
        }
        if k_actual == 0 {
            return Err(anyhow!("Effective k is 0 for matrix ({}x{}).", num_samples, num_features));
        }

        let n_oversamples = 10;
        let seed = cli_args.rfit_seed;
        let tolerance_rfit = None;

        info!(
            "Running efficient_pca rfit: k={}, oversamples={}, seed={:?}, tolerance=None",
            k_actual, n_oversamples, seed
        );
        
        let matrix_for_rfit = genotype_matrix.clone(); // rfit can modify the matrix
        pca_model
            .rfit(matrix_for_rfit, k_actual, n_oversamples, seed, tolerance_rfit)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let computed_k_in_model = pca_model.rotation().map_or(0, |r| r.ncols());
        if computed_k_in_model == 0 && k_actual > 0 {
             warn!("PCA model resulted in 0 components despite requesting {}.", k_actual);
            let transformed_pcs_empty = Array2::<f64>::zeros((num_samples, 0));
            let pc_variances_empty = Vec::new();
            return Ok((pca_model, transformed_pcs_empty, pc_variances_empty));
        } else if computed_k_in_model < k_actual {
            info!("PCA model computed {} components (requested/capped at {}).", computed_k_in_model, k_actual);
        }

        let transformed_pcs = pca_model
            .transform(genotype_matrix) // transform expects original or similar data
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        let pc_variances: Vec<f64> = transformed_pcs
            .axis_iter(Axis(1))
            .map(|pc_column| pc_column.var(0.0))
            .collect();

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
