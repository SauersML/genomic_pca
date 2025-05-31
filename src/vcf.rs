// vcf.rs

// --- External Crate Imports ---
use anyhow::{anyhow, Error, Result};
use clap::Parser;
use efficient_pca::PCA as EfficientPcaModel;
use env_logger;
use indicatif::{ProgressBar, ProgressStyle};
use log::{debug, error, info, warn};
use ndarray::{Array2};
use noodles_vcf::{Record as VcfRecord, Header as VcfHeader}; // Changed
use num_cpus;
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};


pub mod vcf_processing {
    use super::{anyhow, debug, warn, Result, Path, Arc, VcfHeader, VcfRecord}; // Added VcfRecord, ensured cli is not present
    // Removed local noodles_vcf import block that was here
    use noodles_vcf::record::{ // Added new imports
        AlternateBases,
        Samples,
        samples::{
            self, variant::record::samples::series::Value as GenotypeValue, Keys as NoodlesKeys,
            Series as VcfSeries,
        }
    };
    use noodles_vcf::variant::record::AlternateBases as _; // Use _ to import trait methods
    use noodles_vcf::variant::record::samples::Series as SeriesTrait; // For explicit trait method calls

    #[derive(Debug)]
    pub struct SamplesHeaderInfo {
        pub sample_names: Vec<String>,
        pub sample_count: usize,
    }

    impl SamplesHeaderInfo {
        pub fn from_header(header: &VcfHeader, filepath: &Path) -> Result<Self> {
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
    pub struct VariantGenotypeData {
        pub id: String,
        pub chromosome: String,
        pub position: u64,
        pub numerical_genotypes: Vec<u8>,
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

    pub fn process_single_vcf(
        vcf_path: &Path,
        canonical_samples_info: Arc<SamplesHeaderInfo>,
        cli_args: &crate::cli::CliArgs, // Changed to crate::cli::CliArgs
        first_vcf_path_for_error_msg: &Path,
    ) -> Result<Option<Vec<VariantGenotypeData>>> {
        debug!("Processing VCF: {}", vcf_path.display());
        let mut reader = noodles_vcf::io::reader::Builder::default().build_from_path(vcf_path)?; // Changed to noodles_vcf::reader
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
        
        let gt_key_str = "GT"; // Changed to GenotypeKey

        if !current_header.formats().contains_key(gt_key_str) {
            return Err(anyhow!(
                "GT key (FORMAT={}) not found in FORMAT header for VCF {}",
                gt_key_str, vcf_path.display()
            ));
        }

        let mut chromosome_variants_data = Vec::new();
        let mut record_buffer = VcfRecord::default(); // Changed to VcfRecord

        while reader.read_record(&mut record_buffer)? != 0 {
            let record = &record_buffer;

            let ref_bases_str = record.reference_bases();
            let alt_bases_obj = record.alternate_bases();

            if ref_bases_str.len() != 1
                || alt_bases_obj.len() != 1
            // Removed: || alt_bases_obj.as_ref().len() != 1 
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
            let mut current_variant_has_gt_issues = false; 
            
            let samples_obj = record.samples();

            match samples_obj.select(gt_key_str) {
                Some(gt_series_struct) => {
                    for (sample_idx, value_option_result) in <noodles_vcf::record::samples::Series as SeriesTrait>::iter(&gt_series_struct, &current_header).enumerate() {
                        if sample_idx >= canonical_samples_info.sample_count {
                            warn!("More GT values in series than expected samples for variant at {}:{}. VCF: {}. Truncating.",
                                record.reference_sequence_name(),
                                record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                vcf_path.display());
                            current_variant_has_gt_issues = true;
                            break;
                        }
                        match value_option_result {
                            Ok(Some(GenotypeValue::String(gt_string_cow_val))) => { // Changed to GenotypeValue
                                let gt_str_slice = gt_string_cow_val.as_ref();
                                if let Some(gt_val) = parse_gt_to_option_u8(gt_str_slice) {
                                    temp_genotypes_for_variant.push(gt_val);
                                } else {
                                    debug!("Variant at {}:{}: GT field (String type) for sample {} was unparsable ('{}'). Skipping variant.",
                                        record.reference_sequence_name(), record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                        sample_idx, gt_str_slice);
                                    current_variant_has_gt_issues = true;
                                    break;
                                }
                            }
                            Ok(Some(GenotypeValue::Genotype(boxed_gt))) => { // Changed to GenotypeValue
                                let genotype_data = &*boxed_gt;
                                let mut allele_dosage_sum: u8 = 0;
                                let mut alleles_processed_count = 0;
                                let mut current_sample_gt_failed = false;

                                for allele_info_result in genotype_data.iter() {
                                    if alleles_processed_count >= 2 { break; }
                                    match allele_info_result {
                                        Ok((Some(allele_idx), _)) => { // Phasing info ignored with _
                                            if allele_idx == 0 { /* Ref allele */ }
                                            else if allele_idx == 1 { allele_dosage_sum += 1; /* Alt allele */ }
                                            else { 
                                                debug!("Variant at {}:{}: Sample #{} GT has unexpected allele index {} (>1) for bi-allelic site. Invalidating.",
                                                    record.reference_sequence_name(), record.variant_start().map_or(0u64, |r_p| r_p.map_or(0u64, |p|p.get() as u64)),
                                                    sample_idx, allele_idx);
                                                current_sample_gt_failed = true; break;
                                            }
                                        }
                                        Ok((None, _)) => { /* Missing allele '.' */ // Phasing info ignored
                                            debug!("Variant at {}:{}: Sample #{} GT contains missing allele ('.'). Invalidating.",
                                                record.reference_sequence_name(), record.variant_start().map_or(0u64, |r_p| r_p.map_or(0u64, |p|p.get() as u64)),
                                                sample_idx);
                                            current_sample_gt_failed = true; break;
                                        }
                                        Err(e) => {
                                            warn!("Variant at {}:{}: Error iterating GT alleles for sample #{}: {}. Invalidating.",
                                                record.reference_sequence_name(), record.variant_start().map_or(0u64, |r_p| r_p.map_or(0u64, |p|p.get() as u64)),
                                                sample_idx, e);
                                            current_sample_gt_failed = true; break;
                                        }
                                    }
                                    alleles_processed_count += 1;
                                }

                                if current_sample_gt_failed || alleles_processed_count != 2 {
                                    current_variant_has_gt_issues = true; break;
                                } else {
                                    temp_genotypes_for_variant.push(allele_dosage_sum);
                                }
                            }
                            Ok(Some(other_type)) => {
                                debug!("Variant at {}:{}: GT field for sample {} is an unexpected VCF Value type (type: {:?}). Skipping variant.",
                                    record.reference_sequence_name(), record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    sample_idx, other_type);
                                current_variant_has_gt_issues = true;
                                break;
                            }
                            Ok(None) => { 
                                debug!("Variant at {}:{}: GT field for sample {} is missing (None value). Skipping variant.",
                                    record.reference_sequence_name(), record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    sample_idx);
                                current_variant_has_gt_issues = true;
                                break;
                            }
                            Err(e) => { 
                                warn!("Error parsing a genotype value for variant at {}:{} in VCF {}: {}. Skipping variant.",
                                    record.reference_sequence_name(), record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                                    vcf_path.display(), e);
                                current_variant_has_gt_issues = true;
                                break;
                            }
                        }
                    }
                }
                None => { 
                    debug!("Variant at {}:{}: GT series not found in VCF {}. Skipping variant.",
                        record.reference_sequence_name(),
                        record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                        vcf_path.display());
                    current_variant_has_gt_issues = true;
                }
            }

            if current_variant_has_gt_issues
                || temp_genotypes_for_variant.len() != canonical_samples_info.sample_count
            {
                if !current_variant_has_gt_issues {
                    debug!(
                        "Variant at {}:{}:{} in VCF {} dropped (missing/unparsable/incomplete GTs: {}/{} processed).",
                        record.reference_sequence_name(),
                        record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                        record.reference_bases(), 
                        vcf_path.display(),
                        temp_genotypes_for_variant.len(),
                        canonical_samples_info.sample_count
                    );
                }
                continue;
            }

            let allele_sum: u32 = temp_genotypes_for_variant.iter().map(|&g| g as u32).sum();
            let num_alleles_total: u32 = (canonical_samples_info.sample_count * 2) as u32;

            if num_alleles_total == 0 { 
                debug!("Variant at {}:{}: No alleles to calculate MAF (num_alleles_total is 0). Skipping.",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)));
                continue;
            }

            let alt_allele_freq = allele_sum as f64 / num_alleles_total as f64;
            let maf = alt_allele_freq.min(1.0 - alt_allele_freq);
            let maf_threshold = cli_args.maf;

            if maf < maf_threshold {
                debug!("Variant at {}:{}:{} (MAF={:.4}) below threshold ({:.4}). Skipping.",
                    record.reference_sequence_name(),
                    record.variant_start().map_or(0u64, |res_p| res_p.map_or(0u64, |p| p.get() as u64)),
                    record.reference_bases(),
                    maf, maf_threshold);
                continue;
            }
            
            let alt_allele_str = alt_bases_obj.as_ref().to_string(); // Removed .as_ref()
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

pub mod matrix_ops {
    use super::{anyhow, Result, Array2};
    use super::vcf_processing::VariantGenotypeData;

    pub fn aggregate_chromosome_data(
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

    pub fn build_matrix(
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
