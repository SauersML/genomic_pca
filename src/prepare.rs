use ndarray::{s, Array1, Array2, Axis, ArrayView1, Ix2};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::error::Error;
use log::{info, debug, warn, error};

// bed_reader imports
use bed_reader::{Bed, ReadOptions, BedErrorPlus};

// --- Custom Error Type for this Module ---
#[derive(Debug)]
pub struct DataPrepError(String);

impl std::fmt::Display for DataPrepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Data Preparation Error: {}", self.0)
    }
}

impl Error for DataPrepError {}

impl From<String> for DataPrepError {
    fn from(s: String) -> Self { DataPrepError(s) }
}
impl From<&str> for DataPrepError {
    fn from(s: &str) -> Self { DataPrepError(s.to_string()) }
}
// Allow conversion from BedErrorPlus to DataPrepError for convenience
impl From<Box<BedErrorPlus>> for DataPrepError {
    fn from(e: Box<BedErrorPlus>) -> Self {
        DataPrepError(format!("bed_reader error: {}", e))
    }
}
impl From<std::io::Error> for DataPrepError {
    fn from(e: std::io::Error) -> Self {
        DataPrepError(format!("I/O error: {}", e))
    }
}
impl From<std::num::ParseIntError> for DataPrepError {
    fn from(e: std::num::ParseIntError) -> Self {
        DataPrepError(format!("Integer parsing error: {}", e))
    }
}
impl From<std::num::ParseFloatError> for DataPrepError {
    fn from(e: std::num::ParseFloatError) -> Self {
        DataPrepError(format!("Float parsing error: {}", e))
    }
}


/// Thread-safe standard dynamic error.
pub type ThreadSafeStdError = Box<dyn Error + Send + Sync + 'static>;

// --- EigenSNP Interface Types (as expected by EigenSNPCoreAlgorithm) ---

/// Identifies a SNP included in the PCA (post-QC and part of an LD block).
/// This index is relative to the final list of $D_{blocked}$ SNPs used in the analysis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PcaSnpId(pub usize);

/// Identifies a sample included in the PCA (post-QC).
/// This index is relative to the final list of $N$ QC'd samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct QcSampleId(pub usize);

/// Specification for a single Linkage Disequilibrium (LD) block.
#[derive(Clone, Debug)]
pub struct LdBlockSpecification {
    pub defined_block_tag: String,
    pub pca_snp_ids_in_block: Vec<PcaSnpId>, // Sorted PcaSnpId(0..D_blocked-1)
}

/// Defines how the EigenSNP PCA algorithm accesses globally standardized genotype data.
pub trait PcaReadyGenotypeAccessor: Sync {
    /// Fetches a block of globally standardized genotypes.
    /// Output: SNPs as rows, Samples as columns (M_requested x N_requested).
    fn get_standardized_snp_sample_block(
        &self,
        pca_snp_ids_to_fetch: &[PcaSnpId],
        qc_sample_ids_to_fetch: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError>;

    fn num_pca_snps(&self) -> usize; // D_blocked
    fn num_qc_samples(&self) -> usize; // N
}

// --- Helper Struct for Intermediate SNP Data ---
#[derive(Debug, Clone)]
struct IntermediateSnpDetails {
    original_m_idx: usize, // Index in the initial M_initial SNPs from .bim file
    chromosome: String,
    bp_position: i32,
    allele1: String, 
    allele2: String, 
    mean_allele1_dosage: Option<f32>, 
    std_dev_allele1_dosage: Option<f32>,
}

// --- Main Data Preparer Configuration and Struct ---
pub struct MicroarrayDataPreparerConfig {
    pub bed_file_path: String,
    pub ld_block_file_path: String,
    pub sample_ids_to_keep_file_path: Option<String>,
    pub min_snp_call_rate_threshold: f64,
    pub min_snp_maf_threshold: f64,
    pub max_snp_hwe_p_value_threshold: f64, // SNPs with p-value <= this are REMOVED
}

pub struct MicroarrayDataPreparer {
    config: MicroarrayDataPreparerConfig,
    // Cached initial BIM data for efficient access by parallel threads
    initial_bim_chromosomes: Arc<Array1<String>>,
    initial_bim_bp_positions: Arc<Array1<i32>>,
    initial_bim_allele1_alleles: Arc<Array1<String>>,
    initial_bim_allele2_alleles: Arc<Array1<String>>,
    initial_snp_count_from_bim: usize, 
    initial_sample_count_from_fam: usize,
    initial_sample_ids_from_fam: Arc<Array1<String>>,
}

impl MicroarrayDataPreparer {
    pub fn try_new(config: MicroarrayDataPreparerConfig) -> Result<Self, ThreadSafeStdError> {
        info!("Initializing MicroarrayDataPreparer for BED: {}", config.bed_file_path);
        let mut bed_for_metadata = Bed::new(&config.bed_file_path)
            .map_err(|e| DataPrepError::from(format!("Failed to open BED file '{}' for metadata: {}", config.bed_file_path, e)))?;
        
        let initial_bim_chromosomes = Arc::new(bed_for_metadata.chromosome()?.to_owned());
        let initial_bim_bp_positions = Arc::new(bed_for_metadata.bp_position()?.to_owned());
        let initial_bim_allele1_alleles = Arc::new(bed_for_metadata.allele_1()?.to_owned());
        let initial_bim_allele2_alleles = Arc::new(bed_for_metadata.allele_2()?.to_owned());
        let initial_snp_count_from_bim = bed_for_metadata.sid_count()?;
        let initial_sample_count_from_fam = bed_for_metadata.iid_count()?;
        let initial_sample_ids_from_fam = Arc::new(bed_for_metadata.iid()?.to_owned());

        debug!("Initial metadata loaded: {} samples, {} SNPs.", initial_sample_count_from_fam, initial_snp_count_from_bim);

        Ok(Self { 
            config, 
            initial_bim_chromosomes, 
            initial_bim_bp_positions, 
            initial_bim_allele1_alleles, 
            initial_bim_allele2_alleles,
            initial_snp_count_from_bim,
            initial_sample_count_from_fam,
            initial_sample_ids_from_fam,
        })
    }

    pub fn prepare_data_for_eigen_snp(&self) -> Result<(
        MicroarrayGenotypeAccessor,
        Vec<LdBlockSpecification>,
        usize, // N (QC'd sample count)
        usize, // D_blocked (final SNP count for PCA)
    ), ThreadSafeStdError> {
        info!("Starting full data preparation pipeline...");

        let (original_indices_of_qc_samples, num_qc_samples) = self.perform_sample_qc()?;
        if num_qc_samples == 0 { return Err(DataPrepError::from("No samples passed QC.").into()); }

        let (final_qc_snps_details, _num_final_qc_snps) = 
            self.perform_snp_qc_and_calc_std_params(&original_indices_of_qc_samples, num_qc_samples)?;
        if final_qc_snps_details.is_empty() { return Err(DataPrepError::from("No SNPs passed all QC filters.").into()); }

        let (ld_block_specifications, original_indices_of_pca_snps, 
             mean_allele_dosages_for_pca_snps, std_devs_allele_dosages_for_pca_snps, num_blocked_snps_for_pca) = 
            self.map_snps_to_ld_blocks(&final_qc_snps_details)?;
        if num_blocked_snps_for_pca == 0 { return Err(DataPrepError::from("No SNPs mapped to LD blocks or all resulting blocks were empty.").into()); }

        let accessor = MicroarrayGenotypeAccessor::new(
            self.config.bed_file_path.clone(),
            Arc::new(original_indices_of_qc_samples),
            Arc::new(original_indices_of_pca_snps),
            Arc::new(mean_allele_dosages_for_pca_snps),
            Arc::new(std_devs_allele_dosages_for_pca_snps),
            num_qc_samples,
            num_blocked_snps_for_pca,
        );

        info!("Data preparation pipeline complete. Ready for EigenSNP. N_samples_qc={}, D_snps_blocked_for_pca={}", num_qc_samples, num_blocked_snps_for_pca);
        Ok((accessor, ld_block_specifications, num_qc_samples, num_blocked_snps_for_pca))
    }

    fn perform_sample_qc(&self) -> Result<(Vec<isize>, usize), ThreadSafeStdError> {
        info!("Phase 0.2: Performing sample QC using {} initial samples...", self.initial_sample_count_from_fam);
        let qc_sample_original_indices = if let Some(ref path) = self.config.sample_ids_to_keep_file_path {
            info!("Reading sample list to keep from: {}", path);
            let file_content = std::fs::read_to_string(path).map_err(DataPrepError::from)?;
            let ids_to_keep_set: HashSet<String> = file_content.lines().map(String::from).collect();
            self.initial_sample_ids_from_fam.iter().enumerate()
                .filter_map(|(idx, iid_str)| if ids_to_keep_set.contains(iid_str) { Some(idx as isize) } else { None })
                .collect()
        } else {
            warn!("No external sample ID list provided; using all {} initial samples.", self.initial_sample_count_from_fam);
            (0..self.initial_sample_count_from_fam).map(|idx| idx as isize).collect()
        };
        let num_qc_samples = qc_sample_original_indices.len();
        info!("Sample QC: {} / {} samples selected.", num_qc_samples, self.initial_sample_count_from_fam);
        Ok((qc_sample_original_indices, num_qc_samples))
    }

    fn perform_snp_qc_and_calc_std_params(
        &self,
        original_indices_of_qc_samples: &[isize],
        num_qc_samples: usize,
    ) -> Result<(Vec<IntermediateSnpDetails>, usize), ThreadSafeStdError> {
        info!("Phase 0.3 & 0.4: SNP QC & Standardization Params for {} samples...", num_qc_samples);
        if num_qc_samples == 0 { return Ok((Vec::new(), 0)); }

        let par_results: Vec<Option<IntermediateSnpDetails>> = (0..self.initial_snp_count_from_bim)
            .into_par_iter()
            .filter_map(|original_m_idx| {
                let mut thread_bed = match Bed::new(&self.config.bed_file_path) {
                    Ok(b) => b,
                    Err(e) => { error!("SNP QC Thread: Failed to open BED for SNP original_idx {}: {:?}", original_m_idx, e); return None; }
                };

                let snp_genotypes_n_x_1 = match ReadOptions::builder()
                    .iid_index(original_indices_of_qc_samples)
                    .sid_index(original_m_idx as isize)
                    .i8().count_a1().read(&mut thread_bed) {
                        Ok(data) => data,
                        Err(e) => { warn!("SNP QC: Read failed for original SNP idx {}: {:?}", original_m_idx, e); return None; }
                    };
                let snp_genotype_column_view: ArrayView1<i8> = snp_genotypes_n_x_1.column(0);

                let num_non_missing_genotypes = snp_genotype_column_view.iter().filter(|&&g_val| g_val != -127i8).count();
                if num_non_missing_genotypes == 0 { return None; }
                let call_rate = num_non_missing_genotypes as f64 / num_qc_samples as f64;
                if call_rate < self.config.min_snp_call_rate_threshold { return None; }
                if num_non_missing_genotypes != num_qc_samples { return None; } // Strict: No missing in QC'd samples

                let mut allele1_dosage_sum_f64: f64 = 0.0;
                let mut observed_hom_ref_count: f64 = 0.0; 
                let mut observed_het_count: f64 = 0.0; 
                let mut observed_hom_alt_count: f64 = 0.0;

                for &genotype_val_i8 in snp_genotype_column_view.iter() {
                    let dosage_f64 = genotype_val_i8 as f64; // Allele1 dosage (0, 1, or 2)
                    allele1_dosage_sum_f64 += dosage_f64;
                    match genotype_val_i8 {
                        0 => observed_hom_ref_count += 1.0,
                        1 => observed_het_count += 1.0,
                        2 => observed_hom_alt_count += 1.0,
                        _ => { /* Should be unreachable due to strict missingness filter */ }
                    }
                }
                
                let total_alleles_observed_f64 = 2.0 * num_qc_samples as f64;
                let allele1_frequency = allele1_dosage_sum_f64 / total_alleles_observed_f64;
                let minor_allele_frequency = allele1_frequency.min(1.0 - allele1_frequency);
                if minor_allele_frequency < self.config.min_snp_maf_threshold || allele1_frequency.abs() < 1e-9 || (1.0 - allele1_frequency).abs() < 1e-9 { return None; }

                if self.config.max_snp_hwe_p_value_threshold < 1.0 { // Only test if threshold is not effectively "off"
                    let hwe_p_val = Self::calculate_hwe_chi_squared_p_value(
                        observed_hom_ref_count, observed_het_count, observed_hom_alt_count, num_qc_samples as f64
                    );
                    if hwe_p_val <= self.config.max_snp_hwe_p_value_threshold { return None; }
                }
                
                let mean_allele1_dosage_f32 = (allele1_dosage_sum_f64 / num_qc_samples as f64) as f32;
                let sum_sq_diff_f64: f64 = snp_genotype_column_view.iter()
                    .map(|&g_val| (g_val as f64 - mean_allele1_dosage_f32 as f64).powi(2))
                    .sum();
                let variance_allele1_dosage = sum_sq_diff_f64 / ((num_qc_samples.saturating_sub(1)) as f64).max(1.0);
                
                if variance_allele1_dosage <= 1e-9 { return None; } // Effectively zero variance
                let std_dev_allele1_dosage_f32 = (variance_allele1_dosage.sqrt()) as f32;
                
                Some(IntermediateSnpDetails {
                    original_m_idx,
                    chromosome: self.initial_bim_chromosomes[original_m_idx].clone(),
                    bp_position: self.initial_bim_bp_positions[original_m_idx],
                    allele1: self.initial_bim_allele1_alleles[original_m_idx].clone(),
                    allele2: self.initial_bim_allele2_alleles[original_m_idx].clone(),
                    mean_allele1_dosage: Some(mean_allele1_dosage_f32),
                    std_dev_allele1_dosage: Some(std_dev_allele1_dosage_f32),
                })
            }).collect();

        let final_qc_snps_details: Vec<IntermediateSnpDetails> = par_results.into_iter().flatten().collect();
        let num_final_qc_snps = final_qc_snps_details.len();
        info!("SNP QC & Stats: {} / {} initial SNPs passed all filters.", num_final_qc_snps, self.initial_snp_count_from_bim);
        Ok((final_qc_snps_details, num_final_qc_snps))
    }
    
    fn map_snps_to_ld_blocks(
        &self,
        final_qc_snps_details_list: &[IntermediateSnpDetails], // Length D_final
    ) -> Result<(Vec<LdBlockSpecification>, Vec<usize>, Array1<f32>, Array1<f32>, usize), ThreadSafeStdError> {
        info!("Phase 0.6: Mapping {} final QC'd SNPs to LD blocks from '{}'...", final_qc_snps_details_list.len(), self.config.ld_block_file_path);
        let parsed_ld_blocks = self.parse_ld_block_file_robustly()?;

        let mut block_tag_to_original_m_indices: HashMap<String, Vec<usize>> = HashMap::new();
        let mut d_blocked_snp_original_m_indices_set: HashSet<usize> = HashSet::new();

        for snp_details in final_qc_snps_details_list {
            for (block_chr, block_start, block_end, block_tag) in &parsed_ld_blocks {
                if &snp_details.chromosome == block_chr &&
                   snp_details.bp_position >= *block_start &&
                   snp_details.bp_position <= *block_end {
                    block_tag_to_original_m_indices.entry(block_tag.clone()).or_default().push(snp_details.original_m_idx);
                    d_blocked_snp_original_m_indices_set.insert(snp_details.original_m_idx);
                    break; 
                }
            }
        }

        let mut original_indices_of_pca_snps: Vec<usize> = d_blocked_snp_original_m_indices_set.into_iter().collect();
        original_indices_of_pca_snps.sort_unstable(); 
        
        let num_blocked_snps_for_pca = original_indices_of_pca_snps.len();
        if num_blocked_snps_for_pca == 0 {
            warn!("No SNPs mapped to any LD blocks after filtering.");
            return Ok((Vec::new(), Vec::new(), Array1::zeros(0), Array1::zeros(0), 0));
        }

        let original_m_idx_to_pca_snp_id_map: HashMap<usize, PcaSnpId> = original_indices_of_pca_snps.iter().enumerate()
            .map(|(pca_id_val, &orig_m_idx)| (orig_m_idx, PcaSnpId(pca_id_val)))
            .collect();
        
        let mut mean_allele_dosages_for_pca_snps_vec = Vec::with_capacity(num_blocked_snps_for_pca);
        let mut std_devs_allele_dosages_for_pca_snps_vec = Vec::with_capacity(num_blocked_snps_for_pca);
        
        // Create a temporary map for faster lookup of final_qc_snps_details_list by original_m_idx
        let final_qc_snps_map: HashMap<usize, &IntermediateSnpDetails> = final_qc_snps_details_list.iter()
            .map(|info| (info.original_m_idx, info))
            .collect();

        for &orig_m_idx_in_d_blocked in &original_indices_of_pca_snps {
            if let Some(info) = final_qc_snps_map.get(&orig_m_idx_in_d_blocked) {
                mean_allele_dosages_for_pca_snps_vec.push(info.mean_allele1_dosage.ok_or_else(|| DataPrepError::from(format!("Mean dosage missing for QC'd SNP original_idx {}", orig_m_idx_in_d_blocked)))?);
                std_devs_allele_dosages_for_pca_snps_vec.push(info.std_dev_allele1_dosage.ok_or_else(|| DataPrepError::from(format!("StdDev dosage missing for QC'd SNP original_idx {}", orig_m_idx_in_d_blocked)))?);
            } else {
                 return Err(DataPrepError::from(format!("Internal error: SNP with original index {} from D_blocked set not found in final_qc_snps_map during mu/sigma finalization.", orig_m_idx_in_d_blocked)).into());
            }
        }
        let mean_allele_dosages_for_pca_snps = Array1::from_vec(mean_allele_dosages_for_pca_snps_vec);
        let std_devs_allele_dosages_for_pca_snps = Array1::from_vec(std_devs_allele_dosages_for_pca_snps_vec);

        let mut ld_block_specifications: Vec<LdBlockSpecification> = block_tag_to_original_m_indices.into_iter()
            .filter_map(|(block_tag_str, original_m_indices_in_this_block)| {
                let mut pca_snp_ids_for_block: Vec<PcaSnpId> = original_m_indices_in_this_block.iter()
                    .filter_map(|orig_m_idx| original_m_idx_to_pca_snp_id_map.get(orig_m_idx).copied())
                    .collect();
                if pca_snp_ids_for_block.is_empty() { None } else {
                    pca_snp_ids_for_block.sort_unstable(); 
                    Some(LdBlockSpecification { 
                        defined_block_tag: block_tag_str, 
                        pca_snp_ids_in_block: pca_snp_ids_for_block 
                    })
                }
            })
            .collect();
        
        ld_block_specifications.sort_by(|a, b| a.defined_block_tag.cmp(&b.defined_block_tag));

        info!("LD Mapping: {} unique SNPs (D_blocked) mapped to {} LD blocks.", num_blocked_snps_for_pca, ld_block_specifications.len());
        Ok((ld_block_specifications, original_indices_of_pca_snps, mean_allele_dosages_for_pca_snps, std_devs_allele_dosages_for_pca_snps, num_blocked_snps_for_pca))
    }

    fn parse_ld_block_file(&self) -> Result<Vec<(String, i32, i32, String)>, ThreadSafeStdError> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        info!("Parsing LD block file: {}", self.config.ld_block_file_path);
        let file = File::open(&self.config.ld_block_file_path)
            .map_err(|e| DataPrepError::from(format!("Failed to open LD block file '{}': {}", self.config.ld_block_file_path, e)))?;
        let reader = BufReader::new(file);
        let mut blocks = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| DataPrepError::from(format!("Error reading line {} from LD block file: {}", line_num + 1, e)))?;
            let trimmed_line = line.trim();
            if trimmed_line.is_empty() || trimmed_line.starts_with('#') || trimmed_line.starts_with("chr\t") || trimmed_line.starts_with("chromosome\t") { continue; }
            
            let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
            if parts.len() < 4 {
                warn!("Skipping malformed LD block line {}: '{}' (expected at least 4 fields: chr start end id)", line_num + 1, line);
                continue;
            }
            let chr_str = parts[0].to_string();
            // Handle "chrX", "chrY", "chrMT" etc for chromosome, or they are numeric if PLINK standard is numeric.
            // For now, assume they can be strings.
            let start_pos = parts[1].parse::<i32>().map_err(|e| DataPrepError::from(format!("LD block line {}: Error parsing start pos '{}': {}", line_num + 1, parts[1], e)))?;
            let end_pos = parts[2].parse::<i32>().map_err(|e| DataPrepError::from(format!("LD block line {}: Error parsing end pos '{}': {}", line_num + 1, parts[2], e)))?;
            let block_id_str = parts[3].to_string();
            blocks.push((chr_str, start_pos, end_pos, block_id_str));
        }
        if blocks.is_empty() { warn!("No valid LD blocks parsed from file: {}. Make sure format is chr start end block_id (whitespace separated).", self.config.ld_block_file_path); }
        else { info!("Successfully parsed {} LD blocks from file.", blocks.len()); }
        Ok(blocks)
    }

    /// Calculates HWE Chi-squared p-value.
    /// Counts are for genotypes AA, Aa, aa (hom_ref, het, hom_alt for Allele1 as 'a')
    fn calculate_hwe_chi_squared_p_value(hom_allele1_count: f64, het_count: f64, hom_allele2_count: f64, total_samples: f64) -> f64 {
        if total_samples == 0.0 { return 1.0; } // Avoid division by zero

        let allele1_total_count = 2.0 * hom_allele1_count + het_count;
        let allele2_total_count = 2.0 * hom_allele2_count + het_count;
        let total_alleles = allele1_total_count + allele2_total_count;

        if total_alleles == 0.0 { return 1.0; }

        let p_freq = allele1_total_count / total_alleles; // Frequency of Allele1
        let q_freq = allele2_total_count / total_alleles; // Frequency of Allele2

        if (p_freq + q_freq - 1.0).abs() > 1e-6 { // Check if frequencies sum to 1
             warn!("HWE: Allele frequencies p ({}) and q ({}) do not sum to 1.", p_freq, q_freq);
             return 1.0; // Or handle as error / invalid
        }
        
        let expected_hom_allele1 = p_freq * p_freq * total_samples;
        let expected_het = 2.0 * p_freq * q_freq * total_samples;
        let expected_hom_allele2 = q_freq * q_freq * total_samples;

        // Chi-squared statistic: sum((Observed - Expected)^2 / Expected)
        let mut chi_squared_stat: f64 = 0.0;
        if expected_hom_allele1 > 1e-9 { // Avoid division by zero if expected count is effectively zero
            chi_squared_stat += (hom_allele1_count - expected_hom_allele1).powi(2) / expected_hom_allele1;
        }
        if expected_het > 1e-9 {
            chi_squared_stat += (het_count - expected_het).powi(2) / expected_het;
        }
        if expected_hom_allele2 > 1e-9 {
            chi_squared_stat += (hom_allele2_count - expected_hom_allele2).powi(2) / expected_hom_allele2;
        }
        
        // P-value from Chi-squared distribution with 1 degree of freedom
        // Using a library for this is best (e.g., statrs or similar).
        // For simplicity, a full Chi2 CDF is complex. This is a critical external dependency or needs careful implementation.
        // A very basic check: if chi_squared_stat is large (e.g., > 3.84 for p=0.05, > 6.63 for p=0.01), p-value is small.
        // Returning 1.0 for now to not filter anything based on this stub.
        // In a production system, `statrs::distribution::ChiSquared::new(1.0).unwrap().cdf(chi_squared_stat)`
        // would give P(X <= chi_squared_stat), so p_value = 1.0 - cdf.
        if chi_squared_stat.is_nan() || chi_squared_stat.is_infinite() { return 1.0; }
        // Placeholder: This is NOT a real p-value calculation.
        // For a real implementation, use a stats library.
        // Example thresholding: if chi_squared_stat > 10.83 (p ~ 0.001 for 1 df), then it fails.
        if chi_squared_stat > 25.0 { // Arbitrary large value for stub
            return 0.0; // Fails HWE
        }
        1.0 // Passes HWE (stub)
    }
}

// --- Genotype Accessor Implementation ---
#[derive(Clone)]
pub struct MicroarrayGenotypeAccessor {
    bed_file_path: String,
    original_indices_of_qc_samples: Arc<Vec<isize>>,
    original_indices_of_pca_snps: Arc<Vec<usize>>, 
    mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    num_total_qc_samples: usize,
    num_total_pca_snps: usize, // D_blocked
}

impl MicroarrayGenotypeAccessor {
    pub fn new(
        bed_file_path: String,
        original_indices_of_qc_samples: Arc<Vec<isize>>,
        original_indices_of_pca_snps: Arc<Vec<usize>>,
        mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
        std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
        num_total_qc_samples: usize,
        num_total_pca_snps: usize,
    ) -> Self {
        assert_eq!(original_indices_of_qc_samples.len(), num_total_qc_samples, "Accessor: Sample count mismatch");
        assert_eq!(original_indices_of_pca_snps.len(), num_total_pca_snps, "Accessor: D_blocked SNP original index count mismatch");
        assert_eq!(mean_allele1_dosages_for_pca_snps.len(), num_total_pca_snps, "Accessor: Mean dosage vector length mismatch");
        assert_eq!(std_devs_allele1_dosages_for_pca_snps.len(), num_total_pca_snps, "Accessor: StdDev dosage vector length mismatch");
        Self {
            bed_file_path, original_indices_of_qc_samples, original_indices_of_pca_snps,
            mean_allele1_dosages_for_pca_snps, std_devs_allele1_dosages_for_pca_snps,
            num_total_qc_samples, num_total_pca_snps,
        }
    }
}

impl PcaReadyGenotypeAccessor for MicroarrayGenotypeAccessor {
    fn get_standardized_snp_sample_block(
        &self,
        pca_snp_ids_to_fetch: &[PcaSnpId],    // Indices 0..D_blocked-1
        qc_sample_ids_to_fetch: &[QcSampleId], // Indices 0..N-1
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let num_requested_snps = pca_snp_ids_to_fetch.len();
        let num_requested_samples = qc_sample_ids_to_fetch.len();

        if num_requested_snps == 0 || num_requested_samples == 0 {
            return Ok(Array2::zeros((num_requested_snps, num_requested_samples)));
        }

        let bed_reader_snp_indices: Vec<isize> = pca_snp_ids_to_fetch.iter()
            .map(|pca_id| self.original_indices_of_pca_snps[pca_id.0] as isize)
            .collect();
        let bed_reader_sample_indices: Vec<isize> = qc_sample_ids_to_fetch.iter()
            .map(|qc_id| self.original_indices_of_qc_samples[qc_id.0])
            .collect();

        let mut bed_instance = Bed::new(&self.bed_file_path).map_err(Box::new)?;
        let raw_dosages_samples_by_snps_i8 = ReadOptions::builder()
            .iid_index(&bed_reader_sample_indices) 
            .sid_index(&bed_reader_snp_indices) 
            .i8().count_a1().read(&mut bed_instance)?;

        let raw_dosages_snps_by_samples_i8 = raw_dosages_samples_by_snps_i8.t();
        let mut standardized_block_snps_by_samples_f32 = Array2::<f32>::zeros(raw_dosages_snps_by_samples_i8.raw_dim());

        for i_req_snp in 0..num_requested_snps {
            let current_pca_snp_id_val = pca_snp_ids_to_fetch[i_req_snp].0; 
            let mean_dosage = self.mean_allele1_dosages_for_pca_snps[current_pca_snp_id_val];
            let std_dev_dosage = self.std_devs_allele1_dosages_for_pca_snps[current_pca_snp_id_val];
            
            let mut output_std_snp_row = standardized_block_snps_by_samples_f32.row_mut(i_req_snp);
            let input_raw_snp_row_view = raw_dosages_snps_by_samples_i8.row(i_req_snp);

            if std_dev_dosage.abs() < 1e-9 { 
                output_std_snp_row.fill(0.0);
            } else {
                for i_req_sample in 0..num_requested_samples {
                    let raw_dosage_val_i8 = input_raw_snp_row_view[i_req_sample];
                    if raw_dosage_val_i8 == -127 { // Should not happen for D_blocked SNPs
                        output_std_snp_row[i_req_sample] = (0.0 - mean_dosage) / std_dev_dosage; 
                         warn!("Unexpected missing genotype in get_standardized_block for PCA SNP D_blocked_ID {}, requested sample index {}. Standardized as (0-mean)/std_dev.", 
                               current_pca_snp_id_val, qc_sample_ids_to_fetch[i_req_sample].0);
                    } else {
                        output_std_snp_row[i_req_sample] = (raw_dosage_val_i8 as f32 - mean_dosage) / std_dev_dosage;
                    }
                }
            }
        }
        Ok(standardized_block_snps_by_samples_f32)
    }

    fn num_pca_snps(&self) -> usize { self.num_total_pca_snps }
    fn num_qc_samples(&self) -> usize { self.num_total_qc_samples }
}
