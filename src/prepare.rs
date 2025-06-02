use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::error::Error;
use log::{info, debug, warn, error};
use statrs::distribution::{ChiSquared, ContinuousCDF};

// bed_reader imports
use bed_reader::{Bed, ReadOptions, BedErrorPlus};

// efficient_pca::eigensnp imports for types and traits used by MicroarrayDataPreparer
use efficient_pca::eigensnp::{
    PcaSnpId, QcSampleId, LdBlockSpecification, PcaReadyGenotypeAccessor, ThreadSafeStdError,
};

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
    pub max_snp_hwe_p_value_threshold: f64,
}

pub struct MicroarrayDataPreparer {
    config: MicroarrayDataPreparerConfig,
    // Cached initial BIM data for efficient access by parallel threads
    initial_bim_sids: Arc<Array1<String>>, // SNP Identifiers from BIM file
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
        
        let initial_bim_allele1_alleles = Arc::new(bed_for_metadata.allele_1()?.to_owned());
        let initial_bim_allele2_alleles = Arc::new(bed_for_metadata.allele_2()?.to_owned());
        let initial_bim_sids = Arc::new(bed_for_metadata.sid()?.to_owned()); // Load SNP IDs
        let initial_bim_chromosomes = Arc::new(bed_for_metadata.chromosome()?.to_owned()); // Chromosome identifiers from BIM
        let initial_bim_bp_positions = Arc::new(bed_for_metadata.bp_position()?.to_owned()); // Base-pair positions from BIM
        let initial_snp_count_from_bim = bed_for_metadata.sid_count()?;
        let initial_sample_count_from_fam = bed_for_metadata.iid_count()?;
        let initial_sample_ids_from_fam = Arc::new(bed_for_metadata.iid()?.to_owned());

        debug!("Initial metadata loaded: {} samples, {} SNPs.", initial_sample_count_from_fam, initial_snp_count_from_bim);

        Ok(Self { 
            config, 
            initial_bim_sids, 
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

        // MicroarrayGenotypeAccessor::new now returns a Result, so handle it.
        let accessor = MicroarrayGenotypeAccessor::new(
            self.config.bed_file_path.clone(),
            Arc::new(original_indices_of_qc_samples),
            Arc::new(original_indices_of_pca_snps),
            Arc::new(mean_allele_dosages_for_pca_snps),
            Arc::new(std_devs_allele_dosages_for_pca_snps),
            num_qc_samples,
            num_blocked_snps_for_pca,
        ).map_err(|e_accessor_init| -> ThreadSafeStdError {
            error!("Failed to initialize MicroarrayGenotypeAccessor: {}", e_accessor_init);
            Box::new(e_accessor_init) // Convert DataPrepError to ThreadSafeStdError
        })?;

        info!("Data preparation pipeline complete. Ready for EigenSNP. N_samples_qc={}, D_snps_blocked_for_pca={}", num_qc_samples, num_blocked_snps_for_pca);

        info!("Data preparation pipeline complete. Ready for EigenSNP. N_samples_qc={}, D_snps_blocked_for_pca={}", num_qc_samples, num_blocked_snps_for_pca);
        Ok((accessor, ld_block_specifications, num_qc_samples, num_blocked_snps_for_pca))
    }

    fn perform_sample_qc(&self) -> Result<(Vec<isize>, usize), ThreadSafeStdError> {
        info!("Phase 0.2: Performing sample QC using {} initial samples...", self.initial_sample_count_from_fam);
        let qc_sample_original_indices: Vec<isize> = if let Some(ref path) = self.config.sample_ids_to_keep_file_path {
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

    /// Performs SNP quality control (call rate, MAF, HWE) and calculates standardization parameters (mean, std dev).
    /// This version initializes one `Bed` reader instance per Rayon worker thread for efficiency.
    fn perform_snp_qc_and_calc_std_params(
        &self,
        original_indices_of_qc_samples: &[isize], // Original indices of samples that passed sample QC.
        num_qc_samples: usize,                   // Count of samples that passed sample QC.
    ) -> Result<(Vec<IntermediateSnpDetails>, usize), ThreadSafeStdError> {
        info!("Phase 0.3 & 0.4: SNP QC & Standardization Params for {} samples...", num_qc_samples);
        if num_qc_samples == 0 {
            debug!("No QC samples, skipping SNP QC.");
            return Ok((Vec::new(), 0));
        }

        // Clone data needed by closures to satisfy lifetime/capture requirements.
        // Arcs are cheap to clone as they only copy the pointer and increment ref count.
        let bed_file_path_for_init = self.config.bed_file_path.clone();
        let qc_sample_indices_arc = Arc::new(original_indices_of_qc_samples.to_vec());
        
        // Capture necessary configuration and initial BIM data for use in the parallel map closure.
        // These are already Arcs or clonable parts of config.
        let min_snp_call_rate_threshold_val = self.config.min_snp_call_rate_threshold;
        let min_snp_maf_threshold_val = self.config.min_snp_maf_threshold;
        let max_snp_hwe_p_value_threshold_val = self.config.max_snp_hwe_p_value_threshold;
        let initial_bim_chromosomes_arc = self.initial_bim_chromosomes.clone();
        let initial_bim_bp_positions_arc = self.initial_bim_bp_positions.clone();
        let initial_bim_allele1_alleles_arc = self.initial_bim_allele1_alleles.clone();
        let initial_bim_allele2_alleles_arc = self.initial_bim_allele2_alleles.clone();

        // Process SNPs in parallel. Each thread gets its own Bed reader.
        let final_qc_snps_options: Vec<Option<IntermediateSnpDetails>> = (0..self.initial_snp_count_from_bim)
            .into_par_iter()
            .map_init(
                || { // INIT CLOSURE: Runs ONCE PER RAYON WORKER THREAD
                    Bed::new(&bed_file_path_for_init)
                        .map_err(|e| {
                            // Log error here. The map closure will receive Err.
                            error!("SNP QC Thread Init: Failed to open BED '{}': {:?}", bed_file_path_for_init, e);
                            e 
                        })
                },
                |bed_init_result, original_m_idx| { // MAP CLOSURE: Runs for each SNP
                    match bed_init_result {
                        Ok(ref mut thread_local_bed) => { // Successfully initialized Bed for this thread
                            let snp_genotypes_n_x_1_result = ReadOptions::builder()
                                .iid_index(qc_sample_indices_arc.as_slice())
                                .sid_index(original_m_idx as isize)
                                .i8()
                                .count_a1()
                                .read(thread_local_bed);

                            match snp_genotypes_n_x_1_result {
                                Ok(snp_genotypes_n_x_1) => {
                                    let snp_genotype_column_view: ArrayView1<i8> = snp_genotypes_n_x_1.column(0);

                                    let num_non_missing_genotypes = snp_genotype_column_view.iter().filter(|&&g_val| g_val != -127i8).count();
                                    if num_non_missing_genotypes == 0 { return None; }
                                    
                                    let call_rate = num_non_missing_genotypes as f64 / num_qc_samples as f64;
                                    if call_rate < min_snp_call_rate_threshold_val { return None; }
                                    
                                    // Strict filter: no missing genotypes allowed for SNPs included in PCA, among QC'd samples.
                                    if num_non_missing_genotypes != num_qc_samples { return None; }

                                    let mut allele1_dosage_sum_f64: f64 = 0.0;
                                    let mut observed_hom_ref_count: f64 = 0.0;
                                    let mut observed_het_count: f64 = 0.0;
                                    let mut observed_hom_alt_count: f64 = 0.0;

                                    for &genotype_val_i8 in snp_genotype_column_view.iter() {
                                        let dosage_f64 = genotype_val_i8 as f64;
                                        allele1_dosage_sum_f64 += dosage_f64;
                                        match genotype_val_i8 {
                                            0 => observed_hom_ref_count += 1.0,
                                            1 => observed_het_count += 1.0,
                                            2 => observed_hom_alt_count += 1.0,
                                            _ => { /* Unreachable with current strict missingness filter */ }
                                        }
                                    }
                                    
                                    let total_alleles_observed_f64 = 2.0 * num_qc_samples as f64;
                                    let allele1_frequency = allele1_dosage_sum_f64 / total_alleles_observed_f64;
                                    let minor_allele_frequency = allele1_frequency.min(1.0 - allele1_frequency);
                                    
                                    // Filter by MAF. Also filter if monomorphic (freq effectively 0 or 1).
                                    if minor_allele_frequency < min_snp_maf_threshold_val || 
                                       allele1_frequency.abs() < 1e-9 || 
                                       (1.0 - allele1_frequency).abs() < 1e-9 {
                                        return None;
                                    }

                                    // HWE Test if threshold is set to be restrictive (not 1.0)
                                    if max_snp_hwe_p_value_threshold_val < 1.0 {
                                        let hwe_p_val = MicroarrayDataPreparer::calculate_hwe_chi_squared_p_value(
                                            observed_hom_ref_count, observed_het_count, observed_hom_alt_count, num_qc_samples as f64
                                        );
                                        if hwe_p_val <= max_snp_hwe_p_value_threshold_val { return None; }
                                    }
                                    
                                    let mean_allele1_dosage_f32 = (allele1_dosage_sum_f64 / num_qc_samples as f64) as f32;
                                    let sum_sq_diff_f64: f64 = snp_genotype_column_view.iter()
                                        .map(|&g_val| (g_val as f64 - mean_allele1_dosage_f32 as f64).powi(2))
                                        .sum();
                                    
                                    // Sample variance (N-1 denominator if N > 1)
                                    let variance_denominator = (num_qc_samples.saturating_sub(1) as f64).max(1.0);
                                    let variance_allele1_dosage = sum_sq_diff_f64 / variance_denominator;
                                    
                                    // Filter if variance is effectively zero (prevents division by zero later)
                                    if variance_allele1_dosage <= 1e-9 { return None; }
                                    let std_dev_allele1_dosage_f32 = (variance_allele1_dosage.sqrt()) as f32;
                                    
                                    // If all QC checks pass, construct and return the details.
                                    Some(IntermediateSnpDetails {
                                        original_m_idx,
                                        chromosome: initial_bim_chromosomes_arc[original_m_idx].clone(),
                                        bp_position: initial_bim_bp_positions_arc[original_m_idx],
                                        allele1: initial_bim_allele1_alleles_arc[original_m_idx].clone(),
                                        allele2: initial_bim_allele2_alleles_arc[original_m_idx].clone(),
                                        mean_allele1_dosage: Some(mean_allele1_dosage_f32),
                                        std_dev_allele1_dosage: Some(std_dev_allele1_dosage_f32),
                                    })
                                }
                                Err(e) => {
                                    warn!("SNP QC: Read failed for original SNP idx {}: {:?}", original_m_idx, e);
                                    None
                                }
                            }
                        }
                        Err(_e_bed_init) => {
                            // Bed initialization failed for this thread, error already logged by init closure.
                            // This SNP cannot be processed by this particular thread.
                            None
                        }
                    }
                }
            )
            .collect();

        // Filter out the Nones from SNPs that failed QC or couldn't be read
        let final_qc_snps_details: Vec<IntermediateSnpDetails> = final_qc_snps_options.into_iter().flatten().collect();

        let num_final_qc_snps = final_qc_snps_details.len();
        info!("SNP QC & Stats: {} / {} initial SNPs passed all filters.", num_final_qc_snps, self.initial_snp_count_from_bim);
        Ok((final_qc_snps_details, num_final_qc_snps))
    }
    
    fn map_snps_to_ld_blocks(
        &self,
        final_qc_snps_details_list: &[IntermediateSnpDetails], // Length D_final
    ) -> Result<(Vec<LdBlockSpecification>, Vec<usize>, Array1<f32>, Array1<f32>, usize), ThreadSafeStdError> {
        info!("Phase 0.6: Mapping {} final QC'd SNPs to LD blocks from '{}'...", final_qc_snps_details_list.len(), self.config.ld_block_file_path);
        let parsed_ld_blocks = self.parse_ld_block_file()?;

        let mut block_tag_to_original_m_indices: HashMap<String, Vec<usize>> = HashMap::new();
        let mut d_blocked_snp_original_m_indices_set: HashSet<usize> = HashSet::new();

        for snp_details in final_qc_snps_details_list {
            let normalized_snp_chromosome = Self::normalize_chromosome_name(&snp_details.chromosome);
            for (block_chr, block_start, block_end, block_tag) in &parsed_ld_blocks {
                // block_chr is already normalized from parse_ld_block_file
                if &normalized_snp_chromosome == block_chr &&
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
                    user_defined_block_tag: block_tag_str, 
                    pca_snp_ids_in_block: pca_snp_ids_for_block 
                })
            }
        })
        .collect();
    
        ld_block_specifications.sort_by(|a, b| a.user_defined_block_tag.cmp(&b.user_defined_block_tag));

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
            let chr_str_original = parts[0];
            let chr_str = Self::normalize_chromosome_name(chr_str_original);
            let start_pos = parts[1].parse::<i32>().map_err(|e| DataPrepError::from(format!("LD block line {}: Error parsing start pos '{}': {}", line_num + 1, parts[1], e)))?;
            let end_pos = parts[2].parse::<i32>().map_err(|e| DataPrepError::from(format!("LD block line {}: Error parsing end pos '{}': {}", line_num + 1, parts[2], e)))?;
            let block_id_str = parts[3].to_string();
            blocks.push((chr_str, start_pos, end_pos, block_id_str));
        }
        if blocks.is_empty() { warn!("No valid LD blocks parsed from file: {}. Make sure format is chr start end block_id (whitespace separated).", self.config.ld_block_file_path); }
        else { info!("Successfully parsed {} LD blocks from file.", blocks.len()); }
        Ok(blocks)
    }

    /// Normalizes chromosome names to a consistent format (removes "chr" prefix).
    fn normalize_chromosome_name(original_name: &str) -> String {
        let mut name = original_name.to_lowercase();
        if name.starts_with("chr") {
            name = name.trim_start_matches("chr").to_string();
        }
        name
    }
    
    /// Calculates the p-value for Hardy-Weinberg Equilibrium using a Chi-squared test.
    ///
    /// The Chi-squared test statistic is calculated with 1 degree of freedom.
    /// The input counts should correspond to a biallelic marker.
    ///
    /// # Arguments
    /// * `observed_homozygous_allele1_count`: Count of individuals homozygous for Allele 1 (e.g., genotype AA or A1A1).
    /// * `observed_heterozygous_count`: Count of heterozygous individuals (e.g., genotype Aa or A1A2).
    /// * `observed_homozygous_allele2_count`: Count of individuals homozygous for Allele 2 (e.g., genotype aa or A2A2).
    /// * `total_samples_with_genotypes`: Sum of the three observed genotype counts (`obs_hom_A1 + obs_het + obs_hom_A2`). Must be > 0.
    ///
    /// # Returns
    /// The HWE p-value as an `f64`.
    /// Returns `1.0` (non-significant) if:
    ///   - `total_samples_with_genotypes` is zero or negative.
    ///   - Allele frequencies cannot be robustly determined (e.g., total alleles is zero).
    ///   - The SNP is effectively monomorphic.
    ///   - A Chi-squared statistic results in NaN or the CDF calculation fails.
    /// Returns `0.0` if there's an infinite deviation from HWE (e.g., an expected count is zero while observed is not).
    ///
    /// # Notes on Chi-squared Test Applicability
    /// The Chi-squared approximation is generally considered reliable when all expected
    /// genotype counts are reasonably large.
    /// For scenarios with very small expected counts, Fisher's exact test might be more appropriate.
    fn calculate_hwe_chi_squared_p_value(
        observed_homozygous_allele1_count: f64,
        observed_heterozygous_count: f64,
        observed_homozygous_allele2_count: f64,
        total_samples_with_genotypes: f64,
    ) -> f64 {
        if total_samples_with_genotypes <= 1e-9 { // Check if effectively zero samples
            warn!("HWE Test: Total samples ({}) is effectively zero. Cannot compute HWE p-value. Returning 1.0.", total_samples_with_genotypes);
            return 1.0;
        }
    
        // Calculate allele counts from genotype counts
        let count_allele1 = 2.0 * observed_homozygous_allele1_count + observed_heterozygous_count;
        let count_allele2 = 2.0 * observed_homozygous_allele2_count + observed_heterozygous_count;
        let total_alleles_observed = count_allele1 + count_allele2;
    
        // If total_alleles_observed is effectively zero, it implies total_samples_with_genotypes was also zero.
        if total_alleles_observed <= 1e-9 {
            warn!("HWE Test: Total alleles observed ({}) is effectively zero. Cannot compute allele frequencies. Returning 1.0.", total_alleles_observed);
            return 1.0;
        }
    
        // Calculate allele frequencies
        let frequency_allele1 = count_allele1 / total_alleles_observed;
        let frequency_allele2 = count_allele2 / total_alleles_observed;
    
        // Check for monomorphic SNPs. If monomorphic, it's perfectly in HWE (p-value = 1.0).
        // Epsilon comparison for floating point precision.
        const FREQ_EPSILON: f64 = 1e-9;
        if frequency_allele1 < FREQ_EPSILON || frequency_allele2 < FREQ_EPSILON {
            // Effectively monomorphic if one allele has near-zero frequency.
            return 1.0;
        }
        // Sanity check: frequencies should sum to 1.
        if (frequency_allele1 + frequency_allele2 - 1.0).abs() > 1e-6 {
            warn!(
                "HWE Test: Allele frequencies p ({:.4}) and q ({:.4}) do not sum to 1.0. Counts: HomA1={:.0}, Het={:.0}, HomA2={:.0}. Check input counts.",
                frequency_allele1, frequency_allele2, 
                observed_homozygous_allele1_count, observed_heterozygous_count, observed_homozygous_allele2_count
            );
            return 1.0; // Cannot reliably compute HWE
        }
    
        // Calculate expected genotype counts under HWE
        let expected_homozygous_allele1 = frequency_allele1 * frequency_allele1 * total_samples_with_genotypes;
        let expected_heterozygous = 2.0 * frequency_allele1 * frequency_allele2 * total_samples_with_genotypes;
        let expected_homozygous_allele2 = frequency_allele2 * frequency_allele2 * total_samples_with_genotypes;
    
        // Calculate Chi-squared statistic: sum ( (Observed - Expected)^2 / Expected )
        let mut chi_squared_statistic: f64 = 0.0;
        const MIN_EXPECTED_FOR_DIVISION: f64 = 1e-9; // Threshold to prevent division by effective zero
    
        // Term for homozygous Allele 1
        if expected_homozygous_allele1 > MIN_EXPECTED_FOR_DIVISION {
            chi_squared_statistic += (observed_homozygous_allele1_count - expected_homozygous_allele1).powi(2)
                / expected_homozygous_allele1;
        } else if observed_homozygous_allele1_count > MIN_EXPECTED_FOR_DIVISION { // Expected is ~0, but observed is not
            chi_squared_statistic = f64::INFINITY; 
        } // If both observed and expected are ~0, term contribution is 0 (chi_squared_statistic remains unchanged).
    
        // Term for heterozygous
        if chi_squared_statistic.is_finite() { 
            if expected_heterozygous > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic += (observed_heterozygous_count - expected_heterozygous).powi(2)
                    / expected_heterozygous;
            } else if observed_heterozygous_count > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic = f64::INFINITY;
            }
        }
    
        // Term for homozygous Allele 2
        if chi_squared_statistic.is_finite() {
            if expected_homozygous_allele2 > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic += (observed_homozygous_allele2_count - expected_homozygous_allele2).powi(2)
                    / expected_homozygous_allele2;
            } else if observed_homozygous_allele2_count > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic = f64::INFINITY;
            }
        }
        
        if chi_squared_statistic.is_nan() {
            warn!("HWE Test: Chi-squared statistic is NaN. This can occur with extreme deviations or problematic inputs. Counts: HomA1={:.0}, Het={:.0}, HomA2={:.0}. Freqs: p={:.4}, q={:.4}. Exp: E_HomA1={:.2}, E_Het={:.2}, E_HomA2={:.2}. Returning p=1.0.",
                observed_homozygous_allele1_count, observed_heterozygous_count, observed_homozygous_allele2_count,
                frequency_allele1, frequency_allele2,
                expected_homozygous_allele1, expected_heterozygous, expected_homozygous_allele2);
            return 1.0; 
        }
    
        if chi_squared_statistic == f64::INFINITY {
            return 0.0; // Infinite deviation from HWE expectations implies p-value of 0.
        }
    
        // P-value from Chi-squared distribution with 1 degree of freedom
        // P-value = P(X^2 > chi_squared_statistic) = 1 - CDF(chi_squared_statistic)
        match ChiSquared::new(1.0) { // 1 degree of freedom for standard biallelic HWE test
            Ok(chi_sq_dist) => {
                let cdf_value = chi_sq_dist.cdf(chi_squared_statistic);
                // CDF can sometimes slightly exceed 1.0 due to floating point issues for large chi_squared_statistic
                // or return NaN if chi_squared_statistic is NaN (handled above).
                // p-value is well-behaved.
                if cdf_value.is_nan() {
                    warn!("HWE Test: CDF value is NaN for Chi-squared statistic {}. Returning p=1.0.", chi_squared_statistic);
                    1.0
                } else {
                    (1.0 - cdf_value).max(0.0) // p-value is not negative
                }
            }
            Err(e) => {
                // This error means ChiSquared::new(1.0) failed, which is highly unlikely for df=1.0.
                error!("HWE Test: Failed to create ChiSquared distribution (df=1.0): {}. Chi-sq stat was: {}. Returning p=1.0.", e, chi_squared_statistic);
                1.0 
            }      
        }
    }

    // --- Public Accessor Methods for MicroarrayDataPreparer ---

    /// Returns a shared reference to the initial SNP IDs from the BIM file.
    pub fn initial_bim_sids_arc(&self) -> &Arc<Array1<String>> {
        &self.initial_bim_sids
    }

    /// Returns a shared reference to the initial SNP chromosomes from the BIM file.
    pub fn initial_bim_chromosomes_arc(&self) -> &Arc<Array1<String>> {
        &self.initial_bim_chromosomes
    }

    /// Returns a shared reference to the initial SNP basepair positions from the BIM file.
    pub fn initial_bim_bp_positions_arc(&self) -> &Arc<Array1<i32>> {
        &self.initial_bim_bp_positions
    }

    /// Returns a shared reference to the initial sample IDs from the FAM file.
    pub fn initial_sample_ids_from_fam_arc(&self) -> &Arc<Array1<String>> {
        &self.initial_sample_ids_from_fam
    }
}

// --- Genotype Accessor Implementation ---
/// Accessor for genotype data from a BED file, designed to be used by EigenSNP.
/// It opens the BED file on each call to `get_standardized_snp_sample_block`
/// to make sure we have thread safety given the Bed reader's internal structure.
#[derive(Clone)]
pub struct MicroarrayGenotypeAccessor {
    /// Path to the BED file.
    bed_file_path: String,
    original_indices_of_qc_samples: Arc<Vec<isize>>,
    original_indices_of_pca_snps: Arc<Vec<usize>>, 
    mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    num_total_qc_samples: usize,
    num_total_pca_snps: usize, // D_blocked
}

impl MicroarrayGenotypeAccessor {
    /// Creates a new MicroarrayGenotypeAccessor.
    /// Stores the path to the BED file and other necessary metadata.
    pub fn new(
        bed_file_path: String, // Path to the BED file.
        original_indices_of_qc_samples: Arc<Vec<isize>>,
        original_indices_of_pca_snps: Arc<Vec<usize>>,
        mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
        std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
        num_total_qc_samples: usize,
        num_total_pca_snps: usize,
    ) -> Self { // Constructor now returns Self directly
        assert_eq!(original_indices_of_qc_samples.len(), num_total_qc_samples, "Accessor: Sample count mismatch");
        assert_eq!(original_indices_of_pca_snps.len(), num_total_pca_snps, "Accessor: D_blocked SNP original index count mismatch");
        assert_eq!(mean_allele1_dosages_for_pca_snps.len(), num_total_pca_snps, "Accessor: Mean dosage vector length mismatch");
        assert_eq!(std_devs_allele1_dosages_for_pca_snps.len(), num_total_pca_snps, "Accessor: StdDev dosage vector length mismatch");

        Self {
            bed_file_path, // Store the path
            original_indices_of_qc_samples,
            original_indices_of_pca_snps,
            mean_allele1_dosages_for_pca_snps,
            std_devs_allele1_dosages_for_pca_snps,
            num_total_qc_samples,
            num_total_pca_snps,
        }
    }

    // --- Public Accessor Methods for MicroarrayGenotypeAccessor ---

    /// Returns a shared reference to the vector of original (FAM) indices of QC-passed samples.
    /// These indices map to the `initial_sample_ids_from_fam` array in `MicroarrayDataPreparer`.
    pub fn original_indices_of_qc_samples(&self) -> &Arc<Vec<isize>> {
        &self.original_indices_of_qc_samples
    }

    /// Returns a shared reference to the vector of original (BIM) indices of PCA-ready SNPs.
    /// These indices map to the initial BIM metadata arrays (sids, chromosomes, positions)
    /// in `MicroarrayDataPreparer`.
    pub fn original_indices_of_pca_snps(&self) -> &Arc<Vec<usize>> {
        &self.original_indices_of_pca_snps
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

        // Acquire lock on the shared Bed instance.
        // The lock guard ensures exclusive access for the read operation.
        let mut bed_instance_guard = self.bed_reader_instance.lock()
            .map_err(|e_poison| {
                let err_msg = format!("Mutex for Bed reader was poisoned in get_standardized_block: {}", e_poison);
                error!("{}", err_msg); // Log the error
                DataPrepError::from(err_msg)
            })?;

        // Use the locked Bed instance (*bed_instance_guard) for reading.
        let raw_dosages_samples_by_snps_i8 = ReadOptions::builder()
            .iid_index(&bed_reader_sample_indices) 
            .sid_index(&bed_reader_snp_indices) 
            .i8().count_a1().read(&mut *bed_instance_guard) // Dereference guard to get &mut Bed
            .map_err(|e_bed_read| Box::new(DataPrepError::from(format!("Bed read failed in get_standardized_block: {}", e_bed_read))) as ThreadSafeStdError)?;
        // Mutex guard is automatically released when bed_instance_guard goes out of scope here.

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
                    if raw_dosage_val_i8 == -127 { // Missing genotype
                        // We NEVER impute with mean
                        // If there is a missing genotype for anyone, remove the entire row
                        output_std_snp_row[i_req_sample] = (0.0 - mean_dosage) / std_dev_dosage;  // fix later to remove site, no impute
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
