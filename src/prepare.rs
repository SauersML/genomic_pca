// ========================================================================================
//
//               THE PRE-COMPUTATION, RECONCILIATION & PREPARATION ENGINE
//
// ========================================================================================
//
// ### 1. Mission and Philosophy ###
//
// This module is the **sole authority on pre-computation**. Its mission is to be the
// application's "setup phase," transforming raw, un-validated user inputs (file
// paths) into a single, cohesive, and scientifically correct set of data structures
// ready for the high-performance pipeline.
//
// It serves as a strict "airlock" between the messy reality of file formats and
// the clean, logical world of the concurrent processing engine. `main.rs` remains a
// simple conductor, completely insulated from the complexity of parsing, validation,
// and data transformation.
//
// The core design principle is to produce a single, immutable "proof token"—the
// `PreparationResult` struct. The successful creation of this struct is a
// run-time guarantee that all data has been loaded, validated,
// scientifically reconciled, and structured for maximum performance.
//
// ### 2. Architectural Strategy (Data-Oriented) ###
//
// The implementation follows a strict, four-phase data-oriented strategy to maximize
// performance and minimize memory usage.
//
// 1.  **Parse and Flatten:** All weights from the score file are read into a single,
//     contiguous `Vec<f32>` (`flat_weights`). A map is created from SNP ID to its
//     data (start index and effect allele) in this flat buffer. This replaces
//     thousands of small allocations with one large one.
//
// 2.  **Build a Plan:** The `.bim` file is streamed. For each SNP, we look it up in
//     the map. If found, a lightweight, pointer-free `ReconciliationTask` struct
//     is created. This "plan" is a compact vector of all work to be done.
//
// 3.  **Sort the Plan:** The vector of `ReconciliationTask`s is sorted in-place based
//     on the physical row index of SNPs in the `.bed` file. This critical step
//     ensures the I/O engine can perform physically sequential reads.
//
// 4.  **Finalize in a Fused Pass:** A single loop iterates over the sorted plan. In
//     each iteration, it performs a "gather" (random-access read from `flat_weights`)
//     and a "scatter" (a structured, sequential write into the final `interleaved_weights`
//     buffer). This fused pass minimizes memory traffic and is highly cache-efficient.

use ahash::AHashMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::num::ParseFloatError;
use std::path::Path;

// ========================================================================================
//                                  PUBLIC API
// ========================================================================================

/// An instruction for the I/O engine on how to handle a SNP's dosage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reconciliation {
    /// Use the dosage as-is (effect allele is Allele 2).
    Identity,
    /// Flip the dosage: `2 - dosage` (effect allele is Allele 1).
    Flip,
}

/// A comprehensive error type for the preparation phase.
#[derive(Debug)]
pub enum PrepError {
    Io(io::Error),
    Parse(String),
    Header(String),
    InconsistentScores(String),
    ParseFloat(ParseFloatError),
}

/// The final, validated, and pipeline-ready data produced by this module.
#[derive(Debug)]
pub struct PreparationResult {
    /// The weight matrix, interleaved into a SNP-major layout for the compute kernel.
    pub interleaved_weights: Vec<f32>,
    /// The sorted list of physical row indices in the `.bed` file for `io.rs` to read.
    pub required_snp_indices: Vec<usize>,
    /// The parallel list of `Identity`/`Flip` instructions for `io.rs`.
    pub reconciliation_instructions: Vec<Reconciliation>,
    /// The total number of individuals in the genotype file.
    pub num_people: usize,
    /// The total number of SNPs in the `.bim` file, for validation.
    pub total_snps_in_bim: usize,
    /// The number of SNPs that were successfully reconciled between files.
    pub num_reconciled_snps: usize,
    /// The number of scores (i.e., weight columns) per SNP.
    pub num_scores: usize,
}

/// The single entry point for the preparation phase. It orchestrates file parsing,
/// scientific reconciliation, sorting, and data structuring.
pub fn prepare_for_computation(
    plink_prefix: &str,
    score_path: &str,
) -> Result<PreparationResult, PrepError> {
    // Formulate paths
    let fam_path = Path::new(&format!("{}.fam", plink_prefix));
    let bim_path = Path::new(&format!("{}.bim", plink_prefix));
    let score_path = Path::new(score_path);

    // --- PHASE 1: PARSE AND FLATTEN ---
    let (num_scores, flat_weights, mut score_map) = parse_scores_and_flatten(score_path)?;

    // --- PHASE 2: BUILD RECONCILIATION PLAN ---
    // **FIX:** `tasks` is now declared as mutable directly.
    let (mut tasks, total_snps_in_bim) = build_reconciliation_plan(&bim_path, &mut score_map)?;
    let num_reconciled_snps = tasks.len();
    if num_reconciled_snps == 0 {
        return Err(PrepError::Parse(
            "No overlapping SNPs found between the score file and the genotype data.".to_string(),
        ));
    }
    let num_people = parse_fam_file(fam_path)?;

    // --- PHASE 3: THE CRITICAL SORT ---
    // **FIX:** Sorts the mutable `tasks` vector in-place.
    tasks.sort_unstable_by_key(|task| task.bed_row_index);

    // --- PHASE 4: THE FUSED FINALIZATION PASS ---
    let (interleaved_weights, required_snp_indices, reconciliation_instructions) =
        finalize_from_sorted_plan(tasks, &flat_weights, num_scores);

    Ok(PreparationResult {
        interleaved_weights,
        required_snp_indices,
        reconciliation_instructions,
        num_people,
        total_snps_in_bim,
        num_reconciled_snps,
        num_scores,
    })
}

// ========================================================================================
//                           PRIVATE IMPLEMENTATION HELPERS
// ========================================================================================

struct ReconciliationTask {
    bed_row_index: usize,
    instruction: Reconciliation,
    weights_start_index: usize,
}

fn parse_fam_file(fam_path: &Path) -> Result<usize, PrepError> {
    let file = File::open(fam_path)?;
    Ok(BufReader::new(file).lines().count())
}

fn parse_scores_and_flatten(
    score_path: &Path,
) -> Result<(usize, Vec<f32>, AHashMap<String, (usize, u8)>), PrepError> {
    let file = File::open(score_path)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header)?;

    let header_parts: Vec<&str> = header.split_whitespace().collect();
    if header_parts.len() < 2 || !header_parts.contains(&"effect_allele") {
        return Err(PrepError::Header("Score file header is invalid. Must contain at least 'snp_id' and 'effect_allele' columns.".to_string()));
    }
    let num_scores = header_parts.len() - 2;

    let mut flat_weights = Vec::new();
    let mut score_map = AHashMap::new();

    for line_result in reader.lines() {
        let line = line_result?;
        let mut parts = line.split_whitespace();
        let snp_id = match parts.next() {
            Some(id) => id,
            None => continue,
        };
        let effect_allele_str = match parts.next() {
            Some(a) => a,
            None => continue,
        };
        if effect_allele_str.len() != 1 {
            continue;
        }
        let effect_allele_u8 = effect_allele_str.as_bytes()[0];
        let weights_start_index = flat_weights.len();

        // Manually iterate and push directly to the single flat_weights vector.
        let mut weights_count = 0;
        for part in &mut parts {
            weights_count += 1;
            flat_weights.push(part.parse()?);
        }

        // Validate that the correct number of weights were found for this SNP.
        if weights_count != num_scores {
            return Err(PrepError::InconsistentScores(format!(
                "SNP '{}' has {} weights, but header implies {}.",
                snp_id, weights_count, num_scores
            )));
        }

        score_map.insert(snp_id.to_string(), (weights_start_index, effect_allele_u8));
    }
    Ok((num_scores, flat_weights, score_map))
}

fn build_reconciliation_plan(
    bim_path: &Path,
    score_map: &mut AHashMap<String, (usize, u8)>,
) -> Result<(Vec<ReconciliationTask>, usize), PrepError> {
    let file = File::open(bim_path)?;
    let reader = BufReader::new(file);

    let mut tasks = Vec::new();
    let mut total_bim_snps = 0;

    for (line_number, line_result) in reader.lines().enumerate() {
        total_bim_snps += 1;
        let line = line_result?;
        let mut parts = line.split_whitespace();

        let _chromosome = parts.next();
        let snp_id = parts.next();
        let _cm_pos = parts.next();
        let _bp_pos = parts.next();
        let allele1_str = parts.next();
        let allele2_str = parts.next();

        if let (Some(id), Some(a1_str), Some(a2_str)) = (snp_id, allele1_str, allele2_str) {
            // **FIX:** Perform cheap check before expensive hash map lookup.
            if a1_str.len() == 1 && a2_str.len() == 1 {
                if let Some((weights_start_index, effect_allele_u8)) = score_map.remove(id) {
                    let bim_a1 = a1_str.as_bytes()[0];
                    let bim_a2 = a2_str.as_bytes()[0];

                    let instruction = if effect_allele_u8 == bim_a2 {
                        Reconciliation::Identity
                    } else if effect_allele_u8 == bim_a1 {
                        Reconciliation::Flip
                    } else {
                        eprintln!("Warning: Skipping SNP '{}' due to allele mismatch (effect: {}, bim: {}/{})", id, effect_allele_u8 as char, bim_a1 as char, bim_a2 as char);
                        continue;
                    };

                    tasks.push(ReconciliationTask {
                        bed_row_index: line_number,
                        instruction,
                        weights_start_index,
                    });
                }
            }
        } else {
            return Err(PrepError::Parse(format!("Malformed line #{} in .bim file: {}", line_number + 1, line)));
        }
    }
    Ok((tasks, total_bim_snps))
}

fn finalize_from_sorted_plan(
    tasks: Vec<ReconciliationTask>,
    flat_weights: &[f32],
    num_scores: usize,
) -> (Vec<f32>, Vec<usize>, Vec<Reconciliation>) {
    let num_reconciled_snps = tasks.len();

    let mut interleaved_weights = vec![0.0f32; num_reconciled_snps * num_scores];
    let mut required_snp_indices = Vec::with_capacity(num_reconciled_snps);
    let mut reconciliation_instructions = Vec::with_capacity(num_reconciled_snps);

    for (snp_i, task) in tasks.into_iter().enumerate() {
        required_snp_indices.push(task.bed_row_index);
        reconciliation_instructions.push(task.instruction);

        let source_slice = &flat_weights[task.weights_start_index..task.weights_start_index + num_scores];

        for (score_j, &weight) in source_slice.iter().enumerate() {
            let dest_idx = (snp_i * num_scores) + score_j;
            interleaved_weights[dest_idx] = weight;
        }
    }
    (interleaved_weights, required_snp_indices, reconciliation_instructions)
}

// ========================================================================================
//                                 ERROR HANDLING (unchanged)
// ========================================================================================

impl fmt::Display for PrepError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            PrepError::Io(e) => write!(f, "I/O Error: {}", e),
            PrepError::Parse(s) => write!(f, "Parse Error: {}", s),
            PrepError::Header(s) => write!(f, "Invalid Header: {}", s),
            PrepError::InconsistentScores(s) => write!(f, "Inconsistent Data: {}", s),
            PrepError::ParseFloat(e) => write!(f, "Float Parse Error: {}", e),
        }
    }
}

impl Error for PrepError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PrepError::Io(e) => Some(e),
            PrepError::ParseFloat(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for PrepError {
    fn from(err: io::Error) -> Self {
        PrepError::Io(err)
    }
}

impl From<ParseFloatError> for PrepError {
    fn from(err: ParseFloatError) -> Self {
        PrepError::ParseFloat(err)
    }
}
