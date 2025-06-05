use flume;
use log::{debug, error, info, warn};
use ndarray::{Array1, Array2};
use num_cpus;
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, PoisonError};
use thiserror::Error;
use std::simd::Simd; use std::simd::prelude::{SimdPartialEq, SimdFloat, SimdInt};
use std::simd::num::SimdUint;

// SIMD Lane constants
// For i8 to f32 conversions and operations (e.g., standardization)
const LANES_I8_F32_8: usize = 8;
// For f64 operations (e.g., sums and variance parts)
const LANES_F64_4: usize = 4;
// For i8 specific operations like counting
const LANES_I8_16: usize = 16;

// bed_reader imports
use bed_reader::{Bed, BedErrorPlus, ReadOptions};

// efficient_pca::eigensnp imports for types and traits used by MicroarrayDataPreparer
use efficient_pca::eigensnp::{
    LdBlockSpecification, PcaReadyGenotypeAccessor, PcaSnpId, QcSampleId, ThreadSafeStdError,
};

#[derive(Error, Debug)]
pub enum DataPrepError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("BED reader error: {source}")]
    Bed { #[from] source: Box<BedErrorPlus> },

    #[error("Integer parsing error: {0}")]
    ParseInt(#[from] std::num::ParseIntError),

    #[error("Float parsing error: {0}")]
    ParseFloat(#[from] std::num::ParseFloatError),

    #[error("UTF-8 conversion error: {0}")]
    FromUtf8(#[from] std::string::FromUtf8Error),

    #[error("Flume send error: {0}")]
    FlumeSend(String),

    #[error("Flume receive error: {0}")]
    FlumeRecv(String), // Note: This was targeted by a previous subtask for #[from]

    #[error("Flume select error: {0}")]
    FlumeSelect(String), // Note: This was targeted by a previous subtask for #[from]

    #[error("Mutex poisoned: {0}")]
    MutexPoisoned(String),

    #[error("Error: {context}")]
    Contextual {
        context: String,
        #[source]
        source: ThreadSafeStdError, // This is Box<dyn Error + Send + Sync + 'static>
                                     // Note: This was targeted by a previous subtask to change source type
    },

    #[error("{0}")]
    Message(String),
}

// Manual From impl for Flume SendError since it's generic
impl<T> From<flume::SendError<T>> for DataPrepError {
    fn from(e: flume::SendError<T>) -> Self {
        DataPrepError::FlumeSend(format!("Flume send error: {}", e))
    }
}

// Manual From impl for Flume SendTimeoutError
impl<T> From<flume::SendTimeoutError<T>> for DataPrepError {
    fn from(e: flume::SendTimeoutError<T>) -> Self {
        DataPrepError::FlumeSend(format!("Flume send timeout error: {}", e))
    }
}

// Manual From impl for Flume RecvError
impl From<flume::RecvError> for DataPrepError {
    fn from(e: flume::RecvError) -> Self {
        DataPrepError::FlumeRecv(format!("Flume receive error: {}", e))
    }
}

// Manual From impl for Flume RecvTimeoutError
impl From<flume::RecvTimeoutError> for DataPrepError {
    fn from(e: flume::RecvTimeoutError) -> Self {
        DataPrepError::FlumeRecv(format!("Flume receive timeout error: {}", e))
    }
}

// Manual From impl for Flume SelectError
impl From<flume::select::SelectError> for DataPrepError {
    fn from(e: flume::select::SelectError) -> Self {
        DataPrepError::FlumeSelect(format!("Flume select error: {}", e))
    }
}

// Manual From impl for Mutex PoisonError
impl<T> From<PoisonError<T>> for DataPrepError {
    fn from(e: PoisonError<T>) -> Self {
        DataPrepError::MutexPoisoned(format!("Mutex poisoned: {}", e))
    }
}

pub trait WrapErr<T, EOriginal>
where EOriginal: std::error::Error + Send + Sync + 'static
{
    fn wrap_err_with_context(self, context_fn: impl FnOnce() -> String) -> Result<T, ThreadSafeStdError>;
    fn wrap_err_with_str(self, context: &str) -> Result<T, ThreadSafeStdError>;
}

impl<T, EOriginal> WrapErr<T, EOriginal> for Result<T, EOriginal>
where
    EOriginal: std::error::Error + Send + Sync + 'static,
{
    fn wrap_err_with_context(self, context_fn: impl FnOnce() -> String) -> Result<T, ThreadSafeStdError> {
        self.map_err(|e_original| {
            Box::new(DataPrepError::Contextual {
                context: context_fn(),
                source: Box::new(e_original),
            }) as ThreadSafeStdError
        })
    }

    fn wrap_err_with_str(self, context: &str) -> Result<T, ThreadSafeStdError> {
        self.map_err(|e_original| {
            Box::new(DataPrepError::Contextual {
                context: context.to_string(),
                source: Box::new(e_original),
            }) as ThreadSafeStdError
        })
    }
}

#[derive(Debug, Clone)]
struct IntermediateSnpDetails {
    original_m_idx: usize, // Index in the initial M_initial SNPs from .bim file
    chromosome: String,
    bp_position: i32,
    // allele1: String, // Removed due to dead_code warning
    // allele2: String, // Removed due to dead_code warning
    mean_allele1_dosage: Option<f32>,
    std_dev_allele1_dosage: Option<f32>,
}

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
    initial_bim_sids: Arc<Array1<String>>,
    initial_bim_chromosomes: Arc<Array1<String>>,
    initial_bim_bp_positions: Arc<Array1<i32>>,
    initial_snp_count_from_bim: usize,
    initial_sample_count_from_fam: usize,
    initial_sample_ids_from_fam: Arc<Array1<String>>,
    io_service: Arc<io_service_infrastructure::IoService>,
}

mod io_service_infrastructure {
    use super::*;
    use flume;
    use log::{debug, error, info, warn};
    use ndarray::{Array2};
    use std::collections::HashMap;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    pub(crate) const DEFAULT_IO_OPERATION_TIMEOUT: Duration = Duration::from_secs(60);

    #[derive(Debug)]
    pub(crate) enum IoRequest {
        /// Request to fetch genotype data for a chunk of SNPs for QC purposes.
        GetSnpChunkForQc {
            /// Original BIM indices for the SNPs in the requested chunk.
            original_m_indices: Vec<usize>,
            /// Original FAM indices of the samples to include in the QC.
            qc_sample_indices: Arc<Vec<isize>>,
            /// Channel to send the response back.
            response_tx: flume::Sender<IoResponse>,
        },
        GetSnpBlockForEigen {
            original_m_indices_for_bed: Vec<isize>,
            original_sample_indices_for_bed: Arc<Vec<isize>>,
            response_tx: flume::Sender<IoResponse>,
        },
    }

    /// Represents responses sent from IO actors back to the requesting logic.
    /// Ensures responses are Send + Sync as required by flume if passed across threads.
    /// Ndarray Arrays are Send/Sync if their elements are; String is; Result is if Ok/Err are.
    #[derive(Debug)]
    pub(crate) enum IoResponse {
        /// Response containing raw genotype data for a chunk of SNPs requested for QC.
        RawSnpChunkForQc {
            /// Result containing an Array2<i8> of genotypes (samples x SNPs_in_chunk, C-order)
            /// or an error string.
            raw_genotypes_i8_chunk_result: Result<Array2<i8>, String>,
            /// The original BIM indices corresponding to the columns of the returned Array2.
            /// This is crucial for mapping results back to the correct SNPs.
            original_m_indices_in_chunk: Vec<usize>,
        },
        SnpBlockData {
            raw_i8_block_result: Result<Array2<i8>, String>,
        },
        ActorInitStatus {
            actor_id: usize,
            success: bool,
            error_msg: Option<String>,
        },
    }

    pub(crate) struct IoTaskMetrics {
        pub(crate) bytes_read: usize,
    }

    pub(crate) struct IoActorHandle {
        pub(crate) join_handle: std::thread::JoinHandle<()>,
        pub(crate) shutdown_tx: flume::Sender<()>, // To signal individual actor shutdown
    }

    pub(crate) struct IoService {
        pub(crate) bed_file_path: Arc<String>,
        pub(crate) request_tx: flume::Sender<IoRequest>,
        pub(crate) request_rx_shared_for_actors_and_controller_monitoring: flume::Receiver<IoRequest>,
        pub(crate) metrics_tx_for_actors_to_controller: flume::Sender<IoTaskMetrics>,
        pub(crate) metrics_rx_for_controller: flume::Receiver<IoTaskMetrics>,
        pub(crate) active_actors: Arc<Mutex<HashMap<usize /*actor_id*/, IoActorHandle>>>,
        pub(crate) next_actor_id: Arc<AtomicUsize>,
        pub(crate) current_target_actors: Arc<AtomicUsize>,
        pub(crate) absolute_max_actors: usize,
        pub(crate) service_shutdown_signal: Arc<AtomicBool>,
        pub(crate) controller_join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
    }

    /// Constants influencing how the IoService controller adjusts the number of active IO actors.
    /// Justifications are based on common heuristics; optimal values may be system-dependent.
    pub(crate) const MIN_OPERATIONAL_IO_ACTORS: usize = 1;
    pub(crate) const CONTROLLER_ADJUSTMENT_INTERVAL: Duration = Duration::from_millis(750);
    pub(crate) const CONTROLLER_THROUGHPUT_HISTORY_WINDOW_DURATION: Duration = Duration::from_secs(8); // e.g. ~10x adjustment interval
    pub(crate) const TARGET_QUEUE_LENGTH_PER_ACTOR: usize = 3;
    // pub(crate) const MAX_ACCEPTABLE_AVG_TASK_TIME_US: u64 = 150_000; // 150ms as a soft upper limit for "typical" tasks // Removed due to dead_code
    // pub(crate) const MIN_THROUGHPUT_IMPROVEMENT_RATIO_FOR_SCALING_UP: f64 = 0.05; // Removed due to dead_code
    // pub(crate) const MAX_THROUGHPUT_DROP_RATIO_FOR_SCALING_DOWN_REVERSAL: f64 = 0.1; // Removed due to dead_code
    pub(crate) const ACTOR_SCALING_STEP_SIZE: usize = 1;
    pub(crate) const CONTROLLER_SCALING_COOLDOWN_PERIOD: Duration = Duration::from_millis(2000);

    impl IoService {
        pub(crate) fn new(
            bed_file_path: Arc<String>,
            absolute_max_actors: usize,
        ) -> Result<Arc<Self>, ThreadSafeStdError> {
            info!(
                "IoService: Initializing with absolute_max_actors: {}",
                absolute_max_actors
            );

            let request_channel_capacity = (absolute_max_actors.saturating_mul(TARGET_QUEUE_LENGTH_PER_ACTOR).saturating_mul(3)).max(1);
            let metrics_channel_capacity = (absolute_max_actors.saturating_mul(20)).max(1);
            info!(
                "IoService: Request channel capacity: {}, Metrics channel capacity: {}",
                request_channel_capacity, metrics_channel_capacity
            );

            let (request_tx, request_rx_shared) = flume::bounded::<IoRequest>(request_channel_capacity);
            let (metrics_tx, metrics_rx) = flume::bounded::<IoTaskMetrics>(metrics_channel_capacity);

            // Set initial target actors to a more responsive number, e.g., number of logical CPUs,
            // but capped by absolute_max_actors and at least 1.
            // This helps in utilizing resources better from the start for chunked I/O.
            let desired_initial_actors = num_cpus::get(); // Get number of logical CPUs
            // MIN_OPERATIONAL_IO_ACTORS is 1, as defined in io_service_controller_loop.
            // Using 1 directly here for clarity as the const is not in scope.
            const MIN_OP_ACTORS_FOR_INIT: usize = 1;
            let initial_target_actors = desired_initial_actors.max(MIN_OP_ACTORS_FOR_INIT).min(absolute_max_actors);
            info!(
                "IoService: Initial target actors set to {}",
                initial_target_actors
            );

            let service_arc = Arc::new(Self {
                bed_file_path,
                request_tx,
                request_rx_shared_for_actors_and_controller_monitoring: request_rx_shared,
                metrics_tx_for_actors_to_controller: metrics_tx,
                metrics_rx_for_controller: metrics_rx,
                active_actors: Arc::new(Mutex::new(HashMap::new())),
                next_actor_id: Arc::new(AtomicUsize::new(0)),
                current_target_actors: Arc::new(AtomicUsize::new(initial_target_actors)),
                absolute_max_actors,
                service_shutdown_signal: Arc::new(AtomicBool::new(false)),
                controller_join_handle: Mutex::new(None),
            });

            let (init_status_tx, init_status_rx) =
                flume::bounded::<IoResponse>(initial_target_actors);
            let mut successfully_spawned_count = 0;

            for _i in 0..initial_target_actors {
                if service_arc.spawn_new_actor_internal(Some(init_status_tx.clone())) {
                    successfully_spawned_count += 1;
                } else {
                    warn!("IoService: Failed to spawn an initial actor. Expected {} but only {} attempted.", initial_target_actors, successfully_spawned_count);
                }
            }

            if successfully_spawned_count < initial_target_actors {
                warn!("IoService: Not all initial actors ({}) could be spawned. Only {} were launched. This might be due to hitting absolute_max_actors limit early if it's very low.", initial_target_actors, successfully_spawned_count);
                if successfully_spawned_count < MIN_OPERATIONAL_IO_ACTORS {
                    service_arc.shutdown_all_actors_and_controller_immediately();
                    return Err(Box::new(DataPrepError::Message(format!("IoService: Failed to spawn enough initial actors to meet minimum operational requirement (spawned {}, required min {}).", successfully_spawned_count, MIN_OPERATIONAL_IO_ACTORS))) as ThreadSafeStdError);
                }
            }

            for i in 0..successfully_spawned_count {
                match init_status_rx.recv_timeout(Duration::from_secs(10))
                    .wrap_err_with_context(|| format!("IoService: Timed out waiting for actor {} init to report status.", i))? {
                    IoResponse::ActorInitStatus {
                        actor_id,
                        success,
                        error_msg,
                    } => {
                        if !success {
                            let err_message = error_msg.unwrap_or_default();
                            error!(
                                "IoService: Initial actor {} failed to initialize: {:?}",
                                actor_id,
                                err_message
                            );
                            service_arc.shutdown_all_actors_and_controller_immediately();
                            return Err(Box::new(DataPrepError::Message(format!(
                                "IoService: Actor {} failed to initialize. Cause: {}",
                                actor_id, err_message
                            ))) as ThreadSafeStdError);
                        }
                        info!(
                            "IoService: Initial actor {} reported successful initialization.",
                            actor_id
                        );
                    }
                    _ => {
                        error!("IoService: Received unexpected message type on init channel from actor during startup sequence. Actor index {}.", i);
                        service_arc.shutdown_all_actors_and_controller_immediately();
                        return Err(Box::new(DataPrepError::Message(
                            "IoService: Unexpected message during actor init.".to_string(),
                        )) as ThreadSafeStdError);
                    }
                }
            }
            info!(
                "IoService: All {} initial actors successfully initialized.",
                successfully_spawned_count
            );

            let controller_service_arc_clone = Arc::clone(&service_arc);
            let controller_thread_builder =
                std::thread::Builder::new().name("io_service_controller".into());
            match controller_thread_builder
                .spawn(move || io_service_controller_thread(controller_service_arc_clone))
            {
                Ok(handle) => {
                    *service_arc.controller_join_handle.lock()
                        .map_err(|e| DataPrepError::MutexPoisoned(format!("IoService: Controller join handle mutex poisoned during spawn: {}", e)))? = Some(handle);
                    info!("IoService: Controller thread spawned successfully.");
                }
                Err(e) => {
                    error!("IoService: Failed to spawn controller thread: {}", e);
                    service_arc.shutdown_all_actors_and_controller_immediately();
                    return Err(e).wrap_err_with_str("Failed to spawn controller thread");
                }
            }

            Ok(service_arc)
        }

        fn spawn_new_actor_internal(
            &self,
            init_status_tx: Option<flume::Sender<IoResponse>>,
        ) -> bool {
            let mut active_actors_guard = self.active_actors.lock().expect("Mutex poisoned: active_actors lock in spawn_new_actor_internal");

            if active_actors_guard.len() >= self.absolute_max_actors {
                warn!(
                    "IoService: Cannot spawn new actor, absolute_max_actors ({}) reached.",
                    self.absolute_max_actors
                );
                return false;
            }

            let actor_id = self.next_actor_id.fetch_add(1, AtomicOrdering::SeqCst);
            let bed_path_clone = Arc::clone(&self.bed_file_path);
            let request_rx_clone = self
                .request_rx_shared_for_actors_and_controller_monitoring
                .clone();
            let metrics_tx_clone = self.metrics_tx_for_actors_to_controller.clone();
            let global_shutdown_clone = Arc::clone(&self.service_shutdown_signal);

            let (individual_shutdown_tx, individual_shutdown_rx) = flume::bounded::<()>(1);

            let thread_builder = std::thread::Builder::new().name(format!("io_actor_{}", actor_id));

            match thread_builder.spawn(move || {
                io_reader_actor_loop( // Note: Name targeted by prior subtask
                    actor_id,
                    bed_path_clone,
                    request_rx_clone,
                    metrics_tx_clone,
                    individual_shutdown_rx,
                    global_shutdown_clone,
                    init_status_tx,
                )
            }) {
                Ok(join_handle) => {
                    active_actors_guard.insert(
                        actor_id,
                        IoActorHandle {
                            join_handle,
                            shutdown_tx: individual_shutdown_tx,
                        },
                    );
                    info!(
                        "IoService: Actor {} spawned successfully. Total active: {}",
                        actor_id,
                        active_actors_guard.len()
                    );
                    true
                }
                Err(e) => {
                    error!("IoService: Failed to spawn actor {}: {}", actor_id, e);
                    false
                }
            }
        }

        fn shutdown_one_actor(&self) -> bool {
            let mut active_actors_guard = self.active_actors.lock().expect("Mutex poisoned: active_actors lock in shutdown_one_actor");
            if active_actors_guard.is_empty() {
                return false;
            }

            // Simple strategy: remove the actor with the smallest ID (often the oldest).
            if let Some(actor_id_to_remove) = active_actors_guard.keys().min().copied() {
                if let Some(handle) = active_actors_guard.remove(&actor_id_to_remove) {
                    info!("IoService: Shutting down actor {}...", actor_id_to_remove);
                    if let Err(e) = handle.shutdown_tx.send(()) {
                        warn!("IoService: Failed to send shutdown signal to actor {}: {}. It might have already exited.", actor_id_to_remove, e);
                    }
                    info!(
                        "IoService: Actor {} signaled for shutdown. Remaining active: {}",
                        actor_id_to_remove,
                        active_actors_guard.len()
                    );
                    return true;
                }
            }
            warn!("IoService: shutdown_one_actor failed to select an actor to shutdown, though map was not empty.");
            false
        }

        fn shutdown_all_actors_and_controller_immediately(&self) {
            self.service_shutdown_signal
                .store(true, AtomicOrdering::SeqCst);
            let mut active_actors_guard = self.active_actors.lock().expect("Mutex poisoned: active_actors lock in shutdown_all_actors_and_controller_immediately");
            for (id, handle) in active_actors_guard.drain() {
                if handle.shutdown_tx.send(()).is_ok() {
                    info!("IoService: Sent emergency shutdown to actor {}", id);
                }
            }
        }
    }

    fn io_reader_actor_loop( // Note: Name targeted by prior subtask
        actor_id: usize,
        bed_file_path: Arc<String>,
        request_rx: flume::Receiver<IoRequest>,
        metrics_tx: flume::Sender<IoTaskMetrics>,
        individual_shutdown_rx: flume::Receiver<()>,
        global_shutdown_signal: Arc<AtomicBool>,
        init_status_tx: Option<flume::Sender<IoResponse>>,
    ) {
        info!("IoActor [{}]: Starting...", actor_id);

        let mut bed_reader_instance = match Bed::new(bed_file_path.as_str()) {
            Ok(reader) => {
                if let Some(tx) = init_status_tx.as_ref() {
                    if tx
                        .send(IoResponse::ActorInitStatus {
                            actor_id,
                            success: true,
                            error_msg: None,
                        })
                        .is_err()
                    {
                        error!("IoActor [{}]: Failed to send successful init status. IoService might have given up.", actor_id);
                    }
                }
                info!(
                    "IoActor [{}]: Bed reader initialized successfully for '{}'.",
                    actor_id, bed_file_path
                );
                reader
            }
            Err(e) => {
                error!(
                    "IoActor [{}]: Failed to initialize Bed reader for '{}': {:?}",
                    actor_id, bed_file_path, e
                );
                if let Some(tx) = init_status_tx {
                    if tx
                        .send(IoResponse::ActorInitStatus {
                            actor_id,
                            success: false,
                            error_msg: Some(format!("{:?}", e)),
                        })
                        .is_err()
                    {
                        error!("IoActor [{}]: Failed to send error init status. IoService might have given up or channel closed.", actor_id);
                    }
                }
                return;
            }
        };

        /// Represents the different outcomes of a channel select operation within the actor loop.
        /// This helps structure the logic for handling multiplexed channel operations
        /// using the flume::select::Selector builder pattern. Each variant corresponds
        /// to a distinct event that the actor loop needs to handle.
        #[derive(Debug)]
        enum SelectOutcome {
            IndividualShutdown,
            RequestReceived(IoRequest),
            RequestChannelDisconnected,
            Timeout,
        }

        loop {
            if global_shutdown_signal.load(AtomicOrdering::SeqCst) {
                info!(
                    "IoActor [{}]: Global shutdown signal detected. Exiting.",
                    actor_id
                );
                break;
            }
            let selected_event: SelectOutcome;
            {
                let selector = flume::select::Selector::new()
                    .recv(&individual_shutdown_rx, |result| {
                        match result {
                            Ok(_) => SelectOutcome::IndividualShutdown,
                            Err(flume::RecvError::Disconnected) => {
                                debug!(
                                    "IoActor [{}]: Individual shutdown channel disconnected.",
                                    actor_id
                                );
                                SelectOutcome::IndividualShutdown
                            }
                        }
                    })
                    .recv(&request_rx, |result| {
                        match result {
                            Ok(request) => SelectOutcome::RequestReceived(request),
                            Err(flume::RecvError::Disconnected) => {
                                debug!(
                                    "IoActor [{}]: Main request channel disconnected.",
                                    actor_id
                                );
                                SelectOutcome::RequestChannelDisconnected
                            }
                        }
                    });

                match selector.wait_timeout(Duration::from_millis(200)) {
                    Ok(outcome_from_completed_operation) => {
                        selected_event = outcome_from_completed_operation;
                    }
                    Err(flume::select::SelectError::Timeout) => {
                        selected_event = SelectOutcome::Timeout;
                    }
                }
            }

            match selected_event {
                SelectOutcome::IndividualShutdown => {
                    info!("IoActor [{}]: Individual shutdown signal received or its channel disconnected. Exiting.", actor_id);
                    break;
                }
                SelectOutcome::RequestChannelDisconnected => {
                    info!(
                        "IoActor [{}]: Request channel disconnected. Assuming shutdown. Exiting.",
                        actor_id
                    );
                    break;
                }
                SelectOutcome::RequestReceived(request) => {
                    let mut bytes_read_for_task: usize = 0;

                    match request {
                        IoRequest::GetSnpChunkForQc {
                            original_m_indices, // This is Vec<usize>
                            qc_sample_indices,
                            response_tx,
                        } => {
                            let qc_sample_indices_slice: &[isize] = qc_sample_indices.as_slice();
                            // Convert original_m_indices (Vec<usize>) to Vec<isize> for bed_reader
                            let original_m_indices_isize: Vec<isize> = original_m_indices
                                .iter()
                                .map(|&idx| idx as isize)
                                .collect();

                            // Read an entire chunk of SNPs.
                            // Request C-order to get data as (samples, snps_in_chunk),
                            // which is convenient for column-wise (per-SNP) QC processing later.
                            // bed_reader's internal Rayon parallelism will be used here effectively.
                            let read_result = ReadOptions::builder()
                                .sid_index(original_m_indices_isize.as_slice()) // Pass the chunk of SIDs
                                .iid_index(qc_sample_indices_slice)
                                .i8() // Output as i8, suitable for dosage
                                .c()  // Request C-order: (samples x SNPs_in_chunk)
                                .count_a1() // Standard allele counting
                                .num_threads(0) // Allow bed-reader to use its internal default Rayon parallelism
                                .read(&mut bed_reader_instance);

                            let response = match read_result {
                                Ok(array_samples_x_snps_in_chunk) => {
                                    // array_samples_x_snps_in_chunk has dimensions (num_qc_samples, original_m_indices.len())
                                    // Estimate bytes read for the chunk. This is an approximation of packed BED size.
                                    bytes_read_for_task = (array_samples_x_snps_in_chunk.nrows() * array_samples_x_snps_in_chunk.ncols() + 3) / 4;
                                    IoResponse::RawSnpChunkForQc {
                                        raw_genotypes_i8_chunk_result: Ok(array_samples_x_snps_in_chunk),
                                        // Clone original_m_indices here to transfer ownership of the clone to the response,
                                        // while the original original_m_indices remains in scope for subsequent logging.
                                        original_m_indices_in_chunk: original_m_indices.clone(),
                                    }
                                }
                                Err(e) => {
                                    // These logging lines can safely borrow original_m_indices as it has not been moved yet.
                                    let num_snps_in_failed_chunk = original_m_indices.len();
                                    let first_snp_idx_in_failed_chunk = original_m_indices.first().map_or_else(|| "N/A".to_string(), |v| v.to_string());
                                    warn!(
                                        "IoActor [{}]: Bed read failed for GetSnpChunkForQc ({} SNPs, starting original_idx {}): {:?}",
                                        actor_id, num_snps_in_failed_chunk, first_snp_idx_in_failed_chunk, e
                                    );
                                    IoResponse::RawSnpChunkForQc {
                                        raw_genotypes_i8_chunk_result: Err(format!(
                                            "Bed read failed for GetSnpChunkForQc ({} SNPs, starting original_idx {}): {:?}",
                                            num_snps_in_failed_chunk, first_snp_idx_in_failed_chunk, e
                                        )),
                                        // Clone original_m_indices here as well for the error path.
                                        original_m_indices_in_chunk: original_m_indices.clone(),
                                    }
                                }
                            };
                            // The `response` now owns a clone of `original_m_indices`. The `original_m_indices`
                            // in this function's scope is still valid and can be borrowed below.
                            if response_tx.send(response).is_err()
                            {
                                // This borrow of original_m_indices is now safe.
                                let first_snp_idx_in_dropped_chunk = original_m_indices.first().map_or_else(|| "N/A".to_string(), |v| v.to_string());
                                debug!(
                                    "IoActor [{}]: Failed to send RawSnpChunkForQc response for chunk starting with original_idx {}. Receiver likely dropped.",
                                    actor_id, first_snp_idx_in_dropped_chunk
                                );
                            }
                        }
                        IoRequest::GetSnpBlockForEigen {
                            original_m_indices_for_bed,
                            original_sample_indices_for_bed,
                            response_tx,
                        } => {
                            let original_m_indices_slice: &[isize] =
                                original_m_indices_for_bed.as_slice();
                            let original_sample_indices_slice: &[isize] =
                                original_sample_indices_for_bed.as_slice();
                            let read_result = ReadOptions::builder()
                                .sid_index(original_m_indices_slice)
                                .iid_index(original_sample_indices_slice)
                                .i8()
                                .count_a1()
                                .read(&mut bed_reader_instance);

                            let raw_i8_block_result = match read_result {
                                Ok(array_samples_x_snps) => {
                                    // Approx bytes based on genotype dimensions: (num_samples * num_snps + 3) / 4 bytes per genotype (packed BED estimate, ceiling division).
                                    bytes_read_for_task = (array_samples_x_snps.len_of(ndarray::Axis(0)) * array_samples_x_snps.len_of(ndarray::Axis(1)) + 3) / 4;
                                    Ok(array_samples_x_snps.t().as_standard_layout().to_owned())
                                }
                                Err(e) => {
                                    warn!("IoActor [{}]: Bed read failed for GetSnpBlockForEigen: {:?}", actor_id, e);
                                    Err(format!("Bed read failed for GetSnpBlockForEigen: {:?}", e))
                                }
                            };
                            if response_tx
                                .send(IoResponse::SnpBlockData {
                                    raw_i8_block_result,
                                })
                                .is_err()
                            {
                                debug!("IoActor [{}]: Failed to send SnpBlockData response. Receiver likely dropped.", actor_id);
                            }
                        }
                    }
                    match metrics_tx.try_send(IoTaskMetrics {
                        bytes_read: bytes_read_for_task,
                    }) {
                        Ok(_) => {}
                        Err(flume::TrySendError::Full(_)) => {
                            warn!("IoActor [{}]: Metrics channel full. Discarding metric.", actor_id);
                        }
                        Err(flume::TrySendError::Disconnected(_)) => {
                            debug!("IoActor [{}]: Failed to send metrics. Controller might be down.", actor_id);
                        }
                    }
                }
                SelectOutcome::Timeout => {
                    if global_shutdown_signal.load(AtomicOrdering::SeqCst) {
                        info!("IoActor [{}]: Global shutdown signal detected during default check. Exiting.", actor_id);
                        break;
                    }
                }
            }
        }
        info!("IoActor [{}]: Exiting run loop.", actor_id);
    }

    fn io_service_controller_thread(service_arc: Arc<IoService>) { // Note: Name targeted by prior subtask
        info!(
            "IoController: Starting for service with bed file: {}",
            service_arc.bed_file_path
        );
        let mut throughput_history: VecDeque<(Instant, usize)> = VecDeque::with_capacity(100);
        let mut bytes_read_since_last_history_update: usize = 0;
        let mut last_throughput_update_time = Instant::now();
        let mut last_adjustment_time = Instant::now();
        let mut last_scaling_event_time = Instant::now();

        loop {
            if service_arc
                .service_shutdown_signal
                .load(AtomicOrdering::SeqCst)
            {
                info!("IoController: Shutdown signal detected. Exiting.");
                break;
            }

            while let Ok(metric) = service_arc.metrics_rx_for_controller.try_recv() {
                bytes_read_since_last_history_update += metric.bytes_read;
            }

            if last_throughput_update_time.elapsed() >= Duration::from_millis(100) {
                throughput_history
                    .push_back((Instant::now(), bytes_read_since_last_history_update));
                bytes_read_since_last_history_update = 0;
                last_throughput_update_time = Instant::now();

                while let Some((timestamp, _)) = throughput_history.front() {
                    if timestamp.elapsed() > CONTROLLER_THROUGHPUT_HISTORY_WINDOW_DURATION {
                        throughput_history.pop_front();
                    } else {
                        break;
                    }
                }
            }

            if last_adjustment_time.elapsed() >= CONTROLLER_ADJUSTMENT_INTERVAL
                && last_scaling_event_time.elapsed() >= CONTROLLER_SCALING_COOLDOWN_PERIOD
            {
                let current_live_actors = service_arc.active_actors.lock().expect("Mutex poisoned: active_actors lock in controller").len();
                let prev_target_actors = service_arc
                    .current_target_actors
                    .load(AtomicOrdering::Relaxed);
                let current_request_queue_len = service_arc
                    .request_rx_shared_for_actors_and_controller_monitoring
                    .len();

                let total_bytes_in_window: usize =
                    throughput_history.iter().map(|&(_, bytes)| bytes).sum();

                let window_duration_actual_secs = if throughput_history.len() >= 2 {
                    let first_timestamp = throughput_history.front().unwrap().0;
                    let last_timestamp = throughput_history.back().unwrap().0;
                    last_timestamp.duration_since(first_timestamp).as_secs_f64()
                } else {
                    0.0
                };

                let current_avg_throughput_bps = if window_duration_actual_secs > 1e-9 {
                    (total_bytes_in_window as f64 / window_duration_actual_secs) as usize
                } else {
                    0
                };

                debug!("IoController: Eval: LiveActors={}, TargetActors={}, QueueLen={}, ThroughputBps={}",
                       current_live_actors, prev_target_actors, current_request_queue_len, current_avg_throughput_bps);

                let mut new_target_actors = prev_target_actors;
                if current_request_queue_len
                    > TARGET_QUEUE_LENGTH_PER_ACTOR * current_live_actors.max(1)
                    && current_live_actors < service_arc.absolute_max_actors
                {
                    new_target_actors = (prev_target_actors + ACTOR_SCALING_STEP_SIZE)
                        .min(service_arc.absolute_max_actors);
                    info!(
                        "IoController: Queue length high ({}). Scaling UP to {} actors.",
                        current_request_queue_len, new_target_actors
                    );
                } else if current_request_queue_len
                    < (TARGET_QUEUE_LENGTH_PER_ACTOR / 2) * current_live_actors.max(1)
                    && current_live_actors > MIN_OPERATIONAL_IO_ACTORS
                {
                    new_target_actors = prev_target_actors
                        .saturating_sub(ACTOR_SCALING_STEP_SIZE)
                        .max(MIN_OPERATIONAL_IO_ACTORS);
                    info!(
                        "IoController: Queue length low ({}). Scaling DOWN to {} actors.",
                        current_request_queue_len, new_target_actors
                    );
                }

                if new_target_actors != prev_target_actors {
                    service_arc
                        .current_target_actors
                        .store(new_target_actors, AtomicOrdering::SeqCst);
                    info!(
                        "IoController: Adjusting target actors from {} to {}.",
                        prev_target_actors, new_target_actors
                    );
                    last_scaling_event_time = Instant::now();
                }

                if current_live_actors < new_target_actors {
                    if service_arc.spawn_new_actor_internal(None) {
                        info!(
                            "IoController: Spawned one new actor to meet target {}. Live: {}",
                            new_target_actors,
                            service_arc.active_actors.lock().expect("Mutex poisoned: active_actors lock in controller for spawn check").len()
                        );
                    }
                } else if current_live_actors > new_target_actors {
                    if service_arc.shutdown_one_actor() {
                        info!(
                            "IoController: Shutdown one actor to meet target {}. Live: {}",
                            new_target_actors,
                            service_arc.active_actors.lock().expect("Mutex poisoned: active_actors lock in controller for shutdown check").len()
                        );
                    }
                }
                last_adjustment_time = Instant::now();
            }

            std::thread::sleep(Duration::from_millis(100));
        }
        info!("IoController: Exiting run loop.");
    }

    impl Drop for IoService {
        fn drop(&mut self) {
            info!(
                "IoService: Shutting down (Drop invoked)... Bed file: {}",
                self.bed_file_path
            );
            self.service_shutdown_signal
                .store(true, AtomicOrdering::SeqCst);

            match self.controller_join_handle.lock() {
                Ok(mut guard) => {
                    if let Some(controller_handle_owned) = guard.take() {
                        info!("IoService: Waiting for controller thread to exit...");
                        match controller_handle_owned.join() {
                            Ok(_) => info!("IoService: Controller thread successfully joined."),
                            Err(e) => warn!("IoService: Controller thread panicked: {:?}", e),
                        }
                    }
                }
                Err(poison_err) => {
                    warn!("IoService: Controller join handle mutex was poisoned during drop: {}. Cannot join controller thread.", poison_err);
                }
            }

            match self.active_actors.lock() {
                Ok(mut active_actors_guard) => {
                    info!(
                        "IoService: Shutting down {} active actors...",
                        active_actors_guard.len()
                    );
                    let actor_ids: Vec<usize> = active_actors_guard.keys().copied().collect();

                    for actor_id in actor_ids {
                        if let Some(actor_handle) = active_actors_guard.remove(&actor_id) {
                            info!("IoService: Sending shutdown signal to actor {}...", actor_id);
                            if let Err(e) = actor_handle.shutdown_tx.send(()) {
                                warn!("IoService: Failed to send shutdown to actor {}: {}. May have already exited.", actor_id, e);
                            }
                            info!("IoService: Waiting for actor {} to join...", actor_id);
                            if let Err(e_join) = actor_handle.join_handle.join() {
                                warn!("IoService: Actor {} panicked: {:?}", actor_id, e_join);
                            } else {
                                info!("IoService: Actor {} successfully joined.", actor_id);
                            }
                        }
                    }
                    info!(
                        "IoService: Shutdown sequence complete. Active actors remaining (should be 0): {}",
                        active_actors_guard.len()
                    );
                }
                Err(poison_error) => {
                    warn!("IoService: Active actors mutex poisoned during drop. Some actors may not be cleaned up properly: {}.", poison_error);
                }
            }
        }
    }
}

impl MicroarrayDataPreparer {
    pub fn try_new(
        config: MicroarrayDataPreparerConfig,
        absolute_max_io_actors_from_main: usize,
    ) -> Result<Self, ThreadSafeStdError> {
        info!(
            "Initializing MicroarrayDataPreparer for BED: {}",
            config.bed_file_path
        );

        let initial_bim_sids: Arc<Array1<String>>;
        let initial_bim_chromosomes: Arc<Array1<String>>;
        let initial_bim_bp_positions: Arc<Array1<i32>>;
        let initial_snp_count_from_bim: usize;
        let initial_sample_count_from_fam: usize;
        let initial_sample_ids_from_fam: Arc<Array1<String>>;

        {
            let mut bed_for_metadata = Bed::new(&config.bed_file_path)
                .wrap_err_with_context(|| format!("Failed to open BED file '{}' for initial metadata", config.bed_file_path))?;

            initial_bim_sids = Arc::new(
                bed_for_metadata
                    .sid()
                    .wrap_err_with_str("Failed to read SIDs from BIM for initial metadata")?
                    .to_owned(),
            );
            initial_bim_chromosomes = Arc::new(
                bed_for_metadata
                    .chromosome()
                    .wrap_err_with_str("Failed to read chromosomes from BIM for initial metadata")?
                    .to_owned(),
            );
            initial_bim_bp_positions = Arc::new(
                bed_for_metadata
                    .bp_position()
                    .wrap_err_with_str("Failed to read bp_positions from BIM for initial metadata")?
                    .to_owned(),
            );
            initial_snp_count_from_bim = bed_for_metadata.sid_count()
                .wrap_err_with_str("Failed to read sid_count for initial metadata")?;
            initial_sample_count_from_fam = bed_for_metadata.iid_count()
                .wrap_err_with_str("Failed to read iid_count for initial metadata")?;
            initial_sample_ids_from_fam = Arc::new(
                bed_for_metadata
                    .iid()
                    .wrap_err_with_str("Failed to read IIDs from FAM for initial metadata")?
                    .to_owned(),
            );

            debug!("Initial metadata loaded: {} samples, {} SNPs. Bed reader for metadata is now being dropped.", initial_sample_count_from_fam, initial_snp_count_from_bim);
        }

        let bed_file_path_arc = Arc::new(config.bed_file_path.clone());
        let io_service = io_service_infrastructure::IoService::new(
            bed_file_path_arc,
            absolute_max_io_actors_from_main,
        )?;

        info!("IoService initialized successfully for MicroarrayDataPreparer.");

        Ok(Self {
            config,
            initial_bim_sids,
            initial_bim_chromosomes,
            initial_bim_bp_positions,
            initial_snp_count_from_bim,
            initial_sample_count_from_fam,
            initial_sample_ids_from_fam,
            io_service,
        })
    }

    pub fn prepare_data_for_eigen_snp(
        &self,
    ) -> Result<
        (
            MicroarrayGenotypeAccessor,
            Vec<LdBlockSpecification>,
            usize,
            usize,
        ),
        ThreadSafeStdError,
    > {
        info!("Starting full data preparation pipeline...");

        let (original_indices_of_qc_samples_vec, num_qc_samples) = self.perform_sample_qc()?;
        if num_qc_samples == 0 {
            return Err(Box::new(DataPrepError::Message("No samples passed QC.".to_string())) as ThreadSafeStdError);
        }

        let original_indices_of_qc_samples_arc = Arc::new(original_indices_of_qc_samples_vec);

        let (final_qc_snps_details, _num_final_qc_snps) = self.perform_snp_qc_and_calc_std_params(
            &original_indices_of_qc_samples_arc,
            num_qc_samples,
        )?;
        if final_qc_snps_details.is_empty() {
            return Err(Box::new(DataPrepError::Message("No SNPs passed all QC filters.".to_string())) as ThreadSafeStdError);
        }

        let (
            ld_block_specifications,
            original_pca_snp_indices,
            pca_snp_mean_dosages,
            pca_snp_std_dev_dosages,
            num_blocked_snps_for_pca,
        ) = self.map_snps_to_ld_blocks(&final_qc_snps_details)?;
        if num_blocked_snps_for_pca == 0 {
            return Err(Box::new(DataPrepError::Message(
                "No SNPs mapped to LD blocks or all resulting blocks were empty.".to_string(),
            )) as ThreadSafeStdError);
        }

        let original_indices_of_pca_snps_arc = Arc::new(original_pca_snp_indices);
        let mean_allele_dosages_for_pca_snps_arc = Arc::new(pca_snp_mean_dosages);
        let std_devs_allele_dosages_for_pca_snps_arc = Arc::new(pca_snp_std_dev_dosages);

        let accessor = MicroarrayGenotypeAccessor::new(
            self.io_service.request_tx.clone(),
            original_indices_of_qc_samples_arc.clone(),
            num_qc_samples,
            original_indices_of_pca_snps_arc,
            num_blocked_snps_for_pca,
            mean_allele_dosages_for_pca_snps_arc,
            std_devs_allele_dosages_for_pca_snps_arc,
        );
        info!("Data preparation pipeline complete. Ready for EigenSNP. N_samples_qc={}, D_snps_blocked_for_pca={}", num_qc_samples, num_blocked_snps_for_pca);
        Ok((
            accessor,
            ld_block_specifications,
            num_qc_samples,
            num_blocked_snps_for_pca,
        ))
    }

    fn perform_sample_qc(&self) -> Result<(Vec<isize>, usize), ThreadSafeStdError> {
        info!(
            "Performing sample QC using {} initial samples...",
            self.initial_sample_count_from_fam
        );
        let qc_sample_original_indices: Vec<isize> = if let Some(ref path) =
            self.config.sample_ids_to_keep_file_path
        {
            info!("Reading sample list to keep from: {}", path);
            let file_content = std::fs::read_to_string(path)
                .wrap_err_with_context(|| format!("Failed to read sample ID file '{}'", path))?;
            let ids_to_keep_set: HashSet<String> = file_content.lines().map(String::from).collect();
            self.initial_sample_ids_from_fam
                .iter()
                .enumerate()
                .filter_map(|(idx, iid_str)| {
                    if ids_to_keep_set.contains(iid_str) {
                        Some(idx as isize)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            warn!(
                "No external sample ID list provided; using all {} initial samples.",
                self.initial_sample_count_from_fam
            );
            (0..self.initial_sample_count_from_fam)
                .map(|idx| idx as isize)
                .collect()
        };
        let num_qc_samples = qc_sample_original_indices.len();
        info!(
            "Sample QC: {} / {} samples selected.",
            num_qc_samples, self.initial_sample_count_from_fam
        );
        Ok((qc_sample_original_indices, num_qc_samples))
    }

    /// Performs SNP quality control (call rate, MAF, HWE) and calculates standardization parameters (mean, std dev)
    /// using the IoService for batched BED file reading.
    fn perform_snp_qc_and_calc_std_params(
        &self,
        original_indices_of_qc_samples_arc: &Arc<Vec<isize>>,
        num_qc_samples: usize,
    ) -> Result<(Vec<IntermediateSnpDetails>, usize), ThreadSafeStdError> {
        info!("Starting SNP QC & Standardization Params calculation for {} QC'd samples using IoService.", num_qc_samples);

        // Strict 100% Call Rate Policy is enforced.
        if num_qc_samples == 0 {
            debug!("No QC samples provided; skipping SNP QC and returning empty results.");
            return Ok((Vec::new(), 0));
        }

        // Define the chunk size for I/O requests. This determines how many SNPs are read by bed_reader in a single call.
        // A value around 1000-5000 can be a good balance.
        const SNP_IO_CHUNK_SIZE: usize = 2000; // Tunable parameter

        let num_total_initial_snps = self.initial_snp_count_from_bim;
        // Calculate the number of I/O chunks based on the total SNPs and the chosen I/O chunk size.
        let num_io_chunks = (num_total_initial_snps + SNP_IO_CHUNK_SIZE - 1) / SNP_IO_CHUNK_SIZE;

        let mut all_final_qc_snps_details: Vec<IntermediateSnpDetails> =
            Vec::with_capacity(num_total_initial_snps / 2); // Pre-allocate with a rough estimate

        info!(
            "Processing {} initial SNPs in {} I/O chunks of (up to) {} SNPs each for QC.",
            num_total_initial_snps, num_io_chunks, SNP_IO_CHUNK_SIZE
        );

        // Create a vector of original SNP indices (0-based) to be chunked.
        let original_snp_indices_0_based: Vec<usize> = (0..num_total_initial_snps).collect();

        // Iterate over chunks of original SNP indices to create I/O requests.
        // Each iteration processes one I/O chunk: sends request, waits for response, then QCs SNPs in response.
        for (io_chunk_idx, original_m_indices_for_current_io_chunk) in
            original_snp_indices_0_based.chunks(SNP_IO_CHUNK_SIZE).enumerate()
        {
            let (response_tx, response_rx) = flume::bounded(1);
            let request = io_service_infrastructure::IoRequest::GetSnpChunkForQc {
                original_m_indices: original_m_indices_for_current_io_chunk.to_vec(),
                qc_sample_indices: Arc::clone(original_indices_of_qc_samples_arc),
                response_tx,
            };

            // Send the chunked request to the IoService.
            if let Err(e_send) = self.io_service.request_tx.send(request) {
                return Err(e_send).wrap_err_with_context(|| {
                    format!(
                        "Failed to send SNP QC request for I/O chunk {} ({} SNPs). SNP QC cannot proceed.",
                        io_chunk_idx,
                        original_m_indices_for_current_io_chunk.len()
                    )
                });
            }
            debug!(
                "I/O Chunk {}: Dispatched request for {} SNPs to IoService.",
                io_chunk_idx,
                original_m_indices_for_current_io_chunk.len()
            );

            // Receive and process the response for the current I/O chunk.
            match response_rx.recv_timeout(io_service_infrastructure::DEFAULT_IO_OPERATION_TIMEOUT)
            {
                Ok(io_service_infrastructure::IoResponse::RawSnpChunkForQc {
                    raw_genotypes_i8_chunk_result,
                    original_m_indices_in_chunk, // These are the original BIM indices for the columns
                }) => {
                    match raw_genotypes_i8_chunk_result {
                        Ok(genotypes_for_chunk_samples_x_snps) => {
                            // genotypes_for_chunk_samples_x_snps is Array2<i8> (samples x SNPs_in_this_io_chunk)
                            // Verify dimensions as a sanity check.
                            if genotypes_for_chunk_samples_x_snps.ncols() != original_m_indices_in_chunk.len() {
                                 warn!("SNP QC (I/O chunk {}): Received data for {} SNPs, but original_m_indices_in_chunk indicates {}. Mismatch, skipping this chunk.",
                                       io_chunk_idx, genotypes_for_chunk_samples_x_snps.ncols(), original_m_indices_in_chunk.len());
                                 continue; // Skip to the next I/O chunk
                            }
                            if num_qc_samples > 0 && genotypes_for_chunk_samples_x_snps.nrows() != num_qc_samples {
                                warn!("SNP QC (I/O chunk {}): Received data for {} samples, but expected {}. Mismatch, skipping this chunk.",
                                       io_chunk_idx, genotypes_for_chunk_samples_x_snps.nrows(), num_qc_samples);
                                 continue; // Skip to the next I/O chunk
                            }

                            let min_snp_call_rate_thresh = self.config.min_snp_call_rate_threshold;
                            let min_snp_maf_thresh = self.config.min_snp_maf_threshold;
                            let max_snp_hwe_p_thresh = self.config.max_snp_hwe_p_value_threshold;

                            // Parallelize QC over SNPs within this fetched chunk using Rayon.
                            let qc_results_for_this_io_chunk: Vec<IntermediateSnpDetails> =
                                genotypes_for_chunk_samples_x_snps
                                    .axis_iter(ndarray::Axis(1)) // Iterate over columns (each column is one SNP's data for all samples)
                                    .into_par_iter() // Parallelize SNP processing with Rayon
                                    .enumerate() // To get the column index within this specific chunk
                                    .filter_map(|(col_idx_in_io_chunk, snp_column_data_arrayview1)| {
                                        // snp_column_data_arrayview1 is an ArrayView1<i8> for the current SNP
                                        let original_m_idx = original_m_indices_in_chunk[col_idx_in_io_chunk];

                                        // --- Start of existing per-SNP QC logic (mostly unchanged, now operates on snp_column_data_arrayview1) ---
                                        if num_qc_samples > 0 && snp_column_data_arrayview1.len() != num_qc_samples {
                                            warn!("SNP QC (idx {}): Genotype array view length ({}) within chunk does not match num_qc_samples ({}). Skipping.",
                                                  original_m_idx, snp_column_data_arrayview1.len(), num_qc_samples);
                                            return None;
                                        }

                                        let valid_genotypes: Vec<i8> = snp_column_data_arrayview1
                                            .iter()
                                            .filter(|&&g| g != -127i8) // Filter out missing genotypes
                                            .copied()
                                            .collect();

                                        let num_valid_genotypes_for_snp = valid_genotypes.len();

                                        // Call Rate Check
                                        if num_qc_samples > 0 { // Avoid division by zero if num_qc_samples is 0
                                            let call_rate = num_valid_genotypes_for_snp as f64 / num_qc_samples as f64;
                                            if call_rate < min_snp_call_rate_thresh {
                                                debug!("SNP QC (idx {}): Failed call rate check. Call rate: {:.4}, Threshold: {:.4}. Valid genotypes: {}/{}. Skipping.",
                                                       original_m_idx, call_rate, min_snp_call_rate_thresh, num_valid_genotypes_for_snp, num_qc_samples);
                                                return None;
                                            }
                                        } else if num_valid_genotypes_for_snp > 0 {
                                            // This case (no QC samples but SNP has valid genotypes) indicates an inconsistency.
                                            warn!("SNP QC (idx {}): No QC samples (num_qc_samples is 0), but found {} valid genotypes. Skipping.", original_m_idx, num_valid_genotypes_for_snp);
                                            return None;
                                        }
                                        // If num_qc_samples is 0 and num_valid_genotypes_for_snp is 0, it implicitly passes or is irrelevant.

                                        // If no valid genotypes for this SNP among the QC'd samples, it cannot pass subsequent checks.
                                        if num_valid_genotypes_for_snp == 0 {
                                            debug!("SNP QC (idx {}): No valid genotypes found among QC'd samples. Skipping.", original_m_idx);
                                            return None;
                                        }

                                        let valid_genotypes_slice: &[i8] = &valid_genotypes;

                                        // B1. Sum of Dosages (allele1_dosage_sum_f64)
                                        let mut current_sum_f64 = 0.0_f64;
                                        if num_valid_genotypes_for_snp > 0 {
                                            let mut sum_vec_f64 = Simd::<f64, LANES_F64_4>::splat(0.0);
                                            let mut i = 0;
                                            while i + LANES_F64_4 <= num_valid_genotypes_for_snp {
                                                let mut f64_temp_arr = [0.0f64; LANES_F64_4];
                                                for lane_idx in 0..LANES_F64_4 {
                                                     f64_temp_arr[lane_idx] = valid_genotypes_slice[i + lane_idx] as f64;
                                                }
                                                let i8_chunk_as_f64 = Simd::<f64, LANES_F64_4>::from_array(f64_temp_arr);
                                                sum_vec_f64 += i8_chunk_as_f64;
                                                i += LANES_F64_4;
                                            }
                                            current_sum_f64 = sum_vec_f64.reduce_sum();
                                            for k in i..num_valid_genotypes_for_snp {
                                                current_sum_f64 += valid_genotypes_slice[k] as f64;
                                            }
                                        }
                                        let allele1_dosage_sum_f64 = current_sum_f64;

                                        // B2. Genotype Counts (obs_hom_ref, obs_het, obs_hom_alt)
                                        let mut current_obs_hom_ref: usize = 0;
                                        let mut current_obs_het: usize = 0;
                                        let mut current_obs_hom_alt: usize = 0;

                                        if num_valid_genotypes_for_snp > 0 {
                                            let simd_val_0_i8 = Simd::<i8, LANES_I8_16>::splat(0);
                                            let simd_val_1_i8 = Simd::<i8, LANES_I8_16>::splat(1);
                                            let simd_val_2_i8 = Simd::<i8, LANES_I8_16>::splat(2);
                                            // No Simd<usize, ...> vectors here

                                            let mut i = 0;
                                            // Main SIMD loop for counts
                                            while i + LANES_I8_16 <= num_valid_genotypes_for_snp {
                                                let i8_chunk = Simd::<i8, LANES_I8_16>::from_slice(&valid_genotypes_slice[i..i + LANES_I8_16]);

                                                let mask0 = i8_chunk.simd_eq(simd_val_0_i8);
                                                let mask1 = i8_chunk.simd_eq(simd_val_1_i8);
                                                let mask2 = i8_chunk.simd_eq(simd_val_2_i8);

                                                current_obs_hom_ref += mask0.select(Simd::splat(1_u8), Simd::splat(0_u8)).reduce_sum() as usize;
                                                current_obs_het     += mask1.select(Simd::splat(1_u8), Simd::splat(0_u8)).reduce_sum() as usize;
                                                current_obs_hom_alt   += mask2.select(Simd::splat(1_u8), Simd::splat(0_u8)).reduce_sum() as usize;

                                                i += LANES_I8_16;
                                            }
                                            // No horizontal sum of Simd<usize> vectors needed here.

                                            // Scalar remainder loop for counts
                                            for k in i..num_valid_genotypes_for_snp {
                                                match valid_genotypes_slice[k] {
                                                    0 => current_obs_hom_ref += 1,
                                                    1 => current_obs_het += 1,
                                                    2 => current_obs_hom_alt += 1,
                                                    _ => {}
                                                }
                                            }
                                        }
                                        let obs_hom_ref = current_obs_hom_ref;
                                        let obs_het = current_obs_het;
                                        let obs_hom_alt = current_obs_hom_alt;

                                        // MAF Check (calculated based on valid genotypes only)
                                        let total_alleles_obs_f64 = 2.0 * num_valid_genotypes_for_snp as f64;
                                        if total_alleles_obs_f64 < 1e-9 {
                                            debug!("SNP QC (idx {}): Total observed alleles from valid genotypes is effectively zero. Skipping MAF check.", original_m_idx);
                                            return None;
                                        }
                                        let allele1_freq = allele1_dosage_sum_f64 / total_alleles_obs_f64;
                                        let maf = allele1_freq.min(1.0 - allele1_freq);

                                        if maf < min_snp_maf_thresh || allele1_freq.abs() < 1e-9 || (1.0 - allele1_freq).abs() < 1e-9 {
                                            debug!("SNP QC (idx {}): Failed MAF check (MAF={:.4e}, threshold={:.4e} based on {} valid genotypes).", original_m_idx, maf, min_snp_maf_thresh, num_valid_genotypes_for_snp);
                                            return None;
                                        }

                                        if max_snp_hwe_p_thresh < 1.0 {
                                            let hwe_p_val = MicroarrayDataPreparer::calculate_hwe_chi_squared_p_value(obs_hom_ref, obs_het, obs_hom_alt);
                                            if hwe_p_val <= max_snp_hwe_p_thresh {
                                                debug!("SNP QC (idx {}): Failed HWE check (p-val={:.2e}, threshold={:.2e}).", original_m_idx, hwe_p_val, max_snp_hwe_p_thresh);
                                                return None;
                                            }
                                        }

                                        let mean_f32 = (allele1_dosage_sum_f64 / num_valid_genotypes_for_snp as f64) as f32;

                                        // C. Sum of squared differences
                                        let mean_val_f64 = mean_f32 as f64;
                                        let mut current_sum_sq_diff_f64 = 0.0_f64;

                                        if num_valid_genotypes_for_snp > 0 {
                                            let mut sum_sq_diff_vec_f64 = Simd::<f64, LANES_F64_4>::splat(0.0);
                                            let simd_mean_f64 = Simd::<f64, LANES_F64_4>::splat(mean_val_f64);
                                            let mut i = 0;

                                            while i + LANES_F64_4 <= num_valid_genotypes_for_snp {
                                                let mut f64_temp_arr = [0.0f64; LANES_F64_4];
                                                for lane_idx in 0..LANES_F64_4 {
                                                     f64_temp_arr[lane_idx] = valid_genotypes_slice[i + lane_idx] as f64;
                                                }
                                                let i8_chunk_as_f64 = Simd::<f64, LANES_F64_4>::from_array(f64_temp_arr);

                                                let diff = i8_chunk_as_f64 - simd_mean_f64;
                                                let diff_sq = diff * diff;
                                                sum_sq_diff_vec_f64 += diff_sq;
                                                i += LANES_F64_4;
                                            }
                                            current_sum_sq_diff_f64 = sum_sq_diff_vec_f64.reduce_sum();

                                            for k in i..num_valid_genotypes_for_snp {
                                                let diff_scalar = valid_genotypes_slice[k] as f64 - mean_val_f64;
                                                current_sum_sq_diff_f64 += diff_scalar * diff_scalar;
                                            }
                                        }
                                        let sum_sq_diff_f64 = current_sum_sq_diff_f64;

                                        let variance: f64;
                                        if num_valid_genotypes_for_snp >= 2 {
                                            variance = sum_sq_diff_f64 / (num_valid_genotypes_for_snp - 1) as f64;
                                        } else {
                                            // If 0 or 1 valid observation, variance is undefined or zero. Filter these out.
                                            variance = 0.0;
                                        }

                                        // Filter out SNPs with zero or extremely low variance among valid genotypes.
                                        if variance <= 1e-9 {
                                            debug!("SNP QC (idx {}): Failed due to near-zero variance ({:.2e} based on {} valid genotypes).", original_m_idx, variance, num_valid_genotypes_for_snp);
                                            return None;
                                        }
                                        let std_dev_f32 = (variance.sqrt()) as f32;
                                        // --- End of existing per-SNP QC logic ---

                                        // Fetch chromosome and bp_position using original_m_idx from the preparer's stored initial BIM data.
                                        let chromosome = self.initial_bim_chromosomes[original_m_idx].clone();
                                        let bp_pos = self.initial_bim_bp_positions[original_m_idx];

                                        Some(IntermediateSnpDetails {
                                            original_m_idx,
                                            chromosome,
                                            bp_position: bp_pos,
                                            mean_allele1_dosage: Some(mean_f32),
                                            std_dev_allele1_dosage: Some(std_dev_f32),
                                        })
                                    })
                                    .collect();
                            all_final_qc_snps_details.extend(qc_results_for_this_io_chunk);
                        }
                        Err(e_str) => {
                            warn!(
                                "SNP QC (I/O chunk {}): IoService actor reported error for SNP chunk: {}. Original indices in chunk: {:?}",
                                io_chunk_idx, e_str, original_m_indices_in_chunk
                            );
                        }
                    }
                }
                Ok(unexpected_response) => {
                    warn!(
                        "SNP QC (I/O chunk {}): Received unexpected IoResponse type: {:?}. Expected RawSnpChunkForQc.",
                        io_chunk_idx, unexpected_response
                    );
                }
                Err(e_recv) => {
                    warn!(
                        "SNP QC (I/O chunk {}): Failed to receive response from IoService actor (timeout or disconnect): {}",
                        io_chunk_idx, e_recv
                    );
                }
            }

            debug!(
                "I/O Chunk {}: Processed response. Total QC'd SNPs so far: {}.",
                io_chunk_idx,
                all_final_qc_snps_details.len()
            );
            // This logging is useful for tracking progress.
            info!(
                "SNP QC Progress: After I/O chunk {}/{}, total QC'd SNPs found: {}",
                io_chunk_idx + 1, // User-friendly 1-based indexing for progress
                num_io_chunks,
                all_final_qc_snps_details.len()
            );
        }

        let num_final_qc_snps = all_final_qc_snps_details.len();
        info!(
            "SNP QC & Stats calculation complete. {} / {} initial SNPs passed all filters.",
            num_final_qc_snps, num_total_initial_snps
        );
        Ok((all_final_qc_snps_details, num_final_qc_snps))
    }

    fn map_snps_to_ld_blocks(
        &self,
        final_qc_snps_details_list: &[IntermediateSnpDetails],
    ) -> Result<
        (
            Vec<LdBlockSpecification>,
            Vec<usize>,
            Array1<f32>,
            Array1<f32>,
            usize,
        ),
        ThreadSafeStdError,
    > {
        info!(
            "Mapping {} final QC'd SNPs to LD blocks from '{}'...",
            final_qc_snps_details_list.len(),
            self.config.ld_block_file_path
        );
        let parsed_ld_blocks = self.parse_ld_block_file()?;

        let mut block_tag_to_original_m_indices: HashMap<String, Vec<usize>> = HashMap::new();
        let mut d_blocked_snp_original_m_indices_set: HashSet<usize> = HashSet::new();

        for snp_details in final_qc_snps_details_list {
            let normalized_snp_chromosome =
                Self::normalize_chromosome_name(&snp_details.chromosome);
            for (block_chr, block_start, block_end, block_tag) in &parsed_ld_blocks {
                if &normalized_snp_chromosome == block_chr
                    && snp_details.bp_position >= *block_start
                    && snp_details.bp_position <= *block_end
                {
                    block_tag_to_original_m_indices
                        .entry(block_tag.clone())
                        .or_default()
                        .push(snp_details.original_m_idx);
                    d_blocked_snp_original_m_indices_set.insert(snp_details.original_m_idx);
                    break;
                }
            }
        }

        let mut original_indices_of_pca_snps: Vec<usize> =
            d_blocked_snp_original_m_indices_set.into_iter().collect();
        original_indices_of_pca_snps.sort_unstable();

        let num_blocked_snps_for_pca = original_indices_of_pca_snps.len();
        if num_blocked_snps_for_pca == 0 {
            warn!("No SNPs mapped to any LD blocks after filtering.");
            return Ok((
                Vec::new(),
                Vec::new(),
                Array1::zeros(0),
                Array1::zeros(0),
                0,
            ));
        }

        let original_m_idx_to_pca_snp_id_map: HashMap<usize, PcaSnpId> =
            original_indices_of_pca_snps
                .iter()
                .enumerate()
                .map(|(pca_id_val, &orig_m_idx)| (orig_m_idx, PcaSnpId(pca_id_val)))
                .collect();

        let mut mean_allele_dosages_for_pca_snps_vec = Vec::with_capacity(num_blocked_snps_for_pca);
        let mut std_devs_allele_dosages_for_pca_snps_vec =
            Vec::with_capacity(num_blocked_snps_for_pca);

        // Create a temporary map for faster lookup of final_qc_snps_details_list by original_m_idx
        let final_qc_snps_map: HashMap<usize, &IntermediateSnpDetails> = final_qc_snps_details_list
            .iter()
            .map(|info| (info.original_m_idx, info))
            .collect();

        for &orig_m_idx_in_d_blocked in &original_indices_of_pca_snps {
            if let Some(info) = final_qc_snps_map.get(&orig_m_idx_in_d_blocked) {
                mean_allele_dosages_for_pca_snps_vec.push(info.mean_allele1_dosage.ok_or_else(
                    || {
                        Box::new(DataPrepError::Message(format!(
                            "Mean dosage missing for QC'd SNP original_idx {}",
                            orig_m_idx_in_d_blocked
                        ))) as ThreadSafeStdError
                    },
                )?);
                std_devs_allele_dosages_for_pca_snps_vec.push(
                    info.std_dev_allele1_dosage.ok_or_else(|| {
                        Box::new(DataPrepError::Message(format!(
                            "StdDev dosage missing for QC'd SNP original_idx {}",
                            orig_m_idx_in_d_blocked
                        ))) as ThreadSafeStdError
                    })?,
                );
            } else {
                // This indicates a logic error if a SNP in original_indices_of_pca_snps isn't found in final_qc_snps_map.
                return Err(Box::new(DataPrepError::Message(format!("Internal error: SNP with original index {} from D_blocked set not found in final_qc_snps_map during mu/sigma finalization.", orig_m_idx_in_d_blocked))) as ThreadSafeStdError);
            }
        }
        let mean_allele_dosages_for_pca_snps =
            Array1::from_vec(mean_allele_dosages_for_pca_snps_vec);
        let std_devs_allele_dosages_for_pca_snps =
            Array1::from_vec(std_devs_allele_dosages_for_pca_snps_vec);

        let mut ld_block_specifications: Vec<LdBlockSpecification> =
            block_tag_to_original_m_indices
                .into_iter()
                .filter_map(|(block_tag_str, original_m_indices_in_this_block)| {
                    let mut pca_snp_ids_for_block: Vec<PcaSnpId> = original_m_indices_in_this_block
                        .iter()
                        .filter_map(|orig_m_idx| {
                            original_m_idx_to_pca_snp_id_map.get(orig_m_idx).copied()
                        })
                        .collect();
                    if pca_snp_ids_for_block.is_empty() {
                        None
                    } else {
                        pca_snp_ids_for_block.sort_unstable();
                        Some(LdBlockSpecification {
                            user_defined_block_tag: block_tag_str,
                            pca_snp_ids_in_block: pca_snp_ids_for_block,
                        })
                    }
                })
                .collect();

        ld_block_specifications
            .sort_by(|a, b| a.user_defined_block_tag.cmp(&b.user_defined_block_tag));

        info!(
            "LD Mapping: {} unique SNPs (D_blocked) mapped to {} LD blocks.",
            num_blocked_snps_for_pca,
            ld_block_specifications.len()
        );
        Ok((
            ld_block_specifications,
            original_indices_of_pca_snps,
            mean_allele_dosages_for_pca_snps,
            std_devs_allele_dosages_for_pca_snps,
            num_blocked_snps_for_pca,
        ))
    }

    fn parse_ld_block_file(&self) -> Result<Vec<(String, i32, i32, String)>, ThreadSafeStdError> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        info!("Parsing LD block file: {}", self.config.ld_block_file_path);
        let file = File::open(&self.config.ld_block_file_path)
            .wrap_err_with_context(|| format!("Failed to open LD block file '{}'", self.config.ld_block_file_path))?;
        let reader = BufReader::new(file);
        let mut blocks = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.wrap_err_with_context(|| format!("Error reading line {} from LD block file", line_num + 1))?;
            let trimmed_line = line.trim();
            if trimmed_line.is_empty()
                || trimmed_line.starts_with('#')
                || trimmed_line.starts_with("chr\t")
                || trimmed_line.starts_with("chromosome\t")
            {
                continue;
            }

            let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
            if parts.len() < 3 { // Expect at least 3 fields for chr, start, end.
                warn!("Skipping malformed LD block line {}: '{}' (expected at least 3 fields: chr start end)", line_num + 1, line);
                continue;
            }
            let chr_str_original = parts[0];
            let chr_str = Self::normalize_chromosome_name(chr_str_original);
            let start_pos = parts[1].parse::<i32>()
                .wrap_err_with_context(|| format!("LD block line {}: Error parsing start pos '{}'", line_num + 1, parts[1]))?;
            let end_pos = parts[2].parse::<i32>()
                .wrap_err_with_context(|| format!("LD block line {}: Error parsing end pos '{}'", line_num + 1, parts[2]))?;

            // Auto-generate a block ID based on chromosome and coordinates.
            let block_id_str = format!("{}:{}-{}", chr_str, start_pos, end_pos);

            blocks.push((chr_str, start_pos, end_pos, block_id_str));
        }
        if blocks.is_empty() {
            warn!("No valid LD blocks parsed from file: {}. Make sure format is chr start end (whitespace separated). Block IDs are auto-generated.", self.config.ld_block_file_path);
        } else {
            info!("Successfully parsed {} LD blocks from file. Block IDs were auto-generated.", blocks.len());
        }
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
        observed_homozygous_allele1_count: usize,
        observed_heterozygous_count: usize,
        observed_homozygous_allele2_count: usize,
    ) -> f64 {
        let total_samples_with_genotypes = observed_homozygous_allele1_count + observed_heterozygous_count + observed_homozygous_allele2_count;
        if total_samples_with_genotypes == 0 {
            warn!("HWE Test: Total samples (0) is effectively zero. Cannot compute HWE p-value. Returning 1.0.");
            return 1.0;
        }

        let count_allele1 = 2.0 * observed_homozygous_allele1_count as f64 + observed_heterozygous_count as f64;
        let count_allele2 = 2.0 * observed_homozygous_allele2_count as f64 + observed_heterozygous_count as f64;
        let total_alleles_observed = count_allele1 + count_allele2;

        if total_alleles_observed <= 1e-9 {
            warn!("HWE Test: Total alleles observed ({}) is effectively zero. Cannot compute allele frequencies. Returning 1.0.", total_alleles_observed);
            return 1.0;
        }

        let frequency_allele1 = count_allele1 / total_alleles_observed;
        let frequency_allele2 = count_allele2 / total_alleles_observed;

        const FREQ_EPSILON: f64 = 1e-9;
        if frequency_allele1 < FREQ_EPSILON || frequency_allele2 < FREQ_EPSILON {
            return 1.0;
        }
        if (frequency_allele1 + frequency_allele2 - 1.0).abs() > 1e-6 {
            warn!(
                "HWE Test: Allele frequencies p ({:.4}) and q ({:.4}) do not sum to 1.0. Counts: HomA1={}, Het={}, HomA2={}. Check input counts.",
                frequency_allele1, frequency_allele2,
                observed_homozygous_allele1_count, observed_heterozygous_count, observed_homozygous_allele2_count
            );
            return 1.0;
        }

        let expected_homozygous_allele1 =
            frequency_allele1 * frequency_allele1 * total_samples_with_genotypes as f64;
        let expected_heterozygous =
            2.0 * frequency_allele1 * frequency_allele2 * total_samples_with_genotypes as f64;
        let expected_homozygous_allele2 =
            frequency_allele2 * frequency_allele2 * total_samples_with_genotypes as f64;

        let mut chi_squared_statistic: f64 = 0.0;
        const MIN_EXPECTED_FOR_DIVISION: f64 = 1e-9;

        if expected_homozygous_allele1 > MIN_EXPECTED_FOR_DIVISION {
            chi_squared_statistic +=
                (observed_homozygous_allele1_count as f64 - expected_homozygous_allele1).powi(2)
                    / expected_homozygous_allele1;
        } else if (observed_homozygous_allele1_count as f64) > MIN_EXPECTED_FOR_DIVISION {
            chi_squared_statistic = f64::INFINITY;
        }

        if chi_squared_statistic.is_finite() {
            if expected_heterozygous > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic += (observed_heterozygous_count as f64 - expected_heterozygous)
                    .powi(2)
                    / expected_heterozygous;
            } else if (observed_heterozygous_count as f64) > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic = f64::INFINITY;
            }
        }

        if chi_squared_statistic.is_finite() {
            if expected_homozygous_allele2 > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic +=
                    (observed_homozygous_allele2_count as f64 - expected_homozygous_allele2).powi(2)
                        / expected_homozygous_allele2;
            } else if (observed_homozygous_allele2_count as f64) > MIN_EXPECTED_FOR_DIVISION {
                chi_squared_statistic = f64::INFINITY;
            }
        }

        if chi_squared_statistic.is_nan() {
            warn!("HWE Test: Chi-squared statistic is NaN. This can occur with extreme deviations or problematic inputs. Counts: HomA1={}, Het={}, HomA2={}. Freqs: p={:.4}, q={:.4}. Exp: E_HomA1={:.2}, E_Het={:.2}, E_HomA2={:.2}. Returning p=1.0.",
                observed_homozygous_allele1_count, observed_heterozygous_count, observed_homozygous_allele2_count,
                frequency_allele1, frequency_allele2,
                expected_homozygous_allele1, expected_heterozygous, expected_homozygous_allele2);
            return 1.0;
        }

        if chi_squared_statistic == f64::INFINITY {
            return 0.0;
        }

        match ChiSquared::new(1.0) {
            Ok(chi_sq_dist) => {
                let cdf_value = chi_sq_dist.cdf(chi_squared_statistic);
                if cdf_value.is_nan() {
                    warn!(
                        "HWE Test: CDF value is NaN for Chi-squared statistic {}. Returning p=1.0.",
                        chi_squared_statistic
                    );
                    1.0
                } else {
                    (1.0 - cdf_value).max(0.0)
                }
            }
            Err(e) => {
                error!("HWE Test: Failed to create ChiSquared distribution (df=1.0): {}. Chi-sq stat was: {}. Returning p=1.0.", e, chi_squared_statistic);
                1.0
            }
        }
    }

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

/// Accessor for genotype data from a BED file, designed to be used by EigenSNP.
/// It uses an IoService to request data from the BED file.
#[derive(Clone)]
pub struct MicroarrayGenotypeAccessor {
    io_request_tx: flume::Sender<io_service_infrastructure::IoRequest>,
    original_indices_of_qc_samples: Arc<Vec<isize>>,
    num_total_qc_samples: usize,
    original_indices_of_pca_snps: Arc<Vec<usize>>,
    num_total_pca_snps: usize,
    mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
}

impl MicroarrayGenotypeAccessor {
    /// Creates a new MicroarrayGenotypeAccessor.
    pub fn new(
        io_request_tx: flume::Sender<io_service_infrastructure::IoRequest>,
        original_indices_of_qc_samples: Arc<Vec<isize>>,
        num_total_qc_samples: usize,
        original_indices_of_pca_snps: Arc<Vec<usize>>,
        num_total_pca_snps: usize,
        mean_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
        std_devs_allele1_dosages_for_pca_snps: Arc<Array1<f32>>,
    ) -> Self {
        assert_eq!(
            original_indices_of_qc_samples.len(),
            num_total_qc_samples,
            "Accessor: Initial QC sample count mismatch with provided 'original_indices_of_qc_samples' vector length."
        );
        assert_eq!(
            original_indices_of_pca_snps.len(),
            num_total_pca_snps,
            "Accessor: PCA-ready SNP count (D_blocked) mismatch with provided 'original_indices_of_pca_snps' vector length."
        );
        assert_eq!(
            mean_allele1_dosages_for_pca_snps.len(),
            num_total_pca_snps,
            "Accessor: Mean dosage vector length mismatch with D_blocked SNP count."
        );
        assert_eq!(
            std_devs_allele1_dosages_for_pca_snps.len(),
            num_total_pca_snps,
            "Accessor: StdDev dosage vector length mismatch with D_blocked SNP count."
        );

        Self {
            io_request_tx,
            original_indices_of_qc_samples,
            num_total_qc_samples,
            original_indices_of_pca_snps,
            num_total_pca_snps,
            mean_allele1_dosages_for_pca_snps,
            std_devs_allele1_dosages_for_pca_snps,
        }
    }

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
        pca_snp_ids_to_fetch: &[PcaSnpId],
        qc_sample_ids_to_fetch: &[QcSampleId],
    ) -> Result<Array2<f32>, ThreadSafeStdError> {
        let result_internal: Result<Array2<f32>, ThreadSafeStdError> = (|| {
            let num_requested_snps = pca_snp_ids_to_fetch.len();
            let num_requested_samples = qc_sample_ids_to_fetch.len();

            if num_requested_snps == 0 || num_requested_samples == 0 {
                return Ok(Array2::zeros((num_requested_snps, num_requested_samples)));
            }

            let original_m_indices_for_bed: Vec<isize> = pca_snp_ids_to_fetch
                .iter()
                .map(|pca_id| self.original_indices_of_pca_snps[pca_id.0] as isize)
                .collect();
            let requested_original_sample_indices_for_bed: Vec<isize> = qc_sample_ids_to_fetch
                .iter()
                .map(|qc_id| self.original_indices_of_qc_samples[qc_id.0])
                .collect();

            let (response_tx, response_rx) = flume::bounded(1);
            let request = io_service_infrastructure::IoRequest::GetSnpBlockForEigen {
                original_m_indices_for_bed,
                original_sample_indices_for_bed: Arc::new(requested_original_sample_indices_for_bed),
                response_tx,
            };

            self.io_request_tx.send_timeout(request, io_service_infrastructure::DEFAULT_IO_OPERATION_TIMEOUT)
                .wrap_err_with_str("Failed to send SnpBlockForEigen request to IoService")?;

            match response_rx.recv_timeout(io_service_infrastructure::DEFAULT_IO_OPERATION_TIMEOUT)
                .wrap_err_with_str("Failed to receive SnpBlockForEigen response from IoService")? {
                io_service_infrastructure::IoResponse::SnpBlockData { raw_i8_block_result } => {
                    let raw_dosages_snps_by_samples_i8_array2 = raw_i8_block_result
                        .map_err(|s| Box::new(DataPrepError::Message(s)) as ThreadSafeStdError)?;

                    if raw_dosages_snps_by_samples_i8_array2.nrows() != num_requested_snps
                        || raw_dosages_snps_by_samples_i8_array2.ncols() != num_requested_samples {
                        return Err(Box::new(DataPrepError::Message(format!("IoService returned SnpBlockData with unexpected dimensions. Expected: {}x{}, Got: {}x{}",
                                              num_requested_snps, num_requested_samples,
                                              raw_dosages_snps_by_samples_i8_array2.nrows(), raw_dosages_snps_by_samples_i8_array2.ncols()))) as ThreadSafeStdError);
                    }

                    let mut standardized_block_f32 = Array2::<f32>::uninit(raw_dosages_snps_by_samples_i8_array2.raw_dim());
                    for i_req_snp in 0..num_requested_snps {
                        let pca_snp_id_val = pca_snp_ids_to_fetch[i_req_snp].0;
                        let mean_dosage = self.mean_allele1_dosages_for_pca_snps[pca_snp_id_val];
                        let std_dev_dosage = self.std_devs_allele1_dosages_for_pca_snps[pca_snp_id_val];

                        let raw_snp_row = raw_dosages_snps_by_samples_i8_array2.row(i_req_snp);
                        let mut standardized_snp_row_to_fill = standardized_block_f32.row_mut(i_req_snp);

                        let num_samples_in_row = raw_snp_row.len();
                        let raw_slice_option = raw_snp_row.as_slice();
                        let std_mut_slice_option = standardized_snp_row_to_fill.as_slice_mut();

                        let mut i_req_sample = 0;

                        if std_dev_dosage.abs() < 1e-9 {
                            let simd_zeros = Simd::<f32, LANES_I8_F32_8>::splat(0.0f32);
                            let simd_missing_i8 = Simd::<i8, LANES_I8_F32_8>::splat(-127i8);

                            if let (Some(raw_slice), Some(std_mut_slice)) = (raw_slice_option, std_mut_slice_option) {
                                while i_req_sample + LANES_I8_F32_8 <= num_samples_in_row {
                                    let raw_chunk = Simd::<i8, LANES_I8_F32_8>::from_slice(&raw_slice[i_req_sample..i_req_sample + LANES_I8_F32_8]);
                                    if raw_chunk.simd_eq(simd_missing_i8).any() {
                                        for k in 0..LANES_I8_F32_8 {
                                            let current_idx = i_req_sample + k;
                                            if raw_slice[current_idx] == -127i8 {
                                                return Err(Box::new(DataPrepError::Message(format!("Unexpected missing genotype (-127i8) in SnpBlockData for PCA SNP ID {} (original BIM index {}), requested sample index {}. This should have been filtered by QC.",
                                                                      pca_snp_id_val, self.original_indices_of_pca_snps[pca_snp_id_val], qc_sample_ids_to_fetch[current_idx].0))) as ThreadSafeStdError);
                                            }
                                            // If we were to continue processing this chunk after an error (not current logic):
                                            // std_mut_slice[current_idx] = 0.0f32;
                                        }
                                        // This part is currently unreachable due to the return Err above.
                                        // If error handling changes to allow processing past a found error in a chunk,
                                        // this increment and continue would be needed.
                                        // i_req_sample += LANES_I8_F32_8;
                                        // continue 'simd_loop_zero_std_dev;
                                    }
                                    let target_slice: &mut [std::mem::MaybeUninit<f32>] = &mut std_mut_slice[i_req_sample..i_req_sample + LANES_I8_F32_8];
                                    unsafe {
                                        let initialized_slice = std::slice::from_raw_parts_mut(
                                            target_slice.as_mut_ptr() as *mut f32,
                                            LANES_I8_F32_8
                                        );
                                        simd_zeros.write_to_slice_unaligned(initialized_slice);
                                    }
                                    i_req_sample += LANES_I8_F32_8;
                                }
                            }
                            // Scalar remainder loop (or full scalar if slices were not available)
                            for k_sample in i_req_sample..num_samples_in_row {
                                let raw_dosage_val_i8 = if let Some(slice) = raw_slice_option {
                                    slice[k_sample]
                                } else {
                                    *raw_snp_row.uget(k_sample)
                                };
                                if raw_dosage_val_i8 == -127i8 {
                                    return Err(Box::new(DataPrepError::Message(format!("Unexpected missing genotype (-127i8) in SnpBlockData for PCA SNP ID {} (original BIM index {}), requested sample index {}. This should have been filtered by QC.",
                                                          pca_snp_id_val, self.original_indices_of_pca_snps[pca_snp_id_val], qc_sample_ids_to_fetch[k_sample].0))) as ThreadSafeStdError);
                                }
                                if let Some(slice_mut) = std_mut_slice_option {
                                    slice_mut[k_sample] = std::mem::MaybeUninit::new(0.0f32);
                                } else {
                                    standardized_snp_row_to_fill.uget_mut(k_sample).write(0.0f32);
                                }
                            }
                        } else { // std_dev_dosage is NOT near zero
                            let simd_mean = Simd::<f32, LANES_I8_F32_8>::splat(mean_dosage);
                            let simd_std_dev = Simd::<f32, LANES_I8_F32_8>::splat(std_dev_dosage);
                            let simd_missing_i8 = Simd::<i8, LANES_I8_F32_8>::splat(-127i8);

                            if let (Some(raw_slice), Some(std_mut_slice)) = (raw_slice_option, std_mut_slice_option) {
                                while i_req_sample + LANES_I8_F32_8 <= num_samples_in_row {
                                    let raw_i8_chunk = Simd::<i8, LANES_I8_F32_8>::from_slice(&raw_slice[i_req_sample..i_req_sample + LANES_I8_F32_8]);
                                    if raw_i8_chunk.simd_eq(simd_missing_i8).any() {
                                        for k in 0..LANES_I8_F32_8 {
                                            let current_idx = i_req_sample + k;
                                            if raw_slice[current_idx] == -127i8 {
                                                return Err(Box::new(DataPrepError::Message(format!("Unexpected missing genotype (-127i8) in SnpBlockData for PCA SNP ID {} (original BIM index {}), requested sample index {}. This should have been filtered by QC.",
                                                                      pca_snp_id_val, self.original_indices_of_pca_snps[pca_snp_id_val], qc_sample_ids_to_fetch[current_idx].0))) as ThreadSafeStdError);
                                            }
                                            // Scalar processing for this element if error handling changes:
                                            // let standardized_val = (raw_slice[current_idx] as f32 - mean_dosage) / std_dev_dosage;
                                            // std_mut_slice[current_idx] = standardized_val;
                                        }
                                        // As above, currently unreachable.
                                        // i_req_sample += LANES_I8_F32_8;
                                        // continue 'simd_loop_nonzero_std_dev;
                                    }
                                    let raw_f32_chunk: Simd<f32, LANES_I8_F32_8> = raw_i8_chunk.cast();
                                    let standardized_chunk = (raw_f32_chunk - simd_mean) / simd_std_dev;
                                    let target_slice: &mut [std::mem::MaybeUninit<f32>] = &mut std_mut_slice[i_req_sample..i_req_sample + LANES_I8_F32_8];
                                    unsafe {
                                        let initialized_slice = std::slice::from_raw_parts_mut(
                                            target_slice.as_mut_ptr() as *mut f32,
                                            LANES_I8_F32_8
                                        );
                                        standardized_chunk.write_to_slice_unaligned(initialized_slice);
                                    }
                                    i_req_sample += LANES_I8_F32_8;
                                }
                            }
                            // Scalar remainder loop (or full scalar if slices were not available)
                            for k_sample in i_req_sample..num_samples_in_row {
                                let raw_dosage_val_i8 = if let Some(slice) = raw_slice_option {
                                    slice[k_sample]
                                } else {
                                    *raw_snp_row.uget(k_sample)
                                };
                                if raw_dosage_val_i8 == -127i8 {
                                     return Err(Box::new(DataPrepError::Message(format!("Unexpected missing genotype (-127i8) in SnpBlockData for PCA SNP ID {} (original BIM index {}), requested sample index {}. This should have been filtered by QC.",
                                                          pca_snp_id_val, self.original_indices_of_pca_snps[pca_snp_id_val], qc_sample_ids_to_fetch[k_sample].0))) as ThreadSafeStdError);
                                }
                                let standardized_val = (raw_dosage_val_i8 as f32 - mean_dosage) / std_dev_dosage;
                                if let Some(slice_mut) = std_mut_slice_option {
                                    slice_mut[k_sample].write(standardized_val);
                                } else {
                                    standardized_snp_row_to_fill.uget_mut(k_sample).write(standardized_val);
                                }
                            }
                        }
                    }
                    Ok(unsafe { standardized_block_f32.assume_init() })
                }
                unexpected_response => Err(Box::new(DataPrepError::Message(format!("Received unexpected IoResponse type from IoService: {:?}. Expected SnpBlockData.", unexpected_response))) as ThreadSafeStdError),
            }
        })();
        result_internal
    }

    fn num_pca_snps(&self) -> usize {
        self.num_total_pca_snps
    }
    fn num_qc_samples(&self) -> usize {
        self.num_total_qc_samples
    }
}
