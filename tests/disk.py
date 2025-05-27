import mmap
import os
import random
import time
import concurrent.futures
from pathlib import Path
import subprocess # To call gsutil

# --- GCS and Local Path Configuration ---
GCS_MICROARRAY_PLINK_DIR = "gs://fc-aou-datasets-controlled/v8/microarray/plink"
GCS_BED_FILE_PATH = f"{GCS_MICROARRAY_PLINK_DIR}/arrays.bed"
GCS_BIM_FILE_PATH = f"{GCS_MICROARRAY_PLINK_DIR}/arrays.bim"
GCS_FAM_FILE_PATH = f"{GCS_MICROARRAY_PLINK_DIR}/arrays.fam"

# Local paths where the files will be copied to for mmap testing
LOCAL_DATA_DIR = Path("/tmp/aou_microarray_data_for_pca_test")
LOCAL_BED_FILE_PATH = LOCAL_DATA_DIR / "arrays.bed" # Keep original names
LOCAL_BIM_FILE_PATH = LOCAL_DATA_DIR / "arrays.bim"
LOCAL_FAM_FILE_PATH = LOCAL_DATA_DIR / "arrays.fam"

# --- Test Parameters ---
TIME_LIMIT_PER_TEST_SECONDS = 28.0 
NUM_SNPS_PER_STRIP_TEST = 2000
NUM_LD_BLOCKS_TO_TEST = 2000 # Increased to get more work done in time limit
AVG_SNPS_PER_LD_BLOCK_TEST = 100
MAX_SNPS_PER_LD_BLOCK_TEST = 200
MIN_SNPS_PER_LD_BLOCK_TEST = 50
NUM_SAMPLES_SUBSET_NS_TEST = 5000
NUM_PARALLEL_WORKERS = os.cpu_count() or 4

# --- Utility Functions ---

def localize_file_from_gcs(gcs_path: str, local_path: Path, google_project_id: str | None) -> bool:
    """Attempts to download a file from GCS if it doesn't exist locally."""
    if local_path.exists():
        print(f"Local file already exists: {local_path}")
        return True
    
    print(f"Attempting to download {gcs_path} to {local_path}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    gsutil_command = ["gsutil"]
    if google_project_id:
        gsutil_command.extend(["-u", google_project_id])
    gsutil_command.extend(["cp", gcs_path, str(local_path)])
    
    try:
        print(f"Executing: {' '.join(gsutil_command)}")
        process = subprocess.run(gsutil_command, check=True, capture_output=True, text=True)
        print(f"Successfully downloaded {gcs_path}.")
        if process.stderr:
            print(f"gsutil stderr: {process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to download {gcs_path} using gsutil.")
        print(f"  Command: {' '.join(e.cmd)}")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        print(f"Please ensure 'gsutil' is installed, configured, and you have permissions.")
        return False
    except FileNotFoundError:
        print("ERROR: 'gsutil' command not found. Please install and configure the Google Cloud SDK.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during GCS localization: {e}")
        return False

def count_lines(filepath: Path) -> int:
    count = 0
    try:
        with filepath.open("r") as f:
            for _ in f:
                count += 1
    except Exception as e:
        print(f"Error counting lines in {filepath}: {e}")
        return 0
    return count

def get_snp_and_sample_counts(bim_filepath: Path, fam_filepath: Path) -> tuple[int, int]:
    print(f"Attempting to read SNP count from local BIM: {bim_filepath}")
    num_snps = count_lines(bim_filepath)
    
    print(f"Attempting to read sample count from local FAM: {fam_filepath}")
    num_samples = count_lines(fam_filepath)
    
    return num_snps, num_samples

def get_unpacked_genotypes_for_snp(
    mmap_file_view,
    _total_samples_in_snp_row, # Not strictly needed if mmap_file_view is just for this SNP
    target_sample_original_indices_in_row 
    ):
    unpacked_gts = []
    for sample_original_idx_in_row in target_sample_original_indices_in_row:
        byte_offset_for_sample_in_row = sample_original_idx_in_row // 4
        bit_shift_within_byte = (sample_original_idx_in_row % 4) * 2
        
        if byte_offset_for_sample_in_row >= len(mmap_file_view):
            unpacked_gts.append(255) 
            continue

        packed_byte = mmap_file_view[byte_offset_for_sample_in_row]
        genotype_code = (packed_byte >> bit_shift_within_byte) & 0b11
        
        allele_count: int
        if genotype_code == 0b00: allele_count = 0
        elif genotype_code == 0b10: allele_count = 1
        elif genotype_code == 0b11: allele_count = 2
        elif genotype_code == 0b01: allele_count = 255 
        else: raise ValueError(f"Invalid genotype code {genotype_code}")
        unpacked_gts.append(allele_count)
    return unpacked_gts

def process_snp_block_for_accessor_test(
    bed_mmap_obj, 
    snp_original_indices_in_block, 
    target_sample_original_indices, 
    total_samples_in_file 
    ):
    all_snp_data_for_block = []
    bytes_per_snp_row = (total_samples_in_file + 3) // 4

    for snp_original_idx in snp_original_indices_in_block:
        snp_data_start_offset = 3 + snp_original_idx * bytes_per_snp_row
        snp_data_end_offset = snp_data_start_offset + bytes_per_snp_row
        
        if snp_data_end_offset > len(bed_mmap_obj):
            all_snp_data_for_block.append([]) 
            continue

        snp_mmap_slice = bed_mmap_obj[snp_data_start_offset:snp_data_end_offset]
        gts = get_unpacked_genotypes_for_snp(snp_mmap_slice, total_samples_in_file, target_sample_original_indices)
        all_snp_data_for_block.append(gts)
    return all_snp_data_for_block


# --- Test Functions (adapted with time limits) ---

def test_sequential_snp_strips(bed_filepath: Path, num_snps_total: int, num_samples_total: int, 
                               strip_size: int, sample_indices_to_get: list[int]):
    print(f"\n## Test: Sequential SNP Strips (Strip Size: {strip_size}, {len(sample_indices_to_get)} samples, Time Limit: {TIME_LIMIT_PER_TEST_SECONDS}s) ##")
    if num_snps_total == 0 or not sample_indices_to_get:
        print("Skipping test: No SNPs or no samples to process.")
        return 0.0, 0.0, 0

    max_strips_to_process = (num_snps_total + strip_size - 1) // strip_size
    total_gts_processed = 0
    strips_processed_count = 0
    
    overall_start_time = time.perf_counter()
    try:
        with open(bed_filepath, "rb") as f_bed:
            with mmap.mmap(f_bed.fileno(), 0, access=mmap.ACCESS_READ) as mm_bed:
                for i in range(max_strips_to_process):
                    current_elapsed = time.perf_counter() - overall_start_time
                    if current_elapsed > TIME_LIMIT_PER_TEST_SECONDS:
                        print(f"  Time limit ({TIME_LIMIT_PER_TEST_SECONDS}s) reached after {current_elapsed:.2f}s for sequential strip test.")
                        break
                    
                    strip_start_snp_idx = i * strip_size
                    strip_end_snp_idx = min((i + 1) * strip_size, num_snps_total)
                    snp_original_indices_in_strip = list(range(strip_start_snp_idx, strip_end_snp_idx))

                    if not snp_original_indices_in_strip: continue
                    
                    data_block = process_snp_block_for_accessor_test(
                        mm_bed, snp_original_indices_in_strip, sample_indices_to_get, num_samples_total
                    )
                    if data_block: total_gts_processed += sum(len(sg) for sg in data_block if sg) # Check if sg is not empty
                    strips_processed_count +=1
                    if strips_processed_count % max(1, (max_strips_to_process // 20 if max_strips_to_process > 20 else 1)) == 0:
                         print(f"  Processed strip {strips_processed_count}/{max_strips_to_process}... (Elapsed: {current_elapsed:.2f}s)")
    except FileNotFoundError:
        print(f"ERROR: BED file not found at {bed_filepath} during sequential strip test.")
        return 0.0,0.0,0
    except Exception as e:
        print(f"Error during sequential strip test: {e}")
        return 0.0,0.0,0

    duration = time.perf_counter() - overall_start_time # Recalculate actual duration
    gts_per_second = total_gts_processed / duration if duration > 0 else 0
    
    print(f"Processed {strips_processed_count} strips, {total_gts_processed} genotypes.")
    print(f"Duration: {duration:.4f} seconds")
    print(f"Genotypes per second: {gts_per_second:.2f}")
    return duration, gts_per_second, total_gts_processed


def test_ld_block_access(bed_filepath: Path, num_snps_total: int, num_samples_total: int, 
                         num_blocks_config: int, avg_snps_per_block_config: int, 
                         min_snps_config: int, max_snps_config: int,
                         sample_indices_to_get: list[int], test_name_suffix=""):
    print(f"\n## Test: LD Block-like Access ({len(sample_indices_to_get)} samples, {test_name_suffix}, Time Limit: {TIME_LIMIT_PER_TEST_SECONDS}s) ##")
    if num_snps_total == 0 or not sample_indices_to_get: print("Skipping: No SNPs/samples."); return 0.0,0.0,0

    ld_blocks_snp_indices = []
    if num_snps_total > 0 and num_blocks_config > 0 :
        current_snp_start = 0
        for _ in range(num_blocks_config): 
            max_start_pos = max(0, num_snps_total - min_snps_config)
            if current_snp_start > max_start_pos or current_snp_start >= num_snps_total : 
                current_snp_start = random.randint(0, max(0,max_start_pos//2))
            block_start = random.randint(current_snp_start, max(current_snp_start, max_start_pos))
            block_size = min(random.randint(min_snps_config, max_snps_config), num_snps_total - block_start)
            if block_size == 0 and num_snps_total > block_start : block_size = 1
            block_end = block_start + block_size
            if block_start < block_end: ld_blocks_snp_indices.append(list(range(block_start, block_end)))
            current_snp_start = block_end
    
    if not ld_blocks_snp_indices: print("No LD blocks generated for testing."); return 0.0,0.0,0

    total_gts_processed = 0
    blocks_processed_count = 0
    overall_start_time = time.perf_counter()
    try:
        with open(bed_filepath, "rb") as f_bed:
            with mmap.mmap(f_bed.fileno(), 0, access=mmap.ACCESS_READ) as mm_bed:
                for i, snp_indices_in_block in enumerate(ld_blocks_snp_indices):
                    current_elapsed = time.perf_counter() - overall_start_time
                    if current_elapsed > TIME_LIMIT_PER_TEST_SECONDS:
                        print(f"  Time limit ({TIME_LIMIT_PER_TEST_SECONDS}s) reached after {current_elapsed:.2f}s for LD block test.")
                        break
                    if not snp_indices_in_block: continue
                    data_block = process_snp_block_for_accessor_test(
                        mm_bed, snp_indices_in_block, sample_indices_to_get, num_samples_total
                    )
                    if data_block: total_gts_processed += sum(len(sg) for sg in data_block if sg)
                    blocks_processed_count += 1
                    if blocks_processed_count % max(1, (len(ld_blocks_snp_indices) // 10 if len(ld_blocks_snp_indices) > 10 else 1)) == 0:
                         print(f"  Processed LD block {blocks_processed_count}/{len(ld_blocks_snp_indices)}... (Elapsed: {current_elapsed:.2f}s)")
    except FileNotFoundError:
        print(f"ERROR: BED file not found at {bed_filepath} during LD block test.")
        return 0.0,0.0,0
    except Exception as e:
        print(f"Error during LD block test: {e}")
        return 0.0,0.0,0
        
    duration = time.perf_counter() - overall_start_time
    gts_per_second = total_gts_processed / duration if duration > 0 else 0
    
    print(f"Processed {blocks_processed_count} LD blocks, {total_gts_processed} genotypes.")
    print(f"Duration: {duration:.4f} seconds")
    print(f"Genotypes per second: {gts_per_second:.2f}")
    return duration, gts_per_second, total_gts_processed

def parallel_task_ld_block_timed(args_tuple_with_queue):
    bed_filepath_str, snp_indices_in_block, sample_indices_to_get, num_samples_total, process_start_time, queue = args_tuple_with_queue
    if not snp_indices_in_block: return 0

    gts_in_this_block = 0
    try:
        with open(bed_filepath_str, "rb") as f_bed_local:
            with mmap.mmap(f_bed_local.fileno(), 0, access=mmap.ACCESS_READ) as mm_bed_local:
                data_block = process_snp_block_for_accessor_test(
                    mm_bed_local, snp_indices_in_block, sample_indices_to_get, num_samples_total
                )
        if data_block: gts_in_this_block = sum(len(sg) for sg in data_block if sg)
    except Exception as e:
        pass 
    return gts_in_this_block


def test_parallel_ld_block_access(bed_filepath: Path, num_snps_total: int, num_samples_total: int,
                                  num_blocks_config: int, avg_snps_per_block_config: int,
                                  min_snps_config: int, max_snps_config: int,
                                  sample_indices_to_get: list[int], num_workers: int, test_name_suffix=""):
    print(f"\n## Test: Parallel LD Block Access ({num_workers} workers, {len(sample_indices_to_get)} samples, {test_name_suffix}, Time Limit: {TIME_LIMIT_PER_TEST_SECONDS}s) ##")
    if num_snps_total == 0 or not sample_indices_to_get: print("Skipping: No SNPs/samples."); return 0.0,0.0,0

    ld_blocks_snp_indices = []
    if num_snps_total > 0 and num_blocks_config > 0:
        current_snp_start = 0
        for _ in range(num_blocks_config):
            max_start_pos = max(0, num_snps_total - min_snps_config)
            if current_snp_start > max_start_pos or current_snp_start >= num_snps_total : 
                 current_snp_start = random.randint(0, max(0, max_start_pos//2))
            block_start = random.randint(current_snp_start, max(current_snp_start, max_start_pos))
            block_size = min(random.randint(min_snps_config, max_snps_config), num_snps_total - block_start)
            if block_size == 0 and num_snps_total > block_start : block_size = 1
            block_end = block_start + block_size
            if block_start < block_end: ld_blocks_snp_indices.append(list(range(block_start, block_end)))
            current_snp_start = block_end

    if not ld_blocks_snp_indices: print("No LD blocks generated for parallel testing."); return 0.0,0.0,0

    tasks_args_list = [(str(bed_filepath), block_indices, sample_indices_to_get, num_samples_total) 
                       for block_indices in ld_blocks_snp_indices if block_indices]

    total_gts_processed = 0
    tasks_submitted_count = 0
    tasks_completed_count = 0
    
    overall_start_time = time.perf_counter()
    
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures_map = {} # Store future to task_index for better error reporting if needed
            
            for i, task_args in enumerate(tasks_args_list):
                current_elapsed_submit = time.perf_counter() - overall_start_time
                if current_elapsed_submit > TIME_LIMIT_PER_TEST_SECONDS:
                    print(f"  Time limit ({TIME_LIMIT_PER_TEST_SECONDS}s) reached during task submission after {current_elapsed_submit:.2f}s. Submitted {tasks_submitted_count} tasks.")
                    break 
                # Pass overall_start_time for per-task check (though less effective with ProcessPool)
                future = executor.submit(parallel_task_ld_block_timed, task_args + (overall_start_time, None)) # Placeholder for queue
                futures_map[future] = i
                tasks_submitted_count +=1

            print(f"  Submitted {tasks_submitted_count} tasks to ProcessPoolExecutor...")
            for future in concurrent.futures.as_completed(futures_map.keys(), timeout=TIME_LIMIT_PER_TEST_SECONDS + 5): # Overall timeout for waiting
                current_elapsed_results = time.perf_counter() - overall_start_time
                if current_elapsed_results > TIME_LIMIT_PER_TEST_SECONDS + 3: # Give a bit more for graceful finish
                     print(f"  Overall time limit ({TIME_LIMIT_PER_TEST_SECONDS}s) for parallel test exceeded during result collection after {current_elapsed_results:.2f}s.")
                     break
                try:
                    num_gts_in_block = future.result(timeout=10) # Timeout for individual future completion
                    total_gts_processed += num_gts_in_block
                    tasks_completed_count +=1
                    if tasks_completed_count % max(1, (tasks_submitted_count // 10 if tasks_submitted_count > 10 else 1)) == 0:
                        print(f"  Parallel processed task {tasks_completed_count}/{tasks_submitted_count} completed... (Elapsed: {current_elapsed_results:.2f}s)")
                except concurrent.futures.TimeoutError:
                    print(f"  A parallel task retrieval timed out.")
                except concurrent.futures.CancelledError:
                    print(f"  A parallel task was cancelled (likely due to time limit on submission).")
                except Exception as e:
                    task_idx = futures_map.get(future, -1)
                    print(f"  Error processing result from parallel task index {task_idx}: {e}")
    except FileNotFoundError:
        print(f"ERROR: BED file not found at {bed_filepath} during parallel LD block test.")
        return 0.0,0.0,0
    except Exception as e:
        print(f"Error during parallel LD block test execution: {e}")
        return 0.0,0.0,0
            
    duration = time.perf_counter() - overall_start_time
    gts_per_second = total_gts_processed / duration if duration > 0 else 0
    
    print(f"Processed {tasks_completed_count}/{tasks_submitted_count} LD block tasks in parallel, {total_gts_processed} genotypes.")
    print(f"Duration: {duration:.4f} seconds (may include some wind-down time if limit hit).")
    print(f"Genotypes per second: {gts_per_second:.2f}")
    return duration, gts_per_second, total_gts_processed


# --- Main Execution ---
if __name__ == "__main__":
    print("--- EigenSNP I/O Performance Test Script (Using Real Files) ---")
    google_project = os.getenv("GOOGLE_PROJECT")
    if not google_project:
        print("WARNING: GOOGLE_PROJECT environment variable not set. gsutil may require it for billing.")
        print("You can set it with: export GOOGLE_PROJECT='your-gcp-project-id'")
        # Forcing a placeholder here for gsutil command structure, user should ensure it's set
        google_project_for_gsutil = "your-gcp-project-id" 
    else:
        google_project_for_gsutil = google_project

    print(f"\nAttempting to use or download PLINK files to local directory: {LOCAL_DATA_DIR}")
    
    files_to_localize = {
        "BED": (GCS_BED_FILE_PATH, LOCAL_BED_FILE_PATH),
        "BIM": (GCS_BIM_FILE_PATH, LOCAL_BIM_FILE_PATH),
        "FAM": (GCS_FAM_FILE_PATH, LOCAL_FAM_FILE_PATH),
    }
    
    all_files_ready = True
    for ftype, (gcs_path, local_path) in files_to_localize.items():
        print(f"\nChecking/Localizing {ftype} file...")
        if not localize_file_from_gcs(gcs_path, local_path, google_project_for_gsutil):
            all_files_ready = False
            print(f"Failed to make {ftype} file available locally. See gsutil errors above.")
            print(f"Expected GCS path: {gcs_path}")
            print(f"Expected local path: {local_path}")
    
    if not all_files_ready:
        print("\nOne or more essential data files could not be localized. Exiting.")
        exit(1)
    
    print("-" * 60)
    print(f"\nAll PLINK files should now be available locally for testing at: {LOCAL_DATA_DIR}")

    try:
        NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL = get_snp_and_sample_counts(LOCAL_BIM_FILE_PATH, LOCAL_FAM_FILE_PATH)
        print(f"\nActual data dimensions from local files: {NUM_SNPS_TOTAL} SNPs, {NUM_SAMPLES_TOTAL} Samples.")
    except Exception as e:
        print(f"Error reading BIM/FAM files after localization: {e}")
        exit(1)

    if NUM_SNPS_TOTAL == 0 or NUM_SAMPLES_TOTAL == 0:
        print("Error: BIM or FAM file indicates zero SNPs or zero samples after localization. Cannot proceed.")
        exit(1)

    all_sample_original_indices = list(range(NUM_SAMPLES_TOTAL))
    actual_ns_count = min(NUM_SAMPLES_SUBSET_NS_TEST, NUM_SAMPLES_TOTAL)
    subset_ns_sample_original_indices = []
    if actual_ns_count > 0:
        subset_ns_sample_original_indices = random.sample(all_sample_original_indices, actual_ns_count)
        subset_ns_sample_original_indices.sort() 

    print(f"\n--- Starting I/O Performance Tests (each limited to ~{TIME_LIMIT_PER_TEST_SECONDS}s) ---")
    print(f"Using {NUM_PARALLEL_WORKERS} workers for parallel tests.")
    print(f"N_s subset size for tests: {len(subset_ns_sample_original_indices)}")

    test_suite = [
        ("Sequential SNP Strips (All N Samples)", lambda: test_sequential_snp_strips(
            LOCAL_BED_FILE_PATH, NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL, 
            NUM_SNPS_PER_STRIP_TEST, all_sample_original_indices
        )),
        ("LD Block Access (All N Samples)", lambda: test_ld_block_access(
            LOCAL_BED_FILE_PATH, NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL,
            NUM_LD_BLOCKS_TO_TEST, AVG_SNPS_PER_LD_BLOCK_TEST,
            MIN_SNPS_PER_LD_BLOCK_TEST, MAX_SNPS_PER_LD_BLOCK_TEST,
            all_sample_original_indices, "All N Samples"
        )),
        ("LD Block Access (Subset Ns Samples)", lambda: test_ld_block_access(
            LOCAL_BED_FILE_PATH, NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL,
            NUM_LD_BLOCKS_TO_TEST, AVG_SNPS_PER_LD_BLOCK_TEST,
            MIN_SNPS_PER_LD_BLOCK_TEST, MAX_SNPS_PER_LD_BLOCK_TEST,
            subset_ns_sample_original_indices if subset_ns_sample_original_indices else all_sample_original_indices, # Avoid empty list
            f"Subset Ns ({len(subset_ns_sample_original_indices) if subset_ns_sample_original_indices else 0}) Samples"
        ) if subset_ns_sample_original_indices else ("Skipped Test 3: N_s subset empty", lambda: (0,0,0))),
        ("Parallel LD Block Access (All N Samples)", lambda: test_parallel_ld_block_access(
            LOCAL_BED_FILE_PATH, NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL,
            NUM_LD_BLOCKS_TO_TEST, AVG_SNPS_PER_LD_BLOCK_TEST,
            MIN_SNPS_PER_LD_BLOCK_TEST, MAX_SNPS_PER_LD_BLOCK_TEST,
            all_sample_original_indices, NUM_PARALLEL_WORKERS, "All N Samples"
        )),
        ("Parallel LD Block Access (Subset Ns Samples)", lambda: test_parallel_ld_block_access(
            LOCAL_BED_FILE_PATH, NUM_SNPS_TOTAL, NUM_SAMPLES_TOTAL,
            NUM_LD_BLOCKS_TO_TEST, AVG_SNPS_PER_LD_BLOCK_TEST,
            MIN_SNPS_PER_LD_BLOCK_TEST, MAX_SNPS_PER_LD_BLOCK_TEST,
            subset_ns_sample_original_indices if subset_ns_sample_original_indices else all_sample_original_indices,
            NUM_PARALLEL_WORKERS, f"Subset Ns ({len(subset_ns_sample_original_indices) if subset_ns_sample_original_indices else 0}) Samples"
        ) if subset_ns_sample_original_indices else ("Skipped Test 5: N_s subset empty", lambda: (0,0,0))),
    ]

    for i, (name, test_fn) in enumerate(test_suite):
        print(f"\n[Test {i+1}] Running: {name}...")
        if name.startswith("Skipped"):
            print(name)
            continue
        try:
            # test_fn() will print its own summary
            duration, gts_per_sec, total_gts = test_fn()
            if duration > TIME_LIMIT_PER_TEST_SECONDS + 5: # Check if it significantly overran
                print(f"  WARNING: Test '{name}' significantly overran time limit. Duration: {duration:.2f}s")
        except Exception as e:
            print(f"  ERROR during '{name}': {e}")
        print("-" * 30)
        
    print(f"\n--- I/O Performance Tests Finished ---")
