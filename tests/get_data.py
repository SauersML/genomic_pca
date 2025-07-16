# ../data/igsr_samples.tsv
# ../data/hg38_plink1.fam
# ../data/hg38_plink1.bed
# ../data/hg38_plink1.bim

# convert https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst to plink 1.9

from pathlib import Path
from pypdl import Pypdl
import tarfile
import zstandard

# --- Configuration ---
URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
THREAD_COUNT = 8

def main():
    """
    Downloads, decompresses, and extracts reference panel data, skipping
    steps if the corresponding files already exist.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Define file and directory paths ---
    zipped_filename = URL.split("/")[-1]  # pgsc_HGDP+1kGP_v1.tar.zst
    unzipped_tar_filename = zipped_filename.removesuffix('.zst')  # pgsc_HGDP+1kGP_v1.tar
    # Assumes the tarball extracts to a directory with the same name as the tarball
    extracted_dir_name = unzipped_tar_filename.removesuffix('.tar')

    zipped_filepath = OUTPUT_DIR / zipped_filename
    unzipped_tar_filepath = OUTPUT_DIR / unzipped_tar_filename
    final_extracted_path = OUTPUT_DIR / extracted_dir_name

    # --- 1. Pre-flight Check (Caching) ---
    print(f"Checking for data in: {OUTPUT_DIR}")

    # A) Check if the final extracted directory already exists. If so, we are done.
    if final_extracted_path.is_dir():
        print(f"Final extracted directory found: {final_extracted_path.name}")
        print("Process complete. Nothing to do.")
        return

    print("Final data not found. Checking for intermediate files...")

    # B) Check for the unzipped tarball. If it exists, skip download and decompression.
    if not unzipped_tar_filepath.exists():
        # C) Check for the zipped archive. If it exists, skip the download.
        if not zipped_filepath.exists():
            print("Compressed archive not found. Proceeding with download.")
            print("-" * 60)
            # --- 2. Download ---
            dl = Pypdl()
            dl.start(
                url=URL,
                file_path=str(zipped_filepath),
                segments=THREAD_COUNT,
                retries=5
            )
            print(f"Download complete. File saved to: {zipped_filepath}")
        else:
            print(f"Found downloaded (zipped) file: {zipped_filepath.name}")

        # --- 3. Decompress .zst file ---
        print(f"Decompressing {zipped_filepath.name}...")
        try:
            dctx = zstandard.ZstdDecompressor()
            with open(zipped_filepath, 'rb') as ifh, open(unzipped_tar_filepath, 'wb') as ofh:
                dctx.copy_stream(ifh, ofh)
            print(f"Decompression complete. Created: {unzipped_tar_filepath.name}")
        except Exception as e:
            print(f"Error during decompression: {e}")
            # Clean up potentially corrupt file to ensure re-run works
            unzipped_tar_filepath.unlink(missing_ok=True)
            return
    else:
        print(f"Found unzipped tar archive: {unzipped_tar_filepath.name}")

    # --- 4. Extract all files from .tar archive ---
    print(f"Extracting all files from {unzipped_tar_filepath.name}...")
    try:
        with tarfile.open(unzipped_tar_filepath, 'r') as tar:
            tar.extractall(path=OUTPUT_DIR)
        print(f"Extraction complete. Files are located in: {final_extracted_path}")
    except tarfile.TarError as e:
        print(f"Error reading tar archive: {e}")
        return

    print("\nProcess finished. All files are now available in the data directory.")


if __name__ == "__main__":
    main()
