# ../data/igsr_samples.tsv
# ../data/hg38_plink1.fam
# ../data/hg38_plink1.bed
# ../data/hg38_plink1.bim

# convert https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst to plink 1.9

from pathlib import Path
from pypdl import Pypdl

URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/resources/pgsc_HGDP+1kGP_v1.tar.zst"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
THREAD_COUNT = 8


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zipped_filename = URL.split("/")[-1]
    unzipped_filename = zipped_filename.rsplit('.zst', 1)[0]
    zipped_filepath = OUTPUT_DIR / zipped_filename
    unzipped_filepath = OUTPUT_DIR / unzipped_filename

    # --- Pre-flight Check ---
    print(f"Checking for existing data in: {OUTPUT_DIR}")
    if unzipped_filepath.exists():
        print(f"Found final (unzipped) file: {unzipped_filepath.name}")
        return
    if zipped_filepath.exists():
        print(f"Found downloaded (zipped) file: {zipped_filepath.name}")
        return

    print("Local file not found. Proceeding with download.")
    print("-" * 60)

    # Instantiate the Pypdl object.
    dl = Pypdl()

    # Call the start method with all download parameters
    dl.start(
        url=URL,
        file_path=str(zipped_filepath),
        segments=THREAD_COUNT,
        retries=5
    )

    # This line is only reached on successful completion.
    print(f"File saved to: {zipped_filepath}")

if __name__ == "__main__":
    main()
