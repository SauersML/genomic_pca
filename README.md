# Genomic PCA Tool

## Overview

`genomic_pca` is a command-line tool written in Rust for performing Principal Component Analysis (PCA) on genomic variant data from VCF (Variant Call Format) files. It processes multiple VCF files, filters variants based on criteria like Minor Allele Frequency (MAF), constructs a genotype matrix, and then runs PCA to identify principal components.

The tool outputs:
* Principal components for each sample.
* Eigenvalues (variance explained by each PC).
* Variant loadings (contribution of each variant to each PC).

## Prerequisites

* **Rust Toolchain:** You can install it from [rustup.rs](https://rustup.rs/).
* **VCF Files:** Input VCF files (`.vcf` or `.vcf.gz`). It's assumed that all VCF files share the same set of samples in the same order. The sample information is taken from the first VCF file processed.

## Building

```
cargo build --release
```

The executable will be located at `target/release/genomic_pca`.

## Running the Tool

You can run the tool using `cargo run` (which will compile and then run) or by directly executing the compiled binary.

### Using `cargo run`

  * **Debug mode:**
    ```
    cargo run -- [OPTIONS]
    ```
  * **Release mode (recommended for actual analysis):**
    ```
    cargo run --release -- [OPTIONS]
    ```
    The `--` separates arguments for `cargo run` from the arguments for `genomic_pca`.

### Directly Executing the Binary

  * After a debug build:
    ```
    ./target/debug/genomic_pca [OPTIONS]
    ```
  * After a release build:
    ```
    ./target/release/genomic_pca [OPTIONS]
    ```

### Command-Line Options

The following command-line options are available:

| Option                         | Short | Description                                                                      | Required | Default Value |
|--------------------------------|-------|----------------------------------------------------------------------------------|----------|---------------|
| `--vcf-dir <VCF_DIR>`          | `-d`  | Directory containing input VCF/VCF.gz files.                                     | Yes      | N/A           |
| `--out <OUTPUT_PREFIX>`        | `-o`  | Prefix for output files (e.g., "analysis/pca\_results").                       | Yes      | N/A           |
| `--components <COMPONENTS>`    | `-k`  | Number of principal components to compute.                                       | Yes      | N/A           |
| `--maf <MAF>`                  |       | Minimum Minor Allele Frequency (MAF) threshold for variants.                     | No       | `0.01`        |
| `--rfit-seed <RFIT_SEED>`      |       | Seed for randomized PCA (rfit) for reproducible results.                         | No       | None          |
| `--threads <THREADS>`          | `-t`  | Number of threads for parallel operations. Defaults to available physical cores. | No       | (CPU count)   |
| `--log-level <LOG_LEVEL>`      |       | Logging verbosity. Options: `Error`, `Warn`, `Info`, `Debug`, `Trace`.           | No       | `Info`        |
| `--help`                       | `-h`  | Print help information.                                                          | No       | N/A           |
| `--version`                    | `-V`  | Print version information.                                                       | No       | N/A           |

### Example Usage

Assuming your VCF files are in a folder named `vcfs` in the current directory, and you want to compute 10 principal components, outputting files with the prefix `pca_output/run1`:

**Using `cargo run` (release mode):**

```
cargo run --release -- \
    --vcf-dir ./vcfs \
    --out pca_output/run1 \
    --components 10 \
    --maf 0.05 \
    --threads 8 \
    --log-level Info
```

**Using the compiled release binary:**

```
./target/release/genomic_pca \
    --vcf-dir ./vcfs \
    --out pca_output/run1 \
    --components 10 \
    --maf 0.05 \
    --threads 8 \
    --log-level Info
```

This will generate files like:

  * `pca_output/run1.pca.tsv` (principal components for samples)
  * `pca_output/run1.eigenvalues.tsv` (variance explained by PCs)
  * `pca_output/run1.loadings.tsv` (variant loadings)


### Testing

Download the files:
```
BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL"
MANIFEST_URL="$BASE_URL/20190312_biallelic_SNV_and_INDEL_MANIFEST.txt"
curl -s "$MANIFEST_URL" | \
awk '{print substr($1,3)}' | \
grep -E '^ALL\.chr([0-9]+|X)\.shapeit2_integrated_snvindels_v2a_27022019\.GRCh38\.phased\.(vcf\.gz|vcf\.gz\.tbi)$' | \
sed "s|^|$BASE_URL/|" > download_urls.txt
ls -l download_urls.txt
wc -l download_urls.txt
echo "First few URLs:"
head download_urls.txt
if [ -s download_urls.txt ] && [ $(wc -l < download_urls.txt) -gt 1 ]; then
    parallel -j 10 --eta --retries 3 --timeout 300 -a download_urls.txt curl -fLO {}
    echo "GNU Parallel download process complete."
else
    echo "Error: download_urls.txt has insufficient content."
fi
```

### Profiling

Run:
```
cargo build --profile profiling
```

Make sure samply is installed:
```
cargo install samply
```

```
RUST_BACKTRACE=1 samply record ./target/profiling/genomic_pca \
    --vcf-dir ./vcfs \
    --out pca_output/run1 \
    --components 10 \
    --maf 0.05 \
    --threads 8 \
    --log-level Info
```

If there is an error, consider running:
```
echo '1' | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Or, download the GitHub Actions artifact `samply-profile-json.zip`, then run:
```
unzip -o samply-profile-json.zip profile.json && samply load profile.json
```
