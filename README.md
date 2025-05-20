

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
