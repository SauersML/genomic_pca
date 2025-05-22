import os
import glob

def get_chromosome_sort_value(chrom_str):
    """
    Assigns a sort order and numeric value to chromosome strings
    for robust sorting.
    Order: 1-22, X, Y, MT (M), XY, then others alphabetically.
    Handles 'chr' prefix.
    """
    chrom_normalized = chrom_str.lower()
    original_chrom_for_tiebreak = chrom_str

    prefix_removed_val = chrom_normalized[3:] if chrom_normalized.startswith("chr") else chrom_normalized

    # Numeric chromosomes 1-22
    if prefix_removed_val.isdigit():
        num = int(prefix_removed_val)
        if 1 <= num <= 22:
            return (1, num, original_chrom_for_tiebreak)

    # Specific chromosomes
    if prefix_removed_val == 'x':
        return (2, 0, original_chrom_for_tiebreak)
    if prefix_removed_val == 'y':
        return (3, 0, original_chrom_for_tiebreak)
    if prefix_removed_val == 'mt' or prefix_removed_val == 'm':
        return (4, 0, original_chrom_for_tiebreak)
    if prefix_removed_val == 'xy':
        return (5, 0, original_chrom_for_tiebreak)

    # All other chromosomes (e.g., unplaced contigs, custom names)
    return (6, 0, original_chrom_for_tiebreak)


def custom_sort_key(item):
    """
    Sorts items based on chromosome (custom order) and then position (numeric).
    item is a tuple (chrom_str, pos_str where pos_str is pre-validated as digits).
    """
    chrom_str, pos_str = item
    chrom_sort_tuple = get_chromosome_sort_value(chrom_str)
    # pos_str is assumed to be a string of digits due to pre-filtering
    pos_int = int(pos_str)
    return (chrom_sort_tuple, pos_int)

def create_combined_subset_tsv(output_filename="subset.tsv"):
    """
    Finds all .tsv files, extracts unique and valid CHROM/POS pairs,
    sorts them, and writes to output_filename.
    Assumes input .tsv files have a header, CHROM & POS are first two columns.
    Avoids try-except in main data processing as per user request.
    """
    tsv_files = glob.glob("*.tsv")
    all_chrom_pos_pairs = set()
    processed_files_count = 0
    malformed_lines_count = 0
    non_numeric_pos_count = 0

    for filepath in tsv_files:
        if os.path.basename(filepath) == output_filename:
            continue

        with open(filepath, 'r') as f:
            header_line = f.readline()
            if not header_line.strip(): # Skip empty or headerless files
                continue

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    chrom = parts[0]
                    pos_str = parts[1]
                    if pos_str.isdigit():
                        all_chrom_pos_pairs.add((chrom, pos_str))
                    else:
                        non_numeric_pos_count +=1
                else:
                    malformed_lines_count += 1
        processed_files_count += 1

    if not all_chrom_pos_pairs:
        print(f"No valid CHROM/POS data found to write to {output_filename}.")
        if malformed_lines_count > 0:
            print(f"Skipped {malformed_lines_count} malformed lines (insufficient columns).")
        if non_numeric_pos_count > 0:
            print(f"Skipped {non_numeric_pos_count} lines due to non-numeric positions.")
        return

    sorted_pairs = sorted(list(all_chrom_pos_pairs), key=custom_sort_key)

    with open(output_filename, 'w') as outfile:
        outfile.write("CHROM\tPOS\n")
        for chrom, pos in sorted_pairs:
            outfile.write(f"{chrom}\t{pos}\n")
    
    print(f"Created {output_filename} with {len(sorted_pairs)} unique CHROM/POS pairs from {processed_files_count} file(s).")
    if malformed_lines_count > 0:
        print(f"Note: Skipped {malformed_lines_count} malformed lines (insufficient columns) during processing.")
    if non_numeric_pos_count > 0:
        print(f"Note: Skipped {non_numeric_pos_count} lines due to non-numeric positions during processing.")

if __name__ == "__main__":
    create_combined_subset_tsv()
