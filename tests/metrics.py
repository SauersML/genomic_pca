import pandas as pd
import numpy as np
import argparse
import os
import sys # For sys.exit()
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import mutual_info_classif
from scipy.spatial.distance import jensenshannon, cdist

# Helper function for JSD calculation using histograms per PC
def calculate_pairwise_jsd_histograms(pc_data_A, pc_data_B, n_pcs=10, num_bins=30):
    jsd_for_pair_across_pcs = []
    if pc_data_A.shape[0] == 0 or pc_data_B.shape[0] == 0:
        return np.nan

    for j in range(n_pcs):
        pc_A_dim_j = pc_data_A[:, j]
        pc_B_dim_j = pc_data_B[:, j]

        if len(pc_A_dim_j) == 0 or len(pc_B_dim_j) == 0:
            jsd_for_pair_across_pcs.append(np.nan)
            continue

        combined_data = np.concatenate((pc_A_dim_j, pc_B_dim_j))
        if len(np.unique(combined_data)) < 2:
            jsd_for_pair_across_pcs.append(0.0)
            continue
        
        min_val = np.min(combined_data)
        max_val = np.max(combined_data)
        
        if min_val >= max_val: # Ensure range for bins
            if min_val == max_val: # All values identical
                 jsd_for_pair_across_pcs.append(0.0)
                 continue
            max_val = min_val + 1e-6 # Create a tiny range if they were almost identical before unique check
        
        bins = np.linspace(min_val, max_val, num_bins + 1)
        hist_A, _ = np.histogram(pc_A_dim_j, bins=bins, density=False)
        hist_B, _ = np.histogram(pc_B_dim_j, bins=bins, density=False)
        
        # scipy.spatial.distance.jensenshannon normalizes p and q if they don't sum to 1.
        js_distance = jensenshannon(hist_A, hist_B, base=np.e)  # sqrt(JSD) in nats
        jsd_val = js_distance**2 # JSD in nats
        jsd_for_pair_across_pcs.append(jsd_val)

    if not jsd_for_pair_across_pcs or np.all(np.isnan(jsd_for_pair_across_pcs)):
        return np.nan
    return np.nanmean(jsd_for_pair_across_pcs)

def calculate_metrics_per_superpopulation(pca_file, sample_file, num_pcs_to_use=10):
    # --- 1. Data Loading ---
    # No try-except: script will fail if files not found or unreadable
    pca_df = pd.read_csv(pca_file, sep=r'\s+') # Use raw string for regex
    sample_df = pd.read_csv(sample_file, sep='\t') # Assuming igsr_samples.tsv is typically tab-separated

    # --- 2. Data Validation and Merging ---
    if 'SampleID' not in pca_df.columns:
        print(f"Error: 'SampleID' column not found in PCA file '{pca_file}'. Columns found: {pca_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
    
    sample_id_col_sample_info = 'Sample name' 
    pop_col_sample_info = 'Population code'
    superpop_col_sample_info = 'Superpopulation code'

    if sample_id_col_sample_info not in sample_df.columns:
        print(f"Error: '{sample_id_col_sample_info}' column not in sample info file '{sample_file}'. Columns: {sample_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
    if pop_col_sample_info not in sample_df.columns:
        print(f"Error: '{pop_col_sample_info}' column not in sample info file '{sample_file}'. Columns: {sample_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
    if superpop_col_sample_info not in sample_df.columns:
        print(f"Error: '{superpop_col_sample_info}' column not in sample info file '{sample_file}'. Columns: {sample_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    pc_cols = [f'PC{i+1}' for i in range(num_pcs_to_use)]
    missing_pc_cols = [col for col in pc_cols if col not in pca_df.columns]
    if missing_pc_cols:
        print(f"Error: Missing expected PC columns in PCA file: {missing_pc_cols}.", file=sys.stderr)
        print(f"Ensure --num_pcs ({num_pcs_to_use}) matches available PCs. Available: {pca_df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)
    
    sample_df_subset = sample_df[[sample_id_col_sample_info, pop_col_sample_info, superpop_col_sample_info]].copy()
    merged_df = pd.merge(pca_df, sample_df_subset, left_on='SampleID', right_on=sample_id_col_sample_info, how='inner')

    if merged_df.empty:
        print("Error: Merging PCA data and sample info resulted in an empty DataFrame. Check sample IDs and file integrity.", file=sys.stderr)
        sys.exit(1)
        
    superpopulations = merged_df[superpop_col_sample_info].unique()
    print(f"Found {len(superpopulations)} superpopulations: {sorted(superpopulations)}\n") # Print sorted list

    # --- 3. Main Loop per Superpopulation ---
    for superpop_code in sorted(superpopulations):
        print(f"--- Processing Superpopulation: {superpop_code} ---")
        current_superpop_df = merged_df[merged_df[superpop_col_sample_info] == superpop_code].copy()
        
        if current_superpop_df.shape[0] < 2:
            print("  Skipping: Less than 2 samples in this superpopulation.\n")
            continue

        pc_data_current = current_superpop_df[pc_cols].values
        subpop_labels_current_str = current_superpop_df[pop_col_sample_info]
        unique_subpops_in_superpop, counts_subpops = np.unique(subpop_labels_current_str, return_counts=True)
        num_unique_subpops = len(unique_subpops_in_superpop)

        print(f"  Number of samples: {current_superpop_df.shape[0]}")
        print(f"  Number of unique subpopulations: {num_unique_subpops}")
        if num_unique_subpops > 0:
             print(f"  Subpopulations (counts): {dict(zip(unique_subpops_in_superpop, counts_subpops))}")

        # --- Metric Calculations ---
        subpop_labels_numerical = pd.factorize(subpop_labels_current_str)[0]

        # 1. Mutual Information (MI)
        if current_superpop_df.shape[0] > 1 and num_unique_subpops > 1:
            # sklearn's mutual_info_classif handles small n_samples internally for n_neighbors
            mi_scores_per_pc = mutual_info_classif(pc_data_current, subpop_labels_numerical,
                                                   discrete_features=False, random_state=42)
            print(f"  Mutual Information (MI) for first 3 PCs (and PC{num_pcs_to_use} if different):")
            for i in range(min(3, num_pcs_to_use)):
                print(f"    MI PC{i+1}: {mi_scores_per_pc[i]:.4f}")
            if num_pcs_to_use > 3: # Print the last PC's MI if not already printed
                 print(f"    MI PC{num_pcs_to_use}: {mi_scores_per_pc[num_pcs_to_use-1]:.4f}")
            print(f"  Average MI across all {num_pcs_to_use} PCs: {np.mean(mi_scores_per_pc):.4f}")
        else:
            print("  Mutual Information: Not applicable (requires >1 sample and >1 distinct subpopulation).")

        # 2. Average Pairwise Jensen-Shannon Divergence (JSD)
        if num_unique_subpops >= 2:
            pairwise_jsd_values = []
            for i in range(num_unique_subpops):
                for k in range(i + 1, num_unique_subpops):
                    subpop_A_name = unique_subpops_in_superpop[i]
                    subpop_B_name = unique_subpops_in_superpop[k]
                    pc_data_A = current_superpop_df[current_superpop_df[pop_col_sample_info] == subpop_A_name][pc_cols].values
                    pc_data_B = current_superpop_df[current_superpop_df[pop_col_sample_info] == subpop_B_name][pc_cols].values
                    if pc_data_A.shape[0] > 0 and pc_data_B.shape[0] > 0:
                        jsd_ab = calculate_pairwise_jsd_histograms(pc_data_A, pc_data_B, n_pcs=num_pcs_to_use)
                        if not np.isnan(jsd_ab):
                             pairwise_jsd_values.append(jsd_ab)
            if pairwise_jsd_values:
                avg_jsd = np.nanmean(pairwise_jsd_values)
                print(f"  Average Pairwise JSD: {avg_jsd:.4f}")
            else:
                print("  Average Pairwise JSD: Not enough valid pairs or data for calculation.")
        else:
            print("  Average Pairwise JSD: Not applicable (requires at least 2 subpopulations).")

        # 3. Silhouette Score
        if num_unique_subpops >= 2 and current_superpop_df.shape[0] > num_unique_subpops:
            # silhouette_score will raise ValueError if conditions not met (e.g. n_labels < 2 or n_labels > n_samples-1)
            # This will now halt the script if it occurs, as per "no try-except"
            sil_score = silhouette_score(pc_data_current, subpop_labels_current_str)
            print(f"  Silhouette Score: {sil_score:.4f}")
        else:
            print(f"  Silhouette Score: Not applicable (needs >=2 distinct subpopulations and more samples than unique subpop labels).")
            
        # 4. Contrastive Score (Revised, No Arbitrary Margin)
        if num_unique_subpops >= 2:
            violations = []
            for i in range(current_superpop_df.shape[0]):
                sample_pc_vector = pc_data_current[i, :].reshape(1, -1)
                current_sample_subpop_label = subpop_labels_current_str.iloc[i]
                same_subpop_mask = (subpop_labels_current_str == current_sample_subpop_label).values
                # Exclude the sample itself from its own group for D_intra calculation
                temp_self_mask = np.zeros(current_superpop_df.shape[0], dtype=bool)
                temp_self_mask[i] = True
                same_subpop_mask_others = same_subpop_mask & ~temp_self_mask

                pcs_own_subpop = pc_data_current[same_subpop_mask_others, :]
                d_intra = np.mean(cdist(sample_pc_vector, pcs_own_subpop)) if pcs_own_subpop.shape[0] > 0 else 0.0

                other_subpop_mask = (subpop_labels_current_str != current_sample_subpop_label).values
                pcs_other_subpops = pc_data_current[other_subpop_mask, :]
                d_inter_min = np.min(cdist(sample_pc_vector, pcs_other_subpops)) if pcs_other_subpops.shape[0] > 0 else np.inf
                
                violation = max(0, d_intra - d_inter_min) if d_inter_min != np.inf else 0.0
                violations.append(violation)
            
            avg_contrastive_score = np.mean(violations) if violations else 0.0
            print(f"  Contrastive Score (avg V_i): {avg_contrastive_score:.4f}")
        else:
            print("  Contrastive Score: Not applicable (requires at least 2 subpopulations).")
        print("--- End Superpopulation ---\n")

def main():
    parser = argparse.ArgumentParser(
        description="Calculate population structure metrics per superpopulation based on PCs and subpopulation labels.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--pca_file', type=str, help="Path to the PCA eigenvectors file (e.g., 'pca.tsv').")
    parser.add_argument('--sample_file', type=str, help="Path to sample information (e.g., 'samples.tsv').")
    parser.add_argument('--num_pcs', type=int, default=10, help="Number of PCs to use (default: 10).")
    args = parser.parse_args()

    pca_f, sample_f = args.pca_file, args.sample_file
    # Auto-find files logic (simplified, prefers specified path)
    if not pca_f and os.path.exists("chr22_eigensnp.eigensnp.pca.tsv"): pca_f = "chr22_eigensnp.eigensnp.pca.tsv"
    if not sample_f and os.path.exists("igsr_samples.tsv"): sample_f = "igsr_samples.tsv"

    if not pca_f or not os.path.exists(pca_f):
        print(f"Error: PCA file '{pca_f or 'Default PCA file'}' not found or not specified.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    if not sample_f or not os.path.exists(sample_f):
        print(f"Error: Sample info file '{sample_f or 'Default sample file'}' not found or not specified.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
            
    print(f"\nUsing PCA file: {pca_f}")
    print(f"Using sample info file: {sample_f}")
    print(f"Using first {args.num_pcs} PCs for calculations.\n")
    
    calculate_metrics_per_superpopulation(pca_f, sample_f, num_pcs_to_use=args.num_pcs)

if __name__ == "__main__":
    main()
