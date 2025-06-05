"""
e.g.:
python3 metrics.py --pca_file chr22_subset25/chr22_eigensnp.eigensnp.pca.tsv --sample_file igsr_samples.tsv

Outputs one TSV (default population_metrics_summary.tsv) with columns:

  Superpopulation
  Number_of_samples
  Number_of_subpopulations
  logreg_balanced_accuracy_cv
  Mean_multivariate_Jensen_Shannon_divergence_nats
  Median_multivariate_Jensen_Shannon_divergence_nats
  Average_silhouette
  Median_silhouette
  Mean_contrastive_violation
  Median_contrastive_violation
  HDBSCAN_adjusted_mutual_information
"""

from __future__ import annotations
import argparse
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import List # Added import

# Define the list of subpopulations to exclude specifically from Logistic Regression calculations
LOGREG_EXCLUDED_SUBPOPS: List[str] = ["ACB", "ASW", "CLM", "MXL", "PEL", "PUR"]
from sklearn.metrics import silhouette_samples, adjusted_mutual_info_score, make_scorer, balanced_accuracy_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import hdbscan

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────
def fit_gaussian_kdes(
    data_matrix: np.ndarray,
    discrete_labels: np.ndarray,
    bandwidth_rule: str | float = "scott",
) -> Tuple[Dict[str, KernelDensity], KernelDensity]:
    """Fit a Gaussian KDE for each discrete label plus a pooled KDE."""
    kde_per_label: Dict[str, KernelDensity] = {}
    for label in np.unique(discrete_labels):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_rule)
        kde.fit(data_matrix[discrete_labels == label])
        kde_per_label[label] = kde
    pooled_kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_rule).fit(
        data_matrix
    )
    return kde_per_label, pooled_kde


def monte_carlo_mutual_information(
    kde_per_label: Dict[str, KernelDensity],
    pooled_kde: KernelDensity,
    class_priors: Dict[str, float],
    samples_per_label: int = 4_000,
    master_seed: int = 1234,
) -> float:
    """Plug-in Monte-Carlo estimate of I(PCs ; label) in *nats*."""
    mutual_information = 0.0
    for serial, (label, prior) in enumerate(class_priors.items()):
        seed = (master_seed + 7919 * serial) & 0xFFFFFFFF
        synthetic_points = kde_per_label[label].sample(
            samples_per_label, random_state=seed
        )
        log_density_label = kde_per_label[label].score_samples(synthetic_points)
        log_density_pool = pooled_kde.score_samples(synthetic_points)
        mutual_information += prior * np.mean(log_density_label - log_density_pool)
    return float(mutual_information)


def monte_carlo_jsd(
    kde_a: KernelDensity,
    kde_b: KernelDensity,
    mc_samples: int = 4_000,
    seed: int = 42,
) -> float:
    """Monte-Carlo Jensen–Shannon divergence (nats) between two multivariate KDEs."""
    half = mc_samples // 2
    sample_a = kde_a.sample(half, random_state=seed & 0xFFFFFFFF)
    sample_b = kde_b.sample(mc_samples - half, random_state=(seed + 1) & 0xFFFFFFFF)
    all_samples = np.vstack([sample_a, sample_b])

    log_p_a = kde_a.score_samples(all_samples)
    log_p_b = kde_b.score_samples(all_samples)
    log_mix = np.logaddexp(log_p_a, log_p_b) - np.log(2.0)

    return float(0.5 * np.mean(log_p_a - log_mix) + 0.5 * np.mean(log_p_b - log_mix))


def contrastive_violation_statistics(
    data_matrix: np.ndarray, discrete_labels: np.ndarray
) -> Tuple[float, float]:
    """
    For every sample:
      violation = max(0,
                      mean_distance_to_own_subpop − min_distance_to_any_other_subpop)
    """
    distance_matrix = cdist(data_matrix, data_matrix, metric="euclidean")
    violations: List[float] = []

    for idx in range(data_matrix.shape[0]):
        same_mask = discrete_labels == discrete_labels[idx]
        same_mask[idx] = False  # exclude self
        intra_distance = distance_matrix[idx, same_mask].mean() if same_mask.any() else 0.0

        other_mask = ~same_mask
        inter_min_distance = (
            distance_matrix[idx, other_mask].min() if other_mask.any() else np.inf
        )
        violations.append(max(0.0, intra_distance - inter_min_distance)
                          if np.isfinite(inter_min_distance) else 0.0)

    violations = np.asarray(violations)
    return float(violations.mean()), float(np.median(violations))


def best_hdbscan_adjusted_mutual_information(
    data_matrix: np.ndarray,
    true_labels: np.ndarray,
    search_fracs: Tuple[float, ...] = (0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30),
) -> float:
    """
    Try several (min_cluster_size, min_samples) pairs and return the highest
    Adjusted MI between HDBSCAN clusters and true subpopulation labels.
    If no parameter set yields ≥2 clusters, return 0.0.
    """
    n = len(true_labels)
    best_ami = 0.0  # never NaN

    for frac in search_fracs:
        min_cluster_size = max(2, int(round(frac * n)))
        for min_samples in {1, min_cluster_size // 2, min_cluster_size}:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="leaf",
            )
            predicted_labels = clusterer.fit_predict(data_matrix)

            # Ignore noise points (-1) for AMI
            core_mask = predicted_labels != -1
            if core_mask.sum() < 2:
                continue  # too few points left
            core_predicted = predicted_labels[core_mask]
            core_true = true_labels[core_mask]
            if len(np.unique(core_predicted)) < 2:
                continue  # fewer than 2 clusters => AMI undefined/0

            ami_value = adjusted_mutual_info_score(
                core_true, core_predicted, average_method="arithmetic"
            )
            best_ami = max(best_ami, ami_value)

    return float(best_ami)  # could be 0.0 if no real clustering emerged


def logistic_regression_balanced_accuracy(
    pc_matrix: np.ndarray,
    subpopulation_labels: np.ndarray,
    n_cv_splits: int = 5,
    random_seed: int = 42,
) -> float:
    """
    Calculate k-fold cross-validated balanced accuracy of an unregularized
    logistic regression model predicting subpopulation labels from PCs.
    """
    unique_classes = np.unique(subpopulation_labels)
    if len(unique_classes) < 2:
        # Not enough classes for classification
        return np.nan

    class_counts = {label: count for label, count in zip(*np.unique(subpopulation_labels, return_counts=True))}
    if pc_matrix.shape[0] < n_cv_splits or any(count < n_cv_splits for count in class_counts.values()):
        return np.nan

    # Encode string labels to numeric for scikit-learn
    le = LabelEncoder()
    numeric_labels = le.fit_transform(subpopulation_labels)

    # Logistic Regression with L2 regularization
    # L2 regularization is generally robust and helps prevent overfitting.
    # The regularization strength is controlled by C (default C=1.0).
    # For more fine-grained control, C can be tuned using cross-validation
    # (e.g., with LogisticRegressionCV or by wrapping this in a GridSearchCV).
    model = LogisticRegression(
        penalty='l2',  # Apply L2 regularization
        solver='lbfgs', # 'lbfgs' supports L2 regularization
        multi_class='auto', # Automatically handles multi-class
        random_state=random_seed,
        max_iter=300, # Increased iterations
        C=1.0 # Default inverse of regularization strength
    )

    cv_strategy = StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=random_seed)
    balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)

    try:
        scores = cross_val_score(model, pc_matrix, numeric_labels, cv=cv_strategy, scoring=balanced_accuracy_scorer)
        return float(np.mean(scores))
    except ValueError as e:
        return np.nan

# ────────────────────────────────────────────────────────────────────────────────
# Core routine
# ────────────────────────────────────────────────────────────────────────────────
def compute_metrics_for_superpopulations(
    pca_file_path: str,
    sample_file_path: str,
    number_of_pcs: int,
    monte_carlo_samples: int,
    output_tsv_path: str,
) -> None:
    # 1 ─── Load and join PCA with sample metadata
    pca_table = pd.read_csv(pca_file_path, sep=r"\s+")
    sample_table = pd.read_csv(sample_file_path, sep="\t")

    required_pca_columns = ["SampleID"] + [f"PC{i+1}" for i in range(number_of_pcs)]
    required_sample_columns = ["Sample name", "Population code", "Superpopulation code"]

    for col in required_pca_columns:
        if col not in pca_table.columns:
            sys.exit(f"Column '{col}' missing in PCA file.")
    for col in required_sample_columns:
        if col not in sample_table.columns:
            sys.exit(f"Column '{col}' missing in sample file.")

    merged = pca_table.merge(
        sample_table[required_sample_columns],
        left_on="SampleID",
        right_on="Sample name",
        how="inner",
    )
    if merged.empty:
        sys.exit("No overlapping Sample IDs between PCA and sample tables.")

    pc_columns = [f"PC{i+1}" for i in range(number_of_pcs)]
    merged[pc_columns] = StandardScaler().fit_transform(merged[pc_columns])

    # TSV header
    column_names = [
        "Superpopulation",
        "Number_of_samples",
        "Number_of_subpopulations",
        "LogReg_Balanced_Accuracy_CV",
        "LogReg_Normalized_Accuracy_CV",
        "Mean_multivariate_Jensen_Shannon_divergence_nats",
        "Median_multivariate_Jensen_Shannon_divergence_nats",
        "Average_silhouette",
        "Median_silhouette",
        "Mean_contrastive_violation",
        "Median_contrastive_violation",
        "HDBSCAN_adjusted_mutual_information",
    ]
    tsv_lines: List[str] = ["\t".join(column_names)]

    # 2 ─── Iterate over super-populations
    for superpopulation_code in sorted(merged["Superpopulation code"].unique()):
        subset_orig = merged[merged["Superpopulation code"] == superpopulation_code]
        
        # Original data for most metrics and reporting
        pc_matrix_orig = subset_orig[pc_columns].to_numpy()
        subpopulation_labels_orig = subset_orig["Population code"].to_numpy()
        unique_subpops_orig, _ = np.unique(subpopulation_labels_orig, return_counts=True)
        num_subpops_for_reporting = len(unique_subpops_orig)

        # KDEs for MI & JSD (using original, unfiltered data for the superpopulation)
        # Ensure there are samples to fit KDEs
        kde_dict: Dict[str, KernelDensity] = {}
        pooled_kde: KernelDensity | None = None
        if pc_matrix_orig.shape[0] > 0 and len(unique_subpops_orig) > 0:
             kde_dict, pooled_kde = fit_gaussian_kdes(pc_matrix_orig, subpopulation_labels_orig)
        else: # Not enough data to fit KDEs
            # Handle cases where pooled_kde might be expected by other functions if they were to be called
            # For JSD, this will result in nan as jsd_values will be empty
            pass


        # --- Logistic Regression Specific Data Preparation & Calculation ---
        # Filter out excluded subpopulations for Logistic Regression
        logreg_filter_mask = ~subset_orig["Population code"].isin(LOGREG_EXCLUDED_SUBPOPS)
        subset_logreg = subset_orig[logreg_filter_mask]

        pc_matrix_logreg = subset_logreg[pc_columns].to_numpy()
        subpopulation_labels_logreg = subset_logreg["Population code"].to_numpy()

        logreg_balanced_accuracy_cv_raw = np.nan
        logreg_normalized_accuracy_cv = np.nan

        num_unique_subpops_for_logreg_norm = 0
        if pc_matrix_logreg.shape[0] > 0: # Check if any samples remain after filtering
            unique_subpops_logreg_arr = np.unique(subpopulation_labels_logreg)
            num_unique_subpops_for_logreg_norm = len(unique_subpops_logreg_arr)

        # Proceed with LogReg calculation only if there are enough classes and samples
        if num_unique_subpops_for_logreg_norm >= 2:
            logreg_balanced_accuracy_cv_raw = logistic_regression_balanced_accuracy(
                pc_matrix_logreg, subpopulation_labels_logreg
            )
            # Normalized Logistic Regression Balanced Accuracy (using logreg specific counts)
            if pd.notna(logreg_balanced_accuracy_cv_raw): # Check if calculation was successful
                chance_performance_logreg = 1.0 / num_unique_subpops_for_logreg_norm
                if logreg_balanced_accuracy_cv_raw >= chance_performance_logreg:
                    logreg_normalized_accuracy_cv = (
                        logreg_balanced_accuracy_cv_raw - chance_performance_logreg
                    ) / (1.0 - chance_performance_logreg)
                else:
                    logreg_normalized_accuracy_cv = 0.0
            # If logreg_balanced_accuracy_cv_raw is NaN (e.g. from logistic_regression_balanced_accuracy), normalized also remains NaN.
        # If num_unique_subpops_for_logreg_norm < 2, LogReg metrics remain np.nan (as initialized)
        # --- End of Logistic Regression Specific Calculation ---

        # Pairwise JSDs (using original, unfiltered data; unique_subpops_orig for iteration)
        jsd_values: List[float] = []
        if pooled_kde is not None: # Check if KDEs were successfully created
            for i, label_a in enumerate(unique_subpops_orig):
                for label_b in unique_subpops_orig[i + 1 :]:
                    # Ensure KDEs for these labels exist in the dictionary
                    if label_a in kde_dict and label_b in kde_dict:
                        jsd_values.append(
                            monte_carlo_jsd(
                                kde_dict[label_a],
                                kde_dict[label_b],
                                mc_samples=monte_carlo_samples,
                                seed=17 + len(jsd_values),
                            )
                        )
        mean_jsd = float(np.mean(jsd_values)) if jsd_values else np.nan
        median_jsd = float(np.median(jsd_values)) if jsd_values else np.nan

        # Silhouette statistics (using original, unfiltered data)
        average_silhouette = np.nan
        median_silhouette = np.nan
        # silhouette_samples requires at least 2 labels and more samples than labels.
        if num_subpops_for_reporting >= 2 and pc_matrix_orig.shape[0] > num_subpops_for_reporting :
            try:
                silhouette_values = silhouette_samples(pc_matrix_orig, subpopulation_labels_orig)
                average_silhouette = float(silhouette_values.mean())
                median_silhouette = float(np.median(silhouette_values))
            except ValueError: # handles cases where silhouette cannot be computed e.g. all points in one cluster after some internal processing by silhouette_samples
                pass


        # Contrastive violation statistics (using original, unfiltered data)
        mean_contrastive, median_contrastive = contrastive_violation_statistics(
            pc_matrix_orig, subpopulation_labels_orig
        )

        # Adaptive HDBSCAN AMI (using original, unfiltered data)
        hdbscan_ami = best_hdbscan_adjusted_mutual_information(
            pc_matrix_orig, subpopulation_labels_orig
        )

        # TSV row
        row_values = [
            superpopulation_code,
            len(subset_orig), # Number of samples before any LogReg specific filtering
            num_subpops_for_reporting, # Number of subpopulations before LogReg specific filtering
            logreg_balanced_accuracy_cv_raw, # Calculated with filtered data
            logreg_normalized_accuracy_cv, # Calculated with filtered data and correct N_classes
            mean_jsd,
            median_jsd,
            average_silhouette,
            median_silhouette,
            mean_contrastive,
            median_contrastive,
            hdbscan_ami,
        ]

        formatted_values = []
        for val in row_values:
            if isinstance(val, float):
                if np.isnan(val):
                    formatted_values.append("NaN")
                else:
                    formatted_values.append(f"{val:.6f}")
            else:
                formatted_values.append(str(val))
        tsv_lines.append("\t".join(formatted_values))

    # 3 ─── Write TSV
    with open(output_tsv_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(tsv_lines))
    print(
        f"✔ Finished: wrote {len(tsv_lines) - 1} rows to {output_tsv_path}",
        flush=True,
    )


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute multivariate population-structure metrics "
        "for each super-population.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pca_file", required=True, help="Path to PCA TSV file.")
    parser.add_argument("--sample_file", required=True, help="Path to sample TSV file.")
    parser.add_argument(
        "--number_of_pcs", type=int, default=10, help="Number of PCs (columns) to use."
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=4_000,
        help="Monte-Carlo samples per KDE for JSD estimates.",
    )
    parser.add_argument(
        "--output_tsv",
        default="population_metrics_summary.tsv",
        help="Destination TSV file.",
    )
    args = parser.parse_args()

    compute_metrics_for_superpopulations(
        pca_file_path=args.pca_file,
        sample_file_path=args.sample_file,
        number_of_pcs=args.number_of_pcs,
        monte_carlo_samples=args.mc_samples,
        output_tsv_path=args.output_tsv,
    )


if __name__ == "__main__":
    main()
