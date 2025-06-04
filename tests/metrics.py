"""
e.g.:
python3 metrics.py --pca_file chr22_subset25/chr22_eigensnp.eigensnp.pca.tsv --sample_file igsr_samples.tsv

Outputs one TSV (default population_metrics_summary.tsv) with columns:

  Superpopulation
  Number_of_samples
  Number_of_subpopulations
  Mutual_information_nats
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
from sklearn.metrics import silhouette_samples, adjusted_mutual_info_score
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
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
        "Mutual_information_nats",
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
        subset = merged[merged["Superpopulation code"] == superpopulation_code]
        pc_matrix = subset[pc_columns].to_numpy()
        subpopulation_labels = subset["Population code"].to_numpy()

        # KDEs for MI & JSD
        kde_dict, pooled_kde = fit_gaussian_kdes(pc_matrix, subpopulation_labels)

        # Mutual information
        unique_subpops, counts = np.unique(subpopulation_labels, return_counts=True)
        class_priors = {
            label: cnt / len(subset) for label, cnt in zip(unique_subpops, counts)
        }
        mutual_information_nats = monte_carlo_mutual_information(
            kde_dict, pooled_kde, class_priors, samples_per_label=monte_carlo_samples
        )

        # Pairwise JSDs
        jsd_values: List[float] = []
        for i, label_a in enumerate(unique_subpops):
            for label_b in unique_subpops[i + 1 :]:
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

        # Silhouette statistics
        silhouette_values = silhouette_samples(pc_matrix, subpopulation_labels)
        average_silhouette = float(silhouette_values.mean())
        median_silhouette = float(np.median(silhouette_values))

        # Contrastive violation statistics
        mean_contrastive, median_contrastive = contrastive_violation_statistics(
            pc_matrix, subpopulation_labels
        )

        # Adaptive HDBSCAN AMI
        hdbscan_ami = best_hdbscan_adjusted_mutual_information(
            pc_matrix, subpopulation_labels
        )

        # TSV row
        tsv_lines.append(
            "\t".join(
                map(
                    str,
                    [
                        superpopulation_code,
                        len(subset),
                        len(unique_subpops),
                        f"{mutual_information_nats:.6f}",
                        f"{mean_jsd:.6f}",
                        f"{median_jsd:.6f}",
                        f"{average_silhouette:.6f}",
                        f"{median_silhouette:.6f}",
                        f"{mean_contrastive:.6f}",
                        f"{median_contrastive:.6f}",
                        f"{hdbscan_ami:.6f}",
                    ],
                )
            )
        )

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
        help="Monte-Carlo samples per KDE for MI and JSD estimates.",
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
