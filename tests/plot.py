import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys # For sys.exit()
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# --- Configuration ---
# files
MIN_SAMPLES_FOR_ELLIPSE = 3 # Minimum samples in a superpopulation to draw an ellipse
ELLIPSE_STD_DEV = 2

# --- Helper Function for Robust Column Checking ---
def check_columns(df, required_columns, df_name):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"ERROR: DataFrame '{df_name}' is missing required columns: {', '.join(missing_cols)}")
        print(f"       Available columns: {', '.join(df.columns)}")
        sys.exit(f"Exiting due to missing columns in {df_name}.")

# --- Helper function to draw confidence ellipse ---
def plot_confidence_ellipse(ax, x, y, n_std=2.0, facecolor='none', edgecolor='k', **kwargs):
    """
    Creates a plot of the covariance confidence ellipse of *x* and *y*.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse on.
    x, y : array-like, shape (n, )
        Input data.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    if x.size < 2 : # Cannot compute covariance for less than 2 points
        # print(f"Warning: Cannot compute ellipse for less than 2 points. Skipping.")
        return None

    cov = np.cov(x, y)
    # Pearson correlation coefficient
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# --- 1. Read Data Files ---
print("Reading data files...")
try:
    eigenvalues_df = pd.read_csv(EIGENVALUES_FILE, sep='\t')
    check_columns(eigenvalues_df, ['PC', 'Eigenvalue'], 'Eigenvalues')
    print(f"Eigenvalues shape: {eigenvalues_df.shape}")
    print(f"Eigenvalues head:\n{eigenvalues_df.head()}")

    pca_df = pd.read_csv(PCA_FILE, sep='\t')
    pc_cols_expected = [f'PC{i}' for i in range(1, len(eigenvalues_df) + 1)]
    check_columns(pca_df, ['SampleID'] + pc_cols_expected[:2], 'PCA Data')
    pca_df['SampleID'] = pca_df['SampleID'].astype(str).str.strip()
    print(f"\nPCA data shape: {pca_df.shape}")

    loadings_df = pd.read_csv(LOADINGS_FILE, sep='\t')
    check_columns(loadings_df, ['Pos', 'PC1_loading', 'PC2_loading'], 'Loadings Data')
    print(f"\nLoadings shape: {loadings_df.shape}")

    pop_df = pd.read_csv(POPULATION_FILE, sep='\t', header=0)
    check_columns(pop_df, ['Sample name', 'Population code', 'Population name', 'Superpopulation code'], 'Population Data')
    pop_df.rename(columns={
        'Sample name': 'SampleID',
        'Population code': 'Population',
        'Population name': 'PopulationNameLong',
        'Superpopulation code': 'Superpopulation' # This is the key for ellipses
    }, inplace=True)
    pop_df['SampleID'] = pop_df['SampleID'].astype(str).str.strip()
    print(f"\nPopulation data shape: {pop_df.shape}")
    print(f"Superpopulation codes (first 5): {pop_df['Superpopulation'].head().tolist()}")

except FileNotFoundError as e:
    print(f"ERROR: File not found - {e.filename}")
    sys.exit("Exiting due to missing file.")
except Exception as e:
    print(f"An unexpected error occurred during file reading: {e}")
    sys.exit("Exiting due to file reading error.")

# --- 2. Merge PCA Data with Population Labels ---
print("\nMerging PCA and population data...")
pca_merged = pca_df.merge(
    pop_df[['SampleID', 'Population', 'PopulationNameLong', 'Superpopulation']],
    on='SampleID',
    how='left'
)
print(f"Merged data shape: {pca_merged.shape}")
unannotated_count = pca_merged['Population'].isna().sum()
if unannotated_count > 0:
    print(f"WARNING: {unannotated_count} PCA samples could NOT be annotated.")
else:
    print("All PCA samples successfully found in the population file and annotated.")

# --- 3. Visualization Setup ---
fig, axes = plt.subplots(1, 3, figsize=(24, 7.5)) # Adjusted figsize for better legend space
fig.suptitle('PCA Analysis Visualization', fontsize=18, fontweight='bold')

# --- 3.1 Subplot 1: Eigenvalues (Scree Plot) ---
num_eigenvalues = len(eigenvalues_df)
num_samples = len(pca_df)
scree_title = f'Scree Plot (Top {num_eigenvalues} Eigenvalues)'
y_label_cumulative = f'Cumulative % Variance (of Top {num_eigenvalues} PCs)'
if num_eigenvalues < (num_samples -1) and num_eigenvalues > 0 :
    scree_title += f'\nNote: Only top {num_eigenvalues} of ~{num_samples-1} possible PCs shown.'

axes[0].bar(eigenvalues_df['PC'], eigenvalues_df['Eigenvalue'],
            color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Eigenvalue', fontsize=12, color='steelblue')
axes[0].set_title(scree_title, fontsize=14)
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].tick_params(axis='y', labelcolor='steelblue')
if num_eigenvalues > 0:
    axes[0].set_xticks(eigenvalues_df['PC'])
    axes[0].set_xlim(0.5, num_eigenvalues + 0.5)

ax2_scree = axes[0].twinx()
total_eigenvalue_sum_reported = eigenvalues_df['Eigenvalue'].sum()
if total_eigenvalue_sum_reported > 0:
    var_explained_individual = (eigenvalues_df['Eigenvalue'] / total_eigenvalue_sum_reported) * 100
    cumsum_var_explained = var_explained_individual.cumsum()
    ax2_scree.plot(eigenvalues_df['PC'], cumsum_var_explained,
                   'ro-', alpha=0.7, markersize=5, linewidth=2, label='Cumulative % Variance')
    ax2_scree.set_ylabel(y_label_cumulative, color='red', fontsize=12)
    ax2_scree.tick_params(axis='y', labelcolor='red')
    ax2_scree.set_ylim(0, 105 if cumsum_var_explained.iloc[-1] > 99 else cumsum_var_explained.iloc[-1] * 1.05)
else:
    var_explained_individual = pd.Series([0] * len(eigenvalues_df))
axes[0].set_ylim(bottom=0)

# --- 3.2 Subplot 2: PCA Scatter Plot (PC1 vs PC2) ---
ax_pca = axes[1] # Main PCA plot axis

# Define colors for individual populations (points)
unique_short_codes = sorted(pca_merged['Population'].dropna().unique())
pop_point_color_map = {}
if unique_short_codes:
    num_unique_pops = len(unique_short_codes)
    # Using plt.get_cmap to address deprecation
    palette_points = plt.get_cmap('tab20' if num_unique_pops <= 20 else 'tab20b', num_unique_pops if num_unique_pops <=20 else 20)
    colors_for_points = [palette_points(i % palette_points.N) for i in range(num_unique_pops)]
    pop_point_color_map = dict(zip(unique_short_codes, colors_for_points))

# Define colors for superpopulation ellipses
unique_superpopulations = sorted(pca_merged['Superpopulation'].dropna().unique())
superpop_ellipse_color_map = {}
if unique_superpopulations:
    num_unique_superpops = len(unique_superpopulations)
    palette_ellipses = plt.get_cmap('Set2', num_unique_superpops if num_unique_superpops <=8 else 8) # Set2 has 8 distinct colors
    colors_for_ellipses = [palette_ellipses(i % palette_ellipses.N) for i in range(num_unique_superpops)]
    superpop_ellipse_color_map = dict(zip(unique_superpopulations, colors_for_ellipses))

# Plot superpopulation ellipses first (low zorder)
superpop_legend_handles = []
for superpop_code, group_df in pca_merged.groupby('Superpopulation'):
    if pd.notna(superpop_code) and len(group_df) >= MIN_SAMPLES_FOR_ELLIPSE:
        pc1_vals = group_df['PC1']
        pc2_vals = group_df['PC2']
        ellipse_color = superpop_ellipse_color_map.get(superpop_code, 'gray') # Fallback color
        
        plot_confidence_ellipse(ax_pca, pc1_vals, pc2_vals, n_std=ELLIPSE_STD_DEV,
                                facecolor=ellipse_color, alpha=0.2, zorder=1, edgecolor='none')
        # For superpopulation legend
        superpop_legend_handles.append(
            plt.Line2D([0], [0], marker='s', color='w', markersize=10,
                       markerfacecolor=ellipse_color, label=superpop_code, alpha=0.5) # Use square for legend
        )

# Plot individual sample points (higher zorder)
point_color_array = [pop_point_color_map.get(pop, 'dimgray') for pop in pca_merged['Population']]
ax_pca.scatter(pca_merged['PC1'], pca_merged['PC2'],
               c=point_color_array, alpha=0.9, s=60,
               edgecolors='black', linewidth=0.5, zorder=5)

pc1_var_label = f"PC1 ({var_explained_individual.iloc[0]:.2f}%)" if len(var_explained_individual) > 0 else "PC1"
pc2_var_label = f"PC2 ({var_explained_individual.iloc[1]:.2f}%)" if len(var_explained_individual) > 1 else "PC2"
ax_pca.set_xlabel(pc1_var_label, fontsize=12)
ax_pca.set_ylabel(pc2_var_label, fontsize=12)
ax_pca.set_title('PCA Score Plot (PC1 vs PC2)', fontsize=14)
ax_pca.grid(True, linestyle='--', alpha=0.6)

# Create legend for individual populations (points)
pop_legend_elements = []
short_to_long_name_map = pca_merged.dropna(subset=['Population', 'PopulationNameLong']) \
                                   .set_index('Population')['PopulationNameLong'] \
                                   .drop_duplicates().to_dict()
for short_code in unique_short_codes:
    if short_code in pop_point_color_map:
        long_name = short_to_long_name_map.get(short_code, short_code)
        label_text = f"{long_name} ({short_code})" if long_name != short_code else short_code
        pop_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                              markerfacecolor=pop_point_color_map[short_code],
                                              markersize=8, label=label_text, markeredgecolor='black'))
if pca_merged['Population'].isna().any():
    pop_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor='dimgray', markersize=8,
                                          label='Unknown/NA', markeredgecolor='black'))

# Add population legend (main legend)
if pop_legend_elements:
    legend1 = ax_pca.legend(handles=pop_legend_elements, title='Populations (Points)',
                            bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, title_fontsize=10)
    ax_pca.add_artist(legend1) # Important to add it back if we create another legend

# Add superpopulation legend (ellipses)
if superpop_legend_handles:
    ax_pca.legend(handles=superpop_legend_handles, title='Superpopulations (Ellipses)',
                  bbox_to_anchor=(1.02, 0.4), loc='center left', fontsize=8, title_fontsize=10) # Adjust position


# --- 3.3 Subplot 3: Loadings Plot ---
ax_loadings = axes[2]
loadings_sorted = loadings_df.sort_values('Pos')
ax_loadings.scatter(loadings_sorted['Pos'], loadings_sorted['PC1_loading'],
                    alpha=0.6, s=20, label='PC1 Loadings', color='crimson')
ax_loadings.scatter(loadings_sorted['Pos'], loadings_sorted['PC2_loading'],
                    alpha=0.6, s=20, label='PC2 Loadings', color='mediumblue')
ax_loadings.set_xlabel('Genomic Position (Chromosome-wide)', fontsize=12)
ax_loadings.set_ylabel('Loading Value', fontsize=12)
ax_loadings.set_title('SNP Loadings for PC1 & PC2', fontsize=14)
ax_loadings.legend(fontsize=10)
ax_loadings.grid(True, linestyle='--', alpha=0.6)
ax_loadings.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# --- 4. Adjust Layout & Save ---
plt.tight_layout(rect=[0, 0.02, 0.95, 0.96]) # Adjust right margin for legends
plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved as '{OUTPUT_PLOT_FILENAME}'")

# --- 5. Display Basic Statistics ---
print("\n--- Data Summary ---")
# ... (summary statistics code remains largely the same as previous version)
print(f"Number of Principal Components in eigenvalues file: {num_eigenvalues}")
print(f"Number of Samples in PCA data: {num_samples}")
print(f"Number of Variants in Loadings file: {len(loadings_df)}")

if total_eigenvalue_sum_reported > 0 and len(var_explained_individual) >= 2:
    var_explained_pc1_pc2_sum = var_explained_individual.iloc[0] + var_explained_individual.iloc[1]
    print(f"Variance Explained by PC1+PC2 (relative to top {num_eigenvalues} PCs): {var_explained_pc1_pc2_sum:.2f}%")

print(f"\nPopulations found in the PCA dataset ({len(unique_short_codes)} unique):")
if unique_short_codes:
    value_counts_short = pca_merged['Population'].value_counts()
    summary_list_pop = []
    for short_code_iter, count in value_counts_short.items():
        long_name_iter = short_to_long_name_map.get(short_code_iter, "N/A")
        summary_list_pop.append({
            'Population Code': short_code_iter,
            'Population Name': long_name_iter,
            'Count': count
        })
    population_counts_df = pd.DataFrame(summary_list_pop)
    print("\nPopulation counts in PCA data (annotated samples):")
    print(population_counts_df.to_string(index=False))

print(f"\nSuperpopulations found in the PCA dataset ({len(unique_superpopulations)} unique):")
if unique_superpopulations:
    print(f"  {', '.join(unique_superpopulations)}")
    print("\nSuperpopulation counts in PCA data:")
    print(pca_merged['Superpopulation'].value_counts().to_string())


if unannotated_count > 0:
    print(f"\nSamples without population labels: {unannotated_count} out of {num_samples}")

print("\nScript finished.")
