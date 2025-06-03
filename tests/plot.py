import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import glob
import seaborn as sns
import math
from functools import lru_cache
import umap

# --- Configuration ---
EIGENVALUES_SUFFIX = ".eigenvalues.tsv"
PCA_SUFFIXES_TO_TRY = [".eigensnp.pca.tsv", ".pca.tsv"]
LOADINGS_SUFFIXES_TO_TRY = [".eigensnp.loadings.tsv", ".loadings.tsv"]
POPULATION_FILE_NAME = "igsr_samples.tsv"
OUTPUT_PLOT_BASENAME = "pca.png"

MIN_SAMPLES_FOR_KDE = 5
POINT_ALPHA = 0.50 #  transparent for KDE line visibility
KDE_LINE_ALPHA = 0.8 # Alpha for KDE contour lines
KDE_LEVELS = 5
KDE_THRESH = 0.03 #  threshold for  KDE
DEFAULT_MARKER_SIZE = 20 # Smaller markers

SUPERPOP_COLOR_PROFILES = {
    # Color Name: (Base Hex, Hue Center (0-1), Max Hue Span for Subpops)
    'EUR': ('#0077B6', 0.58, 0.12),  # Blue,  span
    'SAS': ('#2ECC71', 0.41, 0.12),  # Green,  span
    'AFR': ('#E74C3C', 0.015, 0.06), # Red,   span
    'EAS': ('#9B59B6', 0.78, 0.12),  # Purple,  span
    'AMR': ('#FFEB3B', 0.155, 0.07), #  Yellow,  span
    'OTH': ('#95A5A6', 0.53, 0.00)   # Grey
}

# Predefined S/V targets for subpopulation color variation
# (Saturation, Value) pairs to cycle through for distinctness
SV_TARGETS = [
    (0.95, 0.90), (0.80, 0.75), (0.65, 0.95), (0.90, 0.60), (0.75, 0.85),
    (1.00, 0.70), (0.60, 0.80), (0.85, 1.00), (0.70, 0.65), (0.90, 0.55) 
]


# --- Helper Functions ---
def find_file_or_exit(search_paths, filename_or_suffixes, description):
    if isinstance(filename_or_suffixes, str) and not filename_or_suffixes.startswith('.'):
        for path_dir in search_paths:
            file_path = os.path.join(path_dir, filename_or_suffixes)
            if os.path.exists(file_path):
                print(f"Found {description}: {file_path}")
                return file_path
        print(f"ERROR: Cannot find {description} '{filename_or_suffixes}' in {search_paths}.")
        sys.exit(1)
    input_dir = search_paths[0]
    suffixes = [filename_or_suffixes] if isinstance(filename_or_suffixes, str) else filename_or_suffixes
    for suffix in suffixes:
        found = glob.glob(os.path.join(input_dir, f"*{suffix}"))
        if found:
            if len(found) > 1: print(f"WARNING: Multiple {description} for '{suffix}'. Using: {found[0]}")
            else: print(f"Found {description}: {found[0]}")
            return found[0]
    print(f"ERROR: Cannot find {description} in '{input_dir}' with suffixes: {suffixes}.")
    sys.exit(1)

def read_data_file(file_path, expected_cols, description, sep=r'\s+'):
    try:
        df = pd.read_csv(file_path, sep=sep, comment='#', low_memory=False)
        if df.empty: print(f"ERROR: {description} file '{file_path}' is empty."); sys.exit(1)
    except Exception as e: print(f"ERROR: Reading {description} '{file_path}'. {e}"); sys.exit(1)
    missing = [col for col in expected_cols if col not in df.columns]
    if missing: print(f"ERROR: {description} '{file_path}' missing columns: {missing}. Has: {list(df.columns)}"); sys.exit(1)
    return df

@lru_cache(maxsize=512)
def get_population_color(superpop_code, num_pops_in_super, pop_idx_in_super):
    profile = SUPERPOP_COLOR_PROFILES.get(superpop_code, SUPERPOP_COLOR_PROFILES['OTH'])
    base_hex, base_hue_center, max_hue_span_total = profile
    
    new_h = base_hue_center
    target_s, target_v = SV_TARGETS[pop_idx_in_super % len(SV_TARGETS)] # Cycle through S/V targets

    if num_pops_in_super > 1 and max_hue_span_total > 0:
        relative_pos_hue = (pop_idx_in_super / (num_pops_in_super - 1)) - 0.5 if num_pops_in_super > 1 else 0.0
        hue_shift = relative_pos_hue * max_hue_span_total
        new_h = (base_hue_center + hue_shift + 1.0) % 1.0
    elif num_pops_in_super == 1: # Single pop, use first S/V target or slightly desaturated base
        target_s, target_v = SV_TARGETS[0] 
        
    if superpop_code == 'OTH': 
        new_h = base_hue_center 
        target_s, target_v = 0.1, 0.6 # Force OTH to be grey

    return mcolors.hsv_to_rgb((new_h, np.clip(target_s,0.1,1.0), np.clip(target_v,0.3,1.0)))

def plot_scatter_with_kde(ax, data_df, x_col, y_col, x_var, y_var,
                             pop_color_map, plot_title_prefix, **plot_params):
    for superpop, group in data_df.groupby('Superpopulation'):
        if not pd.notna(superpop) or len(group) < plot_params['min_samples_kde']:
            continue
        x_vals, y_vals = group[x_col].dropna(), group[y_col].dropna()
        if len(x_vals) < 2 or len(y_vals) < 2: continue
        base_color_hex = SUPERPOP_COLOR_PROFILES.get(superpop, SUPERPOP_COLOR_PROFILES['OTH'])[0]
        try:
            sns.kdeplot(x=x_vals, y=y_vals, ax=ax, levels=plot_params['kde_levels'],
                        thresh=plot_params['kde_thresh'], 
                        alpha=plot_params['kde_line_alpha'], 
                        color=base_color_hex, # KDE lines colored by superpop
                        linewidths=0.9, 
                        zorder=6, warn_singular=False, fill=False) # KDE lines on top
        except Exception as e: print(f"Warning: KDE plot failed for {superpop} on {x_col} vs {y_col}. {e}")

    fallback_rgb_tuple = get_population_color('OTH', 1, 0)
    colors_list = [pop_color_map.get(pop_code, fallback_rgb_tuple) for pop_code in data_df['Population']]
    
    ax.scatter(data_df[x_col], data_df[y_col], c=colors_list, alpha=plot_params['point_alpha'], 
               s=plot_params['marker_size'], edgecolors='none', zorder=5, rasterized=True) # No edgecolors for cleaner look

    var_x_str = f" ({x_var:.2f}%)" if x_var is not None else ""
    var_y_str = f" ({y_var:.2f}%)" if y_var is not None else ""
    ax.set_xlabel(f"{x_col}{var_x_str}", fontsize=10); ax.set_ylabel(f"{y_col}{var_y_str}", fontsize=10)
    ax.set_title(f'{plot_title_prefix}: {x_col} vs {y_col}', fontsize=12, fontweight='medium')
    ax.axhline(0, color='darkgrey', linewidth=0.5, linestyle=':'); ax.axvline(0, color='darkgrey', linewidth=0.5, linestyle=':')

def plot_color_scheme_guide(ax, population_color_map, all_pop_long_names, pca_merged_for_structure):
    ax.axis('off')
    ax.set_title("Color Scheme Guide", fontsize=12, fontweight='bold', pad=15)

    current_y = 0.98
    line_height_super = 0.045
    line_height_sub = 0.030
    indent_sub = 0.05
    patch_x = 0.02
    patch_width = 0.12
    text_x_super = patch_x + patch_width + 0.03
    text_x_sub = text_x_super + indent_sub
    
    # consistent superpopulation order
    # Consider only superpopulations that actually have populations listed in all_pop_long_names or NaN groups
    relevant_superpops = set()
    for pop_code in all_pop_long_names.keys():
        matches = pca_merged_for_structure[pca_merged_for_structure['Population'] == pop_code]['Superpopulation'].dropna()
        if not matches.empty:
            relevant_superpops.add(matches.iloc[0])
        elif pca_merged_for_structure[pca_merged_for_structure['Population'] == pop_code]['Superpopulation'].isna().any():
             # This case is for populations that don't have a superpop, they'll be grouped under OTH later
             pass 
    
    # Add OTH if there are any populations that will fall into it
    if any(pop_code not in population_color_map or 
           pca_merged_for_structure[pca_merged_for_structure['Population'] == pop_code]['Superpopulation'].isna().any() 
           for pop_code in all_pop_long_names.keys()):
        relevant_superpops.add('OTH')


    sorted_superpopulations = sorted(list(relevant_superpops), key=lambda s: (s == 'OTH', s)) # OTH last

    for superpop_code in sorted_superpopulations:
        if current_y < 0.05: break # Stop if out of space

        superpop_profile = SUPERPOP_COLOR_PROFILES.get(superpop_code, SUPERPOP_COLOR_PROFILES['OTH'])
        superpop_base_color_hex = superpop_profile[0]
        
        rect_super = patches.Rectangle((patch_x, current_y - line_height_super*0.8), patch_width, line_height_super*0.7, 
                                       facecolor=superpop_base_color_hex, edgecolor='k', linewidth=0.5)
        ax.add_patch(rect_super)
        ax.text(text_x_super, current_y - line_height_super*0.8/2, superpop_code, 
                va='center', ha='left', fontweight='bold', fontsize=8.5)
        current_y -= line_height_super

        # Get populations for this superpopulation
        pops_in_this_super = []
        for pop_code, long_name in all_pop_long_names.items():
            pop_super = None
            matches = pca_merged_for_structure[pca_merged_for_structure['Population'] == pop_code]['Superpopulation'].dropna()
            if not matches.empty:
                pop_super = matches.iloc[0]
            elif pca_merged_for_structure[pca_merged_for_structure['Population'] == pop_code]['Superpopulation'].isna().any():
                 pop_super = 'OTH' # Assign to OTH for legend grouping

            if pop_super == superpop_code:
                pops_in_this_super.append(pop_code)
        
        for pop_code in sorted(pops_in_this_super):
            if current_y < 0.05: break
            pop_color = population_color_map.get(pop_code, get_population_color('OTH',1,0))
            pop_display_name = f"{all_pop_long_names.get(pop_code, pop_code)} ({pop_code})"
            
            rect_sub = patches.Rectangle((patch_x + indent_sub, current_y - line_height_sub*0.7), patch_width*0.8, line_height_sub*0.6, 
                                         facecolor=pop_color, edgecolor='darkgrey', linewidth=0.3)
            ax.add_patch(rect_sub)
            ax.text(text_x_sub, current_y - line_height_sub*0.7/2, pop_display_name, 
                    va='center', ha='left', fontsize=7)
            current_y -= line_height_sub
        current_y -= line_height_sub * 0.3 # Small gap after subpopulations of a superpop


# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Advanced PCA & UMAP Visualization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_folder", type=str, help="Folder with input data files.")
    args = parser.parse_args()

    input_dir = args.input_folder; os.makedirs(input_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'axes.labelweight': 'normal', 'axes.titleweight': 'bold', 'figure.titleweight': 'bold',
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'savefig.dpi': 300
    })

    print(f"--- Reading data files from: {input_dir} ---")
    eigenvalues_df = read_data_file(find_file_or_exit([input_dir], EIGENVALUES_SUFFIX, "Eigenvalues"), ['PC', 'Eigenvalue'], "Eigenvalues")
    num_pcs_eigen = len(eigenvalues_df)
    min_pca_cols = ['SampleID'] + ([f'PC{i+1}' for i in range(min(1, num_pcs_eigen))] or ['PC1'])
    pca_df_raw = read_data_file(find_file_or_exit([input_dir], PCA_SUFFIXES_TO_TRY, "PCA"), min_pca_cols, "PCA Data")
    pca_df_raw['SampleID'] = pca_df_raw['SampleID'].astype(str).str.strip()
    
    all_pc_cols_in_pca = sorted([col for col in pca_df_raw.columns if col.startswith("PC") and col[2:].isdigit()], key=lambda x: int(x[2:]))
    if not all_pc_cols_in_pca: print("ERROR: No PC columns in PCA data."); sys.exit(1)
    
    num_pcs_available_in_pca_data = int(all_pc_cols_in_pca[-1][2:])
    num_pcs_to_use_for_pca_plots = min(num_pcs_available_in_pca_data, num_pcs_eigen, 4)
    num_pcs_for_umap = min(num_pcs_available_in_pca_data, num_pcs_eigen)
    pc_cols_for_umap = [f"PC{i+1}" for i in range(num_pcs_for_umap) if f"PC{i+1}" in pca_df_raw.columns]

    loadings_df = None
    try:
        loadings_df = read_data_file(find_file_or_exit([input_dir], LOADINGS_SUFFIXES_TO_TRY, "Loadings"), ['Pos', 'PC1_loading'], "Loadings Data")
    except SystemExit: print("Info: Loadings file not found/unreadable. Skipped.")
    
    pop_df = read_data_file(find_file_or_exit([input_dir, os.getcwd()], POPULATION_FILE_NAME, "Population Info"), 
                            ['Sample name', 'Population code', 'Superpopulation code'], "Population Data", sep='\t')
    pop_df.rename(columns={'Sample name': 'SampleID', 'Population code': 'Population',
                           'Superpopulation code': 'Superpopulation', 'Population name': 'PopulationNameLong'}, inplace=True)
    if 'PopulationNameLong' not in pop_df.columns: pop_df['PopulationNameLong'] = pop_df['Population']
    pop_df['SampleID'] = pop_df['SampleID'].astype(str).str.strip()

    print("\n--- Merging & Annotating Data ---")
    pca_cols_to_merge = ['SampleID'] + [col for col in pc_cols_for_umap if col in pca_df_raw.columns]
    pca_df_to_merge = pca_df_raw[list(set(pca_cols_to_merge))] # Use set to unique cols if 'SampleID' was in pc_cols_for_umap
    pca_merged = pca_df_to_merge.merge(pop_df[['SampleID', 'Population', 'PopulationNameLong', 'Superpopulation']],
                              on='SampleID', how='left')
    print(f"{pca_merged['Population'].notna().sum()}/{len(pca_merged)} PCA samples annotated.")

    population_color_map, all_pop_long_names = {}, {}
    if 'Superpopulation' in pca_merged.columns and 'Population' in pca_merged.columns:
        superpop_groups = pca_merged.dropna(subset=['Superpopulation', 'Population']).groupby('Superpopulation')['Population'].unique()
        for superpop, pop_array in superpop_groups.items():
            sorted_pops = sorted(list(pop_array))
            for idx, pop_code in enumerate(sorted_pops):
                population_color_map[pop_code] = get_population_color(superpop, len(sorted_pops), idx)
                long_name_series = pca_merged[pca_merged['Population'] == pop_code]['PopulationNameLong'].dropna()
                all_pop_long_names[pop_code] = long_name_series.iloc[0] if not long_name_series.empty else pop_code
        for pop_code in pca_merged['Population'].dropna().unique(): # Catch any remaining pops (e.g. NaN superpop)
             if pop_code not in population_color_map: 
                population_color_map[pop_code] = get_population_color('OTH', 1, 0)
                long_name_series = pca_merged[pca_merged['Population'] == pop_code]['PopulationNameLong'].dropna()
                all_pop_long_names[pop_code] = long_name_series.iloc[0] if not long_name_series.empty else pop_code
    
    total_eigen_sum = eigenvalues_df['Eigenvalue'].sum()
    pc_variances = {}
    if total_eigen_sum > 1e-9:
        var_explained = (eigenvalues_df['Eigenvalue'] / total_eigen_sum) * 100
        eigen_pc_labels = eigenvalues_df['PC'].astype(str).str.replace('PC', '', regex=False).astype(int)
        for i in range(num_pcs_to_use_for_pca_plots): 
            pc_num = i + 1
            match_idx = eigen_pc_labels[eigen_pc_labels == pc_num].index
            pc_variances[f"PC{pc_num}"] = var_explained.loc[match_idx[0]] if not match_idx.empty else 0.0
    else:
        for i in range(num_pcs_to_use_for_pca_plots): pc_variances[f"PC{i+1}"] = 0.0

    umap_df = None
    if pc_cols_for_umap and len(pc_cols_for_umap) >= 2:
        print("\n--- Calculating UMAP Embedding ---")
        umap_input_data = pca_merged[pc_cols_for_umap].dropna()
        if not umap_input_data.empty and len(umap_input_data) >= 5:
            try:
                n_neighbors_val = min(15, len(umap_input_data)-1) if len(umap_input_data) > 1 else 1
                if n_neighbors_val < 2 : n_neighbors_val = 2
                reducer = umap.UMAP(n_neighbors=n_neighbors_val, min_dist=0.1, n_components=2, random_state=42, low_memory=True)
                embedding = reducer.fit_transform(umap_input_data)
                umap_df_coords = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'], index=umap_input_data.index)
                umap_df = pca_merged.loc[umap_input_data.index, ['SampleID', 'Population', 'Superpopulation']].copy()
                umap_df = umap_df.join(umap_df_coords) 
                print("UMAP calculation complete.")
            except Exception as e: print(f"ERROR UMAP: {e}. Skipped."); umap_df = None
        else: print("Not enough data for UMAP. Skipped.")
    else: print("UMAP skipped: Not enough PC columns.")

    plot_tasks = [{'type': 'scree'}]
    pc_pairs_to_plot = [('PC1', 'PC2')]
    if num_pcs_to_use_for_pca_plots >= 3 and 'PC3' in pca_merged.columns: pc_pairs_to_plot.extend([('PC1', 'PC3'), ('PC2', 'PC3')])
    if num_pcs_to_use_for_pca_plots >= 4 and 'PC4' in pca_merged.columns: pc_pairs_to_plot.append(('PC1', 'PC4'))

    for pc_x, pc_y in pc_pairs_to_plot:
        if pc_x in pca_merged.columns and pc_y in pca_merged.columns:
            plot_tasks.append({'type': 'pca_scatter', 'x_col': pc_x, 'y_col': pc_y,
                               'var_x': pc_variances.get(pc_x), 'var_y': pc_variances.get(pc_y), 'title_prefix': 'PCA'})
    if umap_df is not None:
        plot_tasks.append({'type': 'umap_scatter', 'x_col': 'UMAP1', 'y_col': 'UMAP2',
                           'var_x': None, 'var_y': None, 'title_prefix': 'UMAP'})
    if loadings_df is not None:
        plot_tasks.append({'type': 'loadings_hist', 'pc_load_col': 'PC1_loading', 'title': 'PC1 Loadings Density'})
        if 'PC2_loading' in loadings_df.columns:
            plot_tasks.append({'type': 'loadings_hist', 'pc_load_col': 'PC2_loading', 'title': 'PC2 Loadings Density'})
    
    # Add the color scheme guide as the last plot task
    if population_color_map: # Only if we have colors to show
        plot_tasks.append({'type': 'color_scheme_guide'})


    num_total_subplots = len(plot_tasks)
    ncols_fig = 2 
    if num_total_subplots == 1: ncols_fig = 1
    elif num_total_subplots == 3 or num_total_subplots == 5 : ncols_fig = 2 
    elif num_total_subplots > 6 : ncols_fig = 3
    nrows_fig = math.ceil(num_total_subplots / ncols_fig)
    
    fig_height_per_row = 6.0 
    plot_col_width = 6.5
    
    fig = plt.figure(figsize=(ncols_fig * plot_col_width + 0.5, nrows_fig * fig_height_per_row + 0.5))
    gs = fig.add_gridspec(nrows_fig, ncols_fig, hspace=0.60 if nrows_fig > 1 else 0.2, wspace=0.35) 
    fig.suptitle('Dimensionality Reduction Analysis', fontsize=18, y=0.99 if nrows_fig > 1 else 1.03)

    plot_params_shared = {
        'point_alpha': POINT_ALPHA, 'kde_line_alpha': KDE_LINE_ALPHA, 
        'min_samples_kde': MIN_SAMPLES_FOR_KDE, 'marker_size': DEFAULT_MARKER_SIZE, 
        'kde_levels': KDE_LEVELS, 'kde_thresh': KDE_THRESH,
    }

    for i, task in enumerate(plot_tasks):
        ax = fig.add_subplot(gs[i // ncols_fig, i % ncols_fig]) 
        if task['type'] == 'scree':
            ax.bar(eigenvalues_df['PC'].astype(str), eigenvalues_df['Eigenvalue'], color=SUPERPOP_COLOR_PROFILES['EUR'][0], alpha=0.75, edgecolor='k', lw=0.5)
            ax.set(xlabel='Principal Component', ylabel='Eigenvalue', title='Scree Plot')
            ax.yaxis.label.set_color(SUPERPOP_COLOR_PROFILES['EUR'][0]); ax.tick_params(axis='y', labelcolor=SUPERPOP_COLOR_PROFILES['EUR'][0])
            ax.tick_params(axis='x', rotation=45 if len(eigenvalues_df) > 10 else 0, labelsize=7.5); ax.set_ylim(bottom=0)
            if total_eigen_sum > 1e-9 and 'var_explained' in locals() and not var_explained.empty:
                ax_twin = ax.twinx()
                cumsum_var = var_explained.cumsum()
                ax_twin.plot(eigenvalues_df['PC'].astype(str), cumsum_var.values, color=SUPERPOP_COLOR_PROFILES['AFR'][0], marker='.', ms=7, alpha=0.85)
                ax_twin.set_ylabel('Cumulative % Variance', color=SUPERPOP_COLOR_PROFILES['AFR'][0])
                ax_twin.tick_params(axis='y', labelcolor=SUPERPOP_COLOR_PROFILES['AFR'][0])
                ax_twin.set_ylim(0, max(101, cumsum_var.max() * 1.02 if not cumsum_var.empty else 101))
        elif task['type'] == 'pca_scatter':
            plot_scatter_with_kde(ax, pca_merged, task['x_col'], task['y_col'], task['var_x'], task['var_y'],
                                  population_color_map, task['title_prefix'], **plot_params_shared)
        elif task['type'] == 'umap_scatter' and umap_df is not None:
             plot_scatter_with_kde(ax, umap_df, task['x_col'], task['y_col'], task['var_x'], task['var_y'],
                                   population_color_map, task['title_prefix'], **plot_params_shared)
        elif task['type'] == 'loadings_hist' and loadings_df is not None:
            pc_load_col = task['pc_load_col']
            if pc_load_col in loadings_df.columns:
                valid_loadings = loadings_df[['Pos', pc_load_col]].dropna()
                if not valid_loadings.empty and len(valid_loadings) > 1:
                    hb = ax.hist2d(valid_loadings['Pos'], valid_loadings[pc_load_col], bins=(80, 60), cmap='inferno', cmin=1, rasterized=True) 
                    fig.colorbar(hb[3], ax=ax, label='SNP Density', shrink=0.85, aspect=30)
                    ax.set_xlabel('Genomic Position', fontsize=10); ax.set_ylabel(f'{pc_load_col.replace("_", " ")} Value', fontsize=10)
                    ax.set_title(task['title'], fontsize=12, fontweight='medium')
                    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                else: ax.text(0.5,0.5, "Not enough data for\nloadings histogram.", ha='center',va='center',transform=ax.transAxes); ax.set_title(task['title'])
            else: ax.text(0.5,0.5, f"{pc_load_col} not found.", ha='center',va='center',transform=ax.transAxes); ax.set_title(task['title'])
        elif task['type'] == 'color_scheme_guide':
            plot_color_scheme_guide(ax, population_color_map, all_pop_long_names, pca_merged)
    
    for j in range(num_total_subplots, nrows_fig * ncols_fig): # Hide any final unused subplots
        fig.add_subplot(gs[j // ncols_fig, j % ncols_fig]).set_visible(False)
    
    plt.savefig(os.path.join(input_dir, OUTPUT_PLOT_BASENAME), bbox_inches='tight')
    print(f"\n--- Visualization saved to: {os.path.join(input_dir, OUTPUT_PLOT_BASENAME)} ---")

    print("\n--- Data Summary ---")
    print(f"PCs (Eigenvalues): {num_pcs_eigen}. Max PC plotted: {num_pcs_to_use_for_pca_plots}. PCs for UMAP: {len(pc_cols_for_umap) if umap_df is not None else 'N/A'}.")
    print(f"Samples (PCA): {len(pca_df_raw)}. Variants (Loadings): {len(loadings_df) if loadings_df is not None else 'N/A'}.")
    pc_var_strs = [f"PC{i+1}: {pc_variances.get(f'PC{i+1}', 0):.2f}%" for i in range(min(4, num_pcs_to_use_for_pca_plots))]
    if pc_var_strs: print(f"Variance Explained: {'; '.join(pc_var_strs)}")
    print("\nPopulation Counts (Top 10 shown):")
    pop_summary = pca_merged.groupby(['Superpopulation', 'Population'], dropna=False).size().reset_index(name='Count').sort_values(by=['Superpopulation', 'Population'])
    if not pop_summary.empty: print(pop_summary.to_string(index=False, max_rows=10))
    print("\nScript finished.")

if __name__ == "__main__":
    main()
