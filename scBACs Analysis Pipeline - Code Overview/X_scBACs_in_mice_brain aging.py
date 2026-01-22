#!/usr/bin/env python3
"""
Analysis of scBACs in Mouse Brain Aging
Cell-type-specific aging analysis in young vs old mice and radiation-induced brain injury (RIBI) mice
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Data science and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for better graphic quality

# Statistical analysis libraries
from scipy import stats

# Single-cell analysis libraries
import scanpy as sc
sc.set_figure_params(figsize=(8, 8), frameon=False)

# ============================================================================
# Analysis 1: Cell age comparison between young and old mice brain
# ============================================================================

# Load metadata for young and old mice
data_path = './meta_young&old_mice.csv'
df = pd.read_csv(data_path, index_col=0)

print("=" * 70)
print("ANALYSIS 1: Young vs Old Mice Brain - Cell Age Comparison")
print("=" * 70)
print(f"Dataset shape: {df.shape}")
print(f"Unique cell types: {len(df['celltype'].unique())}")
print(f"Age groups: {df['Age'].unique()}")
print("-" * 70)

# Perform Mann-Whitney U test for each cell type
results = []

for cell_type in df['celltype'].unique():
    # Extract data for old mice (20-month-old stage and over)
    groupA = df.loc[(df['Age'] == '20-month-old stage and over') & 
                    (df['celltype'] == cell_type), 'predicted_cell_age']
    
    # Extract data for young mice (4-week-old stage)
    groupB = df.loc[(df['Age'] == '4-week-old stage') & 
                    (df['celltype'] == cell_type), 'predicted_cell_age']
    
    # Check if both groups have sufficient data
    if len(groupA) > 0 and len(groupB) > 0:
        # Perform Mann-Whitney U test (two-sided)
        statistic, p_value = stats.mannwhitneyu(groupA, groupB, alternative='two-sided')
        
        # Store results
        results.append({
            'cell_type': cell_type,
            'p_value': p_value,
            'statistic': statistic,
            'old_mean': groupA.mean(),
            'young_mean': groupB.mean(),
            'old_count': len(groupA),
            'young_count': len(groupB),
            'old_median': groupA.median(),
            'young_median': groupB.median()
        })
        
        # Format p-value with significance stars
        def format_pvalue(p):
            if p < 0.001:
                return f"{p:.2e} (***)"
            elif p < 0.01:
                return f"{p:.2e} (**)"
            elif p < 0.05:
                return f"{p:.3f} (*)"
            else:
                return f"{p:.3f} (ns)"
        
        # Print results for this cell type
        print(f"\nCell Type: {cell_type}")
        print(f"  P Value: {format_pvalue(p_value)}")
        print(f"  Old (20+ months): mean = {groupA.mean():.2f}, median = {groupA.median():.2f}, n = {len(groupA)}")
        print(f"  Young (4 weeks): mean = {groupB.mean():.2f}, median = {groupB.median():.2f}, n = {len(groupB)}")
        print(f"  Mean difference: {groupA.mean() - groupB.mean():.2f}")

# Create summary DataFrame
results_df = pd.DataFrame(results)

# Identify significant results
significant_results = results_df[results_df['p_value'] < 0.05]

print("\n" + "-" * 70)
print(f"Summary: {len(significant_results)}/{len(results_df)} cell types show significant differences (p < 0.05)")
print("-" * 70)

if len(significant_results) > 0:
    print("\nSignificant cell types (p < 0.05):")
    for _, row in significant_results.iterrows():
        print(f"  {row['cell_type']}: p = {row['p_value']:.3e}, "
              f"Δ = {row['old_mean'] - row['young_mean']:.2f}")

# ============================================================================
# Analysis 2: RIBI Mice - Cell-type-specific brain aging after radiation
# ============================================================================

print("\n\n" + "=" * 70)
print("ANALYSIS 2: RIBI Mice - Cell-type-specific Brain Aging After Radiation")
print("=" * 70)

# Load RIBI mice data
ribi_path = './meta_RIBI_cortex_aging_mice.csv'
data = pd.read_csv(ribi_path, header=0, index_col=0)

print(f"RIBI dataset shape: {data.shape}")
print(f"Unique cell types: {data['celltype'].unique()}")
print(f"Radiation time points: {sorted(data['rib_time'].unique())}")

# Select relevant columns (note: fixing column name from R code - 'celltypist' should be 'celltype')
# In R code they use 'celltypist' but based on the dataset, it should be 'celltype'
df = data[['rib_time', 'celltype', 'predicted_cell_age']].copy()
df.columns = ['rib_time', 'celltype', 'cell_age']  

# Define cell types of interest (Per, End, Ast, Mic)
cell_types_of_interest = ['Per', 'End', 'Ast', 'Mic']

# Filter for selected cell types
subset_df = df[df['celltype'].isin(cell_types_of_interest)].copy()

print(f"\nSelected cell types: {cell_types_of_interest}")
print(f"Filtered data shape: {subset_df.shape}")

# Calculate median cell age for each cell type at each time point
mean_data = subset_df.groupby(['celltype', 'rib_time'], as_index=False)['cell_age'].median()
mean_data = mean_data.rename(columns={'cell_age': 'mean_len'})

print("\nMedian cell age by cell type and time point:")
print(mean_data)

# Sort data for proper plotting
mean_data['celltype'] = pd.Categorical(mean_data['celltype'], 
                                       categories=cell_types_of_interest, 
                                       ordered=True)
mean_data = mean_data.sort_values(['celltype', 'rib_time'])

# Calculate differences between consecutive time points
df_diff_list = []

for cell_type in cell_types_of_interest:
    cell_data = mean_data[mean_data['celltype'] == cell_type].copy()
    cell_data = cell_data.sort_values('rib_time')
    
    # Calculate differences between consecutive time points
    for i in range(1, len(cell_data)):
        current_time = cell_data.iloc[i]['rib_time']
        prev_time = cell_data.iloc[i-1]['rib_time']
        current_mean = cell_data.iloc[i]['mean_len']
        prev_mean = cell_data.iloc[i-1]['mean_len']
        
        diff = current_mean - prev_mean
        time_pair = f"{prev_time} → {current_time}"
        
        df_diff_list.append({
            'celltype': cell_type,
            'time_pair': time_pair,
            'diff': diff,
            'current_time': current_time,
            'prev_time': prev_time
        })

df_diff = pd.DataFrame(df_diff_list)
df_diff['celltype'] = pd.Categorical(df_diff['celltype'], 
                                     categories=cell_types_of_interest, 
                                     ordered=True)

print("\nDifferences between consecutive time points:")
print(df_diff)

# ============================================================================
# Visualization: Heatmap of changes in median length between time points
# ============================================================================

plt.figure(figsize=(10, 6))

# Create pivot table for heatmap
pivot_table = df_diff.pivot(index='celltype', columns='time_pair', values='diff')

# Create heatmap
ax = sns.heatmap(pivot_table, 
                 annot=True, 
                 fmt=".2f", 
                 cmap='RdBu_r', 
                 center=0,
                 linewidths=0.5, 
                 linecolor='white',
                 cbar_kws={'label': 'Δ Median Cell Age'})

# Customize plot appearance
plt.title('Changes in Median Cell Age Between Consecutive Time Points\n(RIBI Mice After Radiation)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Time Point Transition', fontsize=12, fontweight='bold')
plt.ylabel('Cell Type', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save figure
output_path = './ribi_mice_cell_age_changes_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_path}")

# ============================================================================
# End of Analysis
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)