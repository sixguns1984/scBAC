#!/usr/bin/env python3
"""
Application of scBACs in iPSC Neuron Development Process
Analysis of cell age distribution and statistical differences across timepoints
"""
# Data science and visualization libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for better graphic quality

# Statistical and machine learning libraries
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

# Load metadata for iPSC neuron differentiation process
data_path = '/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/dataset/meta_ipsc_neuron_differentiated_human.csv'
df = pd.read_csv(data_path,index_col=0)

print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")
print(f"Unique timepoints: {df['Timepoint2'].unique()}")


# Define timepoint order (equivalent to factor ordering in R)
timepoint_order = ["DIV0", "DIV35-50", "DIV50-65", "DIV>65"]
df['Timepoint2'] = pd.Categorical(df['Differentiated time point'], categories=timepoint_order, ordered=True)


# ============================================================================
# Visualization: Violin plot with boxplot overlay
# ============================================================================

plt.figure(figsize=(12, 8))

# Create violin plot
sns.violinplot(data=df, x='Timepoint2', y='predicted_cell_age', 
               palette='Set2', inner=None, alpha=0.7)

# Add boxplot inside violin
sns.boxplot(data=df, x='Timepoint2', y='predicted_cell_age', 
            width=0.1, color='white', showcaps=False,
            boxprops={'facecolor':'white'}, medianprops={'color':'black'})

# Customize plot appearance
plt.title('Cell Age Distribution Across Timepoints', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Timepoint', fontsize=14, fontweight='bold')
plt.ylabel('Cell Age (years)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--')


# ============================================================================
# Statistical Testing (Mann-Whitney U test)
# ============================================================================

# Define comparison pairs (equivalent to comparisons in R code)
comparisons = [
    ("DIV0", "DIV35-50"),
    ("DIV35-50", "DIV50-65"),
    ("DIV50-65", "DIV>65"),
    ("DIV0", "DIV>65")
]

# Perform statistical tests and format results
def format_pvalue(p):
    """Format p-value with significance stars"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

# Perform Mann-Whitney U tests for each comparison
print("\n" + "="*60)
print("Statistical Test Results (Mann-Whitney U test)")
print("="*60)

for i, (group1, group2) in enumerate(comparisons):
    # Extract data for both groups
    data1 = df[df['Timepoint2'] == group1]['predicted_cell_age'].dropna()
    data2 = df[df['Timepoint2'] == group2]['predicted_cell_age'].dropna()
    
    # Perform Mann-Whitney U test
    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
    
    # Format p-value and significance
    significance = format_pvalue(p_value)
    
    # Print results
    print(f"{group1} vs {group2}:")
    print(f"  Statistic: {stat:.4f}, p-value: {p_value:.6f} ({significance})")
    print(f"  Group sizes: {len(data1)} vs {len(data2)}")
    print(f"  Medians: {np.median(data1):.2f} vs {np.median(data2):.2f}")
    print("-" * 40)

# ============================================================================
# Add statistical annotations to the plot
# ============================================================================

# Manually add significance annotations (simplified version)
# Note: In practice, you might want to use a dedicated function for proper annotation placement
y_max = df['predicted_cell_age'].max()
y_positions = [y_max * (1.05 + 0.05*i) for i in range(len(comparisons))]

for i, ((group1, group2), y_pos) in enumerate(zip(comparisons, y_positions)):
    data1 = df[df['Timepoint2'] == group1]['predicted_cell_age'].dropna()
    data2 = df[df['Timepoint2'] == group2]['predicted_cell_age'].dropna()
    
    if len(data1) > 0 and len(data2) > 0:
        stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        significance = format_pvalue(p_value)
        
        # Calculate x positions for annotation line
        x1 = timepoint_order.index(group1)
        x2 = timepoint_order.index(group2)
        
        # Draw annotation line
        plt.plot([x1, x1, x2, x2], [y_pos-0.02*y_max, y_pos, y_pos, y_pos-0.02*y_max], 
                color='black', linewidth=1.5)
        
        # Add significance text
        plt.text((x1+x2)/2, y_pos+0.01*y_max, significance, 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Adjust layout to prevent label clipping
plt.tight_layout()

# Save the figure
output_path = '/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/dataset/test/cell_age_distribution_timepoints.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")


# ============================================================================
# Additional Analysis: Summary Statistics
# ============================================================================

print("\n" + "="*60)
print("Summary Statistics by Timepoint")
print("="*60)

for timepoint in timepoint_order:
    subset = df[df['Timepoint2'] == timepoint]['predicted_cell_age']
    
    if len(subset) > 0:
        print(f"\n{timepoint}:")
        print(f"  Count: {len(subset):,}")
        print(f"  Mean: {subset.mean():.2f} ± {subset.std():.2f}")
        print(f"  Median: {subset.median():.2f}")
        print(f"  Range: [{subset.min():.2f}, {subset.max():.2f}]")
        print(f"  IQR: [{subset.quantile(0.25):.2f}, {subset.quantile(0.75):.2f}]")
    else:
        print(f"\n{timepoint}: No data available")

# ============================================================================
# Optional: Kruskal-Wallis test for overall differences
# ============================================================================

print("\n" + "="*60)
print("Kruskal-Wallis Test for Overall Differences")
print("="*60)

# Prepare data for Kruskal-Wallis test
groups_data = []
for timepoint in timepoint_order:
    data = df[df['Timepoint2'] == timepoint]['predicted_cell_age'].dropna()
    if len(data) > 0:
        groups_data.append(data)

if len(groups_data) >= 2:
    h_stat, p_value_kw = stats.kruskal(*groups_data)
    print(f"H-statistic: {h_stat:.4f}")
    print(f"p-value: {p_value_kw:.6f}")
    
    if p_value_kw < 0.05:
        print("Result: Significant differences exist between groups (p < 0.05)")
    else:
        print("Result: No significant differences between groups (p ≥ 0.05)")

# ============================================================================
# End of Analysis
# ============================================================================

print("\n" + "="*60)
print("Analysis Complete")
print("="*60)