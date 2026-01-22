"""
Cell-type-specific Analysis of Aging Acceleration Onset in Neurodegenerative Diseases
====================================================================================

Purpose:
- Analyze cellular-level relative age acceleration (RAA) in NDDs
- Identify cell-type-specific individual age at aging acceleration onset
- Investigate relationships between aging cell proportions and clinical features
- Discover genes associated with aging acceleration onset

Neurodegenerative diseases analyzed:
- AD: Alzheimer's Disease
- PD: Parkinson's Disease  
- MCI: Mild Cognitive Impairment
- FTD: Frontotemporal Dementia
- ALS: Amyotrophic Lateral Sclerosis
- FTLD: Frontotemporal Lobar Degeneration

Note: Uses scBACs tool for cellular age analysis
      https://github.com/sixguns1984/scBACs
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

# Core scientific computing libraries
import numpy as np
import pandas as pd

# Single-cell analysis tools
import scanpy as sc

# Statistics
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Visualization libraries
import matplotlib.pyplot as plt

# ============================================================================
# 2. SETUP AND DATA PREPARATION
# ============================================================================

print("\n" + "="*60)
print("Step 1: Setup and Data Preparation")
print("="*60)

# Note: The following sections contain both Python code and command-line instructions
# Command-line instructions are marked with comments starting with "# linux command:"

"""
# linux command: Install scBACs tool
git clone https://github.com/sixguns1984/scBACs.git
cd scBACs
pip install scbac-0.1.0-py3-none-any.whl
"""

# Load metadata
print("Loading metadata...")
meta = pd.read_csv('./meta_human_cortex_scrna_atlas.csv', index_col=0)

# Filter for specific cell types
target_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli', 'Per']
meta = meta.loc[meta['celltype'].isin(target_celltypes), :]
print(f"Metadata shape after filtering: {meta.shape}")
print(f"Cell types included: {target_celltypes}")

# Select neurodegenerative disease to analyze
ND = 'AD'  # Options: AD, PD, MCI, FTD, ALS, FTLD
print(f"\nAnalyzing neurodegenerative disease: {ND}")

# Filter for disease cohorts
nd_cohort = meta.loc[meta['status'] == ND, 'dataset'].unique()
meta2 = meta.loc[meta['dataset'].isin(nd_cohort), :]
df = meta2.loc[(meta2['status'] == ND) | (meta2['status'] == 'CT'), :]

print(f"  Disease datasets: {len(nd_cohort)}")
print(f"  Disease samples: {df[df['status'] == ND].shape[0]}")
print(f"  Control samples: {df[df['status'] == 'CT'].shape[0]}")

# Save data for scBACs analysis
output_path = f'./{ND}_predicted_ages.csv'
df.to_csv(output_path)
print(f"\nData saved for scBACs analysis: {output_path}")

# ============================================================================
# 3. SCBACS ANALYSIS - COMMAND LINE INSTRUCTIONS
# ============================================================================

print("\n" + "="*60)
print("Step 2: scBACs Analysis - Command Line Instructions")
print("="*60)

print("""
# linux command: Analyze cellular level of relative age acceleration (RAA) 
# and cell-type-specific individual age at aging acceleration onset in AD

scbac analyze \\
    --input ./AD_predicted_ages.csv \\
    --output-prefix ./ad_analysis_results \\
    --cell-age-pred-col predicted_cell_age \\
    --disease-name AD \\
    --status-col status \\
    --chronological-age-col Age_at_death \\
    --donor-col donor_id \\
    --celltype-col celltype
""")

print("Note: Run the above command in Linux terminal to perform scBACs analysis.")
print("      The analysis will generate threshold age and positive ratio files.")

# ============================================================================
# 4. ANALYZE RELATIONSHIPS WITH AGING CELL PROPORTIONS
# ============================================================================

print("\n" + "="*60)
print("Step 3: Analyze Relationships with Aging Cell Proportions")
print("="*60)

# After running scBACs, load the results
print("Loading scBACs analysis results...")

# Define paths to results files
base_path = './'
aging_ratio_file = base_path + 'ad_analysis_results_positive_ratio.csv'
threshold_file = base_path + 'ad_analysis_results_thresholds.csv'

# 4.1 Relationship between aging onset age and aging cell proportion
print("\n4.1 Relationship between aging onset age and aging cell proportion")
aging_ratio = pd.read_csv(aging_ratio_file, index_col=0)
threshold_df = pd.read_csv(threshold_file, index_col=0)

# Set indices for merging
aging_ratio.index = aging_ratio['donor_id']
threshold_df.index = threshold_df['donor_id']

print(f"Aging ratio data shape: {aging_ratio.shape}")
print(f"Threshold data shape: {threshold_df.shape}")

# Calculate correlations for each cell type
print("\nCorrelation between threshold age and positive ratio:")
for celltype in threshold_df['celltype'].unique():
    # Filter data for current cell type
    temp1 = threshold_df.loc[threshold_df['celltype'] == celltype, :]
    temp2 = aging_ratio.loc[aging_ratio['celltype'] == celltype, :]
    
    # Ensure matching donors
    common_donors = np.intersect1d(temp1.index, temp2.index)
    temp1 = temp1.loc[common_donors, :]
    temp2 = temp2.loc[common_donors, :]
    
    if len(common_donors) > 3:  # Need sufficient data for correlation
        coef, p_value = spearmanr(temp1['threshold_age'].values, temp2['positive_ratio'].values)
        print(f"  {celltype}: Spearman R = {coef:.3f}, P = {p_value:.4f}")
    else:
        print(f"  {celltype}: Insufficient data (n={len(common_donors)})")

# ============================================================================
# 5. RELATIONSHIP WITH COGNITIVE FUNCTION (MMSE)
# ============================================================================

print("\n" + "="*60)
print("Step 4: Relationship with Cognitive Function (MMSE)")
print("="*60)

print("Analyzing relationship between aging cell proportion and MMSE scores...")

# Load aging ratio data
aging_ratio = pd.read_csv(aging_ratio_file, index_col=0)

# Add MMSE scores
mmse_data = df.drop_duplicates('donor_id')
mmse_data.index = mmse_data['donor_id']
mmse_filtered = mmse_data.loc[aging_ratio['donor_id'].values, :]
aging_ratio['mmse'] = mmse_filtered['MMSE'].values

# Clean data
threshold_df2 = aging_ratio.dropna(subset=['mmse', 'positive_ratio'])
print(f"Donors with MMSE data: {threshold_df2['donor_id'].nunique()}")

# Calculate correlations for each cell type
print("\nCorrelation between MMSE and positive ratio:")
for celltype in threshold_df2['celltype'].unique():
    temp = threshold_df2.loc[threshold_df2['celltype'] == celltype, :]
    if temp.shape[0] > 3:  # Need sufficient data
        coef, p_value = spearmanr(temp['mmse'].values, temp['positive_ratio'].values)
        print(f"  {celltype}: Spearman R = {coef:.3f}, P = {p_value:.4f}")
    else:
        print(f"  {celltype}: Insufficient data (n={temp.shape[0]})")

# ============================================================================
# 6. RELATIONSHIP WITH BRAAK STAGE
# ============================================================================

print("\n" + "="*60)
print("Step 5: Relationship with Braak Stage")
print("="*60)

print("Analyzing relationship between aging cell proportion and Braak stage...")


# Load aging ratio data
aging_ratio = pd.read_csv(aging_ratio_file, index_col=0)

# Add Braak stage data
braak_data = df.drop_duplicates('donor_id')
braak_data.index = braak_data['donor_id']
braak_filtered = braak_data.loc[aging_ratio['donor_id'].values, :]
aging_ratio['braak'] = braak_filtered['Braak_stage'].values

# Clean data
threshold_df2 = aging_ratio.dropna(subset=['braak', 'positive_ratio'])
print(f"Donors with Braak stage data: {threshold_df2['donor_id'].nunique()}")

# Calculate correlations for each cell type
print("\nCorrelation between Braak stage and positive ratio:")
for celltype in threshold_df2['celltype'].unique():
    temp = threshold_df2.loc[threshold_df2['celltype'] == celltype, :]
    if temp.shape[0] > 3:  # Need sufficient data
        coef, p_value = spearmanr(temp['braak'].values, temp['positive_ratio'].values)
        print(f"  {celltype}: Spearman R = {coef:.3f}, P = {p_value:.4f}")
    else:
        print(f"  {celltype}: Insufficient data (n={temp.shape[0]})")


# ============================================================================
# 7. GENE EXPRESSION ANALYSIS FOR AGING ACCELERATION ONSET IN AD
# ============================================================================

print("\n" + "="*60)
print("Step 6: Gene Expression Analysis for Aging Acceleration Onset")
print("="*60)

# Focus on inhibitory neurons (Inh) as an example
celltype = 'Inh'
print(f"Analyzing gene expression associated with aging acceleration onset in {celltype}...")

# Load threshold data
threshold_df = pd.read_csv(threshold_file, index_col=0)
threshold_df = threshold_df.loc[threshold_df['celltype'] == celltype, :]
threshold_df.index = threshold_df['donor_id'].values

print(f"  Donors with threshold data: {threshold_df.shape[0]}")

# Load single-cell data
adata = sc.read_h5ad('./human_brain_scRNA_atlas.h5ad')

# Filter for AD samples of the target cell type
df2 = meta2.loc[(meta2['status'] == ND) & (meta2['celltype'] == celltype), :]
sce = adata[df2.index, :]

print(f"  Single-cell data shape: {sce.shape}")

# Normalize data
sc.pp.normalize_per_cell(sce)
sc.pp.log1p(sce)

# Prepare data for correlation analysis
obs = sce.obs.copy()
rna = sce.to_df()

# Calculate Spearman correlation for each gene
print(f"\n  Calculating correlations for {rna.shape[1]} genes...")
r_values = []
gene_names = []
p_values = []

for i in range(rna.shape[1]):
    # Aggregate expression by donor
    obs['gene'] = rna.iloc[:, i].values.copy()
    obs2 = obs.groupby(['donor_id'], observed=True)[['gene']].mean()
    
    # Find common donors
    common_donors = np.intersect1d(threshold_df.index, obs2.index)
    if len(common_donors) < 5:  # Skip if insufficient donors
        continue
    
    threshold_df2 = threshold_df.loc[common_donors, :]
    obs2 = obs2.loc[common_donors]
    
    # Calculate correlation
    coef, p_value = spearmanr(obs2.iloc[:, 0].values, threshold_df2['threshold_age'].values)
    r_values.append(coef)
    p_values.append(p_value)
    gene_names.append(rna.columns[i])

# Create results dataframe
result = pd.DataFrame({
    'genename': gene_names,
    'r': r_values,
    'p': p_values
})

# Apply FDR correction
_, fdr_p, _, _ = multipletests(result['p'].values, method='fdr_bh')
result['fdr'] = fdr_p

# Save results
output_file = f'./ad_{celltype}_spearman_age_aging_onset.csv'
result.to_csv(output_file, index=False)
print(f"\n  Results saved to: {output_file}")
print(f"  Significant genes (FDR < 0.05): {(result['fdr'] < 0.05).sum()}")



# ============================================================================
# 8. CONTROL GROUP ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("Step 7: Control Group Analysis")
print("="*60)

print("""
# linux command: Analyze aging acceleration onset in control group

scbac analyze \\
    --input ./AD_predicted_ages.csv \\
    --output-prefix ./CT_analysis_results \\
    --cell-age-pred-col predicted_cell_age \\
    --disease-name CT \\
    --status-col status \\
    --chronological-age-col Age_at_death \\
    --donor-col donor_id \\
    --celltype-col celltype
""")

# ============================================================================
# 9. COMPARE AD AND CONTROL GROUPS
# ============================================================================

print("\n" + "="*60)
print("Step 8: Compare AD and Control Groups")
print("="*60)

print("Comparing aging acceleration onset between AD and control groups...")


# Load results from both groups
ad_results = pd.read_csv(base_path + 'ad_analysis_results_thresholds.csv', index_col=0)
ct_results = pd.read_csv(base_path + 'CT_analysis_results_thresholds.csv', index_col=0)

print(f"AD results shape: {ad_results.shape}")
print(f"CT results shape: {ct_results.shape}")

# Compare for each cell type
print("\nComparison of threshold ages (Mann-Whitney U test):")
for celltype in ad_results['celltype'].unique():
    if celltype in ct_results['celltype'].unique():
        # Get data for current cell type
        data_ad = ad_results.loc[(ad_results['celltype'] == celltype), 'threshold_age'].values
        data_ct = ct_results.loc[(ct_results['celltype'] == celltype), 'threshold_age'].values
        
        if len(data_ad) > 3 and len(data_ct) > 3:
            # Perform Mann-Whitney U test
            stat, p_value = mannwhitneyu(data_ad, data_ct, alternative='two-sided')
            
            print(f"\n  {celltype}:")
            print(f"    AD: n={len(data_ad)}, mean={data_ad.mean():.1f}, std={data_ad.std():.1f}")
            print(f"    CT: n={len(data_ct)}, mean={data_ct.mean():.1f}, std={data_ct.std():.1f}")
            print(f"    Mann-Whitney U test: p={p_value:.4f}")
        else:
            print(f"\n  {celltype}: Insufficient data for comparison")

# ============================================================================
# 10. INTEGRATED ANALYSIS: GENES ASSOCIATED WITH AGING ONSET AND AD PATHOLOGY
# ============================================================================

print("\n" + "="*60)
print("Step 9: Integrated Analysis - Genes Associated with Aging Onset and AD Pathology")
print("="*60)

print("Performing integrated analysis of aging onset genes and AD differential expression...")

# Note: This section requires multiple analysis steps

# 10.1 Differential expression analysis between AD and CT in Inh cells
print("\n10.1 Differential expression analysis between AD and CT in Inh cells")

# Load single-cell data
adata = sc.read_h5ad('./human_brain_scRNA_atlas.h5ad')

# Filter for Inh cells in AD and CT groups
celltype = 'Inh'
df_filtered = df.loc[df['celltype'] == celltype, :]
sce = adata[df_filtered.index, :]

# Add group indicator (1 = AD, 0 = CT)
sce.obs.loc[:, 'group'] = 0
sce.obs.loc[sce.obs['status'] == ND, 'group'] = 1

# Normalize and perform differential expression analysis
sc.pp.normalize_total(sce)
sc.pp.log1p(sce)
sc.tl.rank_genes_groups(sce, "status", method="wilcoxon", pts=True)

# Save differential expression results
rank_results = sce.uns['rank_genes_groups']
all_groups_results = pd.DataFrame()

group_labels = sce.obs[rank_results['params']['groupby']].unique()
for group_label in group_labels:
    group_df = sc.get.rank_genes_groups_df(sce, group=group_label)
    group_df = group_df.sort_values(by="scores", ascending=False)
    
    # Filter for genes expressed in at least 10% of cells
    pts = sce.uns['rank_genes_groups']['pts'][group_label]
    selected_genes = pts[pts >= 0.1].index.tolist()
    group_df = group_df[group_df['names'].isin(selected_genes)]
    group_df['group'] = group_label
    all_groups_results = pd.concat([all_groups_results, group_df], ignore_index=True)

# Save DEG results
deg_file = f'./{ND}/{celltype}_AD_CT_DEG_scanpy.csv'
all_groups_results.to_csv(deg_file, index=False)
print(f"  Differential expression results saved to: {deg_file}")

#GENE EXPRESSION ANALYSIS FOR AGING ACCELERATION ONSET IN AD and CT
# Prepare data for correlation analysis
ad_results = pd.read_csv(base_path + 'ad_analysis_results_thresholds.csv', index_col=0)
ct_results = pd.read_csv(base_path + 'CT_analysis_results_thresholds.csv', index_col=0)
threshold_df = pd.concat([ad_results,ct_results],axis=0)

obs = sce.obs.copy()
rna = sce.to_df()

# Calculate Spearman correlation for each gene
print(f"\n  Calculating correlations for {rna.shape[1]} genes...")
r_values = []
gene_names = []
p_values = []

for i in range(rna.shape[1]):
    # Aggregate expression by donor
    obs['gene'] = rna.iloc[:, i].values.copy()
    obs2 = obs.groupby(['donor_id'], observed=True)[['gene']].mean()
    
    # Find common donors
    common_donors = np.intersect1d(threshold_df.index, obs2.index)
    if len(common_donors) < 5:  # Skip if insufficient donors
        continue
    
    threshold_df2 = threshold_df.loc[common_donors, :]
    obs2 = obs2.loc[common_donors]
    
    # Calculate correlation
    coef, p_value = spearmanr(obs2.iloc[:, 0].values, threshold_df2['threshold_age'].values)
    r_values.append(coef)
    p_values.append(p_value)
    gene_names.append(rna.columns[i])

# Create results dataframe
aging_onset_gene_sig = pd.DataFrame({
    'genename': gene_names,
    'r': r_values,
    'p': p_values
})

# Apply FDR correction
_, fdr_p, _, _ = multipletests(aging_onset_gene_sig['p'].values, method='fdr_bh')
aging_onset_gene_sig['fdr'] = fdr_p


# 10.2 Combined analysis of aging onset genes and DEGs
print("\n10.2 Combined analysis of aging onset genes and DEGs")


# Load DEG results
deg = pd.read_csv(deg_file)

# Filter significant results
deg_sig = deg.loc[(deg['group'] == 1) & (deg['pvals_adj'] < 0.05), :]  # CORRECTED: added missing parenthesis
aging_onset_gene_sig = aging_onset_gene.loc[(aging_onset_gene['fdr'] < 0.05), :]

print(f"  Significant DEGs (AD vs CT): {deg_sig.shape[0]}")
print(f"  Significant aging onset genes: {aging_onset_gene_sig.shape[0]}")

# Identify overlapping genes with consistent patterns
print("\n  Genes with consistent patterns:")

# Pattern 1: Upregulated in AD AND negatively correlated with aging onset
upregulated_ad = deg_sig.loc[deg_sig['logfoldchanges'] > 0, 'names'].values
neg_correlated = aging_onset_gene_sig.loc[aging_onset_gene_sig['r'] < 0, 'genename'].values
pattern1_genes = np.intersect1d(upregulated_ad, neg_correlated)
print(f"    Pattern 1 (up in AD, negative correlation): {len(pattern1_genes)} genes")

# Pattern 2: Downregulated in AD AND positively correlated with aging onset
downregulated_ad = deg_sig.loc[deg_sig['logfoldchanges'] < 0, 'names'].values
pos_correlated = aging_onset_gene_sig.loc[aging_onset_gene_sig['r'] > 0, 'genename'].values
pattern2_genes = np.intersect1d(downregulated_ad, pos_correlated)
print(f"    Pattern 2 (down in AD, positive correlation): {len(pattern2_genes)} genes")

# Save overlapping genes
if len(pattern1_genes) > 0:
    pattern1_df = deg_sig[deg_sig['names'].isin(pattern1_genes)]
    pattern1_df.to_csv(f'./{celltype}_pattern1_genes.csv', index=False)
    print(f"    Pattern 1 genes saved to: ./{celltype}_pattern1_genes.csv")

if len(pattern2_genes) > 0:
    pattern2_df = deg_sig[deg_sig['names'].isin(pattern2_genes)]
    pattern2_df.to_csv(f'./{celltype}_pattern2_genes.csv', index=False)
    print(f"    Pattern 2 genes saved to: ./{celltype}_pattern2_genes.csv")


# ============================================================================
print("\n" + "="*60)
print("Analysis Pipeline Complete!")
print("="*60)
print("\nSummary of analysis steps:")
print("  1. Data preparation for scBACs analysis")
print("  2. scBACs command-line analysis (run in Linux terminal)")
print("  3. Analysis of aging cell proportion relationships")
print("  4. Correlation with cognitive function (MMSE)")
print("  5. Correlation with Braak stage")
print("  6. Gene expression analysis for aging acceleration onset")
print("  7. Control group analysis")
print("  8. AD vs Control group comparison")
print("  9. Integrated analysis of aging onset genes and AD pathology")
print("\nNote: Uncomment relevant sections after running scBACs analysis.")