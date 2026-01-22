"""
APOE4 Effect on Aging Acceleration Onset in Alzheimer's Disease
================================================================

Purpose:
- Analyze differences in aging acceleration onset between APOE4 carriers and non-carriers in AD
- Identify genes associated with aging onset in APOE4 carriers vs non-carriers
- Perform discovery and replication analyses for validation

APOE Genotype encoding:
- APOE4 carriers: E3/E4, E2/E4, E4/E4 → encoded as Y
- APOE4 non-carriers: E2/E3, E3/E3, E2/E2 → encoded as N

Cell types analyzed:
- Ast (Astrocytes)
- End (Endothelial cells)
- Exc (Excitatory neurons)
- Inh (Inhibitory neurons)
- Oli (Oligodendrocytes)
- OPC (Oligodendrocyte precursor cells)
- Per (Pericytes)
- Mic (Microglia)
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

# Core scientific computing libraries
import numpy as np
import pandas as pd
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

def prepare_apoe4_data(meta_path, adata_path=None):
    """
    Prepare APOE4 data for aging acceleration analysis
    
    Parameters:
    -----------
    meta_path : str
        Path to metadata CSV file
    adata_path : str, optional
        Path to AnnData file (if available)
    
    Returns:
    --------
    meta_df : pandas.DataFrame
        Processed metadata with APOE4 encoding
    adata : AnnData or None
        Single-cell data (if provided)
    """
    print(f"Loading metadata from: {meta_path}")
    meta_df = pd.read_csv(meta_path, index_col=0)
    
    # Filter for AD samples
    meta_df = meta_df[meta_df['status'] == 'AD'].copy()
    
    # Encode APOE4 status
    apoe_mapping = {
        'E3/E4': 'Y', 'E2/E4': 'Y', 'E4/E4': 'Y',  # APOE4 carriers
        'E2/E3': 'N', 'E3/E3': 'N', 'E2/E2': 'N'   # APOE4 non-carriers
    }
    
    if 'APOE Genotype' in meta_df.columns:
        meta_df['apoe4'] = meta_df['APOE Genotype'].replace(apoe_mapping)
        print(f"APOE4 carriers: {(meta_df['apoe4'] == 'Y').sum()}")
        print(f"APOE4 non-carriers: {(meta_df['apoe4'] == 'N').sum()}")
    else:
        print("Warning: APOE Genotype column not found in metadata")
        meta_df['apoe4'] = np.nan
    
    # Filter for target cell types
    target_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Oli', 'OPC', 'Per', 'Mic']
    meta_df = meta_df[meta_df['celltype'].isin(target_celltypes)]
    
    print(f"\nData summary:")
    print(f"  Total AD samples: {meta_df.shape[0]}")
    print(f"  Unique donors: {meta_df['donor_id'].nunique()}")
    print(f"  Cell types: {meta_df['celltype'].nunique()}")
    
    # Load single-cell data if path provided
    adata = None
    if adata_path:
        print(f"\nLoading single-cell data from: {adata_path}")
        adata = sc.read_h5ad(adata_path)
        # Align with metadata
        common_cells = np.intersect1d(meta_df.index, adata.obs.index)
        adata = adata[common_cells, :]
        meta_df = meta_df.loc[common_cells, :]
        adata.obs = meta_df.copy()
        print(f"  Aligned single-cell data shape: {adata.shape}")
    
    return meta_df, adata

# ============================================================================
# 3. DISCOVERY ANALYSIS SETUP
# ============================================================================

print("\n" + "="*60)
print("Step 2: Discovery Analysis Setup")
print("="*60)

# Discovery data paths
discovery_meta_path = '/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/dataset/meta_APOE4_aging_discovery.csv'

# Prepare discovery data
meta_discovery, adata_discovery = prepare_apoe4_data(
    meta_path=discovery_meta_path,
    adata_path=None  # Update with actual path if available
)

# Save data for scBACs analysis
discovery_output_path = './APOE4_discovery_predicted_ages.csv'
meta_discovery.to_csv(discovery_output_path)
print(f"\nDiscovery data saved for scBACs analysis: {discovery_output_path}")

# ============================================================================
# 4. SCBACS ANALYSIS - DISCOVERY (APOE4 CARRIERS VS NON-CARRIERS)
# ============================================================================

print("\n" + "="*60)
print("Step 3: scBACs Analysis - Discovery (APOE4 Carriers vs Non-Carriers)")
print("="*60)

print("""
# linux command: Analyze aging acceleration onset in APOE4 carriers vs non-carriers

# For APOE4 carriers (apoe4 = Y)
scbac analyze \
    --input ./APOE4_discovery_predicted_ages.csv \
    --output-prefix ./APOE4_carrier_discovery_results \
    --cell-age-pred-col predicted_cell_age \
    --disease-name Y \
    --status-col apoe4 \
    --chronological-age-col Age_at_death \
    --donor-col donor_id \
    --celltype-col celltype 


# For APOE4 non-carriers (apoe4 = N)
scbac analyze \
    --input ./APOE4_discovery_predicted_ages.csv \
    --output-prefix ./APOE4_noncarrier_discovery_results \
    --cell-age-pred-col predicted_cell_age \
    --disease-name N \
    --status-col apoe4 \
    --chronological-age-col Age_at_death \
    --donor-col donor_id \
    --celltype-col celltype

""")

print("Note: Run the above commands in Linux terminal to perform scBACs analysis.")
print("      This will generate separate threshold files for APOE4 carriers and non-carriers.")

# ============================================================================
# 5. REPLICATION ANALYSIS SETUP
# ============================================================================

print("\n" + "="*60)
print("Step 4: Replication Analysis Setup")
print("="*60)

# Replication data paths
replication_adata_path = '/public/labdata/luojunfeng/project_data/spatial_pvm/tool/scMerge/Cell_Brain_age/Total_cell_analysis/prepare_for_paper_submit/dataset/sce_APOE4_aging_replication.h5ad'

# Load replication data
print("Loading replication data...")
adata_replication = sc.read_h5ad(replication_adata_path)
meta_replication = adata_replication.obs.copy()

target_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Oli', 'OPC', 'Per', 'Mic']
meta_replication = meta_replication[meta_replication['celltype'].isin(target_celltypes)]

# Use the loaded metadata
meta_replication['apoe4'] = meta_replication['APOE Genotype'].replace({
    'E3/E4': 'Y', 'E2/E4': 'Y', 'E4/E4': 'Y',
    'E2/E3': 'N', 'E3/E3': 'N', 'E2/E2': 'N'
})
meta_replication = meta_replication.loc[meta_replication['status']=='AD',:]
# Save data for scBACs analysis
replication_output_path = './APOE4_replication_predicted_ages.csv'
meta_replication.to_csv(replication_output_path)
print(f"\nReplication data saved for scBACs analysis: {replication_output_path}")

# ============================================================================
# 6. SCBACS ANALYSIS - REPLICATION
# ============================================================================

print("\n" + "="*60)
print("Step 5: scBACs Analysis - Replication")
print("="*60)

print("""
# linux command: Replication analysis for APOE4 carriers vs non-carriers

# For APOE4 carriers (apoe4 = Y)
scbac analyze \
    --input ./APOE4_replication_predicted_ages.csv \
    --output-prefix ./APOE4_carrier_replication_results \
    --cell-age-pred-col predicted_cell_age \
    --disease-name Y \
    --status-col apoe4 \
    --chronological-age-col Age_at_death \
    --donor-col donor_id \
    --celltype-col celltype 


# For APOE4 non-carriers (apoe4 = N)
scbac analyze \
    --input ./APOE4_replication_predicted_ages.csv \
    --output-prefix ./APOE4_noncarrier_replication_results \
    --cell-age-pred-col predicted_cell_age \
    --disease-name N \
    --status-col apoe4 \
    --chronological-age-col Age_at_death \
    --donor-col donor_id \
    --celltype-col celltype 

""")

# ============================================================================
# 7. COMPARE AGING ACCELERATION ONSET: APOE4 CARRIERS VS NON-CARRIERS
# ============================================================================

print("\n" + "="*60)
print("Step 6: Compare Aging Acceleration Onset: APOE4 Carriers vs Non-Carriers")
print("="*60)

print("Comparing aging acceleration onset between APOE4 carriers and non-carriers...")


def compare_apoe4_thresholds(carrier_path, noncarrier_path, analysis_name="discovery"):
    """
    Compare threshold ages between APOE4 carriers and non-carriers
    """
    print(f"\n{analysis_name.upper()} Analysis - APOE4 Carriers vs Non-Carriers")
    
    # Load threshold data
    carrier_thresholds = pd.read_csv(carrier_path, index_col=0)
    noncarrier_thresholds = pd.read_csv(noncarrier_path, index_col=0)
    
    print(f"  APOE4 carriers: {carrier_thresholds['donor_id'].nunique()} donors")
    print(f"  APOE4 non-carriers: {noncarrier_thresholds['donor_id'].nunique()} donors")
    
    # Define cell types for comparison
    celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Oli', 'OPC', 'Per', 'Mic']
    
    comparison_results = []
    
    for celltype in celltypes:
        # Get data for current cell type
        carrier_data = carrier_thresholds.loc[
            carrier_thresholds['celltype'] == celltype, 'threshold_age'
        ].values
        
        noncarrier_data = noncarrier_thresholds.loc[
            noncarrier_thresholds['celltype'] == celltype, 'threshold_age'
        ].values
        
        # Skip if insufficient data
        if len(carrier_data) < 3 or len(noncarrier_data) < 3:
            print(f"  {celltype}: Insufficient data (carrier: {len(carrier_data)}, non-carrier: {len(noncarrier_data)})")
            continue
        
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(carrier_data, noncarrier_data, alternative='two-sided')
        
        # Calculate effect size (Hedges' g)
        mean_diff = carrier_data.mean() - noncarrier_data.mean()
        pooled_std = np.sqrt(
            ((len(carrier_data)-1)*carrier_data.std()**2 + (len(noncarrier_data)-1)*noncarrier_data.std()**2) /
            (len(carrier_data) + len(noncarrier_data) - 2)
        )
        hedges_g = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Store results
        result = {
            'celltype': celltype,
            'analysis': analysis_name,
            'n_carrier': len(carrier_data),
            'n_noncarrier': len(noncarrier_data),
            'mean_carrier': carrier_data.mean(),
            'mean_noncarrier': noncarrier_data.mean(),
            'mean_diff': mean_diff,
            'hedges_g': hedges_g,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        comparison_results.append(result)
        
        print(f"\n  {celltype}:")
        print(f"    APOE4 carriers: n={len(carrier_data)}, mean={carrier_data.mean():.1f}±{carrier_data.std():.1f}")
        print(f"    APOE4 non-carriers: n={len(noncarrier_data)}, mean={noncarrier_data.mean():.1f}±{noncarrier_data.std():.1f}")
        print(f"    Mean difference: {mean_diff:.1f} years")
        print(f"    Hedges' g: {hedges_g:.3f}")
        print(f"    Mann-Whitney U test: p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(comparison_results)
    
    # Save results
    output_file = f'./APOE4_{analysis_name}_threshold_comparison.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nComparison results saved to: {output_file}")
    
    # Summary
    significant_celltypes = results_df[results_df['significant']]['celltype'].tolist()
    if significant_celltypes:
        print(f"\nSignificant differences found in: {len(significant_celltypes)} cell types")
        print("Cell types with significant APOE4 effect:")
        for ct in significant_celltypes:
            ct_result = results_df[results_df['celltype'] == ct].iloc[0]
            direction = "later" if ct_result['mean_diff'] > 0 else "earlier"
            print(f"  - {ct}: APOE4 carriers show {direction} aging onset (p={ct_result['p_value']:.4f})")
    else:
        print("\nNo significant differences found between APOE4 carriers and non-carriers")
    
    return results_df

# Compare discovery data
print("\n--- DISCOVERY ANALYSIS ---")
discovery_carrier_path = './APOE4_carrier_discovery_results_thresholds.csv'
discovery_noncarrier_path = './APOE4_noncarrier_discovery_results_thresholds.csv'

discovery_comparison = compare_apoe4_thresholds(
    carrier_path=discovery_carrier_path,
    noncarrier_path=discovery_noncarrier_path,
    analysis_name="discovery"
)

# Compare replication data
print("\n--- REPLICATION ANALYSIS ---")
replication_carrier_path = './APOE4_carrier_replication_results_thresholds.csv'
replication_noncarrier_path = './APOE4_noncarrier_replication_results_thresholds.csv'

replication_comparison = compare_apoe4_thresholds(
    carrier_path=replication_carrier_path,
    noncarrier_path=replication_noncarrier_path,
    analysis_name="replication"
)

# Combine discovery and replication results
if discovery_comparison.shape[0] > 0 and replication_comparison.shape[0] > 0:
    combined_results = pd.concat([discovery_comparison, replication_comparison], ignore_index=True)
    
    # Identify consistent findings
    consistent_results = []
    for celltype in discovery_comparison['celltype'].unique():
        disc_result = discovery_comparison[discovery_comparison['celltype'] == celltype]
        rep_result = replication_comparison[replication_comparison['celltype'] == celltype]
        
        if disc_result.shape[0] > 0 and rep_result.shape[0] > 0:
            disc_sig = disc_result['significant'].iloc[0]
            rep_sig = rep_result['significant'].iloc[0]
            same_direction = (disc_result['mean_diff'].iloc[0] * rep_result['mean_diff'].iloc[0]) > 0
            
            if disc_sig and rep_sig and same_direction:
                consistent_results.append(celltype)
    
    if consistent_results:
        print(f"\nConsistent significant findings (discovery and replication):")
        for ct in consistent_results:
            disc_mean_diff = discovery_comparison[discovery_comparison['celltype'] == ct]['mean_diff'].iloc[0]
            direction = "earlier" if disc_mean_diff > 0 else "later"
            print(f"  - {ct}: APOE4 carriers show {direction} aging onset in both cohorts")
    else:
        print("\nNo consistent significant findings between discovery and replication")

# ============================================================================
# 8. GENES ASSOCIATED WITH AGING ONSET IN COMBINED APOE4 CARRIER/NON-CARRIER AD COHORT
# ============================================================================

print("\n" + "="*60)
print("Step 7: Genes Associated with Aging Onset in Combined APOE4 AD Cohort")
print("="*60)

def analyze_combined_aging_onset_genes(adata, thresholds, celltype, analysis_name):
    """
    Analyze genes associated with aging acceleration onset in combined APOE4 carrier/non-carrier AD cohort
    
    Parameters:
    -----------
    adata : AnnData
        Single-cell expression data (AD samples only, both APOE4 carrier and non-carrier)
    thresholds_path : str
        Path to threshold ages CSV file (combined analysis)
    celltype : str
        Target cell type
    analysis_name : str
        Name of analysis for output files ('discovery' or 'replication')
    
    Returns:
    --------
    gene_results : pandas.DataFrame
        Gene-level correlation results
    """
    print(f"\nAnalyzing genes associated with aging onset in {celltype} (combined APOE4 cohort)...")
    
    # Load threshold data (from combined APOE4 analysis)
    thresholds = thresholds[thresholds['celltype'] == celltype]
    thresholds.index = thresholds['donor_id'].values
    
    # Filter single-cell data for target cell type (AD samples only)
    cell_mask = (adata.obs['celltype'] == celltype) & (adata.obs['status'] == 'AD')
    sce = adata[cell_mask, :].copy()
    
    if sce.shape[0] < 30:  # Higher threshold for combined analysis
        print(f"  Insufficient cells: {sce.shape[0]}")
        return pd.DataFrame()
    
    print(f"  Cells included: {sce.shape[0]}")
    print(f"  APOE4 carriers: {(sce.obs['apoe4'] == 1).sum()}")
    print(f"  APOE4 non-carriers: {(sce.obs['apoe4'] == 0).sum()}")
    
    # Normalize expression data
    sc.pp.normalize_per_cell(sce)
    sc.pp.log1p(sce)
    
    # Prepare data for correlation analysis
    obs = sce.obs.copy()
    rna = sce.to_df()
    
    # Calculate Spearman correlation for each gene
    print(f"  Calculating correlations for {rna.shape[1]} genes...")
    r_values = []
    gene_names = []
    p_values = []
    
    for i in range(rna.shape[1]):
        # Aggregate expression by donor
        obs['gene'] = rna.iloc[:, i].values.copy()
        obs2 = obs.groupby(['donor_id'], observed=True)[['gene']].mean()
        
        # Find common donors between expression data and threshold data
        common_donors = np.intersect1d(thresholds.index, obs2.index)
        if len(common_donors) < 8:  # Need more donors for combined analysis
            continue
        
        thresholds_subset = thresholds.loc[common_donors, :]
        obs2 = obs2.loc[common_donors]
        
        # Calculate correlation between gene expression and threshold age
        coef, p_value = spearmanr(obs2.iloc[:, 0].values, thresholds_subset['threshold_age'].values)
        r_values.append(coef)
        p_values.append(p_value)
        gene_names.append(rna.columns[i])
    
    # Create results dataframe
    if len(gene_names) == 0:
        print("  No genes passed filtering criteria")
        return pd.DataFrame()
    
    gene_results = pd.DataFrame({
        'genename': gene_names,
        'r': r_values,
        'p': p_values
    })
    
    # Apply FDR correction
    _, fdr_p, _, _ = multipletests(gene_results['p'].values, method='fdr_bh')
    gene_results['fdr'] = fdr_p
    
    # Sort by absolute correlation coefficient
    gene_results['abs_r'] = gene_results['r'].abs()
    gene_results = gene_results.sort_values('abs_r', ascending=False)
    
    # Save results
    output_file = f'./{analysis_name}_combined_{celltype}_aging_onset_genes.csv'
    gene_results.to_csv(output_file, index=False)
    
    print(f"  Results saved to: {output_file}")
    print(f"  Genes analyzed: {gene_results.shape[0]}")
    print(f"  Significant genes (FDR < 0.05): {(gene_results['fdr'] < 0.05).sum()}")
    
    # Show top genes
    top_genes = gene_results.head(10)
    print(f"\n  Top 10 genes by correlation strength:")
    for idx, row in top_genes.iterrows():
        direction = "positive" if row['r'] > 0 else "negative"
        print(f"    {row['genename']}: r = {row['r']:.3f} ({direction}), FDR = {row['fdr']:.4f}")
    
    return gene_results

# ============================================================================
# 9. APOE4 CARRIER VS NON-CARRIER DIFFERENTIAL EXPRESSION ANALYSIS IN OLI
# ============================================================================

print("\n" + "="*60)
print("Step 8: APOE4 Carrier vs Non-Carrier Differential Expression Analysis in Oli")
print("="*60)

def analyze_apoe4_degs_oli(adata, analysis_name="discovery"):
    """
    Perform differential expression analysis between APOE4 carriers and non-carriers in Oli cells
    
    Parameters:
    -----------
    adata : AnnData
        Single-cell expression data (AD samples only)
    analysis_name : str
        Name of analysis for output files
    
    Returns:
    --------
    deg_results : pandas.DataFrame
        Differential expression results
    """
    print(f"\nPerforming APOE4 carrier vs non-carrier DEG analysis in Oli cells ({analysis_name})...")
    
    # Filter for Oli cells in AD samples
    oli_mask = (adata.obs['celltype'] == 'Oli') & (adata.obs['status'] == 'AD')
    sce_oli = adata[oli_mask, :].copy()
    
    if sce_oli.shape[0] < 50:
        print(f"  Insufficient Oli cells: {sce_oli.shape[0]}")
        return pd.DataFrame()
    
    # Check APOE4 carrier status
    apoe4_carriers = (sce_oli.obs['apoe4'] == 1).sum()
    apoe4_noncarriers = (sce_oli.obs['apoe4'] == 0).sum()
    
    print(f"  Oli cells analyzed: {sce_oli.shape[0]}")
    print(f"  APOE4 carriers: {apoe4_carriers}")
    print(f"  APOE4 non-carriers: {apoe4_noncarriers}")
    
    if apoe4_carriers < 10 or apoe4_noncarriers < 10:
        print("  Insufficient samples in one or both groups for DEG analysis")
        return pd.DataFrame()
    
    # Prepare APOE4 status for DEG analysis
    sce_oli.obs['apoe4_status'] = sce_oli.obs['apoe4'].replace({1: 'carrier', 0: 'noncarrier'})
    
    # Normalize expression data
    sc.pp.normalize_total(sce_oli)
    sc.pp.log1p(sce_oli)
    
    # Perform differential expression analysis
    try:
        sc.tl.rank_genes_groups(sce_oli, "apoe4_status", method="wilcoxon", pts=True)
        
        # Extract results for both comparisons
        rank_results = sce_oli.uns['rank_genes_groups']
        
        # Get results for APOE4 carriers vs non-carriers
        deg_carrier = sc.get.rank_genes_groups_df(sce_oli, group='carrier')
        deg_noncarrier = sc.get.rank_genes_groups_df(sce_oli, group='noncarrier')
        
        # Combine results
        deg_carrier['comparison'] = 'carrier_vs_noncarrier'
        deg_noncarrier['comparison'] = 'noncarrier_vs_carrier'
        
        # Filter for genes expressed in at least 10% of cells
        pts_carrier = sce_oli.uns['rank_genes_groups']['pts']['carrier']
        pts_noncarrier = sce_oli.uns['rank_genes_groups']['pts']['noncarrier']
        
        # Use minimum expression threshold
        min_pts = 0.1
        expressed_genes_carrier = pts_carrier[pts_carrier >= min_pts].index.tolist()
        expressed_genes_noncarrier = pts_noncarrier[pts_noncarrier >= min_pts].index.tolist()
        expressed_genes = list(set(expressed_genes_carrier) | set(expressed_genes_noncarrier))
        
        deg_carrier = deg_carrier[deg_carrier['names'].isin(expressed_genes)]
        deg_noncarrier = deg_noncarrier[deg_noncarrier['names'].isin(expressed_genes)]
        
        # Combine results
        all_degs = pd.concat([deg_carrier, deg_noncarrier], ignore_index=True)
        
        # Add metadata
        all_degs['celltype'] = 'Oli'
        all_degs['analysis'] = analysis_name
        all_degs['n_cells'] = sce_oli.shape[0]
        all_degs['n_carrier'] = apoe4_carriers
        all_degs['n_noncarrier'] = apoe4_noncarriers
        
        # Save results
        output_file = f'./{analysis_name}_Oli_APOE4_carrier_vs_noncarrier_DEG.csv'
        all_degs.to_csv(output_file, index=False)
        
        print(f"  DEG results saved to: {output_file}")
        print(f"  Total DEGs analyzed: {all_degs.shape[0]}")
        
        # Filter significant DEGs (p-adjusted < 0.05)
        sig_degs = all_degs[all_degs['pvals_adj'] < 0.05]
        print(f"  Significant DEGs (padj < 0.05): {sig_degs.shape[0]}")
        
        if sig_degs.shape[0] > 0:
            # Separate up and down regulated genes
            up_genes = sig_degs[sig_degs['logfoldchanges'] > 0]
            down_genes = sig_degs[sig_degs['logfoldchanges'] < 0]
            
            print(f"    Up-regulated in carriers: {up_genes.shape[0]} genes")
            print(f"    Down-regulated in carriers: {down_genes.shape[0]} genes")
            
            # Save significant gene lists
            up_genes['names'].to_csv(f'./{analysis_name}_Oli_APOE4carrier_up_genes.txt', 
                                    index=False, header=False)
            down_genes['names'].to_csv(f'./{analysis_name}_Oli_APOE4carrier_down_genes.txt', 
                                      index=False, header=False)
            
            # Show top DEGs
            print(f"\n  Top 10 DEGs (by absolute log fold change):")
            top_degs = sig_degs.sort_values('logfoldchanges', key=abs, ascending=False).head(10)
            for idx, row in top_degs.iterrows():
                direction = "up" if row['logfoldchanges'] > 0 else "down"
                print(f"    {row['names']}: logFC = {row['logfoldchanges']:.3f} ({direction}), padj = {row['pvals_adj']:.4f}")
        
        return all_degs
        
    except Exception as e:
        print(f"  Error in DEG analysis: {str(e)}")
        return pd.DataFrame()

# ============================================================================
# 10. INTEGRATE AGING ONSET GENES WITH APOE4 DEGs IN OLI
# ============================================================================

print("\n" + "="*60)
print("Step 9: Integrate Aging Onset Genes with APOE4 DEGs in Oli")
print("="*60)

def integrate_oli_aging_onset_with_apoe4_degs(aging_genes_path, deg_path, analysis_name):
    """
    Integrate aging onset genes with APOE4 DEGs specifically in Oli cells
    
    Parameters:
    -----------
    aging_genes_path : str
        Path to combined aging onset gene results for Oli
    deg_path : str
        Path to APOE4 carrier vs non-carrier DEG results for Oli
    analysis_name : str
        Name of analysis ('discovery' or 'replication')
    """
    print(f"\nIntegrating aging onset genes with APOE4 DEGs in Oli ({analysis_name})...")
    
    # Load aging onset genes for Oli
    try:
        aging_genes = pd.read_csv(aging_genes_path)
    except FileNotFoundError:
        print(f"  Aging onset genes file not found: {aging_genes_path}")
        return
    
    # Load APOE4 DEGs for Oli
    try:
        apoe4_degs = pd.read_csv(deg_path)
    except FileNotFoundError:
        print(f"  APOE4 DEGs file not found: {deg_path}")
        return
    
    # Filter significant genes
    aging_sig = aging_genes[aging_genes['fdr'] < 0.05].copy()
    # Use comparison where carriers are reference (logFC > 0 means up in carriers)
    degs_sig = apoe4_degs[(apoe4_degs['pvals_adj'] < 0.05) & 
                          (apoe4_degs['comparison'] == 'carrier_vs_noncarrier')].copy()
    
    print(f"  Significant aging onset genes in Oli: {aging_sig.shape[0]}")
    print(f"  Significant APOE4 DEGs in Oli: {degs_sig.shape[0]}")
    
    if aging_sig.shape[0] == 0 or degs_sig.shape[0] == 0:
        print("  Insufficient significant genes for integration")
        return
    
    # Find overlapping genes
    overlap_genes = np.intersect1d(aging_sig['genename'].values, degs_sig['names'].values)
    
    if len(overlap_genes) > 0:
        print(f"\n  Found {len(overlap_genes)} overlapping genes:")
        
        # Create detailed overlap table
        overlap_details = []
        for gene in overlap_genes:
            aging_info = aging_sig[aging_sig['genename'] == gene].iloc[0]
            deg_info = degs_sig[degs_sig['names'] == gene].iloc[0]
            
            # Determine biological consistency

            # Then it's consistent
            if (aging_info['r'] > 0 and deg_info['logfoldchanges'] > 0) 
                direction_consistency = 'consistent'
            else:
                direction_consistency = 'opposite'
            
            # Determine biological interpretation
            if aging_info['r'] > 0:
                aging_effect = f"later onset (r=+{aging_info['r']:.3f})"
            else:
                aging_effect = f"earlier onset (r={aging_info['r']:.3f})"
            
            if deg_info['logfoldchanges'] > 0:
                apoe4_effect = f"up in carriers (logFC=+{deg_info['logfoldchanges']:.3f})"
            else:
                apoe4_effect = f"down in carriers (logFC={deg_info['logfoldchanges']:.3f})"
            
            overlap_details.append({
                'gene': gene,
                'aging_r': aging_info['r'],
                'aging_fdr': aging_info['fdr'],
                'aging_effect': aging_effect,
                'deg_logfc': deg_info['logfoldchanges'],
                'deg_padj': deg_info['pvals_adj'],
                'apoe4_effect': apoe4_effect,
                'direction_consistency': direction_consistency,
                'biological_interpretation': f"{aging_effect}, {apoe4_effect}"
            })
        
        overlap_df = pd.DataFrame(overlap_details)
        
        # Save overlap results
        overlap_file = f'./{analysis_name}_Oli_APOE4_integrated_overlap_genes.csv'
        overlap_df.to_csv(overlap_file, index=False)
        print(f"  Overlap results saved to: {overlap_file}")
        
        # Categorize genes by pattern
        consistent_genes = overlap_df[overlap_df['direction_consistency'] == 'consistent']
        opposite_genes = overlap_df[overlap_df['direction_consistency'] == 'opposite']
        
        print(f"  Consistent direction: {consistent_genes.shape[0]} genes")
        print(f"  Opposite direction: {opposite_genes.shape[0]} genes")
        
        # Display consistent genes with biological interpretation
        if consistent_genes.shape[0] > 0:
            print(f"\n  Consistent genes (biologically meaningful):")
            for idx, row in consistent_genes.iterrows():
                print(f"    {row['gene']}: {row['biological_interpretation']}")
            
            # Save gene lists for pathway analysis
            consistent_genes['gene'].to_csv(
                f'./{analysis_name}_Oli_APOE4_consistent_genes.txt',
                index=False, header=False
            )
        
        if opposite_genes.shape[0] > 0:
            print(f"\n  Opposite direction genes:")
            for idx, row in opposite_genes.iterrows():
                print(f"    {row['gene']}: {row['biological_interpretation']}")
            
            opposite_genes['gene'].to_csv(
                f'./{analysis_name}_Oli_APOE4_opposite_genes.txt',
                index=False, header=False
            )
        
    else:
        print("  No overlapping genes found")

# ============================================================================
# 11. DISCOVERY ANALYSIS EXECUTION
# ============================================================================

print("\n" + "="*60)
print("Step 10: Discovery Analysis Execution")
print("="*60)

print("Note: Uncomment and execute after running scBACs combined analysis")


test1 = sc.read_h5ad('./sce_test1_0.h5ad')
test2 = sc.read_h5ad('./sce_test1_1.h5ad')
test3 = sc.read_h5ad('./sce_test2.h5ad')
test4 = sc.read_h5ad('./sce_test3.h5ad')

# Combine datasets
adata_discovery = sc.concat([test1, test2, test3, test4], axis=0)


APOE4_carrie = pd.read_csv('./APOE4_carrier_discovery_results_thresholds.csv', index_col=0)
APOE4_noncarrier = pd.read_csv( './APOE4_noncarrier_discovery_results_thresholds.csv', index_col=0)
threshold_df = pd.concat([APOE4_carrie,APOE4_noncarrier],axis=0)

# After running scBACs, perform gene analysis
if adata_discovery is not None:
    # Analyze aging onset genes for key cell types in combined cohort
    celltypes_to_analyze = ['Oli']
    
    for celltype in celltypes_to_analyze:
        combined_genes = analyze_combined_aging_onset_genes(
            adata=adata_discovery,
            thresholds = threshold_df,
            celltype=celltype,
            analysis_name='discovery'
        )

# 11.2 APOE4 carrier vs non-carrier DEG analysis in Oli (discovery)
print("\n10.2 APOE4 Carrier vs Non-Carrier DEG Analysis in Oli (Discovery)")

if adata_discovery is not None:
    oli_degs_discovery = analyze_apoe4_degs_oli(
        adata=adata_discovery,
        analysis_name='discovery'
    )

# 11.3 Integration analysis for Oli (discovery)
print("\n10.3 Integration Analysis for Oli (Discovery)")

integrate_oli_aging_onset_with_apoe4_degs(
    aging_genes_path='./discovery_combined_Oli_aging_onset_genes.csv',
    deg_path='./discovery_Oli_APOE4_carrier_vs_noncarrier_DEG.csv',
    analysis_name='discovery'
)


# ============================================================================
# 12. REPLICATION ANALYSIS EXECUTION
# ============================================================================

print("\n" + "="*60)
print("Step 11: Replication Analysis Execution")
print("="*60)

adata_replication = sc.read_h5ad(
    './sce_APOE4_aging_replication.h5ad'
)



APOE4_carrie = pd.read_csv('./APOE4_carrier_replication_results_thresholds.csv', index_col=0)
APOE4_noncarrier = pd.read_csv( './APOE4_noncarrier_replication_results_thresholds.csv', index_col=0)
threshold_df = pd.concat([APOE4_carrie,APOE4_noncarrier],axis=0)

# After running scBACs, perform gene analysis
if adata_replication is not None:
    # Focus on Oli for replication
    combined_genes_replication = analyze_combined_aging_onset_genes(
        adata=adata_replication,
        thresholds = threshold_df,
        celltype='Oli',
        analysis_name='replication'
    )

# 12.2 APOE4 carrier vs non-carrier DEG analysis in Oli (replication)
print("\n11.2 APOE4 Carrier vs Non-Carrier DEG Analysis in Oli (Replication)")

if adata_replication is not None:
    oli_degs_replication = analyze_apoe4_degs_oli(
        adata=adata_replication,
        analysis_name='replication'
    )

# 12.3 Integration analysis for Oli (replication)
print("\n11.3 Integration Analysis for Oli (Replication)")

integrate_oli_aging_onset_with_apoe4_degs(
    aging_genes_path='./replication_combined_Oli_aging_onset_genes.csv',
    deg_path='./replication_Oli_APOE4_carrier_vs_noncarrier_DEG.csv',
    analysis_name='replication'


# ============================================================================
# 13. DISCOVERY-REPLICATION CONSISTENCY ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("Step 12: Discovery-Replication Consistency Analysis")
print("="*60)

def check_discovery_replication_consistency():
    """
    Check consistency between discovery and replication results for Oli
    """
    print("\nChecking consistency between discovery and replication results...")
    
    # Check aging onset genes consistency
    try:
        disc_aging = pd.read_csv('./discovery_combined_Oli_aging_onset_genes.csv')
        rep_aging = pd.read_csv('./replication_combined_Oli_aging_onset_genes.csv')
        
        # Find common significant genes
        disc_sig = set(disc_aging[disc_aging['fdr'] < 0.05]['genename'])
        rep_sig = set(rep_aging[rep_aging['fdr'] < 0.05]['genename'])
        
        common_aging_genes = disc_sig.intersection(rep_sig)
        print(f"  Common significant aging onset genes: {len(common_aging_genes)}")
        
        # Check correlation direction consistency
        consistent_direction = 0
        for gene in common_aging_genes:
            disc_r = disc_aging[disc_aging['genename'] == gene]['r'].iloc[0]
            rep_r = rep_aging[rep_aging['genename'] == gene]['r'].iloc[0]
            if (disc_r > 0 and rep_r > 0):
                consistent_direction += 1
        
        print(f"  Genes with consistent direction: {consistent_direction}/{len(common_aging_genes)}")
        
    except FileNotFoundError as e:
        print(f"  Warning: Aging onset gene files not found: {e}")
    
    # Check DEG consistency
    try:
        disc_deg = pd.read_csv('./discovery_Oli_APOE4_carrier_vs_noncarrier_DEG.csv')
        rep_deg = pd.read_csv('./replication_Oli_APOE4_carrier_vs_noncarrier_DEG.csv')
        
        # Filter significant DEGs
        disc_sig_deg = disc_deg[(disc_deg['pvals_adj'] < 0.05) & 
                                 (disc_deg['comparison'] == 'carrier_vs_noncarrier')]
        rep_sig_deg = rep_deg[(rep_deg['pvals_adj'] < 0.05) & 
                               (rep_deg['comparison'] == 'carrier_vs_noncarrier')]
        
        common_degs = set(disc_sig_deg['names']).intersection(set(rep_sig_deg['names']))
        print(f"  Common significant DEGs: {len(common_degs)}")
        
        # Check direction consistency
        consistent_deg_direction = 0
        for gene in common_degs:
            disc_fc = disc_sig_deg[disc_sig_deg['names'] == gene]['logfoldchanges'].iloc[0]
            rep_fc = rep_sig_deg[rep_sig_deg['names'] == gene]['logfoldchanges'].iloc[0]
            if (disc_fc > 0 and rep_fc > 0):
                consistent_deg_direction += 1
        
        print(f"  DEGs with consistent direction: {consistent_deg_direction}/{len(common_degs)}")
        
    except FileNotFoundError as e:
        print(f"  Warning: DEG files not found: {e}")
    
    # Check integrated results consistency
    try:
        disc_integrated = pd.read_csv('./discovery_Oli_APOE4_integrated_overlap_genes.csv')
        rep_integrated = pd.read_csv('./replication_Oli_APOE4_integrated_overlap_genes.csv')
        
        common_integrated = set(disc_integrated['gene']).intersection(set(rep_integrated['gene']))
        print(f"  Common integrated genes: {len(common_integrated)}")
        
        if len(common_integrated) > 0:
            print(f"\n  Consistently identified integrated genes:")
            for gene in common_integrated:
                disc_info = disc_integrated[disc_integrated['gene'] == gene].iloc[0]
                rep_info = rep_integrated[rep_integrated['gene'] == gene].iloc[0]
                print(f"    {gene}:")
                print(f"      Discovery: {disc_info['biological_interpretation']}")
                print(f"      Replication: {rep_info['biological_interpretation']}")
    
    except FileNotFoundError as e:
        print(f"  Warning: Integrated results files not found: {e}")

print("Note: Run this function after completing both discovery and replication analyses")
"""
# Execute consistency check
check_discovery_replication_consistency()
"""

# ============================================================================
print("\n" + "="*60)
print("APOE4 Aging Analysis Pipeline Complete!")
print("="*60)
print("\nSummary of modified analysis steps:")
print("  1. Combined aging onset gene analysis (APOE4 carriers + non-carriers in AD)")
print("  2. APOE4 carrier vs non-carrier differential expression analysis in Oli only")
print("  3. Integration of aging onset genes with APOE4 DEGs in Oli")
print("  4. Discovery cohort analysis execution")
print("  5. Replication cohort analysis execution")
print("  6. Discovery-replication consistency analysis")
print("\nFocus on Oli cells for:")
print("  - APOE4 carrier vs non-carrier DEG analysis")
print("  - Integrated analysis of aging onset and APOE4 effects")
print("  - Therapeutic target identification")
