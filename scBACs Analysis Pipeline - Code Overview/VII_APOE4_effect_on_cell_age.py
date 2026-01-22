"""
APOE4 Effect on Cellular Aging in Alzheimer's Disease
======================================================

Purpose:
- Analyze cell-type-specific APOE4 effect on cellular aging in AD progression
- Identify genes associated with APOE4 pathology in different AD stages
- Perform discovery and replication analyses in prefrontal cortex (PFC)

APOE Genotype encoding:
- APOE4 carriers: E3/E4, E2/E4, E4/E4 → encoded as 1
- APOE4 non-carriers: E2/E3, E3/E3, E2/E2 → encoded as 0

AD stages based on Braak stage:
- AD_1: Braak stage < 3 (early AD)
- AD_2: Braak stage 3-4 (intermediate AD)  
- AD_3: Braak stage > 4 (late AD)
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

# Core scientific computing libraries
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import os
import warnings

# Visualization
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics and modeling
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

# Machine learning
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up computational environment
print("Checking GPU availability...")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set configuration
sc.set_figure_params(figsize=(8, 8), frameon=False)

# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def prepare_apoe_data(metadata):
    """
    Prepare metadata for APOE4 analysis
    
    Parameters:
    -----------
    metadata : pandas.DataFrame
        Raw metadata with APOE genotype and clinical information
    
    Returns:
    --------
    prepared_meta : pandas.DataFrame
        Processed metadata with encoded variables
    """
    meta_processed = metadata.copy()
    
    # Encode sex (Male=1, Female=0)
    meta_processed['Sex'] = meta_processed['Sex'].replace(['Male', 'Female'], [1, 0]).astype('float')
    
    # Encode APOE4 status (carrier=1, non-carrier=0)
    apoe_mapping = {
        'E3/E4': 1, 'E2/E4': 1, 'E4/E4': 1,  # APOE4 carriers
        'E2/E3': 0, 'E3/E3': 0, 'E2/E2': 0   # APOE4 non-carriers
    }
    meta_processed['apoe'] = meta_processed['APOE Genotype'].replace(apoe_mapping).astype('float')
    
    # Define AD stages based on Braak stage
    meta_processed['status2'] = meta_processed['status'].astype('str').copy()
    
    # Early AD: Braak stage < 3
    meta_processed.loc[(meta_processed['status'] == 'AD') & 
                       (meta_processed['Braak_stage'] < 3), 'status2'] = 'AD_1'
    
    # Intermediate AD: Braak stage 3-4
    meta_processed.loc[(meta_processed['status'] == 'AD') & 
                       (meta_processed['Braak_stage'] < 5) & 
                       (meta_processed['Braak_stage'] > 2), 'status2'] = 'AD_2'
    
    # Late AD: Braak stage > 4
    meta_processed.loc[(meta_processed['status'] == 'AD') & 
                       (meta_processed['Braak_stage'] > 4), 'status2'] = 'AD_3'
    
    return meta_processed


def analyze_apoe_effect(meta_data, celltypes, stages, analysis_name="discovery"):
    """
    Analyze APOE4 effect on cellular aging using linear regression
    
    Parameters:
    -----------
    meta_data : pandas.DataFrame
        Prepared metadata with encoded variables
    celltypes : list
        List of cell types to analyze
    stages : list
        List of AD stages to analyze
    analysis_name : str
        Name of analysis for output file
    
    Returns:
    --------
    result_df : pandas.DataFrame
        Linear regression results for APOE4 effect
    """
    print(f"\nPerforming {analysis_name} analysis of APOE4 effect...")
    
    # Initialize results dataframe
    n_combinations = len(celltypes) * len(stages)
    result = pd.DataFrame(
        np.zeros((n_combinations, 4)),
        columns=['stage', 'celltype', 'coef', 'pvalue']
    )
    
    m = 0
    print(f"  Analyzing {n_combinations} cell type × stage combinations")
    
    for i, celltype in enumerate(celltypes):
        for j, stage in enumerate(stages):
            # Filter data for current cell type and stage
            meta_temp = meta_data.loc[
                (meta_data['status2'] == stage) & 
                (meta_data['celltype'] == celltype), :
            ].copy()
            
            # Skip if insufficient data
            if meta_temp.shape[0] < 10:
                result.iloc[m, 0] = stage
                result.iloc[m, 1] = celltype
                result.iloc[m, 2] = np.nan
                result.iloc[m, 3] = np.nan
                m += 1
                continue
            
            # Prepare polynomial terms and interactions
            meta_temp['age2'] = meta_temp['Age_at_death'] * meta_temp['Age_at_death']  # Age squared
            meta_temp['Sex*age'] = meta_temp['Sex'] * meta_temp['Age_at_death']
            meta_temp['Sex*age2'] = meta_temp['Sex'] * meta_temp['age2']
            meta_temp['apoe*age'] = meta_temp['apoe'] * meta_temp['Age_at_death']
            meta_temp['apoe*age2'] = meta_temp['apoe'] * meta_temp['age2']
            
            # Define predictors for linear regression
            predictors = [
                'apoe', 'Age_at_death', 'Sex', 'age2',
                'Sex*age', 'Sex*age2', 'apoe*age', 'apoe*age2'
            ]
            
            x = meta_temp.loc[:, predictors]
            x_with_const = sm.add_constant(x)  # Add intercept term
            
            try:
                # Perform OLS regression: predicted_cell_age ~ APOE4 + covariates
                model = sm.OLS(meta_temp['predicted_cell_age'], x_with_const)
                fit0 = model.fit()
                
                # Store APOE4 coefficient and p-value
                result.iloc[m, 0] = stage
                result.iloc[m, 1] = celltype
                result.iloc[m, 2] = fit0.params['apoe']  # APOE4 effect size
                result.iloc[m, 3] = fit0.pvalues['apoe']  # P-value for APOE4 effect
                
            except Exception as e:
                print(f"    Error in {celltype}/{stage}: {str(e)[:50]}...")
                result.iloc[m, 0] = stage
                result.iloc[m, 1] = celltype
                result.iloc[m, 2] = np.nan
                result.iloc[m, 3] = np.nan
            
            m += 1
    
    # Clean results (remove NaN)
    result_clean = result.dropna(subset=['coef', 'pvalue'])
    print(f"  Valid results: {result_clean.shape[0]}/{n_combinations}")
    
    return result_clean


def find_apoe4_deg(adata, celltypes, stages, output_file):
    """
    Find differentially expressed genes associated with APOE4 status
    
    Parameters:
    -----------
    adata : AnnData
        Single-cell expression data with metadata
    celltypes : list
        List of cell types to analyze
    stages : list
        List of AD stages to analyze
    output_file : str
        Path to save results
    
    Returns:
    --------
    all_results : pandas.DataFrame
        Differential expression results
    """
    print(f"\nFinding APOE4-associated differentially expressed genes...")
    
    # Prepare APOE4 status for DEG analysis
    adata.obs['apoe'] = adata.obs['apoe'].replace([0, 1], ['N', 'Y'])
    
    all_results = pd.DataFrame()
    
    for celltype in celltypes:
        for stage in stages:
            print(f"  Processing {celltype} in {stage}...")
            
            # Subset data for current cell type and stage
            subset_mask = (adata.obs['celltype'] == celltype) & (adata.obs['status2'] == stage)
            sce = adata[subset_mask, :].copy()
            
            # Skip if insufficient samples
            if sce.shape[0] < 20 or sce.obs['apoe'].nunique() < 2:
                print(f"    Skipping: insufficient data (n={sce.shape[0]})")
                continue
            
            # Normalize expression data
            sc.pp.normalize_total(sce)
            sc.pp.log1p(sce)
            
            try:
                # Perform differential expression analysis (APOE4 carriers vs non-carriers)
                sc.tl.rank_genes_groups(sce, "apoe", method="wilcoxon", pts=True)
                
                # Extract results
                rank_results = sce.uns['rank_genes_groups']
                group_labels = sce.obs[rank_results['params']['groupby']].unique()
                
                for group_label in group_labels:
                    # Get DEGs for current comparison
                    group_df = sc.get.rank_genes_groups_df(sce, group=group_label)
                    group_df = group_df.sort_values(by="scores", ascending=False)
                    
                    # Filter for genes expressed in at least 10% of cells
                    pts = sce.uns['rank_genes_groups']['pts'][group_label]
                    selected_genes = pts[pts >= 0.1].index.tolist()
                    group_df = group_df[group_df['names'].isin(selected_genes)]
                    
                    # Add metadata
                    group_df['group'] = group_label  # APOE4 status (Y/N)
                    group_df['celltype'] = celltype
                    group_df['status'] = stage
                    
                    all_results = pd.concat([all_results, group_df], ignore_index=True)
                    
                print(f"    Found {group_df.shape[0]} DEGs")
                
            except Exception as e:
                print(f"    Error in DEG analysis: {str(e)[:50]}...")
    
    # Save results
    if all_results.shape[0] > 0:
        all_results.to_csv(output_file, index=False)
        print(f"\nDEG results saved to: {output_file}")
        print(f"Total DEGs found: {all_results.shape[0]}")
    else:
        print("\nNo DEGs found")
    
    return all_results

# ============================================================================
# 3. DISCOVERY ANALYSIS: APOE4 EFFECT ON CELLULAR AGING
# ============================================================================

print("\n" + "="*60)
print("Step 1: Discovery Analysis - APOE4 Effect on Cellular Aging in PFC")
print("="*60)

# Load discovery metadata
print("Loading discovery metadata...")
meta_discovery = pd.read_csv(
    './meta_APOE4_aging_discovery.csv',
    index_col=0
)
print(f"Discovery metadata shape: {meta_discovery.shape}")

# Prepare data for analysis
meta_discovery_prep = prepare_apoe_data(meta_discovery)

# Define cell types and stages for discovery analysis
discovery_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Oli', 'OPC', 'Per', 'Mic']
discovery_stages = meta_discovery_prep['status2'].unique()

print(f"\nDiscovery analysis parameters:")
print(f"  Cell types: {discovery_celltypes}")
print(f"  AD stages: {discovery_stages}")
print(f"  Total samples: {meta_discovery_prep.shape[0]}")
print(f"  APOE4 carriers: {(meta_discovery_prep['apoe'] == 1).sum()}")
print(f"  APOE4 non-carriers: {(meta_discovery_prep['apoe'] == 0).sum()}")

# Analyze APOE4 effect on cellular aging
discovery_results = analyze_apoe_effect(
    meta_data=meta_discovery_prep,
    celltypes=discovery_celltypes,
    stages=discovery_stages,
    analysis_name="discovery"
)

# Save discovery results
if discovery_results.shape[0] > 0:
    discovery_results.to_csv('./apoe4_on_cell_aging_dis.csv', index=False)
    print(f"\nDiscovery results saved to: './apoe4_on_cell_aging_dis.csv'")
    
    # Display significant findings
    significant_discovery = discovery_results[discovery_results['pvalue'] < 0.05]
    if significant_discovery.shape[0] > 0:
        print(f"\nSignificant APOE4 effects (p < 0.05): {significant_discovery.shape[0]}")
        print("\nTop significant associations:")
        print(significant_discovery.sort_values('pvalue').head().to_string(index=False))
    else:
        print("\nNo significant APOE4 effects found (p < 0.05)")

# ============================================================================
# 4. DISCOVERY ANALYSIS: GENES ASSOCIATED WITH APOE4 PATHOLOGY
# ============================================================================

print("\n" + "="*60)
print("Step 2: Discovery Analysis - Genes Associated with APOE4 Pathology")
print("="*60)

print("Loading single-cell data for DEG analysis...")

# Load single-cell data
test1 = sc.read_h5ad('./sce_test1_0.h5ad')
test2 = sc.read_h5ad('./sce_test1_1.h5ad')
test3 = sc.read_h5ad('./sce_test2.h5ad')
test4 = sc.read_h5ad('./sce_test3.h5ad')

# Combine datasets
adata_discovery = sc.concat([test1, test2, test3, test4], axis=0)

# Align with metadata
adata_discovery = adata_discovery[meta_discovery_prep.index, :]
adata_discovery.obs = meta_discovery_prep.copy()

print(f"Combined single-cell data shape: {adata_discovery.shape}")

# Define cell types and stages for DEG analysis
deg_celltypes = ['Ast', 'Exc', 'Inh', 'Oli', 'OPC', 'Mic']  # Focus on major cell types
deg_stages = adata_discovery.obs['status2'].unique()

# Find APOE4-associated differentially expressed genes
discovery_deg = find_apoe4_deg(
    adata=adata_discovery,
    celltypes=deg_celltypes,
    stages=deg_stages,
    output_file='./dis_APOE4_stages_celltypes_DEG_scanpy.csv'
)

# ============================================================================
# 5. REPLICATION ANALYSIS: APOE4 EFFECT ON CELLULAR AGING
# ============================================================================

print("\n" + "="*60)
print("Step 3: Replication Analysis - APOE4 Effect on Cellular Aging in PFC")
print("="*60)

# Load replication data
print("Loading replication data...")
adata_replication = sc.read_h5ad(
    './sce_APOE4_aging_replication.h5ad'
)

# Extract metadata
meta_replication = adata_replication.obs.copy()
print(f"Replication metadata shape: {meta_replication.shape}")

# Prepare replication data
meta_replication_prep = prepare_apoe_data(meta_replication)

# Define cell types and stages for replication analysis
replication_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Oli', 'OPC', 'Per', 'Mic']
replication_stages = meta_replication_prep['status2'].unique()

print(f"\nReplication analysis parameters:")
print(f"  Cell types: {replication_celltypes}")
print(f"  AD stages: {replication_stages}")
print(f"  Total samples: {meta_replication_prep.shape[0]}")
print(f"  APOE4 carriers: {(meta_replication_prep['apoe'] == 1).sum()}")
print(f"  APOE4 non-carriers: {(meta_replication_prep['apoe'] == 0).sum()}")

# Analyze APOE4 effect on cellular aging (replication)
replication_results = analyze_apoe_effect(
    meta_data=meta_replication_prep,
    celltypes=replication_celltypes,
    stages=replication_stages,
    analysis_name="replication"
)

# Save replication results
if replication_results.shape[0] > 0:
    replication_results.to_csv('./apoe4_on_cell_aging_rep.csv', index=False)
    print(f"\nReplication results saved to: './apoe4_on_cell_aging_rep.csv'")
    
    # Display significant findings
    significant_replication = replication_results[replication_results['pvalue'] < 0.05]
    if significant_replication.shape[0] > 0:
        print(f"\nSignificant APOE4 effects (p < 0.05): {significant_replication.shape[0]}")
        print("\nTop significant associations:")
        print(significant_replication.sort_values('pvalue').head().to_string(index=False))
    else:
        print("\nNo significant APOE4 effects found (p < 0.05)")

# ============================================================================
# 6. REPLICATION ANALYSIS: GENES ASSOCIATED WITH APOE4 PATHOLOGY
# ============================================================================

print("\n" + "="*60)
print("Step 4: Replication Analysis - Genes Associated with APOE4 Pathology")
print("="*60)

# Prepare replication data for DEG analysis
print("Preparing replication data for DEG analysis...")

# Update APOE4 status in adata for DEG analysis
adata_replication.obs = meta_replication_prep.copy()

# Define cell types and stages for replication DEG analysis
rep_deg_celltypes = ['Ast', 'Exc', 'Inh', 'Oli', 'OPC', 'Mic']
rep_deg_stages = adata_replication.obs['status2'].unique()

# Find APOE4-associated differentially expressed genes (replication)
replication_deg = find_apoe4_deg(
    adata=adata_replication,
    celltypes=rep_deg_celltypes,
    stages=rep_deg_stages,
    output_file='./rep_APOE4_stages_celltypes_DEG_scanpy.csv'
)

# ============================================================================
# 7. SUMMARY AND INTEGRATION
# ============================================================================

print("\n" + "="*60)
print("Step 5: Summary and Integration of Results")
print("="*60)

print("\nAnalysis completed successfully!")

print("\nGenerated output files:")
print("  1. './apoe4_on_cell_aging_dis.csv' - Discovery analysis of APOE4 effect")
print("  2. './dis_APOE4_stages_celltypes_DEG_scanpy.csv' - Discovery DEG analysis")
print("  3. './apoe4_on_cell_aging_rep.csv' - Replication analysis of APOE4 effect")
print("  4. './rep_APOE4_stages_celltypes_DEG_scanpy.csv' - Replication DEG analysis")

# Note: For further integration of discovery and replication results,
# you could compare effect sizes and identify consistently significant findings

print("\n" + "="*60)
print("APOE4 Effect Analysis Pipeline Complete!")
print("="*60)
print("\nSummary:")
print(f"- Analyzed {len(discovery_celltypes)} cell types across AD progression stages")
print(f"- Performed both discovery and replication analyses")
print(f"- Investigated APOE4 effects on cellular aging and gene expression")
print(f"- Results saved to 4 output files for further investigation")