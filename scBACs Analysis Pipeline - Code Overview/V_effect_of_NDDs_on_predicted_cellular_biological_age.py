"""
Effect of Neurodegenerative Diseases on Predicted Cellular Biological Age
=======================================================================

Purpose:
- Analyze region- and cell-type-specific accelerated aging in neurodegenerative diseases
- Assess the impact of cellular aging on cognitive function (MMSE) in AD
- Perform linear regression analyses to quantify disease effects

Neurodegenerative diseases analyzed:
- AD: Alzheimer's Disease
- PD: Parkinson's Disease  
- MCI: Mild Cognitive Impairment
- FTD: Frontotemporal Dementia
- ALS: Amyotrophic Lateral Sclerosis
- FTLD: Frontotemporal Lobar Degeneration
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

# Visualization libraries
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics and modeling
import statsmodels.api as sm
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds and configuration
sc.set_figure_params(figsize=(8, 8), frameon=False)

# ============================================================================
# 2. DATA LOADING
# ============================================================================

print("\n" + "="*60)
print("Step 1: Data Loading")
print("="*60)

# Load metadata
print("Loading metadata...")
meta = pd.read_csv('./meta_human_cortex_scrna_atlas.csv', index_col=0)
print(f"Metadata shape: {meta.shape}")
print(f"Available disease statuses: {meta['status'].unique()}")
print(f"Available cell types: {meta['celltype'].nunique()}")
print(f"Available brain regions: {meta['sub_tissue'].nunique()}")

# ============================================================================
# 3. REGION- AND CELL-TYPE-SPECIFIC ACCELERATED AGING ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("Step 2: Region- and Cell-type-specific Accelerated Aging Analysis")
print("="*60)

def analyze_disease_effect(ND_disease, meta_data):
    """
    Analyze the effect of a neurodegenerative disease on cellular aging
    
    Parameters:
    -----------
    ND_disease : str
        Neurodegenerative disease code (e.g., 'AD', 'PD', 'MCI', 'FTD', 'ALS', 'FTLD')
    meta_data : pandas.DataFrame
        Metadata containing predicted cell age and clinical information
    
    Returns:
    --------
    result_df : pandas.DataFrame
        Results of linear regression analysis
    """
    print(f"\nAnalyzing {ND_disease} effect on cellular aging...")
    
    # Identify cohorts with the target disease
    nd_cohorts = meta_data.loc[meta_data['status'] == ND_disease, 'dataset'].unique()
    print(f"  Found {len(nd_cohorts)} datasets with {ND_disease}")
    
    # Filter data for disease and control groups
    meta_filtered = meta_data.loc[meta_data['dataset'].isin(nd_cohorts), :]
    df = meta_filtered.loc[(meta_filtered['status'] == ND_disease) | 
                           (meta_filtered['status'] == 'CT'), :]
    
    print(f"  Data shape after filtering: {df.shape}")
    print(f"  {ND_disease} samples: {df[df['status'] == ND_disease].shape[0]}")
    print(f"  Control samples: {df[df['status'] == 'CT'].shape[0]}")
    
    # Select relevant columns and prepare data
    df = df.loc[:, ['celltype', 'predicted_cell_age', 'status', 'Sex', 'Age_at_death', 'sub_tissue']]
    df = df.dropna()
    
    # Encode categorical variables
    df['Sex'] = df['Sex'].replace(['Male', 'Female'], [1, 0]).astype('float')
    df['age2'] = df['Age_at_death'] * df['Age_at_death']  # Age squared term
    df['Sex*age'] = df['Sex'] * df['Age_at_death']
    df['Sex*age2'] = df['Sex'] * df['age2']
    
    # Create disease group indicator (1 = disease, 0 = control)
    df['group'] = 0
    df.loc[df['status'] == ND_disease, 'group'] = 1
    df['group*age'] = df['group'] * df['Age_at_death']
    df['group*age2'] = df['group'] * df['age2']
    
    # Get unique cell types and tissues
    unique_celltypes = df['celltype'].unique()
    unique_tissues = df['sub_tissue'].unique()
    
    print(f"  Unique cell types: {len(unique_celltypes)}")
    print(f"  Unique brain regions: {len(unique_tissues)}")
    
    # Initialize results dataframe
    n_combinations = len(unique_celltypes) * len(unique_tissues)
    result = pd.DataFrame(
        np.zeros((n_combinations, 4)),
        columns=['celltype', 'tissue', 'effect', 'p_value']
    )
    
    # Perform linear regression for each cell type and tissue combination
    m = 0
    print(f"\n  Performing {n_combinations} linear regressions...")
    
    for i, celltype in enumerate(unique_celltypes):
        for j, tissue in enumerate(unique_tissues):
            # Filter data for current cell type and tissue
            df2 = df.loc[(df['sub_tissue'] == tissue) & (df['celltype'] == celltype), :]
            
            # Ensure sufficient control samples
            if df2.loc[df2['status'] == 'CT', :].shape[0] < 100:
                # Add control samples from same cell type across all tissues
                ct_samples = df.loc[(df['status'] == 'CT') & (df['celltype'] == celltype), :]
                df2 = pd.concat([df2, ct_samples], axis=0)
                print(f"    {celltype} in {tissue}: Added controls from other regions")
            
            if df2.shape[0] < 20:  # Skip if insufficient data
                result.iloc[m, 0] = celltype
                result.iloc[m, 1] = tissue
                result.iloc[m, 2] = np.nan
                result.iloc[m, 3] = np.nan
                m += 1
                continue
            
            # Prepare independent variables
            predictors = ['group', 'Sex', 'Age_at_death', 'age2', 'Sex*age', 'Sex*age2', 'group*age', 'group*age2']
            x_with_const = sm.add_constant(df2.loc[:, predictors])
            
            try:
                # Perform OLS regression
                model = sm.OLS(df2['predicted_cell_age'], x_with_const)
                fit0 = model.fit()
                
                # Store results
                result.iloc[m, 0] = celltype
                result.iloc[m, 1] = tissue
                result.iloc[m, 2] = fit0.params['group']  # Disease effect size
                result.iloc[m, 3] = fit0.pvalues['group']  # P-value for disease effect
                
            except Exception as e:
                print(f"    Error in {celltype}/{tissue}: {str(e)[:50]}...")
                result.iloc[m, 0] = celltype
                result.iloc[m, 1] = tissue
                result.iloc[m, 2] = np.nan
                result.iloc[m, 3] = np.nan
            
            m += 1
    
    # Add disease status to results
    result['status'] = ND_disease
    
    # Remove rows with NaN results
    result_clean = result.dropna(subset=['effect', 'p_value'])
    print(f"\n  Valid results: {result_clean.shape[0]}/{n_combinations}")
    
    return result_clean

# ============================================================================
# 4. ANALYZE EFFECT OF NEURODEGENERATIVE DISEASES
# ============================================================================

print("\n" + "="*60)
print("Step 3: Analyze Neurodegenerative Diseases")
print("="*60)

# Define neurodegenerative diseases to analyze
neurodegenerative_diseases = ['AD', 'PD', 'MCI', 'FTD', 'ALS', 'FTLD']
print(f"Analyzing {len(neurodegenerative_diseases)} neurodegenerative diseases")

# Analyze each disease
all_results = []

for disease in neurodegenerative_diseases:
    print(f"\nProcessing {disease}...")
    
    # Check if disease exists in data
    if disease not in meta['status'].unique():
        print(f"  Warning: {disease} not found in data, skipping...")
        continue
    
    # Analyze disease effect
    disease_result = analyze_disease_effect(disease, meta)
    
    if disease_result.shape[0] > 0:
        # Save individual disease results
        output_file = f'./{disease}_effect_on_cell_aging_linear.csv'
        disease_result.to_csv(output_file, index=False)
        print(f"  Results saved to: {output_file}")
        
        all_results.append(disease_result)

# Combine all results if multiple diseases were analyzed
if len(all_results) > 0:
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv('./all_NDDs_effect_on_cell_aging_combined.csv', index=False)
    print(f"\nCombined results saved to: './all_NDDs_effect_on_cell_aging_combined.csv'")



# ============================================================================
# 5. IMPACT OF CELLULAR AGING ON COGNITIVE FUNCTION IN AD
# ============================================================================

print("\n" + "="*60)
print("Step 4: Impact of Cellular Aging on Cognitive Function in AD")
print("="*60)

# Focus on Alzheimer's Disease (AD)
ND = 'AD'
print(f"Analyzing impact of cellular aging on MMSE in {ND}...")

# Filter AD data
if ND not in meta['status'].unique():
    print(f"Warning: {ND} not found in data, skipping cognitive function analysis.")
else:
    df_ad = meta.loc[meta['status'] == ND, :].copy()
    print(f"  AD data shape: {df_ad.shape}")
    
    # Prepare education years variable
    df_ad['eduyr'] = df_ad['eduyr'].astype('float')
    
    # Calculate median values per donor, tissue, and cell type
    print("  Aggregating data by donor, tissue, and cell type...")
    participant_celltype_median = df_ad.groupby(
        ['donor_id', 'sub_tissue', 'celltype']
    )[['predicted_cell_age', 'Age_at_death', 'MMSE', 'eduyr']].median().reset_index()
    
    # Add sex information
    participant_status = df_ad[['donor_id', 'status', 'Sex']].drop_duplicates()
    df_ad_agg = participant_celltype_median.merge(participant_status, on='donor_id')
    
    # Encode sex (Male=1, Female=0)
    df_ad_agg['Sex'] = df_ad_agg['Sex'].replace(['Male', 'Female'], [1.0, 0])
    
    print(f"  Aggregated data shape: {df_ad_agg.shape}")
    print(f"  Unique donors: {df_ad_agg['donor_id'].nunique()}")
    print(f"  Unique cell types: {df_ad_agg['celltype'].nunique()}")
    print(f"  Unique tissues: {df_ad_agg['sub_tissue'].nunique()}")
    
    # Define cell types and tissues for analysis
    target_celltypes = ['Ast', 'End', 'Exc', 'Inh', 'Mic', 'OPC', 'Oli', 'Per']
    target_tissues = df_ad_agg['sub_tissue'].unique()[:4]  # Use first 4 tissues if available
    
    print(f"  Analyzing {len(target_celltypes)} cell types")
    print(f"  Analyzing {len(target_tissues)} brain regions")
    
    # Initialize results dataframe
    result_mmse = pd.DataFrame(
        np.zeros((len(target_celltypes) * len(target_tissues), 4)),
        columns=['celltype', 'tissue', 'coef', 'pvalue']
    )
    
    # Perform linear regression for each combination
    m = 0
    print(f"\n  Performing {len(target_celltypes) * len(target_tissues)} regressions...")
    
    for i, celltype in enumerate(target_celltypes):
        for j, tissue in enumerate(target_tissues):
            # Filter data for current cell type and tissue
            subset = df_ad_agg.loc[
                (df_ad_agg['celltype'] == celltype) & 
                (df_ad_agg['sub_tissue'] == tissue), :
            ]
            
            if subset.shape[0] < 10:  # Skip if insufficient data
                result_mmse.iloc[m, 0] = celltype
                result_mmse.iloc[m, 1] = tissue
                result_mmse.iloc[m, 2] = np.nan
                result_mmse.iloc[m, 3] = np.nan
                m += 1
                continue
            
            # Prepare predictors for regression
            predictors = ['Age_at_death', 'predicted_cell_age', 'eduyr', 'Sex']
            x_with_const = sm.add_constant(subset[predictors])
            x_with_const = x_with_const.dropna()
            
            if x_with_const.shape[0] < 5:  # Skip if insufficient data after dropping NAs
                result_mmse.iloc[m, 0] = celltype
                result_mmse.iloc[m, 1] = tissue
                result_mmse.iloc[m, 2] = np.nan
                result_mmse.iloc[m, 3] = np.nan
                m += 1
                continue
            
            try:
                # Perform OLS regression: MMSE ~ Age + PredictedCellAge + Education + Sex
                model = sm.OLS(subset['MMSE'].iloc[:x_with_const.shape[0]], x_with_const)
                fit0 = model.fit()
                
                # Store coefficient for predicted_cell_age (age-adjusted cellular aging effect)
                result_mmse.iloc[m, 0] = celltype
                result_mmse.iloc[m, 1] = tissue
                result_mmse.iloc[m, 2] = fit0.params['predicted_cell_age']
                result_mmse.iloc[m, 3] = fit0.pvalues['predicted_cell_age']
                
            except Exception as e:
                print(f"    Error in {celltype}/{tissue}: {str(e)[:50]}...")
                result_mmse.iloc[m, 0] = celltype
                result_mmse.iloc[m, 1] = tissue
                result_mmse.iloc[m, 2] = np.nan
                result_mmse.iloc[m, 3] = np.nan
            
            m += 1
    
    # Clean results
    result_mmse_clean = result_mmse.dropna(subset=['coef', 'pvalue'])
    
    if result_mmse_clean.shape[0] > 0:
        # Save results
        output_file = './cell_aging_on_ad_mmse_linear_regression_cov_adjust.csv'
        result_mmse_clean.to_csv(output_file, index=False)
        
        print(f"\n  Results saved to: {output_file}")
        print(f"  Valid results: {result_mmse_clean.shape[0]}")
        
        # Display significant findings
        significant = result_mmse_clean[result_mmse_clean['pvalue'] < 0.05]
        if significant.shape[0] > 0:
            print(f"\n  Significant associations (p < 0.05): {significant.shape[0]}")
            print("\n  Top significant associations:")
            print(significant.sort_values('pvalue').head().to_string(index=False))
        else:
            print("\n  No significant associations found (p < 0.05)")
    else:
        print("\n  No valid results obtained")

# ============================================================================
# 6. SUMMARY AND VISUALIZATION PREPARATION
# ============================================================================

print("\n" + "="*60)
print("Step 5: Summary and Visualization Preparation")
print("="*60)

print("\nAnalysis completed successfully!")
print("\nGenerated output files:")

# List expected output files
expected_files = []
for disease in neurodegenerative_diseases:
    if disease in meta['status'].unique():
        expected_files.append(f"./{disease}_effect_on_cell_aging_linear.csv")

if 'AD' in meta['status'].unique():
    expected_files.append("./cell_aging_on_ad_mmse_linear_regression_cov_adjust.csv")

print("\n".join([f"  - {f}" for f in expected_files]))

print("\n" + "="*60)
print("Analysis Pipeline Complete!")
print("="*60)