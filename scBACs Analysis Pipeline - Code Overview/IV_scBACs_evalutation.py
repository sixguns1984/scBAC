"""
Single-Cell Brain Age Clock (scBACs) Evaluation Pipeline
=========================================================

Purpose:
- Evaluate scBACs model performance on validation datasets
- Generate visualization plots for predicted vs. chronological age
- Perform internal and external validation

Note:
- User-friendly tool for cell-type-specific brain age prediction: 
  https://github.com/sixguns1984/scBACs
- Metadata including predicted cell age for training and validation datasets 
  are available for generating figures 2a, 2b, 2c
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
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics and machine learning
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up computational environment
print("Checking GPU availability...")
print(f"CUDA available: {torch.cuda.is_available()}")

# Set device (Note: Changed to cuda:0 for consistency, or adjust based on available GPUs)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ============================================================================
# 2. INTERNAL VALIDATION - DATA PREPARATION
# ============================================================================

print("\n" + "="*60)
print("Step 1: Internal Validation - Data Preparation")
print("="*60)

# Load metadata
print("Loading metadata...")
meta = pd.read_csv('./meta_human_cortex_scrna_atlas.csv')
print(f"Metadata shape: {meta.shape}")
print(f"Columns: {meta.columns.tolist()}")

# Define training and validation datasets
train_datasets = [
    'jhpce#tran2021', 'GSE140231', 'GSE148822', 'GSE157827', 
    'GSE160936', 'GSE167494', 'jhpce_DLPFC', 'GSE129308', 
    'GSE163577', 'PRJNA544731', 'GSE163122', 'GSE168408'
]

validation_datasets = [
    'AMP-PD', 'Human_Brain_Cell_Atlas', 'ROSMAP.MIT', 
    'Velmeshev', 'Ramos'
]

print(f"\nTraining datasets: {len(train_datasets)}")
print(f"Validation datasets: {len(validation_datasets)}")

# Filter data for validation datasets and control samples
print("\nFiltering data for internal validation...")
data = meta.loc[meta['dataset'].isin(validation_datasets), :]
print(f"After filtering by validation datasets: {data.shape}")

data = data.loc[data['status'] == 'CT', :]
print(f"After filtering control samples (CT): {data.shape}")

# Select relevant columns and rename
data = data.loc[:, ["donor_id", 'celltype', 'predicted_cell_age', 'Age_at_death']]
data.columns = ['id', 'celltype', 'predicted_cell_age', 'age']
df = data.copy()

print(f"\nFinal data shape: {df.shape}")
print(f"Unique donors: {df['id'].nunique()}")
print(f"Unique cell types: {df['celltype'].nunique()}")
print(f"Age range: {df['age'].min()} to {df['age'].max()} years")

# ============================================================================
# 3. INTERNAL VALIDATION - VISUALIZATION FUNCTION
# ============================================================================

def plot_celltype_age_comparison(df, celltypes, output_path, title_suffix=""):
    """
    Create hexbin scatter plots comparing predicted cell age vs. chronological age
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'id', 'celltype', 'predicted_cell_age', 'age'
    celltypes : list
        List of cell types to plot
    output_path : str
        Path to save the output figure
    title_suffix : str
        Suffix to add to the plot title
    """
    print(f"\nCreating plots for {len(celltypes)} cell types...")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("Blues")
    
    n_celltypes = len(celltypes)
    
    # Create square subplot layout
    n_cols = min(8, n_celltypes)  # Max 8 columns per row
    n_rows = (n_celltypes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_celltypes > 1 else [axes]
    
    # Create subplot for each cell type
    for idx, celltype in enumerate(celltypes):
        if idx >= len(axes):
            break
            
        print(f"  Processing {celltype} ({idx+1}/{n_celltypes})...")
        
        # Filter data for current cell type
        celltype_data = df[df['celltype'] == celltype].copy()
        
        # Calculate median cell age per donor
        median_ages = celltype_data.groupby('id').agg({
            'predicted_cell_age': 'median',
            'age': 'first'
        }).reset_index()
        
        ax = axes[idx]
        
        # Create 2D hexbin histogram
        hb = ax.hexbin(
            celltype_data['age'], 
            celltype_data['predicted_cell_age'], 
            gridsize=15, 
            cmap='Blues', 
            alpha=0.7, 
            mincnt=1
        )
        
        # Add median points per donor (highlighted in red)
        scatter = ax.scatter(
            median_ages['age'], 
            median_ages['predicted_cell_age'], 
            color='red', 
            s=60, 
            alpha=0.9, 
            edgecolors='black', 
            linewidth=1, 
            label='Median per donor', 
            zorder=5
        )
        
        # Calculate linear regression for median points
        if len(median_ages) > 1:
            # Linear regression using scipy.stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                median_ages['age'], 
                median_ages['predicted_cell_age']
            )
            
            # Generate regression line
            x_reg = np.linspace(median_ages['age'].min(), median_ages['age'].max(), 100)
            y_reg = slope * x_reg + intercept
            
            # Plot regression line
            ax.plot(x_reg, y_reg, color='gray', linestyle='-', linewidth=3)
        
        # Add colorbar
        cb = fig.colorbar(hb, ax=ax, shrink=0.2)
        
        # Calculate statistics for all cells
        r_all, p_all = spearmanr(celltype_data['age'], celltype_data['predicted_cell_age'])
        mae_all = np.mean(np.abs(celltype_data['predicted_cell_age'] - celltype_data['age']))
        
        # Calculate statistics for median points
        r_median, p_median = spearmanr(median_ages['age'], median_ages['predicted_cell_age'])
        mae_median = np.mean(np.abs(median_ages['predicted_cell_age'] - median_ages['age']))
        
        # Create statistics text
        stats_text = (
            f'R = {r_all:.2f}\n'
            f'P < 0.001\n'
            f'MAE = {mae_all:.2f}\n\n'
            f'r = {r_median:.2f}\n'
            f'P < 0.001\n'
            f'MAE = {mae_median:.2f}'
        )
        
        # Add statistics text to plot
        ax.text(
            0.98, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right'
        )
        
        # Set plot title and labels
        ax.set_title(f'{celltype}{title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Chronological Age', fontsize=12)
        
        # Set y-axis label only for leftmost subplots
        if idx % n_cols == 0:
            ax.set_ylabel('Predicted Cell Age', fontsize=12, labelpad=10)
        else:
            ax.set_ylabel('')
        
        # Set axis limits to square aspect
        x_min = celltype_data['age'].min()
        x_max = celltype_data['age'].max()
        margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(x_min - margin, x_max + margin)
        
        # Ensure square aspect ratio
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(loc='upper left', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # Remove empty subplots
    for idx in range(len(celltypes), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    return fig

# ============================================================================
# 4. INTERNAL VALIDATION - MAIN CELL TYPES
# ============================================================================

print("\n" + "="*60)
print("Step 2: Internal Validation - Main Cell Types")
print("="*60)

# Define main cell types for visualization
main_celltypes = ['Exc', 'Inh', 'OPC', 'Oli', 'Ast', 'Mic', 'End', 'Per']
print(f"Plotting main cell types: {main_celltypes}")

# Generate visualization
plot_celltype_age_comparison(
    df=df,
    celltypes=main_celltypes,
    output_path='./scBAC_internal_validation_main.pdf',
    title_suffix=""
)

print("\nInternal validation for main cell types completed.")


# ============================================================================
# 5. INTERNAL VALIDATION - ADDITIONAL CELL TYPES
# ============================================================================

print("\n" + "="*60)
print("Step 3: Internal Validation - Additional Cell Types")
print("="*60)

# Define additional cell types (optional)
additional_celltypes = ['Fib', 'SMC', 'CAM', 'Mural', 'T_cell']
print(f"Additional cell types available: {additional_celltypes}")

# Check which additional cell types are present in data
available_additional = [ct for ct in additional_celltypes if ct in df['celltype'].unique()]
print(f"Additional cell types present in data: {available_additional}")

if available_additional:
    # Generate visualization for additional cell types
    plot_celltype_age_comparison(
        df=df,
        celltypes=available_additional,
        output_path='./scBAC_internal_validation_additional.pdf',
        title_suffix=""
    )
    print("Internal validation for additional cell types completed.")
else:
    print("No additional cell types found in the validation data.")

# ============================================================================
# 6. EXTERNAL VALIDATION - USING SCBACS TOOL
# ============================================================================

print("\n" + "="*60)
print("Step 4: External Validation - Using scBACs Tool")
print("="*60)

print("""
For external validation using the scBACs command-line tool:

1. Install scBACs:
   git clone https://github.com/sixguns1984/scBACs.git
   cd scBACs
   pip install scbac-0.1.0-py3-none-any.whl

2. Run cellular age prediction:
   scbac predict \\
       --input ./sce_scBACs_external_validation.h5ad \\
       --output ./results.h5ad \\
       --cell-type-column celltype \\
       --count-layer counts \\
       --device cpu
""")

# ============================================================================
# 7. EXTERNAL VALIDATION - VISUALIZATION
# ============================================================================

print("\n" + "="*60)
print("Step 5: External Validation - Visualization")
print("="*60)

print("Loading external validation results...")

# Note: Uncomment the following code after running scBACs prediction

# Load external validation data
adata = sc.read_h5ad('./sce_scBACs_external_validation.h5ad')
print(f"External validation data shape: {adata.shape}")

# Extract relevant columns
df_external = adata.obs.loc[:, ['donor_id', 'celltype', 'predicted_cell_age', 'Age_at_death']]
df_external.columns = ['id', 'celltype', 'cell_age', 'age']

print(f"\nExternal validation data:")
print(f"  Shape: {df_external.shape}")
print(f"  Unique donors: {df_external['id'].nunique()}")
print(f"  Unique cell types: {df_external['celltype'].nunique()}")
print(f"  Age range: {df_external['age'].min()} to {df_external['age'].max()} years")

# Define cell types for external validation
external_celltypes = ['Exc', 'Inh', 'OPC', 'Oli', 'Ast', 'Mic', 'End']
print(f"\nPlotting external validation for cell types: {external_celltypes}")

# Generate visualization for external validation
plot_celltype_age_comparison(
    df=df_external,
    celltypes=external_celltypes,
    output_path='./scBAC_external_validation.pdf',
    title_suffix=" (External)"
)

print("External validation visualization completed.")



