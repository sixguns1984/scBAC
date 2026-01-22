"""
Whole-life Pseudo-bulk Gene Expression Trajectory Analysis Pipeline
===================================================================

Purpose:
- Identify age-associated genes for each cell type
- Perform pseudo-bulk analysis to generate gene expression trajectories across lifespan
- Cluster trajectories to identify distinct aging patterns
- Visualize trajectory clusters by cell type

Input:
- Single-cell RNA-seq data (.h5ad format)
- Cell type annotations and donor metadata

Output:
- Age-associated genes list
- Gene expression trajectories across ages
- Clustered trajectory patterns
- Visualization plots
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
import scipy.io as sio
import matplotlib.pyplot as plt
import warnings

# Single-cell analysis tools
import scvi

# Visualization libraries
import seaborn as sns

# Machine learning and statistics
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.interpolate import BSpline, splrep
from scipy.stats import pearsonr, spearmanr

# Set up computational environment
print(torch.cuda.is_available())  # Check CUDA availability
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Set random seeds and configuration
scvi.settings.seed = 0
scvi.settings.num_threads = 4
sc.set_figure_params(figsize=(8, 8), frameon=False)

# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def batch_sce_norm(sce, batch_key=None):
    """
    Batch-wise normalization of single-cell expression data
    
    Parameters:
    -----------
    sce : AnnData
        Single-cell expression data
    batch_key : str
        Column name in sce.obs for batch information
    
    Returns:
    --------
    sce : AnnData
        Normalized expression data
    """
    # Filter cells with at least one count
    sc.pp.filter_cells(sce, min_counts=1)
    
    # Initialize normalized layer
    if scipy.sparse.issparse(sce.X):
        # Create sparse matrix placeholder
        sce.layers['normalized'] = scipy.sparse.csr_matrix(sce.shape, dtype=np.float32)
    else:
        # Create dense matrix placeholder
        sce.layers['normalized'] = np.zeros(sce.shape, dtype=np.float32)
    
    # Normalize each batch separately
    for batch in np.unique(sce.obs[batch_key]):
        print(f"  Normalizing batch: {batch}")
        
        # Extract sub-batch data
        batch_mask = sce.obs[batch_key] == batch
        sce_batch = sce[batch_mask].copy()
        
        # Perform normalization (scanpy standard pipeline)
        sc.pp.normalize_per_cell(sce_batch)
        
        # Store normalized data in new layer
        sce.layers['normalized'][batch_mask] = sce_batch.X
    
    # Replace original data with normalized data
    sce.X = sce.layers['normalized'].copy()
    
    return sce


# ============================================================================
# 3. IDENTIFY AGE-ASSOCIATED GENES
# ============================================================================

print("\n" + "="*60)
print("Step 1: Identifying Age-associated Genes")
print("="*60)

# Load training data
print("Loading training data...")
adata_train = sc.read_h5ad('./sce_train.h5ad')

# Select only control (CT) samples for age association analysis
adata_train = adata_train[adata_train.obs['status'] == 'CT', :].copy()
print(f"Training data shape (CT samples only): {adata_train.shape}")

# Initialize list to store results
all_age_genes = []

# Identify age-associated genes for each cell type
print("\nIdentifying age-associated genes for each cell type:")
cell_types = adata_train.obs['celltype'].unique()
print(f"Total cell types: {len(cell_types)}")

for i, cell_type in enumerate(cell_types, 1):
    print(f"\n  Processing cell type {i}/{len(cell_types)}: {cell_type}")
    
    # Subset data for current cell type (CORRECTED: changed adata to adata_train)
    adata_temp = adata_train[adata_train.obs['celltype'] == cell_type, :].copy()
    
    # Normalize data
    try:
        adata_temp = batch_sce_norm(adata_temp, batch_key='dataset')
    except Exception as e:
        print(f"    Batch normalization failed, using simple normalization: {e}")
        sc.pp.normalize_per_cell(adata_temp)
    
    # Log-transform and scale
    sc.pp.log1p(adata_temp)
    sc.pp.scale(adata_temp)
    
    # Prepare data for correlation analysis
    obs = adata_temp.obs.copy()
    df = adata_temp.to_df()
    
    # Calculate Spearman correlation between gene expression and age
    print(f"    Calculating correlations for {df.shape[1]} genes...")
    r = []
    genename = []
    p = []
    
    for j in range(df.shape[1]):
        obs['gene'] = df.iloc[:, j].values.copy()
        
        # Aggregate by donor (pseudo-bulk)
        obs2 = obs.groupby(['donor_id'], observed=True)[['gene', 'Age_at_death']].mean()
        
        # Calculate Spearman correlation
        coef, p_value = spearmanr(obs2.iloc[:, 0].values, obs2.iloc[:, 1].values)
        r.append(coef)
        p.append(p_value)
        genename.append(df.columns[j])
    
    # Store results
    result = pd.DataFrame({'genename': genename, 'r': r, 'p': p})
    result['celltype'] = cell_type  # Corrected: was 'i', but 'cell_type' is more descriptive
    
    # Filter significant age-associated genes
    age_genes_sig = result.loc[(result['r'].abs() > 0.3) & (result['p'] < 0.05), :]
    print(f"    Found {len(age_genes_sig)} significant age-associated genes")
    
    all_age_genes.append(result)

# Combine results from all cell types
age_genes = pd.concat(all_age_genes, ignore_index=True)
age_genes.to_csv('./human_cortex_age_genes.csv', index=False)
print(f"\nTotal age-associated genes saved to './human_cortex_age_genes.csv'")
print(f"Total significant associations: {len(age_genes)}")


# ============================================================================
# 4. PSEUDO-BULK GENE EXPRESSION TRAJECTORY ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("Step 2: Whole-life Pseudo-bulk Gene Expression Trajectory Analysis")
print("="*60)

# Load age-associated genes
print("Loading age-associated genes...")
age_genes_df = pd.read_csv('./human_cortex_age_genes.csv')
unique_age_genes = age_genes_df['genename'].unique()
print(f"Unique age-associated genes: {len(unique_age_genes)}")

# Filter to genes present in the data
common_genes = np.intersect1d(unique_age_genes, adata_train.var.index)
print(f"Genes present in data: {len(common_genes)}")

# Subset data to age-associated genes
sce = adata_train[:, common_genes].copy()
sce.obs['age'] = sce.obs['Age_at_death'].copy()

# ============================================================================
# 4.1 EXTRACT GENE EXPRESSION TRAJECTORIES
# ============================================================================

print("\nExtracting mean aging expression trajectories for gene-celltype combinations...")

missingness_threshold = 0.3  # Allow up to 30% missing ages

from scipy.sparse import issparse

def fast_age_trajectories(sce, missingness_threshold=0.3):
    """
    Extract gene expression trajectories across ages for each gene-celltype pair
    
    Parameters:
    -----------
    sce : AnnData
        Single-cell expression data with age and celltype annotations
    missingness_threshold : float
        Maximum allowed proportion of missing ages
    
    Returns:
    --------
    labels : list
        Labels in format "gene_celltype"
    age_expn_vectors : list
        Expression trajectories across ages
    """
    # Get unique ages (sorted for consistency)
    ages = np.sort(sce.obs['age'].unique())
    celltypes = np.unique(sce.obs['celltype'])
    
    print(f"  Ages: {len(ages)} unique values from {ages.min()} to {ages.max()}")
    print(f"  Cell types: {len(celltypes)}")
    
    # Initialize results containers
    labels = []
    age_expn_vectors = []
    
    # Process each cell type
    for ct_idx, ct in enumerate(celltypes, 1):
        print(f"  Processing cell type {ct_idx}/{len(celltypes)}: {ct}")
        
        # Subset to current cell type
        ct_mask = sce.obs['celltype'] == ct
        sub_adata = sce[ct_mask].copy()
        
        # Check if enough ages are present
        present_ages = np.unique(sub_adata.obs['age'])
        if len(present_ages) < (1 - missingness_threshold) * len(ages):
            print(f"    Skipping: insufficient age coverage ({len(present_ages)}/{len(ages)} ages)")
            continue
        
        # Process each gene
        for gene_idx, gene in enumerate(sce.var_names):
            # Generate expression trajectory across ages
            age_expn_vector = []
            
            for age in ages:
                if age in present_ages:
                    # Calculate mean expression at this age
                    age_mask = sub_adata.obs['age'] == age
                    if issparse(sub_adata.X):
                        val = np.mean(sub_adata[age_mask, gene].X.A)
                    else:
                        val = np.mean(sub_adata[age_mask, gene].X)
                    age_expn_vector.append(float(val))
                else:
                    age_expn_vector.append(np.nan)  # Mark as missing
            
            # Impute missing values using Last Observation Carried Forward (LOCF)
            curr_value = 0
            filled_vector = []
            for expn in age_expn_vector:
                if np.isnan(expn):
                    filled_vector.append(curr_value)
                else:
                    curr_value = expn
                    filled_vector.append(curr_value)
            
            # Store results
            labels.append(f"{gene}_{ct}")
            age_expn_vectors.append(filled_vector)
    
    print(f"\n  Total trajectories extracted: {len(labels)}")
    return labels, age_expn_vectors

# Extract trajectories
labels, trajectories = fast_age_trajectories(sce, missingness_threshold)

# Save trajectories as dataframe
print("\nSaving trajectories...")
df_trajectories = pd.DataFrame(np.vstack(trajectories).T, columns=labels)
df_trajectories["age"] = ages
df_trajectories.to_csv("./gene_mean_age_age_genes_expression_raw_LOCF03.csv", index=False)
print("Trajectories saved to './gene_mean_age_age_genes_expression_raw_LOCF03.csv'")

# ============================================================================
# 5. CLUSTER ANALYSIS OF EXPRESSION TRAJECTORIES
# ============================================================================

print("\n" + "="*60)
print("Step 3: Clustering Expression Trajectories")
print("="*60)

# Load trajectory data
print("Loading trajectory data for clustering...")
df = pd.read_csv("./gene_mean_age_age_genes_expression_raw_LOCF03.csv")

# Extract age values and expression matrix
ages = df['age'].values.copy()
age_expn_matrix = df.drop(columns='age').values.T  # Transpose to genes x ages

print(f"Trajectory matrix shape: {age_expn_matrix.shape}")
print(f"Number of gene-celltype combinations: {age_expn_matrix.shape[0]}")
print(f"Number of age points: {age_expn_matrix.shape[1]}")

# Standardize trajectories for clustering
print("Standardizing trajectories...")
age_expn_matrix = StandardScaler().fit_transform(age_expn_matrix)

# Perform K-means clustering
print("Performing K-means clustering...")
kmeans = KMeans(n_clusters=9, random_state=444, n_init="auto").fit(age_expn_matrix)
print(f"Clustering complete. Cluster sizes:")
for i in range(9):
    size = np.sum(kmeans.labels_ == i)
    print(f"  Cluster {i}: {size} trajectories ({size/len(kmeans.labels_)*100:.1f}%)")

# ============================================================================
# 6. VISUALIZATION OF TRAJECTORY CLUSTERS
# ============================================================================

print("\n" + "="*60)
print("Step 4: Visualizing Trajectory Clusters")
print("="*60)

# ============================================================================
# 6.1 9-CLUSTER VISUALIZATION
# ============================================================================

print("\nCreating 9-cluster visualization...")

# Define cluster colors and labels
colors_9 = ['firebrick', 'lightcoral', 'tan',
            'goldenrod', 'grey', 'olive', 
            'darkkhaki', 'cornflowerblue', 'royalblue']

# Cluster reordering for visualization
cluster_order_9 = [7, 8, 0, 5, 3, 4, 6, 2, 1]

# Cluster names
group_names_9 = ["Increasing Gradual", "Increasing", "Decreasing", "Increasing Early",
                 "Midlife Peak", "Late Peak", "Early Peak", "Early Peak", "Early Peak"]

# Create detailed labels with cluster sizes
group_names_detailed_9 = []
for ii in range(len(group_names_9)):
    cluster_idx = cluster_order_9[ii]
    size = np.sum(kmeans.labels_ == cluster_idx)
    group_names_detailed_9.append(f"{group_names_9[ii]}\n(n={size})")

# Position dictionary for annotation placement
pos_dict = {
    "br": (0.3, 0.05),
    "bl": (0.05, 0.05),
    "tr": (0.3, 0.85),
    "tl": (0.05, 0.85)
}

# Annotation positions for each subplot
corner_for_annot = ["tl", "tl", "tl",
                    "tl", "tl", "tl",
                    "bl", "tr", "bl"]

# Create 3x3 grid of subplots
nrows, ncols = 3, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(5, 5), sharex=True, sharey=False)

# Plot each cluster
counter = 0
for i in range(nrows):
    for j in range(ncols):
        # Get current cluster label
        lab = cluster_order_9[counter]
        
        # Subset trajectories for this cluster
        cluster_mask = kmeans.labels_ == lab
        sub_age_expn_matrix = age_expn_matrix[cluster_mask, :]
        
        # Compute median and spread
        median_expn = np.median(sub_age_expn_matrix, axis=0)
        spread_expn_upper = np.percentile(sub_age_expn_matrix, 75, axis=0)
        spread_expn_lower = np.percentile(sub_age_expn_matrix, 25, axis=0)
        
        # Create smooth trajectories using B-spline interpolation
        ages_fine = np.linspace(np.min(ages), np.max(ages), 500)
        
        # Smooth median trajectory
        smoother = BSpline(*splrep(ages, median_expn, s=len(ages)))
        median_expn_smooth = smoother(ages_fine)
        
        # Smooth upper bound
        smoother = BSpline(*splrep(ages, spread_expn_upper, s=len(ages)))
        spread_expn_upper_smooth = smoother(ages_fine)
        
        # Smooth lower bound
        smoother = BSpline(*splrep(ages, spread_expn_lower, s=len(ages)))
        spread_expn_lower_smooth = smoother(ages_fine)
        
        # Plot median trajectory with confidence band
        axs[i, j].plot(ages_fine, median_expn_smooth, color=colors_9[counter], linewidth=2)
        axs[i, j].fill_between(ages_fine, spread_expn_lower_smooth, spread_expn_upper_smooth, 
                               color=colors_9[counter], alpha=0.5)
        
        # Calculate correlation with age
        r, p = pearsonr(ages, median_expn)
        
        # Add correlation annotation
        x, y1 = pos_dict[corner_for_annot[counter]]
        axs[i, j].text(x, y1, f"r={r:.2f}", transform=axs[i, j].transAxes, size=10)
        
        # Set subplot title
        axs[i, j].set_title(group_names_detailed_9[counter], fontsize=11.5)
        
        counter += 1

# Format axes
for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

# Add global labels
fig.supxlabel("Age (Years)", fontsize=16)
fig.supylabel("Mean Expression Z-score", fontsize=16)
plt.tight_layout()

# Save figure
plt.savefig("./paneled_mean_agegenes_trajectories_k9_p5smoothed.pdf", bbox_inches='tight')
print("9-cluster visualization saved to './paneled_mean_agegenes_trajectories_k9_p5smoothed.pdf'")

# ============================================================================
# 6.2 3-CLUSTER VISUALIZATION (SIMPLIFIED)
# ============================================================================

print("\nCreating 3-cluster visualization...")

# Define colors and labels for 3-cluster view
colors_3 = ['lightcoral', 'firebrick', 'cornflowerblue']
cluster_order_3 = [2, 0, 1]
group_names_3 = ["Increasing Early", "Increasing", "Decreasing"]

# Create detailed labels with cluster sizes
group_names_detailed_3 = []
for ii in range(len(group_names_3)):
    cluster_idx = cluster_order_3[ii]
    size = np.sum(kmeans.labels_ == cluster_idx)
    group_names_detailed_3.append(f"{group_names_3[ii]}\n(n={size})")

# Annotation positions for 3-cluster view
corner_for_annot_3 = ["tl", "bl", "bl"]

# Create 1x3 grid of subplots
nrows, ncols = 1, 3
fig, axs = plt.subplots(nrows, ncols, figsize=(5, 2.2), sharex=True, sharey=False)

# Plot each cluster
counter = 0
for j in range(ncols):
    # Get current cluster label
    lab = cluster_order_3[counter]
    
    # Subset trajectories for this cluster
    cluster_mask = kmeans.labels_ == lab
    sub_age_expn_matrix = age_expn_matrix[cluster_mask, :]
    
    # Compute median and spread
    median_expn = np.median(sub_age_expn_matrix, axis=0)
    spread_expn_upper = np.percentile(sub_age_expn_matrix, 75, axis=0)
    spread_expn_lower = np.percentile(sub_age_expn_matrix, 25, axis=0)
    
    # Create smooth trajectories
    ages_fine = np.linspace(np.min(ages), np.max(ages), 500)
    
    # Smooth median trajectory
    smoother = BSpline(*splrep(ages, median_expn, s=len(ages)))
    median_expn_smooth = smoother(ages_fine)
    
    # Smooth bounds
    smoother = BSpline(*splrep(ages, spread_expn_upper, s=len(ages)))
    spread_expn_upper_smooth = smoother(ages_fine)
    
    smoother = BSpline(*splrep(ages, spread_expn_lower, s=len(ages)))
    spread_expn_lower_smooth = smoother(ages_fine)
    
    # Plot
    axs[j].plot(ages_fine, median_expn_smooth, color=colors_3[counter], linewidth=2)
    axs[j].fill_between(ages_fine, spread_expn_lower_smooth, spread_expn_upper_smooth, 
                       color=colors_3[counter], alpha=0.5)
    
    # Calculate correlation with age
    r, p = pearsonr(ages, median_expn)
    
    # Add annotation
    x, y1 = pos_dict[corner_for_annot_3[counter]]
    axs[j].text(x, y1, f"r={r:.2f}", transform=axs[j].transAxes, size=10)
    
    # Set title
    axs[j].set_title(group_names_detailed_3[counter], fontsize=11.5)
    
    counter += 1

# Format axes
for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

# Add global labels
fig.supxlabel("Age (Years)", fontsize=14)
fig.supylabel("Mean Expression Z-score", fontsize=14)
plt.tight_layout()

# Save figure
plt.savefig("./paneled_mean_cellagegenes_trajectories_k3_p5smoothed_temp.pdf", bbox_inches='tight')
print("3-cluster visualization saved to './paneled_mean_cellagegenes_trajectories_k3_p5smoothed_temp.pdf'")

# ============================================================================
# 6.3 CELL TYPE-SPECIFIC TRAJECTORY VISUALIZATION
# ============================================================================

print("\nCreating cell type-specific trajectory visualization...")

# Extract cell type from trajectory labels
celltypes = np.array([x.split("_")[1] for x in labels])

# Define cell type order for visualization
cols = ['Exc', 'Inh', 'Fib', 'Ast', 'Mic', 'Oli', 'OPC', 'End', 'Per', 'CAMs', 'T cells']
rows = group_names_9  # Use 9-cluster names for rows

# Create 9x11 grid of subplots
nrows, ncols = 9, 11
fig, axs = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows), sharex=True, sharey=False)

# Plot each cluster-celltype combination
for i in range(nrows):  # Clusters
    for j in range(ncols):  # Cell types
        # Get current cluster and cell type
        lab = cluster_order_9[i]
        ct = cols[j]
        
        # Subset trajectories for this cluster and cell type
        cluster_mask = kmeans.labels_ == lab
        celltype_mask = celltypes == ct
        combined_mask = cluster_mask & celltype_mask
        
        sub_age_expn_matrix = age_expn_matrix[combined_mask, :]
        
        # Only plot if there are trajectories
        if sub_age_expn_matrix.shape[0] > 0:
            # Compute median and spread
            median_expn = np.median(sub_age_expn_matrix, axis=0)
            spread_expn_upper = np.percentile(sub_age_expn_matrix, 75, axis=0)
            spread_expn_lower = np.percentile(sub_age_expn_matrix, 25, axis=0)
            
            # Create smooth trajectories
            ages_fine = np.linspace(np.min(ages), np.max(ages), 500)
            
            # Smooth median
            smoother = BSpline(*splrep(ages, median_expn, s=len(ages)))
            median_expn_smooth = smoother(ages_fine)
            
            # Smooth bounds
            smoother = BSpline(*splrep(ages, spread_expn_upper, s=len(ages)))
            spread_expn_upper_smooth = smoother(ages_fine)
            
            smoother = BSpline(*splrep(ages, spread_expn_lower, s=len(ages)))
            spread_expn_lower_smooth = smoother(ages_fine)
            
            # Plot
            axs[i, j].plot(ages_fine, median_expn_smooth, color=colors_9[i], linewidth=2)
            axs[i, j].fill_between(ages_fine, spread_expn_lower_smooth, spread_expn_upper_smooth, 
                                   color=colors_9[i], alpha=0.5)
    
    # Add row labels (cluster names)
    axs[i, 0].set_ylabel(rows[i], rotation=90, size='large')

# Add column labels (cell type names)
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

# Format axes
for ax in axs.flatten():
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

# Add global labels
fig.supxlabel("Age (Years)", fontsize=16)
fig.supylabel("Mean Expression Z-score", fontsize=16)
fig.tight_layout()

# Save figure
plt.savefig("./paneled_mean_trajectories_k9_p5smoothed_byCelltype.pdf", bbox_inches='tight')
print("Cell type-specific visualization saved to './paneled_mean_trajectories_k9_p5smoothed_byCelltype.pdf'")

# ============================================================================
print("\n" + "="*60)
print("Pseudo-bulk Gene Expression Trajectory Analysis Complete!")
print("="*60)
        