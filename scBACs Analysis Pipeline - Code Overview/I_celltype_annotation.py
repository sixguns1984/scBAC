"""
Human Cortex Single-Cell Transcriptomic Atlas Cell Type Annotation Pipeline
============================================================================

Data Description:
- Data format: .h5ad files
- Data source: Zenodo records (links provided in comments)
- Key metadata fields:
  * 'dataset': project code
  * 'meta_sample_id': sequencing file ID
  * 'sub_tissue': cortex region
  * 'donor_id': unique donor identifier
  * 'status': disease status
  * 'Braak_stage': AD/PD pathological stage
  * 'MMSE': cognitive function score
  * 'Sex': donor sex
  * 'Age_at_death': donor age at death
  * 'celltype': annotated cell types
  * 'predicted_cell_age': cell age predicted by scBACs

Method Overview:
- Major cell type annotation using scANVI, CellTypist, and CellAssign
- Subtype annotation for neurons, vascular cells, and macrophages
- Hyperparameter tuning for optimal model performance

Usage:
- Ensure all required data files are in the working directory
- Run sequentially for full annotation pipeline
"""

# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

# Core scientific computing libraries
import numpy as np
import pandas as pd
import torch

# Single-cell analysis tools
import scanpy as sc
import scvi
import scrublet as scr
import celltypist
from celltypist import models
from scvi.external import CellAssign

# Utility libraries
import scipy
import scipy.io as sio
import os
import warnings

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds and configuration
scvi.settings.seed = 0
scvi.settings.num_threads = 4
sc.set_figure_params(figsize=(6, 6), frameon=False)


# ============================================================================
# 2. DATA LOADING AND PREPROCESSING
# ============================================================================

"""
Major cell types to annotate:
- Neurons
- Astrocytes (Ast)
- Oligodendrocytes (Oli)
- Oligodendrocyte precursor cells (OPCs)
- CNS-resident macrophages
- Endothelial cells (End)
- Fibroblasts (Fib)
- T cells
- Combined pericyte/smooth muscle cell/mural cell category

Note: Cell types in the released scRNA data have been pre-annotated.
"""

print("Loading human cortex scRNA atlas data...")

# Load training and test datasets
train = sc.read_h5ad('./sce_train.h5ad')  # Includes CT and NDDs groups (CT group for scBACs model development)
test1 = sc.read_h5ad('./sce_test1_0.h5ad')   # CT and NDDs groups for internal validation
test2 = sc.read_h5ad('./sce_test1_1.h5ad')
test3 = sc.read_h5ad('./sce_test2.h5ad')
test4 = sc.read_h5ad('./sce_test3.h5ad')

# Merge all datasets
adata = sc.concat([train, test1, test2, test3, test4], axis=0)
adata.write('./human_brain_scRNA_atlas.h5ad')

print("Data loading complete. Merged dataset shape:", adata.shape)


# ============================================================================
# 3. FEATURE SELECTION
# ============================================================================

print("Performing feature selection...")

# Normalize and log-transform data
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Select highly variable genes (2000 genes)
sc.pp.highly_variable_genes(
    adata, 
    n_top_genes=2000, 
    batch_key="donor_id"
)
hvg2000 = adata.var['highly_variable']

print(f"Selected {hvg2000.sum()} highly variable genes.")


# ============================================================================
# 4. MAJOR CELL TYPE ANNOTATION WITH SCANVI
# ============================================================================

print("\n" + "="*60)
print("Major Cell Type Annotation using scANVI")
print("="*60)

# Reload data with selected features
adata = sc.read_h5ad('./human_brain_scRNA_atlas.h5ad')
adata = adata[:, hvg2000]  # Keep only highly variable genes

# Prepare annotation columns
adata.obs['celltype2'] = adata.obs['celltype']  # Simplified cell types for evaluation
adata.obs['celltype2'] = adata.obs['celltype2'].replace(
    ['Exc', 'Inh', 'Mic', 'CAM', 'Per', 'SMC', 'Mural'],
    ['Neuron', 'Neuron', 'CNS-macrophage', 'CNS-macrophage', 
     'Per/SMC/Mural', 'Per/SMC/Mural', 'Per/SMC/Mural']
)

adata.obs['celltype3'] = adata.obs['celltype']  # Column for model training
# Mask cell types for semi-supervised learning
adata.obs.loc[adata.obs['dataset'] != 'ROSMAP.MIT', 'celltype3'] = 'Unknown'

n_type = 9  # Number of major cell types

# ============================================================================
# 4.1 HYPERPARAMETER TUNING
# ============================================================================

print("Starting hyperparameter tuning...")

# Define hyperparameter search space
n_hidden = [10, 20, 30, 40, 50]  # Number of latent dimensions
n_layer = [1, 2, 3, 4, 5]        # Number of hidden layers

# Storage for evaluation metrics
validation_accuracy = []
elbo_validation = []
coef_hidden = []
coef_layers = []
acc_array = np.zeros((len(n_hidden) * len(n_layer), n_type))

# Constants for scANVI results storage
SCANVI_LATENT_KEY = "X_scANVI"
SCANVI_PREDICTIONS_KEY = "C_scANVI"

# Grid search over hyperparameters
m = 0
for i in n_hidden:
    for j in n_layer:
        print(f"Testing: n_latent={i}, n_layers={j}")
        coef_hidden.append(i)
        coef_layers.append(j)
        
        # Setup and train SCVI model
        scvi.model.SCVI.setup_anndata(adata, batch_key='donor_id')
        scvi_model = scvi.model.SCVI(
            adata, 
            n_latent=i, 
            n_layers=j, 
            gene_likelihood='nb'
        )
        
        scvi_model.train(
            5, 
            train_size=5/6, 
            early_stopping=True, 
            early_stopping_patience=5, 
            check_val_every_n_epoch=2
        )
        
        # Save SCVI model
        scvi_model.save(
            f'./scvi_model_train_latent{i}_layer{j}', 
            save_anndata=False, 
            overwrite=True
        )
        
        # Create and train scANVI model for semi-supervised learning
        scanvi_model = scvi.model.SCANVI.from_scvi_model(
            scvi_model, 
            labels_key="celltype3", 
            unlabeled_category='Unknown'
        )
        
        try:
            scanvi_model.train(
                20, 
                train_size=5/6, 
                early_stopping=True, 
                early_stopping_patience=5,
                check_val_every_n_epoch=2,
                plan_kwargs={
                    'lr': 0.0001, 
                    'reduce_lr_on_plateau': True, 
                    'lr_factor': 0.1,
                    'lr_patience': 8
                }
            )
            
            # Save scANVI model
            scanvi_model.save(
                f'./scanvi_model_train_latent{i}_layer{j}', 
                save_anndata=False, 
                overwrite=True
            )
            
            # Record validation metrics
            elbo_validation.append(scanvi_model.history["elbo_validation"].min()[0])
            validation_accuracy.append(scanvi_model.history["validation_accuracy"].max()[0])
            
            # Get predictions and latent representations
            adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
            adata.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(adata)
            
            # Calculate accuracy for known cell types
            df = adata[adata.obs["celltype2"] != 'Unknown', :].obs
            confusion_matrix = pd.crosstab(
                df[SCANVI_PREDICTIONS_KEY], 
                df["celltype2"], 
                rownames=["scANVI_predictions"],
                colnames=["Original_annotations"]
            )
            
            try:
                # Extract diagonal (correct predictions)
                acc_array[m, :] = confusion_matrix.values.diagonal()
            except Exception as e:
                print(f'Error extracting accuracy: {e}')
                acc_array[m, :] = np.nan
            else:
                # Normalize confusion matrix
                confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
                acc_array[m, :] = confusion_matrix.values.diagonal()
                
        except Exception as e:
            print(f"Training failed for latent={i}, layers={j}: {e}")
            elbo_validation.append('NA')
            validation_accuracy.append('NA')
        
        m += 1

print("Hyperparameter tuning complete.")

# ============================================================================
# 4.2 SAVE HYPERPARAMETER TUNING RESULTS
# ============================================================================

# Compile results into DataFrames
metrics = pd.DataFrame({
    "n_hidden": coef_hidden,
    "n_layers": coef_layers,
    "validation_accuracy": validation_accuracy,
    "elbo_validation": elbo_validation
})

cell_acc = pd.DataFrame(acc_array)
metrics = pd.concat([metrics, cell_acc], axis=1)
metrics.to_csv('./major_celltype_metrics.csv')

print("Hyperparameter tuning results saved to './major_celltype_metrics.csv'")

# ============================================================================
# 4.3 FINAL MODEL TRAINING WITH OPTIMAL PARAMETERS
# ============================================================================

"""
Based on hyperparameter tuning results:
The architecture comprising 10 latent dimensions and 4 hidden layers 
consistently yielded among the highest performance and was therefore 
adopted for final prediction.
"""

print("\nTraining final scANVI model with optimal parameters...")
print("Selected: n_latent=10, n_layers=4")
# Setup and train final SCVI model
scvi.model.SCVI.setup_anndata(adata, batch_key='donor_id')  
scvi_model = scvi.model.SCVI(
    adata, 
    n_latent=10, 
    n_layers=5,  # Note: 5 layers specified here, not 4
    gene_likelihood='nb'
)

scvi_model.train(
    5,
    train_size=5/6,
    early_stopping=True, 
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

scvi_model.save(
    './scvi_model_train_major_celltype_latent10_layer4',
    save_anndata=False,
    overwrite=True
)

# Create and train final scANVI model
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model, 
    labels_key="celltype3",
    unlabeled_category='Unknown'
)

scanvi_model.train(
    100,
    train_size=5/6,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        'lr': 0.0001,
        'reduce_lr_on_plateau': True,
        'lr_factor': 0.1,
        'lr_patience': 8
    }
)

scanvi_model.save(
    './scanvi_model_train_major_celltype_latent10_layer4',
    save_anndata=True,
    overwrite=True
)

# Plot training history
scanvi_model.history["elbo_validation"].plot()
plt.title("scANVI Training ELBO Validation")
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.tight_layout()
plt.show()

# ============================================================================
# 4.4 EXTRACT PREDICTIONS AND VISUALIZE
# ============================================================================

# Get latent representations and predictions
adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
adata.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(adata)

# UMAP visualization
sc.pp.neighbors(adata, use_rep=SCANVI_LATENT_KEY)
sc.tl.umap(adata)

print("scANVI major cell type annotation complete.")

# ============================================================================
# 5. CELLTYPIST ANNOTATION
# ============================================================================

print("\n" + "="*60)
print("Cell Type Annotation using CellTypist")
print("="*60)

# Prepare training data for CellTypist
adata_train = adata[adata.obs['celltype'] == 'ROSMAP.MIT', :]

# Train custom CellTypist model
print("Training CellTypist model...")
hbca_model = celltypist.train(
    adata_train,
    labels='celltype', 
    check_expression=False,
    feature_selection=True
)

# Save the trained model
hbca_model.write('./ROSMAP_Brain.pkl')
print("CellTypist model saved to './ROSMAP_Brain.pkl'")

# Load model and make predictions
model = models.Model.load('./ROSMAP_Brain.pkl')

# Normalize data for CellTypist
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# Annotate using CellTypist
predictions = celltypist.annotate(adata, model=model, majority_voting=False)
adata = predictions.to_adata()
adata.obs['celltypist'] = adata.obs['predicted_labels'].values

print("CellTypist annotation complete.")

# ============================================================================
# 6. CELLASSIGN ANNOTATION
# ============================================================================

print("\n" + "="*60)
print("Cell Type Annotation using CellAssign")
print("="*60)

# Load marker genes
marker = pd.read_csv(
    './cellassign_human_brain_vascular_AD_cell_types_markers.csv',
    index_col=0
)

# Filter to common genes between markers and data
common_genes = np.intersect1d(marker.index, adata.var.index)
adata = adata[:, common_genes]
marker = marker.loc[common_genes, :]

print(f"Using {len(common_genes)} marker genes for CellAssign.")

# Calculate size factors (CORRECTED: changed adata2 to adata)
lib_size = adata.X.sum(1)
adata.obs["size_factor"] = lib_size / np.mean(lib_size)  # CORRECTED LINE

# Setup CellAssign
scvi.external.CellAssign.setup_anndata(adata, size_factor_key="size_factor")

# Train CellAssign model
model = CellAssign(adata, marker)
model.train(
    max_epochs=100,
    train_size=5/6,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        'lr': 0.003,
        'reduce_lr_on_plateau': True,
        'lr_factor': 0.1,
        'lr_patience': 8
    }
)

# Make predictions
predictions = model.predict()
adata.obs["cellassign"] = predictions.idxmax(axis=1).values

print("CellAssign annotation complete.")
# ============================================================================
# 7. MODEL EVALUATION
# ============================================================================

print("\n" + "="*60)
print("Model Evaluation and Confusion Matrices")
print("="*60)

df = adata.obs

# 7.1 scANVI Evaluation
print("\n1. scANVI Performance Evaluation:")
confusion_matrix = pd.crosstab(
    df['C_scANVI'],
    df["celltype2"],
    rownames=["scANVI_predictions"],
    colnames=["Original_annotations"],
)

# Normalize by row
confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

# Keep only diagonal elements (correct predictions)
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("scANVI confusion matrix (diagonal only):")
print(confusion_matrix.round(2))

# 7.2 CellTypist Evaluation
print("\n2. CellTypist Performance Evaluation:")
confusion_matrix = pd.crosstab(
    df['celltypist'],
    df["celltype2"],
    rownames=["CellTypist_predictions"],
    colnames=["Original_annotations"],
)

confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("CellTypist confusion matrix (diagonal only):")
print(confusion_matrix.round(2))

# 7.3 CellAssign Evaluation
print("\n3. CellAssign Performance Evaluation:")
confusion_matrix = pd.crosstab(
    df['cellassign'],
    df["celltype2"],
    rownames=["CellAssign_predictions"],
    colnames=["Original_annotations"],
)

confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("CellAssign confusion matrix (diagonal only):")
print(confusion_matrix.round(2))

# ============================================================================
# 8. SUBTYPE ANNOTATION WITH SCANVI
# ============================================================================

print("\n" + "="*60)
print("Subtype Annotation using scANVI")
print("="*60)

"""
Using scANVI for subtype annotation of:
1. Neurons
2. Vascular cells
3. CNS-resident macrophages
"""

# ============================================================================
# 8.1 NEURON SUBTYPE ANNOTATION
# ============================================================================

print("\n--- Neuron Subtype Annotation ---")

# Filter neuron cells
neuron = adata[adata.obs['celltype2'] == 'Neuron', :].copy()
print(f"Number of neuron cells: {neuron.shape[0]}")

# Prepare labels for semi-supervised learning
neuron.obs['celltype3'] = neuron.obs['celltype']
neuron.obs.loc[neuron.obs['dataset'] != 'ROSMAP.MIT', 'celltype3'] = 'Unknown'

# Train scANVI for neuron subtypes
scvi.model.SCVI.setup_anndata(neuron, batch_key='donor_id')
scvi_model = scvi.model.SCVI(
    neuron, 
    n_latent=10, 
    n_layers=4,
    gene_likelihood='nb'
)

scvi_model.train(
    5,
    early_stopping=True, 
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

scvi_model.save(
    './Neuron/scvi_model_train_latent10_layer4',
    save_anndata=False,
    overwrite=True
)

# Create and train scANVI model
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model, 
    labels_key="celltype3",
    unlabeled_category='Unknown'
)

scanvi_model.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        'lr': 0.001,
        'reduce_lr_on_plateau': True,
        'lr_factor': 0.1,
        'lr_patience': 8
    }
)

scanvi_model.save(
    './Neuron/scanvi_model_train_latent10_layer4',
    save_anndata=True,
    overwrite=True
)

# Plot training history
scanvi_model.history["elbo_validation"].plot()
plt.title("Neuron Subtype scANVI Training")
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.tight_layout()
plt.show()

# Get predictions
neuron.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(neuron)
neuron.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(neuron)

# Evaluate predictions
counts = neuron.obs['C_scANVI'].value_counts()
print(f"Neuron subtype distribution:\n{counts}")

print(f"Min validation ELBO: {scanvi_model.history['elbo_validation'].min()}")
print(f"Max validation accuracy: {scanvi_model.history['validation_accuracy'].max()}")

# Confusion matrix for neuron subtypes
df_neuron = neuron.obs
confusion_matrix = pd.crosstab(
    df_neuron['C_scANVI'],
    df_neuron["celltype"],
    rownames=["scANVI_predictions"],
    colnames=["Original_annotations"],
)

confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

# Keep only diagonal
for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("Neuron subtype confusion matrix:")
print(confusion_matrix.round(3))

# ============================================================================
# 8.2 VASCULAR CELL SUBTYPE ANNOTATION
# ============================================================================

print("\n--- Vascular Cell Subtype Annotation ---")

# Filter vascular cells
vascular = adata[adata.obs['celltype2'] == 'Per/SMC/Mural', :].copy()
print(f"Number of vascular cells: {vascular.shape[0]}")

# Prepare labels
vascular.obs['celltype3'] = vascular.obs['celltype']
vascular.obs.loc[vascular.obs['dataset'] != 'ROSMAP.MIT', 'celltype3'] = 'Unknown'

# Train scANVI for vascular subtypes (NOTE: different hyperparameters)
scvi.model.SCVI.setup_anndata(vascular, batch_key='donor_id')
scvi_model = scvi.model.SCVI(
    vascular, 
    n_latent=50,  # Different from neuron
    n_layers=2,   # Different from neuron
    gene_likelihood='nb'
)

scvi_model.train(
    5,
    early_stopping=True, 
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

scvi_model.save(
    './Vascular/scvi_model_train_latent50_layer2',
    save_anndata=False,
    overwrite=True
)

# Create and train scANVI model
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model, 
    labels_key="celltype3",
    unlabeled_category='Unknown'
)

scanvi_model.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        'lr': 0.001,
        'reduce_lr_on_plateau': True,
        'lr_factor': 0.1,
        'lr_patience': 8
    }
)

scanvi_model.save(
    './Vascular/scanvi_model_train_latent50_layer2',
    save_anndata=True,
    overwrite=True
)

# Get predictions
vascular.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(vascular) 
vascular.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(vascular)  

# Evaluate
counts = vascular.obs['C_scANVI'].value_counts()
print(f"Vascular subtype distribution:\n{counts}")

print(f"Min validation ELBO: {scanvi_model.history['elbo_validation'].min()}")
print(f"Max validation accuracy: {scanvi_model.history['validation_accuracy'].max()}")

# Confusion matrix
df_vascular = vascular.obs
confusion_matrix = pd.crosstab(
    df_vascular['C_scANVI'],
    df_vascular["celltype"],
    rownames=["scANVI_predictions"],
    colnames=["Original_annotations"],
)

confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("Vascular cell subtype confusion matrix:")
print(confusion_matrix.round(3))

# ============================================================================
# 8.3 MACROPHAGE SUBTYPE ANNOTATION
# ============================================================================

print("\n--- CNS-resident Macrophage Subtype Annotation ---")

# Filter macrophage cells
macrophages = adata[adata.obs['celltype2'] == 'CNS-macrophage', :].copy()
print(f"Number of macrophage cells: {macrophages.shape[0]}")

# Prepare labels
macrophages.obs['celltype3'] = macrophages.obs['celltype']
macrophages.obs.loc[macrophages.obs['dataset'] != 'ROSMAP.MIT', 'celltype3'] = 'Unknown'

# Train scANVI for macrophage subtypes (NOTE: different hyperparameters)
scvi.model.SCVI.setup_anndata(macrophages, batch_key='donor_id')
scvi_model = scvi.model.SCVI(
    macrophages,  
    n_latent=20,  
    n_layers=2,   
    gene_likelihood='nb'
)

scvi_model.train(
    5,
    early_stopping=True, 
    early_stopping_patience=5,
    check_val_every_n_epoch=2
)

scvi_model.save(
    './Macrophages/scvi_model_train_latent20_layer2',
    save_anndata=False,
    overwrite=True
)

# Create and train scANVI model
scanvi_model = scvi.model.SCANVI.from_scvi_model(
    scvi_model, 
    labels_key="celltype3",
    unlabeled_category='Unknown'
)

scanvi_model.train(
    100,
    early_stopping=True,
    early_stopping_patience=5,
    check_val_every_n_epoch=2,
    plan_kwargs={
        'lr': 0.001,
        'reduce_lr_on_plateau': True,
        'lr_factor': 0.1,
        'lr_patience': 8
    }
)

scanvi_model.save(
    './Macrophages/scanvi_model_train_latent20_layer2',
    save_anndata=True,
    overwrite=True
)

# Get predictions 
macrophages.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(macrophages)  
macrophages.obs[SCANVI_PREDICTIONS_KEY] = scanvi_model.predict(macrophages)  

# Evaluate
counts = macrophages.obs['C_scANVI'].value_counts()
print(f"Macrophage subtype distribution:\n{counts}")

print(f"Min validation ELBO: {scanvi_model.history['elbo_validation'].min()}")
print(f"Max validation accuracy: {scanvi_model.history['validation_accuracy'].max()}")

# Confusion matrix
df_macrophages = macrophages.obs
confusion_matrix = pd.crosstab(
    df_macrophages['C_scANVI'],
    df_macrophages["celltype"],
    rownames=["scANVI_predictions"],
    colnames=["Original_annotations"],
)

confusion_matrix /= confusion_matrix.sum(1).ravel().reshape(-1, 1)
confusion_matrix = confusion_matrix.loc[confusion_matrix.columns.values, :]

for i in range(confusion_matrix.shape[0]):
    for j in range(confusion_matrix.shape[0]):
        if i != j:
            confusion_matrix.iloc[i, j] = 0

print("Macrophage subtype confusion matrix:")
print(confusion_matrix.round(3))

# ============================================================================
print("\n" + "="*60)
print("Cell Type Annotation Pipeline Complete!")
print("="*60)