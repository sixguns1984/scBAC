# scBACs Analysis Pipeline - Code Overview

This directory contains 10 analysis scripts (I-X) for the scBACs (single-cell Brain Age Clock) study. Each script corresponds to a specific analysis module in the paper.

## Script Overview

### I. Cell Type Annotation
**File**: `I_celltype_annotation.py`

**Purpose**: Major cell type annotation and subtype classification for human cortex single-cell transcriptomic data.

**Key Features**:
- Major cell type annotation using scANVI, CellTypist, and CellAssign
- Subtype annotation for neurons, vascular cells, and macrophages
- Hyperparameter tuning for optimal model performance
- Comprehensive model evaluation with confusion matrices

**Inputs**: 
- `.h5ad` format single-cell data files
- Marker gene lists for CellAssign

**Outputs**:
- Annotated AnnData objects with cell type predictions
- Trained model files for scANVI, CellTypist, and CellAssign
- Confusion matrices for model evaluation

---

### II. Pseudo-bulk Gene Expression Trajectory Analysis
**File**: `II_pseudo_bulk_gene_expression_trajectory_analysis.py`

**Purpose**: Identify age-associated genes and analyze whole-life expression trajectories across cell types.

**Key Features**:
- Identification of age-associated genes using Spearman correlation
- Pseudo-bulk analysis to generate gene expression trajectories
- K-means clustering of trajectories to identify aging patterns
- Visualization of trajectory clusters by cell type

**Inputs**: 
- Annotated single-cell data
- Cell type and age information

**Outputs**:
- Age-associated gene lists for each cell type
- Gene expression trajectories across ages
- Clustered trajectory patterns (3-cluster and 9-cluster views)
- Cell type-specific trajectory visualizations

---

### III. scBACs Model Development
**File**: `III_scBACs_development.py`

**Purpose**: Develop cell-type-specific brain age prediction models using deep learning.

**Key Features**:
- Transformer-based neural network architecture for age prediction
- Feature selection using age-associated genes
- Model training with early stopping and learning rate scheduling
- Internal validation on test datasets
- Identification of cellular biological age-associated genes

**Inputs**:
- Preprocessed single-cell data
- Age-associated gene lists
- Training and test datasets

**Outputs**:
- Trained brain age prediction models (`.pth` files)
- Model evaluation metrics (Spearman R, MAE)
- Cellular biological age-associated gene lists

---

### IV. scBACs Model Evaluation
**File**: `IV_scBACs_evalutation.py`

**Purpose**: Comprehensive evaluation of scBACs model performance on validation datasets.

**Key Features**:
- Internal validation on independent datasets
- Visualization of predicted vs. chronological age relationships
- Hexbin scatter plots for each cell type
- External validation using scBACs command-line tool interface

**Inputs**:
- Validation datasets (CT samples only)
- Predicted cell age data

**Outputs**:
- Hexbin scatter plots for main and additional cell types
- Model performance statistics (R values, p-values, MAE)
- External validation results

---

### V. Effect of Neurodegenerative Diseases on Cellular Aging
**File**: `V_effect_of_NDDs_on_predicted_cellular_biological_age.py`

**Purpose**: Analyze region- and cell-type-specific accelerated aging in neurodegenerative diseases.

**Key Features**:
- Linear regression analysis of disease effects on cellular aging
- Analysis across 6 neurodegenerative diseases (AD, PD, MCI, FTD, ALS, FTLD)
- Assessment of cellular aging impact on cognitive function (MMSE) in AD
- Region-specific aging acceleration analysis

**Inputs**:
- Metadata with disease status and predicted cell age
- Clinical variables (MMSE, Braak stage, etc.)

**Outputs**:
- Disease effect sizes and significance for each cell type/region
- Relationships between cellular aging and cognitive function
- Combined analysis results across all NDDs

---

### VI. Cell-type-specific Aging Acceleration Onset in NDDs
**File**: `VI_celltype_with_earliest_onset_of_aging_acceleration_in_NDDs.py`

**Purpose**: Identify cell-type-specific individual age at aging acceleration onset in neurodegenerative diseases.

**Key Features**:
- scBACs analysis of relative age acceleration (RAA)
- Identification of aging acceleration onset age
- Relationships between aging cell proportions and clinical features
- Gene expression analysis for aging acceleration onset

**Inputs**:
- Metadata with predicted cell ages
- Clinical data (MMSE, Braak stage)

**Outputs**:
- Threshold ages for aging acceleration onset
- Correlations with clinical measures
- Genes associated with aging acceleration onset
- AD vs control group comparisons

---

### VII. APOE4 Effect on Cellular Aging in AD
**File**: `VII_APOE4_effect_on_cell_age.py`

**Purpose**: Analyze cell-type-specific APOE4 effect on cellular aging in Alzheimer's disease progression.

**Key Features**:
- APOE4 carrier vs non-carrier comparison across AD stages
- Linear regression analysis with age and sex adjustments
- Discovery and replication analyses in prefrontal cortex
- Differential expression analysis for APOE4-associated genes

**Inputs**:
- Metadata with APOE genotype and predicted cell age
- Single-cell expression data
- AD stage information (based on Braak stage)

**Outputs**:
- APOE4 effect sizes for each cell type and AD stage
- Differentially expressed genes associated with APOE4 pathology
- Discovery and replication results

---

### VIII. APOE4 Effect on Aging Acceleration Onset
**File**: `VIII_APOE4_age_at_aging_acceleration_onset.py`

**Purpose**: Analyze differences in aging acceleration onset between APOE4 carriers and non-carriers in AD.

**Key Features**:
- scBACs analysis for APOE4 carriers vs non-carriers
- Discovery and replication analyses
- Integration of aging onset genes with APOE4 DEGs
- Focus on oligodendrocyte (Oli) cells for therapeutic target identification

**Inputs**:
- Metadata with APOE genotype
- Single-cell expression data
- Threshold age data from scBACs

**Outputs**:
- Aging acceleration onset comparisons
- Integrated gene lists for Oli cells
- Discovery-replication consistency analysis

---

### IX. scBACs in iPSC Neuron Development
**File**: `IX_scBACs_in_ipsc_neuron_development.py`

**Purpose**: Analyze cell age distribution across iPSC neuron differentiation timepoints.

**Key Features**:
- Violin plots with boxplot overlays for timepoint comparisons
- Mann-Whitney U tests between consecutive timepoints
- Kruskal-Wallis test for overall differences
- Statistical annotations on plots

**Inputs**:
- iPSC neuron differentiation metadata
- Predicted cell age data across timepoints

**Outputs**:
- Cell age distribution visualizations
- Statistical test results for timepoint comparisons
- Summary statistics by timepoint

---

### X. scBACs in Mouse Brain Aging
**File**: `X_scBACs_in_mice_brain_aging.py`

**Purpose**: Cell-type-specific aging analysis in young vs old mice and radiation-induced brain injury models.

**Key Features**:
- Young vs old mice brain cell age comparison (Mann-Whitney U test)
- RIBI mice cell-type-specific brain aging after radiation
- Heatmap visualization of cell age changes over time
- Line plots of median cell age trajectories

**Inputs**:
- Young and old mice metadata
- RIBI mice radiation time course data

**Outputs**:
- Statistical comparisons for young vs old mice
- Heatmaps of cell age changes in RIBI mice
- Trajectory plots of median cell age over time

---

## Dependencies

All scripts require the following Python packages:
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- `scanpy`, `scvi`, `celltypist`
- `torch`, `scikit-learn`
- `statsmodels`

Install dependencies using:
```bash
pip install numpy pandas scipy matplotlib seaborn scanpy scvi-tools celltypist torch scikit-learn statsmodels
