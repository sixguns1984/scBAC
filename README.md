scBACs - Single Cell Brain Age Calculator
<div align="center">
https://img.shields.io/badge/python-3.7%252B-blue
https://img.shields.io/badge/license-MIT-green
https://img.shields.io/badge/status-active-success
https://img.shields.io/badge/DOI-TBD-orange

A deep learning tool for predicting brain cell age from single-cell RNA-seq data

Quick Start | Installation | Documentation | Examples | Contact

</div>
✨ Key Features
🔬 Multi-cell type support: Predict ages for 13 major brain cell types

🧠 Deep learning models: Integrated MLP and Transformer architectures

📊 Advanced analysis: Age gap calculation and senescence turning point detection

🎨 Professional visualizations: Nature journal-style plots

⚡ Efficient computation: CPU/GPU support and parallel processing

🔄 Easy model management: One-click pre-trained model downloads

💻 Command-line interface: User-friendly CLI for all functions

📦 Installation
Basic Installation
bash
pip install scbac
Complete Installation (with analysis module)
bash
pip install "scbac[analysis]"
Development Version
bash
git clone https://github.com/yourusername/scBACs.git
cd scBACs
pip install -e .
Download Pre-trained Models
After installation, download the pre-trained models:

bash
scbac-install-models --cell-type all
🚀 Quick Start
1. Predict Cell Ages from scRNA-seq Data
python
import scanpy as sc
from scbac import predict_cell_age

# Load your single-cell data
adata = sc.read_h5ad("your_data.h5ad")

# Predict cell ages
results = predict_cell_age(
    adata,
    cell_type_column='celltypist',  # Column name for cell type annotations
    count_layer='counts',           # Layer containing raw counts
    device='cpu'                    # Use 'cuda' for GPU acceleration
)

# Results are stored in adata.obs
adata.obs['predicted_age'] = results['cell_ages']
adata.write("results_with_ages.h5ad")
2. Analyze Age Gap and Turning Points
python
import pandas as pd
from scbac import calculate_donor_celltype_age_gap, age_gap_turning_point_analysis

# Load your data with predicted ages
df = pd.read_csv("predicted_ages.csv")

# Calculate age gap (residuals from cell-type specific models)
df_with_gap = calculate_donor_celltype_age_gap(
    df,
    donor_id_col='PaticipantID_unique',
    celltype_col='celltype',
    age_pred_col='age_pred',
    age_death_col='Age_at_death',
    sex_col='Sex'
)

# Analyze turning points in aging trajectories
results = age_gap_turning_point_analysis(
    df_with_gap,
    disease_name='AD',              # Disease group to analyze
    status_col='status',            # Column with disease/control status
    save_prefix='AD_analysis'       # Prefix for output files
)

# View results
print("Cell type aging sequence:")
print(results['analysis_results']['celltype_stats'])
📊 Supported Cell Types
scBACs includes pre-trained models for 13 brain cell types:

Cell Type	Abbreviation	Model Type
Astrocytes	Ast	MLP
Endothelial cells	End	Transformer
Excitatory neurons	Exc	Transformer
Inhibitory neurons	Inh	Transformer
Oligodendrocytes	Oli	Transformer
Oligodendrocyte progenitor cells	OPC	Transformer
Pericytes	Per	Transformer
Microglia	Mic	MLP
Choroid plexus/ependymal cells	CAM	MLP
Fibroblasts	Fib	Transformer
Mural cells	Mural	Transformer
Smooth muscle cells	SMC	Transformer
T cells	Tcell	Transformer
🛠️ Command Line Interface
Predict Cell Ages
bash
scbac predict \
    --input input_data.h5ad \
    --output results.h5ad \
    --cell-type-column celltype \
    --count-layer counts \
    --device cpu
Analyze Age Gap Turning Points
bash
scbac analyze \
    --input predicted_ages.csv \
    --output-prefix analysis_results \
    --disease-name AD \
    --status-col status \
    --donor-col PaticipantID_unique \
    --celltype-col celltype
Model Management
bash
# Interactive model installation
scbac-install-models

# Install all models
scbac-install-models --cell-type all

# Force re-download
scbac-install-models --force-download

# List installed models
scbac-install-models --list-models
📈 Output Files
Prediction Output
predicted_age column added to AnnData.obs

Full observation DataFrame with predictions

Analysis Output
age_gap: Calculated age gap (residuals)

threshold_age: Senescence turning point age

positive_ratio: Proportion of cells with positive age gap

Statistical analysis results (ANOVA, pairwise comparisons)

High-quality publication-ready figures

🎨 Visualizations
scBACs generates comprehensive visualizations:

Individual Donor Curves: Age gap vs. predicted age for each donor

Cell Type Comparisons: Aging thresholds across cell types

Heatmap Clustering: Donor-level aging patterns

Statistical Significance: Pairwise differences between cell types

Positive Ratio Analysis: Proportion of aged cells per donor

🔬 Advanced Usage
Custom Model Directory
python
from scbac import predict_cell_age

results = predict_cell_age(
    adata,
    model_directory="/path/to/custom/models",
    device='cuda'  # Use GPU acceleration
)
Batch Processing
python
# Process multiple datasets
from joblib import Parallel, delayed

def process_dataset(dataset_path):
    adata = sc.read_h5ad(dataset_path)
    results = predict_cell_age(adata)
    return results

# Parallel processing
dataset_paths = ["data1.h5ad", "data2.h5ad", "data3.h5ad"]
all_results = Parallel(n_jobs=3)(delayed(process_dataset)(path) for path in dataset_paths)
Custom Analysis Parameters
python
from scbac import fast_calculate_donor_curve_thresholds, analyze_threshold_differences

# Custom threshold calculation
threshold_df, curve_data = fast_calculate_donor_curve_thresholds(
    df,
    disease_name='AD',
    min_cells=15,      # Minimum cells per donor-celltype
    n_points=200       # Resolution of smooth curves
)

# Custom statistical analysis
analysis_results = analyze_threshold_differences(threshold_df)
📋 Data Requirements
For Age Prediction (AnnData Input)
Cell type annotations in adata.obs

Raw counts in a layer (default: 'counts')

Normalized and scaled data (optional, will be processed if needed)

For Age Gap Analysis (DataFrame Input)
Required columns:

predicted_age: Predicted cell ages

celltype: Cell type annotations

PaticipantID_unique: Donor/participant ID

Age_at_death: Age at death

Sex: Sex (Male/Female or encoded values)

status: Disease status (e.g., 'AD', 'CT', 'ALS', etc.)

📚 Tutorials and Examples
Example 1: Basic Age Prediction Pipeline
python
import scanpy as sc
from scbac import predict_cell_age, preprocess_data

# Load and preprocess data
adata = sc.read_h5ad("brain_data.h5ad")
adata = preprocess_data(adata, normalize=True, scale=True)

# Predict ages
results = predict_cell_age(adata)

# Save results
adata.obs['cell_age'] = results['cell_ages']
sc.pl.umap(adata, color=['celltype', 'cell_age'], save='_cell_age_umap.pdf')
Example 2: Complete Aging Analysis Workflow
python
import pandas as pd
import matplotlib.pyplot as plt
from scbac import (
    calculate_donor_celltype_age_gap,
    age_gap_turning_point_analysis,
    analyze_age_gap_ratio
)

# Load your data
meta_data = pd.read_csv("brain_metadata.csv")

# Calculate age gap
df_with_gap = calculate_donor_celltype_age_gap(meta_data)

# Analyze a specific disease
ad_results = age_gap_turning_point_analysis(
    df_with_gap,
    disease_name='AD',
    save_prefix='AD_results'
)

# Analyze positive age gap ratio
ratio_results = analyze_age_gap_ratio(
    df_with_gap[df_with_gap['status'] == 'AD']
)

# Correlate with clinical scores
clinical_data = pd.read_csv("clinical_scores.csv")
merged_data = pd.merge(ratio_results, clinical_data, on='PaticipantID_unique')

# Plot correlation
plt.figure(figsize=(8, 6))
plt.scatter(merged_data['positive_ratio'], merged_data['MMSE'], alpha=0.7)
plt.xlabel('Positive Age Gap Ratio')
plt.ylabel('MMSE Score')
plt.title('Cognitive Function vs. Cellular Aging')
plt.savefig('correlation_plot.pdf', dpi=300, bbox_inches='tight')
🧪 Testing with Sample Data
To test scBACs with sample data:

python
import numpy as np
import pandas as pd
from scbac import calculate_donor_celltype_age_gap

# Generate sample data
np.random.seed(42)
n_cells = 1000

sample_data = pd.DataFrame({
    'PaticipantID_unique': np.random.choice(['Donor1', 'Donor2', 'Donor3', 'Donor4'], n_cells),
    'celltype': np.random.choice(['Ast', 'End', 'Exc', 'Inh', 'Oli'], n_cells),
    'age_pred': np.random.normal(60, 10, n_cells),
    'Age_at_death': np.random.choice([65, 70, 75, 80], n_cells),
    'Sex': np.random.choice(['Male', 'Female'], n_cells),
    'status': np.random.choice(['AD', 'CT'], n_cells, p=[0.6, 0.4])
})

# Test age gap calculation
df_with_gap = calculate_donor_celltype_age_gap(sample_data)
print(f"Age gap range: {df_with_gap['age_gap'].min():.2f} to {df_with_gap['age_gap'].max():.2f}")
🏗️ Project Structure
text
scbac/
├── scbac/
│   ├── __init__.py                 # Package initialization
│   ├── predictor.py                # Core prediction functions
│   ├── age_gap_analysis.py         # Age gap analysis module
│   ├── data_processing.py          # Data preprocessing
│   ├── models/
│   │   ├── mlp_model.py           # MLP model definition
│   │   └── transformer_model.py   # Transformer model definition
│   ├── cli.py                     # Command-line interface
│   └── install_models.py          # Model installation script
├── examples/
│   ├── basic_usage.py             # Basic usage examples
│   └── advanced_analysis.py       # Advanced analysis examples
├── tests/
│   ├── test_predictor.py          # Unit tests
│   └── test_analysis.py           # Analysis tests
├── setup.py                       # Installation configuration
├── requirements.txt               # Dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
📄 Citation
If you use scBACs in your research, please cite:

bibtex
@software{scBACs2024,
  author = {Luo, Jianfeng},
  title = {scBACs: Single Cell Brain Age Calculator},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/scBACs}
}
🤝 Contributing
We welcome contributions! Here's how to get started:

Fork the repository

Create a feature branch: git checkout -b feature/AmazingFeature

Commit your changes: git commit -m 'Add AmazingFeature'

Push to the branch: git push origin feature/AmazingFeature

Open a Pull Request

Development Setup
bash
# Clone and setup
git clone https://github.com/yourusername/scBACs.git
cd scBACs
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
📞 Contact
Issues: GitHub Issues

Email: luojf35@mail.sysu.edu.cn

Lab Website: TBD

⚖️ License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Thanks to all contributors and testers

Computational resources provided by Sun Yat-sen University

Open-source community for tools and libraries

<div align="center"> <strong>scBACs: Unraveling cellular aging dynamics in the human brain</strong> </div>
