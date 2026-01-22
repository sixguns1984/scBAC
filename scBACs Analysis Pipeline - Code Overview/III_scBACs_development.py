"""
Single-Cell Brain Age Clock (scBACs) Development Pipeline
==========================================================

Purpose:
- Develop cell-type-specific brain age prediction models using deep learning
- Validate models on independent datasets
- Identify cellular biological age-associated genes

Note:
- Hyperparameters for all clocks are available at: 
  https://zenodo.org/api/records/18287820/draft/files/model_hyper_parameter.csv/content
- User-friendly tool for cell-type-specific brain age prediction:
  https://github.com/sixguns1984/scBACs
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

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
        # CORRECTED: changed 'sc.layers' to 'sce.layers'
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
# 3. DATA LOADING AND PREPROCESSING
# ============================================================================

print("\n" + "="*60)
print("Step 1: Data Loading and Preprocessing")
print("="*60)

# Load training data
print("Loading training data...")
adata_train = sc.read_h5ad('./sce_train.h5ad')

# Select only control (CT) samples for model training
adata_train = adata_train[adata_train.obs['status'] == 'CT', :].copy()
print(f"Training data shape (CT samples only): {adata_train.shape}")
print(f"Available cell types: {adata_train.obs['celltype'].unique()}")

# ============================================================================
# 4. EXAMPLE: EXCITATORY NEURON BRAIN AGE CLOCK DEVELOPMENT
# ============================================================================

print("\n" + "="*60)
print("Step 2: Excitatory Neuron Brain Age Clock Development")
print("="*60)

# Set target cell type
celltype = 'Exc'  # Options: 'Exc','Inh','Ast','Oli','OPC','Fib','End','Mic','CAM','Per','SMC','T_cell','Mural'
print(f"\nDeveloping brain age clock for cell type: {celltype}")

# Subset data for target cell type
adata_temp = adata_train[adata_train.obs['celltype'] == celltype, :].copy()
print(f"  Cell count for {celltype}: {adata_temp.shape[0]}")
print(f"  Donor count for {celltype}: {adata_temp.obs['donor_id'].nunique()}")

# Normalize and preprocess data
print("\n  Preprocessing data...")
try:
    adata_temp = batch_sce_norm(adata_temp, batch_key='dataset')
except Exception as e:
    print(f"  Batch normalization failed, using simple normalization: {e}")
    sc.pp.normalize_per_cell(adata_temp)

# Log-transform and scale
sc.pp.log1p(adata_temp)
sc.pp.scale(adata_temp)

# ============================================================================
# 5. FEATURE SELECTION
# ============================================================================

print("\n" + "="*60)
print("Step 3: Feature Selection")
print("="*60)

# Load age-associated genes for the target cell type
print("Loading age-associated genes...")
age_genes_df = pd.read_csv('./cell_age_model_features.csv')
age_genes = age_genes_df.loc[age_genes_df['celltype'] == celltype, 'genename'].values
print(f"  Number of age-associated genes for {celltype}: {len(age_genes)}")

# Extract expression matrix for selected genes
# CORRECTED: removed extra closing bracket
X = adata_temp[:, age_genes].to_df().values
print(f"  Expression matrix shape: {X.shape}")

# Binary transformation from Z-scores
threshold = 0  # Threshold for binarization
X_binary = (X > threshold).astype(int)
print(f"  Binary matrix shape: {X_binary.shape}")

# Prepare target variable (chronological age)
y = adata_temp.obs['Age_at_death'].values
print(f"  Age range: {y.min()} to {y.max()} years")
print(f"  Mean age: {y.mean():.1f} years")

# ============================================================================
# 6. DATA SPLITTING
# ============================================================================

print("\n" + "="*60)
print("Step 4: Data Splitting")
print("="*60)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_binary, y, test_size=0.2, random_state=42)
print(f"  Training set size: {X_train.shape[0]} cells")
print(f"  Validation set size: {X_val.shape[0]} cells")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(f"  Batch size: {batch_size}")
print(f"  Number of training batches: {len(train_loader)}")

# ============================================================================
# 7. MODEL ARCHITECTURE
# ============================================================================

print("\n" + "="*60)
print("Step 5: Model Architecture Definition")
print("="*60)

# Define Transformer-based neural network model
import torch.nn.functional as F

class CellAgePredictor(nn.Module):
    """
    Transformer-based neural network for brain age prediction
    
    Architecture:
    - Transformer encoder layers for feature extraction
    - Fully connected layers for regression
    - Residual connections and layer normalization
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5, num_heads=4, num_layers=2):
        super(CellAgePredictor, self).__init__()
        
        # Ensure input_size is divisible by num_heads
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size,      # Input feature dimension
            nhead=num_heads,         # Number of attention heads
            dim_feedforward=hidden_size,  # Feedforward network hidden layer size
            dropout=dropout_prob,    # Dropout probability
            activation='relu'        # Activation function
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Residual connection
        self.residual = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Predicted age tensor of shape (batch_size, 1)
        """
        # Input x shape: (batch_size, input_size)
        # Transformer encoder requires shape: (seq_len, batch_size, input_size)
        x = x.unsqueeze(0)  # Add sequence dimension
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (seq_len, batch_size, input_size)
        x = x.squeeze(0)  # Remove sequence dimension
        
        # Residual connection + layer normalization
        residual = self.residual(x)
        out = self.fc1(x)
        out = self.norm1(out + residual)  # Residual connection + layer normalization
        out = F.relu(out)
        out = self.dropout(out)
        
        # Second fully connected layer
        out = self.fc2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Output layer
        out = self.fc3(out)
        
        return out

"""
# Alternative: Deep Neural Network (without transformer layers)
# Used for Ast and Mic brain age clocks development

class CellAgePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.3):
        super(CellAgePredictor, self).__init__()
        self.bn_input = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.bn_input(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return self.fc3(x)
"""

# ============================================================================
# 8. MODEL INITIALIZATION
# ============================================================================

print("\n" + "="*60)
print("Step 6: Model Initialization")
print("="*60)

# Set model hyperparameters
input_size = len(age_genes)
hidden_size = 512
output_size = 1
dropout_prob = 0.5
num_heads = 2
num_layers = 1

print(f"  Input size (number of genes): {input_size}")
print(f"  Hidden size: {hidden_size}")
print(f"  Number of attention heads: {num_heads}")
print(f"  Number of transformer layers: {num_layers}")
print(f"  Dropout probability: {dropout_prob}")

# Initialize model
model = CellAgePredictor(input_size, hidden_size, output_size, 
                         dropout_prob, num_heads, num_layers).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

print(f"  Loss function: Mean Squared Error (MSE)")
print(f"  Optimizer: AdamW with weight decay 1e-4")
print(f"  Initial learning rate: 0.001")

# ============================================================================
# 9. MODEL TRAINING
# ============================================================================

print("\n" + "="*60)
print("Step 7: Model Training")
print("="*60)

num_epochs = 100
best_val_loss = float('inf')
patience = 5
epochs_no_improve = 0

print(f"Starting training for {num_epochs} epochs...")
print(f"Early stopping patience: {patience} epochs")

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
    
    # Print progress
    print(f'Epoch [{epoch+1:3d}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    # Early stopping strategy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model
        torch.save(model.state_dict(), './Exc/transf_best_model.pth')
        print(f'  ✓ New best model saved (val_loss: {val_loss:.4f})')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    # Update learning rate
    scheduler.step(val_loss)

print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

# ============================================================================
# 10. MODEL EVALUATION ON TRAINING DATA
# ============================================================================

print("\n" + "="*60)
print("Step 8: Model Evaluation on Training Data")
print("="*60)

# Load best model
model.eval()
model.cpu()
model.load_state_dict(torch.load('./Exc/transf_best_model.pth'))
print("Best model loaded for evaluation")

# Prepare full training data for evaluation
X_full = adata_temp[:, age_genes].to_df().values
X_full_binary = (X_full > threshold).astype(int)
X_tensor = torch.tensor(X_full_binary, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    predicted_ages = model(X_tensor).cpu().numpy().flatten()

# Calculate evaluation metrics
y_true = adata_temp.obs['Age_at_death'].values
coef, p_value = spearmanr(y_true, predicted_ages)
mae = mean_absolute_error(y_true, predicted_ages)
r_squared = r2_score(y_true, predicted_ages)

print("\nTraining Set Evaluation:")
print(f"  Spearman R: {coef:.4f}, P-value: {p_value:.4e}")
print(f"  Mean Absolute Error (MAE): {mae:.2f} years")
print(f"  R-squared: {r_squared:.4f}")

# Store predictions
adata_temp.obs['age_pred'] = predicted_ages

# Calculate median predicted age by actual age
# CORRECTED: changed 'age_at_death' to 'Age_at_death'
age_summary = adata_temp.obs.groupby(['Age_at_death'])[['age_pred']].median()
print("\nMedian predicted age by actual age:")
print(age_summary.head())

# ============================================================================
# 11. INTERNAL VALIDATION ON TEST DATA
# ============================================================================

print("\n" + "="*60)
print("Step 9: Internal Validation on Test Data")
print("="*60)



# Load test datasets (corrected: test1, test2, test3, test4)
test1 = sc.read_h5ad('./sce_test1_0.h5ad')   # CT and NDDs groups for internal validation
test2 = sc.read_h5ad('./sce_test1_1.h5ad')
test3 = sc.read_h5ad('./sce_test2.h5ad')
test4 = sc.read_h5ad('./sce_test3.h5ad')

# Combine test datasets
adata_test = sc.concat([test1, test2, test3, test4], axis=0)

# Select only control samples for validation
adata_test = adata_test[adata_test.obs['status'] == 'CT', :].copy()
print(f"Test data shape: {adata_test.shape}")

# Subset to age-associated genes
adata_test2 = adata_test[:, age_genes].copy()

# Preprocess test data (same as training)
adata_test2 = batch_sce_norm(adata_test2, batch_key='dataset')
sc.pp.log1p(adata_test2)
sc.pp.scale(adata_test2)

# Prepare test data
X_test = adata_test2.to_df().values
X_test_binary = (X_test > threshold).astype(int)
X_test_tensor = torch.tensor(X_test_binary, dtype=torch.float32)
y_test_true = adata_test2.obs['Age_at_death'].values

# Make predictions on test data
with torch.no_grad():
    test_predicted_ages = model(X_test_tensor).cpu().numpy().flatten()

# Calculate evaluation metrics on test set
test_coef, test_p_value = spearmanr(y_test_true, test_predicted_ages)
test_mae = mean_absolute_error(y_test_true, test_predicted_ages)
test_r_squared = r2_score(y_test_true, test_predicted_ages)

print("\nTest Set Evaluation:")
print(f"  Spearman R: {test_coef:.4f}, P-value: {test_p_value:.4e}")
print(f"  Mean Absolute Error (MAE): {test_mae:.2f} years")
print(f"  R-squared: {test_r_squared:.4f}")



# ============================================================================
# 12. IDENTIFY CELLULAR BIOLOGICAL AGE-ASSOCIATED GENES
# ============================================================================

print("\n" + "="*60)
print("Step 10: Identify Cellular Biological Age-associated Genes")
print("="*60)

print("Identifying genes associated with predicted cellular age...")

# Prepare data for correlation analysis
obs = adata_temp.obs.copy()
df = adata_temp.to_df()

# Calculate Spearman correlation between gene expression and predicted cellular age
print(f"Calculating correlations for {df.shape[1]} genes...")
r_values = []
gene_names = []
p_values = []

for i in range(df.shape[1]):
    obs['gene'] = df.iloc[:, i].values.copy()
    
    # Extract gene expression and predicted cellular age
    # Note: 'predicted_cell_age' should be available in obs
    obs2 = obs.loc[:, ['gene', 'predicted_cell_age']]
    
    # Calculate Spearman correlation
    coef, p_value = spearmanr(obs2.iloc[:, 0].values, obs2.iloc[:, 1].values)
    r_values.append(coef)
    p_values.append(p_value)
    gene_names.append(df.columns[i])

# Store results
result = pd.DataFrame({
    'genename': gene_names,
    'r': r_values,
    'p': p_values
})

# Save discovery results
result.to_csv(f'./Exc/spearman_cellage_gene_dis.csv')
print(f"Discovery results saved to './Exc/spearman_cellage_gene_dis.csv'")
print(f"Number of genes analyzed: {len(result)}")


# Replication analysis (requires test data)
print("\nPerforming replication analysis...")
obs_test = adata_test2.obs.copy()
df_test = adata_test2.to_df()

r_rep = []
gene_names_rep = []
p_rep = []

for i in range(df_test.shape[1]):
    obs_test['gene'] = df_test.iloc[:, i].values
    # CORRECTED: 'cellage' should be replaced with actual column name
    obs2_test = obs_test.loc[:, ['gene', 'predicted_cell_age']]  # Use correct column name
    coef, p_value = spearmanr(obs2_test.iloc[:, 0].values, obs2_test.iloc[:, 1].values)
    r_rep.append(coef)
    p_rep.append(p_value)
    # CORRECTED: 'data.columns[i]' should be 'df_test.columns[i]'
    gene_names_rep.append(df_test.columns[i])

rep_result = pd.DataFrame({
    'genename': gene_names_rep,
    'r': r_rep,
    'p': p_rep
})

rep_result.to_csv('./Exc/spearman_cellage_gene_rep.csv')
print(f"Replication results saved to './Exc/spearman_cellage_gene_rep.csv'")


# ============================================================================
# 13. INTEGRATE DISCOVERY AND REPLICATION RESULTS
# ============================================================================

print("\n" + "="*60)
print("Step 11: Integrate Discovery and Replication Results")
print("="*60)

# Note: This section requires both discovery and replication results

# Load discovery and replication results
dis = pd.read_csv('./Exc/spearman_cellage_gene_dis.csv', index_col=0)
rep = pd.read_csv('./Exc/spearman_cellage_gene_rep.csv', index_col=0)

# Set indices for merging
dis.index = dis['genename'].values
rep.index = rep['genename'].values

# Find common genes
common_genes = np.intersect1d(dis.index, rep.index)
print(f"Number of common genes: {len(common_genes)}")

# Merge results
dis_common = dis.loc[common_genes, :].copy()
rep_common = rep.loc[common_genes, :].copy()

# Add replication results to discovery dataframe
dis_common['r.rep'] = rep_common['r'].values
dis_common['p.rep'] = rep_common['p'].values
dis_common['direct'] = dis_common['r'] / dis_common['r.rep']

# Apply FDR correction
_, fdr_p, _, _ = multipletests(dis_common['p'].values, method='fdr_bh')
dis_common['fdr'] = fdr_p

# Filter for consistent and significant associations
dis_filtered = dis_common.loc[
    (dis_common['direct'] > 0) &  # Same direction in discovery and replication
    (dis_common['fdr'] < 0.05) &   # FDR < 0.05 in discovery
    (dis_common['p.rep'] < 0.05),  # P < 0.05 in replication
    :
]

print(f"Number of significant and replicated genes: {len(dis_filtered)}")
print("\nTop 10 cellular biological age-associated genes:")
print(dis_filtered.sort_values('r', ascending=False).head(10))


# ============================================================================
print("\n" + "="*60)
print("scBACs Development Pipeline Complete!")
print("="*60)
print(f"Brain age clock for {celltype} cells has been developed and evaluated.")
print("Model saved to: './Exc/transf_best_model.pth'")