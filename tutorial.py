
"""
scBACs Tutorial Examples
This file contains Python code examples for using scBACs.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scbac import (
    predict_cell_age,
    preprocess_data,
    calculate_donor_celltype_age_gap,
    age_gap_turning_point_analysis,
    analyze_age_gap_ratio,
    fast_calculate_donor_curve_thresholds,
    analyze_threshold_differences
)

# Example 1: Basic Age Prediction Pipeline
def example1_basic_prediction():
    """Basic age prediction example."""
    print("Example 1: Basic Age Prediction Pipeline")
    print("-" * 50)
    
    # Load and preprocess data
    adata = sc.read_h5ad("brain_data.h5ad")
    print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Preprocess data
    adata = preprocess_data(adata, normalize=True, scale=True)
    print("Data preprocessing completed")
    
    # Predict ages
    results = predict_cell_age(adata)
    print("Age prediction completed")
    
    # Save results
    adata.obs['cell_age'] = results['cell_ages']
    adata.write("results_with_ages.h5ad")
    print("Results saved to results_with_ages.h5ad")
    
    # Create visualization
    sc.pl.umap(adata, color=['celltype', 'cell_age'], save='_cell_age_umap.pdf')
    print("UMAP plot saved")

# Example 2: Complete Aging Analysis Workflow
def example2_complete_analysis():
    """Complete aging analysis workflow."""
    print("\nExample 2: Complete Aging Analysis Workflow")
    print("-" * 50)
    
    # Load your data
    meta_data = pd.read_csv("brain_metadata.csv")
    print(f"Loaded metadata with {len(meta_data)} rows")
    
    # Calculate age gap
    df_with_gap = calculate_donor_celltype_age_gap(meta_data)
    print("Age gap calculation completed")
    
    # Analyze a specific disease
    ad_results = age_gap_turning_point_analysis(
        df_with_gap,
        disease_name='AD',
        save_prefix='AD_results'
    )
    print("AD analysis completed")
    
    # Analyze positive age gap ratio
    ratio_results = analyze_age_gap_ratio(
        df_with_gap[df_with_gap['status'] == 'AD']
    )
    print("Positive ratio analysis completed")
    
    # Correlate with clinical scores (if available)
    try:
        clinical_data = pd.read_csv("clinical_scores.csv")
        merged_data = pd.merge(ratio_results, clinical_data, on='PaticipantID_unique')
        
        # Plot correlation
        plt.figure(figsize=(8, 6))
        plt.scatter(merged_data['positive_ratio'], merged_data['MMSE'], alpha=0.7)
        plt.xlabel('Positive Age Gap Ratio')
        plt.ylabel('MMSE Score')
        plt.title('Cognitive Function vs. Cellular Aging')
        plt.savefig('correlation_plot.pdf', dpi=300, bbox_inches='tight')
        print("Correlation plot saved")
    except FileNotFoundError:
        print("Clinical scores file not found, skipping correlation analysis")

# Example 3: Custom Model Usage
def example3_custom_model():
    """Example using custom model directory."""
    print("\nExample 3: Custom Model Usage")
    print("-" * 50)
    
    adata = sc.read_h5ad("your_data.h5ad")
    
    # Use custom model directory
    results = predict_cell_age(
        adata,
        model_directory="/path/to/custom/models",
        device='cuda'  # Use GPU acceleration
    )
    print(f"Predicted ages for {len(results['cell_ages'])} cells using custom models")


# Example 4: Custom Analysis Parameters
def example4_custom_analysis():
    """Example with custom analysis parameters."""
    print("\nExample 5: Custom Analysis Parameters")
    print("-" * 50)
    
    # Load data
    df = pd.read_csv("predicted_ages.csv")
    
    # Custom threshold calculation
    threshold_df, curve_data = fast_calculate_donor_curve_thresholds(
        df,
        disease_name='AD',
        min_cells=15,      # Minimum cells per donor-celltype
        n_points=200       # Resolution of smooth curves
    )
    print(f"Calculated thresholds for {len(threshold_df)} donor-celltype combinations")
    
    # Custom statistical analysis
    analysis_results = analyze_threshold_differences(threshold_df)
    print("Statistical analysis completed")

# Example 5: Testing with Sample Data
def example5_test_with_sample_data():
    """Test scBACs with generated sample data."""
    print("\nExample 6: Testing with Sample Data")
    print("-" * 50)
    
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
    
    print(f"Generated sample data with {n_cells} cells")
    print(f"Columns: {list(sample_data.columns)}")
    
    # Test age gap calculation
    df_with_gap = calculate_donor_celltype_age_gap(sample_data)
    print(f"Age gap range: {df_with_gap['age_gap'].min():.2f} to {df_with_gap['age_gap'].max():.2f}")

# Main function to run all examples
def main():
    """Run all tutorial examples."""
    print("scBACs Tutorial Examples")
    print("=" * 50)
    
    # Note: Uncomment the examples you want to run
    # example1_basic_prediction()
    # example2_complete_analysis()
    # example3_custom_model()
    # example4_custom_analysis()
    example5_test_with_sample_data()
    
    print("\nTutorial completed!")

if __name__ == "__main__":
    main()
