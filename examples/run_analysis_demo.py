import os
from dim_red.analysis.workflow import run_pca_reduction
from dim_red.analysis.plotting import plot_reduced_space

if not os.environ.get("MP_API_KEY"):
    print("Error: MP_API_KEY environment variable is not set.")
    exit(1)

print("Starting analysis workflow...")
try:
    # Fetch and reduce Cubic and Tetragonal systems (10 structures each)
    X_reduced, labels, material_ids = run_pca_reduction(
        crystal_systems=["cubic", "tetragonal"],
        n_components=2,
        limit_per_system=10
    )
    print("Workflow executed successfully!")
    print("Reduced shape:", X_reduced.shape)
    
    # Save plot as dim_red_pca.png in the workspace root
    plot_path = "dim_red_pca.png"
    print("Generating plot...")
    plot_reduced_space(
        X_reduced,
        labels,
        title="PCA of SOAP descriptors (Cubic vs Tetragonal)",
        save_path=plot_path
    )
    print("Done! Plot saved to:", os.path.abspath(plot_path))
except Exception as e:
    print("Error during analysis:", str(e))
