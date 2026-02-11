import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import random
import math



def plot_umap_with_patches(adata, patch_base_folder, 
                           slide_col="Slide_ID", 
                           patch_col="Patch_ID_x", 
                           cluster_col="leiden", 
                           zoom=0.15, 
                           n_patches=50,
                           figsize=(40, 40)):
    """
    Plots UMAP and overlays actual H&E patches on top of the coordinates.
    """
    
    # Extract coordinates and metadata
    umap_coords = adata.obsm["X_umap"]
    clusters = adata.obs[cluster_col].astype(str)
    sample_ids = adata.obs[slide_col].astype(str)
    patch_ids = adata.obs[patch_col].astype(str)

    # Helper function to load image
    def get_image(path, zoom_level):
        return OffsetImage(plt.imread(path), zoom=zoom_level)

    # Create Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c="lightgrey", alpha=0.5, label="Cells")

    print(f"Overlaying patches from: {patch_base_folder}")

    # Process Each Cluster
    for cluster in sorted(clusters.unique()):
        cluster_indices = np.where(clusters == cluster)[0]
        if len(cluster_indices) == 0:
            continue

        # Select random points
        selected_count = min(n_patches, len(cluster_indices))
        selected_indices = random.sample(list(cluster_indices), selected_count)

        for i in selected_indices:
            x0, y0 = umap_coords[i]
            sample_id = sample_ids[i]
            patch_id = patch_ids[i]

            # Construct Path
            patch_folder = os.path.join(patch_base_folder, sample_id)
            patch_path = os.path.join(patch_folder, patch_id)

            if not os.path.exists(patch_path):
                # Try alternative path logic if needed, or just skip
                continue

            # Add Image
            try:
                ab = AnnotationBbox(get_image(patch_path, zoom), (x0, y0), frameon=False, pad=0)
                ax.add_artist(ab)
            except Exception as e:
                print(f"Error loading {patch_path}: {e}")

        # Annotate Cluster Center
        avg_x, avg_y = np.mean(umap_coords[cluster_indices], axis=0)
        ax.annotate(f"Cluster {cluster}", (avg_x, avg_y), 
                    fontsize=20, ha="center", color="black", weight="bold", 
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7))

    ax.set_title(f"UMAP with {n_patches} Patches per Cluster")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    return fig



def create_sample_grid(df, image_dir='/data/', 
                      output_path="/data/", 
                      zoom=0.15, num_rows=10):
    """
    Create a grid of sample images based on ordering in a DataFrame.
    
    Args:
        df: DataFrame containing at least:
            - 'sampleID': unique sample identifiers
            - 'order': numerical ordering for sorting
        image_dir: Directory containing images (named as {sampleID}.png)
        output_path: Where to save the resulting grid
        zoom: Zoom level for images
        num_rows: Number of rows in the grid
    """
    # Sort samples by the specified order
    sorted_df = df.sort_values('pseudotime')
    sample_names = [name.replace('.ome', '') for name in sorted_df['sampleID'].tolist()]
    
    # Filter out samples that don't have corresponding PNG files
    valid_samples = []
    for sample in sample_names:
        img_path = os.path.join(image_dir, f"{sample}.png")
        if os.path.exists(img_path):
            valid_samples.append(sample)
        else:
            print(f"Warning: Image not found for sample {sample} at {img_path}")
    
    num_samples = len(valid_samples)
    
    if num_samples == 0:
        print("No valid images found!")
        return
    
    # Calculate grid dimensions
    num_cols = math.ceil(num_samples / num_rows)
    
    # Prepare figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
    # Flatten axes array for easier indexing
    if num_rows > 1 or num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each valid sample
    for i, sample in enumerate(valid_samples):
        img_path = os.path.join(image_dir, f"{sample}.png")
        ax = axes[i]
        
        # Add image
        img = OffsetImage(plt.imread(img_path), zoom=zoom)
        ab = AnnotationBbox(img, (0.5, 0.5), 
                          frameon=False, 
                          box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
        
        # Minimal labeling
       # ax.set_title(sample, fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig.canvas.draw()  # Force render
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grid with {num_samples} images to {output_path}")