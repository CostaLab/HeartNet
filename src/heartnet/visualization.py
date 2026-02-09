import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import random

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