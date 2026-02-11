import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os
import random
import math
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import random
from PIL import Image
import textwrap


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

def generate_patch_matrix(
    adata_obs=None,
    base_path="/data/",
    unique_classes= [
 'myocardium',
 'damaged myocardium 1',
 'damaged myocardium 2',
 'damaged myocardium 3',
 'fresh infarct with demarcation, early granulation',
 'granulation tissue, perivascular scar with immune cells (older)',
 'fibrotic tissue, scar tissue',
 'necrotic tissue',
 'fat tissue, adipose tissue',
],
    class_colors={
    'myocardium':'#b01111',
    'damaged myocardium 1':'#b4451f',
    'damaged myocardium 2':'#dd9f40',
    'damaged myocardium 3':'#e7d87d',
    'fresh infarct with demarcation, early granulation':'blue',
    'granulation tissue, perivascular scar with immune cells (older)':'#984ea3',
    'fibrotic tissue, scar tissue':'green',
    'necrotic tissue':'#A9A9A9',
    'fat tissue, adipose tissue':'#FF8DE6',
}
,
    save_manifest="selected_patches_per_class.csv",
    save_fig_prefix="patch_matrix",
    n_cols=4,
    annotation_width=3,
    color_width=0.4,
    patch_size_inch=2,
    dpi=300,
    Sample_ID='Sample_ID',
    Patch_ID='Patch_ID',
):
    """
    Generates a matrix of image patches organized by class with a color legend column.
    """
    
    def check_and_load_patch(p_id, s_id):
        file_path = os.path.join(base_path, f"{s_id}.ome", f"{p_id}")
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path)
                return np.array(img), file_path
            except:
                return None, None
        return None, None

    # 1. Load or Select Patches
    class_selected_patches = {}
    
    if os.path.exists(save_manifest):
        selected_df = pd.read_csv(save_manifest)
        for class_name in unique_classes:
            rows = selected_df[selected_df["class"] == class_name]
            patches_list = []
            for _, r in rows.iterrows():
                if os.path.exists(r["file_path"]):
                    img = np.array(Image.open(r["file_path"]))
                    patches_list.append({'img_array': img})
            class_selected_patches[class_name] = patches_list
    else:
        print("Selecting patches randomly...")
        selected_records = []
        for class_name in unique_classes:
            class_data = adata_obs[adata_obs['patch_annotation'] == class_name]
            existing_patches = []
            
            # Shuffle and check existence
            indices = list(range(len(class_data)))
            random.shuffle(indices)
            
            for idx in indices:
                if len(existing_patches) >= n_cols:
                    break
                row = class_data.iloc[idx]
                img_array, fp = check_and_load_patch(row[Patch_ID], row[Sample_ID])
                if img_array is not None:
                    existing_patches.append({'img_array': img_array})
                    selected_records.append({
                        'class': class_name, 'patch_id': row[Patch_ID],
                        'sample_id': row[Sample_ID], 'file_path': fp
                    })
            class_selected_patches[class_name] = existing_patches
        
        pd.DataFrame(selected_records).to_csv(save_manifest, index=False)
    

    # 2. Setup Figure
    n_rows = len(unique_classes)
    fig, axes = plt.subplots(
        n_rows, n_cols + 2,
        figsize=(12 + annotation_width, patch_size_inch * n_rows),
        gridspec_kw={'width_ratios': [annotation_width] + [2] * n_cols + [color_width]}
    )
    if n_rows == 1: axes = axes.reshape(1, -1)

    # 3. Fill Matrix
    for i, class_name in enumerate(unique_classes):
        # Column 0: Text Label
        ax_ann = axes[i, 0]
        ax_ann.axis('off')
        ax_ann.text(0.5, 0.5, textwrap.fill(str(class_name), width=30), 
                    fontsize=14, ha='center', va='center')

        # Columns 1 to N: Patches
        patches_list = class_selected_patches.get(class_name, [])
        for j in range(n_cols):
            ax = axes[i, j + 1]
            ax.axis('off')
            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                fill=False, edgecolor='black', linewidth=2)
            
            if j < len(patches_list):
                ax.imshow(patches_list[j]['img_array'])
            else:
                ax.text(0.5, 0.5, "No patch", ha='center', va='center', color='gray')
            ax.add_patch(rect)

        # Last Column: Color Indicator
        ax_color = axes[i, n_cols + 1]
        ax_color.axis("off")
        color = class_colors.get(class_name, 'white')
        ax_color.add_patch(patches.Rectangle((0, 0), 1, 1, transform=ax_color.transAxes, color=color))

    plt.tight_layout()
    plt.savefig(f"{save_fig_prefix}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{save_fig_prefix}.pdf", bbox_inches='tight')
    plt.show()

