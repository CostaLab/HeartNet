import torch
import torch.nn as nn
import joblib
import os
import torch
import numpy as np
import anndata as ad
import scanpy as sc

class TissueClassifier(nn.Module):
    """
    HeartNet Model Architecture
    Simple MLP for tissue classification.
    """
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

def load_heartnet(checkpoint_dir, timestamp, device='cpu'):
    """
    Loads the HeartNet model weights and configuration.
    
    Args:
        checkpoint_dir (str): Path to the folder containing .pth and .pkl files
        timestamp (str): The timestamp string (e.g., '20251015_121453')
        device (str): 'cpu' or 'cuda'
        
    Returns:
        model (nn.Module): The loaded PyTorch model (eval mode)
        model_info (dict): Metadata including input_dim and class names
    """
    # Construct paths
    info_path = os.path.join(checkpoint_dir, f'{timestamp}.pkl')
    weights_path = os.path.join(checkpoint_dir, f'{timestamp}.pth')
    
    # Validation
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Info file not found: {info_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
    # Load metadata
    model_info = joblib.load(info_path)
    
    # Initialize model
    model = TissueClassifier(
        input_dim=model_info['input_dim'], 
        num_classes=model_info['num_classes']
    )
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print(f"✅ HeartNet loaded successfully from {timestamp}")
    return model, model_info





def predict_and_annotate(model, adata, class_names, input_key='X_pca', device='cpu'):
    """
    Runs inference on AnnData and adds predictions to .obs
    
    Args:
        model: Loaded PyTorch model
        adata: AnnData object containing embeddings
        class_names: List or dict of class names (from model_info)
        input_key: Key in adata.obsm to use (e.g. 'X_pca'). Use 'X' for raw counts.
        device: 'cuda' or 'cpu'
    """

    
    print(f"Running inference on {len(adata)} patches...")
    
    # 1. Prepare Data
    if input_key == 'X':
        data = adata.X
    elif input_key in adata.obsm:
        data = adata.obsm[input_key]
    else:
        raise ValueError(f"Key '{input_key}' not found in adata.obsm or X.")

    # Convert to Tensor
    if not isinstance(data, torch.Tensor):
        # Check if it's a pandas dataframe or numpy
        if hasattr(data, 'values'):
            data = data.values
        tensor_data = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        tensor_data = data.to(device)

    # 2. Inference
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        logits = model(tensor_data)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    # 3. Process Results
    probs_cpu = probabilities.cpu().numpy()
    preds_cpu = predictions.cpu().numpy()
    
    # Get max confidence
    confidence_scores = np.max(probs_cpu, axis=1)
    
    # Map indices to names
    predicted_labels = [class_names[p] for p in preds_cpu]

    # 4. Save to AnnData
    adata.obs['HeartNet_Pred'] = predicted_labels
    adata.obs['HeartNet_Confidence'] = confidence_scores
    
    print(f"Predictions added to adata.obs['HeartNet_Pred']")
    return adata
