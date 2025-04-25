"""
CODE FOR GETTING UMAP'S METRICS
"""

import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from datasetmetricas import get_datasets_and_loaders
from ReconstructionUMAP import prepare_model
from ablationv1UMAP import prepare_model as modelablationv1
from ablationv2UMAP import prepare_model as modelablationv2
from ablationv3UMAP import prepare_model as modelablationv3
from metrics import compare_mae, compare_psnr, ssim_val, compare_fid, compare_lpips
import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

### Logging setup
logging.basicConfig(filename="UMAP.txt", level=logging.INFO, format="%(message)s")
def logprint(message):
    print(message)
    logging.info(message)

### Model paths and labels (specify the model's path)
models = [
    ("Reconstruction", r".\reconstruction.pth"), 
    ("AblationV1", r".\ablationv1.pth"),
    ("AblationV2", r".\ablationv2.pth"),
    ("AblationV3", r".\ablationv3.pth"),
]

### Config
images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"
save_folder = r"E:\inference_outputs" ### Path of the output folder
os.makedirs(save_folder, exist_ok=True)
img_size = 256
val_bin_count = 1500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Inference function
def run_inference_for_model(model_name, checkpoint_path):
    logprint(f"\n\n### {model_name.upper()} ###")

    # Load model
    if model_name == "Modelo Completo":    
        model, _ = prepare_model()
    if model_name == "AblationV1":
        model, _ = modelablationv1()
    elif model_name == "AblationV2":
        model,_ = modelablationv2()
    elif model_name == "AblationV3":
        model, _ = modelablationv3()
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    for val_bin_index in range(6):
        logprint(f"\n=== BIN {val_bin_index} ===")

        _, valid_dataset, _, valid_loader = get_datasets_and_loaders(
            images_folder, masks_folder, img_size,
            train_count=27000, valid_count=val_bin_count,
            batch_size=8, seed=42, num_workers=4,
            val_bin_index=val_bin_index
        )

        all_features = []
        all_img_ids = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(valid_loader, desc=f"{model_name} - Bin {val_bin_index}")):
                mask, target = data[1].to(device), data[0].to(device)
                input_data = target * (1 - mask) + mask

                features = model(input_data, return_features=True)  ### [B, C, H, W]
                B, C, H, W = features.shape
                features = features.view(B, -1).cpu().numpy()
                all_features.append(features)
                all_img_ids.extend([f"{val_bin_index}_{batch_idx}_{i}" for i in range(B)])

        all_features = np.concatenate(all_features, axis=0)  ### Shape: [N, D]
        all_features = StandardScaler().fit_transform(all_features)

        ### UMAP projection
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors = 20) ### n_heigbors can be modified
        embedding = reducer.fit_transform(all_features)  ### [N, 2]
        
        kmeans = KMeans(n_clusters=1, random_state=42) ### n_clusters can be modified
        cluster_labels = kmeans.fit_predict(all_features)

        ### Save embeddings and features
        np.save(f"{model_name}_bin{val_bin_index}_umap_embeddings.npy", embedding)
        np.save(f"{model_name}_bin{val_bin_index}_features.npy", all_features)

        ### Metrics
        try:
            ### Silhouette Score: no labels, so use dummy labels
            sil_score = silhouette_score(all_features, cluster_labels)
            db_index = davies_bouldin_score(all_features, cluster_labels)
        except Exception as e:
            sil_score = db_index = None
            logprint(f"Metric computation error (clustering-based): {e}")

        ### Mean pairwise distance in UMAP space
        mean_distance = np.mean(pairwise_distances(embedding))

        ### Variance explained by PCA (for comparison)
        try:
            pca = PCA(n_components=2)
            pca.fit(all_features)
            explained_variance = np.sum(pca.explained_variance_ratio_)
        except Exception as e:
            explained_variance = None
            logprint(f"PCA variance computation error: {e}")

        ### Log metrics
        logprint(f"Silhouette Score (dummy labels): {sil_score:.4f}" if sil_score is not None else "Silhouette Score: error")
        logprint(f"Davies-Bouldin Index (dummy labels): {db_index:.4f}" if db_index is not None else "Davies-Bouldin Index: error")
        logprint(f"Mean Pairwise Distance in UMAP space: {mean_distance:.4f}")
        logprint(f"Variance Explained by PCA (2D): {explained_variance:.4f}" if explained_variance is not None else "Variance Explained: error")

        ### Plot UMAP 
        plt.figure(figsize=(10, 8))
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10, alpha=0.7)
        plt.title(f"UMAP of {model_name} - Bin {val_bin_index}")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{model_name}_bin{val_bin_index}_umap.png")
        plt.show()

# === MAIN ===
if __name__ == "__main__":
    for model_name, checkpoint_path in models:
        run_inference_for_model(model_name, checkpoint_path)