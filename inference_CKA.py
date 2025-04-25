"""
CODE FOR GETTING CKA VALUES BETWEEN ENCODER'S LEVEL 0
"""

import torch
import numpy as np
from tqdm import tqdm
from datasetmetricas import get_datasets_and_loaders
from reconstruction import prepare_model ### ensemble model
from sklearn.metrics.pairwise import linear_kernel

### Calculate CKA involves the gram matrices
def gram_linear(x):
    return linear_kernel(x, x)

def center_gram(gram):
    n = gram.shape[0]
    ones = np.ones((n, n)) / n
    return gram - ones @ gram - gram @ ones + ones @ gram @ ones

def cka(X, Y):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    K = center_gram(gram_linear(X))
    L = center_gram(gram_linear(Y))
    return (K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L))


def extract_features(model, dataloader, device):
    model.eval()
    model.to(device)
    features_m1, features_m2 = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            mask, target = batch[1].to(device), batch[0].to(device)
            input_data = target * (1 - mask) + mask
            
            ### We get each encoder's models level 0 outputs
            e1 = model.feature_extractor(input_data).cpu().numpy() ### self.feature_extractor (ResNet) output
            e2 = model.encoder(input_data).cpu().numpy() ### self.encoder ouutput    
            features_m1.append(e1.reshape(e1.shape[0], -1))
            features_m2.append(e2.reshape(e2.shape[0], -1))

    ### Concatenation of all features
    f1 = np.concatenate(features_m1, axis=0)
    f2 = np.concatenate(features_m2, axis=0)
    return f1, f2


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = prepare_model()
    checkpoint_path = r"reconstruction.pth" ### Path to the model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    images_folder = r".\CelebA-HQ-img" ### Path to the folder "CelebA-HQ-img"
    masks_folder = r".\testing_mask_dataset" ### Path to the folder "testing_mask_dataset"
    _, valid_dataset, _, valid_loader = get_datasets_and_loaders(
        images_folder, masks_folder, 256,
        train_count=27000, valid_count=3000,  
        batch_size=8, seed=42, num_workers=4, val_bin_index=0 ### val_bin_index range is 0 up to 5.
    )

    f1, f2 = extract_features(model, valid_loader, device)
    print("CKA(M1, M2):", cka(f1, f2)) ### CKA between models level 0 in the encoder

