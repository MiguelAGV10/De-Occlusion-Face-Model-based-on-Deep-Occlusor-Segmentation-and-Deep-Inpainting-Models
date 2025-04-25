"""
CODE FOR PERFORMING INFERENCES AS WELL AS CALCULATING METRICS OF ALL THE AVAILABLE MODELS
(ENSEMBLE AND ABLATIONS) SPECIFIED IN models (LINE 27)
"""

import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from datasetmetricas import get_datasets_and_loaders
from Reconstruction import prepare_model
from ablationv1 import prepare_model as modelablationv1
from ablationv2 import prepare_model as modelablationv2
from ablationv3 import prepare_model as modelablationv3
from metrics import compare_mae, compare_psnr, ssim_val, compare_fid, compare_lpips
import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

### Logging setup
logging.basicConfig(filename="Metrics.txt", level=logging.INFO, format="%(message)s") ### Recommend to change the .txt name if we change the model
def logprint(message):
    print(message)
    logging.info(message)

### Model paths and labels
models = [
    ("Reconstruction", r".\reconstruction.pth"), 
    ("AblationV1", r".\ablationv1.pth"),
    ("AblationV2", r".\ablationv2.pth"),
    ("AblationV3", r".\ablationv3.pth"),
]

### Config
images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"
save_folder = r".\inference_outputs" ### Inference outputs folder
os.makedirs(save_folder, exist_ok=True)
img_size = 256
val_bin_count = 3000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Inference function
def run_inference_for_model(model_name, checkpoint_path):
    logprint(f"\n\n### {model_name.upper()} ###")
    ### Load model
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

        ### Load dataset for this bin
        _, valid_dataset, _, valid_loader = get_datasets_and_loaders(
            images_folder, masks_folder, img_size,
            train_count=27000, valid_count=val_bin_count,
            batch_size=8, seed=42, num_workers=4,
            val_bin_index=val_bin_index
        )

        ### Metrics initizaliation
        mae_metric = 0
        psnr_metric = 0
        ssim_metric = 0
        fidmetric = 0
        lpipsmetric = 0
        counter = 0

        ### Propagation and inference stage
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(valid_loader, desc=f"{model_name} - Bin {val_bin_index}")):
                mask, target = data[1].to(device), data[0].to(device)
                input_data = target * (1 - mask) + mask
                inpainted_image = model(input_data)
                merged = target * (1 - mask) + inpainted_image * mask

                target_detached = target.detach()
                merged_detached = merged.detach()

                mae_metric += compare_mae(merged_detached, target_detached)
                psnr_metric += compare_psnr(merged_detached, target_detached)
                ssim_metric += ssim_val(merged_detached, target_detached)

                
                fidmetric += compare_fid(merged_detached, target)
                lpipsmetric += compare_lpips(merged_detached, target).item()
                    
                if counter % 50 == 0:  
                    img_idx = 0

                    ### Detach tensors and clamp to [0, 1] if needed
                    img_mask = target[img_idx].detach().cpu().clamp(0, 1)
                    img_data = input_data[img_idx].detach().cpu().clamp(0, 1)
                    img_inpainted = merged[img_idx].detach().cpu().clamp(0, 1)
                    
                    ### Convert to PIL images
                    img_mask_pil = TF.to_pil_image(img_mask)
                    img_data_pil = TF.to_pil_image(img_data)
                    img_inpainted_pil = TF.to_pil_image(img_inpainted)
                    
                    ### Plotting
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(img_mask_pil)
                    axs[0].set_title("Original image")
                    axs[0].axis('off')
                    
                    axs[1].imshow(img_data_pil)
                    axs[1].set_title("Input image")
                    axs[1].axis('off')
                    
                    axs[2].imshow(img_inpainted_pil)
                    axs[2].set_title("Inpainted Output")
                    axs[2].axis('off')
                    
                    plt.tight_layout()
                    plt.show()

                counter += 1
                
        ### Averages
        logprint(f"MAE   : {mae_metric / counter:.4f}")
        logprint(f"SSIM  : {ssim_metric / counter:.4f}")
        logprint(f"PSNR  : {psnr_metric / counter:.2f}")
        logprint(f"LPIPS : {lpipsmetric / counter:.4f}")
        logprint(f"FID   : {fidmetric / counter:.4f}")

if __name__ == "__main__":
    for model_name, checkpoint_path in models:
        run_inference_for_model(model_name, checkpoint_path)