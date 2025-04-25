"""
CODE FOR CALCULATING INFERENCE TIME STATISTICS.
"""

import os
import torch
from tqdm import tqdm
from datasetmetricas import get_datasets_and_loaders
from Reconstruction import prepare_model
import logging
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import time

### Logging
logging.basicConfig(filename="Time_statistics.txt", level=logging.INFO, format="%(message)s")
def logprint(message):
    print(message)
    logging.info(message)
    
if __name__ == "__main__":
    
    images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
    masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"
    
    ### The ensemble learning and the ablation ones can be used to measure the inference time
    checkpoint_path = r"./reconstruction.pth" ### Path of the "reconstruction" model (ensemble model)
    # checkpoint_path = r".\best_loss_ablationv1_loss.pth" ### Path of the "ablationv1" model (AblationV1 file)
    # checkpoint_path = r".\best_loss_ablationv2_loss.pth" ### Path of the "ablationv2" model (AblationV2 file)
    # checkpoint_path = r".\best_loss_ablationv3_loss.pth" ### Path of the "ablationv3" model (AblationV3 file)
    
    save_folder = r"E:\inference_outputs"
    os.makedirs(save_folder, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ### Download the trained model
    model, _ = prepare_model()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    ### Specify the bin you want (0 to 5)
    val_bin_index = 1  ### For bin 1–10%
    val_bin_count = 3000  ### How many val images/masks to sample
    
    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder, masks_folder, img_size=256,
        train_count=27000, valid_count=val_bin_count,
        batch_size=8, seed=42, num_workers=2,
        val_bin_index=val_bin_index
    )
    
    ### Time variables
    total_time = 0.0
    total_images = 0
    
    ### Inference loop
    with torch.no_grad():
        
        counter = 0
        
        for batch_idx, data in enumerate(tqdm(valid_loader, desc="Running Inference")):
            
            ### Start timing
            start_time = time.time()
            
            mask, target = data[1].to(device), data[0].to(device)
            data_input = target * (1 - mask) + mask
            
            ### Forward pass
            inpainted_image = model(data_input)
            
            ### If using GPU, synchronize to get accurate time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            
            ### Count how many images in this batch
            batch_size = target.size(0)
            total_images += batch_size
            ### End timing 
            
            # (Optional) Merge step or other post-processing
            merged = target*(1 - mask) + inpainted_image*mask
            
            # Visualization every 5th batch
            if counter % 5 == 0:
                img_idx = 0
                img_mask = target[img_idx].detach().cpu().clamp(0, 1)
                img_data = data_input[img_idx].detach().cpu().clamp(0, 1)
                img_inpainted = inpainted_image[img_idx].detach().cpu().clamp(0, 1)
                
                img_mask_pil = TF.to_pil_image(img_mask)
                img_data_pil = TF.to_pil_image(img_data)
                img_inpainted_pil = TF.to_pil_image(img_inpainted)
                
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

    ### Inference time statistics
    num_batches = batch_idx + 1  ### total loop iterations
    avg_time_per_batch = total_time / num_batches
    avg_time_per_image = total_time / total_images if total_images > 0 else 0
    fps = 1.0 / avg_time_per_image if avg_time_per_image > 0 else 0
    
    print("\n### Inference Time Results ###")
    print(f"Total Batches   : {num_batches}")
    print(f"Total Images    : {total_images}")
    print(f"Total Time      : {total_time:.4f} seconds")
    print(f"Time per Batch  : {avg_time_per_batch:.4f} seconds")
    print(f"Time per Image  : {avg_time_per_image:.4f} seconds")
    print(f"Approx. FPS     : {fps:.2f}")
    
    logprint("\n### Inference Time Results ###")
    logprint(f"Total Batches   : {num_batches}")
    logprint(f"Total Images    : {total_images}")
    logprint(f"Total Time      : {total_time:.4f} seconds")
    logprint(f"Time per Batch  : {avg_time_per_batch:.4f} seconds")
    logprint(f"Time per Image  : {avg_time_per_image:.4f} seconds")
    logprint(f"Approx. FPS     : {fps:.2f}")