"""
CODE FOR TRAINING THE ENSEMBLE INPAINTING MODEL FOR FACE INPAINTING.
RECOMMEND TO RUN THIS CODE FROM CONSOLE. THE USER SELECTS THE MODEL TO BE TRAINED
train_inpainting.py --model [Reconstruction, ablationv1, ablationv2, ablationv3]
"""

import torch
import torch.nn as nn
import os
import argparse
import logging

# User-defined model mapping
model_choices = {
    "reconstruction": "Reconstruction", ### Ensemble model
    "ablationv1": "ablationv1",
    "ablationv2": "ablationv2",
    "ablationv3": "ablationv3"
}

# CLI Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=model_choices.keys(), required=True, help='Model version to train')
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--imgsz', default=256, type=int)
args = parser.parse_args()

# Dynamically import the correct prepare_model
selected_module = __import__(model_choices[args.model], fromlist=['prepare_model'])
prepare_model = selected_module.prepare_model

# Logging and model save name based on model choice
log_filename = f"{args.model}_model_log.txt"
model_save_name = args.model.capitalize()

logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(message)s")

from datasetv3 import get_datasets_and_loaders
from EngineComplete import train, validate
from loss import total_loss  
from utils import save_model, SaveBestModel, save_plots

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

### Save information a .txt file
def logprint(message):
    print(message)
    logging.info(message)

if __name__ == '__main__':
    out_dir = os.path.join(r'.\outputs')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ### Models to be trained
    modelo, gan = prepare_model()
    modelo = modelo.to(device)
    gan = gan.to(device)
    
    ### Hyperparameters for the loss function
    a, b, c, d = 1.7, 4.0, 3.6, 4.7 ### L1, L2, L3, and L4
    total_trainable_params_modelo = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logging.info(f"{total_trainable_params_modelo:,} training parameters for model.")
    total_trainable_params_GAN = sum(p.numel() for p in gan.parameters() if p.requires_grad)
    logging.info(f"{total_trainable_params_GAN:,} training parameters for GAN.")
    
    ### Optimizer and losses for the reconstruction model and discriminator GAN model
    b1, b2 = 0.9, 0.999
    optimizer = torch.optim.Adam(modelo.parameters(), lr=args.lr, betas=(b1, b2))
    optimizer_gan = torch.optim.Adam(gan.parameters(), lr=args.lr, betas=(b1, b2))
    criterion = total_loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    ### Datasets path
    images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
    masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"

    train_dataset, valid_dataset, train_dataloader, valid_dataloader = get_datasets_and_loaders(
        images_folder, masks_folder, batch_size=args.batch, train_count=27000, valid_count=3000, seed=42, img_size=args.imgsz
    )
    
    save_best_model = SaveBestModel()
    EPOCHS = args.epochs
    train_loss, valid_loss = [], []
    best_valid_loss = float("inf")

    ### Train and validation loops
    for epoch in range(EPOCHS):
        logprint(f"EPOCH: {epoch + 1}")
        train_epoch_loss, mae_metric_train, psnr_metric_train, ssim_train, train_fid, train_lpips = train(
            modelo, gan, train_dataset, train_dataloader, device, optimizer, optimizer_gan,
            criterion, criterion_gan, a, b, c, d
        )
        valid_epoch_loss, mae_metric_valid, psnr_metric_valid, ssim_valid, valid_fid, valid_lpips = validate(
            modelo, gan, valid_dataset, valid_dataloader, device,
            criterion, criterion_gan, epoch, a, b, c, d
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        save_best_model(valid_epoch_loss, mae_metric_valid, psnr_metric_valid, epoch, modelo, out_dir, name=model_save_name)

        # Metrics Logging
        for name, value in zip(
            ["Train Loss", "Valid Loss", "Train MAE", "Valid MAE", "Train PSNR", "Valid PSNR", "Train SSIM", "Valid SSIM", "Train FID", "Valid FID", "Train LPIPS", "Valid LPIPS"],
            [train_epoch_loss, valid_epoch_loss, mae_metric_train, mae_metric_valid, psnr_metric_train, psnr_metric_valid, ssim_train, ssim_valid, train_fid, valid_fid, train_lpips, valid_lpips]
        ):
            logprint(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")

    save_model(EPOCHS, modelo, optimizer, criterion, out_dir, name=model_save_name)
    save_plots(train_loss, valid_loss, out_dir)
    print('TRAINING COMPLETE')