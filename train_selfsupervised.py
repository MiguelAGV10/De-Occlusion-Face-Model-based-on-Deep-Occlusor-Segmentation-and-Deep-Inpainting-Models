"""
CODE FOR TRAINING THE ENSEMBLE INPAINTING MODEL FOR FACE INPAINTING USING SELF-SUPERVISED LEARNING.
SOME RANDOM OCCLUSORS AND SYNTHETIC ONES ARE ADDED RANDOMLY INTO FACES, THE MODEL BY ITSELF LEARNS
THE REGIONS TO BE INPAINTED.
"""

import torch
import torch.nn as nn
import os
import argparse
from datasetself import get_datasets_and_loaders
# from torch.optim.lr_scheduler import MultiStepLR
from Reconstruction import prepare_model #EXP1
from engineself import train, validate
from lossself import total_loss  
from utils import save_model, SaveBestModel, save_plots
import logging

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

### Hyperparameters (hyperparameters can change)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--imgsz', default=256, type=int)
args = parser.parse_args()

logging.basicConfig(filename="Selfsupervised.txt", level=logging.INFO, format="%(message)s")

def logprint(message):
    print(message)
    logging.info(message)

if __name__ == '__main__':
    
    # Create a directory with the model name for outputs.
    out_dir = os.path.join(r'.\outputs')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelo, gan = prepare_model()
    modelo = modelo.to(device)
    gan = gan.to(device)
    a = 1.7
    b = 4.0
    c = 3.6
    d = 4.7

    # Modelo
    total_params = sum(p.numel() for p in modelo.parameters())
    print (total_params)
    total_trainable_params_modelo = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"{total_trainable_params_modelo:,} training parameters.")
    
    # GAN
    total_params = sum(p.numel() for p in gan.parameters())
    print (total_params)
    total_trainable_params_GAN = sum(p.numel() for p in gan.parameters() if p.requires_grad)
    print(f"{total_trainable_params_GAN:,} training parameters.")
    
    b1 = 0.0
    b2 = 0.9
    
    optimizer = torch.optim.Adam(modelo.parameters(), lr=args.lr,betas=(b1,b2))
    optimizer_gan = torch.optim.Adam(gan.parameters(), lr=args.lr,betas=(b1,b2))
    criterion = total_loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    ### Dataset
    images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
    occluder_folder = r".\Selfsupervisedocclusors\train" ### Path of the folder "Selfsupervisedocclusors\train"

    # Create datasets and dataloaders
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = get_datasets_and_loaders(
        images_folder, occluder_folder, batch_size=4, train_count=27000, valid_count=3000, seed=42, img_size=256
    )
    
    save_best_model = SaveBestModel()
    EPOCHS = args.epochs
    train_loss = []
    valid_loss = []
    best_valid_loss = float("inf")
    
    ### Training and validation loop stages
    for epoch in range (EPOCHS):
        logprint(f"EPOCH: {epoch + 1}")
        train_epoch_loss, mae_metric_train, psnr_metric_train,ssim_train, train_fid, train_lpips = train(
            modelo,
            gan,
            train_dataset,
            train_dataloader,
            device,
            optimizer,
            optimizer_gan,
            criterion,
            criterion_gan,
            a,
            b,
            c,
            d
        )

        valid_epoch_loss, mae_metric_valid, psnr_metric_valid,ssim_valid,valid_fid, valid_lpips = validate(
            modelo,
            gan,
            valid_dataset,
            valid_dataloader,
            device,
            criterion, 
            criterion_gan,
            epoch,
            a,
            b,
            c,
            d
        )
        
        ### Save best models
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        save_best_model(
            valid_epoch_loss,mae_metric_valid, psnr_metric_valid,epoch, modelo, out_dir, name='Selfsupervised_loss'
        )
        
        ### Losses and metrics
        for name, value in zip(
            ["Train Loss", "Valid Loss", "Train MAE", "Valid MAE", "Train PSNR", "Valid PSNR", "Train SSIM", "Valid SSIM", "Train FID", "Valid FID", "Train LPIPS", "Valid LPIPS"],
            [train_epoch_loss, valid_epoch_loss, mae_metric_train, mae_metric_valid, psnr_metric_train, psnr_metric_valid, ssim_train, ssim_valid, train_fid, valid_fid, train_lpips, valid_lpips]
        ):
            logprint(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        
    ### Save the loss and accuracy plots    
    save_model(EPOCHS, modelo, optimizer, criterion, out_dir, name='Selfsupervised')
    save_plots(
        train_loss, valid_loss, out_dir
    )
    print('TRAINING COMPLETE')
