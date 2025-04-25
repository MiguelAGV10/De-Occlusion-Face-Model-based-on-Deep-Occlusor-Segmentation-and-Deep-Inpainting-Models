"""
CODE FOR TRAINING THE ENSEMBLE INPAINTING MODEL FOR GENERAL INPAINTING TASK PURPOSE.
IT TRAIN USING STL DATASET. WHICH HAS 7000 TRAINING IMAGES AND 1000 VALIDATION IMAGES.
SUCH DATASET HAS 10 DIFFERENT CLASSES
"""

import torch
import torch.nn as nn
import os
import argparse
from datasetgeneralv2 import get_datasets_and_loaders
from Reconstruction import prepare_model ### Ensemble model
from enginegeneral import train, validate
from loss import total_loss  
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

### Logging
logging.basicConfig(filename="GeneralInpainting.txt", level=logging.INFO, format="%(message)s")
def logprint(message):
    print(message)
    logging.info(message)

if __name__ == '__main__':

    ### Create a directory with the model name for outputs.
    out_dir = os.path.join(r'.\outputs')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelo, gan = prepare_model()
    modelo = modelo.to(device)
    gan = gan.to(device)
    a = 1.7
    b = 4.0
    c = 3.6
    d = 4.7

    ### Modelo
    total_params = sum(p.numel() for p in modelo.parameters())
    print (total_params)
    total_trainable_params_modelo = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"{total_trainable_params_modelo:,} training parameters.")
    
    ### GAN
    total_params = sum(p.numel() for p in gan.parameters())
    print (total_params)
    total_trainable_params_GAN = sum(p.numel() for p in gan.parameters() if p.requires_grad)
    print(f"{total_trainable_params_GAN:,} training parameters.")
    
    ### Optimizer and losses
    b1 = 0.9
    b2 = 0.999
    optimizer = torch.optim.Adam(modelo.parameters(), lr=args.lr,betas=(b1,b2))
    optimizer_gan = torch.optim.Adam(gan.parameters(), lr=args.lr,betas=(b1,b2))
    criterion = total_loss()
    criterion_gan = nn.BCEWithLogitsLoss()
    
    ###Dataset
    images_folder = r".\General\test_images" ### Path of the folder "General\test_images"
    masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"

    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder, masks_folder, args.img_size,
        train_count=7000, valid_count=1000,
        batch_size=8, seed=42, num_workers=2
    )
    
    save_best_model = SaveBestModel()

    EPOCHS = args.epochs
    train_loss = []
    valid_loss = []
    best_valid_loss = float("inf")
    
    ### Traning and validation loop stages
    for epoch in range (EPOCHS):
        logprint(f"EPOCH: {epoch + 1}")
        train_epoch_loss, mae_metric_train, psnr_metric_train,ssim_train, train_fid, train_lpips = train(
            modelo,
            gan,
            train_dataset,
            train_loader,
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
            valid_loader,
            device,
            criterion, 
            criterion_gan,
            epoch,
            a,
            b,
            c,
            d
        )
            
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        save_best_model(
            valid_epoch_loss,mae_metric_valid, psnr_metric_valid,epoch, modelo, out_dir, name='GeneralInpainting_loss'
        )
    
        ### Losses and metrics
        for name, value in zip(
            ["Train Loss", "Valid Loss", "Train MAE", "Valid MAE", "Train PSNR", "Valid PSNR", "Train SSIM", "Valid SSIM", "Train FID", "Valid FID", "Train LPIPS", "Valid LPIPS"],
            [train_epoch_loss, valid_epoch_loss, mae_metric_train, mae_metric_valid, psnr_metric_train, psnr_metric_valid, ssim_train, ssim_valid, train_fid, valid_fid, train_lpips, valid_lpips]
        ):
            logprint(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        
    save_model(EPOCHS, modelo, optimizer, criterion, out_dir, name='GeneralInpaintinBSDS300g')
    
    ### Save the loss and accuracy plots.
    save_plots(
        train_loss, valid_loss, out_dir
    )
    print('TRAINING COMPLETE')


 