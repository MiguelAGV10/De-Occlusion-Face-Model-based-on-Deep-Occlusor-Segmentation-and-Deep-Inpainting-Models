
"""
CODE FOR FINDING THE BEST INPAINTING LOSS' HYPERPARAMETERS.
THE MODEL TRAINS FOR N NUMBER OF EPOCHS M NUMBER OF TRIALS. DURING CERTAIN TRIALS
THE HYPERPARAMETERS ARE CHOSEN RANDOMLY. THEN, THE TPE OPTIMIZATION ALGORITHM STARTS
FINDING THE BEST HYPERPARAMETERS THAT MINIMIZE THE VALIDATION LOSS FUNCTION
"""
import torch
import torch.nn as nn
import os
import argparse
from DatasetOptuna import get_datasets_and_loaders
from Reconstruction import prepare_model 
from EngineComplete import train, validate
from loss import total_loss  
from utils import save_model, SaveBestModel, save_plots
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc
import logging

### Seeds
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

### Hyperparameters (hyperparameters can change)
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--batch', default=8, type=int)
parser.add_argument('--imgsz', default=256, type=int)
args = parser.parse_args()

### For logging
logging.basicConfig(filename="TPE_optimization.txt", level=logging.INFO, format="%(message)s")
def logprint(message):
    print(message)
    logging.info(message)
    
### Dataset
images_folder = r".\CelebA-HQ-img" ### Path of the folder "testing_mask_dataset"
masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"


def reset_seed(seed=42):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def objective(trial):
    logprint(f"Starting trial {trial.number}")
    reset_seed(seed=trial.number + 100)
    
    ### Get datasets
    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder=images_folder,
        masks_folder=masks_folder,
        img_size=256,
        train_count=27000, ### 27000 images are for training
        valid_count=3000, ### 3000 images are for validating
        batch_size=8,
        train_subset_count=5400, ### For TPE optimization we select a sample of 5400 random images per epoch during training (27000 can be set, but it will take longer)
        valid_subset_count=1500 ### For TPE optimization we select a sample of 1500 random images per epoch during training (3000 can be set, but it will take longer)
    )
    
    ### Search space 
    a = trial.suggest_float('a', 1.0, 10.0, step = 0.1)
    b = trial.suggest_float('b', 1.0, 10.0, step = 0.1)
    c = trial.suggest_float('c', 1.0, 10.0, step = 0.1)
    d = trial.suggest_float('d', 1.0, 10.0, step = 0.1)
    
    logprint(f"[Trial {trial.number}] Hyperparameters: a={a}, b={b}, c={c}, d={d}")
    
    # Create a directory with the model name for outputs.
    out_dir = os.path.join(r'.\outputs') ### Path of the folder in which the model is saved
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    modelo, gan = prepare_model()
    modelo = modelo.to(device)
    gan = gan.to(device)
  

    # Model
    total_params = sum(p.numel() for p in modelo.parameters())
    print (total_params)
    total_trainable_params_modelo = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logprint(f"{total_trainable_params_modelo:,} training parameters.")
    
    # GAN
    total_params = sum(p.numel() for p in gan.parameters())
    print (total_params)
    total_trainable_params_GAN = sum(p.numel() for p in gan.parameters() if p.requires_grad)
    logprint(f"{total_trainable_params_GAN:,} training parameters.")
    
    ### Optimizer and losses
    b1 = 0.9
    b2 = 0.999
    optimizer = torch.optim.Adam(modelo.parameters(), lr=args.lr,betas=(b1,b2))
    optimizer_gan = torch.optim.Adam(gan.parameters(), lr=args.lr,betas=(b1,b2))
    criterion = total_loss()
    criterion_gan = nn.BCEWithLogitsLoss()

    save_best_model = SaveBestModel()

    EPOCHS = args.epochs
    train_loss = []
    valid_loss = []
    best_valid_loss = float("inf")
    
    def combined(outputs, targets):
        loss_values = criterion(outputs, targets)  # returns l1, l2, l3
        loss4 = criterion_gan(outputs, targets)    # l4
    
        # Clip each loss between 0 and 1 for stability
        loss_values = torch.clamp(torch.tensor(loss_values, device=outputs.device), min=0.0, max=1.0)
        loss4 = torch.clamp(loss4, min=0.0, max=1.0)
    
        # Normalize weights
        total_weight = a + b + c + d
        a_ = a / total_weight
        b_ = b / total_weight
        c_ = c / total_weight
        d_ = d / total_weight
    
        # Logging individual losses for debug
        logprint(f"   L1: {loss_values[0].item():.4f}, L2: {loss_values[1].item():.4f}, L3: {loss_values[2].item():.4f}, L4: {loss4.item():.4f}")
        logprint(f"   Weights: a={a_:.2f}, b={b_:.2f}, c={c_:.2f}, d={d_:.2f}")
    
        # Return the weighted sum
        return a_ * loss_values[0] + b_ * loss_values[1] + c_ * loss_values[2] + d_ * loss4
    
    ### Training and validation loop stage
    for epoch in range (EPOCHS):
        logprint(f"[Trial {trial.number}] Epoch: {epoch + 1}")
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
        
        ### If a trial is worse it ends before reaching the total number of epochs
        trial.report(valid_epoch_loss, step = epoch)
        if trial.should_prune():
            logprint(f"[Trial {trial.number}] Pruned at epoch {epoch + 1}")
            raise optuna.exceptions.TrialPruned()
           
        ### Find the best validation loss
        if valid_epoch_loss < best_valid_loss:
            best_valid_loss = valid_epoch_loss
            
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        
        save_best_model(
            valid_epoch_loss,mae_metric_valid, psnr_metric_valid,epoch, modelo, out_dir, name='TPE_opt'
        )

        for name, value in zip(
            ["Train Loss", "Valid Loss", "Train MAE", "Valid MAE", "Train PSNR", "Valid PSNR", "Train SSIM", "Valid SSIM", "Train FID", "Valid FID", "Train LPIPS", "Valid LPIPS"],
            [train_epoch_loss, valid_epoch_loss, mae_metric_train, mae_metric_valid, psnr_metric_train, psnr_metric_valid, ssim_train, ssim_valid, train_fid, valid_fid, train_lpips, valid_lpips]
        ):
            logprint(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")
        
    save_model(EPOCHS, modelo, optimizer, criterion, out_dir, name='TPE_opt')
    
    # Save the loss and accuracy plots.
    save_plots(
            train_loss, valid_loss, out_dir
    )
    logprint('TRAINING COMPLETE')
        
    del modelo, gan, optimizer, optimizer_gan
    torch.cuda.empty_cache()
    gc.collect()
        
    return best_valid_loss

import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    pruner = MedianPruner(n_warmup_steps=5) ### Control of number of steps to decide if a trial should be stopped.
    study = optuna.create_study(direction='minimize', sampler = TPESampler(n_startup_trials=40, multivariate=False), pruner = pruner) ### TPE is used. During first n_startup_trials is random, the rest it uses TPE optimization
    study.optimize(objective, n_trials=50) ### Number of trials. Each trial run for the number of epochs chosen.
    logging("Sampler", study.sampler)
    logging("Best trial:") 
    trial = study.best_trial
    logging("  Value: {}".format(trial.value))
    logging("  Params: ") ### The hyperparameters that minimized the validation loss.
    for key, value in trial.params.items():
        logging("    {}: {}".format(key, value)) 

    trials = study.trials
    
    # Extract trial numbers and objective values
    trial_numbers = [t.number for t in trials]
    objective_values = [t.value for t in trials]
    
    plt.figure()
    plt.plot(trial_numbers, objective_values, marker='o', linestyle='-')
    plt.xlabel('Trial Number')
    plt.ylabel('Objective Value')
    plt.title('Optimization History')
    plt.show()
    

    
    