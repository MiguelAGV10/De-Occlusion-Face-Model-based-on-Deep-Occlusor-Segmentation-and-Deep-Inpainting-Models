
import torch
import os
import matplotlib.pyplot as plt

class SaveBestModel:
    """
    Save the best model while training. It compares the current validation loss to the best one
    """
    def __init__(self, best_valid_loss=float('inf'), best_valid_mae=float('inf'), best_valid_psnr=float('-inf'), ):
        self.best_valid_loss = best_valid_loss
        self.best_valid_mae = best_valid_mae
        self.best_valid_psnr = best_valid_psnr
        
    def __call__(
        self, current_valid_loss, current_valid_mae, current_valid_psnr, epoch, model, out_dir, name='model'
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_loss_'+name+'.pth'))
            
        if current_valid_mae < self.best_valid_mae:
            self.best_valid_mae = current_valid_mae
            print(f"\nBest validation mae: {self.best_valid_mae}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_mae_'+name+'.pth'))
            
        if current_valid_psnr>self.best_valid_psnr:
            self.best_valid_psnr = current_valid_psnr
            print(f"\nBest validation psnr: {self.best_valid_psnr}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                }, os.path.join(out_dir, 'best_psnr_'+name+'.pth'))
        

def save_model(epochs, model, optimizer, criterion, out_dir, name='model'):
    """
    Save the model.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, os.path.join(out_dir, name +'.pth'))

def save_plots(
    train_loss, valid_loss,
    out_dir
):
    """
    Save the losses plot.
    """
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))

