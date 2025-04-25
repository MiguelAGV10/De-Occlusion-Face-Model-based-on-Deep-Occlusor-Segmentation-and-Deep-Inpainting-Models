
import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips  

### To calculate the FID and LPIP metrics
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### FID and LPIP metrics definitions
fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
loss_fn = lpips.LPIPS(net='alex').to(DEVICE)
loss_fn.eval()

def compare_fid(real, fake):
    """
    Function to calculate the FID metric using FrechetInceptionDistance using default net.
    """

    real8 = (real*255).to(torch.uint8)
    fake8 = (fake*255).to(torch.uint8)

    fid_metric.update(real8, real=True)
    fid_metric.update(fake8, real=False)
    
    fid = fid_metric.compute()
    return fid

def compare_lpips(real, fake):
    """
    Function to calculate the LPIPS metric using AlexNet.
    """
    # print("min:", real.min().item(), "max:", real.max().item())
    # print("min:", fake.min().item(), "max:", fake.max().item())
    ### To be in range
    real = 2*real-1 
    fake = 2*real-1
    lpips = loss_fn(real, fake)
    # print('LPIPS', lpips)
    # print('LPIPS', lpips.shape)
    return lpips.mean()

def mae_val(real,fake):
    """
    Function to calculate the L1 metric.
    """
    real, fake = real.cpu(), fake.cpu()
    real, fake = real.numpy(), fake.numpy()
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return np.sum(np.abs(real-fake))/np.sum(real+fake)

def psnr_val(real, fake):
    """
    Function to calculate the PSNR metric using skimage package.
    """
    real, fake = real.cpu(), fake.cpu()
    real, fake = real.numpy(), fake.numpy()
    real, fake = real.astype(np.float32), fake.astype(np.float32)
    return peak_signal_noise_ratio(real, fake)

def ssim_val(real, fake):
    """
    Function to calculate the SSIM metric using skimage package.
    """
    real, fake = real.cpu(), fake.cpu()
    real, fake = real.numpy(), fake.numpy()
    real, fake = np.squeeze(real), np.squeeze(fake)
    
    real_r, fake_r = real[:, :, 0], fake[:,:,0]
    real_g, fake_g = real[:,:,1], fake[:,:,1]
    real_b, fake_b = real[:,:,2], fake[:,:,2]
    
    ssim_r = structural_similarity(real_r, fake_r,win_size =3)
    ssim_g =  structural_similarity(real_g, fake_g,win_size =3) 
    ssim_b = structural_similarity(real_b, fake_b,win_size =3)
    
    output = (ssim_r+ssim_g+ssim_b)/3
    return output


