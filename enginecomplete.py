
import torch
from tqdm import tqdm
from metrics import compare_mae,compare_psnr, mae_val, psnr_val, ssim_val, compare_fid, compare_lpips
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_images(inpainted_image, data, target, title_prefix):
    """
    It displays the target image, masked image, and inpainted image
    """
    if len(inpainted_image.shape) == 4:
        inpainted_image = inpainted_image.squeeze(0)
    if len(data.shape) == 4:
        data = data.squeeze(0)
    if len(target.shape) == 4:
        target = target.squeeze(0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax in axs:
        ax.axis('off')
    axs[2].imshow(TF.to_pil_image(inpainted_image))
    axs[2].set_title(f"{title_prefix} Inpainted Image")
    
    axs[0].imshow(TF.to_pil_image(data))
    axs[0].set_title(f"{title_prefix} Data")
    
    axs[1].imshow(TF.to_pil_image(target))
    axs[1].set_title(f"{title_prefix} Target")
    
    plt.show()

def train(model, gan, train_dataset, train_dataloader, device, optimizer, optimizer_gan, criterion, criterion_gan, a, b, c, d):
    """
    Function to train the model. It computes the corresponding masking stage, then the masked image is the input of the model.
    The corresponding loss calculation, propagation and weigths updating is applied.
    Also, the metrics are calculated.
    
    Inputs:
        model = Inpainting model
        gan = discriminator
        train_datasatet = images that the model will use to be train with
        train_dataloader = it creates the defined loader 
        device = CUDA if available
        optimizer = optimizer for the model optimization
        optimizer_gan = optimizer for the gan optimization
        criterion = to calculate L1, perceptual, and style loss 
        criterion_gan = to calculate adversarial loss
        a, b, c, d = hyperparameters of the total loss function
    """
    model.train()
    gan.train()
    train_running_loss = 0.0
    mae_metric = 0
    psnr_metric = 0
    ssim_metric = 0
    fidmetric = 0
    lpipsmetric = 0
    num_batches = int(len(train_dataset)/train_dataloader.batch_size)
    prog_bar = tqdm(train_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    counter = 0
    display_interval = 500
    eval_frequency = 5
    
    for i, data in enumerate(prog_bar):
        counter += 1
        optimizer.zero_grad()
        optimizer_gan.zero_grad()
        mask, target = data[1].to(device), data[0].to(device)
        data = target*(1-mask) + mask
        
            
        inpainted_image = model(data)
        # inpainted_image_mask = inpainted_image*mask
    
        
        real_labels = torch.ones_like(gan(target))
        fake_labels = torch.zeros_like(gan(inpainted_image.detach()))
            
        adversarial_loss = criterion_gan(gan(inpainted_image),real_labels)
            
        perceptual,style, content = criterion(inpainted_image,target, mask)
            # reconstruction_loss_global = perceptual  + style + content
        total_loss = a*perceptual + b*style + c*content + d*adversarial_loss
        
        train_running_loss += total_loss.item()
        total_loss.backward()
        optimizer.step()

        real_loss = criterion_gan(gan(target),real_labels)
        fake_loss = criterion_gan(gan(inpainted_image.detach()), fake_labels)
        discriminator_loss  = 0.5*(real_loss + fake_loss)
            
        discriminator_loss.backward()
        optimizer_gan.step()

        target_numpy, outputs_numpy = target.detach(), inpainted_image.detach()
        mae_metric += compare_mae(outputs_numpy,target_numpy)
        psnr_metric += compare_psnr(outputs_numpy,target_numpy)
        ssim_metric += ssim_val(outputs_numpy, target)
        
        ### It calculates FID and LPIPS metrics (eval_frequency can be set to 1, meaning it calculates these metrics every batch)
        if counter % eval_frequency == 0:
            fidmetric += compare_fid(inpainted_image, target)
            lpipsmetric += compare_lpips(inpainted_image, target)
 
        ### Comment if run from console. Meant to keep track of the reconstruction during training. It displays results in the GUI. 
        if counter % display_interval == 0:
            show_images(inpainted_image[0], data[0], target[0], f"Training (Batch {counter})")
        
    train_loss = train_running_loss / counter
    train_mae = mae_metric/counter
    train_psnr = psnr_metric/counter
    train_ssim = ssim_metric/counter
    train_fid = fidmetric/counter
    train_lpips = lpipsmetric/counter
    # train_ssim = 0
    return train_loss, train_mae, train_psnr, train_ssim, train_fid, train_lpips

def validate(model, gan, valid_dataset, valid_dataloader, device, criterion, criterion_gan, epoch, a, b, c, d):
    
    """
    Function to evaluate the model. It computes the corresponding masking stage, then the masked image is the input of the model.
    The metrics and losses are calculated.
    
    Inputs:
        model = Inpainting model
        gan = discriminator
        train_datasatet = images that the model will use to be train with
        train_dataloader = it creates the defined loader 
        device = CUDA if available
        criterion = to calculate L1, perceptual, and style loss 
        criterion_gan = to calculate adversarial loss
        a, b, c, d = hyperparameters of the total loss function
    """
    
    model.eval()
    valid_running_loss = 0.0
    mae_metric = 0
    psnr_metric = 0
    ssim_metric = 0
    fidmetric = 0
    lpipsmetric = 0
    display_interval = 20
    eval_frequency = 5
    # ssim_metric = 0
    num_batches = int(len(valid_dataset)/valid_dataloader.batch_size)
    
    with torch.no_grad():
        prog_bar = tqdm(valid_dataloader, total=num_batches, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            mask, target = data[1].to(device), data[0].to(device)
            data = target*(1-mask) + mask ### Image with missing pixels
            out_M = model(data) ### Model's prediction
            
            perceptual, style, content = criterion(out_M, target,mask)
            
            ### GAN
            real_labels = torch.ones_like(gan(target))     
            adversarial_loss = criterion_gan(gan(out_M),real_labels)
            
            perceptual,style, content = criterion(out_M,target, mask)
            custom_loss = a*perceptual + b*style + c*content + d*adversarial_loss
            valid_running_loss += custom_loss.item()
            
            mae_metric += mae_val(out_M,target)
            psnr_metric += psnr_val(out_M,target)
            ssim_metric += ssim_val(out_M,target)
            
            ### It calculates FID and LPIPS metrics (eval_frequency can be set to 1, meaning it calculates these metrics every batch)
            if i % eval_frequency == 0:
                fidmetric += compare_fid(out_M, target)
                lpipsmetric += compare_lpips(out_M, target)
            ### Comment if run from console. Meant to keep track of the reconstruction during training. It displays results in the GUI. 
            if counter % display_interval == 0:
                show_images(out_M[0], data[0], target[0], f"Valid (Batch {counter})")

        valid_mae = mae_metric/counter
        valid_psnr = psnr_metric/counter
        valid_ssim = ssim_metric/counter
        valid_fid = fidmetric/counter
        valid_lpips = lpipsmetric/counter
        valid_loss = valid_running_loss / counter
        return valid_loss, valid_mae, valid_psnr, valid_ssim, valid_fid, valid_lpips
