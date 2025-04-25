# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 17:29:46 2025

@author: javie
"""

import os
import glob
import random
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

### Stratified Mask Sampling Functions
def sample_mask_stratified(mask_pool):
    """
    Randomly samples a mask from the mask_pool using stratified selection.
    The mask_pool is assumed to be organized as:
      - Bin i (i=0,...,5) corresponds to masks [i*2000, i*2000+2000).
      - Within each bin, sub-bin 0: [i*2000, i*2000+1000) and sub-bin 1: [i*2000+1000, i*2000+2000).
    """
    bin_num = random.randint(0, 5)
    sub_bin = random.randint(0, 1)
    start = bin_num * 2000 + sub_bin * 1000
    end = start + 1000
    return random.choice(mask_pool[start:end])

def precompute_fixed_masks(mask_pool, val_count=3000, fixed_seed=42):
    """
    Precomputes a fixed list of masks for validation.
    It divides the mask pool into 12 sub-bins (6 bins × 2 sub-bins) and randomly
    selects the same number (val_count/12) of masks from each sub-bin.
    For example, with 3000 validation images, each sub-bin contributes 250 masks.
    """
    num_bins = 6
    per_subbin = val_count // (num_bins * 2)  # should be 250 if val_count==3000
    fixed_masks = []
    random.seed(fixed_seed)
    for bin_num in range(num_bins):
        for sub_bin in range(2):
            start = bin_num * 2000 + sub_bin * 1000
            end = start + 1000
            sub_bin_masks = mask_pool[start:end]
            if len(sub_bin_masks) < per_subbin:
                raise ValueError("Not enough masks in sub-bin for validation.")
            selected = random.sample(sub_bin_masks, per_subbin)
            fixed_masks.extend(selected)
    random.shuffle(fixed_masks)
    return fixed_masks

### Splitting Images into Training and Validation Sets
def split_images(images_folder, train_count=27000, valid_count=3000, seed=42):
    """
    Splits all images in the given folder into two lists: training and validation
    """
    all_images = sorted(glob.glob(os.path.join(images_folder, "*")))
    random.seed(seed)
    random.shuffle(all_images)
    total_needed = train_count + valid_count
    if len(all_images) < total_needed:
        raise ValueError(f"Not enough images. Needed {total_needed}, found {len(all_images)}.")
    train_images = all_images[:train_count]
    valid_images = all_images[train_count:train_count + valid_count]
    return train_images, valid_images

def get_data_paths(images_folder, masks_folder, train_count=27000, valid_count=3000, seed=42):
    """
    Returns:
      - train_images: list of training image paths (split from images_folder)
      - valid_images: list of validation image paths (split from images_folder)
      - mask_pool: list of all mask paths (assumed ordered as described)
    """
    train_images, valid_images = split_images(images_folder, train_count, valid_count, seed)
    mask_pool = sorted(glob.glob(os.path.join(masks_folder, "*")))
    return train_images, valid_images, mask_pool

### Normalization and Augmentation Transforms
def normalize():
    """
    Transform to normalize image.
    """
    transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            always_apply=True
        )
    ])
    return transform

def train_transforms(img_size):
    """
    Transforms for training images and masks.
    """
    
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
        # The next Resize is redundant—if needed, remove it.
        # A.HorizontalFlip(p=0.1),
        # A.VerticalFlip(p=0.1),
        # A.Rotate(limit=360, p=0.01),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms for validation images and masks.
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ], is_check_shapes = False)
    return valid_image_transform

### Custom Dataset Class (Using Stratified Mask Sampling)
class CustomDataSet(Dataset):
    def __init__(self, image_paths, mask_pool, tfms, norm_tfms,
                 random_mask, fixed_masks=None):
        """
        :param image_paths: List of image file paths (training or validation).
        :param mask_pool: List of mask file paths (from one folder).
        :param tfms: Albumentations transforms (augmentations).
        :param norm_tfms: Albumentations normalization transforms.
        :param random_mask: Boolean; if True, a random mask is chosen for each image.
        :param fixed_masks: Optional list of fixed mask paths (used for validation).
        """
        self.image_paths = image_paths
        self.mask_pool = mask_pool
        self.tfms = tfms
        self.norm_tfms = norm_tfms
        self.random_mask = random_mask
        self.fixed_masks = fixed_masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load image
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        image = self.norm_tfms(image=image)['image']

        # Select mask:
        if self.fixed_masks is not None:
            # Use fixed masks for validation (each image gets its predetermined mask)
            mask_path = self.fixed_masks[index]
        elif self.random_mask:
            # Use stratified sampling for training masks
            mask_path = sample_mask_stratified(self.mask_pool)
        else:
            # Otherwise, use mask with the same index as image (if aligned)
            mask_path = self.mask_pool[index]

        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask = self.norm_tfms(image=mask)['image']

        # Apply augmentations (both image and mask)
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        # Change from (H, W, C) to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)

        return image, mask

### Functions to Create Datasets and DataLoaders
def get_datasets_and_loaders(images_folder, masks_folder, img_size,
                            train_count=27000, valid_count=3000,
                            batch_size=8, seed=42, num_workers=4):
    """
    Creates training and validation datasets and dataloaders using a single images folder
    and a mask pool.
    For training, masks are selected randomly via stratified sampling.
    For validation, a fixed set of masks is precomputed.
    """
    # 
    train_images, valid_images, mask_pool = get_data_paths(
        images_folder, masks_folder, train_count, valid_count, seed
    )

    #
    fixed_masks = precompute_fixed_masks(mask_pool, val_count=valid_count, fixed_seed=seed)

    ### Transformations.
    norm_tfms = normalize()
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    ### Dataset creation.
    train_dataset = CustomDataSet(
        image_paths=train_images,
        mask_pool=mask_pool,
        tfms=train_tfms,
        norm_tfms=norm_tfms,
        random_mask=True,
        fixed_masks=None
    )
    valid_dataset = CustomDataSet(
        image_paths=valid_images,
        mask_pool=mask_pool,
        tfms=valid_tfms,
        norm_tfms=norm_tfms,
        random_mask=False,
        fixed_masks=fixed_masks
    )

    #### Dataloaders creation.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,  persistent_workers=True, pin_memory=True)

    return train_dataset, valid_dataset, train_loader, valid_loader

def display_image_and_mask(images, masks):
    ### First image of the batch
    image = images[0]
    mask = masks[0]
    img = image*(1-mask)+mask
    image_pil = TF.to_pil_image(image)
    mask_pil = TF.to_pil_image(mask)
    img = TF.to_pil_image(img)
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image_pil)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(img)
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    ### Dataset path
    images_folder = r".\CelebA-HQ-img" ### Path of the folder "CelebA-HQ-img"
    masks_folder = r".\testing_mask_dataset" ### Path of the folder "testing_mask_dataset"
    img_size = 256

    ### Create dataloader
    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder, masks_folder, img_size,
        train_count=27000, valid_count=3000,
        batch_size=8, seed=42, num_workers=2
    )

    ### Visualization
    train_batch = next(iter(train_loader))
    images, masks = train_batch
    print("Images batch shape:", images.shape)
    print("Masks batch shape:", masks.shape)
    display_image_and_mask(images, masks)