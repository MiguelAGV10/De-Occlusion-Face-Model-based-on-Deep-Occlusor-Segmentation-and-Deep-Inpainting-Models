

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

### Stratified Mask Sampling Functions: A random mask is chosen
def sample_mask_stratified(mask_pool):
    bin_num = random.randint(0, 5)
    sub_bin = random.randint(0, 1)
    start = bin_num * 2000 + sub_bin * 1000
    end = start + 1000
    return random.choice(mask_pool[start:end])

### Splitting Images into Training and Validation Sets
def split_images(images_folder, train_count=7000, valid_count=1000, seed=42):
    all_images = sorted(glob.glob(os.path.join(images_folder, "*")))
    random.seed(seed)
    random.shuffle(all_images)
    total_needed = train_count + valid_count
    if len(all_images) < total_needed:
        raise ValueError(f"Not enough images. Needed {total_needed}, found {len(all_images)}.")
    train_images = all_images[:train_count]
    valid_images = all_images[train_count:train_count + valid_count]
    return train_images, valid_images

def get_data_paths(images_folder, masks_folder, train_count=7000, valid_count=1000, seed=42):
    train_images, valid_images = split_images(images_folder, train_count, valid_count, seed)
    mask_pool = sorted(glob.glob(os.path.join(masks_folder, "*")))
    return train_images, valid_images, mask_pool

### Normalization and Augmentation Transforms
def normalize():
    transform = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            always_apply=True
        )
    ])
    return transform

def train_transforms(img_size):
    train_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ], is_check_shapes=False)
    return train_image_transform

def valid_transforms(img_size):
    valid_image_transform = A.Compose([
        A.Resize(img_size, img_size, always_apply=True),
    ], is_check_shapes=False)
    return valid_image_transform

### Custom Dataset Class (Using Stratified Mask Sampling)
class CustomDataSet(Dataset):
    def __init__(self, image_paths, mask_pool, tfms, norm_tfms,
                 random_mask, fixed_masks=None):
        self.image_paths = image_paths
        self.mask_pool = mask_pool
        self.tfms = tfms
        self.norm_tfms = norm_tfms
        self.random_mask = random_mask
        self.fixed_masks = fixed_masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.array(Image.open(self.image_paths[index]).convert('RGB'))
        image = self.norm_tfms(image=image)['image']

        if self.fixed_masks is not None:
            mask_path = self.fixed_masks[index % len(self.fixed_masks)]
        elif self.random_mask:
            mask_path = sample_mask_stratified(self.mask_pool)
        else:
            mask_path = self.mask_pool[index]

        mask = np.array(Image.open(mask_path).convert('RGB'))
        mask = self.norm_tfms(image=mask)['image']

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float)
        mask = torch.tensor(mask, dtype=torch.float)

        return image, mask

### Functions to Create Datasets and DataLoaders
def get_datasets_and_loaders(images_folder, masks_folder, img_size,
                            train_count=7000, valid_count=1000,
                            batch_size=8, seed=42, num_workers=4):
    train_images, valid_images, mask_pool = get_data_paths(
        images_folder, masks_folder, train_count, valid_count, seed
    )

    random.seed(seed)
    if len(mask_pool) >= valid_count:
        fixed_masks = random.sample(mask_pool, valid_count)
    else:
        fixed_masks = [random.choice(mask_pool) for _ in range(valid_count)]

    norm_tfms = normalize()
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, persistent_workers=True, pin_memory=True,
                              drop_last=True)

    return train_dataset, valid_dataset, train_loader, valid_loader

### Visualization
def display_batch(images, masks, N=4):
    N = min(N, len(images))
    fig, axs = plt.subplots(N, 2, figsize=(8, 4 * N))

    for i in range(N):
        image = images[i]
        mask = masks[i]
        masked_img = image * (1 - mask) + mask

        image_pil = TF.to_pil_image(image)
        masked_pil = TF.to_pil_image(masked_img)

        axs[i, 0].imshow(image_pil)
        axs[i, 0].set_title(f"Image {i}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(masked_pil)
        axs[i, 1].set_title(f"Masked {i}")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    ### Dataset path
    images_folder = r"D:\Miguel\Revisionarticulo\Inpaintinggeneral\test_images"
    masks_folder = r"E:\celeba\test_mask\mask\testing_mask_dataset"
    img_size = 256

    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder, masks_folder, img_size,
        train_count=7000, valid_count=1000,
        batch_size=8, seed=42, num_workers=2
    )

    train_batch = next(iter(train_loader))
    images, masks = train_batch
    print("Images batch shape:", images.shape)
    print("Masks batch shape:", masks.shape)
    display_batch(images, masks)
