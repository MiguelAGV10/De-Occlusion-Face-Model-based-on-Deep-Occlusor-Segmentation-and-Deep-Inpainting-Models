
import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

### Split images and get occluders
def split_images(images_folder, train_count=27000, valid_count=3000, seed=42):
    all_images = sorted(glob.glob(os.path.join(images_folder, "*")))
    random.seed(seed)
    random.shuffle(all_images)
    total_needed = train_count + valid_count
    if len(all_images) < total_needed:
        raise ValueError(f"Not enough images. Needed {total_needed}, found {len(all_images)}.")
    train_images = all_images[:train_count]
    valid_images = all_images[train_count:train_count + valid_count]
    return train_images, valid_images

def get_data_paths(images_folder, occluder_folder, train_count=27000, valid_count=3000, seed=42, single_occluder_folder=True):
    train_images, valid_images = split_images(images_folder, train_count, valid_count, seed)

    if single_occluder_folder:
        all_occluders = sorted(glob.glob(os.path.join(occluder_folder, "*.png")))
        train_occluders = all_occluders
        valid_occluders = all_occluders
    else:
        train_occluders = sorted(glob.glob(os.path.join(occluder_folder, "train", "*.png")))
        valid_occluders = sorted(glob.glob(os.path.join(occluder_folder, "val", "*.png")))

    if len(train_occluders) == 0 or len(valid_occluders) == 0:
        raise ValueError("Occluder folder is empty or incorrectly set.")

    return train_images, valid_images, train_occluders, valid_occluders

### Albumentations Transforms
def normalize():
    return A.Compose([
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], always_apply=True),
    ])

def train_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.RandomShadow(p=0.3),
    ])

def valid_transforms(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
    ])

### Dataset with Clean Alpha Pasting + Random Occlusion

class OcclusionAugmentDataset(Dataset):
    def __init__(self, image_paths, occluder_paths, tfms_occluded, tfms_original, norm_tfms):
        self.image_paths = image_paths
        self.occluder_paths = occluder_paths
        self.tfms_occluded = tfms_occluded
        self.tfms_original = tfms_original
        self.norm_tfms = norm_tfms

    def __len__(self):
        return len(self.image_paths)

    def paste_occluder(self, face_img, occ_img):
        face_w, face_h = face_img.size
        occ_img = occ_img.convert("RGBA")
        occ = np.array(occ_img)

        # Create alpha mask where occluder is not black
        alpha = np.any(occ[:, :, :3] != 0, axis=-1).astype(np.uint8) * 255
        occ[:, :, 3] = alpha
        occ_img = Image.fromarray(occ, mode="RGBA")

        # Resize occluder if too large
        occ_w, occ_h = occ_img.size
        if occ_w >= face_w or occ_h >= face_h:
            occ_img = occ_img.resize((face_w // 3, face_h // 3))
            occ_w, occ_h = occ_img.size

        x = random.randint(0, face_w - occ_w)
        y = random.randint(0, face_h - occ_h)

        face_img = face_img.convert("RGBA")
        face_img.paste(occ_img, (x, y), mask=occ_img)
        return face_img.convert("RGB")

    def draw_random_shapes(self, img):
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for _ in range(random.randint(1, 3)):
            shape_type = random.choice(['rectangle', 'ellipse'])
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = x1 + random.randint(20, w//2), y1 + random.randint(20, h//2)
            color = tuple(random.randint(0, 255) for _ in range(3))
            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], fill=color)
            else:
                draw.ellipse([x1, y1, x2, y2], fill=color)
        return img

    def __getitem__(self, index):
        original = Image.open(self.image_paths[index]).convert('RGB')

        if not self.occluder_paths:
            raise ValueError("No occluders found. Check the occluder folder path.")

        occ_path = random.choice(self.occluder_paths)
        occ_img = Image.open(occ_path).convert('RGBA')
        occluded = self.paste_occluder(original.copy(), occ_img)

        if random.random() < 0.5:
            occluded = self.draw_random_shapes(occluded)

        occluded_np = np.array(occluded)
        original_np = np.array(original)

        occluded_np = self.tfms_occluded(image=occluded_np)['image']
        original_np = self.tfms_original(image=original_np)['image']

        occluded_np = self.norm_tfms(image=occluded_np)['image']
        original_np = self.norm_tfms(image=original_np)['image']

        occluded_tensor = torch.tensor(occluded_np.transpose(2, 0, 1), dtype=torch.float)
        original_tensor = torch.tensor(original_np.transpose(2, 0, 1), dtype=torch.float)

        return occluded_tensor, original_tensor

### Create Datasets and Dataloaders

def get_datasets_and_loaders(images_folder, occluder_folder, img_size,
                             train_count=27000, valid_count=3000,
                             batch_size=8, seed=42, num_workers=4,
                             single_occluder_folder=True):
    train_images, valid_images, train_occluders, valid_occluders = get_data_paths(
        images_folder, occluder_folder, train_count, valid_count, seed,
        single_occluder_folder=single_occluder_folder
    )

    norm_tfms = normalize()
    tfms_augmented = train_transforms(img_size)
    tfms_simple = valid_transforms(img_size)

    train_dataset = OcclusionAugmentDataset(train_images, train_occluders, tfms_augmented, tfms_simple, norm_tfms)
    valid_dataset = OcclusionAugmentDataset(valid_images, valid_occluders, tfms_simple, tfms_simple, norm_tfms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

    return train_dataset, valid_dataset, train_loader, valid_loader

### Visualization

def display_occluded_images(batch, num_to_show=4):
    occluded_batch, original_batch = batch
    plt.figure(figsize=(num_to_show * 3, 5))

    for i in range(min(num_to_show, len(occluded_batch))):
        occluded = TF.to_pil_image(occluded_batch[i])
        original = TF.to_pil_image(original_batch[i])

        plt.subplot(2, num_to_show, i + 1)
        plt.imshow(occluded)
        plt.axis("off")
        plt.title(f"Occluded {i+1}")

        plt.subplot(2, num_to_show, i + 1 + num_to_show)
        plt.imshow(original)
        plt.axis("off")
        plt.title(f"Original {i+1}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    images_folder = r"E:\celeba\CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img"
    occluder_folder = r"D:\Miguel\Revisionarticulo\RealOcc\occ\train"

    train_dataset, valid_dataset, train_loader, valid_loader = get_datasets_and_loaders(
        images_folder, occluder_folder, img_size=256,
        train_count=27000, valid_count=3000,
        batch_size=8, seed=42, num_workers=2,
        single_occluder_folder=True
    )

    batch = next(iter(train_loader))
    display_occluded_images(batch, num_to_show=6)