import numpy as np
import torch.utils.data
import os
from torchvision import transforms

from PIL import Image


class ImageMaskDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        self.image_path = os.path.join(path, 'images')
        self.mask_path = os.path.join(path, 'masks')
        self.transform = transform

        self.samples = []

        classes = os.listdir(self.mask_path)

        for i, cls in enumerate(classes):
            img_folder = os.path.join(self.image_path, cls)
            mask_folder = os.path.join(self.mask_path, cls)

            img_files = os.listdir(img_folder)
            mask_files = os.listdir(mask_folder)

            for img_name, mask_name in zip(img_files, mask_files):
                img_path = os.path.join(img_folder, img_name)
                mask_path = os.path.join(mask_folder, mask_name)
                self.samples.append((img_path, mask_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, cls = self.samples[idx]

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        img = transform(img)
        mask = transform(mask)

        masked_img = img * mask


        if self.transform:
            masked_img = self.transform(masked_img)

        return masked_img, cls