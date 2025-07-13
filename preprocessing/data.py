import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import shutil
import random

from preprocessing.masked_dataset import ImageMaskDataset


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def train_test_split(data_path, split_ratio = 0.8):
    images_root =os.path.join(data_path,'all_images')
    train_path = os.path.join(data_path,'train')
    test_path = os.path.join(data_path,'test')

    clean_dir(train_path)
    clean_dir(test_path)

    dirs = os.listdir(images_root)


    for dir in dirs:
        class_dir = os.path.join(images_root,dir)

        images = os.listdir(os.path.join(class_dir,'images'))
        masks = os.listdir(os.path.join(class_dir,'masks'))
        list_len = len(images)
        indices = list(range(list_len))
        split_point = int(list_len * split_ratio)

        random.shuffle(indices)

        images_shuffled = [images[i] for i in indices]
        masks_shuffled = [masks[i] for i in indices]

        train_images, train_masks = images_shuffled[:split_point], masks_shuffled[:split_point]
        test_images, test_masks = images_shuffled[split_point:], masks_shuffled[split_point:]

        train_class_dir_img = os.path.join(train_path,'images',dir)
        train_class_dir_mask = os.path.join(train_path,'masks',dir)
        test_class_dir_img = os.path.join(test_path,'images',dir)
        test_class_dir_mask = os.path.join(test_path,'masks',dir)

        os.makedirs(train_class_dir_img,exist_ok=True)
        os.makedirs(train_class_dir_mask,exist_ok=True)
        os.makedirs(test_class_dir_img,exist_ok=True)
        os.makedirs(test_class_dir_mask,exist_ok=True)

        for img, mask in zip(train_images, train_masks):
            shutil.copy(os.path.join(class_dir, 'images', img), os.path.join(train_class_dir_img, img))
            shutil.copy(os.path.join(class_dir, 'masks', mask), os.path.join(train_class_dir_mask, mask))

        for img, mask in zip(test_images, test_masks):
            shutil.copy(os.path.join(class_dir, 'images', img), os.path.join(test_class_dir_img, img))
            shutil.copy(os.path.join(class_dir, 'masks', mask), os.path.join(test_class_dir_mask, mask))

    print("Data successfully split into train and test sets.")


#train_test_split('data')

def check_split(data_path):
    train_path = os.path.join(data_path, 'train', 'images')
    test_path = os.path.join(data_path, 'test', 'images')

    dirs = os.listdir(train_path)
    for dir in dirs:
        train_class_dir = os.path.join(train_path,dir)
        test_class_dir = os.path.join(test_path,dir)

        train_imgages = os.listdir(train_class_dir)
        test_imgages = os.listdir(test_class_dir)

        common_images = list(set(train_imgages) & set(test_imgages))
        print(dir)
        print(f"{len(common_images)} images common to train and test sets.")
        print(common_images)
        print('*'*10)

def create_datasets(data_path, mean, std):

    train_transforms = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
    ])

    test_transforms = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
    ])
    train_dataset = ImageMaskDataset(os.path.join(data_path, 'train'), transform=train_transforms)
    test_dataset = ImageMaskDataset(os.path.join(data_path, 'test'), transform = test_transforms)

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_loaders(data_path, mean, std, batch_size):

    train_dataset, test_dataset = create_datasets(data_path, mean, std)
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size)

    return train_loader, test_loader

def calculate_mean_std(data_path, batch_size):
    train_transforms = torchvision.transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    train_dataset = ImageMaskDataset(os.path.join(data_path, 'train'), transform=train_transforms)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mean = 0.
    std = 0.
    total_images = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean, std



#train_dataset, test_dataset = create_datasets('data/train', 'data/test', mean=[0.485], std=[0.229])
#train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, batch_size=32)
