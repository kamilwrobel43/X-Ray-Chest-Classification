import torch
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms
from preprocessing.masked_dataset import ImageMaskDataset
from torch import nn


def get_weighted_loss(data_path='data'):
    dirs = os.listdir(os.path.join(data_path, 'train', 'images'))
    class_weights = []

    for dir in dirs:
        files = os.listdir(os.path.join(data_path, 'train', 'images', dir))
        class_weights.append(1 / len(files))

    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    loss = nn.CrossEntropyLoss(weight=class_weights)
    return loss


def get_loaders_with_oversampling(root_dir = 'data', batch_size = 32, mean = [0.5], std = [0.5]):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
        ]
    )

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=torch.Tensor(mean), std=torch.Tensor(std)),
    ])

    train_dataset = ImageMaskDataset(os.path.join(root_dir, 'train'), transform=train_transforms)
    test_dataset = ImageMaskDataset(os.path.join(root_dir, 'test'), transform=test_transforms)

    dirs = os.listdir(os.path.join(root_dir, 'train', 'images'))
    class_weights = []

    for dir in dirs:
        files = os.listdir(os.path.join(root_dir, 'train', 'images', dir))
        class_weights.append(1 / len(files))

    sample_weights = [0] * len(train_dataset)

    for idx, (data, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

