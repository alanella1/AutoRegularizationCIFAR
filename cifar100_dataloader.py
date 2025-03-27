import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# CIFAR-100 mean and standard deviation (computed from dataset)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomCrop(32, padding=4),  # Data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)  # Normalize
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)  # Normalize (no augmentation)
])


# Load CIFAR-100 dataset
def get_cifar100_dataloader(batch_size=128, num_workers=4):
    train_dataset = datasets.CIFAR100(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR100(root="./data", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
