from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

## From i-ResNet paper (https://arxiv.org/pdf/1608.04101.pdf)
train_chain = [
    transforms.Pad(4, padding_mode="symmetric"),
    transforms.RandomCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
test_chain = [transforms.ToTensor()]
dens_est_chain = [
    lambda x: (255.0 * x) + torch.zeros_like(x).uniform_(0.0, 1.0),
    lambda x: x / 256.0,
    lambda x: x - 0.5,
]


def get_cifar10_data(
    batch_size,
    train_transform = train_chain,
    test_transform = test_chain,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    train = CIFAR10(root="data", train=True, download=True, transform=train_transform)
    test = CIFAR10(root="data", train=False, download=True, transform=test_transform)

    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainloader, testloader


def get_cifar100_data(
    batch_size,
    train_transform,
    test_transform,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
):
    
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    train = CIFAR100(root="data", train=True, download=True, transform=train_transform)
    test = CIFAR100(root="data", train=False, download=True, transform=test_transform)

    trainloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    testloader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return trainloader, testloader
