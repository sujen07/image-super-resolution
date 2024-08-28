from torch.utils.data import DataLoader
from torchvision import transforms
from models import ImageDataset
import os

def load_data(downscaling_factor, batch_size, hr_crop_size, train_dir, val_dir):
    train_hr_dir = os.path.join(train_dir, 'hr')
    train_lr_dir = os.path.join(train_dir, 'lr')
    val_hr_dir = os.path.join(val_dir, 'hr')
    val_lr_dir = os.path.join(val_dir, 'lr') 


    # Transforms for HR images
    hr_transform = transforms.Compose([
        transforms.CenterCrop(hr_crop_size),
        transforms.ToTensor(),
    ])

    # Adjust the LR crop size according to the downscaling factor
    lr_crop_size = hr_crop_size // downscaling_factor

    # Transforms for LR images
    lr_transform = transforms.Compose([
        transforms.CenterCrop(lr_crop_size),
        transforms.ToTensor(),
    ])


    train_dataset = ImageDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, hr_transform=hr_transform, lr_transform=lr_transform)
    val_dataset = ImageDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, hr_transform=hr_transform, lr_transform=lr_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader