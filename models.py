import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset
import os
from PIL import Image




class ResidualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out
        
        
class Generator(nn.Module):
    def __init__(self, in_channels=3, num_rrdb_blocks=23):
        super(Generator, self).__init__()
        self.rrdb_blocks = self._make_layers(num_rrdb_blocks)
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upsample by 2x
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Conv2d(256, in_channels, kernel_size=3, stride=1, padding=1)

        
    def _make_layers(self, num_rrdb_blocks):
        layers = []
        for _ in range(num_rrdb_blocks):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.rrdb_blocks(x)
        out = self.upsample(out)
        out = self.conv(out)
        return out
    
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Add more layers as needed
        )
        # Placeholder for the fully connected layers; to be initialized later
        self.fc = None

    def forward(self, x):
        x = self.convs(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)) # Add this line for global average pooling
        x = torch.flatten(x, 1) # Adjust this line accordingly
        if self.fc is None:
            n_size = x.size(1)
            self.fc = nn.Sequential(
                nn.Linear(n_size, 1024),
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1)
            ).to(x.device)
        return self.fc(x)
    
class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

    
    
class PerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatures()
        self.criterion = torch.nn.MSELoss()

    def forward(self, x, y):
        
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = sum(self.criterion(x_feat, y_feat) for x_feat, y_feat in zip(x_vgg, y_vgg))
        return loss


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, hr_transform=None, lr_transform=None):
        """
        Args:
            hr_dir (string): Directory with high-resolution images.
            lr_dir (string): Directory with low-resolution images.
            hr_transform (callable, optional): Transform to be applied on HR images.
            lr_transform (callable, optional): Transform to be applied on LR images.
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform
        self.image_files = [f for f in os.listdir(hr_dir) if os.path.isfile(os.path.join(hr_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_img_name = os.path.join(self.hr_dir, self.image_files[idx])
        lr_img_name = os.path.join(self.lr_dir, self.image_files[idx])

        hr_image = Image.open(hr_img_name).convert('RGB')
        lr_image = Image.open(lr_img_name).convert('RGB')

        if self.hr_transform:
            hr_image = self.hr_transform(hr_image)
        if self.lr_transform:
            lr_image = self.lr_transform(lr_image)

        return lr_image, hr_image

        
        