import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class ResiudualBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(growth_rate, in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        
        
class Generator(nn.Module):
    def __init__(self, self, in_channels=3, num_rrdb_blocks=23):
        super(Generator, self).__init__()
        self.rrdb_blocks = self._make_layers(num_rrdb_blocks)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upsample by 2x
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2)
        )
        self.conv = nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1)

        
    def _make_layers(self, num_rrdb_blocks):
        layers = []
        for _ in range(num_rrdb_blocks):
            layers.append(RRDB())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.rrdb_blocks(out)
        out = self.upsample(out)
        out = self.conv(out)
        return out
    
    
    
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 128 * 128, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.model(x)

    
class VGGFeatures(torch.nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features

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



        
        