import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import pad
from models import Generator
from torch.optim.lr_scheduler import StepLR

from models import ImageDataset
from models import PerceptualLoss
from models import Discriminator
import matplotlib.pyplot as plt
from PIL import Image
import time

device = torch.device("cuda")

train_dir = 'data/train'
val_dir = 'data/validation'

train_hr_dir = os.path.join(train_dir, 'hr')
train_lr_dir = os.path.join(train_dir, 'lr')
val_hr_dir = os.path.join(val_dir, 'hr')
val_lr_dir = os.path.join(val_dir, 'lr')


downscaling_factor = 4  # Adjust as per your downscaling factor
hr_crop_size = 256  # Example crop size for HR images

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

batch_size = 5
lambda_perceptual=0.3
num_epochs=1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

model = Generator()
model = model.to(device)
loss = PerceptualLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

discriminator = Discriminator().to(device)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)

g_scheduler = StepLR(optimizer, step_size=50, gamma=0.5) 
d_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.5) 

def save_imgs(lr_input, target_hr):
    lr_img = lr_input.permute(1, 2, 0).numpy()
    plt.imshow(lr_img)
    plt.savefig('image_lr.png', bbox_inches='tight', pad_inches=0)
    
    hr_img = target_hr.permute(1, 2, 0).numpy()
    plt.imshow(hr_img)
    plt.savefig('image_hr.png', bbox_inches='tight', pad_inches=0)
    
    lr_input = lr_input.unsqueeze(0).to(device)
    target = target_hr.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(lr_input)
        print(loss(out, target).item())
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    plt.imshow(out)
    plt.savefig('image_model.png', bbox_inches='tight', pad_inches=0)

# Training loop
for epoch in range(num_epochs):
    batch_idx=0
    for lr_imgs, hr_imgs in train_loader:
        start_time = time.time()
        ### Discriminator Training
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        discriminator.zero_grad()

        # Train with real images
        real_preds = discriminator(hr_imgs.to(device))

        # Train with fake images
        fake_images = model(lr_imgs.to(device))
        fake_preds = discriminator(fake_images.detach())
        d_loss_real = criterion_GAN(real_preds - torch.mean(fake_preds), real_labels)
        d_loss_fake = criterion_GAN(fake_preds - torch.mean(real_preds), fake_labels)


        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        d_optimizer.step()

        model.zero_grad()

        # Adversarial loss for generator
        fake_preds_for_generator = discriminator(fake_images)
        g_loss_gan = criterion_GAN(fake_preds_for_generator - torch.mean(real_preds.detach()), real_labels)

        # Perceptual loss
        perceptual_loss = loss(fake_images, hr_imgs.to(device))

        # Combined loss
        g_loss = g_loss_gan + perceptual_loss * lambda_perceptual  

        g_loss.backward()
        optimizer.step()
        end_time = time.time()
        one_batch_time = end_time - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
              f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
              f'G_GAN Loss: {g_loss_gan.item():.4f}, Perceptual Loss: {perceptual_loss.item():.4f}, Time: {one_batch_time:.4f} seconds')
        batch_idx+=1
    g_scheduler.step()
    d_scheduler.step()
    if epoch % 50 == 0:
        torch.save(model.state_dict(), 'model_4.pth')
        print('Successfully Saved Checkpoint at model_4.pth')
        
    

torch.save(model.state_dict(), 'model_4.pth')
print('Successfuly saved model to model.pth')    

    
    
    
