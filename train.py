import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.functional import pad
from models import Generator

from models import ImageDataset
from models import PerceptualLoss

device = torch.device("cuda")

train_dir = 'data/train'
val_dir = 'data/validation'

train_hr_dir = os.path.join(train_dir, 'hr')
train_lr_dir = os.path.join(train_dir, 'lr')
val_hr_dir = os.path.join(val_dir, 'hr')
val_lr_dir = os.path.join(val_dir, 'lr')

def collate_fn(batch):
    images, targets = zip(*batch)

    return list(images), list(targets)


transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = ImageDataset(hr_dir=train_hr_dir, lr_dir=train_lr_dir, transform=transform)
val_dataset = ImageDataset(hr_dir=val_hr_dir, lr_dir=val_lr_dir, transform=transform)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

model = Generator()
model = model.to(device)
loss = PerceptualLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# Training loop
for lr_imgs, hr_imgs in train_loader:
    #losses = []
    print('test')
    for lr_img, hr_img in zip(lr_imgs, hr_imgs):
        optimizer.zero_grad()
        lr_img = lr_img.unsqueeze(0).to(device)
        hr_img = hr_img.unsqueeze(0).to(device)
        out = model(lr_img)
        out = torch.clamp(out, 0, 1)
        print('ran model')
        img_loss = loss(out, hr_img)
        #losses.append(img_loss)
        img_loss.backward()
        optimizer.step()
        
    #batch_loss = torch.stack(losses).mean()
    #batch_loss.backward()
    #optimizer.step()
    torch.cuda.empty_cache()
    #print(len(losses))
    print(f'Epoch test, loss: {img_loss.item()}')
    

    
new_lr_img = lr_img.unsqueeze(0).to(device)