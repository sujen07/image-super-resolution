import os
import torch
from torch.nn.functional import pad
from models import Generator
from torch.optim.lr_scheduler import StepLR
from models import PerceptualLoss
from models import Discriminator
from load_data import load_data
import matplotlib.pyplot as plt
from PIL import Image
import time
from argparse import ArgumentParser
import wandb



# Default Hyperparameters
default_downscaling_factor = 4
default_hr_crop_size = 256
default_batch_size = 5
default_lambda_perceptual = 0.7
default_learning_rate = 0.0001
default_num_epochs = 1000
default_model_name = 'model.pth'
default_out_dir = 'out'
default_wandb_log = False



# Check for command line args for hyperparameters
def parse_args():
    parser = ArgumentParser()
    # Use the default values defined above
    parser.add_argument("--downscaling_factor", type=int, default=default_downscaling_factor)
    parser.add_argument("--hr_crop_size", type=int, default=default_hr_crop_size)
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--lambda_perceptual", type=float, default=default_lambda_perceptual)
    parser.add_argument("--num_epochs", type=int, default=default_num_epochs)
    parser.add_argument("--model_name", type=str, default=default_model_name)
    parser.add_argument("--out_dir", type=str, default=default_out_dir)
    parser.add_argument("--wandb_log", type=bool, default=default_wandb_log)
    parser.add_argument("--lr", type=bool, default=default_learning_rate)
    return parser.parse_args()


def get_val_loss(model, val_loader, loss, device):
    total_loss = 0
    with torch.no_grad():
        for val_lr, val_hr in val_loader:
            output = model(val_lr.to(device))
            total_loss += loss(output, val_hr.to(device)).item()
    total_loss = total_loss / len(val_loader)
    return total_loss




def train(downscaling_factor, hr_crop_size, batch_size, lambda_perceptual, learning_rate, num_epochs, model_path, wandb_log):
    print(f"Training configuration:\n"
          f"Model Path: {model_path}\n"
          f"Epochs: {num_epochs}\n"
          f"Batch Size: {batch_size}\n"
          f"Downscaling Factor: {downscaling_factor}\n"
          f"High Resolution Crop Size: {hr_crop_size}\n"
          f"Lambda Perceptual: {lambda_perceptual}\n")

    train_dir = 'data/train'
    val_dir = 'data/validation'
    train_loader, val_loader = load_data(downscaling_factor, batch_size, hr_crop_size, train_dir, val_dir)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using Device: {device}')
    model = Generator()
    model = model.to(device)
    loss = PerceptualLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    discriminator = Discriminator().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)

    g_scheduler = StepLR(optimizer, step_size=50, gamma=0.5) 
    d_scheduler = StepLR(d_optimizer, step_size=50, gamma=0.5) 


    # Training loop
    for epoch in range(num_epochs):
        batch_idx=0
        epoch_loss = 0
        d_loss_epoch = 0
        g_loss_epoch = 0
        start_time = time.time()
        for lr_imgs, hr_imgs in train_loader:
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
            d_loss_epoch += d_loss.item()
            d_loss.backward()
            d_optimizer.step()

            model.zero_grad()

            # Adversarial loss for generator
            fake_preds_for_generator = discriminator(fake_images)
            g_loss_gan = criterion_GAN(fake_preds_for_generator - torch.mean(real_preds.detach()), real_labels)
            g_loss_epoch += g_loss_gan.item()

            # Perceptual loss
            perceptual_loss = loss(fake_images, hr_imgs.to(device))

            # Combined loss
            g_loss = g_loss_gan + perceptual_loss * lambda_perceptual  
            epoch_loss += perceptual_loss.item()

            g_loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, '
                f'G_GAN Loss: {g_loss_gan.item():.4f}, Perceptual Loss: {perceptual_loss.item():.4f}')
            batch_idx+=1
        g_scheduler.step()
        d_scheduler.step()
        end_time = time.time()
        one_epoch_time = end_time - start_time
        train_loss = epoch_loss / len(train_loader)
        d_loss_epoch = d_loss_epoch / len(train_loader)
        g_loss_epoch = g_loss_epoch / len(train_loader)
        if wandb_log:
            wandb.log({'d_loss': d_loss_epoch, 'g_loss': g_loss_epoch})
        print(f'Time for Epoch {epoch+1}: {one_epoch_time} Seconds')
        if epoch % 20 == 0:
            val_loss = get_val_loss(model, val_loader, loss, device)
            if wandb_log:
                wandb.log({"Validation Perceptual Loss": val_loss, "Training Perceptual Loss": train_loss})
            print(f'Perceptual Loss Train: {train_loss}, Val: {val_loss}, Time for Epoch {epoch+1}: {one_epoch_time} Seconds')
            torch.save(model.state_dict(), model_path)
            print(f'Successfully Saved Checkpoint at {model_path}')
            
        

    torch.save(model.state_dict(), model_path)
    print(f'Successfuly saved model to {model_path}')    


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    model_path = os.path.join(args.out_dir, args.model_name)
    if args.wandb_log:
        wandb.login()
        config = vars(args)
        run = wandb.init(
            project="image-resolution",
            # Track hyperparameters and run metadata
            config=config
        )
    train(args.downscaling_factor, args.hr_crop_size, args.batch_size, args.lambda_perceptual, args.lr, args.num_epochs, model_path, args.wandb_log)