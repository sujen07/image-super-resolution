import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import numpy as np

# Configuration
num_proc = 4  # Number of CPU Cores // 2
out_dir = "data"  # Output directory for preprocessed images
scale_factor = 4  # Scale factor for resizing, adjust based on your requirements


# Preprocess and save function
def preprocess_and_save(data, subset, start_ind=0):
    os.makedirs(f"{out_dir}/{subset}", exist_ok=True)  # Create output directory
    os.makedirs(f"{out_dir}/{subset}/lr", exist_ok=True)  # Low-res images directory
    os.makedirs(f"{out_dir}/{subset}/hr", exist_ok=True)  # High-res images directory
    
    for i, item in tqdm(enumerate(data), total=len(data)):
        # Load images
        lr_img_path, hr_img_path = item['lr'], item['hr']
        lr_img = Image.open(lr_img_path)
        hr_img = Image.open(hr_img_path)


        # Save processed images
        lr_img.save(f"{out_dir}/{subset}/lr/{i+start_ind}.png")
        hr_img.save(f"{out_dir}/{subset}/hr/{i+start_ind}.png")

# Process and save training and validation datasets
# Load dataset
if __name__ == '__main__':
    dataset = load_dataset('div2k.py', 'bicubic_x2')        

    preprocess_and_save(dataset['train'], 'train')
    preprocess_and_save(dataset['validation'], 'validation')

    print("Data preprocessing completed.")