import os
import glob
from PIL import Image
from tqdm import tqdm

from prepare_data import preprocess_and_save

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

train_data = []
img_folder = 'Humans'

all_imgs = [file for file in os.listdir(img_folder) if is_image_file(file)]

for image_path in tqdm(all_imgs):
    img_path = os.path.join(img_folder, image_path)
    img = Image.open(img_path)

    new_width = img.width // 4
    new_height = img.height // 4

    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.LANCZOS)

    # Path to save the downscaled image
    downscaled_image_path = os.path.join(img_folder, 'lr_' + os.path.splitext(image_path)[0] + '.png')

    # Save the downscaled image
    img_resized.save(downscaled_image_path)

    train_data.append({'lr': downscaled_image_path, 'hr': img_path})

preprocess_and_save(train_data, 'train', start_ind=801)
