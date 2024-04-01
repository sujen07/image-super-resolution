import os
import glob
from PIL import Image

from prepare_data import preprocess_and_save


train_data = []
img_folder = 'img_folder/flowers'

all_imgs = glob.glob(os.path.join(img_folder, '*.png'))

for image_path in all_imgs:
    img = Image.open(image_path)

    new_width = img.width // 4
    new_height = img.height // 4

    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Path to save the downscaled image
    downscaled_image_path = os.path.join(img_folder, 'lr_' + os.path.basename(image_path))

    # Save the downscaled image
    img_resized.save(downscaled_image_path)

    train_data.append({'lr': downscaled_image_path, 'hr': image_path})

preprocess_and_save(train_data, 'train', start_ind=800)



