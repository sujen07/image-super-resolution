from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import os
import tqdm

def create_synthetic_text_image(text, font_path=None, image_size=(512, 512), font_size=24):
    image = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(image)
    
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
                # Handle default font scaling manually
                text_width, text_height = draw.textsize(text, font=font)
                text_width = int(text_width * (font_size / 10))
                text_height = int(text_height * (font_size / 10))
                image = Image.new('RGB', (text_width, text_height), color='white')
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

    if font.path != "default":
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    else:
        text_width, text_height = draw.textsize(text, font=font)

    # Calculate the position to center the text
    position_x = (image_size[0] - text_width) // 2
    position_y = (image_size[1] - text_height) // 2 + (text_bbox[1] - text_bbox[3]) // 2

    draw.text((position_x, position_y), text, fill='black', font=font)
    return image

# Example usage
font_path = None  # Set to 'None' to test the fallback
for i in tqdm.tqdm(range(2000)):
    text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=1))
    image = create_synthetic_text_image(text, font_path, font_size=600)
    image.save(f'img_folder/text{i}.png')
