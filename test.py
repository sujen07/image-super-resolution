from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import string
import os

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

    position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
    print(text_width, text_height)
    draw.text(position, text, fill='black', font=font)
    return image

# Example usage
text = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
font_path = None  # Set to 'None' to test the fallback
for i in range(2000):
    image = create_synthetic_text_image(text, font_path, font_size=100)
    image.save(f'img_folder/text{i}.png')
