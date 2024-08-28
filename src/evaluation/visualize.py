import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.training.models import *
import pdb
from PIL import Image


# Load the trained model
model = Generator()
model.load_state_dict(torch.load('out/model.pth'))
model.eval()

# Load and preprocess the test image
transform = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

image_path = 'data/train/lr/9.png'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension

# Function to register hooks and store activations
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks to the layers you want to visualize
model.first_conv.register_forward_hook(get_activation('first_conv'))
for idx, layer in enumerate(model.rrdb_blocks):
    layer.register_forward_hook(get_activation(f'rrdb_block_{idx}'))

# Pass the image through the model
to_pil = transforms.ToPILImage()

device = 'cuda'
model = model.to(device)
image = image.to(device)
output = model(image)
pil_image = to_pil(output.squeeze(0))
save_path = 'activation_image.jpg'
pil_image.save(save_path)


# Save the activations as image files
def save_activations(activations, layer_name, save_dir='activations'):
    os.makedirs(save_dir, exist_ok=True)
    act = activations[layer_name].squeeze(0)  # Remove batch dimension
    num_activations = act.size(0)
    num_cols = 8
    num_rows = (num_activations + num_cols - 1) // num_cols  # Adjust rows to fit all activations
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < num_activations:
            ax.imshow(act[i].cpu().numpy(), cmap='viridis')
            ax.axis('off')
    plt.suptitle(f'Activations of {layer_name}', fontsize=16)
    save_path = os.path.join(save_dir, f'{layer_name}.png')
    plt.savefig(save_path)
    plt.close()

# Save activations for a specific layer
save_activations(activations, 'first_conv')
# Save activations for a specific RRDB block
save_activations(activations, 'rrdb_block_0')  # Change the index as needed
