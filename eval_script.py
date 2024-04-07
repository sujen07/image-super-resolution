import torch
from models import Generator
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image


model = Generator()
model.load_state_dict(torch.load('out/model.pth'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()

img = Image.open('27.png').convert('RGB')

transform = transforms.Compose([
        transforms.ToTensor(),
    ])
img = transform(img).unsqueeze(0).to(device)


with torch.no_grad():
    out = model(img)
out = out.squeeze(0).cpu()

save_image(out, 'output_image.png')


