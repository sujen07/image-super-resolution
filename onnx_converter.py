import torch.onnx
import torch
import onnx
from models import Generator
from PIL import Image
from torchvision import transforms

onnx_path = "out/model.onnx"

model = Generator()
model.load_state_dict(torch.load('model.pth'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img = Image.open('84.png').convert('RGB')
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
dummy_input = transform(img).unsqueeze(0).to(device)

torch.onnx.export(model, 
                  dummy_input, 
                  onnx_path, 
                  export_params=True, 
                  do_constant_folding=True)