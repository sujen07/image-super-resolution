import torch.onnx
import torch
import onnx
from models import Generator
from PIL import Image
from torchvision import transforms
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch.nn.utils.prune as prune

onnx_path = "out/model.onnx"

model = Generator()
model.load_state_dict(torch.load('model.pth'))

for layer in model.upsample:
    if isinstance(layer, torch.nn.Conv2d):
        prune.l1_unstructured(layer, name='weight', amount=0.3)

for layer in model.upsample:
    if isinstance(layer, torch.nn.Conv2d):
        prune.remove(layer, 'weight')

for layer in model.rrdb_blocks:
    if isinstance(layer, torch.nn.Conv2d):
        prune.l1_unstructured(layer, name='weight', amount=0.2)

for layer in model.rrdb_blocks:
    if isinstance(layer, torch.nn.Conv2d):
        prune.remove(layer, 'weight')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img = Image.open('84.png').convert('RGB')
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
dummy_input = transform(img).unsqueeze(0).to(device)

input_names = ['input_image']
output_names = ['output_image']
torch.onnx.export(model, 
                  dummy_input, 
                  onnx_path, 
                  export_params=True, 
                  input_names=input_names, 
                  output_names=output_names,
                  dynamic_axes={
                                'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
                                'output_image': {0: 'batch_size'}
                                },
                  do_constant_folding=True,
                  )
