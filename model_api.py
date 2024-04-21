from flask import Flask, request, send_file, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch
import io

app = Flask(__name__)

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def prepare_image(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img = transform(img).unsqueeze(0).numpy()
    return img

def run_inference(session, input_tensor):
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    return output[0]

def image_to_bytes(image_tensor):
    image = transforms.ToPILImage()(torch.tensor(image_tensor.squeeze(0)).clamp(0, 1))
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return byte_arr

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        input_tensor = prepare_image(img)
        output_tensor = run_inference(model_session, input_tensor)
        output_image = image_to_bytes(output_tensor)
        return send_file(output_image, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    model_path = 'model.onnx'
    model_session = load_model(model_path)
    app.run(host='0.0.0.0', port=5001)
