# Image Enhancement Model API

This repository contains a state-of-the-art image enhancement model API built on the Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN) architecture. Our model is trained on the high-quality DIV2K dataset, designed to provide significant improvements in photo resolution and quality. The API takes in an image and upscales it by 4x resolution.

## Features

- **High-Resolution Output:** Enhance images up to 4x their original resolution without losing detail.
- **Pre-Trained Model:** Utilizes the powerful DIV2K dataset for training, which includes a diverse set of high-resolution images.
- **Easy Integration:** Simple API for easy integration with existing projects or applications.
- **Real-Time Enhancement:** Capable of enhancing images in real-time with minimal latency.

## Getting Started

### Prerequisites

- Python 3.8+
- Flask
- TensorFlow 2.x
- CUDA (for GPU support)

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/image-enhancement-model-api.git
cd image-enhancement-model-api
pip install -r requirements.txt
```

## API Usage

### Enhance an Image

To enhance an image, make a POST request to the `/enhance` endpoint with the image data:

```bash
curl -X POST -F "image=@path_to_your_image.jpg" http://localhost:5000/enhance
```

The API will return the enhanced image in the response.

## Model Details

The model is based on the ESRGAN architecture, which is a robust approach to enhancing image resolution through generative adversarial networks. The key features of this model include:

- **Residual-in-Residual Dense Blocks (RRDB):** These blocks help in reconstructing more details from low-resolution images.
- **GAN-based Architecture:** Utilizes a discriminator network to guide the super-resolution process, resulting in more realistic images.

### Model Architecture

!(ESRGAN Generator and Discriminator Archiecture)(esrgan_architecture.png)

## Dataset

This model has been trained on the DIV2K dataset, which is a benchmark for image super-resolution techniques. It contains 800 high-quality images, which are diverse in scenes and subjects, providing a robust training set for high-resolution image enhancement.

## Training Results

Here are the hyperparameters I used in training the model:

- *Residual-in-Residual Dense Blocks:* 23
- *Batch Size: 10*
- *Crop Size: 356*
- *Lambda Weight: 0.3*
- *Learning Rate: 0.0001
- *Epochs: 240*

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Acknowledgments

- Thanks to the authors of the ESRGAN paper and the creators of the DIV2K dataset for their contributions to the field of image super-resolution.
- This project would not be possible without the support from the open-source community.



