# UTMIST-OpenCV-Image-Enchancement -- Mock Nvidia DLSS

## Overview
This project is a mock clone of Nvidia's super resolution technology. The goal is to experiment with and explore deep learning techniques for enhancing low-resolution images. By mimicking aspects of Nvidia's advanced models, the project implements a convolutional neural network with residual blocks and sub-pixel convolution (PixelShuffle) layers to upscale images while preserving and enhancing details.

## Motivation
Recent advancements in deep learning have revolutionized the field of image processing, particularly for super resolution tasks. Inspired by Nvidia's groundbreaking work in improving image quality for gaming, media, and computer vision, this project aims to:
- Provide an accessible platform for learning and experimentation with super resolution techniques.
- Enable users to upscale images without resorting to proprietary software.
- Foster innovation by replicating a state-of-the-art method using open-source tools.

## Features
- **Deep Learning Upscaling:** Uses a CNN architecture with residual connections and PixelShuffle layers to effectively enhance low-resolution images.
- **Custom PyTorch Dataset:** Automatically manages high-resolution images and creates low-resolution inputs by downscaling, forming paired datasets for training.
- **Training Pipeline:** Integrates data preprocessing, model training (with Mean Squared Error loss and Adam optimizer), and periodic logging of training metrics.
- **Inference Capabilities:** Offers an inference script to upscale new images using the trained model.

## Requirements
- Python 3.7 or later
- OpenCV
- NumPy
- Matplotlib
- PyTorch & Torchvision

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/YourUsername/Mock-Nvidia-Super-Resolution.git
cd Mock-Nvidia-Super-Resolution
pip install -r requirements.txt
