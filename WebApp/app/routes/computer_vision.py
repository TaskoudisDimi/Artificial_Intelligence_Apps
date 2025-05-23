from flask import Blueprint, request, jsonify, render_template, current_app
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from app.models.mnist import PredictMnist

computer_vision_bp = Blueprint('computer_vision', __name__)

# Define image transformations
mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28 for MNIST
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@computer_vision_bp.route('/computer_vision/mnist', methods=['POST'])
def predict_upload_mnist():
    img = Image.open(request.files['img']).convert('L')
    img = mnist_transform(img).unsqueeze(0)
    # Get models from app context
    models = current_app.models  
    with torch.no_grad():
        preds = models["Mnist_model"](img)
        predicted_class = torch.argmax(preds, dim=1).item()  # Use torch.argmax for consistency

    return render_template('ComputerVision_MNIST_Up_image.html', predicted_value=predicted_class)

@computer_vision_bp.route('/computer_vision/cifar', methods=['POST'])
def predict_cifar():
    img = Image.open(request.files['image'])
    img = cifar_transform(img).unsqueeze(0)
    models = current_app.models
    with torch.no_grad():
        output = models["Cifar_model"](img)
        _, predicted_class_idx = torch.max(output, 1)
        predicted_class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                'dog', 'frog', 'horse', 'ship', 'truck'][predicted_class_idx.item()]

    return render_template('ComputerVision_CIFAR10.html', predicted_value=predicted_class_name)

@computer_vision_bp.route('/computer_vision', methods=['GET'])
def computer_vision():
    return render_template('ComputerVision.html')