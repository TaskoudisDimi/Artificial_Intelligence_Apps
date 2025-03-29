from flask import Blueprint, request, jsonify, render_template
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from app.models.cifar import Net as CifarModel
from app.models.mnist import Model as MnistModel

computer_vision_bp = Blueprint('computer_vision', __name__)

# Load models
mnist_model = MnistModel()
cifar_model = CifarModel()

mnist_model.load_state_dict(torch.load('models/Mnist/model.pth', map_location=torch.device('cpu')))
mnist_model.eval()

cifar_model.load_state_dict(torch.load('models/Cifar/Net.pth', map_location=torch.device('cpu')))
cifar_model.eval()

# Define image transformations
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cifar_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@computer_vision_bp.route('/computer_vision/mnist', methods=['POST'])
def predict_mnist():
    img = Image.open(request.files['img']).convert('L')
    img = mnist_transform(img).unsqueeze(0)

    with torch.no_grad():
        preds = mnist_model(img)
        predicted_class = np.argmax(preds.numpy())

    return jsonify({'predicted_class': int(predicted_class)})

@computer_vision_bp.route('/computer_vision/cifar', methods=['POST'])
def predict_cifar():
    img = Image.open(request.files['image'])
    img = cifar_transform(img).unsqueeze(0)

    with torch.no_grad():
        output = cifar_model(img)
        _, predicted_class_idx = torch.max(output, 1)
        predicted_class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                'dog', 'frog', 'horse', 'ship', 'truck'][predicted_class_idx.item()]

    return jsonify({'predicted_class': predicted_class_name})

@computer_vision_bp.route('/computer_vision', methods=['GET'])
def computer_vision():
    return render_template('ComputerVision.html')