from pathlib import Path
import torch
from torchvision import transforms
from Old_Models.Cifar.Net import Net

class CifarModel:
    def __init__(self, model_path):
        self.model = Net()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()  # Set the model to evaluation mode
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def predict(self, image):
        image = self.transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            output = self.model(image)
        return output.argmax(dim=1).item()  # Return the predicted class index

# Load the CIFAR-10 model
model_path = Path(__file__).resolve().parent.parent / 'downloaded_models' / 'Cifar' / 'Cifar_model_filename.pth'
cifar_model = CifarModel(model_path)