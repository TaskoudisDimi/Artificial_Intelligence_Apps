from matplotlib import transforms
import torch
from PIL import ImageChops
import os
import gdown
import joblib
from io import BytesIO
import requests
import gdown
import joblib
import os
from pathlib import Path
import tensorflow as tf


def load_model(model_path):
    """Load a PyTorch model from the specified path."""
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image, size=(32, 32)):
    """Preprocess the input image for model prediction."""
    image = image.resize(size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)  # Add a batch dimension

def center_image(img):
    """Center the image by cropping it to the bounding box of the non-zero pixels."""
    w, h = img.size[:2]
    left, top, right, bottom = w, h, -1, -1
    imgpix = img.getdata()

    for y in range(h):
        offset_y = y * w
        for x in range(w):
            if imgpix[offset_y + x] > 0:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    shift_x = (left + (right - left) // 2) - w // 2
    shift_y = (top + (bottom - top) // 2) - h // 2
    return ImageChops.offset(img, -shift_x, -shift_y)

def load_pickle_model_from_cloud(model_url, cache_dir="downloaded_models"):
    """Load a pickle model directly from a cloud URL using gdown."""
    try:
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Extract file ID from URL
        file_id = model_url.split("id=")[-1]
        output_path = os.path.join(cache_dir, f"{file_id}.pkl")
        
        # Download file using gdown if not already cached
        if not os.path.exists(output_path):
            gdown.download(id=file_id, output=output_path, quiet=False)
        
        # Load the pickle file
        return joblib.load(output_path)
    except Exception as e:
        raise Exception(f"Failed to load model from {model_url}. Error: {str(e)}")
    
def load_torch_model_from_cloud(model_class, model_url, device="cpu"):
    """Load a PyTorch model directly from a cloud URL."""
    response = requests.get(model_url)
    if response.status_code == 200:
        model = model_class().to(device)
        model.load_state_dict(torch.load(BytesIO(response.content), map_location=torch.device(device)))
        model.eval()
        return model
    else:
        raise Exception(f"Failed to load model from {model_url}. Status code: {response.status_code}")
    

