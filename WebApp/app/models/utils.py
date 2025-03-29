from matplotlib import transforms
import torch


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