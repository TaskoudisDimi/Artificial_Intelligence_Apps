from torchvision import transforms
import torch
from Models.Mnist.model import Model
from PIL import Image, ImageChops, ImageOps


# SAVE_MODEL_PATH = "/media/dimitristaskoudis/HardDisk1/Codes/Artificial_Intelligence_Apps/WebApp/Models/best_accuracy.pth"



# class Predict():
#     def __init__(self):
#         device = torch.device("cpu")
#         self.model = Model().to(device)
#         self.model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=device))
#         self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

#     def _centering_img(self, img):
#         w, h = img.size[:2]
#         left, top, right, bottom = w, h, -1, -1
#         imgpix = img.getdata()

#         for y in range(h):
#             offset_y = y * w
#             for x in range(w):
#                 if imgpix[offset_y + x] > 0:
#                     left = min(left, x)
#                     top = min(top, y)
#                     right = max(right, x)
#                     bottom = max(bottom, y)

#         shift_x = (left + (right - left) // 2) - w // 2
#         shift_y = (top + (bottom - top) // 2) - h // 2
#         return ImageChops.offset(img, -shift_x, -shift_y)

#     def __call__(self, img):
#         img = ImageOps.invert(img)  # MNIST image is inverted
#         img = self._centering_img(img)
#         img = img.resize((28, 28), Image.BICUBIC)  # resize to 28x28
#         tensor = self.transform(img)
#         tensor = tensor.unsqueeze_(0)  # 1,1,28,28

#         self.model.eval()
#         with torch.no_grad():
#             preds = self.model(tensor)
#             preds = preds.detach().numpy()[0]

#         return preds
    

