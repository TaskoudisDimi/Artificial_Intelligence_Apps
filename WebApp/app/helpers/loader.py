import os
from .utils import load_torch_model_from_cloud, load_pickle_model_from_cloud
from Old_Models.Cifar.Net import Net
from Old_Models.Mnist.model import Model

def load_models_from_cloud(config):
    """Load all models directly from the cloud."""
    #model_urls = config.get('MODEL_URLS', {})
    model_urls = config.get('GDRIVE_IDS', {})  # ✅ Correct Key

    models = {
        "SVM_Iris_model": load_pickle_model_from_cloud(model_urls["SVM_Iris_model"]),
        "KNN_Iris_model": load_pickle_model_from_cloud(model_urls["KNN_Iris_model"]),
        "Regression_Iris_model": load_pickle_model_from_cloud(model_urls["Regression_Iris_model"]),
        "Regression_House_model": load_pickle_model_from_cloud(model_urls["Regression_House_model"]),
        "Cifar_model": load_torch_model_from_cloud(Net, model_urls["Cifar_model_filename"]),
        "Mnist_model": load_torch_model_from_cloud(Model, model_urls["Mnist_model"]),
    }
    print("Loaded model URLs:", model_urls)  # Check if model URLs are correct

    return models