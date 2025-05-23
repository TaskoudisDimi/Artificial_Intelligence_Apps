import os
from .utils import load_torch_model_from_cloud, load_pickle_model_from_cloud
from Old_Models.Cifar.Net import Net
from Old_Models.Mnist.model import Model

def load_models_from_cloud(config):
    model_urls = config.get('GDRIVE_IDS', {})

    models = {
        "SVM_Iris_model": load_pickle_model_from_cloud(model_urls["SVM_Iris_model"]),
        "KNN_Iris_model": load_pickle_model_from_cloud(model_urls["KNN_Iris_model"]),
        "KNearestCentroid_Iris_model": load_pickle_model_from_cloud(model_urls["KNearestCentroid_Iris_model"]),
        "KMeans_Iris_model": load_pickle_model_from_cloud(model_urls["KMeans_Iris_model"]),
        "KMeans_breast_cancer_model": load_pickle_model_from_cloud(model_urls["KMeans_breast_cancer_model"]),
        "Clustering_Breast_Cancer": load_pickle_model_from_cloud(model_urls["Clustering_Breast_Cancer"]),
        "Regression_Iris_model": load_pickle_model_from_cloud(model_urls["Regression_Iris_model"]),
        "regression_house": load_pickle_model_from_cloud(model_urls["regression_house"]),
        "Cifar_model": load_torch_model_from_cloud(Net, model_urls["Cifar_model_filename"]),
        "Mnist_model": load_torch_model_from_cloud(Model, model_urls["Mnist_model"]),
    }
    print("Loaded model URLs:", model_urls)

    return models