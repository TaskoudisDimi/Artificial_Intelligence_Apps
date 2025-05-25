from pathlib import Path

class Config:
    SECRET_KEY = 'your_secret_key_here'
    MODEL_DIR = Path(__file__).parent / 'downloaded_models'
    DEBUG = True  # Set to False in production
    ALLOWED_EXTENSIONS = {'pth', 'pkl'}
    
   
    GDRIVE_IDS = {
        "SVM_Iris_model": "https://drive.google.com/uc?id=1GF3Ried3YdNBmNI7Ry-GCRgCr7sEokfA",
        "KNN_Iris_model": "https://drive.google.com/uc?id=18hsEbOutisA25ht3B1i8VU5Lez_wxwY8",
        "KNearestCentroid_Iris_model": "https://drive.google.com/uc?id=1nlqU7JNlsCbV4hnMGUAh7YaVWCeg4SNl",
        "KMeans_Iris_model": "https://drive.google.com/uc?id=1h_z4xvavJ73_FQugBeJwZfu5T42eGUoB",
        "KMeans_breast_cancer_model": "https://drive.google.com/uc?id=1yjWkQ1QLdzmNXF4GZ7KS_kziGHmsttsQ",
        "Clustering_Breast_Cancer": "https://drive.google.com/uc?id=1DK_iId3gl8MhgA0Pw0re-tCoFhAAOIry",
        "Regression_Iris_model": "https://drive.google.com/uc?id=1VokC-aVW8o4zA_3a6NLRTKsF4Rpq9r6J",
        "regression_house": "https://drive.google.com/uc?id=1dwi0NbPU66mFcj2Vq_v5zxBIHyb3cKXE",
        "Cifar_model_filename": "https://drive.google.com/uc?id=1jmO3pah-IfGqH_kzySgSLVWpOXy4SYTu",
        "Mnist_model": "https://drive.google.com/uc?id=1hOyDxX5vglHOYWhNItk4I4cy9on2UJ_4",
    }
    