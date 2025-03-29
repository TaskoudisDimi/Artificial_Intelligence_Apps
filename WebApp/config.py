from pathlib import Path

class Config:
    SECRET_KEY = 'your_secret_key_here'
    MODEL_DIR = Path(__file__).parent / 'models'
    DEBUG = True  # Set to False in production
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Google Drive File IDs
    GDRIVE_IDS = {
        "SVM_Iris_model": "1GF3Ried3YdNBmNI7Ry-GCRgCr7sEokfA",
        "KNN_Iris_model": "18hsEbOutisA25ht3B1i8VU5Lez_wxwY8",
        "KNearestCentroid_Iris_model": "1nlqU7JNlsCbV4hnMGUAh7YaVWCeg4SNl",
        "KMeans_Iris_model": "1h_z4xvavJ73_FQugBeJwZfu5T42eGUoB",
        "KMeans_breast_cancer_model": "1yjWkQ1QLdzmNXF4GZ7KS_kziGHmsttsQ",
        "scaler": "1DK_iId3gl8MhgA0Pw0re-tCoFhAAOIry",
        "Regression_Iris_model": "1VokC-aVW8o4zA_3a6NLRTKsF4Rpq9r6J",
        "Regression_House_model": "1K0EtqmjLnRSJrsTsupsH-Tf6cFmHjkeU",
        "Cifar_model_filename": "1jmO3pah-IfGqH_kzySgSLVWpOXy4SYTu",
        "Mnist_model": "1hOyDxX5vglHOYWhNItk4I4cy9on2UJ_4",
        "Activity_Recognize": "122RG_xW7_h9e0lkwkBZQpgFtgSWqChHZ",
    }