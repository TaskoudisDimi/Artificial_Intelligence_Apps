from sklearn.cluster import KMeans
import joblib
import numpy as np

class BreastCancerModel:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input_data):
        scaled_data = self.scaler.transform(input_data)
        prediction = self.model.predict(scaled_data)
        return prediction

# Example usage:
# model = BreastCancerModel('models/breast_cancer_model.pkl', 'models/scaler.pkl')
# result = model.predict(np.array([[mean_radius, mean_texture]]))