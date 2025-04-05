import joblib
import numpy as np

class IrisModel:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, features):
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)
        return prediction[0] if prediction else None

    def predict_proba(self, features):
        features = np.array(features).reshape(1, -1)
        probabilities = self.model.predict_proba(features)
        return probabilities[0] if probabilities else None