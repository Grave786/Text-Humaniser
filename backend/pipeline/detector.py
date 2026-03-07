from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

# Detector class that loads a machine learning model from a specified path and provides a method to predict probabilities based on input feature vectors. It includes error handling for model loading and prediction, and ensures that the input feature vector is compatible with the expected feature space of the loaded model.
class Detector:
    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self._load_model()
# Load the model from the specified path, handling errors and ensuring that the model is only loaded once.
    def _load_model(self) -> None:
        if joblib is None:
            self.model = None
            return
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None
# Predict the probability of a feature vector using the loaded model, with error handling and input length adjustment to match the expected feature space of the model. If the model is not loaded or an error occurs during prediction, return a default probability of 0.5.
    def predict_probability(self, feature_vector: List[float]) -> float:
        if self.model is None:
            return 0.5

        expected_features = getattr(self.model, "n_features_in_", None)
        vector = np.array(feature_vector, dtype=float)

        # Keep API stable even if loaded model expects a different feature space.
        if isinstance(expected_features, (int, np.integer)) and expected_features > 0:
            if vector.shape[0] < expected_features:
                vector = np.pad(vector, (0, expected_features - vector.shape[0]), mode="constant")
            elif vector.shape[0] > expected_features:
                vector = vector[:expected_features]

        x = vector.reshape(1, -1)

        try:
            if hasattr(self.model, "predict_proba"):
                score = float(self.model.predict_proba(x)[0][1])
                return round(score, 4)

            prediction = float(self.model.predict(x)[0])
            return round(prediction, 4)
        except Exception:
            return 0.5
