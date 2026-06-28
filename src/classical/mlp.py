import numpy as np 
from sklearn.neural_network import MLPClassifier
from models.base import BaseChurnModel
from config import PipelineConfig

class MLPChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig):
        self._model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=config.classical.max_iter,
                random_state=config.classical.random_state,
                early_stopping=True,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPChurnModel":
        self._model.fit(X, y)
        self._threshold = y.sum() / len(y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._model.predict_proba(X)[:, 1]>=self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
