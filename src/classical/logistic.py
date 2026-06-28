import numpy as np 
from sklearn.linear_model import LogisticRegression
from models.base import BaseChurnModel
from config import PipelineConfig

class LogisticChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig): 
        self._model = LogisticRegression(
                max_iter=config.classical.max_iter, 
                random_state=config.classical.random_state,
                class_weight='balanced',
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticChurnModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return np.abs(self._model.coef_[0])
