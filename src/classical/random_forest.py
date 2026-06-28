
import numpy as np 
from sklearn.ensemble import RandomForestClassifier  
from models.base import BaseChurnModel
from config import PipelineConfig

class RandomForestChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig):
        self._model = RandomForestClassifier(
                n_estimators=config.classical.n_estimators, 
                random_state=config.classical.random_state,
                class_weight='balanced',
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestChurnModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

