import numpy as np
from sklearn.ensemble import VotingClassifier
from models.base import BaseChurnModel
from config import PipelineConfig
from classical.logistic import LogisticChurnModel
from classical.random_forest import RandomForestChurnModel
from classical.xgboost_model import XGBoostChurnModel


class EnsembleChurnModel(BaseChurnModel):
    def __init__(self, config):
        self._model = VotingClassifier(
                estimators=[
                    ('logistic', LogisticChurnModel(config)._model),
                    ('rf', RandomForestChurnModel(config)._model),
                    ('xgb', XGBoostChurnModel(config)._model),
                ],
                voting='soft',
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleChurnModel":
        self._model.fit(X, y)
        return self
        
    def predict(self, X: np.ndarray) -> np.ndarray: 
        return self._model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
