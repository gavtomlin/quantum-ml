
import numpy as np 
from xgboost import XGBClassifier 
from models.base import BaseChurnModel
from config import PipelineConfig

class XGBoostChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig):
        self._model = XGBClassifier(
                n_estimators=config.classical.n_estimators,
                random_state=config.classical.random_state,
                eval_metric='logloss',
                nthread=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostChurnModel":
        neg = (y == 0).sum()
        pos = (y == 1).sum()
        self._model.set_params(scale_pos_weight=neg / pos)
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_
    
