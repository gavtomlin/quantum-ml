from abc import ABC, abstractmethod
import numpy as np 
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report 

class BaseChurnModel(ABC):

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseChurnModel": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict: 
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "roc_auc": roc_auc_score(y, y_proba), 
            "report": classification_report(y, y_pred),
        }

    @property
    def name(self) -> str:
        return self.__class__.__name__

