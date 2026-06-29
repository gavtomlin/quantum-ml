import time
import numpy as np 
from sklearn.svm import SVC 
from models.base import BaseChurnModel
from config import PipelineConfig
from quantum.kernel import build_kernel

class QuantumSVMModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.n_qubits = config.quantum.n_qubits
        self._max_samples = config.quantum.max_kernel_samples
        self._kernel = build_kernel(self.n_qubits)
        self._model = SVC(kernel='precomputed', class_weight='balanced',probability=True)
        self._X_train = None

    def _kernel_matrix(self, X1, X2, label=""):
        n = len(X1)
        rows = []
        for i, x1 in enumerate(X1):
            rows.append([self._kernel(x1, x2) for x2 in X2])
            if (i + 1) % 10 == 0 or (i + 1) == n:
                print(f"  {label}row {i + 1}/{n}", end="\r", flush=True)
        print()
        return np.array(rows)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumSVMModel":
        X, y = X[:self._max_samples], y[:self._max_samples]
        self._X_train = X
        print(f"QuantumSVM: building {len(X)}×{len(X)} kernel matrix ({self.n_qubits} qubits)...")
        t0 = time.time()
        K_train = self._kernel_matrix(X, X, label="train ")
        print(f"QuantumSVM: kernel matrix done in {time.time() - t0:.1f}s, fitting SVC...")
        self._model.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel_matrix(X, self._X_train, label="test ")
        return self._model.predict(K_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        K_test = self._kernel_matrix(X, self._X_train, label="test ")
        return self._model.predict_proba(K_test)

