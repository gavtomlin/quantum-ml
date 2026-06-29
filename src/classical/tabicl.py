import sys
import time
import numpy as np
import torch
from tabicl import TabICLClassifier
from models.base import BaseChurnModel
from config import PipelineConfig

class TabICLChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig): 
        self._model = TabICLClassifier(
            random_state=config.classical.random_state,
            device='cpu',
            batch_size=4,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabICLChurnModel":
        X, y = X[:200], y[:200]
        print(f"TabICL: fitting on {len(X)} samples, {X.shape[1]} features (cpu)...")
        t0 = time.time()
        original_argv = sys.argv[:]
        sys.argv = sys.argv[:1]
        try:
            torch.set_num_threads(1)
            self._model.fit(X, y)
            print(f"TabICL: done in {time.time() - t0:.1f}s")
        except SystemExit as e:
            print(f"TabICL: sys.exit({e.code}) caught — skipping")
        except Exception as e:
            print(f"TabICL: error — {e}")
        finally:
            sys.argv = original_argv
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray: 
        return self._model.predict_proba(X)

