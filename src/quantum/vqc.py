import numpy as np
import pennylane.numpy as pnp
import pennylane as qml
from models.base import BaseChurnModel 
from config import PipelineConfig 
from quantum.encoding import angle_encoding_layer, amplitude_encoding_layer 

class VQCChurnModel(BaseChurnModel):

    def __init__(self, config: PipelineConfig): 
        self.config = config
        self.n_qubits = config.quantum.n_qubits
        self.n_layers = config.quantum.n_layers
        self.encoding = config.quantum.encoding
        self._dev = qml.device('lightning.qubit', wires=self.n_qubits)
        self._params = None
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self._dev, diff_method='adjoint')
        def circuit(x, params): 
            if self.encoding == 'angle':
                angle_encoding_layer(x * pnp.pi, range(self.n_qubits))
            else:
                n_amplitudes = 2 ** self.n_qubits
                x_padded = pnp.pad(x, (0, max(0, n_amplitudes - len(x))))[:n_amplitudes]
                amplitude_encoding_layer(x_padded, range(self.n_qubits))
            for layer in range(self.n_layers):
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
                for i in range(self.n_qubits):
                    qml.RY(params[layer][i], wires=i)
            return qml.expval(qml.PauliZ(0))
        return circuit

    def _cost(self, params, X, y):
        preds = pnp.array([self._circuit(x, params) for x in X])
        probs = pnp.clip((preds + 1) / 2, 1e-7, 1 - 1e-7)
        pos_weight = (y == 0).sum() / (y == 1).sum()
        weights = pnp.where(y == 1, pos_weight, 1.0)
        return -pnp.mean(weights * (y * pnp.log(probs) + (1 - y) * pnp.log(1 - probs)))
    
    def _get_optimizer(self): 
        opts = {
            'adam': qml.AdamOptimizer, 
            'gradient_descent': qml.GradientDescentOptimizer, 
            'adagrad': qml.AdagradOptimizer,
            'momentum': qml.MomentumOptimizer,
        }
        cls = opts.get(self.config.quantum.optimizer, qml.AdamOptimizer)
        return cls(stepsize=self.config.quantum.step_size)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "VQCChurnModel":
        max_samples = getattr(self.config.quantum, 'max_train_samples', 300)
        X, y = X[:max_samples], y[:max_samples]
        print(f"VQC: {self.n_qubits} qubits, {self.n_layers} layers, {len(X)} samples")

        self._params = pnp.random.randn(self.n_layers, self.n_qubits, requires_grad=True) * 0.1
        opt = self._get_optimizer()

        for step in range(self.config.quantum.n_steps):
            self._params, cost = opt.step_and_cost(
                    lambda p: self._cost(p, X, y), self._params
            )
            if (step + 1) % 10 == 0:
                print(f" step {step + 1}/{self.config.quantum.n_steps} - cost: {cost:.4f}")
        return self 

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.array([float((self._circuit(x, self._params) + 1) / 2) for x in X])
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

