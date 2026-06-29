import numpy as np 
import pennylane as qml 

def build_kernel(n_qubits: int): 
    dev = qml.device('lightning.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1 * np.pi, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2 * np.pi, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def kernel(x1, x2): 
        return float(kernel_circuit(x1, x2)[0])
    
    return kernel
