import pennylane as qml
import numpy as np 

def angle_encoding_layer(x, wires): 
    for i, wire in enumerate(wires): 
        qml.RY(x[i], wires=wire)

def amplitude_encoding_layer(x, wires):
    qml.AmplitudeEmbedding(x, wires=wires, normalize=True)

