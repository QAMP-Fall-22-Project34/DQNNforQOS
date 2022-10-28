'''
File: dataset.py
Project: dqnn
Created Date: Fri Oct 28 2022 07:03:34 AM +09:00
Author: Siheon Park (sihoney97@kaist.ac.kr)
-----
Last Modified: Fr Oct 28 2022 07:05:34 AM +09:00
Modified By: Siheon Park (sihoney97@kaist.ac.kr)
-----
Copyright (c) 2022 Siheon Park
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import numpy as np
from qiskit.quantum_info import random_statevector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.states.quantum_state import Operator


def generate_random_training_data(numSamples:int, numInputQubits:int, seed:int=None):
    """ Generate random training data 
    Args:
        numSamples: number of samples to generate
        numInputQubits: number of input qubits
    Returns:
        trainingData: array of input states of shape (batch_size, 2**numInputQubits)
    """
    trainingData = np.empty(shape=(numSamples, 2**numInputQubits), dtype=np.complex128)
    for i in range(numSamples):
        trainingData[i, :] = random_statevector(2**numInputQubits, seed=(seed+i) if seed is not None else None).data
    return trainingData

def generate_target_from_unitary(channel:Operator, trainingData:list):
    """ Generate target states from a channel
    Args:
        channel: quantum channel or unitary to apply to the training data
        trainingData: list of input states
    Returns:
        targets: array of target states of shape (batch_size, output_dim) or (batch_size, output_dim, output_dim)
    """
    targetStates = []
    for inputState in trainingData:
        inputState = Statevector(inputState)
        targetStates.append(
            inputState.evolve(channel).data
        )
    targetStates = np.stack(targetStates)
    return targetStates