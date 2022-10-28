'''
File: opflow.py
Project: dqnn
Created Date: Fri Oct 28 2022 03:27:07 AM +09:00
Author: Siheon Park (sihoney97@kaist.ac.kr)
-----
Last Modified: Fr Oct 28 2022 06:58:58 AM +09:00
Modified By: Siheon Park (sihoney97@kaist.ac.kr)
-----
Copyright (c) 2022 Siheon Park
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import numpy as np
from typing import Union, List
from collections import deque

from qiskit.quantum_info import random_unitary
from qiskit.quantum_info import (
    DensityMatrix,
    Statevector,
    partial_trace,
    state_fidelity,
)
from qiskit.quantum_info.states.quantum_state import Operator
from qiskit.extensions import HamiltonianGate


class OpflowDQNN(object):
    def __init__(self, qnnArch: List[int], epsilon: float = 0.1, lamda: float = 1):
        """
        Deep Quantum Neural Network.

        Args:
            qnnArch: list of integers representing the architecture of the DQNN. e.g. [2, 3, 2] represents 2q input, 3q
            hidden, 2q output
        """
        self.qnnArch = qnnArch
        self.epsilon = epsilon
        self.lamda = lamda

    def makeRandomUnitaries(self, seed: int = None):
        """Randomly initialize the unitaries of the DQNN
        Args:
            seed: random seed
        """
        unitaries = []
        for l in range(len(self.qnnArch) - 1):
            unitaries.append(
                np.empty(
                    shape=(
                        self.qnnArch[l + 1],
                        2 ** (self.qnnArch[l] + 1),
                        2 ** (self.qnnArch[l] + 1),
                    ),
                    dtype=np.complex128,
                )
            )
            for j in range(self.qnnArch[l + 1]):
                unitaries[l][j, ::] = random_unitary(
                    2 ** (self.qnnArch[l] + 1),
                    seed=(seed + j) if seed is not None else None,
                ).data
        return unitaries

    def applyLayerChannel(
        self, unitaries: List[np.ndarray], l: int, inputState: DensityMatrix
    ):

        numInputQubits = self.qnnArch[l]
        numOutputQubits = self.qnnArch[l + 1]
        inputWires = list(range(numInputQubits))
        outputWires = list(range(numInputQubits, numInputQubits + numOutputQubits))
        layerUnitaries = unitaries[l]
        # inputState = DensityMatrix(inputState)
        inputState = inputState.expand(
            DensityMatrix.from_int(0, dims=2**numOutputQubits)
        )
        # apply layer unitaries
        for j in range(numOutputQubits):
            # assert Operator(layerUnitaries[j, :, :]).is_unitary(), f"`unitaries[{l}][{j}, :, :]` is not unitary"
            inputState = inputState.evolve(
                Operator(layerUnitaries[j, :, :]), qargs=inputWires + [outputWires[j]]
            )
        return partial_trace(inputState, qargs=inputWires)

    def applyAdjointLayerChannel(
        self, unitaries: List[np.ndarray], l: int, outputState: DensityMatrix
    ):
        numInputQubits = self.qnnArch[l]
        numOutputQubits = self.qnnArch[l + 1]
        inputWires = list(range(numInputQubits))
        outputWires = list(range(numInputQubits, numInputQubits + numOutputQubits))
        layerUnitaries = unitaries[l]
        # outputState = DensityMatrix(outputState)
        outputState = DensityMatrix(np.eye(2**numInputQubits)).expand(outputState)
        projOp = Operator(np.eye(2**numInputQubits)).expand(
            DensityMatrix.from_int(0, dims=2**numOutputQubits).to_operator()
        )
        # apply adjoing layer unitaries
        for j in range(numOutputQubits - 1, -1, -1):
            # assert Operator(layerUnitaries[j, :, :]).is_unitary(), f"`unitaries[{l}][{j}, :, :]` is not unitary"
            outputState = outputState.evolve(
                Operator(layerUnitaries[j, :, :]).adjoint(),
                qargs=inputWires + [outputWires[j]],
            )
        return partial_trace(
            DensityMatrix(projOp @ outputState.to_operator()), outputWires
        )

    def feedforward(
        self,
        unitaries: List[np.ndarray],
        inputs: List[Union[Statevector, DensityMatrix, np.ndarray]],
    ):
        """Forward pass of the DQNN
        Args:
            unitaries: list of unitary parameters per layer
            inputs: array of input states of shape (batch_size, input_dim, input_dim) or (batch_size, input_dim)

        Returns:
            outputs: deque of "array of layer states of shape (batch_size, output_dim, output_dim)" per layer
        """
        forward_process = []
        for quantum_state in inputs:
            layer_output = deque()
            quantum_state = DensityMatrix(quantum_state)
            layer_output.append(quantum_state)
            for l in range(len(self.qnnArch) - 1):
                quantum_state = self.applyLayerChannel(unitaries, l, quantum_state)
                layer_output.append(quantum_state)
            forward_process.append(layer_output)

        layer_outputs = []
        for l in range(len(self.qnnArch)):
            layer_outputs.append(
                np.stack([forward_process[x][l] for x in range(len(inputs))])
            )
        return layer_outputs

    def backpropagation(
        self,
        unitaries: List[np.ndarray],
        targets: List[Union[Statevector, DensityMatrix, np.ndarray]],
    ):
        """Backward pass of the DQNN
        Args:
            unitaries: list of unitary parameters per layer
            targets: array of target states of shape (batch_size, output_dim, output_dim) or (batch_size, output_dim)

        Returns:
            outputs: deque of "array of layer states of shape (batch_size, input_dim, input_dim)" per layer
        """
        backward_process = []
        for povm_measurement in targets:
            layer_output = deque()
            povm_measurement = DensityMatrix(povm_measurement)
            layer_output.appendleft(povm_measurement)
            for l in range(len(self.qnnArch) - 2, -1, -1):
                povm_measurement = self.applyAdjointLayerChannel(
                    unitaries, l, povm_measurement
                )
                layer_output.appendleft(povm_measurement)
            backward_process.append(layer_output)

        layer_outputs = []
        for l in range(len(self.qnnArch)):
            layer_outputs.append(
                np.stack([backward_process[x][l] for x in range(len(targets))])
            )
        return layer_outputs

    def step(
        self,
        unitaries: List[np.ndarray],
        inputs: List[Union[Statevector, DensityMatrix, np.ndarray]],
        targets: List[Union[Statevector, DensityMatrix, np.ndarray]],
    ):
        """Perform a single step of the DQNN
        Args:
            unitaries: list of unitary parameters per layer. This will be updated in-place.
            inputs: array of input states of shape (batch_size, input_dim, input_dim) or (batch_size, input_dim)
            targets: array of target states of shape (batch_size, output_dim, output_dim) or (batch_size, output_dim)

        Returns:
            outputstate: array of (previous) output states of shape (batch_size, output_dim, output_dim) or (batch_size, output_dim)
        Note:
            This update is only suitable for state fidelity cost function.
        """
        # feedforward & backpropagation
        feedforward_results = self.feedforward(unitaries, inputs)
        backpropagation_results = self.backpropagation(unitaries, targets)

        for l in range(len(self.qnnArch) - 1):
            xTrMmatrices = []
            numInputQubits = self.qnnArch[l]
            numOutputQubits = self.qnnArch[l + 1]
            inputWires = list(range(numInputQubits))
            outputWires = list(range(numInputQubits, numInputQubits + numOutputQubits))
            layerUnitaries = unitaries[l]
            layerInputStates = feedforward_results[l]
            layerOutputStates = backpropagation_results[l + 1]
            # make update matrices
            for layerInputState, layerOutputState in zip(
                layerInputStates, layerOutputStates
            ):
                Amatrices = deque()
                Bmatrices = deque()
                TrMmatrices = list()
                astate = DensityMatrix(layerInputState).expand(
                    DensityMatrix.from_int(0, dims=2**numOutputQubits)
                )
                bstate = DensityMatrix(np.eye(2**numInputQubits)).expand(
                    DensityMatrix(layerOutputState)
                )
                for j in range(numOutputQubits):
                    astate = astate.evolve(
                        layerUnitaries[j, :, :], qargs=inputWires + [outputWires[j]]
                    )
                    Amatrices.append(astate.data)
                for j in range(numOutputQubits - 1, -1, -1):
                    Bmatrices.appendleft(bstate.data)
                    if j != 0:
                        bstate = bstate.evolve(
                            Operator(layerUnitaries[j, :, :]).adjoint(),
                            qargs=inputWires + [outputWires[j]],
                        )
                for j in range(numOutputQubits):
                    # assert np.isclose(np.trace(Amatrices[j] @ Bmatrices[j]) , np.mean(measout[:, x])), "A or B matrix is incorrect"
                    TrMmatrices.append(
                        partial_trace(
                            self._commutator(Amatrices[j], Bmatrices[j]),
                            qargs=outputWires[:j] + outputWires[j + 1 :],
                        ).data
                    )
                    # assert TrMmatrices[j].shape == layerUnitaries[j].shape, f"TrM matrix is incorrect ({TrMmatrices[-1].shape} != {layerUnitaries.shape})"
                xTrMmatrices.append(TrMmatrices)
            xTrMmatrices = np.stack(xTrMmatrices)
            Kmatrices = (
                1j * (2**numInputQubits) / self.lamda * xTrMmatrices.mean(axis=0)
            )
            # assert len(xTrMmatrices)==len(targets)
            # replace unitaries
            for j in range(numOutputQubits):
                # assert Operator(Kmatrices[j]) == Operator(Kmatrices[j]).adjoint()
                unitaries[l][j, :, :] = np.matmul(
                    HamiltonianGate(Kmatrices[j], time=-self.epsilon).to_matrix(),
                    unitaries[l][j, :, :],
                )
                # assert Operator(unitaries[l][j, :, :]).is_unitary()
        return self.cost(feedforward_results[-1], targets)

    @staticmethod
    def cost(outputs: np.ndarray, targets: np.ndarray):
        """Calculate the cost of the DQNN
        Args:
            outputs: array of output states of shape (batch_size, output_dim, output_dim) or (batch_size, output_dim)
            targets: array of target states of shape (batch_size, output_dim, output_dim) or (batch_size, output_dim)

        Returns:
            cost: float
        """
        # assert len(outputs) == len(targets)
        cost = 0
        for output, target in zip(outputs, targets):
            cost += state_fidelity(DensityMatrix(output), DensityMatrix(target))
        return cost / len(targets)

    @staticmethod
    def _commutator(A, B):
        # assert A.shape == B.shape
        # assert len(A.shape) == 2
        return np.matmul(A, B) - np.matmul(B, A)

    def __call__(self, unitaries, inputs):
        return self.feedforward(unitaries, inputs)[-1]


if __name__ == "__main__":
    help(OpflowDQNN)
