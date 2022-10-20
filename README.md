# DQNNforQOS

This repo is for learning quantum open system with deep quantum neural network.

## Summaries & Presentations

- [Slide for *Data-Driven Time Propagation of Quantum Systems with Neural Networks*](https://www.slideshare.net/secret/GwG1Qh6mUSZ0Q2)
- [Slide for *Training deep quantum neural networks*](https://www.slideshare.net/secret/zijNV2VvA9OcHk)

## Tutorials

- `DQNN_pennylane.ipynb`: Basic implementation of DQNN with pennylane, proposed in *Beer et al. 2020 <https://doi.org/10.1038/s41467-020-14454-2>*
- `DQNN_qiskit.ipynb`: Basic implementation of DQNN with qiskit, proposed in *Beer et al. 2020 <https://doi.org/10.1038/s41467-020-14454-2>*
  - `class OpflowDQNN(object)` is the DQNN that utilizes `qiskit.quantum_info` to construct and update unitaries.
  - `class CircuitDQNN(object)` is the DQNN that utilizes `qiskit.circuit` to construct channels. TODO: working progress.

## Published Papers & Preprints

## References

- Previous implementations: <https://github.com/qigitphannover/DeepQuantumNeuralNetworks>
- Parent repo: <https://github.com/QAMP-Fall-22-Project34/OQS-dynamics>
