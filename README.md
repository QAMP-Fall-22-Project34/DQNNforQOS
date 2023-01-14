# DQNNforQOS

This repository features part of a Qiskit Advocate Mentorship Program Fall-22 program project titled: QML for reduced order density matrix time propagation qiskit-advocate/qamp-fall-22#34
The main function of this repository is learning quantum open system with deep quantum neural network.

## Summaries & Presentations

- [Slide for *Data-Driven Time Propagation of Quantum Systems with Neural Networks*](https://www.slideshare.net/secret/GwG1Qh6mUSZ0Q2)
- [Slide for *Training deep quantum neural networks*](https://www.slideshare.net/secret/zijNV2VvA9OcHk)

## Tutorials

- `DQNN_pennylane.ipynb`: Basic implementation of DQNN with pennylane, proposed in *Beer et al. 2020 <https://doi.org/10.1038/s41467-020-14454-2>*
- `DQNN_qiskit.ipynb`: Basic implementation of DQNN with qiskit, proposed in *Beer et al. 2020 <https://doi.org/10.1038/s41467-020-14454-2>*
  - `class OpflowDQNN(object)` is the DQNN that utilizes `qiskit.quantum_info` to construct and update unitaries.
  - `class CircuitDQNN(object)` is the DQNN that utilizes `qiskit.circuit` to construct channels. *Update: Deparicated*
- `dqnn`: Python package for DQNN simulation
  - `opflow`: Copy of `OpflowDQNN` class in `DQNN_qiskit.ipynb`
  - `dataset`: Copy of functions regarding dataset generation in `DQNN_qiskit.ipynb`

## Published Papers & Preprints

## References

- Previous implementations: <https://github.com/qigitphannover/DeepQuantumNeuralNetworks>
- Parent repo: <https://github.com/QAMP-Fall-22-Project34/OQS-dynamics>
