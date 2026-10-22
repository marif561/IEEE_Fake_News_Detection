"""The quantum layer: angle embedding (RY) + trainable entangling layers + ⟨Z⟩ read-out.

Circuit on ``n`` qubits, for an input ``x ∈ R^n`` and weights ``θ ∈ R^{L×n}``:

    for i:            RY(x_i) on qubit i                         (angle embedding)
    for each layer l: RX(θ_{l,i}) on qubit i, then CNOT ring    (BasicEntanglerLayers)
                      CNOT(0,1), CNOT(1,2), …, CNOT(n-1,0)
    output:           ⟨Z_i⟩ for every qubit                      (n values in [-1, 1])

Two interchangeable back-ends:
  * ``pennylane`` — PennyLane ``default.qubit`` wrapped as a ``TorchLayer`` (as typically used)
  * ``torch``     — a small exact state-vector simulator written in PyTorch (no extra
                    dependency, fully differentiable, fast for ≤ ~12 qubits)
Both compute the same circuit; the tests check that their outputs agree.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------- PyTorch state-vector simulator
def _apply_1q(state: torch.Tensor, gate: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    """state: (B, 2, …, 2) complex; gate: (B, 2, 2) or (2, 2)."""
    state = state.movedim(wire + 1, -1)
    if gate.dim() == 2:
        state = state @ gate.T
    else:
        shape = state.shape
        state = torch.einsum("bij,b...j->b...i", gate, state.reshape(shape[0], -1, 2)).reshape(shape)
    return state.movedim(-1, wire + 1)


def _cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
    s = state.clone()
    idx1 = [slice(None)] * state.dim()
    idx1[control + 1] = 1
    sub = s[tuple(idx1)]
    t = target if target < control else target - 1  # axis index after removing the control axis
    s[tuple(idx1)] = sub.flip(t + 1)
    return s


def _ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2).to(torch.complex64)


def _rx(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta / 2).to(torch.complex64)
    s = (-1j * torch.sin(theta / 2)).to(torch.complex64)
    return torch.stack([torch.stack([c, s], -1), torch.stack([s, c], -1)], -2)


def simulate(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Exact expectation values ⟨Z_i⟩ of the circuit above. x: (B, n); weights: (L, n)."""
    b, n = x.shape
    state = torch.zeros((b,) + (2,) * n, dtype=torch.complex64, device=x.device)
    state[(slice(None),) + (0,) * n] = 1.0
    for i in range(n):
        state = _apply_1q(state, _ry(x[:, i]), i, n)
    for layer in weights:
        for i in range(n):
            state = _apply_1q(state, _rx(layer[i]), i, n)
        if n == 2:
            state = _cnot(state, 0, 1)
        elif n > 2:
            for i in range(n):
                state = _cnot(state, i, (i + 1) % n)
    probs = state.abs() ** 2
    out = []
    for i in range(n):
        p = probs.movedim(i + 1, 1).reshape(b, 2, -1).sum(-1)
        out.append(p[:, 0] - p[:, 1])
    return torch.stack(out, 1)


class TorchQuantumLayer(nn.Module):
    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(n_layers, n_qubits) * 2 * math.pi)

    def forward(self, x):
        return simulate(x, self.weights)


# ---------------------------------------------------------------- PennyLane back-end
def pennylane_layer(n_qubits: int, n_layers: int, device: str = "default.qubit") -> nn.Module:
    import pennylane as qml

    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits), rotation=qml.RX)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, {"weights": (n_layers, n_qubits)})


def quantum_layer(n_qubits: int = 6, n_layers: int = 2, backend: str = "torch") -> nn.Module:
    if backend == "pennylane":
        return pennylane_layer(n_qubits, n_layers)
    if backend == "torch":
        return TorchQuantumLayer(n_qubits, n_layers)
    raise ValueError("backend must be 'torch' or 'pennylane'")


class HQNN(nn.Module):
    """Quantum layer (n qubits) followed by a linear layer to 2 classes (softmax via the loss)."""

    def __init__(self, n_qubits: int = 6, n_layers: int = 2, backend: str = "torch"):
        super().__init__()
        self.quantum = quantum_layer(n_qubits, n_layers, backend)
        self.fc = nn.Linear(n_qubits, 2)

    def forward(self, x):
        return self.fc(self.quantum(x).float())
