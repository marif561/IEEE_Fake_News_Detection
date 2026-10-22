"""Circuit correctness and training checks (skipped without PyTorch)."""
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from hqnn.quantum import HQNN, simulate  # noqa: E402
from hqnn.train import main as train_main  # noqa: E402

S = Path(__file__).resolve().parents[1] / "data" / "sample"


def test_angle_embedding_single_qubit():
    x = torch.tensor([[0.0], [math.pi / 3], [math.pi]])
    z = simulate(x, torch.zeros(0, 1))
    assert torch.allclose(z[:, 0], torch.cos(x[:, 0]), atol=1e-5)  # <Z> = cos(x) after RY(x)


def test_cnot_correlates_qubits():
    x = torch.tensor([[0.7, 1.1]])
    z = simulate(x, torch.zeros(1, 2))  # RX(0) = identity, then CNOT(0, 1)
    assert torch.allclose(z[0, 0], torch.cos(x[0, 0]), atol=1e-5)
    assert torch.allclose(z[0, 1], torch.cos(x[0, 0]) * torch.cos(x[0, 1]), atol=1e-5)


def test_outputs_bounded_and_differentiable():
    model = HQNN(n_qubits=6, n_layers=2)
    x = torch.rand(5, 6) * math.pi
    q = model.quantum(x)
    assert q.shape == (5, 6) and q.abs().max() <= 1 + 1e-5
    model(x).sum().backward()
    assert model.quantum.weights.grad is not None and model.quantum.weights.grad.abs().sum() > 0


def test_matches_pennylane():
    qml = pytest.importorskip("pennylane")
    torch.manual_seed(0)
    x = torch.rand(4, 6) * math.pi
    w = torch.rand(2, 6) * 2 * math.pi
    dev = qml.device("default.qubit", wires=6)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(6), rotation="Y")
        qml.BasicEntanglerLayers(weights, wires=range(6), rotation=qml.RX)
        return [qml.expval(qml.PauliZ(i)) for i in range(6)]

    ref = torch.stack([torch.stack([torch.as_tensor(v) for v in circuit(xi, w)]) for xi in x]).float()
    assert torch.allclose(simulate(x, w), ref, atol=1e-4)


@pytest.mark.parametrize("backend", ["torch", "pennylane"])
def test_cross_validation_end_to_end(tmp_path, backend):
    if backend == "pennylane":
        pytest.importorskip("pennylane")
    train_main(["--data", str(S / "labelled.csv"), "--name", f"sample_{backend}", "--folds", "2", "--epochs", "2",
                "--backend", backend, "--baselines", "--out", str(tmp_path)])
    import json
    res = json.loads((tmp_path / f"sample_{backend}" / "results.json").read_text())
    assert set(res["hqnn"]) == {"accuracy", "precision", "recall", "f1"} and len(res["folds"]) == 2
    assert len(res["curves"][0]) == 2 and "lr_tfidf" in res["baselines"]
