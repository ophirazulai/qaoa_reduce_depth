import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from qaqa_reduce_depth.compiler import configure_threads  # noqa: E402

configure_threads()


@pytest.fixture(autouse=True)
def worker_imports(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))


@pytest.fixture
def inputs(tmp_path):
    from qiskit import QuantumCircuit, qpy

    source = tmp_path / "inputs"
    source.mkdir()
    first = QuantumCircuit(6, name="nonlocal")
    first.h(0)
    first.rz(0.37, 5)
    first.cx(0, 5)
    first.cx(1, 4)
    first.cx(2, 5)
    first.rx(0.19, 1)
    second = QuantumCircuit(6, name="parallel")
    second.cx(0, 1)
    second.cx(2, 3)
    with (source / "bundle.qpy").open("wb") as stream:
        qpy.dump([first, second], stream)
    return source
