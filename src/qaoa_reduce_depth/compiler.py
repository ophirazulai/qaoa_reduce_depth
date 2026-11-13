"""Explicit SABRE configuration for the pinned wait-time branch."""

from __future__ import annotations

import importlib
import inspect
import math
import os
import re

QISKIT_REVISION = "9dedcf6f64a66344cb31364e846f71d020adcf9d"
DEFAULT_TARGET = "qiskit_ibm_runtime.fake_provider:FakeBoston"
DEFAULT_WEIGHTS = (0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001)
DEFAULT_SABRE_TRIALS = 20
SABRE_ITERATIONS_BY_LEVEL = {0: 1, 1: 2, 2: 2, 3: 4}


def configure_threads():
    # Set before importing Qiskit/numpy in the coordinator and spawned workers.
    for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS", "QISKIT_NUM_PROCS"):
        os.environ[name] = "1"
    os.environ["QISKIT_PARALLEL"] = "FALSE"


def resolve_target(spec):
    if spec in {"FakeBoston", "FakeBoston()"}:
        spec = DEFAULT_TARGET
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", spec):
        raise ValueError("target must be an importable module:factory (or FakeBoston)")
    module, factory = spec.split(":")
    backend = getattr(importlib.import_module(module), factory)()
    if not hasattr(backend, "target") or not backend.target.num_qubits:
        raise ValueError("target factory must return a BackendV2 with a nonempty target")
    return backend


def make_manager(backend, seed, weight, optimization_level=3, trials=DEFAULT_SABRE_TRIALS):
    from qiskit._accelerate.sabre import Heuristic, SetScaling
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import SabreLayout
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    if not hasattr(Heuristic, "with_wait_time"):
        raise RuntimeError("Qiskit lacks wait-time support; install this project's pinned fork")
    if "heuristic" not in inspect.signature(SabreLayout.__init__).parameters:
        raise RuntimeError("Qiskit lacks SabreLayout.heuristic; install the pinned fork")
    if weight is not None and (
        isinstance(weight, bool) or not isinstance(weight, (int, float))
        or not math.isfinite(weight) or weight < 0
    ):
        raise ValueError("weight must be finite and nonnegative")
    width = backend.target.num_qubits
    heuristic = (
        Heuristic(attempt_limit=10 * width)
        .with_basic(1.0, SetScaling.Constant)
        .with_lookahead([0.5 / width], SetScaling.Constant)
    )
    # Baseline uses decay; weighted runs use wait time without decay.
    heuristic = (heuristic.with_decay(0.001, 5) if weight is None
                 else heuristic.with_wait_time(weight))
    manager = generate_preset_pass_manager(
        backend=backend, seed_transpiler=seed, optimization_level=optimization_level,
        routing_method="none",  # SabreLayout performs routing internally.
    )
    manager.layout = PassManager([
        SabreLayout(
            backend.target, heuristic=heuristic, seed=seed,
            # Preserve the pinned preset's search budget at each optimization level.
            max_iterations=SABRE_ITERATIONS_BY_LEVEL[optimization_level],
            layout_trials=trials, swap_trials=trials,
        ),
    ])
    return manager


def measure(circuit):
    """N2 counts two-qubit gates; D2 is their longest dependency path.

    Zero-cost operations, including barriers, still synchronize their wires.
    Dynamic control flow is deliberately rejected instead of guessed.
    """
    from qiskit.circuit import ControlFlowOp, Gate

    depths = dict.fromkeys([*circuit.qubits, *circuit.clbits], 0)
    n2 = 0
    for instruction in circuit.data:
        op = instruction.operation
        if isinstance(op, ControlFlowOp) or getattr(op, "condition", None) is not None:
            raise ValueError("D2 measurement supports static circuits only")
        if len(instruction.qubits) > 2 and op.name != "barrier":
            raise ValueError(f"Unexpected multi-qubit output instruction: {op.name}")
        cost = int(isinstance(op, Gate) and len(instruction.qubits) == 2)
        wires = (*instruction.qubits, *instruction.clbits)
        depth = max((depths[wire] for wire in wires), default=0) + cost
        for wire in wires:
            depths[wire] = depth
        n2 += cost
    return {"d2": max(depths.values(), default=0), "n2": n2}


def rank(row):
    """Prefer D2, then N2, baseline, smaller weight, smaller seed."""
    return (row["d2"], row["n2"], row["weight"] is not None,
            row["weight"] or 0.0, row["seed"])
