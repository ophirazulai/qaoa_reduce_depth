# QAOA reduce depth

Try different random seeds and SABRE wait-time weights to find a transpilation
with lower two-qubit depth. Each circuit is also run with a baseline that has
wait time disabled. The best result is saved as a QPY file, along with its seed,
weight, depth (D2), and two-qubit gate count (N2).

## Install

You'll need Python 3.12+, uv, Git, a C/C++ compiler, and Rust (`cargo` and `rustc`).
From the project directory:

```bash
uv venv --python 3.12
source .venv/bin/activate
CARGO_BUILD_JOBS=4 uv pip install -e .
```

This builds the pinned Qiskit fork with wait-time support, which can take a few
minutes. Keep this version installed; the regular PyPI release won't work here.
No IBM account is needed.

## Run

Put your `.qpy` files in `circuits/`, then run:

```bash
source .venv/bin/activate
qaoa-reduce-depth run
```

The default run uses FakeBoston, 30 seeds, six wait-time weights plus baseline,
and five worker processes: **210 transpilation trials per input circuit**.
It uses optimization level 3 and 20 internal SABRE layout/swap trials.

## How the search works

The program recompiles the same circuit with different settings and keeps the
result with the lowest two-qubit depth (D2).

Two settings change: the **seed** makes SABRE explore different qubit placements
and routing choices; the **weight** controls how strongly wait time affects those
choices.

For each input circuit, the default search does this:

1. With seed `0`, compile seven versions: one with wait time disabled (baseline),
   then one for each weight: `0.00001`, `0.00005`, `0.0001`, `0.00025`, `0.0005`,
   and `0.001`.
2. Repeat those seven settings with seeds `1` through `29`. That gives
   `30 × 7 = 210` compiled versions, each starting from the original input.
3. Compare their D2 values and save the version with the smallest D2. If several
   have the same D2, prefer the one with fewer two-qubit gates (N2).

Five worker processes share this work. The report also compares the best baseline
with the best wait-time result. The saved circuit is the best of these 210 trials;
a different seed or weight could still produce a better result.

## Results

Open `output/report.md` for the winners and baseline comparison.

- `best/`: the selected QPY circuit for each input.
- `results.json`: winners and comparisons in JSON.
- `trials.json`: measurements, timings, and errors for every trial.

The winner has the lowest D2; ties favor fewer two-qubit gates (N2).
D2 counts two-qubit gates along the longest dependency path. The report's
wait-time/baseline ratios are below 1 when wait time improves the result.
The baseline uses decay; wait-time trials use wait time without decay.
An input gets a winner only when all its trials succeed.



## Run the heuristic directly

With the environment above activated, this runs one transpilation using the
wait-time heuristic. Change `weight` to adjust its contribution to SABRE scoring.

```python
from qiskit import QuantumCircuit
from qiskit._accelerate.sabre import Heuristic, SetScaling
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SabreLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeBoston

circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cx(0, 2)
circuit.cx(2, 1)

backend = FakeBoston()
weight = 0.0005
seed = 7
heuristic = (
    Heuristic(attempt_limit=10 * backend.num_qubits)
    .with_basic(1.0, SetScaling.Constant)
    .with_lookahead([0.5 / backend.num_qubits], SetScaling.Constant)
    .with_wait_time(weight)
)
manager = generate_preset_pass_manager(
    backend=backend, optimization_level=3, seed_transpiler=seed,
    routing_method="none",  # SabreLayout below also performs routing.
)
manager.layout = PassManager([
    SabreLayout(
        backend.target, heuristic=heuristic, seed=seed,
        max_iterations=4, layout_trials=20, swap_trials=20,
    ),
])
transpiled = manager.run(circuit)
print(transpiled.count_ops())
```

The weight applies to both layout search and routing.

