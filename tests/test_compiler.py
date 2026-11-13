import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import CheckMap, SabreLayout, SabreSwap

from qaqa_reduce_depth.compiler import make_manager, measure, rank, resolve_target, sabre_passes
from tests.support import backend


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("weight", [None, 0.0, 0.00001, 0.001])
def test_both_passes_configured_and_semantics_preserved(level, weight):
    target = backend()
    manager = make_manager(target, 7, weight, level, trials=2)
    passes = list(sabre_passes(manager))
    assert {type(task) for task in passes} == {SabreLayout, SabreSwap}
    for task in passes:
        heuristic = repr(task.heuristic)
        if weight is None:
            assert "wait_time=None" in heuristic
            assert "decay=DecayHeuristic" in heuristic
        else:
            assert f"wait_time=WaitTimeHeuristic(weight={weight}," in heuristic
            assert "decay=None" in heuristic
        if isinstance(task, SabreLayout):
            assert task.swap_trials == task.layout_trials == 2
        else:
            assert task.trials == 2
    circuit = QuantumCircuit(6)
    circuit.h(0)
    circuit.rz(0.37, 5)
    circuit.cx(0, 5)
    circuit.cx(1, 4)
    circuit.cx(5, 1)
    output = manager.run(circuit)
    assert Operator.from_circuit(output).equiv(Operator(circuit))
    check = CheckMap(target.coupling_map)
    check(output)
    assert check.property_set["is_swap_mapped"]
    assert measure(output)["n2"] == output.count_ops().get("cx", 0)
    assert output == make_manager(target, 7, weight, level, trials=2).run(circuit)


def test_depth_counts_parallel_layers_and_barrier_dependencies():
    circuit = QuantumCircuit(4)
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    assert measure(circuit) == {"d2": 1, "n2": 2}
    circuit.cx(0, 1)
    circuit.barrier(1, 2)
    circuit.cx(2, 3)
    assert measure(circuit) == {"d2": 3, "n2": 4}
    assert measure(QuantumCircuit(0)) == {"d2": 0, "n2": 0}


def test_measure_rejects_dynamic_and_non_native_output():
    circuit = QuantumCircuit(3, 1)
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.cx(0, 1)
    with pytest.raises(ValueError, match="static"):
        measure(circuit)
    circuit = QuantumCircuit(3)
    circuit.ccx(0, 1, 2)
    with pytest.raises(ValueError, match="multi-qubit"):
        measure(circuit)


def test_rank_prioritizes_d2_before_n2_and_has_stable_ties():
    baseline = {"d2": 2, "n2": 10, "weight": None, "seed": 1}
    assert rank(baseline) < rank(dict(baseline, d2=3, n2=3))
    assert rank(dict(baseline, n2=9)) < rank(baseline)
    assert rank(baseline) < rank(dict(baseline, weight=0.0001, seed=0))
    assert rank(dict(baseline, weight=0.0001)) < rank(dict(baseline, weight=0.001))
    assert rank(dict(baseline, seed=0)) < rank(baseline)


def test_fake_boston_and_safe_factory_resolution():
    assert resolve_target("FakeBoston()").name == "fake_boston"
    with pytest.raises(ValueError, match="module:factory"):
        resolve_target("__import__('os').system('echo oops')")
