import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import CheckMap, SabreLayout, SabreSwap

from qaoa_reduce_depth.compiler import make_manager, measure, rank, resolve_target
from tests.support import backend


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("weight", [None, 0.0, 0.00001, 0.001])
def test_integrated_layout_and_routing_preserve_semantics(level, weight):
    target = backend()
    manager = make_manager(target, 7, weight, level, trials=2)
    (layout,) = manager.layout.to_flow_controller().tasks
    assert isinstance(layout, SabreLayout)
    assert not layout.skip_routing
    assert layout.swap_trials == layout.layout_trials == 2
    assert layout.max_iterations == (1, 2, 2, 4)[level]
    heuristic = repr(layout.heuristic)
    if weight is None:
        assert "wait_time=None" in heuristic
        assert "decay=DecayHeuristic" in heuristic
    else:
        assert f"wait_time=WaitTimeHeuristic(weight={weight}," in heuristic
        assert "decay=None" in heuristic
    circuit = QuantumCircuit(6)
    circuit.h(0)
    circuit.rz(0.37, 5)
    circuit.cx(0, 5)
    circuit.cx(1, 4)
    circuit.cx(5, 1)
    # A triangle cannot fit the line backend under any fixed layout.
    circuit.cx(1, 0)
    executed = []
    output = manager.run(circuit, callback=lambda **kwargs: executed.append(kwargs["pass_"]))
    assert any(isinstance(task, SabreLayout) for task in executed)
    assert not any(isinstance(task, SabreSwap) for task in executed)
    assert manager.property_set["final_layout"] is not None
    assert Operator.from_circuit(output).equiv(Operator(circuit))
    check = CheckMap(target.coupling_map)
    check(output)
    assert check.property_set["is_swap_mapped"]
    assert measure(output)["n2"] == output.count_ops().get("cx", 0)
    assert output == make_manager(target, 7, weight, level, trials=2).run(circuit)


def test_integrated_layout_preserves_parameters_and_measurement_mapping():
    theta = Parameter("theta")
    circuit = QuantumCircuit(6, 6)
    circuit.h(0)
    circuit.rz(theta, 0)
    circuit.cx(0, 5)
    circuit.cx(5, 1)
    circuit.barrier()
    circuit.cx(1, 0)
    classical_order = [2, 0, 1, 5, 3, 4]
    circuit.measure(range(6), classical_order)

    output = make_manager(backend(), 7, 0.0001, trials=2).run(circuit)
    assert output.parameters == {theta}
    measured = {
        output.find_bit(item.clbits[0]).index: output.find_bit(item.qubits[0]).index
        for item in output.data if item.operation.name == "measure"
    }
    final_indices = output.layout.final_index_layout()
    assert measured == {classical: final_indices[qubit]
                        for qubit, classical in enumerate(classical_order)}
    original = circuit.assign_parameters({theta: 0.37}).remove_final_measurements(inplace=False)
    compiled = output.assign_parameters({theta: 0.37}).remove_final_measurements(inplace=False)
    assert Operator.from_circuit(compiled).equiv(Operator(original))


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
