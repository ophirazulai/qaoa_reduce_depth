def backend():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    return GenericBackendV2(6, coupling_map=CouplingMap.from_line(6),
                            basis_gates=["rz", "sx", "x", "cx"], seed=17)
