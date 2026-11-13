import time


def backend():
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit.transpiler import CouplingMap

    return GenericBackendV2(6, coupling_map=CouplingMap.from_line(6),
                            basis_gates=["rz", "sx", "x", "cx"], seed=17)


def slow_backend():
    # Workers resolve the target once per batch, so every batch takes long
    # enough for a test to interrupt the sweep while most are still pending.
    time.sleep(0.5)
    return backend()
