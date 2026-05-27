"""
soft_info.codes.noise
---------------------
Noise-channel injection for Stim circuits.

Models
------
phenomenological_css_circuit(H, L, T, p_data, p_meas, noise='X')
    Phenomenological CSS memory circuit: T noisy rounds + 1 perfect round.
    Noiseless CNOTs; independent single-qubit errors only.
      noise='X' : X_ERROR(p_data) on data  +  Hz stabilisers  →  Lz observable
      noise='Z' : Z_ERROR(p_data) on data  +  Hx stabilisers  →  Lx observable
    Measurement flip: X_ERROR(p_meas) on ancilla before M each noisy round.
    Detector coordinates [check_idx, round_idx] for mwpf spatial matching.

standard_depolarizing_noise_model(circuit, data_qubits, ...)
    Circuit-level depolarizing noise; inserts errors at every gate.
      after_clifford_depolarization    — DEPOLARIZE1/2 after single/two-qubit gates
      after_reset_flip_probability     — X_ERROR after R/RZ, Z_ERROR after RX
      before_measure_flip_probability  — X_ERROR before M/MR, Z_ERROR before MX
      before_round_data_depolarization — DEPOLARIZE1 on data qubits at every TICK

si1000_noise_model(circuit, data_qubits, probability)
    Superconducting SI-1000-inspired noise model (rates in units of p):
      Two-qubit gates (CNOT/CX)  : p depolarizing
      Single-qubit gates         : p/10 depolarizing
      Measure-reset (MR)         : p depol + 5p readout flip + 2p post-reset flip
      Measurement (M)            : p depol + 5p readout flip
      Idle (at every TICK)       : 2p DEPOLARIZE1 on data qubits
      First reset                : 2p reset flip + 2p idle on unused qubits

bravyi_noise_model(circuit, error_rate)
    Minimal Pauli noise (Bravyi et al. conventions):
      Two-qubit gates (CNOT/CX/CZ) : DEPOLARIZE2
      Measurement (M/MX/MZ)        : pre-measurement flip
      Reset (R/RX/RZ)              : post-reset flip
      Identity gates (I)           : DEPOLARIZE1
"""

import stim
import numpy as np

def phenomenological_css_circuit(
    H: np.ndarray,
    L: np.ndarray,
    T: int,
    p_data: float,
    p_meas: float,
    noise: str = 'X',
) -> stim.Circuit:
    if noise not in ('X', 'Z'):
        raise ValueError(f"noise must be 'X' or 'Z'; got {noise!r}")
    L = np.atleast_2d(L)
    m, n = H.shape
    data = list(range(n))
    anc  = list(range(n, n + m))

    circ = stim.Circuit()
    circ.append("R", data + anc)
    if noise == 'Z':
        circ.append("H", data)
    for t in range(T + 1):
            perfect = (t == T)
            if not perfect and p_data > 0:
                circ.append("X_ERROR" if noise == 'X' else "Z_ERROR", data, p_data)

            circ.append("R", anc)
            if noise == 'Z': circ.append("H", anc)
            for i in range(m):
                for q in np.where(H[i])[0]:
                    if noise == 'Z': circ.append("CNOT", [anc[i], int(q)])
                    else: circ.append("CNOT", [int(q), anc[i]])
            if noise == 'Z': circ.append("H", anc)

            if not perfect and p_meas > 0:
                circ.append("X_ERROR", anc, p_meas)
            circ.append("M", anc)

            for i in range(m):
                rec_cur = stim.target_rec(i - m)
                if t == 0:
                    circ.append("DETECTOR", [rec_cur], [float(i), float(t)])
                else:
                    circ.append("DETECTOR", [rec_cur, stim.target_rec(i - 2 * m)], [float(i), float(t)])
            circ.append("TICK")

    if noise == 'Z':
        circ.append("H", data)
    circ.append("M", data)
    for j, lop in enumerate(L):
        targets = [stim.target_rec(int(q) - n) for q in np.where(lop)[0]]
        circ.append("OBSERVABLE_INCLUDE", targets, j)

    return circ


def circuit_level_css_circuit(
    H: np.ndarray,
    L: np.ndarray,
    T: int,
    p: float,
    noise: str = 'X',
) -> stim.Circuit:
    if noise not in ('X', 'Z'):
        raise ValueError(f"noise must be 'X' or 'Z'; got {noise!r}")
    L = np.atleast_2d(L)
    m, n = H.shape
    data = list(range(n))
    anc  = list(range(n, n + m))

    def _one_round(t: int, perfect: bool) -> stim.Circuit:
        c = stim.Circuit()
        c.append("R", anc)
        if noise == 'Z':
            c.append("H", anc)
        for i in range(m):
            for q in np.where(H[i])[0]:
                if noise == 'Z':
                    c.append("CNOT", [anc[i], int(q)])
                else:
                    c.append("CNOT", [int(q), anc[i]])
        if noise == 'Z':
            c.append("H", anc)
        c.append("M", anc)
        c.append("TICK")
        for i in range(m):
            rec_cur = stim.target_rec(i - m)
            if t == 0:
                c.append("DETECTOR", [rec_cur], [float(i), float(t)])
            else:
                c.append("DETECTOR", [rec_cur, stim.target_rec(i - 2 * m)], [float(i), float(t)])
        return c

    clean = stim.Circuit()
    clean.append("R", data + anc)
    if noise == 'Z':
        clean.append("H", data)
    clean.append("TICK")
    for t in range(T):
        clean += _one_round(t, perfect=False)

    # final perfect round
    clean += _one_round(T, perfect=True)

    if noise == 'Z':
        clean.append("H", data)
    clean.append("M", data)
    for j, lop in enumerate(L):
        targets = [stim.target_rec(int(q) - n) for q in np.where(lop)[0]]
        clean.append("OBSERVABLE_INCLUDE", targets, j)

    return standard_depolarizing_noise_model(
        clean, data,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=0.0,
    )


def standard_depolarizing_noise_model(
        circuit: stim.Circuit,
        data_qubits: list[int],
        after_clifford_depolarization: float,
        after_reset_flip_probability: float,
        before_measure_flip_probability: float,
        before_round_data_depolarization: float,
        ancilla_qubits: list[int] | None = None,
) -> stim.Circuit:
    result = stim.Circuit()
    initialized = False
    ancillas = list(ancilla_qubits) if ancilla_qubits else []

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=standard_depolarizing_noise_model(
                    instruction.body_copy(), data_qubits,
                    after_clifford_depolarization, after_reset_flip_probability,
                    before_measure_flip_probability, before_round_data_depolarization,
                    ancilla_qubits=ancillas,
                )))
        elif instruction.name in ('R', 'RZ'):
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
            initialized = True
        elif instruction.name == 'RX':
            result.append(instruction)
            result.append('Z_ERROR', instruction.targets_copy(), after_reset_flip_probability)
            initialized = True
        elif instruction.name in ('H', 'S', 'S_DAG', 'H_DAG', 'C_XYZ', 'C_ZYX'):
            result.append(instruction)
            result.append('DEPOLARIZE1', instruction.targets_copy(), after_clifford_depolarization)
        elif instruction.name in ('CNOT', 'CX', 'CZ'):
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), after_clifford_depolarization)
        elif instruction.name == 'MR':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), after_reset_flip_probability)
        elif instruction.name == 'MX':
            result.append('Z_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), before_measure_flip_probability)
            result.append(instruction)
        elif instruction.name == 'TICK':
            if initialized and before_round_data_depolarization > 0:
                if data_qubits:
                    result.append('DEPOLARIZE1', data_qubits, before_round_data_depolarization)
                if ancillas:
                    result.append('DEPOLARIZE1', ancillas, before_round_data_depolarization)
            result.append(instruction)
        else:
            result.append(instruction)

    return result


def si1000_noise_model(
        circuit: stim.Circuit,
        data_qubits: list[int],
        probability: float
) -> stim.Circuit:
    all_qubits_in_circuit = set()
    for op in circuit.flattened():
        for t in op.targets_copy():
            if t.is_qubit_target:
                all_qubits_in_circuit.add(t.value)

    result = stim.Circuit()
    first_reset_seen = False

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=si1000_noise_model(instruction.body_copy(), data_qubits, probability)
            ))
            continue

        targets = [t.value for t in instruction.targets_copy() if t.is_qubit_target]

        if instruction.name == 'R' and not first_reset_seen:
            result.append(instruction)
            result.append('X_ERROR', targets, 2 * probability)
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
            first_reset_seen = True
        elif instruction.name in ['H', 'S', 'X', 'Y', 'Z', 'S_DAG', 'H_DAG']:
            result.append(instruction)
            result.append('DEPOLARIZE1', targets, probability / 10)
        elif instruction.name in ['CNOT', 'CX']:
            result.append(instruction)
            result.append('DEPOLARIZE2', targets, probability)
        elif instruction.name == 'MR':
            result.append('DEPOLARIZE1', targets, probability)
            result.append('X_ERROR', targets, 5 * probability)
            result.append(instruction)
            result.append('X_ERROR', targets, 2 * probability)
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
        elif instruction.name == 'M':
            result.append('DEPOLARIZE1', targets, probability)
            result.append('X_ERROR', targets, 5 * probability)
            result.append(instruction)
            idle_qubits = list(all_qubits_in_circuit - set(targets))
            if idle_qubits:
                result.append('DEPOLARIZE1', idle_qubits, 2 * probability)
        elif instruction.name == 'TICK':
            result.append(instruction)
            if first_reset_seen and data_qubits:
                result.append('DEPOLARIZE1', data_qubits, probability * 2)
        else:
            result.append(instruction)

    return result


def bravyi_noise_model(
        circuit: stim.Circuit,
        error_rate: float
) -> stim.Circuit:
    result = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(stim.CircuitRepeatBlock(
                repeat_count=instruction.repeat_count,
                body=bravyi_noise_model(instruction.body_copy(), error_rate)
            ))
        elif instruction.name in ('R', 'RZ'):
            result.append(instruction)
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
        elif instruction.name == 'RX':
            result.append(instruction)
            result.append('Z_ERROR', instruction.targets_copy(), error_rate)
        elif instruction.name == 'MX':
            result.append('Z_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        elif instruction.name == 'MZ':
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        elif instruction.name == 'I':
            result.append(instruction)
            result.append('DEPOLARIZE1', instruction.targets_copy(), error_rate)
        elif instruction.name in ['CNOT', 'CX', 'CZ']:
            result.append(instruction)
            result.append('DEPOLARIZE2', instruction.targets_copy(), error_rate)
        elif instruction.name == 'M':
            result.append('X_ERROR', instruction.targets_copy(), error_rate)
            result.append(instruction)
        else:
            result.append(instruction)
    return result
