"""
circuit.py
----------
Stim circuit builders for CSS quantum memory experiments.

Circuit topologies
------------------
Two topologies are selected automatically by generate_css_memory_experiment
based on whether Hx == Hz:

  Self-dual schedule
      For all self-dual codes (Hx == Hz): triangular, square, GO03, QR, EQR.
      Shared ancilla set.  Within each round: X-checks first (H-gated ancilla
      as control), then Z-checks (ancilla as target).  Detectors reference the
      previous round's measurements of the same type.

  Standard CSS schedule
      For non-self-dual codes (Hx ≠ Hz): JA25 and others.
      Separate X-ancilla and Z-ancilla sets.  X and Z stabilizers measured
      simultaneously within each round using independent ancilla registers.

Schedule files
--------------
Optimised CNOT orderings produced by optimize_schedule.py are stored as JSON:
    {"ancilla_row,data_qubit_idx": tick_priority, ...}
Loaded by load_schedule() and injected via _build_cnot_ops.  When no schedule
file is present the natural column order within each stabilizer row is used.

Public API
----------
    load_schedule(filepath)                              -> dict
    generate_css_memory_experiment(Hx, Hz, rounds, ...) -> stim.Circuit
    generate_experiment_with_noise(Hx, Hz, rounds, ...) -> stim.Circuit

Private helpers
---------------
    _row_start_x          — centering helper for qubit coordinates
    _build_cnot_ops       — returns sorted (priority, ctrl, targ) CNOT list
    _generate_color_code_schedule
    _generate_self_dual_schedule
    _generate_standard_schedule
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import stim

from helpers import find_logical_operator
from noise_models import (standard_depolarizing_noise_model, si1000_noise_model,
                          phenomenological_css_circuit, circuit_level_css_circuit)


def load_schedule(filepath: str) -> Dict[Tuple[int, int], int]:
    with open(filepath) as f:
        data = json.load(f)
    return {(int(k.split(',')[0]), int(k.split(',')[1])): v for k, v in data.items()}


def generate_css_memory_experiment(
    Hx: np.ndarray,
    Hz: np.ndarray,
    rounds: int,
    memory_basis: str = "Z",
    schedule: Optional[dict] = None,
    code_type: str = "unknown",
    circuit_type: Optional[str] = None,
) -> stim.Circuit:
    """
    circuit_type overrides the default dispatch when set:
      'color'     — C_XYZ frame-rotation (triangular/square color code schedule)
      'self_dual' — shared ancillas, X then Z per round
      'standard'  — separate X and Z ancilla sets measured simultaneously
    """
    is_self_dual = (Hx.shape == Hz.shape) and np.array_equal(Hx, Hz)

    effective = circuit_type
    if effective is None:
        if is_self_dual:
            effective = "self_dual"
        else:
            effective = "standard"

    if effective == "color":
        return _generate_color_code_schedule(Hx, rounds, schedule)
    elif effective == "self_dual":
        return _generate_self_dual_schedule(Hx, rounds, memory_basis, schedule)
    else:
        return _generate_standard_schedule(Hx, Hz, rounds, memory_basis, schedule)


def generate_experiment_with_noise(
    Hx: np.ndarray,
    Hz: np.ndarray,
    rounds: int,
    noise_model_name: str,
    noise_params: dict,
    memory_basis: str = "Z",
    schedule: Optional[dict] = None,
    code_type: str = "unknown",
    circuit_type: Optional[str] = None,
    idle_noise_on_ancillas: bool = False,
) -> stim.Circuit:
    clean = generate_css_memory_experiment(
        Hx, Hz, rounds,
        memory_basis=memory_basis,
        schedule=schedule,
        code_type=code_type,
        circuit_type=circuit_type,
    )
    num_data = Hx.shape[1]
    data_qubits = list(range(num_data))
    p = noise_params['p']

    ancilla_qubits: Optional[List[int]] = None
    if idle_noise_on_ancillas:
        is_self_dual = (Hx.shape == Hz.shape) and np.array_equal(Hx, Hz)
        num_anc = Hx.shape[0] if is_self_dual else Hx.shape[0] + Hz.shape[0]
        ancilla_qubits = list(range(num_data, num_data + num_anc))

    if noise_model_name == "depolarizing":
        return standard_depolarizing_noise_model(
            circuit=clean,
            data_qubits=data_qubits,
            after_clifford_depolarization=noise_params.get('p_clifford', p),
            after_reset_flip_probability=noise_params.get('p_reset', p),
            before_measure_flip_probability=noise_params.get('p_meas', p),
            before_round_data_depolarization=noise_params.get('p_data_round', p),
            ancilla_qubits=ancilla_qubits,
        )
    elif noise_model_name == "si1000":
        return si1000_noise_model(circuit=clean, data_qubits=data_qubits, probability=p)
    elif noise_model_name == "phenomenological":
        # noiseless CNOTs; basis selects X- or Z-memory
        basis  = noise_params.get('basis', 'X')
        p_data = noise_params.get('p_data', p)
        p_meas = noise_params.get('p_meas', p)
        if basis not in ('X', 'Z'):
            raise ValueError(f"phenomenological basis must be 'X' or 'Z'; got {basis!r}")
        if basis == 'Z':
            H = Hx
            L = np.atleast_2d(find_logical_operator(Hx, Hz, basis='X')).astype(np.uint8)
        else:
            H = Hz
            L = np.atleast_2d(find_logical_operator(Hx, Hz, basis='Z')).astype(np.uint8)
        return phenomenological_css_circuit(H, L, rounds, p_data, p_meas, noise=basis)
    else:
        raise ValueError(f"Unknown noise model: {noise_model_name!r}")


def _row_start_x(num_items: int, spacing: float, center: float) -> float:
    return center - ((num_items - 1) * spacing / 2.0) if num_items > 1 else center


def _build_cnot_ops(
    check_matrix: np.ndarray,
    ancilla_idxs: List[int],
    data_idxs: List[int],
    ancilla_is_control: bool,
    schedule: Optional[Dict[Tuple[int, int], int]],
    schedule_offset: int = 0,
) -> List[Tuple[int, int, int]]:
    data_idx_map = {v: i for i, v in enumerate(data_idxs)}
    ops = []
    for i, row in enumerate(check_matrix):
        anc = ancilla_idxs[i]
        targets = [data_idxs[q] for q in np.flatnonzero(row)]
        for t_idx, dat in enumerate(targets):
            key = (i + schedule_offset, data_idx_map[dat])
            priority = schedule.get(key, t_idx) if schedule else t_idx
            ops.append((priority, anc, dat) if ancilla_is_control else (priority, dat, anc))
    return ops


def _generate_color_code_schedule(
    H: np.ndarray,
    rounds: int,
    schedule: Optional[dict] = None,
) -> stim.Circuit:
    num_data = H.shape[1]
    num_checks = H.shape[0]
    data_qubits = list(range(num_data))
    ancillas = list(range(num_data, num_data + num_checks))
    center = float(num_data - 1)

    circuit = stim.Circuit()
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])

    anc_start = _row_start_x(num_checks, 4.0, center)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        anc_coords.append(pos)

    circuit.append("R", data_qubits)
    circuit.append("R", ancillas)
    circuit.append("TICK")

    def _round(c: stim.Circuit, round_idx: int) -> None:
        ops = _build_cnot_ops(H, ancillas, data_qubits, ancilla_is_control=False,
                              schedule=schedule, schedule_offset=0)
        ops.sort()
        for _, ctrl, targ in ops:
            c.append("CNOT", [ctrl, targ])
        c.append("MR", ancillas)
        c.append("C_XYZ", data_qubits)
        c.append("TICK")

        if round_idx >= 3:
            for i in range(num_checks):
                rec_now  = stim.target_rec(-num_checks + i)
                rec_prev = stim.target_rec(-num_checks * 4 + i)
                c.append("DETECTOR", [rec_now, rec_prev],
                         [anc_coords[i][0], anc_coords[i][1], round_idx % 3])

    for r in range(rounds):
        _round(circuit, r)

    final_basis = rounds % 3
    if final_basis == 0:
        circuit.append("M",  data_qubits)
    elif final_basis == 1:
        circuit.append("MX", data_qubits)
    else:
        circuit.append("MY", data_qubits)

    if rounds >= 3:
        for i, row in enumerate(H):
            rec_data = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec_prev = stim.target_rec(-(num_data + num_checks * 3 - i))
            circuit.append("DETECTOR", rec_data + [rec_prev],
                           [anc_coords[i][0], anc_coords[i][1], final_basis])

    op = find_logical_operator(H, H, basis="Z")
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit


def _generate_self_dual_schedule(
    H: np.ndarray,
    rounds: int,
    memory_basis: str,
    schedule: Optional[dict],
) -> stim.Circuit:
    num_data = H.shape[1]
    num_checks = H.shape[0]
    data_qubits = list(range(num_data))
    ancillas = list(range(num_data, num_data + num_checks))
    center = float(num_data - 1)

    circuit = stim.Circuit()
    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])

    anc_start = _row_start_x(num_checks, 4.0, center)
    anc_coords = []
    for i, q in enumerate(ancillas):
        pos = [anc_start + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        anc_coords.append(pos)

    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", ancillas)
    circuit.append("TICK")

    def _round(c: stim.Circuit, is_first: bool) -> None:
        total_m = 2 * num_checks

        # X-checks
        c.append("H", ancillas)
        c.append("TICK")
        x_ops = _build_cnot_ops(H, ancillas, data_qubits, ancilla_is_control=True,
                                schedule=schedule, schedule_offset=0)
        x_ops.sort()
        for _, ctrl, targ in x_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("H", ancillas)
        c.append("M", ancillas)
        c.append("R", ancillas)
        c.append("TICK")

        # Z-checks
        z_ops = _build_cnot_ops(H, ancillas, data_qubits, ancilla_is_control=False,
                                schedule=schedule, schedule_offset=num_checks)
        z_ops.sort()
        for _, ctrl, targ in z_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("M", ancillas)
        c.append("R", ancillas)
        c.append("TICK")

        # X detectors
        for i in range(num_checks):
            rec_now  = stim.target_rec(-total_m + i)
            rec_prev = stim.target_rec(-total_m * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X":
                c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 0])

        # Z detectors
        for i in range(num_checks):
            rec_now  = stim.target_rec(-num_checks + i)
            rec_prev = stim.target_rec(-num_checks - total_m + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z":
                c.append("DETECTOR", args, [anc_coords[i][0], anc_coords[i][1], 1])

    _round(circuit, is_first=True)
    if rounds > 1:
        loop = stim.Circuit()
        _round(loop, is_first=False)
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

    # Final data measurements
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_checks - i)))
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 1])
    else:
        circuit.append("MX", data_qubits)
        for i, row in enumerate(H):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + 2 * num_checks - i)))
            circuit.append("DETECTOR", rec, [anc_coords[i][0], anc_coords[i][1], 0])

    op = find_logical_operator(H, H, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit


def _generate_standard_schedule(
    Hx: np.ndarray,
    Hz: np.ndarray,
    rounds: int,
    memory_basis: str,
    schedule: Optional[dict],
) -> stim.Circuit:
    num_data = Hx.shape[1]
    num_x = Hx.shape[0]
    num_z = Hz.shape[0]
    data_qubits = list(range(num_data))
    x_ancillas = list(range(num_data, num_data + num_x))
    z_ancillas = list(range(num_data + num_x, num_data + num_x + num_z))
    center = float(num_data - 1)

    circuit = stim.Circuit()

    x_start = _row_start_x(num_x, 4.0, center)
    x_coords = []
    for i, q in enumerate(x_ancillas):
        pos = [x_start + i * 4.0, 0.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        x_coords.append(pos)

    for i, q in enumerate(data_qubits):
        circuit.append("QUBIT_COORDS", [q], [i * 2.0, 2.0])

    z_start = _row_start_x(num_z, 4.0, center)
    z_coords = []
    for i, q in enumerate(z_ancillas):
        pos = [z_start + i * 4.0, 4.0]
        circuit.append("QUBIT_COORDS", [q], pos)
        z_coords.append(pos)

    circuit.append("R" if memory_basis == "Z" else "RX", data_qubits)
    circuit.append("R", x_ancillas + z_ancillas)
    circuit.append("TICK")

    def _round(c: stim.Circuit, is_first: bool) -> None:
        tot_meas = num_x + num_z

        # X stabilizers
        c.append("H", x_ancillas)
        c.append("TICK")
        x_ops = _build_cnot_ops(Hx, x_ancillas, data_qubits, ancilla_is_control=True,
                                schedule=schedule, schedule_offset=0)
        x_ops.sort()
        for _, ctrl, targ in x_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("TICK")
        c.append("H", x_ancillas)
        c.append("TICK")
        c.append("MR", x_ancillas)

        # Z stabilizers
        z_ops = _build_cnot_ops(Hz, z_ancillas, data_qubits, ancilla_is_control=False,
                                schedule=schedule, schedule_offset=num_x)
        z_ops.sort()
        for _, ctrl, targ in z_ops:
            c.append("CNOT", [ctrl, targ])
        c.append("TICK")
        c.append("MR", z_ancillas)

        # X detectors
        for i in range(num_x):
            rec_now  = stim.target_rec(-tot_meas + i)
            rec_prev = stim.target_rec(-tot_meas * 2 + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "X":
                c.append("DETECTOR", args, [x_coords[i][0], x_coords[i][1], 0])

        # Z detectors
        for i in range(num_z):
            rec_now  = stim.target_rec(-num_z + i)
            rec_prev = stim.target_rec(-tot_meas - num_z + i)
            args = [rec_now, rec_prev] if not is_first else [rec_now]
            if not is_first or memory_basis == "Z":
                c.append("DETECTOR", args, [z_coords[i][0], z_coords[i][1], 0])

        # Advance time coordinate so each round gets a distinct z-index.
        # Inside a REPEAT block stim accumulates this per iteration.
        c.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

    _round(circuit, is_first=True)
    if rounds > 1:
        loop = stim.Circuit()
        _round(loop, is_first=False)
        circuit.append(stim.CircuitRepeatBlock(rounds - 1, loop))

    # Final data measurements
    if memory_basis == "Z":
        circuit.append("M", data_qubits)
        for i, row in enumerate(Hz):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z - i)))
            circuit.append("DETECTOR", rec, [z_coords[i][0], z_coords[i][1], 0])
    elif memory_basis == "X":
        circuit.append("MX", data_qubits)
        for i, row in enumerate(Hx):
            rec = [stim.target_rec(-(num_data - q)) for q in np.flatnonzero(row)]
            rec.append(stim.target_rec(-(num_data + num_z + num_x - i)))
            circuit.append("DETECTOR", rec, [x_coords[i][0], x_coords[i][1], 0])

    op = find_logical_operator(Hx, Hz, basis=memory_basis)
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(-(num_data - k)) for k in np.flatnonzero(op)], 0)
    return circuit
