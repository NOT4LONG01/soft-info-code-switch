"""
soft_info.decoders.sinter
-------------------------
Sinter-compatible decoder wrappers for all supported QEC decoders.

Supported decoders
------------------
  mwpf            Minimum-Weight Parity Factor (local submodule).
  tesseract       Tesseract beam-search decoder.
  bp_osd          Belief-Propagation + Ordered Statistics Decoding (stimbposd).
  relay_bp        Relay-BP.

Each wrapper implements sinter.Decoder.compile_decoder_for_dem() and returns a
sinter.CompiledDecoder whose decode_shots_bit_packed() method handles a batch
of bit-packed detection events.

Trace files
-----------
Every wrapper optionally writes a binary timing trace when trace_filename is set.
Records are 4 little-endian floats: (cpu_time_per_shot, obj_lower, obj_upper, 0.0).
Decoders without an objective value write 0.0 for the bound fields.
Tesseract writes mean detcost (sum of log-likelihood costs of predicted errors) in both obj_lower and obj_upper.
Use read_trace(path) to load traces as a DataFrame with columns
cpu_time / obj_lower / obj_upper.  This format is identical to the one written
by the MWPF submodule so all decoders can be read uniformly.

RELAY_PARAMS
    Default hyperparameters for RelayBP; imported by optimize_schedule.py.

Usage
-----
    from soft_info.decoders.sinter import build_decoder, ALL_DECODERS

    decoder = build_decoder("mwpf", trace_filename="/tmp/trace.bin")
    decoder = build_decoder("tesseract")
"""

import glob
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import sinter
import stim

try:
    from mwpf.sinter_decoders import SinterMWPFDecoder as _SinterMWPFDecoder
    HAS_MWPF = True
except ImportError:
    HAS_MWPF = False


if HAS_MWPF:
    class _SinterMWPFDecoderExact(_SinterMWPFDecoder):
        @property
        def config(self) -> dict:
            return dict(
                cluster_node_limit=0,
                only_solve_primal_once=False,
            )

try:
    from tesseract_decoder import tesseract as _tesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import stimbposd
    HAS_BPOSD = True
except ImportError:
    HAS_BPOSD = False

try:
    import relay_bp
    import relay_bp.stim as _relay_bp_stim
    HAS_RELAYBP = True
except ImportError:
    HAS_RELAYBP = False

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

try:
    import fusion_blossom
    HAS_FUSION_BLOSSOM = True
except ImportError:
    HAS_FUSION_BLOSSOM = False

ALL_DECODERS      = ["mwpf", "tesseract"]           # default set for --decoder all
EXTRA_DECODERS    = ["bp_osd", "relay_bp"]          # available but not in default run

RELAY_PARAMS = {
    "gamma0": 0.1,
    "pre_iter": 80,
    "num_sets": 100,
    "set_max_iter": 60,
    "gamma_dist_interval": (-0.24, 0.66),
    "stop_nconv": 2,
}

def _write_trace(filename: str, cpu_time_per_shot: float, num_shots: int,
                 obj_lower: float = 0.0, obj_upper: float = 0.0) -> None:
    record = struct.pack("ffff", cpu_time_per_shot, obj_lower, obj_upper, 0.0)
    try:
        with open(filename, 'ab') as f:
            for _ in range(num_shots):
                f.write(record)
    except OSError:
        pass


def read_trace(path: str) -> 'pd.DataFrame':
    import pandas as pd
    records = []
    for fpath in sorted(glob.glob(f"{path}*")) or [path]:
        try:
            with open(fpath, 'rb') as f:
                while chunk := f.read(16):
                    if len(chunk) == 16:
                        cpu, lo, hi, _ = struct.unpack("ffff", chunk)
                        records.append((cpu, lo, hi))
        except OSError:
            pass
    return pd.DataFrame(records, columns=['cpu_time', 'obj_lower', 'obj_upper'])


def SinterMWPFDecoder(cluster_node_limit: int = 50,
                      trace_filename: Optional[str] = None) -> sinter.Decoder:
    if not HAS_MWPF:
        raise ImportError("mwpf submodule is not available.")
    return _SinterMWPFDecoder(cluster_node_limit=cluster_node_limit,
                              trace_filename=trace_filename)


def SinterMWPFDecoderExact(trace_filename: Optional[str] = None) -> sinter.Decoder:
    if not HAS_MWPF:
        raise ImportError("mwpf submodule is not available.")
    return _SinterMWPFDecoderExact(trace_filename=trace_filename)


class _CompiledTesseract(sinter.CompiledDecoder):
    def __init__(self, decoder, trace_filename: Optional[str]):
        self._decoder = decoder
        self._trace_filename = trace_filename

    def decode_shots_bit_packed(self, *,
                                bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        num_det = self._decoder.num_detectors
        num_obs = self._decoder.num_observables
        num_shots = bit_packed_detection_event_data.shape[0]

        syndromes = np.unpackbits(
            bit_packed_detection_event_data, axis=1, count=num_det, bitorder='little'
        ).astype(bool)

        t0 = time.perf_counter()
        obs_list = []
        total_cost = 0.0
        for syndrome in syndromes:
            errors = self._decoder.decode_to_errors(syndrome)
            total_cost += self._decoder.cost_from_errors(errors)
            obs_list.append(self._decoder.get_observables_from_errors(errors))
        obs_batch = np.array(obs_list, dtype=bool)
        elapsed = time.perf_counter() - t0

        if self._trace_filename is not None:
            mean_cost = total_cost / num_shots if num_shots else 0.0
            _write_trace(self._trace_filename,
                         elapsed / num_shots if num_shots else 0.0, num_shots,
                         obj_lower=mean_cost, obj_upper=mean_cost)

        pad = (-num_obs) % 8
        if pad:
            obs_batch = np.pad(obs_batch, ((0, 0), (0, pad)))
        num_obs_bytes = (num_obs + 7) // 8
        return np.packbits(obs_batch, axis=1, bitorder='little')[:, :num_obs_bytes]


@dataclass
class SinterTesseractDecoder(sinter.Decoder):
    det_beam: int = 50
    beam_climbing: bool = False
    no_revisit_dets: bool = True
    merge_errors: bool = True
    pqlimit: int = 200_000
    det_penalty: float = 0.0
    trace_filename: Optional[str] = None

    def compile_decoder_for_dem(self, *,
                                dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        if not HAS_TESSERACT:
            raise ImportError("tesseract_decoder package is not installed.")
        config = _tesseract.TesseractConfig(
            dem=dem,
            det_beam=self.det_beam,
            beam_climbing=self.beam_climbing,
            no_revisit_dets=self.no_revisit_dets,
            merge_errors=self.merge_errors,
            pqlimit=self.pqlimit,
            det_penalty=self.det_penalty,
        )
        return _CompiledTesseract(config.compile_decoder(),
                                  trace_filename=self.trace_filename)


class _CompiledPyMatching(sinter.CompiledDecoder):
    def __init__(self, matcher, trace_filename: Optional[str]):
        self._matcher = matcher
        self._trace_filename = trace_filename

    def decode_shots_bit_packed(self, *,
                                bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]

        t0 = time.perf_counter()
        result = self._matcher.decode_batch(bit_packed_detection_event_data, bit_packed_shots=True)
        elapsed = time.perf_counter() - t0

        if self._trace_filename is not None:
            _write_trace(self._trace_filename,
                         elapsed / num_shots if num_shots else 0.0, num_shots)
        return result


@dataclass
class SinterPyMatchingDecoder(sinter.Decoder):
    trace_filename: Optional[str] = None

    def compile_decoder_for_dem(self, *,
                                dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        if not HAS_PYMATCHING:
            raise ImportError("pymatching package is not installed.")
        matcher = pymatching.Matching.from_detector_error_model(dem)
        return _CompiledPyMatching(matcher, trace_filename=self.trace_filename)


class _CompiledWrapped(sinter.CompiledDecoder):
    def __init__(self, inner: sinter.CompiledDecoder, trace_filename: Optional[str]):
        self._inner = inner
        self._trace_filename = trace_filename

    def decode_shots_bit_packed(self, *,
                                bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        num_shots = bit_packed_detection_event_data.shape[0]
        t0 = time.perf_counter()
        result = self._inner.decode_shots_bit_packed(
            bit_packed_detection_event_data=bit_packed_detection_event_data)
        elapsed = time.perf_counter() - t0
        if self._trace_filename is not None:
            _write_trace(self._trace_filename,
                         elapsed / num_shots if num_shots else 0.0, num_shots)
        return result


@dataclass
class SinterBPOSDDecoder(sinter.Decoder):
    max_bp_iters: int = 20
    trace_filename: Optional[str] = None

    def compile_decoder_for_dem(self, *,
                                dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        if not HAS_BPOSD:
            raise ImportError("stimbposd package is not installed.")
        inner = stimbposd.SinterDecoder_BPOSD(
            max_bp_iters=self.max_bp_iters
        ).compile_decoder_for_dem(dem=dem)
        return _CompiledWrapped(inner, trace_filename=self.trace_filename)


@dataclass
class SinterFusionBlossomDecoder(sinter.Decoder):
    trace_filename: Optional[str] = None

    def compile_decoder_for_dem(self, *,
                                dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        if not HAS_FUSION_BLOSSOM:
            raise ImportError("fusion_blossom package is not installed.")
        inner = fusion_blossom.SinterDecoder().compile_decoder_for_dem(dem=dem)
        return _CompiledWrapped(inner, trace_filename=self.trace_filename)


@dataclass
class SinterRelayBPDecoder(sinter.Decoder):
    params: dict = field(default_factory=lambda: dict(RELAY_PARAMS))
    trace_filename: Optional[str] = None

    def compile_decoder_for_dem(self, *,
                                dem: stim.DetectorErrorModel) -> sinter.CompiledDecoder:
        if not HAS_RELAYBP:
            raise ImportError("relay_bp package is not installed.")
        decoders = _relay_bp_stim.sinter_decoders(**self.params)
        inner_decoder = next(iter(decoders.values()))
        inner = inner_decoder.compile_decoder_for_dem(dem=dem)
        num_det_bytes = (dem.num_detectors + 7) // 8
        inner.decode_shots_bit_packed(
            bit_packed_detection_event_data=np.zeros((1, num_det_bytes), dtype=np.uint8))
        return _CompiledWrapped(inner, trace_filename=self.trace_filename)


def build_decoder(name: str, trace_filename: Optional[str] = None) -> sinter.Decoder:
    kw = {"trace_filename": trace_filename} if trace_filename else {}
    if name == "mwpf":             return SinterMWPFDecoder(cluster_node_limit=50, **kw)
    elif name == "mwpf_exact":     return SinterMWPFDecoderExact(**kw)
    elif name == "tesseract":      return SinterTesseractDecoder(det_beam=50, **kw)
    elif name == "pymatching":     return SinterPyMatchingDecoder(**kw)
    elif name == "bp_osd":         return SinterBPOSDDecoder(max_bp_iters=1000, **kw)
    elif name == "fusion_blossom": return SinterFusionBlossomDecoder(**kw)
    elif name == "relay_bp":       return SinterRelayBPDecoder(**kw)
    else:
        raise ValueError(f"Unknown decoder {name!r}. Choices: {ALL_DECODERS}")
