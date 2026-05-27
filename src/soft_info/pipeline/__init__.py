"""
soft_info.pipeline
------------------
Experiment-driver entry points (``python -m soft_info.pipeline.<module>``).

Submodules
~~~~~~~~~~
- ``main``              : Sinter-driven color-code memory experiments. CLI:
                          ``python -m soft_info.pipeline.main --code_type … --decoder …``.
- ``single_shot``       : Single-shot-property sweep (fix T sweep W, or fix W sweep T).
- ``overlapping``       : Sliding-window CSS memory decoding (:func:`ler_windowed`).
- ``optimize_schedule`` : MCTS / ILP syndrome-extraction schedule optimiser; writes
                          to ``src/pcms/<code_type>_codes/``.

The soft-info post-selection driver (per-shot gap → acceptance-rate sweep) is
intended to live here as well in a follow-up.
"""
