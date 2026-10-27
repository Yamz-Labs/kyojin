"""bigconv: model-agnostic converter for too-big MoE checkpoints (EXL3, streaming per-layer capture).

The DS4.1 pipeline (tools/ds41) is the reference. Everything arch-specific lives in an ArchAdapter
(tensor roles, expert enumeration, raw tensors, capture forward); the plan, sheet, memory-guarded
capture, claims, spool and quantizer are shared.
"""
