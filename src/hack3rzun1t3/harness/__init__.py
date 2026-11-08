"""Torch-free core of the distributed harness: planning, checkpointing, aggregation.

Nothing here imports torch, so it runs and is tested anywhere. The GPU side
(layer streaming, hooks) will sit on top of these pieces.
"""
