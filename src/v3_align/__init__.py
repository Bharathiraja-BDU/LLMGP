"""Training-only alignment objective for the unchanged V3 architecture."""

from .alignment_loss import class_discriminative_alignment

__all__ = ["class_discriminative_alignment"]
