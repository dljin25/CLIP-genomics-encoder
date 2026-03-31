"""Model architectures for genomics set encoding."""

from .deep_sets import DeepSetsEncoder
from .set_transformer import SetTransformerEncoder

__all__ = ["DeepSetsEncoder", "SetTransformerEncoder"]
