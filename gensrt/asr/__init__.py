"""ASR engine layer.

The pipeline imports :func:`get_engine_for_model` from this package
rather than touching individual engine classes directly.

Routing rules and the engine inventory are documented in
:mod:`gensrt.asr.factory`.
"""

from gensrt.asr.factory import get_engine_for_model
from gensrt.asr.base import ASREngine

__all__ = ["get_engine_for_model", "ASREngine"]
