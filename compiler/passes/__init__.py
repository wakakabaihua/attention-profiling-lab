"""compiler.passes — 编译 Pass 模块。"""

from compiler.passes.pattern_match import (
    MatchResult,
    match_scale_mask_softmax,
    match_qk_scale_mask_softmax,
    find_all_patterns,
)
from compiler.passes.fusion import ScaleMaskSoftmaxFusionPass, FusionResult
from compiler.passes.canonicalize import CanonicalizationPass
from compiler.passes.validation import ValidationPass, ValidationResult

__all__ = [
    "MatchResult",
    "match_scale_mask_softmax",
    "match_qk_scale_mask_softmax",
    "find_all_patterns",
    "ScaleMaskSoftmaxFusionPass",
    "FusionResult",
    "CanonicalizationPass",
    "ValidationPass",
    "ValidationResult",
]
