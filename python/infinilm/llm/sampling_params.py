"""
Sampling Parameters - Configuration for text generation sampling.
"""

from dataclasses import dataclass
from typing import List, Optional


def validate_stop_strings(value) -> None:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(marker, str) or not marker for marker in value
    ):
        raise ValueError("stop must contain nonempty strings")


def validate_stop_token_ids(value) -> None:
    if not isinstance(value, (list, tuple)) or any(
        type(token) is not int or not 0 <= token < (1 << 63) for token in value
    ):
        raise ValueError("stop_token_ids must contain non-negative int64 token IDs")


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    max_tokens: int = 512
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    seed: Optional[int] = None
    do_sample: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []

    def clone(self) -> "SamplingParams":
        """Create a copy of this SamplingParams instance."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            do_sample=self.do_sample,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop=list(self.stop) if isinstance(self.stop, (list, tuple)) else self.stop,
            stop_token_ids=(
                list(self.stop_token_ids)
                if isinstance(self.stop_token_ids, (list, tuple))
                else self.stop_token_ids
            ),
            ignore_eos=self.ignore_eos,
        )
