"""Probability arithmetic and owned host logit rows for classic NextN sampling."""

from collections import Counter
import ctypes
import math

import numpy as np


def validate_penalties(presence_penalty=0, frequency_penalty=0, repetition_penalty=1):
    for name, value in (("presence_penalty", presence_penalty),
                        ("frequency_penalty", frequency_penalty),
                        ("repetition_penalty", repetition_penalty)):
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"NextN {name} must be finite")
    if not -2 <= presence_penalty <= 2 or not -2 <= frequency_penalty <= 2:
        raise ValueError("NextN presence and frequency penalties must be in [-2, 2]")
    if repetition_penalty <= 0:
        raise ValueError("NextN repetition penalty must be positive")


def penalized_logits(logits, *, prompt_token_ids=(), generated_token_ids=(),
                     presence_penalty=0, frequency_penalty=0, repetition_penalty=1):
    """Repetition uses prompt and generated history. Other penalties use output only."""
    validate_penalties(presence_penalty, frequency_penalty, repetition_penalty)
    scores = np.asarray(logits, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("Expected a vocabulary logit row")
    if (presence_penalty, frequency_penalty, repetition_penalty) == (0, 0, 1):
        return scores
    counts = Counter(generated_token_ids)
    repeated = set(prompt_token_ids) | counts.keys()
    if any(type(token) is not int or not 0 <= token < scores.size for token in repeated):
        raise ValueError("Invalid NextN penalty history token ID")
    scores = scores.copy()
    for token in repeated:
        if scores[token] > 0:
            scores[token] /= repetition_penalty
        else:
            scores[token] *= repetition_penalty
    for token, count in counts.items():
        scores[token] -= presence_penalty + frequency_penalty * count
    return scores


def _validate_filters(temperature, top_k, top_p):
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("NextN sampling temperature must be positive and finite")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("NextN sampling top_k must be a nonnegative integer")
    if type(top_p) not in (float, int) or not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("NextN sampling top_p must satisfy 0 < top_p <= 1")


def probabilities(logits, temperature, top_k, top_p, **penalties):
    """Apply penalties, temperature, stable top-k and inclusive nucleus truncation."""
    _validate_filters(temperature, top_k, top_p)
    scores = penalized_logits(logits, **penalties)
    if np.isnan(scores).any() or np.isposinf(scores).any() or not np.isfinite(scores).any():
        raise ValueError("Expected finite or -inf logits with at least one finite token")
    order = np.lexsort((np.arange(scores.size), -scores))
    if top_k:
        order = order[:top_k]
    selected = scores[order]
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp((selected - selected.max()) / temperature)
    weights /= weights.sum()
    keep = min(len(weights), int(np.searchsorted(np.cumsum(weights), top_p, side="left")) + 1)
    result = np.zeros(scores.size, dtype=np.float64)
    result[order[:keep]] = weights[:keep] / weights[:keep].sum()
    return result


def _validate_distribution(probs):
    if (probs.ndim != 1 or not probs.size or not np.isfinite(probs).all()
            or np.any(probs < 0) or not np.isclose(probs.sum(), 1)):
        raise ValueError("Expected a normalized nonnegative finite distribution")


def sample(probs, rng):
    """Sample from a vocabulary distribution using only the supplied request RNG."""
    _validate_distribution(probs)
    cumulative = np.cumsum(probs)
    value = rng.random() * cumulative[-1]
    # Clamping to the last supported token prevents tail rounding from sampling
    # a zero-probability token when a draw is close to one.
    index = int(np.searchsorted(cumulative, value, side="right"))
    if index == len(probs):
        return int(np.flatnonzero(probs)[-1])
    return index


def accept_or_resample(draft_token, p, q, rng):
    """Accept with min(1,p/q), or sample correction from normalized (p-q)+."""
    _validate_distribution(p)
    _validate_distribution(q)
    if p.shape != q.shape or type(draft_token) is not int or not 0 <= draft_token < len(p):
        raise ValueError("Mismatched p/q vocabulary or draft token")
    if q[draft_token] <= 0:
        raise ValueError("A draft token must have positive proposal probability")
    if rng.random() < min(1.0, p[draft_token] / q[draft_token]):
        return True, None
    residual = np.maximum(p - q, 0.0)
    mass = float(residual.sum())
    if mass <= 0:
        raise RuntimeError("Rejected draft has zero residual mass")
    return False, sample(residual / mass, rng)


def host_logit_rows(logits, expected_rows, vocab_size, *, start=0, count=None):
    """Read an owning float32 host copy from a completed V4 forward result."""
    from infinilm import _core

    if logits is None or tuple(logits.shape) not in (
        (1, expected_rows, vocab_size), (expected_rows, vocab_size)
    ):
        raise RuntimeError("NextN sampling requires full-vocabulary logits in native row order")
    if count is None:
        count = expected_rows - start
    if not 0 <= start < start + count <= expected_rows or vocab_size < 1:
        raise ValueError("Invalid NextN logit row range")
    if start != 0 or count != expected_rows:
        logits = logits.narrow(len(logits.shape) - 2, start, count)
    if logits.dtype != _core.bfloat16:
        return logits.to_numpy().astype(np.float32, copy=False).reshape(count, vocab_size)

    # NumPy's native dtype list has no BF16. The high half of each float32
    # bit pattern is exactly the BF16 value, including signed zero and -inf.
    source = logits if logits.device.type == "cpu" else logits.to("cpu")
    source = source.contiguous()
    bits = np.empty((count, vocab_size), dtype=np.uint16)
    ctypes.memmove(bits.ctypes.data, source.data_ptr(), bits.nbytes)
    return np.left_shift(bits.astype(np.uint32), 16).view(np.float32)


def validate_request_sampling(params):
    penalties = (params.presence_penalty, params.frequency_penalty, params.repetition_penalty)
    validate_penalties(*penalties)
    if type(params.do_sample) is not bool:
        raise ValueError("NextN do_sample must be a boolean")
    if not params.do_sample:
        if penalties != (0, 0, 1):
            raise ValueError("NextN nondefault penalties require do_sample=True")
        if type(params.top_k) is not int or params.top_k != 1:
            raise ValueError("classic greedy MTP requires integer top_k=1")
        return
    if params.seed is not None and (type(params.seed) is not int or params.seed < 0):
        raise ValueError("NextN seed must be a nonnegative integer")
    _validate_filters(params.temperature, params.top_k, params.top_p)


class RequestSampling:
    """A live request's deterministic conditioning prefix and private RNG."""

    def __init__(self, request):
        self.params = request.sampling_params
        self.prompt = tuple(request.prompt_token_ids)
        self.history = tuple(request.generated_token_ids)
        self.rng = request._sampling_rng

    def distribution(self, logits, extension=()):
        params = self.params
        return probabilities(
            logits, params.temperature, params.top_k, params.top_p,
            prompt_token_ids=self.prompt, generated_token_ids=self.history + tuple(extension),
            presence_penalty=params.presence_penalty, frequency_penalty=params.frequency_penalty,
            repetition_penalty=params.repetition_penalty,
        )

    def select(self, logits, extension=()):
        return sample(self.distribution(logits, extension), self.rng)
