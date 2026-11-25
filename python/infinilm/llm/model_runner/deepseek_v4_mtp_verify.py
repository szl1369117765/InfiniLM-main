"""Packed Target verify and commit intent for classic greedy DeepSeek V4 MTP."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal


_INT32_MAX = (1 << 31) - 1
_INT64_MAX = (1 << 63) - 1


def _is_token(value: int) -> bool:
    return type(value) is int and 0 <= value <= _INT64_MAX


@dataclass(frozen=True)
class GreedyVerifyRequest:
    owner: str
    start: int
    remaining: int
    anchor: int
    draft_tokens: tuple[int, ...]
    eos_token_ids: frozenset[int] = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "draft_tokens", tuple(self.draft_tokens))
        object.__setattr__(self, "eos_token_ids", frozenset(self.eos_token_ids))


@dataclass(frozen=True, eq=False)
class VerifyRow:
    """Identity belongs to a specific batch and flat output position."""

    owner: str
    flat_index: int
    position: int
    input_token: int


@dataclass(frozen=True)
class TargetVerifyRow:
    row: VerifyRow
    predicted_token: int


@dataclass(frozen=True)
class GreedyCommitIntent:
    owner: str
    emitted_tokens: tuple[int, ...]
    retain_length: int
    materialized_rows: tuple[VerifyRow, ...]
    terminal: bool
    next_anchor: int | None
    next_anchor_kind: Literal["correction", "bonus"] | None


@dataclass(frozen=True)
class GreedyVerifyBatch:
    requests: tuple[GreedyVerifyRequest, ...]
    rows: tuple[VerifyRow, ...]
    input_offsets: tuple[int, ...]
    past_kv_lengths: tuple[int, ...]
    total_kv_lengths: tuple[int, ...]
    cu_seqlens: tuple[int, ...]

    @property
    def input_ids(self) -> tuple[int, ...]:
        return tuple(row.input_token for row in self.rows)

    @property
    def position_ids(self) -> tuple[int, ...]:
        return tuple(row.position for row in self.rows)

    @classmethod
    def pack(
        cls, requests: Iterable[GreedyVerifyRequest], max_context: int, *, scratch_step: bool = False,
    ) -> "GreedyVerifyBatch":
        requests = tuple(requests)
        if not requests or type(max_context) is not int or not 1 <= max_context <= _INT32_MAX:
            raise ValueError("classic MTP verify requires requests and an int32 context limit")

        owners: set[str] = set()
        rows: list[VerifyRow] = []
        offsets, past, total, cumulative = [0], [], [], [0]
        for request in requests:
            owner = request.owner
            if not isinstance(owner, str) or not owner or owner in owners:
                raise ValueError("classic MTP verify requires distinct request owners")
            owners.add(owner)

            tokens = (request.anchor, *request.draft_tokens)
            if (
                type(request.remaining) is not int
                or not (
                    len(tokens) == request.remaining == 1 if scratch_step
                    else 2 <= len(tokens) <= min(4, request.remaining)
                )
            ):
                raise ValueError(f"invalid classic MTP verify query or output budget for {owner}")
            if (
                type(request.start) is not int
                or not 0 <= request.start <= max_context - len(tokens)
            ):
                raise ValueError(f"classic MTP verify exceeds the context for {owner}")
            if not all(_is_token(token) for token in (*tokens, *request.eos_token_ids)):
                raise ValueError(f"invalid classic MTP token ID for {owner}")
            if any(token in request.eos_token_ids for token in tokens[:-1]):
                raise ValueError(f"classic MTP proposal continues past EOS for {owner}")

            start = request.start
            end = start + len(tokens)
            rows.extend(
                VerifyRow(owner, offsets[-1] + local, start + local, token)
                for local, token in enumerate(tokens)
            )
            offsets.append(len(rows))
            past.append(start)
            total.append(end)
            cumulative.append(cumulative[-1] + end)

        if cumulative[-1] > _INT32_MAX:
            raise ValueError("classic MTP verify exceeds int32 cumulative sequence length")
        return cls(
            requests, tuple(rows), tuple(offsets), tuple(past), tuple(total), tuple(cumulative)
        )

    @property
    def ends_in_known_stop(self) -> bool:
        return (
            len(self.requests) == 1 and len(self.rows) >= 2
            and self.rows[-1].input_token in self.requests[0].eos_token_ids
        )

    def resolve(
        self, output: Sequence[TargetVerifyRow], *, omit_terminal: bool = False,
        early_rejection: bool = False,
    ) -> tuple[GreedyCommitIntent, ...]:
        if omit_terminal and not self.ends_in_known_stop:
            raise ValueError("Target verify can omit only a known terminal proposal")
        if early_rejection and (omit_terminal or len(self.requests) != 1 or not 1 <= len(output) <= len(self.rows)):
            raise ValueError("Stepwise Target verification requires one nonempty branch prefix")
        submitted_rows = self.rows[:len(output)] if early_rejection else (
            self.rows[:-1] if omit_terminal else self.rows
        )
        if len(output) != len(submitted_rows):
            raise ValueError("Target verify result count differs from the submitted batch")
        for expected, result in zip(submitted_rows, output):
            if result.row is not expected:
                raise ValueError("Target verify result does not belong to the submitted row")
            if not _is_token(result.predicted_token):
                raise ValueError("Target verify requires non-negative int64 token IDs")
        if (early_rejection and len(output) < len(self.rows)
                and output[-1].predicted_token == self.rows[len(output)].input_token):
            raise ValueError("Stepwise Target verification omitted a matching Draft row")

        intentions = []
        for index, request in enumerate(self.requests):
            begin, end = self.input_offsets[index : index + 2]
            accepted_end = begin + 1  # The anchor was already selected by Target.
            while accepted_end < end:
                proposed = self.rows[accepted_end].input_token
                if proposed != output[accepted_end - 1].predicted_token:
                    break
                accepted_end += 1
            if early_rejection and accepted_end != len(submitted_rows):
                raise ValueError("Stepwise Target verification continued past a rejection")

            accepted = self.rows[begin:accepted_end]
            emitted = tuple(row.input_token for row in accepted)
            terminal = len(emitted) == request.remaining or emitted[-1] in request.eos_token_ids
            # The preceding prediction verifies a terminal proposal. An accepted
            # stop is emitted, but the input that would predict its successor
            # was not submitted and does not own a retained cache row.
            retained = submitted_rows[begin:accepted_end]
            anchor, kind = None, None
            if not terminal:
                anchor = output[accepted_end - 1].predicted_token
                kind = "bonus" if accepted_end == end else "correction"
            intentions.append(
                GreedyCommitIntent(
                    request.owner, emitted, request.start + len(retained),
                    retained, terminal, anchor, kind,
                )
            )
        return tuple(intentions)


def run_greedy_verify(
    requests: Iterable[GreedyVerifyRequest],
    verify_target: Callable[[GreedyVerifyBatch], Sequence[TargetVerifyRow]],
    *,
    max_context: int,
) -> tuple[GreedyCommitIntent, ...]:
    """Use the MTP-owned Target transport and resolve intents before any commit.

    The transport owns cache protection and must return rows bound to this
    submission in packed order. The caller owns commit, rollback and Draft repair.
    """
    batch = GreedyVerifyBatch.pack(requests, max_context)
    return batch.resolve(verify_target(batch))
