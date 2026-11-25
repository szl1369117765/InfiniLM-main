"""Owner-bound eager classic NextN transport for a V4 engine."""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

from .deepseek_v4_mtp_sampling import accept_or_resample, host_logit_rows, sample
from .deepseek_v4_mtp_verify import GreedyCommitIntent, GreedyVerifyBatch, TargetVerifyRow


logger = logging.getLogger(__name__)


def _slice_hidden(tensors, start, length):
    return tuple(tensor.narrow(0, start, length) for tensor in tensors)


@dataclass(frozen=True)
class _DraftSeed:
    token: int
    hidden: tuple
    pages: tuple[int, ...]


@dataclass
class _DraftState:
    hidden: tuple
    anchor: int | None
    position: int
    proposal: tuple[int, ...] | None = None
    proposal_table: tuple[int, ...] | None = None
    seed: _DraftSeed | None = None
    proposal_law: tuple | None = None


@dataclass(frozen=True)
class _DraftSegment:
    tokens: tuple[int, ...]
    start: int
    table: tuple[int, ...]
    hidden: tuple


@dataclass(frozen=True)
class _ScratchCommit:
    table: tuple[int, ...]
    hidden_rows: tuple


class _TargetTransaction:
    def __init__(self, transport, batch, tables, tokens, verify_hidden=None,
                 proposal_laws=(), logits=None, *, omit_terminal=False, early_rejection=False,
                 scratch_commit=None, batch_draft_seed=False, k1_graph_seed=False):
        self.transport = transport
        self.batch = batch
        self.tables = tables
        self.omit_terminal = omit_terminal
        self.early_rejection = early_rejection
        submitted_rows = batch.rows[:len(tokens)] if early_rejection else (
            batch.rows[:-1] if omit_terminal else batch.rows
        )
        self.rows = tuple(
            TargetVerifyRow(row, token) for row, token in zip(submitted_rows, tokens, strict=True)
        )
        self.verify_hidden = verify_hidden
        self.proposal_laws = proposal_laws
        self.logits = logits
        self.scratch_commit = scratch_commit
        self.batch_draft_seed = batch_draft_seed
        self.k1_graph_seed = k1_graph_seed
        self._resolved = None
        self.committed = False

    def resolve_stochastic(self):
        if not self.proposal_laws:
            raise RuntimeError("MTP stochastic resolution requires a sampled Draft proposal")
        if self._resolved is not None:
            return self._resolved
        intents = list(self.batch.resolve(self.rows))
        for index, law in enumerate(self.proposal_laws):
            if law is not None:
                intents[index] = self._sample_owner(index, law)
        self._resolved = tuple(intents)
        return self._resolved

    def _sample_owner(self, index, law):
        request = self.batch.requests[index]
        begin = self.batch.input_offsets[index]
        sampling, q_rows = law
        proposals = request.draft_tokens
        retained_count = 1
        correction = None
        for i, (token, q) in enumerate(zip(proposals, q_rows, strict=True)):
            p = sampling.distribution(self.logits[begin + i], proposals[:i])
            accepted, correction = accept_or_resample(token, p, q, sampling.rng)
            if not accepted:
                break
            retained_count += 1
        retained = self.batch.rows[begin:begin + retained_count]
        emitted = tuple(row.input_token for row in retained)
        terminal = retained_count == request.remaining or emitted[-1] in request.eos_token_ids
        anchor, kind = None, None
        if not terminal:
            if correction is not None:
                anchor, kind = correction, "correction"
            else:
                anchor = sampling.select(self.logits[begin + retained_count - 1], proposals)
                kind = "bonus"
        return GreedyCommitIntent(
            request.owner, emitted, request.start + retained_count, retained, terminal, anchor, kind,
        )

    def commit(self, intent):
        decisions = intent if isinstance(intent, tuple) else (intent,)
        expected = (
            self._resolved if self.proposal_laws
            else self.batch.resolve(self.rows, omit_terminal=self.omit_terminal, early_rejection=self.early_rejection)
        )
        if self.committed or decisions != expected:
            raise ValueError("MTP commit must match the current verify decision exactly once")
        if self.batch_draft_seed:
            self._commit_k1_draft_batch(decisions)
        else:
            for index, (decision, table) in enumerate(zip(decisions, self.tables)):
                self._commit_owner(decision, table, self.batch.input_offsets[index])
        self.committed = True

    def _retained_target(self, intent, table, begin):
        count = len(intent.materialized_rows)
        start = self.rows[begin].row.position
        retained = self.rows[begin:begin + count]
        tokens = tuple(row.row.input_token for row in retained)
        hidden = self.verify_hidden
        if hidden is None:
            actual, hidden = self.transport._forward(
                self.transport.target, tokens, start, table
            )
            expected = tuple(row.predicted_token for row in retained)
            if actual != expected:
                raise RuntimeError(
                    f"MTP Target replay disagrees for {intent.owner} at {start}: "
                    f"verify={expected}, replay={actual}"
                )
        return tokens, start, hidden

    def _commit_k1_draft_batch(self, decisions):
        updates, segments, seed_owners = {}, [], []
        for index, (intent, table) in enumerate(zip(decisions, self.tables, strict=True)):
            tokens, start, hidden = self._retained_target(intent, table, self.batch.input_offsets[index])
            count = len(tokens)
            updates[intent.owner] = _DraftState(
                _slice_hidden(hidden, count - 1, 1), intent.next_anchor, intent.retain_length,
            )
            if intent.terminal:
                continue
            seed_next = self._has_next_proposal(intent, count, index)
            # The proposal already wrote the old anchor at start - 1.
            # Target HC at start + i drives the next Draft input at start + i.
            inputs = (*tokens[1:], intent.next_anchor) if seed_next else tokens[1:]
            if inputs:
                segments.append(_DraftSegment(inputs, start, table, _slice_hidden(hidden, 0, len(inputs))))
                seed_owners.append(intent.owner if seed_next else None)

        if segments and (any(seed_owners) or not self.transport.draft_cache_only):
            ids, _ = self.transport._forward_draft_segments(segments, return_hidden=False)
            for owner, token, segment in zip(seed_owners, ids, segments, strict=True):
                if owner is not None:
                    state = updates[owner]
                    page_count = (state.position - 1) // self.transport.page_size + 1
                    state.seed = _DraftSeed(token, (), segment.table[:page_count])
        else:
            # The native output-free policy accepts only a single span.
            for segment in segments:
                self.transport._write_draft_cache(segment.tokens, segment.start, segment.table, segment.hidden)
        self.transport.states.update(updates)

    def _commit_owner(self, intent, table, begin):
        if self.scratch_commit is not None:
            start = self.rows[begin].row.position
            tokens = tuple(row.input_token for row in intent.materialized_rows)
            self._commit_scratch(intent, table, start, tokens)
            return
        tokens, start, hidden = self._retained_target(intent, table, begin)
        count = len(tokens)
        seed = None
        if not intent.terminal:
            # The first Draft row used the previous Target hidden state.
            # Rebuild each subsequent retained row from the committed Target state.
            if self._seed_next_anchor(intent, count, begin):
                (token,), draft_hidden = self.transport._forward(
                    self.transport.draft, (*tokens[1:], intent.next_anchor), start, table,
                    draft_hidden=hidden, sample_all_positions=False, return_hidden=not self.k1_graph_seed,
                )
                page_count = (intent.retain_length - 1) // self.transport.page_size + 1
                seed_hidden = () if self.k1_graph_seed else _slice_hidden(draft_hidden, count - 1, 1)
                seed = _DraftSeed(token, seed_hidden, tuple(table[:page_count]))
            else:
                for offset in range(1, count):
                    self.transport._write_draft_cache(
                        tokens[offset:offset + 1], start + offset - 1, table,
                        _slice_hidden(hidden, offset - 1, 1),
                    )
        self.transport.states[intent.owner] = _DraftState(
            _slice_hidden(hidden, count - 1, 1), intent.next_anchor, intent.retain_length, seed=seed,
        )

    def _commit_scratch(self, intent, table, start, tokens):
        scratch = self.scratch_commit
        if (
            len(intent.materialized_rows) != len(scratch.hidden_rows)
            or intent.materialized_rows != tuple(row.row for row in self.rows)
            or intent.retain_length != start + len(scratch.hidden_rows)
        ):
            raise RuntimeError("MTP winner does not match the materialized scratch rows")
        page_size = self.transport.page_size
        first_page = start // page_size
        end_page = (intent.retain_length - 1) // page_size + 1
        if end_page > min(len(scratch.table), len(table)):
            raise RuntimeError("MTP winner page span exceeds its scratch or parent table")
        # The scheduler keeps both tables reserved until the native copy and Draft repair complete.
        self.transport.target.clone_v4_cache_pages(list(zip(
            scratch.table[first_page:end_page], table[first_page:end_page], strict=True,
        )))
        if not intent.terminal:
            for offset in range(1, len(tokens)):
                self.transport._write_draft_cache(
                    tokens[offset:offset + 1], start + offset - 1, table,
                    scratch.hidden_rows[offset - 1],
                )
        self.transport.states[intent.owner] = _DraftState(
            scratch.hidden_rows[-1], intent.next_anchor, intent.retain_length,
        )

    def _seed_next_anchor(self, intent, count, begin):
        """Fuse only when a retained suffix and another proposal both exist."""
        if count < 2 or not (self.transport.reuse_seed or self.k1_graph_seed):
            return False
        index = self.batch.input_offsets.index(begin)
        if self.proposal_laws and self.proposal_laws[index] is not None:
            return False
        return self._has_next_proposal(intent, count, index)

    def _has_next_proposal(self, intent, count, index):
        request = self.batch.requests[index]
        # The current anchor and the next published anchor both occupy budget.
        remaining_after_anchor = request.remaining - count - 1
        return remaining_after_anchor > 1 and intent.next_anchor not in request.eos_token_ids


class _V4Transport:
    def __init__(self, target, draft, core, page_size, *, direct_k1_commit=False,
                 direct_k23_commit=False, reuse_seed=False, draft_cache_only=False,
                 terminal_tail_trim=False, cpu_metadata_shadow=False,
                 shared_compression_plan=False, stepwise_direct_commit=False,
                 batched_draft=False, last_logits=False, k1_graph_seed=False):
        self.target = target
        self.draft = draft
        self.core = core
        self.page_size = page_size
        self.direct_k1_commit = direct_k1_commit
        self.direct_k23_commit = direct_k23_commit
        self.reuse_seed = reuse_seed
        self.draft_cache_only = draft_cache_only
        self.terminal_tail_trim = terminal_tail_trim
        self.cpu_metadata_shadow = cpu_metadata_shadow
        self.shared_compression_plan = shared_compression_plan
        self.stepwise_direct_commit = stepwise_direct_commit
        self.batched_draft = batched_draft
        self.last_logits = last_logits
        self.k1_graph_seed = k1_graph_seed
        self.states = {}
        self.transaction = None

    @property
    def owner(self):
        return next(iter(self.states)) if len(self.states) == 1 else None

    @property
    def hidden(self):
        state = self.states.get(self.owner)
        return state.hidden if state else None

    @property
    def anchor(self):
        state = self.states.get(self.owner)
        return state.anchor if state else None

    @property
    def position(self):
        state = self.states.get(self.owner)
        return state.position if state else 0

    @staticmethod
    def _decode(result, count):
        tokens = tuple(result["output_ids"].to_numpy().tolist())
        if len(tokens) != count or any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeError("MTP native forward returned invalid token IDs")
        return tokens, result.get("hidden_states_by_rank")

    def _slots(self, table, start, end):
        if start < 0 or start >= end or end > self.page_size * len(table):
            raise ValueError("MTP token span exceeds the scheduler's page table")
        return [table[pos // self.page_size] * self.page_size + pos % self.page_size
                for pos in range(start, end)]

    def _host_metadata_options(self):
        if self.shared_compression_plan:
            return {"mtp_shared_compression_plan": True}
        if self.cpu_metadata_shadow:
            return {"mtp_cpu_metadata_shadow": True}
        return {}

    def _forward(self, engine, tokens, start, table, *, draft_hidden=None,
                 return_hidden=True, rollback=False, sample_all_positions=True,
                 return_logits=False, cache_only=False):
        if cache_only and (
            engine is not self.draft or not draft_hidden or return_hidden or rollback or return_logits
        ):
            raise ValueError("Classic NextN cache-only is limited to an output-free Draft cache write")
        end = start + len(tokens)
        slots = self._slots(table, start, end)
        core = self.core
        last_logits = (
            self.last_logits and engine is self.draft and draft_hidden and len(tokens) > 1
            and return_hidden and not sample_all_positions and not return_logits and not rollback and not cache_only
        )
        result = engine.forward_raw(
            input_ids=core.from_list([list(tokens)], dtype=core.int64),
            position_ids=core.from_list(list(range(start, end)), dtype=core.int64),
            past_kv_lengths=core.from_list([start], dtype=core.int32),
            total_kv_lengths=core.from_list([end], dtype=core.int32),
            input_offsets=core.from_list([0, len(tokens)], dtype=core.int32),
            cu_seqlens=core.from_list([0, end], dtype=core.int32),
            block_tables=core.from_list([list(table)], dtype=core.int32),
            slot_mapping=core.from_list(slots, dtype=core.int64),
            sample_all_positions=sample_all_positions,
            top_k=1,
            return_draft_hidden_states=return_hidden,
            rollback_v4_cache=rollback,
            target_hidden_states_by_rank=draft_hidden,
            **({"mtp_draft_cache_only": True} if cache_only else {}),
            **({"mtp_last_logits": True} if last_logits else {}),
            **self._host_metadata_options(),
        )
        if last_logits and result.get("mtp_last_logits_used") is not True:
            raise RuntimeError("Classic NextN Draft seed did not select the last logits row")
        if cache_only:
            if result.get("mtp_draft_cache_only_used") is not True or any(
                result.get(key) is not None for key in ("output_ids", "logits", "hidden_states")
            ):
                raise RuntimeError("Classic NextN Draft cache-only did not return a completed cache write")
            return
        decoded = self._decode(result, len(tokens) if sample_all_positions else 1)
        if not return_logits:
            return decoded
        return (*decoded, host_logit_rows(result["logits"], len(tokens), engine.hf_config["vocab_size"]))

    def _write_draft_cache(self, tokens, start, table, hidden):
        self._forward(
            self.draft, tokens, start, table, draft_hidden=hidden, return_hidden=False,
            **({"cache_only": True} if self.draft_cache_only else {}),
        )

    def _join_draft_hidden(self, segments):
        core = self.core
        # Native returns completed rank-local views. Cat submits on the frontend
        # runtime for each device. Finish those writes before the Draft workers
        # consume the joined tensors on their separate rank streams.
        hidden = tuple(core.cat(rank_rows, dim=0) for rank_rows in zip(
            *(segment.hidden for segment in segments), strict=True,
        ))
        previous = core.get_device()
        try:
            for tensor in hidden:
                core.set_device(tensor.device)
                core.sync_stream()
        finally:
            core.set_device(previous)
        return hidden

    def _forward_draft_segments(self, segments, *, return_hidden):
        if len(segments) == 1:
            (segment,) = segments
            return self._forward(
                self.draft, segment.tokens, segment.start, segment.table,
                draft_hidden=segment.hidden, return_hidden=return_hidden, sample_all_positions=False,
            )
        tokens, positions, past, total, slots = [], [], [], [], []
        offsets, cumulative = [0], [0]
        tables = []
        width = max(len(segment.table) for segment in segments)
        for segment in segments:
            start, table = segment.start, segment.table
            end = start + len(segment.tokens)
            tokens.extend(segment.tokens)
            positions.extend(range(start, end))
            slots.extend(self._slots(table, start, end))
            past.append(start)
            total.append(end)
            offsets.append(len(tokens))
            cumulative.append(cumulative[-1] + end)
            tables.append(list(table) + [-1] * (width - len(table)))
        if len(set(slots)) != len(slots):
            raise ValueError("MTP packed Draft has overlapping physical slots")
        core = self.core
        result = self.draft.forward_raw(
            input_ids=core.from_list([tokens], dtype=core.int64),
            position_ids=core.from_list(positions, dtype=core.int64),
            past_kv_lengths=core.from_list(past, dtype=core.int32),
            total_kv_lengths=core.from_list(total, dtype=core.int32),
            input_offsets=core.from_list(offsets, dtype=core.int32),
            cu_seqlens=core.from_list(cumulative, dtype=core.int32),
            block_tables=core.from_list(tables, dtype=core.int32),
            slot_mapping=core.from_list(slots, dtype=core.int64),
            sample_all_positions=False,
            top_k=1,
            return_draft_hidden_states=return_hidden,
            target_hidden_states_by_rank=self._join_draft_hidden(segments),
            **self._host_metadata_options(),
        )
        return self._decode(result, len(segments))

    def _forward_packed(self, engine, batch, tables, *, return_hidden=False, rollback=False,
                        return_logits=False, scratch_step=False):
        width = max(map(len, tables))
        slots = []
        for table, start, end in zip(tables, batch.past_kv_lengths, batch.total_kv_lengths):
            slots.extend(self._slots(table, start, end))
        if len(set(slots)) != len(slots):
            raise ValueError("MTP packed verify has overlapping physical slots")
        core = self.core
        result = engine.forward_raw(
            input_ids=core.from_list([list(batch.input_ids)], dtype=core.int64),
            position_ids=core.from_list(list(batch.position_ids), dtype=core.int64),
            past_kv_lengths=core.from_list(list(batch.past_kv_lengths), dtype=core.int32),
            total_kv_lengths=core.from_list(list(batch.total_kv_lengths), dtype=core.int32),
            input_offsets=core.from_list(list(batch.input_offsets), dtype=core.int32),
            cu_seqlens=core.from_list(list(batch.cu_seqlens), dtype=core.int32),
            block_tables=core.from_list(
                [list(table) + [-1] * (width - len(table)) for table in tables], dtype=core.int32,
            ),
            slot_mapping=core.from_list(slots, dtype=core.int64),
            sample_all_positions=True,
            top_k=1,
            return_draft_hidden_states=return_hidden,
            rollback_v4_cache=rollback,
            **self._host_metadata_options(),
            **({"mtp_scratch_step": True} if scratch_step else {}),
        )
        decoded = self._decode(result, len(batch.rows))
        if not return_logits:
            return decoded
        return (*decoded, host_logit_rows(result["logits"], len(batch.rows), engine.hf_config["vocab_size"]))

    def prefill(self, owner, model_input, *, sampling=None):
        (anchor,) = self.prefill_many((owner,), model_input, sampling=(sampling,) if sampling is not None else None)
        return anchor

    def prefill_many(self, owners, model_input, *, sampling=None):
        count = len(owners)
        if len(set(owners)) != count or any(not owner or owner in self.states for owner in owners):
            raise ValueError("MTP prefill requires distinct unused request owners")
        if sampling is not None and len(sampling) != count:
            raise ValueError("MTP stochastic prefill needs one sampling state per owner")
        (tokens,) = model_input["input_ids"].to_numpy().tolist()
        tables = model_input["block_tables"].to_numpy().tolist()
        offsets = (
            (0, len(tokens)) if count == 1 else model_input["input_offsets"].to_numpy().tolist()
        )
        if (
            model_input["past_kv_lengths"].to_numpy().tolist() != [0] * count
            or len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(tokens)
            or len(tables) != count or any(start >= end for start, end in zip(offsets, offsets[1:]))
        ):
            raise ValueError("MTP requires a fresh prefill without prefix reuse")
        model_input = {**model_input, "sample_all_positions": False, "top_k": 1,
                       "return_draft_hidden_states": True}
        last_logits = (
            self.last_logits and len(tokens) > count
            and (sampling is None or all(state is None for state in sampling))
        )
        if last_logits:
            model_input["mtp_last_logits"] = True
        result = self.target.forward_raw(**model_input)
        if last_logits and result.get("mtp_last_logits_used") is not True:
            raise RuntimeError("Classic NextN Target prefill did not select each owner's last logits row")
        anchors, hidden = self._decode(result, count)
        anchors = list(anchors)
        for index, (owner, table) in enumerate(zip(owners, tables)):
            start, end = offsets[index:index + 2]
            if sampling is not None and sampling[index] is not None:
                logits = host_logit_rows(result["logits"], len(tokens), self.target.hf_config["vocab_size"],
                                         start=end - 1, count=1)
                anchors[index] = sampling[index].select(logits[0])
            if end - start > 1:
                self._write_draft_cache(tokens[start + 1:end], 0, table,
                                        _slice_hidden(hidden, start, end - start - 1))
            self.states[owner] = _DraftState(_slice_hidden(hidden, end - 1, 1), anchors[index], end - start)
        return tuple(anchors)

    def propose(self, owner, anchor, position, block_table):
        (proposal,) = self.propose_many(owner, anchor, position, block_table, 1, ())
        return proposal

    def propose_many(self, owner, anchor, position, block_table, count, eos, *, sampling=None):
        if type(count) is not int or count not in (1, 2, 3):
            raise ValueError("MTP proposal depth must be 1, 2, or 3")
        state = self._ready_state(owner, anchor, position)
        token, hidden = anchor, state.hidden
        proposals = []
        q_rows = []
        seed = self._take_seed(state, block_table, count)
        if sampling is None and seed is not None:
            token, hidden = seed.token, seed.hidden
            proposals.append(token)
        # The uncompressed Draft writes only its current slot and reads up to
        # that position. Unaccepted suffix slots fall beyond the next length.
        for depth in range(len(proposals), count):
            if proposals and token in eos:
                break
            result = self._forward(
                self.draft, (token,), position - 1 + depth, block_table,
                draft_hidden=hidden, return_hidden=depth + 1 < count,
                sample_all_positions=False, return_logits=sampling is not None,
            )
            (token,), hidden = result[:2]
            if sampling is not None:
                q = sampling.distribution(result[2][0], proposals)
                token = sample(q, sampling.rng)
                q_rows.append(q)
            proposals.append(token)
        state.proposal = tuple(proposals)
        state.proposal_table = tuple(block_table)
        if sampling is not None:
            state.proposal_law = (sampling, tuple(q_rows))
        return state.proposal

    @staticmethod
    def _take_seed(state, table, count):
        seed, state.seed = state.seed, None
        if seed is not None and (count == 1 or seed.hidden) and seed.pages == tuple(table[:len(seed.pages)]):
            return seed
        return None

    def propose_batched(self, requests, tables, counts):
        states, chains, hidden = [], [], []
        for request, table, count in zip(requests, tables, counts, strict=True):
            state = self._ready_state(request.owner, request.anchor, request.start)
            seed = self._take_seed(state, table, count)
            states.append(state)
            chains.append([seed.token] if seed is not None else [])
            hidden.append(seed.hidden if seed is not None else state.hidden)
        for depth in range(max(counts)):
            active, segments = [], []
            for index, (request, table, count) in enumerate(zip(requests, tables, counts, strict=True)):
                chain = chains[index]
                token = chain[-1] if chain else request.anchor
                if len(chain) == depth < count and (not chain or token not in request.eos_token_ids):
                    active.append(index)
                    segments.append(_DraftSegment((token,), request.start - 1 + depth, table, hidden[index]))
            if not segments:
                continue
            next_depth = any(depth + 1 < counts[index] for index in active)
            ids, rows = self._forward_draft_segments(segments, return_hidden=next_depth)
            if next_depth and (
                not rows or len(rows) != len(segments[0].hidden)
                or any(tensor.shape[0] != len(active) for tensor in rows)
            ):
                raise RuntimeError("MTP packed Draft must return one HC row per live owner and rank")
            for row, (index, token) in enumerate(zip(active, ids, strict=True)):
                chains[index].append(token)
                if depth + 1 < counts[index] and token not in requests[index].eos_token_ids:
                    hidden[index] = _slice_hidden(rows, row, 1)
        for state, table, chain in zip(states, tables, chains, strict=True):
            state.proposal = tuple(chain)
            state.proposal_table = table
        return tuple(state.proposal for state in states)

    def _ready_state(self, owner, anchor, position):
        state = self.states.get(owner)
        if (
            state is None or anchor != state.anchor or position != state.position
            or state.proposal is not None or self.transaction is not None
        ):
            raise ValueError("MTP step disagrees with the current owner, anchor or position")
        return state

    def target_tail(self, owner, anchor, position, table, *, sampling=None):
        self._ready_state(owner, anchor, position)
        if sampling is not None:
            _, _, logits = self._forward(self.target, (anchor,), position, table,
                                         return_hidden=False, return_logits=True)
            return sampling.select(logits[0])
        (token,), _ = self._forward(self.target, (anchor,), position, table, return_hidden=False)
        return token

    @contextmanager
    def target_transaction(self, batch, block_table, *, batch_draft_seed=False):
        tables = (tuple(block_table),) if len(batch.requests) == 1 else tuple(block_table)
        if self.transaction is not None or len(tables) != len(batch.requests):
            raise ValueError("MTP verify must match the current owner and proposal")
        for request, table in zip(batch.requests, tables):
            state = self.states.get(request.owner)
            if state is None or (request.anchor, request.start, request.draft_tokens, tuple(table)) != (
                state.anchor, state.position, state.proposal, state.proposal_table,
            ):
                raise ValueError("MTP verify must match the current owner and proposal")
        verify_hidden = None
        proposal_laws = tuple(self.states[request.owner].proposal_law for request in batch.requests)
        if not any(law is not None for law in proposal_laws):
            proposal_laws = ()
        stochastic = bool(proposal_laws)
        omit_terminal = (
            self.terminal_tail_trim and not stochastic and len(self.states) == 1
            and batch.ends_in_known_stop
        )
        if len(tables) == 1:
            # A K1 rejection advances to the first discarded position. Deeper
            # verifies need an expanded ring to keep the history for that input
            # live after the whole speculative suffix has written.
            query = len(batch.rows)
            direct = not omit_terminal and not stochastic and len(self.states) == 1 and (
                (self.direct_k1_commit and query == 2)
                or (self.direct_k23_commit and 2 <= query <= 4)
            )
            result = self._forward(
                self.target, batch.input_ids[:-1] if omit_terminal else batch.input_ids,
                batch.position_ids[0], tables[0],
                return_hidden=direct, rollback=not direct, return_logits=stochastic,
            )
            if direct:
                hidden = result[1]
                if not hidden:
                    raise RuntimeError("Direct MTP verification requires Target hidden states")
                verify_hidden = hidden
        else:
            result = self._forward_packed(self.target, batch, tables, rollback=True, return_logits=stochastic)
        transaction = _TargetTransaction(
            self, batch, tables, result[0], verify_hidden, proposal_laws, result[2] if stochastic else None,
            omit_terminal=omit_terminal, batch_draft_seed=batch_draft_seed,
            k1_graph_seed=self.k1_graph_seed and not stochastic and all(
                len(request.draft_tokens) == 1 for request in batch.requests
            ),
        )
        with self._publishing(transaction):
            yield transaction

    def _check_branch_trials(self, branches, block_table):
        table = tuple(block_table)
        owner = self.owner
        if self.transaction is not None or owner is None or not branches:
            raise ValueError("MTP branching requires one uncommitted owner")
        state = self.states[owner]
        proposal = state.proposal
        if not proposal or state.proposal_table != table or state.proposal_law is not None:
            raise ValueError("MTP branching requires the current draft chain")
        max_depth = 0
        (first,) = branches[0].requests
        for branch in branches:
            (request,) = branch.requests
            depth = len(request.draft_tokens)
            if (
                request.owner != owner
                or request.anchor != state.anchor
                or request.start != state.position
                or request.remaining != first.remaining
                or request.eos_token_ids != first.eos_token_ids
                or not 1 <= depth <= len(proposal)
                or request.draft_tokens != proposal[:depth]
            ):
                raise ValueError("MTP branch differs from its owner draft prefix")
            max_depth = max(max_depth, depth)
        if max_depth != len(proposal):
            raise ValueError("MTP branching lacks its longest candidate")
        return table

    def _branch_winner(self, branches, table, predictions, *, stepwise=False, scratch_commits=None):
        winner = None
        for index, (branch, tokens) in enumerate(zip(branches, predictions, strict=True)):
            omit_terminal = (
                stepwise and branch.ends_in_known_stop and len(tokens) == len(branch.rows) - 1
                and tokens[-1] == branch.rows[-1].input_token
            )
            early_rejection = stepwise and not omit_terminal
            transaction = _TargetTransaction(
                self, branch, (table,), tokens,
                omit_terminal=omit_terminal, early_rejection=early_rejection,
                scratch_commit=scratch_commits[index] if scratch_commits is not None else None,
            )
            (intent,) = branch.resolve(transaction.rows, omit_terminal=omit_terminal,
                                       early_rejection=early_rejection)
            if winner is None or len(intent.emitted_tokens) > len(winner[1].emitted_tokens):
                winner = (transaction, intent)
        return winner

    @contextmanager
    def serial_branch_transaction(self, branches, block_table):
        """Verify each depth against one rolled-back Target prefix."""
        table = self._check_branch_trials(branches, block_table)
        # All drafts share the same owner, anchor and deterministic Draft prefix.
        # Only the Target query varies. Native rollback restores each trial before
        # the next, and only the longest accepted branch publishes its state.
        predictions = (
            self._forward(
                self.target, branch.input_ids, branch.position_ids[0], table,
                return_hidden=False, rollback=True,
            )[0]
            for branch in branches
        )
        winner = self._branch_winner(branches, table, predictions)
        with self._publishing(winner[0]):
            yield winner

    @contextmanager
    def packed_branch_transaction(self, branches, block_table, scratch_tables, max_context):
        """Verify candidate depths in one Target call on private copies of the parent cache."""
        table = self._check_branch_trials(branches, block_table)
        packed = GreedyVerifyBatch.pack(
            (replace(branch.requests[0], owner=f"branch-{index}")
             for index, branch in enumerate(branches)),
            max_context,
        )
        self._clone_branch_prefixes(branches, table, scratch_tables)
        tokens, _ = self._forward_packed(
            self.target, packed, scratch_tables, return_hidden=False, rollback=True,
        )
        predictions = (tokens[start:end] for start, end in zip(packed.input_offsets, packed.input_offsets[1:]))
        winner = self._branch_winner(branches, table, predictions)
        with self._publishing(winner[0]):
            yield winner

    def _clone_branch_prefixes(self, branches, table, scratch_tables):
        width = (branches[0].past_kv_lengths[0] + self.page_size - 1) // self.page_size
        if len(scratch_tables) != len(branches) or any(
            len(scratch) < (branch.total_kv_lengths[0] + self.page_size - 1) // self.page_size
            for scratch, branch in zip(scratch_tables, branches)
        ):
            raise ValueError("MTP scratch page table cannot hold its branch")
        copies = [(src, dst) for scratch in scratch_tables for src, dst in zip(table[:width], scratch[:width])]
        if copies:
            self.target.clone_v4_cache_pages(copies)

    @contextmanager
    def stepwise_branch_transaction(self, branches, block_table, scratch_tables, max_context):
        """Advance only live siblings on their private pages."""
        table = self._check_branch_trials(branches, block_table)
        self._clone_branch_prefixes(branches, table, scratch_tables)
        predictions = [[] for _ in branches]
        direct_commit = self.stepwise_direct_commit
        hidden_rows = [[] for _ in branches] if direct_commit else None
        if direct_commit:
            rank_count = len(self.states[branches[0].requests[0].owner].hidden)
        active = tuple(range(len(branches)))
        for step in range(max(len(branch.rows) for branch in branches)):
            batch = GreedyVerifyBatch.pack(
                (replace(branches[index].requests[0], owner=f"branch-{index}",
                         start=branches[index].rows[step].position,
                         anchor=branches[index].rows[step].input_token, remaining=1, draft_tokens=())
                 for index in active),
                max_context, scratch_step=True,
            )
            tokens, hidden = self._forward_packed(
                self.target, batch, tuple(scratch_tables[index] for index in active),
                scratch_step=True, return_hidden=direct_commit,
            )
            if direct_commit and (
                not hidden or len(hidden) != rank_count
                or any(tensor.shape[0] != len(active) for tensor in hidden)
            ):
                raise RuntimeError("MTP scratch q1 did not return a hidden row per active branch and rank")
            continuing = []
            for row_index, (index, predicted) in enumerate(zip(active, tokens, strict=True)):
                predictions[index].append(predicted)
                if direct_commit:
                    hidden_rows[index].append(_slice_hidden(hidden, row_index, 1))
                branch = branches[index]
                next_row = step + 1
                if (next_row < len(branch.rows) and predicted == branch.rows[next_row].input_token
                        and predicted not in branch.requests[0].eos_token_ids):
                    continuing.append(index)
            if not continuing:
                break
            active = tuple(continuing)
        scratch_commits = None
        if direct_commit:
            scratch_commits = tuple(
                _ScratchCommit(tuple(scratch), tuple(rows))
                for scratch, rows in zip(scratch_tables, hidden_rows, strict=True)
            )
        winner = self._branch_winner(
            branches, table, predictions, stepwise=True, scratch_commits=scratch_commits,
        )
        with self._publishing(winner[0]):
            yield winner

    @contextmanager
    def _publishing(self, transaction):
        self.transaction = transaction
        try:
            yield
            if not transaction.committed:
                raise RuntimeError("MTP transaction ended before committing its retained prefix")
        finally:
            self.transaction = None

    def release(self, owner):
        if not self.states:
            return
        if owner not in self.states or self.transaction is not None:
            raise ValueError("MTP cache lease belongs to a different owner or a live transaction")
        del self.states[owner]

    def close(self):
        self.states.clear()
        self.draft = self.target = None


def require_classic_v4_mtp_abi():
    from infinilm.infer_engine import InferEngine

    if not hasattr(InferEngine.Output, "hidden_states_by_rank"):
        raise RuntimeError("The native extension lacks the classic V4 MTP hidden-state ABI")


def _draft_cache_only_enabled(config, engine_type):
    enabled = os.environ.get("INFINILM_MTP_DRAFT_CACHE_ONLY") == "1" and not config.enable_graph
    if enabled and getattr(engine_type, "mtp_draft_cache_only_version", 0) != 1:
        raise RuntimeError("Classic NextN Draft cache-only requires the native output policy")
    return enabled


def _host_metadata_modes(config, engine_type):
    shadow = (
        os.environ.get("INFINILM_MTP_CPU_METADATA_SHADOW") == "1" and not config.enable_graph
    )
    shared = (
        os.environ.get("INFINILM_MTP_SHARED_COMPRESSION_PLAN") == "1" and not config.enable_graph
    )
    if (shadow or shared) and getattr(engine_type, "mtp_cpu_metadata_shadow_version", 0) != 1:
        raise RuntimeError("Classic NextN requires native host metadata shadow support")
    if shared and getattr(engine_type, "mtp_shared_compression_plan_version", 0) != 1:
        raise RuntimeError("Classic NextN requires native shared compression plan support")
    return shadow, shared


def _direct_k1_commit_enabled(config):
    return (
        os.environ.get("INFINILM_MTP_EAGER_K1_DIRECT_COMMIT") == "1"
        and not config.enable_graph
        and config.max_batch_size == 1
        and config.num_draft_tokens == 1
        and config.nextn_branch_draft_lengths is None
    )


def _batched_draft_enabled(config):
    return (
        os.environ.get("INFINILM_MTP_BATCHED_DRAFT") == "1"
        and not config.enable_graph
        and config.nextn_branch_draft_lengths is None
    )


def _k1_graph_seed_enabled(config, engine_type):
    enabled = (
        os.environ.get("INFINILM_MTP_K1_GRAPH_SEED") == "1"
        and config.enable_graph
        and config.max_batch_size in (1, 2, 3)
        and config.num_draft_tokens == 1
        and config.nextn_branch_draft_lengths is None
    )
    if enabled and getattr(engine_type, "mtp_k1_graph_seed_version", 0) != 1:
        raise RuntimeError("Classic NextN K1 Graph seed requires native Draft q2 graph support")
    return enabled


def _direct_k23_commit_enabled(config, target_engine):
    if (
        os.environ.get("INFINILM_MTP_EAGER_K23_DIRECT_COMMIT") != "1"
        or config.enable_graph
        or config.max_batch_size != 1
        or config.num_draft_tokens not in (2, 3)
        or config.nextn_branch_draft_lengths is not None
    ):
        return False
    if getattr(target_engine, "deepseek_v4_raw_ring_rows", None) != (16, 256):
        raise RuntimeError("Direct K2/K3 verification requires the expanded native compressor rings")
    return True


def create_deepseek_v4_mtp_transport(config, target_engine, device):
    from infinilm import _core
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    require_classic_v4_mtp_abi()
    if target_engine.model_type != "deepseek_v4":
        raise ValueError("classic MTP requires DeepSeek V4 Target")
    if config.nextn_branch_execution in ("packed", "packed-stepwise") and not callable(
        getattr(target_engine, "clone_v4_cache_pages", None)
    ):
        raise RuntimeError("Packed classic NextN branches require the native V4 page clone API")
    if (config.nextn_branch_execution == "packed-stepwise"
            and getattr(target_engine, "mtp_scratch_step_version", 0) != 1):
        raise RuntimeError("stepwise packed classic NextN requires native scratch q1 support")
    stepwise_direct_commit = (
        config.nextn_branch_execution == "packed-stepwise"
        and os.getenv("INFINILM_MTP_STEPWISE_DIRECT_COMMIT") == "1"
    )
    if stepwise_direct_commit and getattr(target_engine, "mtp_scratch_hc_version", 0) != 1:
        raise RuntimeError("Stepwise direct commit requires native owned scratch q1 hidden states")
    direct_k23_commit = _direct_k23_commit_enabled(config, target_engine)
    draft_cache_only = _draft_cache_only_enabled(config, InferEngine)
    batched_draft = _batched_draft_enabled(config)
    k1_graph_seed = _k1_graph_seed_enabled(config, InferEngine)
    last_logits = os.getenv("INFINILM_MTP_LAST_LOGITS") == "1" and not config.enable_graph
    if last_logits and getattr(InferEngine, "mtp_last_logits_version", 0) != 1:
        raise RuntimeError("Classic NextN last logits require native head selection support")
    cpu_metadata_shadow, shared_compression_plan = _host_metadata_modes(config, InferEngine)
    draft = InferEngine(
        config.draft_model_path,
        device=device,
        distributed_config=DistConfig(config.tensor_parallel_size),
        cache_config=PagedKVCacheConfig(
            num_blocks=config.num_blocks, block_size=config.block_size,
            max_batch_size=config.max_batch_size if batched_draft else 1,
        ),
        enable_graph_compiling=config.enable_graph,
        attention_backend="paged-attn",
        classic_mtp_draft=True,
        weight_load_mode=config.weight_load_mode,
        pre_transpose=config.pre_transpose,
    )
    load_model_state_dict_by_file(draft, config.draft_model_path, dtype=draft.dtype)
    reuse_seed = (
        os.getenv("INFINILM_V4_MTP_K23_SEED") == "1"
        and not config.enable_graph and config.num_draft_tokens in (2, 3)
        and config.nextn_branch_draft_lengths is None
    )
    direct_k1_commit = _direct_k1_commit_enabled(config)
    terminal_tail_trim = (
        os.getenv("INFINILM_MTP_TARGET_TAIL_TRIM") == "1" and not config.enable_graph
        and config.max_batch_size == 1 and config.nextn_branch_draft_lengths is None
    )
    logger.info(
        "Classic NextN K=%s graph=%s branches=%s branch_execution=%s direct_k1=%s direct_k23=%s "
        "draft_seed=%s draft_cache_only=%s batched_draft=%s k1_graph_seed=%s last_logits=%s terminal_tail_trim=%s "
        "cpu_metadata_shadow=%s shared_compression_plan=%s stepwise_direct_commit=%s",
        config.num_draft_tokens, config.enable_graph, config.nextn_branch_draft_lengths,
        config.nextn_branch_execution, direct_k1_commit, direct_k23_commit, reuse_seed,
        draft_cache_only, batched_draft, k1_graph_seed, last_logits, terminal_tail_trim,
        cpu_metadata_shadow or shared_compression_plan,
        shared_compression_plan, stepwise_direct_commit,
    )
    return _V4Transport(
        target_engine, draft, _core, config.block_size,
        direct_k1_commit=direct_k1_commit, direct_k23_commit=direct_k23_commit,
        reuse_seed=reuse_seed, draft_cache_only=draft_cache_only,
        terminal_tail_trim=terminal_tail_trim,
        cpu_metadata_shadow=cpu_metadata_shadow, shared_compression_plan=shared_compression_plan,
        stepwise_direct_commit=stepwise_direct_commit, batched_draft=batched_draft, last_logits=last_logits,
        k1_graph_seed=k1_graph_seed,
    )
