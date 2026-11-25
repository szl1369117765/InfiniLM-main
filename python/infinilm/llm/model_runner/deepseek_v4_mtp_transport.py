"""Owner-bound eager classic NextN transport for a V4 engine."""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass

from .deepseek_v4_mtp_sampling import accept_or_resample, host_logit_rows, sample
from .deepseek_v4_mtp_verify import GreedyCommitIntent, TargetVerifyRow


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


class _TargetTransaction:
    def __init__(self, transport, batch, tables, tokens, verify_hidden=None,
                 proposal_laws=(), logits=None):
        self.transport = transport
        self.batch = batch
        self.tables = tables
        self.rows = tuple(TargetVerifyRow(row, token) for row, token in zip(batch.rows, tokens))
        self.verify_hidden = verify_hidden
        self.proposal_laws = proposal_laws
        self.logits = logits
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
        expected = self._resolved if self.proposal_laws else self.batch.resolve(self.rows)
        if self.committed or decisions != expected:
            raise ValueError("MTP commit must match the current verify decision exactly once")
        for index, (decision, table) in enumerate(zip(decisions, self.tables)):
            self._commit_owner(decision, table, self.batch.input_offsets[index])
        self.committed = True

    def _commit_owner(self, intent, table, begin):
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
        seed = None
        if not intent.terminal:
            # The first Draft row used the previous Target hidden state.
            # Rebuild each subsequent retained row from the committed Target state.
            if self._seed_next_anchor(intent, count, begin):
                (token,), draft_hidden = self.transport._forward(
                    self.transport.draft, (*tokens[1:], intent.next_anchor), start, table,
                    draft_hidden=hidden, sample_all_positions=False,
                )
                page_count = (intent.retain_length - 1) // self.transport.page_size + 1
                seed = _DraftSeed(token, _slice_hidden(draft_hidden, count - 1, 1), tuple(table[:page_count]))
            else:
                for offset in range(1, count):
                    self.transport._forward(
                        self.transport.draft, tokens[offset:offset + 1], start + offset - 1, table,
                        draft_hidden=_slice_hidden(hidden, offset - 1, 1), return_hidden=False,
                    )
        self.transport.states[intent.owner] = _DraftState(
            _slice_hidden(hidden, count - 1, 1), intent.next_anchor, intent.retain_length, seed=seed,
        )

    def _seed_next_anchor(self, intent, count, begin):
        """Fuse only when a retained suffix and another proposal both exist."""
        if not self.transport.reuse_seed or count < 2:
            return False
        index = self.batch.input_offsets.index(begin)
        if self.proposal_laws and self.proposal_laws[index] is not None:
            return False
        request = self.batch.requests[index]
        # The current Target replay owns every HC row used by the fused repair.
        # The next anchor runs in the last retained page, including after truncation.
        remaining_after_anchor = request.remaining - count - 1
        return remaining_after_anchor > 1 and intent.next_anchor not in request.eos_token_ids


class _V4Transport:
    def __init__(self, target, draft, core, page_size, *, direct_k1_commit=False,
                 direct_k23_commit=False, reuse_seed=False):
        self.target = target
        self.draft = draft
        self.core = core
        self.page_size = page_size
        self.direct_k1_commit = direct_k1_commit
        self.direct_k23_commit = direct_k23_commit
        self.reuse_seed = reuse_seed
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

    def _forward(self, engine, tokens, start, table, *, draft_hidden=None,
                 return_hidden=True, rollback=False, sample_all_positions=True,
                 return_logits=False):
        end = start + len(tokens)
        slots = self._slots(table, start, end)
        core = self.core
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
        )
        decoded = self._decode(result, len(tokens) if sample_all_positions else 1)
        if not return_logits:
            return decoded
        return (*decoded, host_logit_rows(result["logits"], len(tokens), engine.hf_config["vocab_size"]))

    def _forward_packed(self, engine, batch, tables, *, return_hidden=False, rollback=False,
                        return_logits=False):
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
        result = self.target.forward_raw(**model_input)
        anchors, hidden = self._decode(result, count)
        anchors = list(anchors)
        for index, (owner, table) in enumerate(zip(owners, tables)):
            start, end = offsets[index:index + 2]
            if sampling is not None and sampling[index] is not None:
                logits = host_logit_rows(result["logits"], len(tokens), self.target.hf_config["vocab_size"],
                                         start=end - 1, count=1)
                anchors[index] = sampling[index].select(logits[0])
            if end - start > 1:
                self._forward(self.draft, tokens[start + 1:end], 0, table,
                              draft_hidden=_slice_hidden(hidden, start, end - start - 1),
                              return_hidden=False)
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
        seed = state.seed
        state.seed = None
        if sampling is None and seed is not None and seed.pages == tuple(block_table[:len(seed.pages)]):
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
    def target_transaction(self, batch, block_table):
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
        if len(tables) == 1:
            # A K1 rejection advances to the first discarded position. Deeper
            # verifies need an expanded ring to keep the history for that input
            # live after the whole speculative suffix has written.
            query = len(batch.rows)
            direct = not stochastic and len(self.states) == 1 and (
                (self.direct_k1_commit and query == 2)
                or (self.direct_k23_commit and 2 <= query <= 4)
            )
            result = self._forward(
                self.target, batch.input_ids, batch.position_ids[0], tables[0],
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
        )
        with self._publishing(transaction):
            yield transaction

    @contextmanager
    def serial_branch_transaction(self, branches, block_table):
        """Verify each depth against one rolled-back Target prefix."""
        table = tuple(block_table)
        owner = self.owner
        if self.transaction is not None or owner is None or not branches:
            raise ValueError("serial MTP branching requires one uncommitted owner")
        state = self.states[owner]
        proposal = state.proposal
        if not proposal or state.proposal_table != table or state.proposal_law is not None:
            raise ValueError("serial MTP branching requires the current draft chain")
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
                raise ValueError("serial MTP branch differs from its owner draft prefix")
            max_depth = max(max_depth, depth)
        if max_depth != len(proposal):
            raise ValueError("serial MTP branching lacks its longest candidate")

        # All drafts share the same owner, anchor and deterministic Draft prefix.
        # Only the Target query varies. Native rollback restores each trial before
        # the next, and only the longest accepted branch publishes its state.
        winner = None
        for branch in branches:
            tokens, _ = self._forward(
                self.target, branch.input_ids, branch.position_ids[0], table,
                return_hidden=False, rollback=True,
            )
            transaction = _TargetTransaction(self, branch, (table,), tokens)
            (intent,) = branch.resolve(transaction.rows)
            if winner is None or intent.retain_length > winner[1].retain_length:
                winner = (transaction, intent)
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


def _direct_k1_commit_enabled(config):
    return (
        os.environ.get("INFINILM_MTP_EAGER_K1_DIRECT_COMMIT") == "1"
        and not config.enable_graph
        and config.max_batch_size == 1
        and config.num_draft_tokens == 1
        and config.nextn_branch_draft_lengths is None
    )


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
    direct_k23_commit = _direct_k23_commit_enabled(config, target_engine)
    draft = InferEngine(
        config.draft_model_path,
        device=device,
        distributed_config=DistConfig(config.tensor_parallel_size),
        cache_config=PagedKVCacheConfig(
            num_blocks=config.num_blocks, block_size=config.block_size, max_batch_size=1
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
    logger.info("Classic NextN K=%s graph=%s serial_branches=%s direct_k1=%s direct_k23=%s draft_seed=%s",
                config.num_draft_tokens, config.enable_graph, config.nextn_branch_draft_lengths,
                direct_k1_commit, direct_k23_commit, reuse_seed)
    return _V4Transport(
        target_engine, draft, _core, config.block_size,
        direct_k1_commit=direct_k1_commit, direct_k23_commit=direct_k23_commit,
        reuse_seed=reuse_seed,
    )
