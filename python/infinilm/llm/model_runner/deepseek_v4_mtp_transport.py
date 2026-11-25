"""Owner-bound eager classic NextN transport for a V4 engine."""

from contextlib import contextmanager
from dataclasses import dataclass

from .deepseek_v4_mtp_verify import TargetVerifyRow


def _slice_hidden(tensors, start, length):
    return tuple(tensor.narrow(0, start, length) for tensor in tensors)


@dataclass
class _DraftState:
    hidden: tuple
    anchor: int | None
    position: int
    proposal: tuple[int, ...] | None = None
    proposal_table: tuple[int, ...] | None = None


class _TargetTransaction:
    def __init__(self, transport, batch, tables, tokens):
        self.transport = transport
        self.batch = batch
        self.tables = tables
        self.rows = tuple(TargetVerifyRow(row, token) for row, token in zip(batch.rows, tokens))
        self.committed = False

    def commit(self, intent):
        decisions = intent if isinstance(intent, tuple) else (intent,)
        if self.committed or decisions != self.batch.resolve(self.rows):
            raise ValueError("MTP commit must match the current verify decision exactly once")
        for index, (decision, table) in enumerate(zip(decisions, self.tables)):
            self._replay_owner(decision, table, self.batch.input_offsets[index])
        self.committed = True

    def _replay_owner(self, intent, table, begin):
        count = len(intent.materialized_rows)
        start = self.rows[begin].row.position
        retained = self.rows[begin:begin + count]
        tokens = tuple(row.row.input_token for row in retained)
        actual, hidden = self.transport._forward(
            self.transport.target, tokens, start, table
        )
        expected = tuple(row.predicted_token for row in retained)
        if actual != expected:
            raise RuntimeError(
                f"MTP Target replay disagrees for {intent.owner} at {start}: "
                f"verify={expected}, replay={actual}"
            )
        if not intent.terminal:
            # The first Draft row used the previous Target hidden state.
            # Rebuild each subsequent retained row from the replayed Target state.
            for offset in range(1, count):
                self.transport._forward(
                    self.transport.draft, tokens[offset:offset + 1], start + offset - 1, table,
                    draft_hidden=_slice_hidden(hidden, offset - 1, 1), return_hidden=False,
                )
        self.transport.states[intent.owner] = _DraftState(
            _slice_hidden(hidden, count - 1, 1), intent.next_anchor, intent.retain_length,
        )


class _V4Transport:
    def __init__(self, target, draft, core, page_size):
        self.target = target
        self.draft = draft
        self.core = core
        self.page_size = page_size
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
                 return_hidden=True, rollback=False, sample_all_positions=True):
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
        return self._decode(result, len(tokens))

    def _forward_packed(self, engine, batch, tables, *, return_hidden=False, rollback=False):
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
        return self._decode(result, len(batch.rows))

    def prefill(self, owner, model_input):
        (anchor,) = self.prefill_many((owner,), model_input)
        return anchor

    def prefill_many(self, owners, model_input):
        count = len(owners)
        if len(set(owners)) != count or any(not owner or owner in self.states for owner in owners):
            raise ValueError("MTP prefill requires distinct unused request owners")
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
        anchors, hidden = self._decode(self.target.forward_raw(**model_input), count)
        for index, (owner, anchor, table) in enumerate(zip(owners, anchors, tables)):
            start, end = offsets[index:index + 2]
            if end - start > 1:
                self._forward(self.draft, tokens[start + 1:end], 0, table,
                              draft_hidden=_slice_hidden(hidden, start, end - start - 1),
                              return_hidden=False)
            self.states[owner] = _DraftState(_slice_hidden(hidden, end - 1, 1), anchor, end - start)
        return anchors

    def propose(self, owner, anchor, position, block_table):
        (proposal,) = self.propose_many(owner, anchor, position, block_table, 1, ())
        return proposal

    def propose_many(self, owner, anchor, position, block_table, count, eos):
        if type(count) is not int or count not in (1, 2, 3):
            raise ValueError("MTP proposal depth must be 1, 2, or 3")
        state = self._ready_state(owner, anchor, position)
        token, hidden = anchor, state.hidden
        proposals = []
        # The uncompressed Draft writes only its current slot and reads up to
        # that position. Unaccepted suffix slots fall beyond the next length.
        for depth in range(count):
            (token,), hidden = self._forward(
                self.draft, (token,), position - 1 + depth, block_table,
                draft_hidden=hidden, return_hidden=depth + 1 < count,
                sample_all_positions=False,
            )
            proposals.append(token)
            if token in eos:
                break
        state.proposal = tuple(proposals)
        state.proposal_table = tuple(block_table)
        return state.proposal

    def _ready_state(self, owner, anchor, position):
        state = self.states.get(owner)
        if (
            state is None or anchor != state.anchor or position != state.position
            or state.proposal is not None or self.transaction is not None
        ):
            raise ValueError("MTP step disagrees with the current owner, anchor or position")
        return state

    def target_tail(self, owner, anchor, position, table):
        self._ready_state(owner, anchor, position)
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
        if len(tables) == 1:
            tokens, _ = self._forward(
                self.target, batch.input_ids, batch.position_ids[0], tables[0],
                return_hidden=False, rollback=True,
            )
        else:
            tokens, _ = self._forward_packed(self.target, batch, tables, rollback=True)
        transaction = _TargetTransaction(self, batch, tables, tokens)
        self.transaction = transaction
        try:
            yield transaction
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


def create_deepseek_v4_mtp_transport(config, target_engine, device):
    from infinilm import _core
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    require_classic_v4_mtp_abi()
    if target_engine.model_type != "deepseek_v4":
        raise ValueError("classic MTP requires DeepSeek V4 Target")
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
    return _V4Transport(target_engine, draft, _core, config.block_size)
