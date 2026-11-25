"""Compare packed and serial Draft work through the classic NextN runner on CPU."""

import os
import sys
import unittest
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


package = ModuleType("batched_draft_test_llm")
package.__path__ = [str(Path(__file__).resolve().parents[2] / "python/infinilm/llm")]
sys.modules[package.__name__] = package
from batched_draft_test_llm.model_runner.deepseek_v4_mtp_runner import DeepSeekV4MTPRunner
from batched_draft_test_llm.model_runner.deepseek_v4_mtp_transport import (
    _V4Transport,
    _batched_draft_enabled,
    _k1_graph_seed_enabled,
)
from batched_draft_test_llm.sampling_params import SamplingParams


PAGE_SIZE = 256
RANKS = 8


class Tensor:
    def __init__(self, rows, device="cpu"):
        self.rows = rows
        self.device = device
        self.shape = (len(rows), 4, 1)

    def to_numpy(self):
        return self

    def tolist(self):
        return self.rows

    def narrow(self, dim, start, count):
        assert dim == 0
        return Tensor(self.rows[start:start + count], self.device)


class Core:
    int64 = 64
    int32 = 32

    def from_list(self, rows, *, dtype):
        return Tensor(rows)

    def cat(self, tensors, *, dim):
        raise AssertionError("Frontend cannot concatenate Draft worker-owned hidden states")

    def sync_stream(self):
        raise AssertionError("Frontend cannot synchronize a Draft worker stream")


class Engine:
    hf_config = {"max_position_embeddings": 2048}
    eos_token_id = None

    def __init__(self, *, draft=False):
        self.draft = draft
        self.calls = []
        self.return_hidden = []
        self.cache = {}
        self.input_hidden = {}

    def forward_raw(self, **options):
        (tokens,) = options["input_ids"].rows
        offsets = options["input_offsets"].rows
        positions = options["position_ids"].rows
        slots = options["slot_mapping"].rows
        tables = options["block_tables"].rows
        past = options["past_kv_lengths"].rows
        total = options["total_kv_lengths"].rows
        cumulative = options["cu_seqlens"].rows
        all_positions = options["sample_all_positions"]
        assert offsets[0] == cumulative[0] == 0 and offsets[-1] == len(tokens)
        assert len(set(slots)) == len(tokens)
        assert options["top_k"] == 1
        if self.draft:
            if len(offsets) > 2:
                assert "target_hidden_states_by_rank" not in options
                rank_segments = options["target_hidden_segments_by_rank"]
                assert isinstance(rank_segments, tuple)
                lengths = [end - begin for begin, end in zip(offsets, offsets[1:])]
                rank_hidden = []
                for rank, segments in enumerate(rank_segments):
                    assert isinstance(segments, tuple)
                    assert [len(tensor.rows) for tensor in segments] == lengths
                    assert all(tensor.device == rank for tensor in segments)
                    rank_hidden.append(Tensor([row for tensor in segments for row in tensor.rows], rank))
            else:
                assert "target_hidden_segments_by_rank" not in options
                rank_hidden = options["target_hidden_states_by_rank"]
            assert len(rank_hidden) == RANKS
            assert all(tensor.device == rank and len(tensor.rows) == len(tokens)
                       for rank, tensor in enumerate(rank_hidden))
        predictions, hidden = [], [[] for _ in range(RANKS)]
        spans = []
        for index, (begin, end) in enumerate(zip(offsets, offsets[1:])):
            owner = tables[index][0]
            query = positions[begin:end]
            assert query == list(range(past[index], total[index]))
            assert cumulative[index + 1] - cumulative[index] == total[index]
            spans.append((owner, tuple(query), tuple(slots[begin:end])))
            for row in range(begin, end):
                token, pos = tokens[row], positions[row]
                assert slots[row] == tables[index][pos // PAGE_SIZE] * PAGE_SIZE + pos % PAGE_SIZE
                if self.draft:
                    self.input_hidden.setdefault((owner, pos, token), []).append(
                        tuple(tensor.rows[row] for tensor in rank_hidden)
                    )
                # Only owner 4 at this position rejects its K1 proposal.
                prediction = token + (2 if not self.draft and (owner, pos) == (4, 255) else 1)
                if all_positions or row == end - 1:
                    predictions.append(prediction)
                if not options.get("rollback_v4_cache"):
                    self.cache[slots[row]] = token
                for rank, output in enumerate(hidden):
                    upstream = ()
                    if self.draft:
                        upstream = rank_hidden[rank].rows[row]
                        assert upstream[:2] == (rank, owner)
                        assert upstream[2] == pos
                    output.append((rank, owner, pos + int(self.draft), token, upstream))
        self.calls.append((tuple(spans), all_positions))
        self.return_hidden.append(options["return_draft_hidden_states"])
        result = {"output_ids": Tensor(predictions)}
        if options.get("mtp_graph_direct_commit"):
            assert not self.draft and not options["rollback_v4_cache"]
            result["execution_report"] = {
                "mode": "device_graph", "device_graph_segments": 1, "host_segments": 0,
            }
        if options["return_draft_hidden_states"]:
            result["hidden_states_by_rank"] = tuple(Tensor(rows, rank) for rank, rows in enumerate(hidden))
        return result


class Request:
    has_multimodal_inputs = False
    num_local_cached_tokens = 0
    eos_token_ids = ()

    def __init__(self, owner, first_page, budget, *, stop=()):
        self.request_id = owner
        self.block_table = [first_page]
        self.next_pages = deque((first_page + 1, first_page + 2))
        self.generated_token_ids = []
        self.sampling_params = SamplingParams(max_tokens=budget, stop_token_ids=tuple(stop))

    def get_prompt_length(self):
        return 255

    def get_num_generated_tokens(self):
        return len(self.generated_token_ids)

    def get_total_length(self):
        return self.get_prompt_length() + self.get_num_generated_tokens()


class CacheOps:
    def __init__(self, *, permit_direct=False):
        self.permit_direct = permit_direct

    @contextmanager
    def verify_batch(self, candidates):
        for row in candidates:
            request = row["req"]
            table = list(request.block_table)
            while len(table) * PAGE_SIZE < row["base_len"] + row["draft_count"]:
                table.append(request.next_pages.popleft())
            row["block_table"] = table

        def commit(lengths):
            for row, length in zip(candidates, lengths, strict=True):
                row["req"].block_table = row["block_table"][:(length + PAGE_SIZE - 1) // PAGE_SIZE]

        yield SimpleNamespace(mark_submitted=lambda: None, mark_completed=lambda: None, commit=commit,
                              permits_packed_direct_target=lambda tables: self.permit_direct)


def make_case(enabled, depth, specs, *, transport_options=None, modes=None):
    target, draft, core = Engine(), Engine(draft=True), Core()
    transport = _V4Transport(target, draft, core, PAGE_SIZE, batched_draft=enabled, **(transport_options or {}))
    if modes is not None:
        proposal, repair = modes
        transport.set_c3_draft_modes(proposal=proposal, repair=repair)
    runner = DeepSeekV4MTPRunner(target, transport, 2048, num_draft_tokens=depth)
    requests = [Request(name, page, budget, stop=stop) for name, page, budget, stop in specs]
    pages = [request.block_table[0] for request in requests]
    width = len(pages)
    length = requests[0].get_prompt_length()
    inputs = {
        "input_ids": Tensor([[page * 10 for page in pages for _ in range(length)]]),
        "position_ids": Tensor(list(range(length)) * width),
        "past_kv_lengths": Tensor([0] * width),
        "total_kv_lengths": Tensor([length] * width),
        "input_offsets": Tensor([index * length for index in range(width + 1)]),
        "cu_seqlens": Tensor([index * length for index in range(width + 1)]),
        "block_tables": Tensor([[page] for page in pages]),
        "slot_mapping": Tensor([page * PAGE_SIZE + pos for page in pages for pos in range(length)]),
    }
    anchors = runner.forward(SimpleNamespace(is_prefill=True, scheduled_requests=requests), inputs)
    for request, anchor in zip(requests, anchors, strict=True):
        request.generated_token_ids.append(anchor)
    draft.calls.clear()
    draft.return_hidden.clear()
    draft.input_hidden.clear()
    return runner, requests, draft, core


def decode(runner, requests, *, cache_ops=None):
    output = runner.forward(SimpleNamespace(
        is_prefill=False, scheduled_requests=requests,
        speculative_cache_ops=CacheOps() if cache_ops is None else cache_ops,
    ), {})
    for request, tokens in zip(requests, output, strict=True):
        request.generated_token_ids.extend(tokens)
    return output


class BatchedDraftTest(unittest.TestCase):
    def test_opt_in_applies_only_to_eager_single_chains(self):
        config = SimpleNamespace(enable_graph=False, nextn_branch_draft_lengths=None)
        with patch.dict(os.environ, {"INFINILM_MTP_BATCHED_DRAFT": "0"}):
            self.assertFalse(_batched_draft_enabled(config))
        with patch.dict(os.environ, {"INFINILM_MTP_BATCHED_DRAFT": "1"}):
            self.assertTrue(_batched_draft_enabled(config))
            config.enable_graph = True
            self.assertFalse(_batched_draft_enabled(config))
            config.enable_graph = False
            config.nextn_branch_draft_lengths = (1, 3)
            self.assertFalse(_batched_draft_enabled(config))

    def test_k1_graph_seed_requires_an_eligible_config_and_native_abi(self):
        settings = dict(enable_graph=True, max_batch_size=3, num_draft_tokens=1,
                        nextn_branch_draft_lengths=None)
        config = SimpleNamespace(**settings)
        native = SimpleNamespace(mtp_k1_graph_seed_version=1)
        with patch.dict(os.environ, {"INFINILM_MTP_K1_GRAPH_SEED": "0"}):
            self.assertFalse(_k1_graph_seed_enabled(config, object()))
        with patch.dict(os.environ, {"INFINILM_MTP_K1_GRAPH_SEED": "1"}):
            self.assertTrue(_k1_graph_seed_enabled(config, native))
            with self.assertRaisesRegex(RuntimeError, "native Draft q2 graph"):
                _k1_graph_seed_enabled(config, object())
            for key, value in (("enable_graph", False), ("max_batch_size", 4),
                               ("num_draft_tokens", 2), ("nextn_branch_draft_lengths", (1,))):
                config = SimpleNamespace(**{**settings, key: value})
                self.assertFalse(_k1_graph_seed_enabled(config, object()))

    def test_k1_repair_and_seed_across_a_rejected_page(self):
        specs = (("a", 1, 8, ()), ("reject", 4, 8, ()), ("c", 7, 8, ()))
        c1 = specs[:1]
        oracles = []
        for spec in specs:
            runner, requests, _, _ = make_case(False, 1, (spec,))
            oracles.append(tuple(decode(runner, requests)[0] for _ in range(2)))
        for cohort, expected in ((c1, ((1, 1), (1,))), (specs, ((3, 3), (3,)))):
            with self.subTest(owners=len(cohort)):
                serial, old_requests, old_draft, _ = make_case(False, 1, cohort)
                batched, requests, draft, _ = make_case(True, 1, cohort)
                actual = decode(batched, requests)
                self.assertEqual(actual, decode(serial, old_requests))
                self.assertEqual(actual, [row[0] for row in oracles[:len(cohort)]])
                self.assertEqual(tuple(len(spans) for spans, _ in draft.calls), expected[0])
                self.assertEqual(len(old_draft.calls), 2 if len(cohort) == 1 else 5)
                if len(cohort) == 3:
                    repair, all_positions = draft.calls[-1]
                    self.assertEqual(tuple(span[1] for span in repair), ((255, 256), (255,), (255, 256)))
                    self.assertFalse(all_positions)
                    self.assertEqual(requests[1].block_table, [4])
                    self.assertEqual(batched.transport.states["reject"].seed.pages, (4,))
                    self.assertEqual(requests[0].block_table, [1, 2])
                    self.assertEqual(batched.transport.states["a"].seed.pages, (1, 2))
                for state in batched.transport.states.values():
                    self.assertEqual(state.seed.hidden, ())
                old_draft.calls.clear()
                draft.calls.clear()
                actual = decode(batched, requests)
                self.assertEqual(actual, decode(serial, old_requests))
                self.assertEqual(actual, [row[1] for row in oracles[:len(cohort)]])
                self.assertEqual(tuple(len(spans) for spans, _ in draft.calls), expected[1])
                self.assertEqual(len(old_draft.calls), len(cohort) * 2)
                if len(cohort) == 3:
                    self.assertEqual(requests[1].block_table, [4, 6])
                    self.assertNotIn(5 * PAGE_SIZE, draft.cache)
                    self.assertEqual(draft.cache[6 * PAGE_SIZE], 44)

    def test_changed_retained_page_recomputes_only_its_seed(self):
        specs = (("a", 1, 8, ()), ("b", 7, 8, ()))
        runner, requests, draft, _ = make_case(True, 1, specs)
        decode(runner, requests)
        requests[0].block_table[1] = 3
        draft.calls.clear()
        decode(runner, requests)
        self.assertEqual([len(spans) for spans, _ in draft.calls], [1, 2])
        self.assertEqual(draft.calls[0][0], ((1, (256,), (3 * PAGE_SIZE,)),))

    def test_k1_graph_seed_uses_serial_q2_only_after_acceptance(self):
        specs = (("a", 1, 8, ()), ("reject", 4, 8, ()), ("c", 7, 8, ()))
        for cohort in (specs[:1], specs):
            with self.subTest(owners=len(cohort)):
                baseline, old_requests, old_draft, _ = make_case(False, 1, cohort)
                runner, requests, draft, _ = make_case(False, 1, cohort)
                runner.transport.k1_graph_seed = True
                self.assertEqual(decode(runner, requests), decode(baseline, old_requests))
                self.assertEqual(len(draft.calls), len(old_draft.calls))
                fused = [
                    (spans[0], sample_all, return_hidden)
                    for (spans, sample_all), return_hidden in zip(draft.calls, draft.return_hidden, strict=True)
                    if len(spans[0][1]) == 2
                ]
                owners = [1] if len(cohort) == 1 else [1, 7]
                self.assertEqual(fused, [
                    ((page, (255, 256), (page * PAGE_SIZE + 255, (page + 1) * PAGE_SIZE)), False, False)
                    for page in owners
                ])
                state = runner.transport.states["a"]
                self.assertEqual((state.seed.hidden, state.seed.pages), ((), (1, 2)))
                if len(cohort) == 3:
                    self.assertIsNone(runner.transport.states["reject"].seed)
                    self.assertEqual(requests[1].block_table, [4])
                draft.calls.clear()
                draft.return_hidden.clear()
                old_draft.calls.clear()
                self.assertEqual(decode(runner, requests), decode(baseline, old_requests))
                self.assertEqual([len(spans[0][1]) for spans, _ in draft.calls],
                                 [2] if len(cohort) == 1 else [1, 2, 2, 2])
                self.assertTrue(all(len(spans) == 1 for spans, _ in draft.calls))
                self.assertEqual(len(old_draft.calls), 2 * len(cohort))
                if len(cohort) == 3:
                    self.assertEqual(requests[1].block_table, [4, 6])
                    self.assertNotIn(5 * PAGE_SIZE, draft.cache)

    def test_k1_graph_seed_skips_owners_without_another_proposal(self):
        specs = (("budget", 1, 4, ()), ("stop", 7, 8, (73,)))
        runner, requests, draft, _ = make_case(False, 1, specs)
        runner.transport.k1_graph_seed = True
        decode(runner, requests)
        self.assertTrue(all(state.seed is None for state in runner.transport.states.values()))
        self.assertTrue(all(len(spans[0][1]) == 1 for spans, _ in draft.calls))

    def test_depth_batch_shrinks_on_stop_and_budget(self):
        specs = (("stop", 1, 9, (12,)), ("short", 4, 4, ()), ("deep", 7, 9, ()))
        for depth, expected in ((2, (3, 2)), (3, (3, 2, 1))):
            with self.subTest(depth=depth):
                serial, old_requests, old_draft, _ = make_case(False, depth, specs)
                batched, requests, draft, _ = make_case(True, depth, specs)
                self.assertEqual(decode(batched, requests), decode(serial, old_requests))
                proposals = [spans for spans, all_positions in draft.calls if not all_positions]
                old_proposals = [spans for spans, all_positions in old_draft.calls if not all_positions]
                self.assertEqual(tuple(len(spans) for spans in proposals), expected)
                self.assertEqual(len(old_proposals), sum(expected))
                self.assertEqual(tuple(span[0] for span in proposals[0]), (1, 4, 7))
                self.assertEqual(tuple(span[0] for span in proposals[1]), (4, 7))
                if depth == 3:
                    self.assertEqual(tuple(span[0] for span in proposals[2]), (7,))
                self.assertEqual(draft.input_hidden, old_draft.input_hidden)
                for owner, state in batched.transport.states.items():
                    self.assertEqual([tensor.rows for tensor in state.hidden],
                                     [tensor.rows for tensor in serial.transport.states[owner].hidden])


if __name__ == "__main__":
    unittest.main()
