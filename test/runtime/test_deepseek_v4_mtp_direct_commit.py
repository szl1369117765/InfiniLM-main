"""Exercise eager K1 Target commit without loading the native engines."""

import os
import sys
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch


package = types.ModuleType("direct_commit_test_runner")
package.__path__ = [str(Path(__file__).resolve().parents[2] / "python/infinilm/llm/model_runner")]
sys.modules[package.__name__] = package
from direct_commit_test_runner.deepseek_v4_mtp_transport import (
    _V4Transport,
    _direct_k1_commit_enabled,
    _direct_k23_commit_enabled,
)
from direct_commit_test_runner.deepseek_v4_mtp_verify import GreedyVerifyBatch, GreedyVerifyRequest


class Tensor:
    def __init__(self, values):
        self.values = values

    def to_numpy(self):
        return self

    def tolist(self):
        return self.values

    def narrow(self, dim, start, count):
        assert dim == 0
        return Tensor(self.values[start:start + count])


class Engine:
    def __init__(self, *responses):
        self.responses = deque(responses)
        self.calls = []

    def forward_raw(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.popleft()


def output(ids, source=None, *, rows=2):
    result = {"output_ids": Tensor(ids)}
    if source is not None:
        result["hidden_states_by_rank"] = tuple(
            Tensor([(source, rank, row) for row in range(rows)]) for rank in range(8)
        )
    return result


class DirectCommitTest(unittest.TestCase):
    def make_step(self, accept, *, direct=True, remaining=4, missing_hidden=False):
        verify = [9, 10] if accept else [8, 10]
        target = Engine(output([7], "prefill"), output(verify, None if missing_hidden else "verify"))
        if not direct:
            target.responses.append(output(verify if accept else verify[:1], "replay"))
        draft = Engine(output([1]), output([9]))
        if accept and remaining > 2 and not missing_hidden:
            draft.responses.append(output([10]))
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        port = _V4Transport(target, draft, core, 256, direct_k1_commit=direct)
        prompt = dict(input_ids=Tensor([[1, 3]]), block_tables=Tensor([[6]]), past_kv_lengths=Tensor([0]))
        self.assertEqual(port.prefill("r", prompt), 7)
        self.assertEqual(port.propose("r", 7, 2, [6]), 9)
        batch = GreedyVerifyBatch.pack([GreedyVerifyRequest("r", 2, remaining, 7, (9,))], 256)
        return port, target, draft, batch

    def commit(self, port, batch):
        with port.target_transaction(batch, (6,)) as transaction:
            (intent,) = batch.resolve(transaction.rows)
            transaction.commit(intent)
        return intent

    def assert_hidden_row(self, tensors, source, row):
        self.assertEqual(len(tensors), 8)
        for rank, tensor in enumerate(tensors):
            self.assertEqual(tensor.values, [(source, rank, row)])

    def test_accept_retains_both_target_rows_and_repairs_draft(self):
        port, target, draft, batch = self.make_step(True)
        self.commit(port, batch)
        self.assertEqual(len(target.calls), 2)
        self.assertFalse(target.calls[1]["rollback_v4_cache"])
        self.assertTrue(target.calls[1]["return_draft_hidden_states"])
        self.assertEqual(draft.calls[2]["input_ids"].values, [[9]])
        self.assertEqual(draft.calls[2]["position_ids"].values, [2])
        self.assert_hidden_row(draft.calls[2]["target_hidden_states_by_rank"], "verify", 0)
        self.assert_hidden_row(port.hidden, "verify", 1)
        self.assertEqual((port.anchor, port.position), (10, 4))

    def test_reject_keeps_first_target_row_for_next_proposal(self):
        port, target, draft, batch = self.make_step(False)
        self.commit(port, batch)
        self.assertEqual(len(target.calls), 2)
        self.assertEqual(len(draft.calls), 2)
        self.assertEqual((port.anchor, port.position), (8, 3))
        draft.responses.append(output([14]))
        self.assertEqual(port.propose("r", 8, 3, [6]), 14)
        self.assert_hidden_row(draft.calls[2]["target_hidden_states_by_rank"], "verify", 0)
        self.assertEqual(draft.calls[2]["position_ids"].values, [2])

    def test_terminal_accept_skips_draft_repair_and_releases_owner(self):
        port, target, draft, batch = self.make_step(True, remaining=2)
        self.assertTrue(self.commit(port, batch).terminal)
        self.assertEqual((len(target.calls), len(draft.calls)), (2, 2))
        self.assertIsNone(port.anchor)
        port.release("r")
        self.assertIsNone(port.hidden)

    def test_protected_path_still_rolls_back_and_replays_retained_prefix(self):
        port, target, draft, batch = self.make_step(False, direct=False)
        self.commit(port, batch)
        self.assertTrue(target.calls[1]["rollback_v4_cache"])
        self.assertFalse(target.calls[1]["return_draft_hidden_states"])
        self.assertEqual(target.calls[2]["input_ids"].values, [[7]])
        self.assert_hidden_row(port.hidden, "replay", 0)

    def test_missing_verify_hidden_raises_before_replay(self):
        port, target, draft, batch = self.make_step(False, missing_hidden=True)
        with self.assertRaisesRegex(RuntimeError, "requires Target hidden states"):
            self.commit(port, batch)
        self.assertEqual((len(target.calls), len(draft.calls)), (2, 2))

    def test_opt_in_is_limited_to_eager_c1_k1(self):
        config = {"enable_graph": False, "max_batch_size": 1, "num_draft_tokens": 1,
                  "nextn_branch_draft_lengths": None}
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_K1_DIRECT_COMMIT": "0"}):
            self.assertFalse(_direct_k1_commit_enabled(types.SimpleNamespace(**config)))
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_K1_DIRECT_COMMIT": "1"}):
            self.assertTrue(_direct_k1_commit_enabled(types.SimpleNamespace(**config)))
            for key, value in (("enable_graph", True), ("max_batch_size", 2), ("num_draft_tokens", 2),
                               ("nextn_branch_draft_lengths", (1, 3))):
                with self.subTest(key=key):
                    self.assertFalse(_direct_k1_commit_enabled(types.SimpleNamespace(**{**config, key: value})))

    def test_k23_opt_in_requires_eager_c1_and_the_expanded_native_layout(self):
        settings = {"enable_graph": False, "max_batch_size": 1, "num_draft_tokens": 3,
                    "nextn_branch_draft_lengths": None}
        config = types.SimpleNamespace(**settings)
        expanded = types.SimpleNamespace(deepseek_v4_raw_ring_rows=(16, 256))
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_K23_DIRECT_COMMIT": "0"}):
            self.assertFalse(_direct_k23_commit_enabled(config, object()))
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_K23_DIRECT_COMMIT": "1"}):
            self.assertTrue(_direct_k23_commit_enabled(config, expanded))
            with self.assertRaisesRegex(RuntimeError, "expanded native compressor rings"):
                _direct_k23_commit_enabled(config, object())
            for key, value in (("enable_graph", True), ("max_batch_size", 2), ("num_draft_tokens", 1),
                               ("nextn_branch_draft_lengths", (1, 3))):
                other = types.SimpleNamespace(**{**settings, key: value})
                with self.subTest(key=key):
                    self.assertFalse(_direct_k23_commit_enabled(other, expanded))

    def test_k23_verify_keeps_every_possible_prefix(self):
        for query in (2, 3, 4):
            for retained in range(1, query + 1):
                with self.subTest(query=query, retained=retained):
                    self.check_k23_prefix(query, retained)

    def check_k23_prefix(self, query, retained):
        candidates = tuple(range(9, 8 + query))
        verify = [*candidates[:retained - 1], 90, *([91] * (query - retained))]
        target = Engine(output([7], "prefill"), output(verify, "verify", rows=query))
        draft = Engine(output([1]), *(output([token], "draft") for token in candidates),
                       *(output([1]) for _ in range(retained - 1)))
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        port = _V4Transport(target, draft, core, 256, direct_k23_commit=True)
        prompt = dict(input_ids=Tensor([[1, 3]]), block_tables=Tensor([[6]]), past_kv_lengths=Tensor([0]))
        self.assertEqual(port.prefill("r", prompt), 7)
        self.assertEqual(port.propose_many("r", 7, 2, [6], query - 1, ()), candidates)
        batch = GreedyVerifyBatch.pack([GreedyVerifyRequest("r", 2, 8, 7, candidates)], 256)
        self.assertEqual(len(self.commit(port, batch).materialized_rows), retained)
        self.assertEqual(len(target.calls), 2)
        self.assertFalse(target.calls[1]["rollback_v4_cache"])
        self.assertTrue(target.calls[1]["return_draft_hidden_states"])
        self.assertEqual((port.anchor, port.position), (90, 2 + retained))
        self.assert_hidden_row(port.hidden, "verify", retained - 1)
        repair_calls = draft.calls[query:]
        self.assertEqual(len(repair_calls), retained - 1)
        for offset, call in enumerate(repair_calls):
            self.assertEqual(call["position_ids"].values, [2 + offset])
            self.assertEqual(call["input_ids"].values, [[candidates[offset]]])
            self.assert_hidden_row(call["target_hidden_states_by_rank"], "verify", offset)


if __name__ == "__main__":
    unittest.main()
