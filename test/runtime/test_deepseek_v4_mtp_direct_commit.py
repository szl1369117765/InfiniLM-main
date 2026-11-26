"""Exercise eager and packed Graph Target commit with native call substitutes."""

# The stub package must be registered before importing the real transport.
# ruff: noqa: E402

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
    _direct_k1_commit_enabled,
    _direct_k23_commit_enabled,
    _draft_cache_only_enabled,
    _DraftState,
    _eager_packed_direct_available,
    _graph_k3_draft_seed_enabled,
    _host_metadata_modes,
    _V4Transport,
)
from direct_commit_test_runner.deepseek_v4_mtp_verify import (
    GreedyVerifyBatch,
    GreedyVerifyRequest,
)


class Tensor:
    def __init__(self, values):
        self.values = values

    def to_numpy(self):
        return self

    def tolist(self):
        return self.values

    @property
    def shape(self):
        return (len(self.values), 1, 1)

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


def cache_write_receipt():
    return {"mtp_draft_cache_only_used": True, "output_ids": None, "logits": None, "hidden_states": None}


class DirectCommitTest(unittest.TestCase):
    def packed_direct_step(self, *, enabled, hidden=True, graph=True, eager=False, eager_selected=False, owner_count=3):
        requests = (
            GreedyVerifyRequest("a", 253, 4, 1, (2, 3, 4)),
            GreedyVerifyRequest("b", 253, 5, 11, (12, 13), frozenset({13})),
            GreedyVerifyRequest("c", 255, 5, 21, (22,)),
        )[:owner_count]
        # B's full three-token reservation crosses p256, but Draft stops at
        # EOS. The last reserved page must stay leased and out of native input.
        tables = ((3, 4), (5, 8), (6, 7))[:owner_count]
        batch = GreedyVerifyBatch.pack(requests, 512)
        predictions = [2, 3, 4, 5, 15, 16, 17, 22, 23][:len(batch.rows)]
        verify = output(predictions, "verify" if enabled and hidden else None, rows=len(batch.rows))
        verify["execution_report"] = ({
            "mode": "eager",
            "device_graph_segments": 0,
            "host_segments": 0,
            "mtp_eager_packed_direct_commit": enabled,
        } if eager else {
            "mode": "device_graph" if graph else "mixed",
            "device_graph_segments": 1,
            "host_segments": 0 if graph else 1,
        })
        target = Engine(
            verify,
            *(() if enabled else (
                output(predictions[:4], "a", rows=4),
                output(predictions[4:5], "b", rows=1),
                output(predictions[7:], "c", rows=2),
            )[:owner_count]),
        )
        draft = Engine(output([29]))
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        port = _V4Transport(target, draft, core, 256, graph_packed_direct_available=not eager,
                            eager_packed_direct_available=eager)
        if eager:
            port.set_eager_packed_direct_commit(eager_selected)
        for req, table in zip(requests, tables, strict=True):
            port.states[req.owner] = _DraftState((), req.anchor, req.start, req.draft_tokens, table)
        return port, target, draft, batch, tables

    def test_packed_graph_direct_slices_owner_hc_and_trims_only_native_pages(self):
        port, target, draft, batch, tables = self.packed_direct_step(enabled=True)
        with patch.dict(os.environ, {"INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT": "1"}):
            with port.target_transaction(batch, tables, graph_packed_direct_commit=True) as tx:
                self.assertEqual(tx.tables, tables)
                self.assertEqual(port.states["b"].proposal_table, (5, 8))
                decisions = batch.resolve(tx.rows)
                self.assertEqual([(d.retain_length, d.terminal) for d in decisions],
                                 [(257, True), (254, False), (257, False)])
                tx.commit(decisions)
        self.assertEqual(len(target.calls), 1)
        self.assertTrue(target.calls[0]["mtp_graph_direct_commit"])
        self.assertFalse(target.calls[0]["rollback_v4_cache"])
        self.assertEqual(target.calls[0]["input_offsets"].values, [0, 4, 7, 9])
        self.assertEqual(target.calls[0]["block_tables"].values, [[3, 4], [5, -1], [6, 7]])
        self.assertEqual(len(draft.calls), 1)
        self.assert_hidden_row(draft.calls[0]["target_hidden_states_by_rank"], "verify", 7)
        for owner, row in (("a", 3), ("b", 4), ("c", 8)):
            self.assert_hidden_row(port.states[owner].hidden, "verify", row)

    def test_packed_graph_direct_defaults_to_protection(self):
        port, target, _, batch, tables = self.packed_direct_step(enabled=False)
        with patch.dict(os.environ, {"INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT": "0"}):
            with port.target_transaction(batch, tables, graph_packed_direct_commit=True) as tx:
                tx.commit(batch.resolve(tx.rows))
        self.assertEqual([call["rollback_v4_cache"] for call in target.calls], [True, False, False, False])
        self.assertEqual(target.calls[0]["block_tables"].values, [list(table) for table in tables])
        self.assertNotIn("mtp_graph_direct_commit", target.calls[0])

    def test_single_owner_graph_direct_reuses_target_hidden(self):
        port, target, _, _, tables = self.packed_direct_step(enabled=True)
        req = GreedyVerifyRequest("a", 253, 4, 1, (2, 3, 4))
        batch = GreedyVerifyBatch.pack((req,), 512)
        verify = output([2, 3, 4, 5], "verify", rows=4)
        verify["execution_report"] = {
            "mode": "device_graph", "device_graph_segments": 1, "host_segments": 0,
        }
        target = port.target = Engine(verify)
        port.states = {"a": port.states["a"]}
        port.graph_single_direct_available = True
        with patch.dict(os.environ, {"INFINILM_MTP_GRAPH_K3_DIRECT_COMMIT": "1"}):
            with port.target_transaction(batch, tables[0], graph_packed_direct_commit=True) as tx:
                tx.commit(batch.resolve(tx.rows)[0])
        self.assertEqual(len(target.calls), 1)
        self.assertTrue(target.calls[0]["mtp_graph_direct_commit"])
        self.assertEqual(target.calls[0]["input_offsets"].values, [0, 4])
        self.assertEqual(target.calls[0]["block_tables"].values, [[3, 4]])

    def test_packed_graph_direct_bad_receipt_does_not_retry_after_native_write(self):
        for options, message in (
            ({"hidden": False}, "complete Target HC"),
            ({"graph": False}, "single Target device Graph"),
        ):
            with self.subTest(options=options):
                port, target, draft, batch, tables = self.packed_direct_step(enabled=True, **options)
                with patch.dict(os.environ, {"INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT": "1"}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        with port.target_transaction(batch, tables, graph_packed_direct_commit=True):
                            self.fail("invalid native receipt reached commit")
                self.assertEqual(len(target.calls), 1)
                self.assertEqual(draft.calls, [])

    def test_packed_eager_direct_reuses_verified_hc_and_trims_reserved_pages(self):
        port, target, draft, batch, tables = self.packed_direct_step(enabled=True, eager=True, eager_selected=True)
        with port.target_transaction(batch, tables, eager_packed_direct_commit=True) as tx:
            tx.commit(batch.resolve(tx.rows))
        self.assertEqual(len(target.calls), 1)
        self.assertTrue(target.calls[0]["mtp_eager_packed_direct_commit"])
        self.assertNotIn("mtp_graph_direct_commit", target.calls[0])
        self.assertFalse(target.calls[0]["rollback_v4_cache"])
        self.assertEqual(target.calls[0]["block_tables"].values, [[3, 4], [5, -1], [6, 7]])
        self.assert_hidden_row(draft.calls[0]["target_hidden_states_by_rank"], "verify", 7)
        for owner, row in (("a", 3), ("b", 4), ("c", 8)):
            self.assert_hidden_row(port.states[owner].hidden, "verify", row)

    def test_eager_c3_cache_only_uses_the_same_single_row_metadata_and_hc(self):
        calls = []
        for enabled in (False, True):
            port, target, draft, batch, tables = self.packed_direct_step(
                enabled=True, eager=True, eager_selected=True,
            )
            port.shared_compression_plan = True
            draft.mtp_eager_c3_draft_repair_cache_only_version = 1
            if enabled:
                draft.responses[0] = cache_write_receipt()
            with patch.dict(os.environ, {"INFINILM_MTP_EAGER_C3_DRAFT_REPAIR_CACHE_ONLY": str(int(enabled))}):
                with port.target_transaction(batch, tables, eager_packed_direct_commit=True) as tx:
                    tx.commit(batch.resolve(tx.rows))
            self.assertEqual(len(target.calls), 1)
            self.assertEqual(len(draft.calls), 1)
            self.assert_hidden_row(port.states["c"].hidden, "verify", 8)
            calls.append(draft.calls[0])
        before, after = calls
        for key in ("input_ids", "input_offsets", "position_ids", "past_kv_lengths", "total_kv_lengths",
                    "cu_seqlens", "block_tables", "slot_mapping"):
            self.assertEqual(before[key].values, after[key].values, key)
        self.assertEqual(after["input_offsets"].values, [0, 1])
        self.assertEqual(after["position_ids"].values, [255])
        self.assert_hidden_row(before["target_hidden_states_by_rank"], "verify", 7)
        self.assert_hidden_row(after["target_hidden_states_by_rank"], "verify", 7)
        for key in ("sample_all_positions", "top_k", "return_draft_hidden_states", "mtp_shared_compression_plan"):
            self.assertEqual(before[key], after[key], key)
        self.assertNotIn("mtp_eager_c3_draft_repair_cache_only", before)
        self.assertTrue(after["mtp_eager_c3_draft_repair_cache_only"])
        self.assertTrue(after["mtp_draft_cache_only"])

    def test_eager_c3_cache_only_switch_does_not_affect_protected_or_short_cohorts(self):
        for owner_count, selected in ((3, False), (2, True)):
            with self.subTest(owner_count=owner_count, selected=selected):
                port, target, draft, batch, tables = self.packed_direct_step(
                    enabled=False, eager=True, eager_selected=selected, owner_count=owner_count,
                )
                with patch.dict(os.environ, {"INFINILM_MTP_EAGER_C3_DRAFT_REPAIR_CACHE_ONLY": "1"}):
                    with port.target_transaction(batch, tables, eager_packed_direct_commit=True) as tx:
                        tx.commit(batch.resolve(tx.rows))
                self.assertTrue(target.calls[0]["rollback_v4_cache"])
                self.assertTrue(all("mtp_eager_c3_draft_repair_cache_only" not in call for call in draft.calls))

    def test_eager_c3_cache_only_requires_abi_before_writes_and_checks_native_receipt(self):
        cases = (
            (False, cache_write_receipt(), "native output-free", 0),
            (True, output([29]), "completed cache write", 1),
            (True, {**cache_write_receipt(), "hidden_states_by_rank": (Tensor([]),)}, "completed cache write", 1),
        )
        for abi, receipt, message, expected_writes in cases:
            with self.subTest(abi=abi, receipt=receipt):
                port, target, draft, batch, tables = self.packed_direct_step(
                    enabled=True, eager=True, eager_selected=True,
                )
                if abi:
                    draft.mtp_eager_c3_draft_repair_cache_only_version = 1
                draft.responses[0] = receipt
                with patch.dict(os.environ, {"INFINILM_MTP_EAGER_C3_DRAFT_REPAIR_CACHE_ONLY": "1"}):
                    with self.assertRaisesRegex(RuntimeError, message):
                        with port.target_transaction(batch, tables, eager_packed_direct_commit=True) as tx:
                            tx.commit(batch.resolve(tx.rows))
                self.assertEqual(len(target.calls), expected_writes)
                self.assertEqual(len(draft.calls), expected_writes)
                self.assertEqual(port.states["c"].position, 255)

    def test_packed_eager_direct_leaves_arm_off_and_short_cohorts_protected(self):
        for owner_count, selected in ((3, False), (2, True)):
            with self.subTest(owner_count=owner_count, selected=selected):
                port, target, _, batch, tables = self.packed_direct_step(
                    enabled=False, eager=True, eager_selected=selected, owner_count=owner_count,
                )
                with port.target_transaction(batch, tables, eager_packed_direct_commit=True) as tx:
                    tx.commit(batch.resolve(tx.rows))
                self.assertEqual([call["rollback_v4_cache"] for call in target.calls], [True] + [False] * owner_count)
                self.assertNotIn("mtp_eager_packed_direct_commit", target.calls[0])

    def test_packed_eager_direct_bad_receipt_does_not_retry_after_native_write(self):
        for hidden, changes, message in (
            (False, {}, "complete Target HC"),
            (True, {"mtp_eager_packed_direct_commit": False}, "verified eager Target"),
            (True, {"host_segments": 1}, "verified eager Target"),
        ):
            with self.subTest(hidden=hidden, changes=changes):
                port, target, draft, batch, tables = self.packed_direct_step(
                    enabled=True, hidden=hidden, eager=True, eager_selected=True,
                )
                target.responses[0]["execution_report"].update(changes)
                with self.assertRaisesRegex(RuntimeError, message):
                    with port.target_transaction(batch, tables, eager_packed_direct_commit=True):
                        self.fail("invalid native receipt reached commit")
                self.assertEqual(len(target.calls), 1)
                self.assertEqual(draft.calls, [])

    def test_packed_eager_direct_is_prepared_only_for_eligible_native_engines(self):
        config = types.SimpleNamespace(
            enable_graph=False, max_batch_size=3, num_draft_tokens=3,
            pipeline_parallel_size=1, tensor_parallel_size=8, moe_ep_size=1, block_size=256,
            nextn_branch_draft_lengths=None, enable_prefix_caching=False,
        )
        target = types.SimpleNamespace(mtp_eager_packed_direct_commit_version=1, deepseek_v4_raw_ring_rows=(16, 256))
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_C3_DIRECT_COMMIT": "0"}):
            self.assertFalse(_eager_packed_direct_available(config, target))
        with patch.dict(os.environ, {"INFINILM_MTP_EAGER_C3_DIRECT_COMMIT": "1"}):
            self.assertTrue(_eager_packed_direct_available(config, target))
            with patch.object(config, "enable_graph", True):
                self.assertFalse(_eager_packed_direct_available(config, target))
            with self.assertRaisesRegex(RuntimeError, "native ABI"):
                _eager_packed_direct_available(config, object())
        port = _V4Transport(None, None, None, 256, eager_packed_direct_available=True)
        self.assertFalse(port.eager_packed_direct_enabled())
        port.set_eager_packed_direct_commit(True)
        self.assertTrue(port.eager_packed_direct_enabled())
        with patch.object(port, "states", {"a": object()}), self.assertRaisesRegex(RuntimeError, "active owners"):
            port.set_eager_packed_direct_commit(False)

    def make_step(self, accept, *, direct=True, remaining=4, missing_hidden=False,
                  draft_cache_only=False, reuse_seed=False):
        verify = [9, 10] if accept else [8, 10]
        target = Engine(output([7], "prefill"), output(verify, None if missing_hidden else "verify"))
        if not direct:
            target.responses.append(output(verify if accept else verify[:1], "replay"))
        draft = Engine(cache_write_receipt() if draft_cache_only else output([1]), output([9]))
        if accept and remaining > 2 and not missing_hidden:
            repair = output([10], "seed" if reuse_seed else None)
            if draft_cache_only and not reuse_seed:
                repair = cache_write_receipt()
            draft.responses.append(repair)
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        port = _V4Transport(target, draft, core, 256, direct_k1_commit=direct,
                            draft_cache_only=draft_cache_only, reuse_seed=reuse_seed)
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

    def test_host_metadata_modes_preserve_protected_target_queries(self):
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        for shadow, shared, kwargs in (
            (False, False, {}),
            (True, False, {"mtp_cpu_metadata_shadow": True}),
            (False, True, {"mtp_shared_compression_plan": True}),
        ):
            for query in (2, 3, 4):
                with self.subTest(shadow=shadow, shared=shared, query=query):
                    tokens = tuple(range(query))
                    target = Engine(output(list(tokens)))
                    port = _V4Transport(
                        target, None, core, 256, cpu_metadata_shadow=shadow,
                        shared_compression_plan=shared,
                    )
                    self.assertEqual(
                        port._forward(target, tokens, 255, (3, 1), return_hidden=False, rollback=True),
                        (tokens, None),
                    )
                    call = target.calls[0]
                    self.assertEqual(call["input_offsets"].values, [0, query])
                    self.assertTrue(call["rollback_v4_cache"])
                    self.assertEqual(
                        {key: value for key, value in call.items() if key.startswith("mtp_")},
                        kwargs,
                    )

    def test_host_metadata_requires_the_requested_eager_native_capability(self):
        eager = types.SimpleNamespace(enable_graph=False)
        graph = types.SimpleNamespace(enable_graph=True)
        native = types.SimpleNamespace(mtp_cpu_metadata_shadow_version=1)
        with patch.dict(os.environ, {
            "INFINILM_MTP_CPU_METADATA_SHADOW": "0",
            "INFINILM_MTP_SHARED_COMPRESSION_PLAN": "1",
        }):
            self.assertEqual(_host_metadata_modes(graph, object()), (False, False))
            with self.assertRaisesRegex(RuntimeError, "host metadata shadow"):
                _host_metadata_modes(eager, object())
            with self.assertRaisesRegex(RuntimeError, "shared compression plan"):
                _host_metadata_modes(eager, native)
            native.mtp_shared_compression_plan_version = 1
            self.assertEqual(_host_metadata_modes(eager, native), (False, True))

    def test_cache_only_writes_prefill_and_draft_repair_but_keeps_verify_and_replay(self):
        port, target, draft, batch = self.make_step(True, direct=False, draft_cache_only=True)
        self.commit(port, batch)
        self.assertEqual([call.get("mtp_draft_cache_only", False) for call in draft.calls],
                         [True, False, True])
        self.assertTrue(all("mtp_draft_cache_only" not in call for call in target.calls))
        self.assertTrue(target.calls[1]["rollback_v4_cache"])
        self.assertEqual(target.calls[2]["input_ids"].values, [[7, 9]])
        self.assertEqual((port.anchor, port.position), (10, 4))
        self.assert_hidden_row(port.hidden, "replay", 1)

    def test_cache_only_preserves_seed_output_and_rejects_nonwrite_calls(self):
        port, target, draft, batch = self.make_step(True, remaining=5, draft_cache_only=True, reuse_seed=True)
        self.commit(port, batch)
        self.assertEqual([call.get("mtp_draft_cache_only", False) for call in draft.calls],
                         [True, False, False])
        self.assertTrue(draft.calls[-1]["return_draft_hidden_states"])
        self.assertEqual(port.propose("r", 10, 4, [6]), 10)
        self.assertEqual(len(draft.calls), 3)
        for engine, options in (
            (target, {"return_hidden": False}),
            (draft, {"return_hidden": True}),
            (draft, {"return_hidden": False, "return_logits": True}),
            (draft, {"return_hidden": False, "rollback": True}),
        ):
            with self.assertRaisesRegex(ValueError, "output-free Draft cache write"):
                port._forward(engine, (10,), 3, [6], draft_hidden=port.hidden, cache_only=True, **options)

    def test_cache_only_does_not_accept_a_missing_completion_receipt(self):
        port, _, draft, _ = self.make_step(True, draft_cache_only=True)
        draft.responses.clear()
        draft.responses.append(output([10]))
        with self.assertRaisesRegex(RuntimeError, "completed cache write"):
            port._write_draft_cache((10,), 2, [6], port.hidden)

    def test_cache_only_opt_in_requires_eager_and_native_capability(self):
        eager = types.SimpleNamespace(enable_graph=False)
        graph = types.SimpleNamespace(enable_graph=True)
        native = types.SimpleNamespace(mtp_draft_cache_only_version=1)
        with patch.dict(os.environ, {"INFINILM_MTP_DRAFT_CACHE_ONLY": "0"}):
            self.assertFalse(_draft_cache_only_enabled(eager, native))
        with patch.dict(os.environ, {"INFINILM_MTP_DRAFT_CACHE_ONLY": "1"}):
            self.assertTrue(_draft_cache_only_enabled(eager, native))
            self.assertFalse(_draft_cache_only_enabled(graph, object()))
            with self.assertRaisesRegex(RuntimeError, "requires the native output policy"):
                _draft_cache_only_enabled(eager, object())

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


class GraphK3DraftSeedTest(unittest.TestCase):
    table = (6, 2)
    candidates = (8, 9, 10)

    def make_verify(self, accepted, *, remaining=9, eos=(), enabled=True, start=255):
        predictions = [11] * 4
        predictions[:accepted] = self.candidates[:accepted]
        target_hc = output(predictions, "verify", rows=4)
        target_hc["execution_report"] = {
            "mode": "device_graph", "device_graph_segments": 1, "host_segments": 0,
        }
        target = Engine(target_hc)
        draft = Engine(*(output([token], "proposal", rows=1) for token in self.candidates))
        core = types.SimpleNamespace(from_list=lambda data, dtype: Tensor(data), int64=64, int32=32)
        port = _V4Transport(target, draft, core, 256, graph_single_direct_available=True,
                            graph_k3_draft_seed=enabled)
        port.states["r"] = _DraftState(output([], "prefill", rows=1)["hidden_states_by_rank"], 7, start)
        self.assertEqual(port.propose_many("r", 7, start, self.table, 3, eos), self.candidates)
        batch = GreedyVerifyBatch.pack([GreedyVerifyRequest("r", start, remaining, 7, self.candidates, eos)], 512)
        return port, target, draft, batch

    def commit(self, port, batch):
        with patch.dict(os.environ, {"INFINILM_MTP_GRAPH_K3_DIRECT_COMMIT": "1"}):
            with port.target_transaction(batch, self.table, graph_packed_direct_commit=True) as transaction:
                (intent,) = batch.resolve(transaction.rows)
                transaction.commit(intent)
        return intent

    def seed_receipt(self, length, segments=1, last=True):
        return {**output([90], "seed", rows=length), "execution_report": {"device_graph_segments": segments},
                "mtp_last_logits_used": last}

    def test_fused_draft_uses_only_retained_hc_across_pages(self):
        for accepted in (1, 2, 3):
            with self.subTest(accepted=accepted):
                port, target, draft, batch = self.make_verify(accepted)
                count = accepted + 1
                draft.responses.append(self.seed_receipt(count))
                self.assertEqual(len(self.commit(port, batch).materialized_rows), count)
                self.assertEqual(len(target.calls), 1)
                self.assertEqual(len(draft.calls), 4)
                fused = draft.calls[-1]
                self.assertIs(fused["mtp_graph_draft_seed"], True)
                self.assertFalse(fused["sample_all_positions"])
                self.assertTrue(fused["return_draft_hidden_states"])
                self.assertNotIn("mtp_last_logits", fused)
                self.assertEqual(fused["input_ids"].values, [[*self.candidates[:accepted], 11]])
                self.assertEqual(fused["position_ids"].values, list(range(255, 255 + count)))
                self.assertEqual(fused["slot_mapping"].values, [6 * 256 + 255, *range(512, 512 + accepted)])
                for rank, tensor in enumerate(fused["target_hidden_states_by_rank"]):
                    self.assertEqual(tensor.values, [("verify", rank, row) for row in range(count)])
                seed = port.states["r"].seed
                self.assertEqual(seed.pages, self.table)
                for rank, tensor in enumerate(seed.hidden):
                    self.assertEqual(tensor.values, [("seed", rank, count - 1)])
                draft.responses.extend((output([91], "next", rows=1), output([92])))
                self.assertEqual(port.propose_many("r", 11, 255 + count, self.table, 3, ()), (90, 91, 92))
                self.assertEqual(draft.calls[-2]["target_hidden_states_by_rank"], seed.hidden)
                self.assertEqual(draft.calls[-2]["position_ids"].values, [255 + count])
                self.assertIsNone(port.states["r"].seed)

    def test_fused_draft_does_not_expose_reserved_page_after_rejection(self):
        port, target, draft, batch = self.make_verify(1, start=254)
        draft.responses.append(self.seed_receipt(2))
        self.commit(port, batch)
        self.assertEqual(target.calls[-1]["block_tables"].values, [[6, 2]])
        self.assertEqual(draft.calls[-1]["block_tables"].values, [[6]])
        self.assertEqual(draft.calls[-1]["slot_mapping"].values, [6 * 256 + 254, 6 * 256 + 255])
        self.assertEqual(port.states["r"].seed.pages, (6,))

    def test_zero_acceptance_terminal_tail_short_budget_and_off_keep_serial_behavior(self):
        for accepted, remaining, eos, enabled, repairs in (
            (0, 9, (), True, 0), (1, 4, (), True, 1), (3, 4, (), True, 0),
            (1, 9, (11,), True, 1), (2, 9, (), False, 2),
        ):
            with self.subTest(accepted=accepted, remaining=remaining, eos=eos, enabled=enabled):
                port, target, draft, batch = self.make_verify(accepted, remaining=remaining, eos=eos, enabled=enabled)
                draft.responses.extend(output([1]) for _ in range(repairs))
                self.commit(port, batch)
                self.assertEqual(len(target.calls), 1)
                self.assertEqual(len(draft.calls), 3 + repairs)
                self.assertTrue(all("mtp_graph_draft_seed" not in call for call in draft.calls))
                self.assertIsNone(port.states["r"].seed)

    def test_submitted_fused_draft_rejects_bad_receipt_without_retry(self):
        for segments, last, expected in ((0, True, "device Graph"), (1, False, "last logits")):
            with self.subTest(segments=segments, last=last):
                port, target, draft, batch = self.make_verify(1)
                draft.responses.append(self.seed_receipt(2, segments, last))
                with self.assertRaisesRegex(RuntimeError, expected):
                    self.commit(port, batch)
                self.assertEqual(len(target.calls), 1)
                self.assertEqual(len(draft.calls), 4)
                self.assertIsNone(port.states["r"].seed)
                self.assertEqual(port.position, 255)
                self.assertIsNone(port.transaction)

    def test_opt_in_requires_direct_single_stage_serial_graph_and_native_abi(self):
        config = types.SimpleNamespace(pipeline_parallel_size=1, nextn_branch_execution="serial")
        native = types.SimpleNamespace(mtp_graph_draft_seed_version=1)
        flag = "INFINILM_MTP_GRAPH_K3_DRAFT_SEED"
        with patch.dict(os.environ, {flag: "0"}):
            self.assertFalse(_graph_k3_draft_seed_enabled(config, True, native))
        with patch.dict(os.environ, {flag: "1"}):
            self.assertTrue(_graph_k3_draft_seed_enabled(config, True, native))
            self.assertFalse(_graph_k3_draft_seed_enabled(config, False, native))
            with self.assertRaisesRegex(RuntimeError, "native owned Draft HC"):
                _graph_k3_draft_seed_enabled(config, True, object())
            config.pipeline_parallel_size = 2
            self.assertFalse(_graph_k3_draft_seed_enabled(config, True, native))
            config.pipeline_parallel_size = 1
            config.nextn_branch_execution = "packed"
            self.assertFalse(_graph_k3_draft_seed_enabled(config, True, native))


if __name__ == "__main__":
    unittest.main()
