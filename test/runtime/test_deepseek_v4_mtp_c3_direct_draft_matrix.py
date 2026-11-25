"""Exercise both C3 Draft modes through the runner with CPU native substitutes."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from test_deepseek_v4_mtp_batched_draft import CacheOps, decode, make_case


SPECS = (("a", 1, 13, ()), ("b", 4, 10, ()), ("c", 7, 13, ()))
OPTIONS = {
    "graph_packed_direct_available": True,
    "graph_batched_draft_proposal": True,
    "graph_batched_direct_draft_repair": True,
    "batched_target_replay": True,
    "batched_draft_repair": True,
}
DIRECT = {"INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT": "1"}


def run_arm(proposal, repair, *, specs=SPECS, rounds=2, permit_direct=True):
    runner, requests, draft, core = make_case(
        False, 3, specs, transport_options=OPTIONS, modes=(proposal, repair),
    )
    transport = runner.transport
    original = transport.target_transaction
    cache_ops = CacheOps(permit_direct=permit_direct)
    outputs, calls, tables = [], [], []

    for _ in range(rounds):
        begin_draft = len(draft.calls)
        begin_target = len(transport.target.calls)
        proposal_end = None

        @contextmanager
        def observe(*args, **kwargs):
            nonlocal proposal_end
            proposal_end = len(draft.calls)
            with original(*args, **kwargs) as transaction:
                yield transaction

        with patch.dict(os.environ, DIRECT), patch.object(transport, "target_transaction", observe):
            outputs.append(decode(runner, requests, cache_ops=cache_ops))
        tables.append([tuple(req.block_table) for req in requests])
        proposal_shapes = [
            (len(spans), return_hc) for (spans, _), return_hc in
            zip(draft.calls[begin_draft:proposal_end], draft.return_hidden[begin_draft:proposal_end], strict=True)
        ]
        repairs = [tuple((owner, positions) for owner, positions, _ in spans)
                   for spans, _ in draft.calls[proposal_end:]]
        calls.append((proposal_shapes, repairs, len(transport.target.calls) - begin_target))

    owner_state = {owner: (state.anchor, state.position, [tensor.rows for tensor in state.hidden])
                   for owner, state in transport.states.items()}
    assert core.device == "caller" and not core.pending
    return (outputs, tables, owner_state, draft.cache, draft.input_hidden, transport.target.cache), calls


class C3DirectDraftMatrixTest(unittest.TestCase):
    def test_four_arms_keep_tokens_pages_and_hc_with_one_target_call(self):
        baseline, _ = run_arm(False, False)
        for proposal, repair, expected in (
            (False, False, (9, 6)), (True, False, (5, 6)),
            (False, True, (9, 3)), (True, True, (5, 3)),
        ):
            with self.subTest(proposal=proposal, repair=repair):
                state, calls = run_arm(proposal, repair)
                self.assertEqual(state, baseline)
                self.assertEqual([count for _, _, count in calls], [1, 1])
                proposals, repairs, _ = calls[0]
                self.assertEqual((len(proposals), len(repairs)), expected)
                if proposal:
                    self.assertEqual(proposals, [(3, True), (3, True), (1, False), (1, False), (1, False)])
                if repair:
                    self.assertEqual(repairs, [((1, (255,)), (7, (255,))),
                                               ((1, (256,)), (7, (256,))),
                                               ((1, (257,)), (7, (257,)))])
        # Owner b rejected at the page boundary. Its speculative page was not retained.
        self.assertEqual(baseline[1][0], [(1, 2), (4,), (7, 8)])

    def test_budget_tail_is_single_and_terminal_owner_skips_repair(self):
        specs = (SPECS[0], ("b", 4, 3, ()), ("c", 7, 13, (72,)))
        baseline, baseline_calls = run_arm(False, False, specs=specs, rounds=1)
        state, calls = run_arm(True, True, specs=specs, rounds=1)
        self.assertEqual(state, baseline)
        proposals, repairs, target = calls[0]
        self.assertEqual(proposals, [(1, False), (2, True), (1, True), (1, False)])
        self.assertEqual(repairs, [((1, (255,)),), ((1, (256,)),), ((1, (257,)),)])
        self.assertEqual(repairs, baseline_calls[0][1])
        self.assertEqual(target, 1)

    def test_protected_fallback_keeps_the_existing_repair_mode(self):
        baseline = run_arm(False, False, permit_direct=False)
        self.assertEqual(run_arm(False, True, permit_direct=False), baseline)
        self.assertEqual([count for _, _, count in baseline[1]], [2, 2])

    def test_modes_can_be_selected_independently_only_when_prepared_and_idle(self):
        runner, _, _, _ = make_case(False, 3, SPECS, transport_options=OPTIONS)
        transport = runner.transport
        with self.assertRaisesRegex(RuntimeError, "idle"):
            transport.set_c3_draft_modes(proposal=False, repair=False)
        for owner, *_ in SPECS:
            transport.release(owner)
        for proposal, repair in ((False, False), (True, False), (False, True), (True, True)):
            transport.set_c3_draft_modes(proposal=proposal, repair=repair)
            self.assertEqual((transport.graph_batched_draft_proposal,
                              transport.graph_batched_direct_draft_repair), (proposal, repair))
        unprepared, _, _, _ = make_case(False, 3, SPECS)
        for owner, *_ in SPECS:
            unprepared.transport.release(owner)
        with self.assertRaisesRegex(RuntimeError, "prepared"):
            unprepared.transport.set_c3_draft_modes(proposal=True, repair=True)

    def test_failed_direct_batched_repair_never_retries_or_publishes_owner_state(self):
        runner, requests, _, _ = make_case(
            False, 3, SPECS, transport_options=OPTIONS, modes=(False, True),
        )
        transport = runner.transport
        before = dict(transport.states)
        initial = [list(req.generated_token_ids) for req in requests]
        target_calls = len(transport.target.calls)
        with (patch.dict(os.environ, DIRECT),
              patch.object(transport, "_forward_draft_segments", side_effect=RuntimeError("Draft write failed")) as repair):
            with self.assertRaisesRegex(RuntimeError, "Draft write failed"):
                decode(runner, requests, cache_ops=CacheOps(permit_direct=True))
        self.assertEqual(repair.call_count, 1)
        self.assertTrue(runner.failed)
        self.assertTrue(all(transport.states[owner] is state for owner, state in before.items()))
        self.assertEqual([req.generated_token_ids for req in requests], initial)
        self.assertEqual(len(transport.target.calls) - target_calls, 1)


if __name__ == "__main__":
    unittest.main()
