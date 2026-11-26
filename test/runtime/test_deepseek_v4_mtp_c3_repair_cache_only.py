"""Exercise the C3 Draft repair cache-only routing through the runner."""

import os
import unittest
from unittest.mock import patch

from test_deepseek_v4_mtp_batched_draft import CacheOps, decode, make_case
from test_deepseek_v4_mtp_c3_direct_draft_matrix import DIRECT, OPTIONS, SPECS

FLAG = "INFINILM_MTP_GRAPH_C3_DRAFT_REPAIR_CACHE_ONLY"
ALL_ACCEPT = (SPECS[0], ("b", 10, 13, ()), SPECS[2])


def run(enabled, *, specs, permit_direct=True):
    runner, requests, draft, _ = make_case(False, 3, specs, transport_options=OPTIONS)
    draft.mtp_c3_draft_repair_cache_only_version = 1
    native_forward = draft.forward_raw
    cache_calls = []

    def forward(**options):
        result = native_forward(**options)
        if not options.get("mtp_draft_cache_only"):
            return result
        assert options["input_offsets"].rows == [0, 1, 2, 3]
        assert "target_hidden_states_by_rank" not in options
        assert len(options["target_hidden_segments_by_rank"]) == 8
        assert (
            not options["return_draft_hidden_states"]
            and not options["sample_all_positions"]
        )
        cache_calls.append(draft.calls[-1])
        return {"mtp_draft_cache_only_used": True}

    with (
        patch.dict(os.environ, {**DIRECT, FLAG: str(int(enabled))}),
        patch.object(draft, "forward_raw", side_effect=forward),
    ):
        outputs = [
            decode(runner, requests, cache_ops=CacheOps(permit_direct=permit_direct))
            for _ in range(2)
        ]

    states = {
        owner: (state.position, state.anchor, [tensor.rows for tensor in state.hidden])
        for owner, state in runner.transport.states.items()
    }
    return (
        outputs,
        [request.block_table for request in requests],
        states,
        draft.calls,
        draft.cache,
        draft.input_hidden,
    ), cache_calls


class C3DraftRepairCacheOnlyTest(unittest.TestCase):
    def test_only_repair_triples_return_without_logits_and_keep_all_slots(self):
        baseline, _ = run(False, specs=ALL_ACCEPT)
        candidate, cache_calls = run(True, specs=ALL_ACCEPT)
        self.assertEqual(candidate, baseline)
        self.assertEqual(len(cache_calls), 6)
        self.assertEqual(
            [[positions for _, positions, _ in spans] for spans, _ in cache_calls],
            [[(position,)] * 3 for position in (255, 256, 257, 259, 260, 261)],
        )

    def test_protected_and_two_owner_repair_reuse_previous_route(self):
        baseline, _ = run(False, specs=ALL_ACCEPT, permit_direct=False)
        self.assertEqual(
            run(True, specs=ALL_ACCEPT, permit_direct=False), (baseline, [])
        )
        mixed_baseline, _ = run(False, specs=SPECS)
        mixed_candidate, cache_calls = run(True, specs=SPECS)
        self.assertEqual(mixed_candidate, mixed_baseline)
        self.assertEqual(
            [[positions[0] for _, positions, _ in spans] for spans, _ in cache_calls],
            [[259, 256, 259], [260, 257, 260], [261, 258, 261]],
        )

    def test_bad_receipt_stops_the_transaction_before_publication(self):
        for response in (
            {},
            {"mtp_draft_cache_only_used": True, "output_ids": object()},
        ):
            with self.subTest(response=response):
                runner, requests, draft, _ = make_case(
                    False, 3, ALL_ACCEPT, transport_options=OPTIONS
                )
                draft.mtp_c3_draft_repair_cache_only_version = 1
                native_forward = draft.forward_raw
                states = dict(runner.transport.states)
                published = [list(request.generated_token_ids) for request in requests]

                def forward(**options):
                    return (
                        response
                        if options.get("mtp_draft_cache_only")
                        else native_forward(**options)
                    )

                with (
                    patch.dict(os.environ, {**DIRECT, FLAG: "1"}),
                    patch.object(draft, "forward_raw", side_effect=forward) as native,
                ):
                    with self.assertRaisesRegex(RuntimeError, "completed cache write"):
                        decode(runner, requests, cache_ops=CacheOps(permit_direct=True))
                self.assertEqual(
                    sum(
                        bool(call.kwargs.get("mtp_draft_cache_only"))
                        for call in native.call_args_list
                    ),
                    1,
                )
                self.assertTrue(runner.failed)
                self.assertTrue(
                    all(
                        runner.transport.states[owner] is state
                        for owner, state in states.items()
                    )
                )
                self.assertEqual(
                    [request.generated_token_ids for request in requests], published
                )


if __name__ == "__main__":
    unittest.main()
