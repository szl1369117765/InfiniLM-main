"""Exercise the opt-in C3 final Draft batch with real transport and rank-local CPU substitutes."""

import os
import unittest
from unittest.mock import patch

from test_deepseek_v4_mtp_batched_draft import CacheOps, decode, make_case
from test_deepseek_v4_mtp_c3_direct_draft_matrix import OPTIONS, SPECS, run_arm

FLAG = "INFINILM_MTP_GRAPH_C3_BATCH_FINAL_DRAFT"


def proposal(specs, *, enabled, direct=True, permit_direct=True, keep_final_graph=True):
    runner, requests, draft, _ = make_case(
        False,
        3,
        specs,
        transport_options=OPTIONS,
        modes=(keep_final_graph, True),
    )
    transport = runner.transport
    propose = transport.propose_batched
    observed, proposed = [], []

    def observe(*args, **kwargs):
        with patch.object(draft, "forward_raw", wraps=draft.forward_raw) as native:
            chains = propose(*args, **kwargs)
        observed.extend(call.kwargs for call in native.call_args_list)
        proposed.extend(chains)
        return chains

    with (
        patch.dict(
            os.environ,
            {
                FLAG: "1" if enabled else "0",
                "INFINILM_MTP_GRAPH_C3_DIRECT_COMMIT": "1" if direct else "0",
            },
        ),
        patch.object(transport, "propose_batched", side_effect=observe) as batched,
    ):
        tokens = decode(
            runner, requests, cache_ops=CacheOps(permit_direct=permit_direct)
        )
    return tokens, observed, proposed, batched.call_count


class C3FinalDraftTest(unittest.TestCase):
    def test_packs_three_final_rows_in_owner_order_and_preserves_state(self):
        with patch.dict(os.environ):
            os.environ.pop(FLAG, None)
            baseline, baseline_calls = run_arm(True, True)
        with patch.dict(os.environ, {FLAG: "1"}):
            candidate, candidate_calls = run_arm(True, True)
        self.assertEqual(candidate, baseline)
        self.assertEqual([len(row[0]) for row in baseline_calls], [5, 5])
        self.assertEqual([len(row[0]) for row in candidate_calls], [3, 3])

        tokens, calls, proposed, _ = proposal(SPECS, enabled=True)
        self.assertEqual(tokens, baseline[0][0])
        self.assertEqual(proposed, [(12, 13, 14), (42, 43, 44), (72, 73, 74)])
        self.assertEqual(
            [row["return_draft_hidden_states"] for row in calls], [True, True, False]
        )
        third = calls[2]
        self.assertEqual(third["input_ids"].rows, [[13, 43, 73]])
        self.assertEqual(third["input_offsets"].rows, [0, 1, 2, 3])
        self.assertEqual(third["position_ids"].rows, [256, 256, 256])
        self.assertEqual(third["block_tables"].rows, [[1, 2], [4, 5], [7, 8]])
        self.assertEqual(third["slot_mapping"].rows, [512, 1280, 2048])
        self.assertFalse(third["sample_all_positions"])
        self.assertNotIn("target_hidden_states_by_rank", third)
        rank_views = third["target_hidden_segments_by_rank"]
        self.assertEqual(len(rank_views), 8)
        self.assertTrue(
            all(
                [view.rows[0][:3] for view in views]
                == [(rank, owner, 256) for owner in (1, 4, 7)]
                for rank, views in enumerate(rank_views)
            )
        )

    def test_scoped_fallbacks_keep_existing_calls(self):
        for label, specs, options in (
            ("protected intent", SPECS, {"direct": False}),
            ("private lease denied", SPECS, {"permit_direct": False}),
            ("mixed budgets", (SPECS[0], ("b", 4, 3, ()), SPECS[2]), {}),
            ("short cohort", SPECS[:2], {}),
            ("known stop at first proposal", (*SPECS[:2], ("c", 7, 13, (72,))), {}),
            ("known stop before final proposal", (*SPECS[:2], ("c", 7, 13, (73,))), {}),
            ("base proposal optimization disabled", SPECS, {"keep_final_graph": False}),
        ):
            with self.subTest(label=label):
                expected, original, _, original_count = proposal(
                    specs, enabled=False, **options
                )
                actual, candidate, _, candidate_count = proposal(
                    specs, enabled=True, **options
                )
                self.assertEqual(actual, expected)
                self.assertEqual(candidate_count, original_count)
                self.assertEqual(
                    [call["input_ids"].rows for call in candidate],
                    [call["input_ids"].rows for call in original],
                )
                self.assertEqual(
                    [call["input_offsets"].rows for call in candidate],
                    [call["input_offsets"].rows for call in original],
                )


if __name__ == "__main__":
    unittest.main()
