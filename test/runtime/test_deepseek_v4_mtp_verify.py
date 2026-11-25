import unittest
from unittest.mock import Mock

from infinilm.llm.model_runner.deepseek_v4_mtp_verify import (
    GreedyVerifyBatch,
    GreedyVerifyRequest,
    TargetVerifyRow,
    run_greedy_verify,
)


class DeepseekV4MtpVerifyTest(unittest.TestCase):
    def test_mixed_queries_keep_correction_and_bonus_for_next_round(self):
        requests = (
            GreedyVerifyRequest("a", 254, 5, 10, (11, 12, 13)),
            GreedyVerifyRequest("b", 129, 4, 20, (21, 22)),
            GreedyVerifyRequest("c", 80, 3, 30, (31,), frozenset({31})),
        )

        def verify(batch):
            self.assertEqual(batch.input_offsets, (0, 4, 7, 9))
            self.assertEqual(batch.input_ids, (10, 11, 12, 13, 20, 21, 22, 30, 31))
            self.assertEqual(batch.position_ids, (254, 255, 256, 257, 129, 130, 131, 80, 81))
            self.assertEqual(batch.past_kv_lengths, (254, 129, 80))
            self.assertEqual(batch.total_kv_lengths, (258, 132, 82))
            self.assertEqual(batch.cu_seqlens, (0, 258, 390, 472))
            predicted = (11, 91, 13, 92, 21, 22, 93, 31, 94)
            return tuple(TargetVerifyRow(row, token) for row, token in zip(batch.rows, predicted))

        intents = run_greedy_verify(requests, verify, max_context=512)
        self.assertEqual(
            [
                (r.owner, r.emitted_tokens, r.retain_length, r.next_anchor, r.next_anchor_kind, r.terminal)
                for r in intents
            ],
            [
                ("a", (10, 11), 256, 91, "correction", False),
                ("b", (20, 21, 22), 132, 93, "bonus", False),
                ("c", (30, 31), 82, None, None, True),
            ],
        )
        self.assertEqual(
            [[row.flat_index for row in r.materialized_rows] for r in intents],
            [[0, 1], [4, 5, 6], [7, 8]],
        )

    def test_cross_owner_and_stale_row_tags_fail_before_commit_intents(self):
        requests = (
            GreedyVerifyRequest("a", 125, 3, 10, (11,)),
            GreedyVerifyRequest("b", 125, 3, 10, (11,)),
        )
        stale = GreedyVerifyBatch.pack(requests, 512)
        attempts = {
            "swapped owners": lambda rows: (*rows[2:], *rows[:2]),
            "stale but same logical row": lambda rows: (stale.rows[0], *rows[1:]),
            "reordered local rows": lambda rows: (rows[1], rows[0], *rows[2:]),
        }
        for description, transform in attempts.items():
            with self.subTest(description=description), self.assertRaisesRegex(
                ValueError, "does not belong to the submitted row"
            ):
                run_greedy_verify(
                    requests,
                    lambda batch: tuple(TargetVerifyRow(row, 11) for row in transform(batch.rows)),
                    max_context=512,
                )

    def test_invalid_owner_and_budget_fail_before_forward(self):
        request = GreedyVerifyRequest("a", 125, 2, 10, (11,))
        for requests, message in (
            ((request, request), "distinct request owners"),
            ((GreedyVerifyRequest("a", 125, 1, 10, (11,)),), "query or output budget"),
            ((GreedyVerifyRequest("a", 511, 2, 10, (11,)),), "exceeds the context"),
        ):
            with self.subTest(reason=message):
                target = Mock()
                with self.assertRaisesRegex(ValueError, message):
                    run_greedy_verify(requests, target, max_context=512)
                target.assert_not_called()


if __name__ == "__main__":
    unittest.main()
