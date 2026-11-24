"""Check protected K3 Draft repair with real runner and CPU engine substitutes."""

import unittest
from unittest.mock import patch

from test_deepseek_v4_mtp_batched_draft import decode, make_case


def run_case(enabled, *, first_stop=(), target_replay=True, fail_packed=False):
    specs = (("long", 1, 12, first_stop), ("reject", 4, 9, ()), ("partial", 7, 12, ()))
    runner, requests, draft, core = make_case(False, 3, specs)
    transport = runner.transport
    transport.batched_target_replay = target_replay
    transport.batched_draft_repair = enabled
    original_target = transport.target.forward_raw
    original_single = transport._write_draft_cache
    original_packed = transport._forward_draft_segments
    repairs = []
    prefilled = dict(transport.states)

    def target_forward(**options):
        result = original_target(**options)
        for span, table in enumerate(options["block_tables"].rows):
            start, end = options["input_offsets"].rows[span:span + 2]
            for row in range(start, end):
                if table[0] == 7 and options["position_ids"].rows[row] == 257:
                    result["output_ids"].rows[row] += 1
        return result

    def single(tokens, start, table, hidden):
        repairs.append(((table[0], start),))
        return original_single(tokens, start, table, hidden)

    def packed(segments, *, return_hidden):
        if return_hidden:
            return original_packed(segments, return_hidden=True)
        repairs.append(tuple((segment.table[0], segment.start) for segment in segments))
        if fail_packed:
            raise RuntimeError("simulated packed Draft repair failure")
        assert all(len(segment.tokens) == 1 for segment in segments)
        return original_packed(segments, return_hidden=False)

    with (
        patch.object(transport.target, "forward_raw", target_forward),
        patch.object(transport, "_write_draft_cache", single),
        patch.object(transport, "_forward_draft_segments", packed),
    ):
        try:
            output = decode(runner, requests)
        except RuntimeError as error:
            if fail_packed and str(error) == "simulated packed Draft repair failure":
                return runner, transport, prefilled, repairs
            raise

    state = {
        owner: (value.anchor, value.position, [tensor.rows for tensor in value.hidden])
        for owner, value in transport.states.items()
    }
    assert core.device == "caller" and not core.pending
    return output, repairs, draft.cache, draft.input_hidden, transport.target.cache, state


class DraftRepairBatchTest(unittest.TestCase):
    def test_mixed_acceptance_both_sides_of_page_boundary(self):
        serial = run_case(False)
        packed = run_case(True)
        self.assertEqual(packed[0], serial[0])
        self.assertEqual(packed[2:], serial[2:])
        self.assertEqual(serial[1], [((1, 255),), ((1, 256),), ((1, 257),), ((7, 255),), ((7, 256),)])
        self.assertEqual(packed[1], [((1, 255), (7, 255)), ((1, 256), (7, 256)), ((1, 257),)])

    def test_terminal_owner_skips_repair(self):
        serial = run_case(False, first_stop=(12,))
        packed = run_case(True, first_stop=(12,))
        self.assertEqual(packed, serial)
        self.assertEqual(packed[1], [((7, 255),), ((7, 256),)])

    def test_disabled_target_replay_keeps_serial_repair(self):
        self.assertEqual(run_case(True, target_replay=False), run_case(False, target_replay=False))

    def test_failed_packed_write_does_not_publish_owner_states(self):
        runner, transport, prefilled, repairs = run_case(True, fail_packed=True)
        self.assertTrue(runner.failed)
        self.assertEqual(repairs, [((1, 255), (7, 255))])
        for owner, state in prefilled.items():
            self.assertIs(transport.states[owner], state)


if __name__ == "__main__":
    unittest.main()
