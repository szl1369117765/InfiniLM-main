"""Exercise classic MTP request retirement through the real paged LLM engine."""

import importlib
import queue
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[2] / "python" / "infinilm"


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


class NativeTransport:
    def __init__(self):
        self.calls = []
        self.after_prefill = None
        self.release_error = None

    def prefill(self, owner, model_input):
        self.calls.append(("prefill", owner))
        if self.after_prefill is not None:
            self.after_prefill()
        return 71

    def release(self, owner):
        self.calls.append(("release", owner))
        if self.release_error is not None:
            raise self.release_error

    def prefill_many(self, owners, model_input):
        self.calls.append(("prefill_many", tuple(owners)))
        return [71] * len(owners)


class ClassicMTPLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = {"model_type": "deepseek_v4", "max_position_embeddings": 2048}
        stubs = {
            name: module(name, __path__=[str(SOURCE.joinpath(*name.split(".")[1:]))])
            for name in (
                "infinilm",
                "infinilm.config",
                "infinilm.llm",
                "infinilm.llm.model_runner",
            )
        }
        names = {
            "janus": {"Queue": lambda: SimpleNamespace(sync_q=queue.Queue())},
            "xxhash": {},  # Classic MTP disables prefix hashing.
            "infinilm.infer_engine": {
                "model_uses_mamba_cache": lambda _: False,
                "read_hf_config": lambda _: config,
            },
            "infinilm.kv_connector": {
                "KVConnectorFactory": None,
                "KVConnectorRole": None,
            },
            "infinilm.llm.model_runner.model_runner": {"ModelRunner": None},
            "infinilm.multimodal.multimodal": {"resolve_multimodal_inputs": None},
            "infinilm.processors.basic_llm_processor": {"BasicLLMProcessor": None},
        }
        stubs.update({name: module(name, **attrs) for name, attrs in names.items()})
        # A native-free fixture must bind this checkout even when another test
        # previously imported the installed Python package.
        patcher = patch.dict(sys.modules, stubs)
        patcher.start()
        cls.addClassCleanup(patcher.stop)
        for name in list(sys.modules):
            if name.startswith("infinilm.") and name not in stubs:
                del sys.modules[name]
        cls.llm = importlib.import_module("infinilm.llm.llm")
        cls.reason = cls.llm.FinishReason
        cls.runner = importlib.import_module(
            "infinilm.llm.model_runner.deepseek_v4_mtp_runner"
        ).DeepSeekV4MTPRunner
        cls.target = SimpleNamespace(hf_config=config, model_type="deepseek_v4")

    def setUp(self):
        self.transport = NativeTransport()
        self.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[-1]))
        self.engine = self.make_engine()

    def make_engine(self, max_batch_size=1):
        runner = self.runner(self.target, self.transport, 1024)

        def execute_model(batch):
            rows = runner.forward(batch, {})
            return SimpleNamespace(
                req_ids=[req.request_id for req in batch.scheduled_requests],
                sampled_token_ids=rows,
                kv_connector_output=None,
            )

        model_runner = SimpleNamespace(
            device="cpu",
            dtype="bfloat16",
            eos_token_id=[],
            processor=SimpleNamespace(get_tokenizer=lambda: self.tokenizer),
            model_engine=self.target,
            deepseek_v4_mtp_runner=runner,
            execute_model=execute_model,
        )
        with patch.object(self.llm, "ModelRunner", return_value=model_runner):
            return self.llm.LLMEngine(
                self.llm.EngineConfig(
                    model_path="target",
                    draft_model_path="draft",
                    tensor_parallel_size=8,
                    max_batch_size=max_batch_size,
                    num_draft_tokens=1,
                    num_blocks=4,
                    attn_backend="paged-attn",
                    enable_prefix_caching=False,
                    enable_deepseek_v4_mtp=True,
                )
            )

    def enqueue(self, owner, budget=2):
        request = self.llm.InferenceRequest(
            owner,
            prompt_token_ids=[7],
            sampling_params=self.llm.SamplingParams(top_k=1, max_tokens=budget),
        )
        request.output_queue
        self.engine.add_request(request)
        return request

    def outputs(self, req):
        outputs = []
        stream = req.output_queue.sync_q
        while not stream.empty():
            result = stream.get_nowait()
            outputs.append((result.token_id, result.finished, result.finish_reason))
        return outputs

    def test_waiting_and_running_abort_skip_native_and_release_once(self):
        active = self.enqueue("active")
        waiting = self.enqueue("waiting")
        replacement = self.enqueue("replacement", budget=1)
        self.assertTrue(self.engine.step())
        active.abort()
        waiting.abort()
        self.engine.step()
        self.assertTrue(self.engine.step())
        self.assertFalse(self.engine.step())
        self.assertEqual(
            self.outputs(active), [(71, False, None), (-1, True, self.reason.CANCELED)]
        )
        self.assertEqual(self.outputs(waiting), [(-1, True, self.reason.CANCELED)])
        self.assertEqual(self.outputs(replacement), [(71, True, self.reason.LENGTH)])
        self.assertEqual(
            self.transport.calls,
            [
                ("prefill", "active"),
                ("release", "active"),
                ("prefill", "replacement"),
                ("release", "replacement"),
            ],
        )
        self.assert_usable_pages(4)

    def test_abort_after_schedule_skips_native(self):
        request = self.enqueue("scheduled")
        schedule = self.engine.scheduler.schedule

        def abort_after_schedule():
            batch = schedule()
            request.abort()
            return batch

        with patch.object(self.engine.scheduler, "schedule", abort_after_schedule):
            self.assertTrue(self.engine.step())
        self.assertEqual(self.transport.calls, [])
        self.assertEqual(self.outputs(request), [(-1, True, self.reason.CANCELED)])
        self.assert_usable_pages(4)

    def test_abort_after_c3_schedule_preserves_uncanceled_prefill_rows(self):
        self.engine = self.make_engine(max_batch_size=3)
        canceled = self.enqueue("canceled", budget=1)
        remaining = [self.enqueue(owner, budget=1) for owner in ("left", "right")]
        schedule = self.engine.scheduler.schedule

        def cancel_one():
            batch = schedule()
            self.assertEqual(batch.num_requests, 3)
            canceled.abort()
            return batch

        with patch.object(self.engine.scheduler, "schedule", cancel_one):
            self.assertTrue(self.engine.step())
        self.assertEqual(self.outputs(canceled), [(-1, True, self.reason.CANCELED)])
        for request in remaining:
            self.assertEqual(self.outputs(request), [(71, True, self.reason.LENGTH)])
        self.assertEqual(
            self.transport.calls,
            [("prefill_many", ("left", "right")), ("release", "left"), ("release", "right")],
        )
        self.assert_usable_pages(4)

    def test_inflight_abort_blocks_token_and_releases_request(self):
        for checkpoint in ("native", "tokenizer"):
            with self.subTest(checkpoint=checkpoint):
                request = self.enqueue(checkpoint)
                if checkpoint == "native":
                    self.transport.after_prefill = request.abort
                    self.assertTrue(self.engine.step())
                    self.transport.after_prefill = None
                else:

                    def abort_in_decode(tokens):
                        request.abort()
                        return "discarded"

                    with patch.object(self.tokenizer, "decode", abort_in_decode):
                        self.assertTrue(self.engine.step())
                self.assertEqual(
                    self.outputs(request), [(-1, True, self.reason.CANCELED)]
                )
                self.assertEqual(list(request.generated_token_ids), [])
                self.assertEqual(
                    self.transport.calls[-2:],
                    [("prefill", checkpoint), ("release", checkpoint)],
                )
                self.assert_usable_pages(4)

    def test_release_failure_preserves_delivered_terminal_and_notifies_waiter(self):
        complete = self.enqueue("complete", budget=1)
        waiting = self.enqueue("waiting")
        failure = RuntimeError("injected release fault")
        self.transport.release_error = failure
        with self.assertRaises(RuntimeError) as caught:
            self.engine.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.outputs(complete), [(71, True, self.reason.LENGTH)])
        self.assertEqual(self.outputs(waiting), [(-1, True, self.reason.ERROR)])
        self.assertEqual(complete.finish_reason, self.reason.LENGTH)
        self.assert_faulted()
        self.assertEqual(
            self.transport.calls, [("prefill", "complete"), ("release", "complete")]
        )

    def test_post_forward_failure_fails_closed_with_pages_still_owned(self):
        request = self.enqueue("active")
        waiting = self.enqueue("waiting")
        failure = RuntimeError("injected tokenizer fault")
        with patch.object(self.tokenizer, "decode", side_effect=failure):
            with self.assertRaises(RuntimeError) as caught:
                self.engine.step()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.outputs(request), [(-1, True, self.reason.ERROR)])
        self.assertEqual(self.outputs(waiting), [(-1, True, self.reason.ERROR)])
        self.assertIs(self.engine.scheduler._active_requests["active"], request)
        self.assert_usable_pages(3)
        self.assert_faulted()
        self.assertEqual(self.transport.calls, [("prefill", "active")])

    def assert_faulted(self):
        with self.assertRaisesRegex(RuntimeError, "must be closed after a failed step"):
            self.engine.step()
        with self.assertRaisesRegex(RuntimeError, "must be closed after a failed step"):
            self.enqueue("late")

    def assert_usable_pages(self, expected):
        self.assertEqual(
            self.engine.scheduler.cache_manager.get_total_usable_blocks(), expected
        )


if __name__ == "__main__":
    unittest.main()
