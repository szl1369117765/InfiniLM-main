"""Exercise classic MTP request retirement through the real paged LLM engine."""

import ast
import asyncio
from contextlib import contextmanager
import importlib
import queue
import sys
import threading
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
        cls.verify_row = importlib.import_module(
            "infinilm.llm.model_runner.deepseek_v4_mtp_verify"
        ).TargetVerifyRow
        cls.native_transport = importlib.import_module(
            "infinilm.llm.model_runner.deepseek_v4_mtp_transport"
        )
        cls.target = SimpleNamespace(
            hf_config=config, model_type="deepseek_v4", eos_token_id=None
        )

    def setUp(self):
        self.transport = NativeTransport()
        self.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[-1]))
        self.engine = self.make_engine()

    def make_engine(self, max_batch_size=1, num_draft_tokens=1, branch_lengths=None):
        runner = self.runner(
            self.target, self.transport, 1024, num_draft_tokens=num_draft_tokens,
            branch_draft_lengths=branch_lengths,
        )

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
                    num_draft_tokens=num_draft_tokens,
                    nextn_branch_draft_lengths=branch_lengths,
                    num_blocks=4,
                    attn_backend="paged-attn",
                    enable_prefix_caching=False,
                    enable_deepseek_v4_mtp=True,
                )
            )

    def enqueue(self, owner, budget=2, **sampling):
        request = self.llm.InferenceRequest(
            owner,
            prompt_token_ids=[7],
            sampling_params=self.llm.SamplingParams(
                top_k=1, max_tokens=budget, **sampling
            ),
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

    def test_zero_budget_finishes_without_scheduler_or_native_ownership(self):
        with patch.object(self.engine.scheduler, "add_request") as submit:
            zero = self.enqueue("empty", budget=0)
            submit.assert_not_called()
        self.assertTrue(zero.is_finished())
        self.assertEqual(list(zero.generated_token_ids), [])
        self.assertEqual(self.outputs(zero), [(-1, True, self.reason.LENGTH)])
        self.assertEqual(zero.block_table, [])
        self.assertFalse(self.engine.step())
        self.assertEqual(self.transport.calls, [])
        with self.assertRaisesRegex(ValueError, "new waiting request"):
            self.engine.add_request(zero)
        self.assert_usable_pages(4)

    def test_prefill_stop_id_wins_over_ignore_eos_and_hides_text(self):
        request = self.enqueue("stop", stop_token_ids=[71], ignore_eos=True)
        self.assertTrue(self.engine.step())
        self.assertEqual(self.outputs(request), [(71, True, self.reason.EOS_TOKEN)])
        self.assertEqual(list(request.generated_token_ids), [71])
        self.assertEqual(request.generated_text, "")
        self.assertEqual(
            self.transport.calls, [("prefill", "stop"), ("release", "stop")]
        )
        self.assert_usable_pages(4)

    def test_packed_stop_id_only_shortens_its_own_verify(self):
        self.engine = self.make_engine(max_batch_size=2, num_draft_tokens=3)
        stopped = self.enqueue(
            "stop", budget=5, stop_token_ids=[72], ignore_eos=True
        )
        survivor = self.enqueue("survivor", budget=5, ignore_eos=True)
        queries = []
        commits = []

        def propose_many(owner, anchor, position, table, count, stop_ids):
            proposals = []
            for token in range(anchor + 1, anchor + count + 1):
                proposals.append(token)
                if token in stop_ids:
                    break
            return tuple(proposals)

        @contextmanager
        def target_transaction(batch, tables):
            queries.append(batch.input_offsets)
            rows = tuple(
                self.verify_row(row, row.input_token + 1) for row in batch.rows
            )
            yield SimpleNamespace(rows=rows, commit=commits.append)

        self.assertTrue(self.engine.step())
        with (
            patch.object(self.transport, "propose_many", propose_many, create=True),
            patch.object(
                self.transport, "target_transaction", target_transaction, create=True
            ),
        ):
            self.assertTrue(self.engine.step())
        self.assertEqual(queries, [(0, 2, 6)])
        self.assertEqual(
            [(row.owner, row.terminal) for row in commits[0]],
            [("stop", True), ("survivor", False)],
        )
        self.assertEqual(
            self.outputs(stopped), [(71, False, None), (72, True, self.reason.EOS_TOKEN)]
        )
        self.assertEqual(list(survivor.generated_token_ids), [71, 72, 73, 74, 75])
        self.assertTrue(survivor.is_finished())
        self.assert_usable_pages(4)

    def test_text_stop_masks_a_cross_token_marker_and_retires_cross_page_suffix(self):
        self.engine = self.make_engine(max_batch_size=2, num_draft_tokens=3)
        stopped = self.llm.InferenceRequest(
            "stop",
            prompt_token_ids=[7] * 254,
            sampling_params=self.llm.SamplingParams(
                max_tokens=5, stop=["END"], ignore_eos=True
            ),
        )
        stopped.output_queue
        self.engine.add_request(stopped)
        survivor = self.enqueue("survivor", budget=5)
        chunks = {71: "left E", 72: "ND suffix", 73: " next", 74: " last", 75: "!"}
        self.tokenizer.decode = lambda tokens: chunks[tokens[-1]]
        self.assertTrue(self.engine.step())
        first = stopped.output_queue.sync_q.get_nowait()
        self.assertEqual((first.token_text, first.generated_text), ("left ", "left "))
        self.assertFalse(first.finished)
        self.assertEqual(len(stopped.block_table), 1)
        commits = []

        def propose_many(owner, anchor, position, table, count, stop_ids):
            return tuple(range(anchor + 1, anchor + count + 1))

        @contextmanager
        def target_transaction(batch, tables):
            self.assertEqual(batch.input_offsets, (0, 4, 8))
            self.assertEqual([len(table) for table in tables], [2, 1])
            rows = tuple(
                self.verify_row(row, row.input_token + 1) for row in batch.rows
            )
            yield SimpleNamespace(rows=rows, commit=commits.append)

        with (
            patch.object(self.transport, "propose_many", propose_many, create=True),
            patch.object(
                self.transport, "target_transaction", target_transaction, create=True
            ),
        ):
            self.assertTrue(self.engine.step())
        end = stopped.output_queue.sync_q.get_nowait()
        self.assertEqual(
            (end.token_id, end.token_text, end.generated_text, end.finish_reason),
            (72, "", "left ", self.reason.STOP_STRING),
        )
        self.assertTrue(end.finished)
        self.assertTrue(stopped.output_queue.sync_q.empty())
        self.assertEqual(list(stopped.generated_token_ids), [71, 72])
        self.assertEqual(stopped.to_request_output().outputs[0].text, "left ")
        self.assertEqual(len(commits[0][0].materialized_rows), 4)
        self.assertEqual(list(survivor.generated_token_ids), [71, 72, 73, 74, 75])
        self.assertEqual(survivor.finish_reason, self.reason.LENGTH)
        self.assertEqual(
            self.transport.calls[-2:], [("release", "stop"), ("release", "survivor")]
        )
        self.assert_usable_pages(4)
        followup = self.enqueue("stop", budget=1)
        self.assertTrue(self.engine.step())
        self.assertEqual(list(followup.generated_token_ids), [71])
        self.assert_usable_pages(4)

    def test_text_stop_flushes_a_partial_marker_at_length_limit(self):
        self.tokenizer.decode = lambda _: "left E"
        request = self.enqueue("length", budget=1, stop=["END"])
        self.assertTrue(self.engine.step())
        end = request.output_queue.sync_q.get_nowait()
        self.assertEqual((end.token_text, end.generated_text), ("left E", "left E"))
        self.assertEqual(end.finish_reason, self.reason.LENGTH)

    def test_text_stop_detects_the_marker_before_an_incomplete_utf8_suffix(self):
        self.tokenizer.decode = lambda _: "left END\ufffd"
        request = self.enqueue("stop", budget=2, stop=["END"])
        self.assertTrue(self.engine.step())
        end = request.output_queue.sync_q.get_nowait()
        self.assertEqual((end.token_text, end.generated_text), ("left ", "left "))
        self.assertEqual(end.finish_reason, self.reason.STOP_STRING)
        self.assertEqual(list(request.generated_token_ids), [71])
        self.assert_usable_pages(4)

    def test_serial_greedy_branches_publish_only_the_ordered_winner(self):
        class Hidden:
            def narrow(self, axis, start, length):
                return self

        for tie in (False, True):
            with self.subTest(tie=tie):
                draft = object()
                hidden = (Hidden(),)
                self.transport = self.native_transport._V4Transport(
                    self.target, draft, None, 256
                )
                self.engine = self.make_engine(num_draft_tokens=3, branch_lengths=(1, 3))
                calls = []

                def prefill(owner, model_input):
                    self.transport.states[owner] = self.native_transport._DraftState(
                        hidden, 71, 254
                    )
                    return 71

                def forward(engine, tokens, start, table, *, rollback=False, **options):
                    calls.append((engine is self.target, tuple(tokens), rollback))
                    predicted = [token + 1 for token in tokens]
                    if engine is self.target and rollback and len(tokens) == 4 and tie:
                        predicted[1] = 90
                    return tuple(predicted), hidden

                request = self.llm.InferenceRequest(
                    "branch",
                    prompt_token_ids=[7] * 254,
                    sampling_params=self.llm.SamplingParams(max_tokens=5),
                )
                self.engine.add_request(request)
                with (
                    patch.object(self.transport, "prefill", prefill),
                    patch.object(self.transport, "_forward", forward),
                ):
                    self.assertTrue(self.engine.step())
                    self.assertTrue(self.engine.step())
                winner = (71, 72) if tie else (71, 72, 73, 74)
                self.assertEqual(
                    [row[1:] for row in calls if row[0]],
                    [((71, 72), True), ((71, 72, 73, 74), True), (winner, False)],
                )
                self.assertEqual(
                    [row[1] for row in calls[:3]], [(71,), (72,), (73,)]
                )
                self.assertEqual(
                    list(request.generated_token_ids),
                    [71, 72, 73] if tie else [71, 72, 73, 74, 75],
                )
                if tie:
                    self.assertEqual(self.transport.states["branch"].anchor, 73)
                    request.abort()
                    self.engine.step()
                self.assertFalse(self.transport.states)
                self.assert_usable_pages(4)

    def test_classic_serial_branch_configuration_rejects_unsupported_modes(self):
        configure = self.llm.EngineConfig
        required = {
            "model_path": "target", "draft_model_path": "draft",
            "num_draft_tokens": 3, "max_batch_size": 1,
            "attn_backend": "paged-attn", "enable_prefix_caching": False,
            "enable_deepseek_v4_mtp": True, "nextn_branch_draft_lengths": [3, 1],
        }
        self.assertEqual(configure(**required).nextn_branch_draft_lengths, (3, 1))
        for change in (
            {"enable_graph": True}, {"max_batch_size": 2},
            {"nextn_branch_draft_lengths": [1, 1]},
        ):
            with self.subTest(change=change), self.assertRaisesRegex(
                ValueError, "serial branches require eager C1"
            ):
                configure(**{**required, **change})

    def test_real_http_sampling_parser_accepts_mtp_stop_ids_and_zero_alias(self):
        path = SOURCE / "server" / "inference_server.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        server_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "InferenceServer"
        )
        method = next(
            node for node in server_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_sampling_params"
        )
        sampling = importlib.import_module("infinilm.llm.sampling_params")
        scope = {
            "SamplingParams": sampling.SamplingParams,
            "validate_stop_strings": sampling.validate_stop_strings,
            "validate_stop_token_ids": sampling.validate_stop_token_ids,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope
        )
        parse = scope[method.name]
        server = SimpleNamespace(
            enable_deepseek_v4_mtp=True, ignore_eos=False,
            max_tokens=6, temperature=1.0, top_p=1.0, top_k=1,
        )
        params = parse(
            server, {"max_completion_tokens": 0, "sampling_params": {"stop_token_ids": [72]}}
        )
        self.assertEqual((params.max_tokens, params.stop_token_ids), (0, [72]))
        with self.assertRaisesRegex(ValueError, "stop_token_ids"):
            parse(server, {"stop_token_ids": [True]})
        self.assertEqual(parse(server, {"stop": "END"}).stop, ["END"])
        with self.assertRaisesRegex(ValueError, "supports integer n=1 only"):
            parse(server, {"n": 2})
        with self.assertRaisesRegex(ValueError, "stop must contain nonempty strings"):
            parse(server, {"stop": [""]})
        server.enable_deepseek_v4_mtp = False
        with self.assertRaisesRegex(ValueError, "max_tokens must be greater than 0"):
            parse(server, {"max_tokens": 0})

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
        for async_owner in (False, True):
            with self.subTest(async_owner=async_owner):
                self.transport.calls.clear()
                self.engine = self.make_engine(max_batch_size=3)
                self.engine._async_abort_owner = async_owner
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

    def test_async_owner_flag_only_abort_retires_running_and_waiting(self):
        self.engine._async_abort_owner = True
        active = self.enqueue("async-active")
        waiting = self.enqueue("async-waiting")
        self.assertTrue(self.engine.step())
        active.abort()
        waiting.abort()

        native_result = SimpleNamespace(to_numpy=lambda: SimpleNamespace(tolist=lambda: [72]))
        with patch.object(self.target, "forward", return_value=native_result, create=True) as tail:
            self.engine.step()
            self.engine.step()
            tail.assert_not_called()
        self.assertEqual(
            self.outputs(active), [(71, False, None), (-1, True, self.reason.CANCELED)]
        )
        self.assertEqual(self.outputs(waiting), [(-1, True, self.reason.CANCELED)])
        self.assertEqual(
            self.transport.calls, [("prefill", "async-active"), ("release", "async-active")]
        )
        self.assertFalse(self.engine.step())
        self.assertFalse(self.engine.model_runner.deepseek_v4_mtp_runner.requests)
        self.assert_usable_pages(4)

    def test_async_owner_flag_only_abort_during_native_or_tokenization(self):
        for checkpoint in ("native", "tokenizer", "close"):
            with self.subTest(checkpoint=checkpoint):
                self.transport = NativeTransport()
                self.engine = self.make_engine()
                self.engine._async_abort_owner = True
                request = self.enqueue(f"async-{checkpoint}")
                if checkpoint == "tokenizer":
                    def abort_in_decode(tokens):
                        request.abort()
                        return "discarded"

                    with patch.object(self.tokenizer, "decode", abort_in_decode):
                        self.assertTrue(self.engine.step())
                elif checkpoint == "close":
                    async def wait_closed():
                        return None

                    async def wait_for(awaitable, timeout):
                        return await awaitable

                    def close_request():
                        with patch.object(asyncio, "wait_for", wait_for):
                            with self.assertRaises(StopIteration):
                                request.close().send(None)

                    stream = request.output_queue
                    stream.async_q = queue.Queue()
                    stream.close = lambda: None
                    stream.wait_closed = wait_closed
                    self.transport.after_prefill = close_request
                    with patch.object(stream.sync_q, "put_nowait", side_effect=RuntimeError("queue closed")):
                        self.assertTrue(self.engine.step())
                else:
                    self.transport.after_prefill = request.abort
                    self.assertTrue(self.engine.step())
                wanted = [] if checkpoint == "close" else [(-1, True, self.reason.CANCELED)]
                self.assertEqual(self.outputs(request), wanted)
                self.assertEqual(request.finish_reason, self.reason.CANCELED)
                self.assertEqual(list(request.generated_token_ids), [])
                self.assertFalse(self.engine.model_runner.deepseek_v4_mtp_runner.requests)
                self.assert_usable_pages(4)

    def test_async_abort_during_packed_cross_page_commit_retires_only_that_owner(self):
        self.engine = self.make_engine(max_batch_size=3, num_draft_tokens=3)
        self.engine._async_abort_owner = True
        canceled = self.llm.InferenceRequest(
            "cross-page", prompt_token_ids=[7] * 255,
            sampling_params=self.llm.SamplingParams(max_tokens=5),
        )
        canceled.output_queue
        self.engine.add_request(canceled)
        peers = [self.enqueue(owner, budget=5) for owner in ("peer-a", "peer-b")]
        self.assertTrue(self.engine.step())
        original = tuple(canceled.block_table)
        self.assertEqual(len(original), 1)
        cache_ops = self.engine.scheduler.speculative_cache_ops

        def propose_many(owner, anchor, position, table, count, stop_ids):
            return tuple(range(anchor + 1, anchor + count + 1))

        @contextmanager
        def target_transaction(batch, tables):
            self.assertEqual(batch.input_offsets, (0, 4, 8, 12))
            self.assertEqual([len(table) for table in tables], [2, 1, 1])
            rows = tuple(self.verify_row(row, row.input_token + 1) for row in batch.rows)

            def commit(intents):
                self.assertEqual(intents[0].retain_length, 259)
                self.assertTrue(cache_ops._pending.submitted)
                self.assertEqual(self.engine.scheduler.cache_manager.get_total_usable_blocks(), 0)
                canceled.abort()

            yield SimpleNamespace(rows=rows, commit=commit)

        with (
            patch.object(self.transport, "propose_many", propose_many, create=True),
            patch.object(self.transport, "target_transaction", target_transaction, create=True),
        ):
            self.assertTrue(self.engine.step())

        self.assertEqual(tuple(canceled.block_table), original)
        self.assertIsNone(cache_ops._pending)
        self.assertEqual(self.outputs(canceled), [(71, False, None), (-1, True, self.reason.CANCELED)])
        for peer in peers:
            self.assertEqual(list(peer.generated_token_ids), [71, 72, 73, 74, 75])
            self.assertEqual(peer.finish_reason, self.reason.LENGTH)
        self.assertFalse(self.engine.model_runner.deepseek_v4_mtp_runner.requests)
        self.assert_usable_pages(4)

    def test_async_queued_timeout_owns_terminal_reason_after_prefill(self):
        self.engine._async_abort_owner = True
        async_engine = self.llm.AsyncLLMEngine.__new__(self.llm.AsyncLLMEngine)
        async_engine._abort_queue = SimpleNamespace(sync_q=queue.Queue())
        async_engine._stop_lock = threading.Lock()
        async_engine._healthy = True
        request = self.enqueue("async-queued")
        self.transport.after_prefill = lambda: async_engine.add_aborted_req(
            request, self.reason.TIMEOUT
        )

        self.assertTrue(self.engine.step())
        self.assertFalse(request.is_finished())
        self.assertEqual(self.outputs(request), [])
        self.assertTrue(request._abort_enqueued)
        async_engine._drain_abort_queue()
        self.assertFalse(self.engine.step())
        self.assertEqual(self.outputs(request), [(-1, True, self.reason.TIMEOUT)])
        self.assertEqual(
            self.transport.calls, [("prefill", "async-queued"), ("release", "async-queued")]
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
