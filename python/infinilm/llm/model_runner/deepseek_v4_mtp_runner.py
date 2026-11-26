"""Eager classic DeepSeek V4 NextN execution."""

import os
from dataclasses import replace

from ..sampling_params import validate_stop_strings, validate_stop_token_ids
from .deepseek_v4_mtp_sampling import RequestSampling, validate_request_sampling
from .deepseek_v4_mtp_verify import GreedyVerifyBatch, GreedyVerifyRequest


class DeepSeekV4MTPRunner:
    def __init__(
        self, target, transport, cache_token_capacity, *, num_draft_tokens=1,
        branch_draft_lengths=None, branch_execution="serial", allow_stochastic=False,
    ):
        self.target = target
        self.transport = transport
        self.num_draft_tokens = num_draft_tokens
        self.branch_draft_lengths = tuple(
            k for k in (branch_draft_lengths or ()) if k <= num_draft_tokens
        )
        self.branch_execution = branch_execution
        self.cache_token_capacity = cache_token_capacity
        self.allow_stochastic = allow_stochastic
        self.max_context = min(
            target.hf_config["max_position_embeddings"], cache_token_capacity
        )
        self.requests = {}
        self.failed = False

    @property
    def request(self):
        return next(iter(self.requests.values())) if len(self.requests) == 1 else None

    def validate_request(self, request):
        params = request.sampling_params
        validate_request_sampling(params)
        if params.do_sample and not self.allow_stochastic:
            raise ValueError("classic stochastic MTP requires eager execution")
        if params.do_sample and self.branch_draft_lengths:
            raise ValueError("classic stochastic MTP requires a single draft chain")
        validate_stop_strings(params.stop)
        validate_stop_token_ids(params.stop_token_ids)
        if (
            request.has_multimodal_inputs
            or type(params.max_tokens) is not int
            or params.max_tokens < 0
            or request.get_prompt_length() < 1
        ):
            raise ValueError(
                "classic DeepSeek V4 MTP requires a nonempty text prompt, "
                "and a non-negative output limit"
            )
        if request.get_prompt_length() + params.max_tokens > self.max_context:
            raise ValueError("classic DeepSeek V4 MTP request exceeds available context")
        if self.branch_execution in ("packed", "packed-stepwise") and params.max_tokens >= 4:
            count = len(set(min(k, params.max_tokens - 2) for k in self.branch_draft_lengths))
            if count > 1:
                page_size = self.transport.page_size
                last_input = request.get_prompt_length() + params.max_tokens - 1
                pages = (last_input + page_size - 1) // page_size
                if pages * (count + 1) > self.cache_token_capacity // page_size:
                    raise ValueError("classic NextN packed branches exceed available private cache pages")

    def retire(self):
        if self.failed:
            raise RuntimeError("classic DeepSeek V4 MTP must be closed after a failed step")
        try:
            for request in tuple(self.requests.values()):
                if not request.is_finished():
                    continue
                self.transport.release(request.request_id)
                del self.requests[request.request_id]
        except BaseException:
            self.failed = True
            raise

    def forward(self, scheduler_output, model_input):
        if self.failed:
            raise RuntimeError("classic DeepSeek V4 MTP must be closed after a failed step")
        requests = scheduler_output.scheduled_requests
        if not requests:
            raise RuntimeError("classic DeepSeek V4 MTP requires a scheduled request")
        # Native output IDs remain greedy. Stochastic choices use the raw logits
        # and consume only their requesting owner's Python RNG.
        model_input = {**model_input, "temperature": 1.0, "top_p": 1.0, "top_k": 1}
        try:
            for request in requests:
                self.validate_request(request)
            if self.branch_draft_lengths and len(requests) != 1:
                raise RuntimeError("classic NextN branches require one scheduled owner")
            if scheduler_output.is_prefill:
                return self._prefill(requests, model_input)
            for request in requests:
                if self.requests.get(request.request_id) is not request or not request.generated_token_ids:
                    raise RuntimeError("classic DeepSeek V4 MTP decode lacks its prefill owner")

            remaining = [
                request.sampling_params.max_tokens - request.get_num_generated_tokens()
                for request in requests
            ]
            if remaining == [1] and not requests[0].sampling_params.do_sample:
                # There is no room to publish both a proposal and its bonus.
                return self.target.forward(
                    **model_input, sampling_rows=[(1.0, 1.0, 1)], random_values=[0.0]
                ).to_numpy().tolist()
            depth = max(self.branch_draft_lengths, default=self.num_draft_tokens)
            candidates = [
                {"req": req, "base_len": req.get_total_length(),
                 "draft_count": min(depth, limit - 1), "remaining": limit}
                for req, limit in zip(requests, remaining) if limit > 1
            ]
            outputs = self._verify(scheduler_output.speculative_cache_ops, candidates) if candidates else {}
            for request, limit in zip(requests, remaining):
                if limit == 1:
                    outputs[request.request_id] = self.transport.target_tail(
                        request.request_id, request.generated_token_ids[-1],
                        request.get_total_length() - 1, request.block_table,
                        **({"sampling": RequestSampling(request)} if request.sampling_params.do_sample else {}),
                    )
            return [outputs[request.request_id] for request in requests]
        except BaseException:
            self.failed = True
            raise

    def _prefill(self, requests, model_input):
        self.retire()
        owners = tuple(request.request_id for request in requests)
        if any(
            request.request_id in self.requests or request.num_local_cached_tokens != 0
            for request in requests
        ):
            raise RuntimeError("classic DeepSeek V4 MTP requires a fresh full prefill")
        if len(owners) == 1 and requests[0].sampling_params.do_sample:
            anchors = [self.transport.prefill(owners[0], model_input, sampling=RequestSampling(requests[0]))]
        elif len(owners) == 1:
            anchors = [self.transport.prefill(owners[0], model_input)]
        else:
            sampling = tuple(RequestSampling(req) if req.sampling_params.do_sample else None
                             for req in requests)
            if any(state is not None for state in sampling):
                anchors = list(self.transport.prefill_many(owners, model_input, sampling=sampling))
            else:
                anchors = list(self.transport.prefill_many(owners, model_input))
        if len(anchors) != len(owners) or any(type(token) is not int or token < 0 for token in anchors):
            raise RuntimeError("classic DeepSeek V4 MTP prefill returned invalid tokens")
        self.requests.update(zip(owners, requests))
        return anchors

    def _proposal_request(self, candidate):
        request = candidate["req"]
        start = candidate["base_len"] - 1
        owner, anchor = request.request_id, request.generated_token_ids[-1]
        eos = request.eos_token_ids or self.target.eos_token_id or ()
        if isinstance(eos, int):
            eos = (eos,)
        stop_ids = frozenset(request.sampling_params.stop_token_ids)
        if not request.sampling_params.ignore_eos:
            stop_ids = stop_ids.union(eos)
        return GreedyVerifyRequest(owner, start, candidate["remaining"] + 1, anchor, (), stop_ids)

    def _propose(self, candidate, table):
        request = self._proposal_request(candidate)
        owner, anchor, start = request.owner, request.anchor, request.start
        count, stop_ids = candidate["draft_count"], request.eos_token_ids
        source = candidate["req"]
        if source.sampling_params.do_sample:
            proposals = self.transport.propose_many(
                owner, anchor, start, table, count, stop_ids, sampling=RequestSampling(source)
            )
        elif count == 1:
            proposals = (self.transport.propose(owner, anchor, start, table),)
        else:
            proposals = self.transport.propose_many(
                owner, anchor, start, table, count, stop_ids
            )
        return replace(request, draft_tokens=proposals)

    def _verify(self, cache_ops, candidates):
        with cache_ops.verify_batch(candidates) as reservation:
            scratch_tables = ()
            if self.branch_execution in ("packed", "packed-stepwise"):
                (parent,) = candidates
                planned_depths = dict.fromkeys(
                    min(k, parent["draft_count"]) for k in self.branch_draft_lengths
                )
                if len(planned_depths) > 1:
                    scratch_tables = reservation.reserve_branch_tables(
                        tuple(parent["base_len"] + k for k in planned_depths)
                    )
            # The proposal also writes native cache state. Keep provisional pages
            # reserved until the transport has resolved the full transaction.
            reservation.mark_submitted()
            tables = tuple(tuple(candidate["block_table"]) for candidate in candidates)
            keep_final_graph = self.num_draft_tokens == 3 and getattr(
                self.transport, "graph_batched_draft_proposal", False
            )
            batched = (
                (getattr(self.transport, "batched_draft", False) or keep_final_graph)
                and not self.branch_draft_lengths
                and (self.num_draft_tokens == 1 or len(candidates) > 1)
                and all(not candidate["req"].sampling_params.do_sample for candidate in candidates)
            )
            if batched:
                proposals = [self._proposal_request(candidate) for candidate in candidates]
                counts = tuple(candidate["draft_count"] for candidate in candidates)
                proposal_options = {"keep_final_graph": True} if keep_final_graph else {}
                if keep_final_graph and os.getenv("INFINILM_MTP_GRAPH_C3_BATCH_FINAL_DRAFT") == "1":
                    # Recheck the live lease at the last hop, after earlier Draft writes.
                    proposal_options["permit_final_batch"] = lambda: (
                        self.transport.graph_packed_direct_enabled() and reservation.permits_packed_direct_target(tables)
                    )
                chains = self.transport.propose_batched(
                    proposals, tables, counts, **proposal_options,
                )
                requests = [replace(request, draft_tokens=chain)
                            for request, chain in zip(proposals, chains, strict=True)]
            else:
                requests = [
                    self._propose(candidate, table) for candidate, table in zip(candidates, tables)
                ]
            if self.branch_draft_lengths:
                (request,) = requests
                depths = dict.fromkeys(
                    min(k, len(request.draft_tokens)) for k in self.branch_draft_lengths
                )
                branches = tuple(
                    GreedyVerifyBatch.pack(
                        (replace(request, draft_tokens=request.draft_tokens[:k]),),
                        self.max_context,
                    )
                    for k in depths
                )
                if scratch_tables and len(branches) > 1:
                    verify = (
                        self.transport.stepwise_branch_transaction if self.branch_execution == "packed-stepwise"
                        else self.transport.packed_branch_transaction
                    )
                    branch_verify = verify(branches, tables[0], scratch_tables[:len(branches)], self.max_context)
                else:
                    branch_verify = self.transport.serial_branch_transaction(branches, tables[0])
                with branch_verify as (transaction, winner):
                    intents = (winner,)
                    transaction.commit(winner)
            else:
                batch = GreedyVerifyBatch.pack(requests, self.max_context)
                graph_direct_selector = {
                    1: "graph_single_direct_enabled", 3: "graph_packed_direct_enabled",
                }.get(len(candidates), "")
                packed_direct = (
                    self.num_draft_tokens == 3
                    and getattr(self.transport, graph_direct_selector, lambda: False)()
                    and all(not item["req"].sampling_params.do_sample for item in candidates)
                    and reservation.permits_packed_direct_target(tables)
                )
                with self.transport.target_transaction(
                    batch, tables[0] if len(candidates) == 1 else tables,
                    **({"batch_draft_seed": True} if batched and self.num_draft_tokens == 1 else {}),
                    **({"graph_packed_direct_commit": True} if packed_direct else {}),
                ) as transaction:
                    intents = (
                        transaction.resolve_stochastic()
                        if any(row["req"].sampling_params.do_sample for row in candidates)
                        else batch.resolve(
                            transaction.rows, omit_terminal=getattr(transaction, "omit_terminal", False),
                        )
                    )
                    transaction.commit(intents[0] if len(candidates) == 1 else intents)
            reservation.mark_completed()
            reservation.commit([intent.retain_length for intent in intents])

        # The previous step has already published the anchor. The regular
        # scheduler publishes newly accepted tokens and the one uncomputed tail.
        outputs = {}
        for intent in intents:
            tokens = list(intent.emitted_tokens[1:])
            if intent.next_anchor is not None:
                tokens.append(intent.next_anchor)
            outputs[intent.owner] = tokens
        return outputs

    def close(self):
        self.transport.close()
        self.requests.clear()
