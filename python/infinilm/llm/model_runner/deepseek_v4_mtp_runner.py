"""Eager classic DeepSeek V4 NextN execution."""

from .deepseek_v4_mtp_verify import GreedyVerifyBatch, GreedyVerifyRequest


class DeepSeekV4MTPRunner:
    def __init__(self, target, transport, cache_token_capacity, *, num_draft_tokens=1):
        self.target = target
        self.transport = transport
        self.num_draft_tokens = num_draft_tokens
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
        if (
            params.top_k != 1
            or params.stop
            or params.stop_token_ids
            or request.has_multimodal_inputs
            or type(params.max_tokens) is not int
            or params.max_tokens < 1
            or request.get_prompt_length() < 1
        ):
            raise ValueError(
                "classic DeepSeek V4 MTP requires a nonempty text prompt, "
                "a positive output limit, top_k=1 and no custom stop rules"
            )
        if request.get_prompt_length() + params.max_tokens > self.max_context:
            raise ValueError("classic DeepSeek V4 MTP request exceeds available context")

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
        # Classic greedy sampling never consumes RNG state.
        model_input = {**model_input, "temperature": 1.0, "top_p": 1.0, "top_k": 1}
        try:
            for request in requests:
                self.validate_request(request)
            if scheduler_output.is_prefill:
                return self._prefill(requests, model_input)
            for request in requests:
                if self.requests.get(request.request_id) is not request or not request.generated_token_ids:
                    raise RuntimeError("classic DeepSeek V4 MTP decode lacks its prefill owner")

            remaining = [
                request.sampling_params.max_tokens - request.get_num_generated_tokens()
                for request in requests
            ]
            if remaining == [1]:
                # There is no room to publish both a proposal and its bonus.
                return self.target.forward(
                    **model_input, sampling_rows=[(1.0, 1.0, 1)], random_values=[0.0]
                ).to_numpy().tolist()
            candidates = [
                {"req": req, "base_len": req.get_total_length(),
                 "draft_count": min(self.num_draft_tokens, limit - 1), "remaining": limit}
                for req, limit in zip(requests, remaining) if limit > 1
            ]
            outputs = self._verify(scheduler_output.speculative_cache_ops, candidates) if candidates else {}
            for request, limit in zip(requests, remaining):
                if limit == 1:
                    outputs[request.request_id] = self.transport.target_tail(
                        request.request_id, request.generated_token_ids[-1],
                        request.get_total_length() - 1, request.block_table,
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
        if len(owners) == 1:
            anchors = [self.transport.prefill(owners[0], model_input)]
        else:
            anchors = list(self.transport.prefill_many(owners, model_input))
        if len(anchors) != len(owners) or any(type(token) is not int or token < 0 for token in anchors):
            raise RuntimeError("classic DeepSeek V4 MTP prefill returned invalid tokens")
        self.requests.update(zip(owners, requests))
        return anchors

    def _propose(self, candidate, table):
        request = candidate["req"]
        start = candidate["base_len"] - 1
        count = candidate["draft_count"]
        owner, anchor = request.request_id, request.generated_token_ids[-1]
        eos = request.eos_token_ids or self.target.eos_token_id or ()
        if isinstance(eos, int):
            eos = (eos,)
        eos = frozenset() if request.sampling_params.ignore_eos else frozenset(eos)
        if count == 1:
            proposals = (self.transport.propose(owner, anchor, start, table),)
        else:
            proposals = self.transport.propose_many(owner, anchor, start, table, count, eos)
        return GreedyVerifyRequest(owner, start, candidate["remaining"] + 1, anchor, proposals, eos)

    def _verify(self, cache_ops, candidates):
        with cache_ops.verify_batch(candidates) as reservation:
            # The proposal also writes native cache state. Keep provisional pages
            # reserved until the transport has resolved the full transaction.
            reservation.mark_submitted()
            tables = tuple(tuple(candidate["block_table"]) for candidate in candidates)
            batch = GreedyVerifyBatch.pack(
                [self._propose(candidate, table) for candidate, table in zip(candidates, tables)],
                self.max_context,
            )
            with self.transport.target_transaction(
                batch, tables[0] if len(candidates) == 1 else tables,
            ) as transaction:
                intents = batch.resolve(transaction.rows)
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
