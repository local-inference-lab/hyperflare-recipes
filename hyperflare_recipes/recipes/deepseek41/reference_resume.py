# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sample a restored reference-model prefix through shared request handling."""

import torch

from hyperflare_recipes.components.resume import PreparedResumeState
from ignition.components.sampling import PreparedSampling


class ReferenceResume:
    def __init__(self, model, requests, pool, inputs, completion, frame, sequence, *, rows, session,
                 speculation_enabled=True):
        if requests.vocab_size != model.cfg.vocab_size:
            raise ValueError("reference resume vocabulary differs from request storage")
        if type(speculation_enabled) is not bool:
            raise TypeError("speculation_enabled must be a boolean")
        self.model = model
        self.state = PreparedResumeState(requests, pool.reserved_region("last_hidden"), inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, world_size=1,
            state_rows=inputs.request_state_rows).functional()
        example = torch.empty((rows, model.cfg.vocab_size), dtype=torch.float32,
                              device=requests.generation.device)
        self.sampling = PreparedSampling(example, requests, inputs.row_slot_ids[:rows],
            inputs.row_generations[:rows], inputs.live_rows, completion, frame, sequence,
            draft_context_lag=0 if speculation_enabled else None)
        self.sampling.prepare()
        self.sampling.functional(session=session)

    def __call__(self):
        from deepinfer.model_executor.layers.sampler_utils import get_logits

        validation, present = self.state.validate()
        hidden = self.state.gather(validation, present=present)
        logits = get_logits(hidden, self.model.lm_head.weight, self.model.cfg.vocab_size)
        self.sampling.run(logits)
