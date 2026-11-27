# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind SDXL's canonical attention processor before forward compilation."""

import torch
from diffusers.models.attention_processor import Attention, AttnProcessor2_0


class PreparedAttention(torch.nn.Module):
    """Retain canonical attention arithmetic without runtime signature inspection."""

    def __init__(self, attention):
        super().__init__()
        if type(attention) is not Attention or type(attention.processor) is not AttnProcessor2_0:
            raise ValueError("SDXL preparation requires canonical Attention with AttnProcessor2_0")
        if attention._forward_hooks or attention._forward_pre_hooks:
            raise ValueError("SDXL attention preparation requires modules without forward hooks")
        self.attention = attention
        self.train(attention.training)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                **cross_attention_kwargs):
        if cross_attention_kwargs:
            raise ValueError("SDXL prepared attention does not accept extra processor arguments")
        return self.attention.processor(
            self.attention, hidden_states, encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask, temb=temb)


def bind_attention_processors(model):
    """Resolve attention dispatch while retaining the original modules and weights."""
    replacements = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, Attention):
                replacements.append((parent, name, PreparedAttention(child)))
    for parent, name, prepared in replacements:
        setattr(parent, name, prepared)
    return len(replacements)
