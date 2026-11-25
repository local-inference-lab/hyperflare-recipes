# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Inkling2 draft feature mixing, rotary positions, and committed context selection."""

import torch


def mix_features(features, alpha, weight, eps):
    mixed = torch.zeros_like(features[:, 0], dtype=torch.float32)
    for index in range(features.shape[1]):
        mixed = mixed + features[:, index].float() * alpha[index].float()
    scale = torch.rsqrt(mixed.square().mean(-1, keepdim=True) + eps)
    return (mixed * scale * weight.float()).bfloat16()


def rotary(values, positions, inverse):
    angle = positions.clamp_min(0).float()[:, None] * inverse[None]
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    left, right = values.float().chunk(2, -1)
    return torch.cat((left * cosine - right * sine, right * cosine + left * sine), -1).to(values.dtype)


def _cast_kv(values, dtype):
    if values.dtype == dtype:
        return values
    if dtype == torch.float8_e4m3fn:
        return values.clamp(-448.0, 448.0).to(dtype)
    return values.to(dtype)


def stage_window(history, values, metadata, pages):
    """Stage the same committed suffix before every query in a noncausal block."""
    m = metadata
    rows, window = m.starts.numel(), history.shape[1]
    capacity = pages * 128
    offsets = torch.arange(capacity, device=values.device)[None]
    prefix = m.starts.clamp(max=window)[:, None]
    positions = m.starts[:, None] - prefix + offsets
    previous = history[m.state_rows[:, None], positions.remainder(window)]
    indices = (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, values.shape[0] - 1)
    current = _cast_kv(values, history.dtype)[indices]
    valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
    zero = torch.zeros((), dtype=history.dtype, device=values.device)
    return torch.where(valid[:, :, None, None],
        torch.where((offsets < prefix)[:, :, None, None], previous, current), zero).view(
            rows * pages, 128, *values.shape[1:])


def stage_paged_tail(history, values, metadata, pages):
    """Copy the committed partial page and proposal rows into private attention pages."""
    m = metadata
    rows, page_tokens = m.starts.numel(), history.shape[1]
    offsets = torch.arange(pages * page_tokens, device=values.device)[None]
    prefix = m.starts.remainder(page_tokens)[:, None]
    columns = (m.starts // page_tokens).clamp(max=m.page_table.shape[1] - 1)
    requests = torch.arange(rows, device=values.device)
    source_pages = m.page_table[requests, columns].long().clamp_min(0)
    previous = history[source_pages[:, None], offsets.remainder(page_tokens)]
    indices = (m.query_start_loc[:-1, None] + offsets - prefix).clamp(0, values.shape[0] - 1)
    current = _cast_kv(values, history.dtype)[indices]
    valid = (m.lengths[:, None] > 0) & (offsets < prefix + m.lengths[:, None])
    zero = torch.zeros((), dtype=history.dtype, device=values.device)
    return torch.where(valid[:, :, None, None],
        torch.where((offsets < prefix)[:, :, None, None], previous, current), zero).view(
            rows * pages, page_tokens, *values.shape[1:])


def accepted_rows(metadata, counts, *, window=None):
    offset = metadata.positions - metadata.starts[metadata.requests]
    count = counts[metadata.requests]
    live = (metadata.positions >= 0) & (offset < count) & (offset >= 0)
    if window is not None:
        live = live & (offset >= (count - window).clamp_min(0))
    return live


def context_slots(metadata, counts, *, window=None, page_tokens=128):
    live = accepted_rows(metadata, counts, window=window)
    if window is not None:
        slot = metadata.state_rows[metadata.requests] * window + metadata.positions.remainder(window)
    else:
        columns = metadata.positions.clamp_min(0) // page_tokens
        rows = torch.arange(metadata.positions.numel(), device=metadata.positions.device)
        page = metadata.token_page_table[rows, columns.clamp(max=metadata.token_page_table.shape[1] - 1)]
        live = live & (page >= 0) & (columns < metadata.token_page_table.shape[1])
        slot = page.long() * page_tokens + metadata.positions.remainder(page_tokens)
    return torch.where(live, slot, -1)
