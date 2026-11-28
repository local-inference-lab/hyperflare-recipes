# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native DeepSeek TP1 text replay with mapped Grace experts and MTP.

Set FLARE_RUN_CUDA_TESTS=1 and FLARE_DEEPSEEK41_SM103_TEST_URL to an idle
release Flare service prepared with explicit SM103 placement: concurrency one,
256-token prefill and seven adaptive drafts, or concurrency sixteen, 8192-token
prefill and three adaptive drafts. This checks serving integration, not numerical
agreement with a reference model or other graph capacities.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import threading
import time
import urllib.request

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_DEEPSEEK41_SM103_TEST_URL"),
    reason="requires an idle native DeepSeek SM103 TP1 MTP service",
)


def test_native_text_replay_and_prefix_reuse():
    url = os.environ["FLARE_DEEPSEEK41_SM103_TEST_URL"].rstrip("/")

    def get(path):
        with urllib.request.urlopen(url + path, timeout=30) as response:
            return response.read().decode()

    def metrics():
        return {fields[0]: float(fields[1]) for line in get("/metrics").splitlines()
                if not line.startswith("#") and len(fields := line.split()) == 2}

    model = json.loads(get("/v1/models"))["data"][0]
    assert "deepseek" in model["id"].lower() and "4.1" in model["id"]
    concurrency = model["flare"]["max_concurrency"]
    chunk = model["flare"]["prefill_chunk_size"]
    assert (concurrency, chunk) in ((1, 256), (16, 8192)), model

    def ask(prompt, barrier=None):
        # Sampling follows the core's declared defaults; only seed is overridden.
        body = dict(model=model["id"], messages=[dict(role="user", content=prompt)],
                    seed=1234, max_tokens=1536 if concurrency == 16 else 256,
                    enable_thinking=False, return_token_ids=True)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        if barrier is not None:
            barrier.wait(timeout=30)
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        choice = result["choices"][0]
        assert choice["finish_reason"] == "stop", result
        assert len(choice["token_ids"]) == result["usage"]["completion_tokens"], result
        return result, choice["message"]["content"].strip(), choice["token_ids"]

    gauges = ("flare_active_requests", "flare_pending_requests", "flare_retiring_request_slots",
              "flare_http_inflight_requests", "flare_frontend_active_workers")
    before = metrics()
    assert all(before[key] == 0 for key in gauges), before
    prompt = "What is 17 times 23? Reply with only the integer."
    _, first, ids = ask(prompt)
    _, repeated, repeated_ids = ask(prompt)
    assert first == repeated == "391"
    assert ids == repeated_ids

    items = 1536 if concurrency == 16 else 96
    context = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(items))
    prompt = context + "\nWhat code does item 37 record? Reply with only the integer."
    result, first, ids = ask(prompt)
    assert result["usage"]["prompt_tokens"] > chunk
    result, repeated, repeated_ids = ask(prompt)
    assert first == repeated == "629"
    assert ids == repeated_ids
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > chunk
    _, answer, _ = ask(context + "\nWhat code does item 71 record? Reply with only the integer.")
    assert answer == "210"
    _, answer, _ = ask("Write the integers from 1 through 40 inclusive, separated by commas, with no other text.")
    assert [int(value) for value in re.findall(r"\d+", answer)] == list(range(1, 41)), answer

    if concurrency == 16:
        barrier = threading.Barrier(concurrency)

        def concurrent_request(index):
            start = index * 1000 + 1
            _, answer, _ = ask(
                f"Write every integer from {start} through {start + 239} inclusive, "
                "separated by commas, with no other text.", barrier,
            )
            assert [int(value) for value in re.findall(r"\d+", answer)] == list(range(start, start + 240)), answer

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(concurrent_request, range(concurrency)))

    deadline = time.monotonic() + 30
    while True:
        after = metrics()
        if all(after[key] == 0 for key in gauges):
            break
        assert time.monotonic() < deadline, {key: after[key] for key in gauges}
        time.sleep(.01)
    expected_requests = 22 if concurrency == 16 else 6
    assert after["flare_requests_completed_total"] - before["flare_requests_completed_total"] == expected_requests
    assert after["flare_requests_failed_total"] == before["flare_requests_failed_total"]
    assert after["flare_cache_reused_tokens_total"] > before["flare_cache_reused_tokens_total"]
    assert sum(value - before.get(key, 0) for key, value in after.items()
               if key.startswith('flare_speculative_offered_drafts_total{batch="1",')) > 0
    if concurrency == 16:
        assert sum(value - before.get(key, 0) for key, value in after.items()
                   if key.startswith('flare_speculative_offered_drafts_total{batch="16",')) > 0
