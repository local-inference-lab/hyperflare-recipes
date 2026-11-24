# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native Mini serving, loop-history restore, and chunked attention.

Set FLARE_INKLING2_MINI_TEST_DRAFTS to 0 for plain decoding or 7 when the
server uses --no-mtp-adaptive-verification. Leave it unset for adaptive
selection, which may legitimately offer no drafts on an individual request.
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest


@pytest.fixture(scope="module")
def service():
    url = os.environ.get("FLARE_INKLING2_MINI_TEST_URL")
    if not url:
        pytest.skip("requires a native Mini server at FLARE_INKLING2_MINI_TEST_URL")
    url = url.rstrip("/")
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]
    assert model["id"] in ("inkling-2-mini", "TML-d24-varloop-research")
    assert "text" in model["flare"]["input_modalities"]

    def ask(prompt, **options):
        body = dict(model=model["id"], messages=[dict(role="user", content=prompt)],
                    seed=1234, max_tokens=768, return_token_ids=True)
        body.update(options)
        request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.load(response)
        assert result["choices"][0]["finish_reason"] == "stop", result
        assert result["usage"]["completion_tokens"] == len(result["choices"][0]["token_ids"])
        return result

    return ask, model["flare"], model["id"]


def answer(result):
    return result["choices"][0]["message"]["content"].strip()


def test_text_reasoning_and_request_isolation(service):
    ask, settings, _ = service
    result = ask("What is 17 times 23? Put only the integer in the final answer.")
    assert answer(result) == "391"
    assert result["choices"][0]["message"]["reasoning_content"].strip()
    count = max(4, settings["max_concurrency"])
    prompts = [f"Repeat exactly this code and nothing else: FLARE{i}128" for i in range(count)]
    with ThreadPoolExecutor(max_workers=count) as workers:
        results = list(workers.map(ask, prompts))
    assert [answer(result) for result in results] == [f"FLARE{i}128" for i in range(count)]


def test_streamed_reasoning_and_answer(service):
    _, _, model_id = service
    url = os.environ["FLARE_INKLING2_MINI_TEST_URL"].rstrip("/")
    body = dict(model=model_id, messages=[dict(role="user",
        content="What is 17 times 23? Put only the integer in the final answer.")],
        max_tokens=768, seed=1234, stream=True, stream_options=dict(include_usage=True))
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    events, done = [], False
    with urllib.request.urlopen(request, timeout=600) as response:
        for line in response:
            if line.startswith(b"data: "):
                payload = line.removeprefix(b"data: ").strip()
                if payload == b"[DONE]":
                    done = True
                    break
                events.append(json.loads(payload))
    assert done
    choices = [choice for event in events for choice in event["choices"]]
    reasoning = "".join(choice.get("delta", {}).get("reasoning_content") or "" for choice in choices)
    content = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
    assert reasoning.strip() and content.strip() == "391"
    assert choices[-1]["finish_reason"] == "stop"
    assert any(event.get("usage", {}).get("completion_tokens", 0) > 0 for event in events if event.get("usage"))


def test_chunked_prefix_restore_and_branching(service):
    ask, settings, model_id = service
    # Cross the 1024-token local-attention ring and the largest prefill graph.
    count = max(200, settings["prefill_chunk_size"] // 8)
    prefix = " ".join(f"Item {i} records code {i * 17 % 997}." for i in range(count))
    history = [dict(role="system", content="Use these records to answer the user's lookup. " + prefix)]
    # At effort 0.9, reasoning may check the entire record list before answering.
    # Grow its allowance with the input while preserving natural EOS and the answer checks.
    output_budget = max(2048, 2 * settings["prefill_chunk_size"])
    for index in (11, count - 1, 11):
        messages = history + [dict(role="user",
            content=f"What code does item {index} record? Reply with only the integer.")]
        first, second = (ask("", messages=messages, max_tokens=output_budget),
                         ask("", messages=messages, max_tokens=output_budget))
        assert first["usage"]["prompt_tokens"] > max(1024, settings["prefill_chunk_size"])
        assert second["usage"]["prompt_tokens_details"]["cached_tokens"] >= 1024
        assert answer(first) == answer(second) == str(index * 17 % 997)
        # Timing probes can change decode capacity within a request. Exact
        # state transfer is checked separately, without cross-capacity arithmetic.
        prefix_ids = first["prompt_token_ids"] + first["choices"][0]["token_ids"][:64]
        body = dict(model=model_id, prompt=prefix_ids, max_tokens=1, seed=1234,
                    logprobs=20, return_tokens_as_token_ids=True, ignore_eos=True)
        completions = []
        url = os.environ["FLARE_INKLING2_MINI_TEST_URL"].rstrip("/")
        for _ in range(2):
            request = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=600) as response:
                completions.append(json.load(response))
        assert completions[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == len(prefix_ids)
        distributions = [r["choices"][0]["logprobs"]["top_logprobs"][0] for r in completions]
        assert max(distributions[0], key=distributions[0].get) == max(distributions[1], key=distributions[1].get)


def test_tml_tool_call(service):
    _, _, model_id = service
    tools = [dict(type="function", function=dict(name="lookup_weather", description="Get a city's weather.",
        parameters=dict(type="object", properties=dict(city=dict(type="string")), required=["city"])))]
    # Tool calls terminate with tool_calls, so inspect this response separately.
    url = os.environ["FLARE_INKLING2_MINI_TEST_URL"].rstrip("/")
    body = dict(model=model_id, messages=[dict(role="user", content="Use lookup_weather for Paris.")],
                tools=tools, tool_choice="required", max_tokens=768, seed=1234)
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call, = choice["message"]["tool_calls"]
    assert call["function"]["name"] == "lookup_weather"
    assert json.loads(call["function"]["arguments"])["city"] == "Paris"


def test_images_and_cached_image_prefixes(service):
    from .test_minimax_m3 import image

    ask, settings, _ = service
    if "image" not in settings["input_modalities"]:
        pytest.skip("requires a vision-enabled Mini core")
    labels = ("FIRST 128", "SECOND 256", "THIRD 512", "FOURTH 1024")
    pictures = [image(color, label) for color, label in
                zip(("red", "blue", "green", "orange"), labels, strict=True)]
    content = [dict(type="text", text="filler " * (settings["prefill_chunk_size"] + 100)),
               *pictures, dict(type="text", text="Read all four image labels in order. Only output the labels.")]
    # A complete system message supplies an anchor independently of timing costs.
    # Prime snapshot and prefill estimates before testing optional prompt-end retention.
    calibration_notes = " ".join(
        f"Entry {i} calibration note." for i in range(settings["prefill_chunk_size"] // 4 + 100)
    )
    calibration = ask("", messages=[
        dict(role="system", content="Read these notes before answering. " + calibration_notes),
        dict(role="user", content="Repeat exactly the code CALIBRATED.")])
    assert answer(calibration) == "CALIBRATED"
    assert calibration["usage"]["prompt_tokens"] > settings["prefill_chunk_size"]
    for _ in range(3):
        result = ask(content, max_tokens=2048)
        text = answer(result)
        assert result["usage"]["prompt_tokens"] > settings["prefill_chunk_size"]
        assert all(label in text for label in labels), text
        assert [text.index(label) for label in labels] == sorted(text.index(label) for label in labels)
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] > settings["prefill_chunk_size"]
    prompts = [[picture, dict(type="text", text="Read the image label. Only output the label.")]
               for picture in pictures]
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda prompt: ask(prompt, max_tokens=2048), prompts))
    assert [answer(result) for result in results] == list(labels)


def test_draft_acceptance_matches_serving_mode(service):
    expected = os.environ.get("FLARE_INKLING2_MINI_TEST_DRAFTS")
    if expected is None:
        pytest.skip("set FLARE_INKLING2_MINI_TEST_DRAFTS to the server's draft depth")
    url = os.environ["FLARE_INKLING2_MINI_TEST_URL"].rstrip("/")

    def metrics():
        with urllib.request.urlopen(url + "/metrics", timeout=10) as response:
            lines = response.read().decode().splitlines()
        return {parts[0]: float(parts[1]) for line in lines
                if line and not line.startswith("#") and "{" not in line for parts in [line.split()]}

    before = metrics()
    result = service[0]("Write the integers from 1 through 40, separated by commas. Output only that list.")
    values = [int(value.strip()) for value in answer(result).rstrip(".").split(",")]
    assert values == list(range(1, 41))
    after = metrics()
    rounds = after["flare_draft_request_rounds_total"] - before["flare_draft_request_rounds_total"]
    accepted = after["flare_accepted_draft_tokens_total"] - before["flare_accepted_draft_tokens_total"]
    if int(expected):
        assert rounds > 0 and accepted > 0, (rounds, accepted)
    else:
        assert rounds == accepted == 0, (rounds, accepted)
