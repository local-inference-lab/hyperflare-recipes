# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare native image inputs with DeepInfer's CPU and GPU preprocessing."""

import io
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer

from ignition.preparation.frontend import prepare_frontend
from hyperflare_recipes.recipes.deepseek41.contract import INPUT_VOCABULARY, VISION, VISION_FRONTEND

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned GPU and DeepInfer")


@pytest.fixture(scope="module")
def native_image_processor():
    pytest.importorskip("deepinfer")
    path = os.environ.get("FLARE_DEEPSEEK41_CSF_SNAPSHOT")
    if not path:
        pytest.skip("requires the DeepSeek V4.1 checkpoint frontend")
    target = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "helios/target"))
    subprocess.run(["cargo", "build", "--quiet", "--manifest-path", str(ROOT / "helios/Cargo.toml"),
                    "--target-dir", str(target), "--features", "server,test-tools",
                    "--bin", "test_deepseek4_images"], check=True)
    return Path(path), target / "debug/test_deepseek4_images"


@pytest.mark.parametrize("sizes", [[(1, 1)], [(701, 503)], [(17, 3000)], [(503, 701), (149, 31)]])
@torch.no_grad()
def test_native_images_match_deepinfer(native_image_processor, tmp_path, sizes):
    from deepinfer.hugging_face import build_mm_processor, load_processor_config
    from deepinfer.model_executor.layers.tilelang_ops.image_embed_ops import rgb_pad_to_patches
    from deepinfer.multimodal_tokens import TokenType
    from deepinfer.ve.processor.common import ImageInput

    snapshot, binary = native_image_processor
    config = load_processor_config(snapshot)
    processor = build_mm_processor(config).image_processor
    vocab = json.loads((snapshot / "config.json").read_text())["text_config"]["vocab_size"]
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    prompt = "Describe " + "<｜deepseek_image｜> then " * len(sizes)
    inputs, references, patches, grids, counts, positions, cu = [], [], [], [], [], [], [0]
    for index, (width, height) in enumerate(sizes):
        rgb = np.random.default_rng(index + 51).integers(0, 256, (height, width, 3), dtype=np.uint8)
        source = tmp_path / f"image{index}.rgb"
        source.write_bytes(rgb.tobytes())
        inputs.append(dict(width=width, height=height, pixels=str(source)))
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, format="PNG")
        reference = processor.process(ImageInput(width, height, encoded.getvalue()))
        references.append(reference)
        resized = reference["images"][0].load().pixels
        crop = reference["images_spatial_crop"]
        w, h = crop[0].tolist()
        count = ((h + 2) // 3) * ((w + 2) // 3) * 9
        expected = rgb_pad_to_patches(resized.flatten().cuda(),
            torch.tensor([0, resized.numel()], device="cuda", dtype=torch.int64),
            torch.tensor([[resized.shape[1], resized.shape[0]]], device="cuda", dtype=torch.int32),
            crop.cuda(), torch.tensor([0, h * w], device="cuda", dtype=torch.int32),
            h * w, config["patch_size"], processor.image_transform.mean[0],
            processor.image_transform.std[0], int(processor.image_transform.mean[0] * 255))
        patches.append(torch.nn.functional.pad(expected.reshape(h * w, -1).cpu(), (0, 0, 0, count - h * w)))
        grids.append([1, h, w])
        counts.append(count)
        positions.extend(reference["images_global_attn_pos_id"].tolist() + [[0, 0]] * (count - h * w))
        cu.append(cu[-1] + h * w)
    tokens, indices, spans, image, offset = [], [], [], 0, 0
    offsets = {TokenType.IMAGE_START: 0, TokenType.IMAGE: 1, TokenType.IMAGE_NEW_LINE: 2, TokenType.IMAGE_END: 3}
    for token in tokenizer.encode(prompt, add_special_tokens=False).ids:
        if token != config["image_token_id"]:
            tokens.append(token)
            indices.append(-1)
            continue
        begin = len(tokens)
        for kind in references[image]["mm_token_type"].tolist():
            tokens.append(vocab + offsets[TokenType(kind)])
            indices.append(offset if kind == TokenType.IMAGE else -1)
            offset += int(kind == TokenType.IMAGE)
        spans.append([begin, len(tokens)])
        image += 1
    assert image == len(sizes)
    contract = json.loads((ROOT / "helios/tests/support/contracts/deepseek41.json").read_text())
    contract.update(input_vocabulary=INPUT_VOCABULARY, frontend=VISION_FRONTEND, vision={**VISION, "feature_slots": 4})
    settings = prepare_frontend(snapshot, eos_token_ids=[1], image_profile="deepseek_row_major")["image"]
    case = tmp_path / "case.json"
    case.write_text(json.dumps(dict(images=inputs, prompt=prompt, contract=contract, image_settings=settings)))
    subprocess.run([str(binary), str(snapshot), str(ROOT / "ignition/recipes/deepseek41/assets"),
                    str(case), str(tmp_path)], check=True)
    actual = json.loads((tmp_path / "metadata.json").read_text())
    assert actual == dict(tokens=tokens, indices=indices, spans=spans, grids=grids, patches=counts,
                          positions=positions, cu_seqlens=cu)
    pixels = torch.from_numpy(np.fromfile(tmp_path / "pixels.f32", dtype=np.float32)).to(torch.bfloat16)
    torch.testing.assert_close(pixels, torch.cat(patches).flatten(), rtol=0, atol=0)
