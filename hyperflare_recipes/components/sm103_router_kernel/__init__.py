# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation build and shared-memory setup for the SM103 BF16 MoE router."""

from functools import cache
import hashlib
from pathlib import Path


def initialize(kernel_cache: Path):
    """Build the SM103 router extension with Ignition's Torch ABI before capture."""
    import torch
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("the SM103 BF16 router requires an SM103 device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("the SM103 BF16 router must be prepared before CUDA graph capture")
    return _build(Path(kernel_cache).resolve())


@cache
def _build(kernel_cache: Path):
    import torch
    from torch.utils.cpp_extension import load

    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.cu")):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    digest.update((torch.__version__ + str(torch.version.cuda)).encode())
    build = Path(kernel_cache) / "sm103-router" / digest.hexdigest()
    build.mkdir(parents=True, exist_ok=True)
    return load(
        name="flare_sm103_router",
        sources=[str(root / "csrc/sm103_router.cu")],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "-lineinfo", "--use_fast_math",
            "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        build_directory=str(build),
        verbose=False,
    )


def prepare(kernel_cache: Path):
    """Configure dynamic shared-memory limits on the active SM103 device before capture."""
    module = initialize(kernel_cache)
    module.sm103_router_prepare()
    return module
