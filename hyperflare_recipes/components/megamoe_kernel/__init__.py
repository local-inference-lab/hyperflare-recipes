# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation binding for the pinned DeepGEMM SM103 MegaMoE library."""

from functools import cache
from pathlib import Path


@cache
def initialize(kernel_cache: Path):
    """Return the installed DeepGEMM module providing MegaMoE preparation kernels."""
    del kernel_cache
    import deep_gemm
    return deep_gemm
