# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests verifying Helios history copy-draft buffer declarations on Inkling-2 and DFlash2."""

from __future__ import annotations

import torch


def test_copy_draft_buffer_contract():
    from ignition.preparation import contract as c

    inputs = c.text_inputs("speculative")
    for field in ("copy_token_ids", "copy_token_count", "copy_expected_consumed", "copy_generation"):
        assert field in inputs


def test_dspark_and_dflash2_expose_buffers_method():
    from hyperflare_recipes.components.dflash2_preparation import DFlash2Preparation
    from hyperflare_recipes.recipes.inkling2.dspark import SpeculativePreparation

    assert hasattr(DFlash2Preparation, "buffers")
    assert hasattr(SpeculativePreparation, "buffers")
