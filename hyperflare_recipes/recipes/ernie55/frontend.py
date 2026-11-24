# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Annotated Jinja chat template matching ERNIE 5.5 Flash Preview's `encoding_ernie.py`."""

import hashlib
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


def chat_template_asset():
    """Return the annotated ERNIE 5.5 Jinja template asset for Helios."""
    contents = (ASSETS / "chat_template.jinja").read_bytes()
    return {
        "text": contents.decode("utf-8"),
        "sha256": hashlib.sha256(contents).hexdigest(),
    }
