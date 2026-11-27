# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope="session")
def megamoe_core_storage():
    """Share the process-wide native allocator across MegaMoE component checks."""
    import torch
    from ignition.preparation.core_storage import CoreStorage

    storage = CoreStorage()
    with storage:
        torch.cuda.set_device(0)
        yield storage
    storage.close()
