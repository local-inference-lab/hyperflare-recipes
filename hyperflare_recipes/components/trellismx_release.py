# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TrellisMX overlay checkpoint layout and manifest validation.

A TrellisMX release overlays per-layer, per-rank routed-expert sidecars on a
carrier directory. The release root holds the manifest, sidecars, and encoder
transform while the safetensors index, tokenizer, chat template, and generation
configuration live under ``carrier/``.

``TRELLISMX_OVERLAY`` plugs this layout into core Ignition's generic
``CheckpointOverlay`` contract without importing PyTorch, Transformers, or
Heliopause.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from ignition.declaration import CheckpointOverlay
from ignition.preparation.checkpoint import (
    checkpoint_assets_root,
    overlay_files_present,
    overlay_stored_bytes,
    read_overlay_manifest,
)

MANIFEST_NAME = "trellismx-manifest.json"
MANIFEST_SCHEMA = "trellismx.hf-overlay-release.v1"
CARRIER_DIRECTORY = "carrier"
SIDECAR_DIRECTORY = "sidecars"
# The encoder transform the sidecars pin, which preparation reads beside them.
TRANSFORM_PATH = "design/transform.json"


def sidecar_path(layer: int, rank: int, world_size: int) -> str:
    """Return the release-relative path of one layer and tensor-parallel rank's sidecar."""
    return f"{SIDECAR_DIRECTORY}/p8-layer-{layer:03d}-tp{world_size}-rank-{rank}.safetensors"


def _allocation(manifest: Mapping[str, object]) -> dict[int, int]:
    """Return each allocated layer index and the tensor-parallel size it is sliced for."""
    allocation = manifest.get("allocation")
    if not isinstance(allocation, dict) or not allocation:
        raise ValueError("TrellisMX manifest allocation must be a nonempty object of layer indices")
    layers = {}
    for key, world_size in allocation.items():
        try:
            layer = int(key)
        except (TypeError, ValueError):
            layer = None
        if layer is None or key != str(layer) or layer < 0 or type(world_size) is not int or world_size <= 0:
            raise ValueError(f"TrellisMX manifest allocation entry {key!r}: {world_size!r} is invalid")
        layers[layer] = world_size
    return layers


def validate_trellismx_manifest(manifest: Mapping[str, object]) -> None:
    """Validate that ``manifest['files']`` matches the canonical per-layer, per-rank sidecar inventory.

    The canonical inventory contains exactly one rank-named sidecar for every
    layer and rank declared in ``manifest['allocation']``. Omitted, duplicated,
    respelled, or extra entries raise ``ValueError`` so an incomplete or altered
    download cannot appear complete.
    """
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("TrellisMX manifest lists no files")
    entries = []
    for entry in files:
        path = entry.get("path") if isinstance(entry, dict) else None
        size = entry.get("bytes") if isinstance(entry, dict) else None
        if (not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts
                or type(size) is not int or size <= 0):
            raise ValueError(f"TrellisMX manifest file entry {entry!r} is invalid")
        entries.append((path, size))
    inventory = {sidecar_path(layer, rank, world_size)
                 for layer, world_size in _allocation(manifest).items() for rank in range(world_size)}
    listed = [path for path, _ in entries]
    if len(listed) != len(set(listed)) or set(listed) != inventory:
        missing, unexpected = sorted(inventory - set(listed)), sorted(set(listed) - inventory)
        repeated = sorted({path for path in listed if listed.count(path) > 1})
        raise ValueError("TrellisMX manifest files must be one sidecar per allocated layer and rank:"
                         f" missing {missing}, unexpected {unexpected}, repeated {repeated}")


TRELLISMX_OVERLAY = CheckpointOverlay(
    manifest=MANIFEST_NAME,
    schema=MANIFEST_SCHEMA,
    assets_subdir=CARRIER_DIRECTORY,
    required_files=(TRANSFORM_PATH,),
    validate_manifest=validate_trellismx_manifest,
)


def trellismx_assets_root(root: str | Path) -> Path:
    """Return the directory holding ``root``'s safetensors index and frontend assets.

    Returns ``root / carrier`` when ``root`` declares the TrellisMX overlay schema
    and carries a carrier index or single-file checkpoint. Raises ``CheckpointError``
    when the overlay manifest is unreadable or declares this schema without carrier
    safetensors assets, and returns ``root`` for plain checkpoints and other schemas.
    """
    return checkpoint_assets_root(root, TRELLISMX_OVERLAY)


def overlay_schema(root: str | Path) -> str | None:
    """Return the TrellisMX overlay schema when ``root`` declares it, else None."""
    return MANIFEST_SCHEMA if read_overlay_manifest(root, TRELLISMX_OVERLAY) is not None else None


def trellismx_files_present(root: str | Path) -> bool:
    """Report whether every file a TrellisMX release needs beside its carrier exists at its declared size.

    Returns True for a root that is not a TrellisMX release. A manifest whose file
    list does not match the canonical inventory raises ``ValueError``.
    """
    if read_overlay_manifest(root, TRELLISMX_OVERLAY) is None:
        return True
    return overlay_files_present(root, TRELLISMX_OVERLAY)


def trellismx_stored_bytes(root: str | Path) -> int | None:
    """Return stored payload bytes of a TrellisMX release, or None for a non-release root."""
    if read_overlay_manifest(root, TRELLISMX_OVERLAY) is None:
        return None
    return overlay_stored_bytes(root, TRELLISMX_OVERLAY)


__all__ = [
    "CARRIER_DIRECTORY", "MANIFEST_NAME", "MANIFEST_SCHEMA", "SIDECAR_DIRECTORY", "TRANSFORM_PATH",
    "TRELLISMX_OVERLAY", "overlay_schema", "sidecar_path", "trellismx_assets_root",
    "trellismx_files_present", "trellismx_stored_bytes", "validate_trellismx_manifest",
]
