# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bind the reference vision computation to native patch and feature storage."""

from hyperflare_recipes.components.deepseek_vision import merge_image_features


class ReferenceVision:
    """Consume validated native patches with the checkpoint's encoder/projector.

    Native merge indices encode the same channel-major spatial concatenation
    as the reference image processor. Shared vision generation validates the
    request and publishes only its declared feature prefix.
    """

    def __init__(self, source, inputs, *, capacity):
        if (type(capacity) is not int or capacity < 9 or capacity % 9
                or capacity > inputs.vision_pixel_values.shape[0]
                or source.visual_encoder.patch_size != 14
                or source.aligner.cfg.downsample_ratio != 3
                or source.aligner.cfg.projector_type != "downsample_mlp_gelu"):
            raise ValueError("reference vision requires matching 14-pixel patches and 3-by-3 feature groups")
        self.source, self.inputs, self.capacity = source, inputs, capacity
        self.position_limit = min(capacity, source.visual_encoder.rope_table.shape[0])

    def __call__(self, pixels, positions, cu, indices):
        patches = pixels.view(self.capacity, 3, 14, 14)
        features = self.source.visual_encoder(patches, positions, cu, self.capacity)
        return self.source.aligner.layers(merge_image_features(features, indices))
