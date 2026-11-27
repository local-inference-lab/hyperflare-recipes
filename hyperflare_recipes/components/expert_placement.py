# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Canonical expert coverage for explicit HBM and mapped-host weight rows."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillExpertStaging:
    """Prepared row threshold and preceding-call cold demand for temporary copies."""

    min_rows: int = 1024
    min_cold_experts: int = 8

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in (self.min_rows, self.min_cold_experts)):
            raise ValueError("prefill expert staging thresholds must be positive integers")

    def to_dict(self):
        return dict(min_rows=self.min_rows, min_cold_experts=self.min_cold_experts)


@dataclass(frozen=True)
class ExpertPlacement:
    """Physical row order; an HBM copy overrides the same expert's host copy."""

    experts: int
    host: tuple[int, ...]
    hbm: tuple[int, ...]

    def __post_init__(self):
        if (type(self.experts) is not int or self.experts <= 0
                or not isinstance(self.host, tuple) or not isinstance(self.hbm, tuple)
                or not self.hbm
                or any(type(i) is not int or not 0 <= i < self.experts for i in self.host + self.hbm)
                or any(len(set(rows)) != len(rows) for rows in (self.host, self.hbm))
                or set(self.host + self.hbm) != set(range(self.experts))):
            raise ValueError("expert placement requires complete canonical coverage and unique rows in each tier, with nonempty HBM")

    @property
    def physical_rows(self):
        return len(self.host) + len(self.hbm)

    def to_dict(self):
        return dict(experts=self.experts, host=list(self.host), hbm=list(self.hbm))
