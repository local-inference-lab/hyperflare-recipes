# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare PixArt's deterministic DPM-Solver++ updates for graph capture."""

import math

import torch


class PreparedDPMStep:
    """Freeze one canonical midpoint update and its CPU-computed scalar factors.

    Diffusers 0.43.0 keeps schedule scalars on the CPU. Resolving them before
    capture preserves their CUDA arithmetic without recording host computations.
    Converted predictions remain explicit tensor state between graph executions.
    CPU FP16 execution is unsupported: PyTorch rounds CPU tensor coefficients
    differently from Python scalar coefficients when multiplying FP16 tensors.
    """

    def __init__(self, scheduler, index):
        config = scheduler.config
        required = dict(algorithm_type="dpmsolver++", prediction_type="epsilon",
                        solver_order=2, solver_type="midpoint", thresholding=False,
                        variance_type=None, use_flow_sigmas=False)
        if any(getattr(config, name) != value for name, value in required.items()):
            raise ValueError("PixArt preparation requires deterministic epsilon DPM-Solver++ midpoint order two")
        steps = scheduler.num_inference_steps
        if steps is None or steps != len(scheduler.timesteps):
            raise ValueError("DPM-Solver timesteps must be resolved before preparation")
        if type(index) is not int or not 0 <= index < steps:
            raise ValueError("DPM-Solver step index lies outside the prepared schedule")
        if (scheduler.sigmas.device.type != "cpu" or scheduler.sigmas.dtype != torch.float32
                or scheduler.sigmas.ndim != 1 or len(scheduler.sigmas) != steps + 1):
            raise ValueError("DPM-Solver preparation requires the canonical CPU float32 sigma table")

        final_first_order = index == steps - 1 and (
            config.euler_at_final or (config.lower_order_final and steps < 15)
            or config.final_sigmas_type == "zero")
        self.order = 1 if index == 0 or final_first_order else 2
        self.index = index
        alpha_s, sigma_s = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas[index])
        alpha_t, sigma_t = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas[index + 1])
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        h = lambda_t - lambda_s
        prediction_factor = alpha_t * (torch.exp(-h) - 1.0)
        self.alpha = alpha_s.item()
        self.sigma = sigma_s.item()
        self.sample_factor = (sigma_t / sigma_s).item()
        self.prediction_factor = prediction_factor.item()
        self.history_factor = 0.0
        self.correction_factor = 0.0
        if self.order == 2:
            alpha_previous, sigma_previous = scheduler._sigma_to_alpha_sigma_t(scheduler.sigmas[index - 1])
            lambda_previous = torch.log(alpha_previous) - torch.log(sigma_previous)
            h_previous = lambda_s - lambda_previous
            ratio = h_previous / h
            self.history_factor = (1.0 / ratio).item()
            self.correction_factor = (0.5 * prediction_factor).item()
        factors = (self.alpha, self.sigma, self.sample_factor, self.prediction_factor,
                   self.history_factor, self.correction_factor)
        if not all(math.isfinite(value) for value in factors) or self.alpha <= 0:
            raise ValueError("DPM-Solver schedule produces invalid prepared coefficients")

    def __call__(self, noise_prediction, sample, previous_converted=None):
        # Conversion precedes the sample upcast in the canonical scheduler.
        converted = (sample - self.sigma * noise_prediction) / self.alpha
        sample = sample.to(torch.float32)
        if self.order == 1:
            previous = self.sample_factor * sample - self.prediction_factor * converted
        else:
            if previous_converted is None:
                raise ValueError("second-order DPM-Solver update requires the previous converted prediction")
            derivative = self.history_factor * (converted - previous_converted)
            previous = (self.sample_factor * sample - self.prediction_factor * converted
                        - self.correction_factor * derivative)
        return converted, previous.to(converted.dtype)
