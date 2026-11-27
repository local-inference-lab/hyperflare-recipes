# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare the prepared CUDA solver with the independent Diffusers scheduler."""

import os
import pickle

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers import DPMSolverMultistepScheduler
from hyperflare_recipes.recipes.pixart.scheduler import PreparedDPMStep


def scheduler(steps=20, **overrides):
    config = dict(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02,
                  beta_schedule="linear", algorithm_type="dpmsolver++", solver_order=2,
                  solver_type="midpoint", prediction_type="epsilon", thresholding=False,
                  lower_order_final=True, euler_at_final=False, final_sigmas_type="zero",
                  timestep_spacing="linspace", steps_offset=0, variance_type=None)
    config.update(overrides)
    result = DPMSolverMultistepScheduler(**config)
    result.set_timesteps(steps)
    return result


def compare_trajectory(*, steps, seed, dtype, device):
    reference = scheduler(steps)
    prepared = [PreparedDPMStep(reference, index) for index in range(steps)]
    prepared = pickle.loads(pickle.dumps(prepared))
    generator = torch.Generator().manual_seed(seed)
    sample = torch.randn((1, 4, 8, 8), generator=generator, dtype=dtype).to(device)
    previous_converted = None
    for index, timestep in enumerate(reference.timesteps):
        noise = torch.randn(sample.shape, generator=generator, dtype=dtype).to(device) * 0.15
        actual_converted, actual = prepared[index](noise, sample, previous_converted)
        expected = reference.step(noise, timestep, sample, return_dict=False)[0]
        expected_converted = reference.model_outputs[-1]
        assert actual.dtype == expected.dtype == dtype
        assert torch.isfinite(actual).all(), index
        assert torch.equal(actual_converted, expected_converted), (index, "converted prediction")
        assert torch.equal(actual, expected), (index, "previous latent")
        previous_converted = actual_converted
        sample = expected
    assert [step.order for step in prepared] == [1, *([2] * (steps - 2)), 1]


@pytest.mark.parametrize("steps", [4, 20])
@pytest.mark.parametrize("seed", [42, 43])
def test_prepared_solver_matches_every_canonical_float32_step_on_cpu(steps, seed):
    compare_trajectory(steps=steps, seed=seed, dtype=torch.float32, device="cpu")


def test_cpu_half_tensor_scalar_rounding_is_not_the_cuda_preparation_contract():
    # Torch 2.14.1 CPU tensor multiplication rounds this coefficient before
    # multiplication; the Python scalar form preserves it until the result cast.
    coefficient = torch.tensor(-0.004044429864734411, dtype=torch.float32)
    prediction = torch.tensor([-293.5], dtype=torch.float16)
    tensor_product = coefficient * prediction
    scalar_product = coefficient.item() * prediction
    assert torch.equal(tensor_product, coefficient.half() * prediction)
    assert torch.equal(scalar_product, (coefficient.item() * prediction.float()).half())
    assert tensor_product.item() == 1.1865234375
    assert scalar_product.item() == 1.1875


@pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="CUDA tests require FLARE_RUN_CUDA_TESTS=1")
@pytest.mark.parametrize("steps", [4, 20])
@pytest.mark.parametrize("seed", [42, 43])
def test_prepared_solver_matches_canonical_cpu_scalar_promotion_on_cuda(steps, seed):
    from ignition import runtime_patches

    runtime_patches.apply()
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    compare_trajectory(steps=steps, seed=seed, dtype=torch.float16, device="cuda")


def test_prepared_coefficients_are_independent_of_scheduler_mutation():
    reference = scheduler()
    prepared = PreparedDPMStep(reference, 1)
    sample = torch.tensor([0.25, -0.75], dtype=torch.float32)
    noise = torch.tensor([-0.5, 0.125], dtype=torch.float32)
    history = torch.tensor([1.0, -1.0], dtype=torch.float32)
    expected = prepared(noise, sample, history)
    reference.set_timesteps(4)
    reference.sigmas.zero_()
    actual = prepared(noise, sample, history)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_second_order_requires_history():
    prepared = PreparedDPMStep(scheduler(), 1)
    with pytest.raises(ValueError, match="previous converted prediction"):
        prepared(torch.ones(2), torch.ones(2))


@pytest.mark.parametrize("overrides", [dict(solver_type="heun"), dict(solver_order=3),
                                      dict(prediction_type="sample"), dict(thresholding=True)])
def test_preparation_rejects_unimplemented_solver_semantics(overrides):
    with pytest.raises(ValueError, match="requires deterministic epsilon"):
        PreparedDPMStep(scheduler(**overrides), 0)


@pytest.mark.parametrize("index", [-1, 20, True])
def test_preparation_rejects_invalid_step_index(index):
    with pytest.raises(ValueError, match="step index"):
        PreparedDPMStep(scheduler(), index)
