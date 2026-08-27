import random

import numpy as np
import pytest

pytest.importorskip("torch")

from models import PairAugmentConfig, augment_pair_crop


def _config(**overrides):
    values = {
        "enabled": True,
        "rotation_probability": 0.0,
        "horizontal_flip_probability": 0.0,
        "vertical_flip_probability": 0.0,
        "scale_min": 1.0,
        "scale_max": 1.0,
        "small_rotation_probability": 0.0,
        "swap_probability": 0.0,
        "identity_probability": 0.0,
        "cutout_probability": 0.0,
        "copy_paste_probability": 0.0,
        "shift_probability": 0.0,
        "exposure_probability": 0.0,
        "white_balance_probability": 0.0,
        "illumination_probability": 0.0,
        "specular_probability": 0.0,
        "noise_probability": 0.0,
        "jpeg_probability": 0.0,
        "sync_photometric_probability": 0.0,
        "sync_gamma_probability": 0.0,
        "sync_motion_blur_probability": 0.0,
    }
    values.update(overrides)
    return PairAugmentConfig(**values)


def _sample(size=64):
    y, x = np.indices((size, size))
    original = np.stack((x * 3, y * 3, (x + y) * 2), axis=2).clip(0, 255).astype(np.uint8)
    erased = original.copy()
    erased[28:36, 28:36] = 0
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[28:36, 28:36] = 1
    return original, erased, mask


def _augment(config, *, strength=1.0, donor_provider=None):
    return augment_pair_crop(
        *_sample(), crop_size=64, biased_probability=0.0,
        rng=random.Random(17), config=config,
        curriculum_strength=strength, donor_provider=donor_provider,
    )


def test_independent_photometric_noise_does_not_change_fixed_mask():
    original, _, mask = _sample()
    result_a, result_b, result_mask = augment_pair_crop(
        original, original.copy(), np.zeros_like(mask),
        crop_size=64, biased_probability=0.0, rng=random.Random(4),
        config=_config(exposure_probability=1.0, max_exposure_delta=0.12),
        curriculum_strength=1.0,
    )

    assert np.any(result_a != result_b)
    assert not result_mask.any()


def test_curriculum_zero_disables_independent_photometric_noise():
    original, _, mask = _sample()
    result_a, result_b, result_mask = augment_pair_crop(
        original, original.copy(), np.zeros_like(mask),
        crop_size=64, biased_probability=0.0, rng=random.Random(4),
        config=_config(exposure_probability=1.0), curriculum_strength=0.0,
    )

    assert np.array_equal(result_a, result_b)
    assert not result_mask.any()


def test_identity_pair_zeros_mask_and_makes_views_equal():
    result_a, result_b, result_mask = _augment(_config(identity_probability=1.0))

    assert np.array_equal(result_a, result_b)
    assert not result_mask.any()


def test_cutout_updates_mask_where_the_selected_view_changes():
    result_a, result_b, result_mask = _augment(_config(cutout_probability=1.0))

    synthetic_difference = np.any(result_a != result_b, axis=2)
    assert result_mask.sum() > 64
    assert np.all(result_mask[synthetic_difference] == 1)


def test_independent_shift_dilates_positive_mask():
    _, _, original_mask = _sample()
    _, _, shifted_mask = _augment(
        _config(shift_probability=1.0, max_shift_pixels=2), strength=1.0
    )

    assert shifted_mask.sum() > original_mask.sum()


def test_copy_paste_loads_donor_lazily_and_updates_mask():
    calls = []

    def donor_provider():
        calls.append(True)
        donor = np.full((64, 64, 3), 255, dtype=np.uint8)
        donor_mask = np.zeros((64, 64), dtype=np.uint8)
        donor_mask[24:40, 24:40] = 1
        return donor, donor_mask

    _, _, result_mask = _augment(
        _config(copy_paste_probability=1.0), donor_provider=donor_provider
    )

    assert calls == [True]
    assert result_mask.sum() > 64


def test_disabled_augmentation_still_returns_a_fixed_crop():
    a, b, mask = _augment(PairAugmentConfig(enabled=False))

    assert a.shape == b.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
