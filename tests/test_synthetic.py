"""The synthetic waterfall generator is the offline stand-in for real SatNOGS
data, so the pipeline's correctness rests on its two invariants: a carrier
visibly brightens the image, and generation is deterministic per seed."""
import numpy as np

from make_synthetic import add_carrier, colorize, make_noise


def test_noise_is_normalized_grayscale():
    rng = np.random.default_rng(0)
    img = make_noise(64, 64, rng)
    assert img.shape == (64, 64)
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_carrier_brightens_image():
    rng = np.random.default_rng(1)
    base = make_noise(128, 128, rng)
    with_signal = add_carrier(base.copy(), rng)
    # A "good" waterfall must be distinguishable from noise: total energy rises
    # and stays clipped to [0, 1].
    assert with_signal.sum() > base.sum()
    assert with_signal.max() <= 1.0


def test_generation_is_deterministic_per_seed():
    a = add_carrier(make_noise(64, 64, np.random.default_rng(7)),
                    np.random.default_rng(7))
    b = add_carrier(make_noise(64, 64, np.random.default_rng(7)),
                    np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_colorize_produces_rgb_uint8():
    rng = np.random.default_rng(2)
    rgb = colorize(make_noise(32, 32, rng))
    assert rgb.shape == (32, 32, 3)
    assert rgb.dtype == np.uint8
