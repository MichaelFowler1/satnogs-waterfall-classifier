"""make_loaders is where the two documented data-handling guarantees live:
the split is deterministic per seed with no train/val/test leakage, and class
weights are computed from the train split only (real SatNOGS data is
imbalanced). These tests pin both."""
import numpy as np
import pytest
from PIL import Image

from dataset import make_loaders
from make_synthetic import add_carrier, colorize, make_noise


@pytest.fixture(scope="module")
def image_folder(tmp_path_factory):
    """A small imbalanced ImageFolder tree: 12 bad (noise) vs 6 good (signal)."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for label, count in (("bad", 12), ("good", 6)):
        d = root / label
        d.mkdir()
        for i in range(count):
            g = make_noise(32, 32, rng)
            if label == "good":
                g = add_carrier(g, rng)
            Image.fromarray(colorize(g)).save(d / f"{label}_{i:03d}.png")
    return str(root)


def _split_indices(loaders):
    return {name: sorted(dl.dataset.indices) for name, dl in loaders.items()}


def test_split_is_deterministic_per_seed(image_folder):
    a, _, _ = make_loaders(image_folder, img_size=32, batch_size=4,
                           seed=0, num_workers=0)
    b, _, _ = make_loaders(image_folder, img_size=32, batch_size=4,
                           seed=0, num_workers=0)
    assert _split_indices(a) == _split_indices(b)


def test_no_leakage_between_splits(image_folder):
    loaders, _, _ = make_loaders(image_folder, img_size=32, batch_size=4,
                                 seed=0, num_workers=0)
    idx = _split_indices(loaders)
    assert not (set(idx["train"]) & set(idx["val"]))
    assert not (set(idx["train"]) & set(idx["test"]))
    assert not (set(idx["val"]) & set(idx["test"]))
    assert len(idx["train"]) + len(idx["val"]) + len(idx["test"]) == 18


def test_class_weights_upweight_minority(image_folder):
    loaders, classes, weights = make_loaders(image_folder, img_size=32,
                                             batch_size=4, seed=0,
                                             num_workers=0)
    assert sorted(classes) == ["bad", "good"]
    assert weights.shape == (2,)
    # 'good' is the minority class (6 vs 12) so it must carry the larger
    # loss weight.
    minority = classes.index("good")
    majority = classes.index("bad")
    assert weights[minority] > weights[majority]
