"""The shipped model.pt is part of the repo's contract: it must load on CPU
and produce a calibrated two-class prediction for an arbitrary waterfall
image. metrics.json must stay consistent with the checkpoint's classes."""
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from infer import load_model, predict
from make_synthetic import add_carrier, colorize, make_noise

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "model.pt")
METRICS = os.path.join(REPO, "metrics.json")


@pytest.fixture(scope="module")
def model_bundle():
    return load_model(MODEL)


def test_checkpoint_loads_with_two_classes(model_bundle):
    _, classes, _ = model_bundle
    assert sorted(classes) == ["bad", "good"]


def test_predict_returns_valid_distribution(model_bundle, tmp_path):
    model, classes, tf = model_bundle
    rng = np.random.default_rng(0)
    img = tmp_path / "waterfall.png"
    Image.fromarray(colorize(add_carrier(make_noise(128, 128, rng), rng))).save(img)

    label, conf, probs = predict(model, classes, tf, str(img))
    assert label in classes
    assert 0.0 <= conf <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 0.01


def test_predict_is_deterministic(model_bundle, tmp_path):
    model, classes, tf = model_bundle
    rng = np.random.default_rng(3)
    img = tmp_path / "waterfall.png"
    Image.fromarray(colorize(make_noise(128, 128, rng))).save(img)

    first = predict(model, classes, tf, str(img))
    second = predict(model, classes, tf, str(img))
    assert first == second


def test_metrics_match_checkpoint_classes(model_bundle):
    _, classes, _ = model_bundle
    metrics = json.load(open(METRICS))
    assert metrics["classes"] == classes
    assert set(metrics["per_class"]) == set(classes)
    assert 0.0 <= metrics["test_accuracy"] <= 1.0
