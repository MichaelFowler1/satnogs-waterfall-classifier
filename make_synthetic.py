#!/usr/bin/env python3
"""
Generate synthetic SatNOGS-style waterfalls so you can test the whole training
pipeline in two minutes, before downloading any real data.

A real SatNOGS waterfall is a spectrogram (frequency across, time down) of what a
ground station heard during a satellite pass:
  • "good"  / with-signal  -> a bright trace (often a doppler-curved carrier line)
  • "bad"   / without-signal -> just noise

These fakes capture that difference so the CNN has an easy version of the real
task to prove itself on. Output layout is ImageFolder-ready:
    data/good/*.png   data/bad/*.png

Usage:
    pip install pillow numpy
    python make_synthetic.py --per-class 200 --out data
"""
import argparse, os, math, random
import numpy as np
from PIL import Image


def make_noise(h, w, rng):
    return rng.normal(0.35, 0.12, (h, w)).clip(0, 1)


def add_carrier(img, rng):
    """Draw a doppler-curved bright vertical-ish trace (a real signal)."""
    h, w = img.shape
    cx = rng.uniform(0.25, 0.75) * w           # center frequency column
    drift = rng.uniform(-0.18, 0.18) * w        # doppler sweep across the pass
    width = rng.uniform(0.8, 2.2)               # trace thickness in columns
    bright = rng.uniform(0.55, 0.95)
    for row in range(h):
        t = row / h
        # S-curve doppler shift over the pass
        col = cx + drift * (t - 0.5) + 0.04 * w * math.sin(2 * math.pi * t)
        lo, hi = int(col - 3 * width), int(col + 3 * width) + 1
        for c in range(max(0, lo), min(w, hi)):
            img[row, c] = min(1.0, img[row, c] + bright * math.exp(-((c - col) ** 2) / (2 * width ** 2)))
    return img


def colorize(gray):
    """Apply a viridis-ish colormap so it looks like a real waterfall PNG."""
    g = gray
    r = np.clip(1.4 * g - 0.4, 0, 1)
    gr = np.clip(1.1 * g, 0, 1)
    b = np.clip(1.2 - 1.3 * g, 0, 1)
    return (np.stack([r, gr, b], -1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--out", default="data")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for label in ("good", "bad"):
        d = os.path.join(a.out, label)
        os.makedirs(d, exist_ok=True)
        for i in range(a.per_class):
            g = make_noise(a.size, a.size, rng)
            if label == "good":
                for _ in range(rng.integers(1, 3)):
                    g = add_carrier(g, rng)
            Image.fromarray(colorize(g)).save(os.path.join(d, f"{label}_{i:04d}.png"))
    print(f"Wrote {a.per_class} good + {a.per_class} bad waterfalls to {a.out}/")


if __name__ == "__main__":
    main()
