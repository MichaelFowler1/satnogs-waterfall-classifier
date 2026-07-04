#!/usr/bin/env python3
"""
Generate docs/hero.png - the README image.

A simple, honest illustration of the task: real SatNOGS waterfalls, three
human-vetted GOOD (signal present) next to three human-vetted BAD (noise only).
No model output is shown - just the data and the problem.

Run:  python make_hero.py
"""
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

SEED = 7
PER_CLASS = 3

BG, INK, DIM = "#070b12", "#d7e2f0", "#6b7d95"
GOOD, BAD = "#25d07d", "#ff4d4d"

rng = random.Random(SEED)
picks = []
for label in ("good", "bad"):
    files = sorted(glob.glob(os.path.join("data", label, "*.png")))
    picks += [(p, label) for p in rng.sample(files, PER_CLASS)]

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
fig = plt.figure(figsize=(13, 6.2), facecolor=BG)
fig.text(0.05, 0.94, "SATNOGS WATERFALL CLASSIFIER", fontsize=17, fontweight="bold")
fig.text(0.05, 0.885, "Can you tell which passes caught a satellite? Real ground-station "
                      "spectrograms, human-vetted: signal present vs noise only.",
         fontsize=10.5, color=DIM)
fig.text(0.95, 0.94, "GOOD = signal", ha="right", fontsize=11,
         fontweight="bold", color=GOOD)
fig.text(0.95, 0.895, "BAD = noise", ha="right", fontsize=11,
         fontweight="bold", color=BAD)

cols = len(picks)
for k, (path, label) in enumerate(picks):
    ax = fig.add_axes([0.05 + k * 0.90 / cols, 0.09, 0.90 / cols - 0.012, 0.70])
    ax.imshow(Image.open(path).convert("RGB"), aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    col = GOOD if label == "good" else BAD
    for s in ax.spines.values():
        s.set_edgecolor(col); s.set_linewidth(2.5)
    ax.set_title(label.upper(), fontsize=13, fontweight="bold", color=col, pad=8)

fig.text(0.05, 0.015, "Data: SatNOGS Network / Libre Space Foundation (CC-BY-SA)",
         fontsize=8, color=DIM)

os.makedirs("docs", exist_ok=True)
fig.savefig("docs/hero.png", dpi=140, facecolor=BG)
print("[+] wrote docs/hero.png")
for p, l in picks:
    print(f"    {l}: {os.path.basename(p)}")
