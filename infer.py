#!/usr/bin/env python3
"""
Run a trained model on new SatNOGS waterfalls and predict good vs bad.

Usage:
    python infer.py --image path/to/waterfall.png
    python infer.py --dir folder/of/waterfalls      # writes predictions.csv
"""
import argparse, csv, glob, os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from train import build_model
from dataset import IMAGENET_MEAN, IMAGENET_STD


def load_model(path="model.pt"):
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["arch"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return model, ckpt["classes"], tf


@torch.no_grad()
def predict(model, classes, tf, path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
    probs = F.softmax(model(x), 1)[0]
    i = int(probs.argmax())
    return classes[i], float(probs[i]), {c: round(float(p), 3) for c, p in zip(classes, probs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--image")
    ap.add_argument("--dir")
    a = ap.parse_args()
    model, classes, tf = load_model(a.model)

    if a.image:
        label, conf, allp = predict(model, classes, tf, a.image)
        print(f"{os.path.basename(a.image)} -> {label}  ({conf:.1%})   {allp}")
    elif a.dir:
        rows = []
        files = sorted(glob.glob(os.path.join(a.dir, "*.png")) +
                       glob.glob(os.path.join(a.dir, "*.jpg")))
        for f in files:
            label, conf, _ = predict(model, classes, tf, f)
            rows.append((os.path.basename(f), label, round(conf, 3)))
            print(f"{os.path.basename(f):>28} -> {label:>5}  ({conf:.1%})")
        with open("predictions.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["file", "prediction", "confidence"]); w.writerows(rows)
        print(f"\nwrote predictions.csv ({len(rows)} files)")
    else:
        ap.error("pass --image or --dir")


if __name__ == "__main__":
    main()
