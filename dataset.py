#!/usr/bin/env python3
"""
Build train/val/test DataLoaders from an ImageFolder directory:
    data/good/*.png
    data/bad/*.png

Same deterministic split for train vs eval (via identical shuffle seed) so we can
apply augmentation to train only while keeping val/test clean.
"""
import random
import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _transforms(img_size, train):
    t = [transforms.Resize((img_size, img_size))]
    if train:
        # waterfalls: time axis is vertical, frequency horizontal. A small
        # vertical flip (time reversal) is a safe, label-preserving augmentation.
        t += [transforms.RandomVerticalFlip(0.3)]
    t += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(t)


def make_loaders(data_dir, img_size=128, batch_size=32,
                 val_frac=0.15, test_frac=0.15, seed=0, num_workers=2):
    train_full = datasets.ImageFolder(data_dir, transform=_transforms(img_size, True))
    eval_full = datasets.ImageFolder(data_dir, transform=_transforms(img_size, False))
    classes = train_full.classes

    n = len(train_full)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(test_frac * n)
    n_val = int(val_frac * n)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(eval_full, val_idx)
    test_ds = Subset(eval_full, test_idx)

    # class weights from the TRAIN split only (real SatNOGS data is imbalanced)
    targets = train_full.targets
    counts = [0] * len(classes)
    for i in train_idx:
        counts[targets[i]] += 1
    total = sum(counts)
    weights = torch.tensor([total / (len(classes) * max(1, c)) for c in counts],
                           dtype=torch.float32)

    mk = lambda ds, shuf: DataLoader(ds, batch_size=batch_size, shuffle=shuf,
                                     num_workers=num_workers, pin_memory=True)
    loaders = {"train": mk(train_ds, True), "val": mk(val_ds, False), "test": mk(test_ds, False)}
    print(f"dataset: {n} images | classes={classes} | "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} | "
          f"train class counts={dict(zip(classes, counts))}")
    return loaders, classes, weights
