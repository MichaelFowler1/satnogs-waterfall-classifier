#!/usr/bin/env python3
"""
Train a good-vs-bad SatNOGS waterfall classifier.

On your RTX 3080 this uses the GPU automatically and trains a transfer-learned
ResNet18 in minutes. A tiny from-scratch CNN (--arch tiny) is included so the
pipeline also runs with no internet / no pretrained-weights download.

Usage (after make_synthetic.py or fetch.py have populated ./data):
    pip install torch torchvision pillow numpy matplotlib
    python train.py --arch resnet18 --epochs 8        # real use, on the 3080
    python train.py --arch tiny --epochs 3 --img-size 64   # quick offline check

Outputs: model.pt, metrics.json, confusion_matrix.png
"""
import argparse, json, time
import torch
import torch.nn as nn
from dataset import make_loaders


class TinyCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def build_model(arch, n_classes):
    if arch == "resnet18":
        from torchvision import models
        try:
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("loaded ImageNet-pretrained ResNet18 (transfer learning)")
        except Exception as e:
            print(f"could not fetch pretrained weights ({e}); training from scratch")
            m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    return TinyCNN(n_classes)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    correct = total = 0
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1
    return correct / max(1, total), cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--arch", choices=["resnet18", "tiny"], default="resnet18")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    loaders, classes, weights = make_loaders(
        a.data_dir, img_size=a.img_size, batch_size=a.batch_size, seed=a.seed,
        num_workers=0 if device == "cpu" else 2)
    n_classes = len(classes)

    model = build_model(a.arch, n_classes).to(device)
    crit = nn.CrossEntropyLoss(weight=weights.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best_val, best_state = 0.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
        val_acc, _ = evaluate(model, loaders["val"], device, n_classes)
        print(f"epoch {ep:2d} | loss {run/len(loaders['train'].dataset):.3f} "
              f"| val_acc {val_acc:.3f} | {time.time()-t0:.1f}s")
        if val_acc >= best_val:
            best_val, best_state = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, cm = evaluate(model, loaders["test"], device, n_classes)
    print(f"\nTEST accuracy: {test_acc:.3f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{c:>10}" for c in classes))
    per_class = {}
    for i, c in enumerate(classes):
        row = cm[i]
        tp = row[i].item(); fn = (row.sum() - tp).item()
        col = cm[:, i]; fp = (col.sum() - tp).item()
        recall = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        per_class[c] = {"precision": round(prec, 3), "recall": round(recall, 3)}
        print(f"{c:>8}  " + "  ".join(f"{v.item():>10}" for v in row))
    for c, m in per_class.items():
        print(f"  {c:>8}: precision {m['precision']:.3f}  recall {m['recall']:.3f}")

    torch.save({"state_dict": model.state_dict(), "classes": classes,
                "arch": a.arch, "img_size": a.img_size}, "model.pt")
    json.dump({"test_accuracy": round(test_acc, 4), "best_val_accuracy": round(best_val, 4),
               "per_class": per_class, "confusion_matrix": cm.tolist(), "classes": classes},
              open("metrics.json", "w"), indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(n_classes)); ax.set_xticklabels(classes)
        ax.set_yticks(range(n_classes)); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, cm[i, j].item(), ha="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"SatNOGS waterfall  acc={test_acc:.2f}")
        fig.tight_layout(); fig.savefig("confusion_matrix.png", dpi=120)
        print("saved confusion_matrix.png")
    except Exception as e:
        print(f"(skipped confusion-matrix plot: {e})")
    print("saved model.pt + metrics.json")


if __name__ == "__main__":
    main()
