SatNOGS Waterfall Classifier
Automatically tell whether a satellite ground station actually caught a signal — a CNN that classifies SatNOGS observation waterfalls as good (signal present) or bad (noise only).

Why this project
SatNOGS is a worldwide network of open-source satellite ground stations. Every observation produces a waterfall — a spectrogram of what the station heard during a pass — and a human vets each one as having a signal or not. That manual vetting is the network's bottleneck, and it's a clean supervised-learning problem: image in, good/bad out.

Automating "is there a signal in this spectrum or not" is also the core primitive of RF surveillance and space domain awareness, which is what makes this more than a toy: it's a real, useful triage model on real RF data.

Pipeline
fetch.py          → download real labeled waterfalls from the SatNOGS open API
make_synthetic.py → generate fake waterfalls to test the pipeline offline
dataset.py        → ImageFolder → train/val/test loaders (deterministic split)
train.py          → ResNet18 transfer learning on your GPU → model.pt + metrics
infer.py          → classify new waterfalls with confidence scores
Quickstart
pip install torch torchvision pillow numpy matplotlib requests

# A) Prove it works in 2 minutes, no API, no internet needed:
python make_synthetic.py --per-class 200
python train.py --arch tiny --epochs 4 --img-size 64
python infer.py --image data/good/good_0000.png

# B) The real thing, on your RTX 3080:
python fetch.py --per-class 800          # real SatNOGS waterfalls + labels
python train.py --arch resnet18 --epochs 8
python infer.py --dir some_new_waterfalls/
train.py uses the GPU automatically (cuda if available). Outputs: model.pt, metrics.json, and confusion_matrix.png.

Data
SatNOGS Network API — public read access, no key required. Labels come from the waterfall_status field (with-signal → good, without-signal → bad).
Data is CC-BY-SA (SatNOGS / Libre Space Foundation). Credit them if you publish.
Honest notes (read these — they're the interesting part)
Label noise. The "good/bad" labels are crowd-sourced human vetting, so some are wrong. Your accuracy ceiling is the labelers' consistency, not 100%. Report where the model and humans disagree — some of those are human mistakes the model caught, which is a great result to surface.
Class imbalance. Real observations skew toward one class; training applies class weights to compensate. Watch per-class recall, not just overall accuracy.
Distribution shift. Different satellites, bands, and stations produce very different-looking waterfalls. A model trained on one slice may not generalize — test across satellites/stations and document it.
Synthetic ≠ real. make_synthetic.py exists only to prove the pipeline. Real results come from fetch.py data. Don't report synthetic numbers as real ones.
Roadmap
Modulation/signal-type classification instead of just good/bad (multi-class).
Anomaly detection — flag a signal where none was expected for that satellite, the SIGINT / space-domain-awareness angle.
Decode, not just detect — pair with gr-satellites to demodulate the passes the model flags as good.
What this demonstrates
End-to-end applied ML on real sensor data: pulling and labeling a dataset from a live API, transfer learning on a GPU, honest evaluation (confusion matrix, per-class precision/recall), and clear-eyed treatment of label noise and distribution shift — the parts that separate a real model from a leaderboard score.
