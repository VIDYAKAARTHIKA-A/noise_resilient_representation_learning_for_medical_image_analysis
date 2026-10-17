"""
Run this BEFORE training. Per-class AUROC of "is this label clean?" for different detectors,
using the ground-truth is_clean flags. If kNN on frozen features beats the loss on the rare
classes (DF, VASC, SCC), the method is worth building on.

  python diagnose_detection.py --noise 0.3 --noise_type uniform_global
  python diagnose_detection.py --noise 0.3 --ckpt checkpoints/baseline_uniform_global_n30_clean_s42.pth
"""
import os
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.utils import load_config
from src.dataset import get_dataloaders, CLASSES
from src.models import ResNet18Classifier, build_reference_encoder
from src.detection import extract_features, knn_neighbors, neighbor_votes, predict_probs, per_class_auroc


def report(name, score, clean, labels, C):
    per = per_class_auroc(score, clean, labels, C)
    overall = roc_auc_score(clean.astype(int), score)
    print(f"{name:<28} overall {overall:.3f} | " +
          " ".join(f"{CLASSES[c]} {a:.2f}" for c, a in enumerate(per)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--noise', type=float, default=0.3)
    ap.add_argument('--noise_type', default='uniform_global')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ks', nargs='+', type=int, default=[10, 20, 50])
    ap.add_argument('--ckpt', default=None, help="trained classifier checkpoint for the loss-based score")
    a = ap.parse_args()

    cfg = load_config('config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = get_dataloaders(cfg, a.noise, a.noise_type, 'clean', a.seed)
    ds, C = d['train_dataset'], d['num_classes']
    labels, clean = np.asarray(ds.labels), np.asarray(ds.is_clean).astype(bool)
    ar = np.arange(len(labels))
    print(f"noise={a.noise:.0%} type={a.noise_type}: actual clean rate {clean.mean():.3f}")
    print("per-class clean rate:", " ".join(f"{CLASSES[c]} {clean[labels == c].mean():.2f}" for c in range(C)))

    kinds = ['imagenet'] + (['simclr'] if os.path.exists('checkpoints/simclr_best.pth') else [])
    for kind in kinds:
        feats = extract_features(build_reference_encoder(kind).to(device), d['score'], device)
        for k in a.ks:
            votes = neighbor_votes(knn_neighbors(feats, k), labels, C)
            report(f"kNN agreement {kind} k={k}", votes[ar, labels], clean, labels, C)

    if a.ckpt:
        m = ResNet18Classifier(C, init='scratch').to(device)
        m.load_state_dict(torch.load(a.ckpt, map_location=device))
        probs = predict_probs(m, d['score'], device, len(labels), C)
        report("loss (negative CE)", np.log(np.clip(probs[ar, labels], 1e-7, 1)), clean, labels, C)
