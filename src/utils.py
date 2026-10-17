import json
import os
import random
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=5, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric, model, path):
        """After the call, counter == 0 means this epoch was the new best (and was saved)."""
        score = metric if self.mode == 'max' else -metric
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def evaluate_model(model, dataloader, device, num_classes, logit_adjust=None):
    """logit_adjust: optional tensor (C,) SUBTRACTED from logits (post-hoc logit adjustment)."""
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            out = model(inputs.to(device))
            if logit_adjust is not None:
                out = out - logit_adjust.to(device)
            preds_all.extend(out.argmax(1).cpu().numpy())
            targets_all.extend(targets.numpy())

    labels = list(range(num_classes))
    return {
        'accuracy': accuracy_score(targets_all, preds_all),
        'macro_precision': precision_score(targets_all, preds_all, average='macro', zero_division=0),
        'macro_recall': recall_score(targets_all, preds_all, average='macro', zero_division=0),
        'balanced_accuracy': recall_score(targets_all, preds_all, labels=labels, average='macro', zero_division=0),
        'macro_f1': f1_score(targets_all, preds_all, average='macro', zero_division=0),
        'weighted_f1': f1_score(targets_all, preds_all, average='weighted', zero_division=0),
        'per_class_f1': f1_score(targets_all, preds_all, labels=labels, average=None, zero_division=0).tolist(),
        'per_class_recall': recall_score(targets_all, preds_all, labels=labels, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(targets_all, preds_all, labels=labels).tolist(),
    }


def save_results(results, filepath):
    """Upsert by run_name, so re-running a run never creates duplicate rows."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    data = [d for d in data if d.get('run_name') != results.get('run_name')]
    data.append(results)
    tmp = filepath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=4)
    os.replace(tmp, filepath)


def load_config(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(cm, classes, filename):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
