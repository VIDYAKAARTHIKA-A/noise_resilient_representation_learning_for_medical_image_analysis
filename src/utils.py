import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class EarlyStopping:
    def __init__(self, patience=5, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric, model, path):
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, path)
            self.counter = 0
            
    def save_checkpoint(self, model, path):
        torch.save(model.state_dict(), path)

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets, _, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    mac_p = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    mac_r = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    mac_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    wt_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    
    return {
        'accuracy': acc,
        'macro_precision': mac_p,
        'macro_recall': mac_r,
        'macro_f1': mac_f1,
        'weighted_f1': wt_f1,
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist()
    }

def save_results(results, filepath):
    # Load existing or create new
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
    else:
        data = []
        
    data.append(results)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)
        
def load_config(filepath):
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config

def plot_confusion_matrix(cm, classes, filename):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
