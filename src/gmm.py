import numpy as np
from sklearn.mixture import GaussianMixture
import torch

def get_losses(model, dataloader, device, criterion):
    model.eval()
    all_losses = []
    all_indices = []
    
    with torch.no_grad():
        for inputs, targets, indices, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # compute loss per sample
            loss = criterion(outputs, targets).cpu().numpy()
            all_losses.extend(loss)
            all_indices.extend(indices.numpy())
            
    all_losses = np.array(all_losses)
    all_indices = np.array(all_indices)
    
    # Sort back by original index if needed, or return a dict
    idx_to_loss = {idx: loss for idx, loss in zip(all_indices, all_losses)}
    return idx_to_loss, all_losses, all_indices

def fit_gmm(losses):
    # Losses is a 1D numpy array
    losses = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
    losses = losses.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(losses)
    
    prob = gmm.predict_proba(losses)
    
    # Determine which component is 'clean' (should have smaller mean loss)
    clean_idx = gmm.means_.argmin()
    clean_prob = prob[:, clean_idx]
    
    return clean_prob, gmm

def estimate_clean_probabilities(model, dataloader, device):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    idx_to_loss, all_losses, all_indices = get_losses(model, dataloader, device, criterion)
    
    clean_prob, gmm = fit_gmm(all_losses)
    
    # Map back to indices
    idx_to_prob = {int(idx): prob for idx, prob in zip(all_indices, clean_prob)}
    
    return idx_to_prob, all_losses, clean_prob, gmm
