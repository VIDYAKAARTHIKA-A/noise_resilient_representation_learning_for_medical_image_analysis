import numpy as np
from sklearn.mixture import GaussianMixture
import torch

def get_losses(model, dataloader, device, criterion):
    model.eval()
    all_losses = []
    all_indices = []
    all_targets = []  # Added to track class labels for class-wise GMM fitting
    
    with torch.no_grad():
        for inputs, targets, indices, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # compute loss per sample
            loss = criterion(outputs, targets).cpu().numpy()
            all_losses.extend(loss)
            all_indices.extend(indices.numpy())
            all_targets.extend(targets.cpu().numpy()) # Capture targets
            
    all_losses = np.array(all_losses)
    all_indices = np.array(all_indices)
    all_targets = np.array(all_targets)
    
    idx_to_loss = {idx: loss for idx, loss in zip(all_indices, all_losses)}
    return idx_to_loss, all_losses, all_indices, all_targets

def estimate_clean_probabilities(model, dataloader, device, nu=10.0):
    """
    Estimates clean sample probabilities using a Class-Wise GMM regularized 
    by an Empirical Bayes Shared-Variance Prior to handle class imbalance.
    """
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # 1. Gather losses, indices, and corresponding target classes
    idx_to_loss, all_losses, all_indices, all_targets = get_losses(model, dataloader, device, criterion)
    
    # 2. Perform global normalization to stabilize loss space scales
    normalized_losses = (all_losses - all_losses.min()) / (all_losses.max() - all_losses.min() + 1e-8)
    
    # 3. Compute the Global Variance Prior across the deep feature space
    global_loss_variance = np.var(normalized_losses)
    
    # Pre-allocate array for clean probabilities across all samples
    clean_prob = np.zeros_like(normalized_losses)
    unique_classes = np.unique(all_targets)
    class_gmms = {}

    # 4. Class-Wise Probabilistic Noise Partitioning Loop
    for c in unique_classes:
        class_mask = (all_targets == c)
        class_losses = normalized_losses[class_mask].reshape(-1, 1)
        N_c = len(class_losses)
        
        # Guard rail against statistical failure for ultra-rare classes
        if N_c < 2:
            clean_prob[class_mask] = 1.0  # Default to clean if sample count is non-statistical
            continue
            
        # Fit 2-component GMM on the specific class distribution
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(class_losses)
        
        # Extract raw empirical variances computed by scikit-learn
        raw_variances = gmm.covariances_.flatten() # Shape: (2,)
        
        # Apply the Hierarchical Shared-Variance Prior formula
        adjusted_variances = (N_c * raw_variances + nu * global_loss_variance) / (N_c + nu)
        
        # Override the GMM's covariance matrix with regularized variables before prediction
        gmm.covariances_ = adjusted_variances.reshape(-1, 1, 1)
        
        # Compute posteriors based on the regularized variance matrices
        prob = gmm.predict_proba(class_losses)
        
        # Map the clean component to the smaller mean loss cluster within this class
        clean_idx = gmm.means_.argmin()
        clean_prob[class_mask] = prob[:, clean_idx]
        class_gmms[c] = gmm
    
    # 5. Map probabilities back to their unique original indices
    idx_to_prob = {int(idx): prob for idx, prob in zip(all_indices, clean_prob)}
    
    # Retained original signature variables along with target array needed for curriculum step
    return idx_to_prob, all_losses, clean_prob, class_gmms, all_targets
