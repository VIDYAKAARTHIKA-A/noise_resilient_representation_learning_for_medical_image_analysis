import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, class_weights=None):
        # Compute individual unweighted cross-entropy per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply dynamic clean-effective weights if provided
        if class_weights is not None:
            focal_loss = focal_loss * class_weights[targets]
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=8, reduction='mean'):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.reduction = reduction
        
    def forward(self, pred, labels, class_weights=None):
        # pred: raw network logits
        pred_softmax = F.softmax(pred, dim=1)
        pred_softmax = torch.clamp(pred_softmax, min=1e-7, max=1.0)
        
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        # Forward and Reverse Cross Entropy computed per-sample
        ce = -1 * torch.sum(label_one_hot * torch.log(pred_softmax), dim=1)
        rce = -1 * torch.sum(pred_softmax * torch.log(label_one_hot), dim=1)
        
        sce_loss = self.alpha * ce + self.beta * rce
        
        # Apply dynamic clean-effective weights if provided
        if class_weights is not None:
            sce_loss = sce_loss * class_weights[labels]
            
        if self.reduction == 'mean':
            return sce_loss.mean()
        elif self.reduction == 'sum':
            return sce_loss.sum()
        else:
            return sce_loss

class UnifiedDynamicHybridLoss(nn.Module):
    """
    Unified Optimization Engine that dynamically couples Focal Loss and SCE
    using real-time Clean-Effective Sample Volumes (E_c) calculated per epoch.
    """
    def __init__(self, num_classes=8, alpha_hybrid=1.0, beta_hybrid=1.0, gamma_focal=2.0, beta_eff=0.9999):
        super(UnifiedDynamicHybridLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_hybrid = alpha_hybrid
        self.beta_hybrid = beta_hybrid
        self.beta_eff = beta_eff
        
        # Initialize internal loss definitions with 'none' reduction to allow custom pooling
        self.focal_loss_fn = FocalLoss(gamma=gamma_focal, reduction='none')
        self.sce_loss_fn = SymmetricCrossEntropy(num_classes=num_classes, reduction='none')
        
        # Initialize uniform weights as default baseline
        self.register_buffer('class_weights', torch.ones(num_classes))
        
    def update_dynamic_weights(self, selected_indices, all_targets, all_clean_probs, all_indices):
        """
        Closed-Loop update method called at the turn of every epoch.
        Calculates E_c(t) and updates the cost-sensitive class weights.

        selected_indices: dataset indices kept by the curriculum.
        all_targets / all_clean_probs / all_indices: aligned arrays (one entry per
        sample) so that position i in each array refers to dataset index all_indices[i].
        """
        all_targets = np.asarray(all_targets)
        all_clean_probs = np.asarray(all_clean_probs, dtype=np.float64)
        all_indices = np.asarray(all_indices)

        E_c = np.zeros(self.num_classes)

        # 1. Sum up the expected clean probability scores inside the selected curriculum filter
        selected_mask = np.isin(all_indices, np.asarray(selected_indices))
        for c in range(self.num_classes):
            E_c[c] = all_clean_probs[selected_mask & (all_targets == c)].sum()

        # 2. Compute the dynamic inverse effective volume weights
        new_weights = np.ones(self.num_classes)
        for c in range(self.num_classes):
            if E_c[c] > 0:
                new_weights[c] = (1.0 - self.beta_eff) / (1.0 - np.power(self.beta_eff, E_c[c]))
            else:
                new_weights[c] = 1.0  # Fallback for rare cases with zero sample pass
                
        # 3. Normalize weights to maintain gradient magnitude stability
        new_weights = new_weights / np.mean(new_weights)
        
        # 4. Push updated array directly onto the GPU buffer
        self.class_weights = torch.tensor(new_weights, dtype=torch.float32, device=self.class_weights.device)
        
    def forward(self, inputs, targets):
        # Compute individual unweighted sample arrays
        focal_samples = self.focal_loss_fn(inputs, targets, class_weights=None)
        sce_samples = self.sce_loss_fn(inputs, targets, class_weights=None)
        
        # Combine using the hybrid coefficients
        combined_loss = self.alpha_hybrid * focal_samples + self.beta_hybrid * sce_samples
        
        # Weigh the combined loss using the epoch-updated clean-effective tensor w_c(t)
        weighted_loss = combined_loss * self.class_weights[targets]
        
        return weighted_loss.mean()

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        similarity_matrix = similarity_matrix / self.temperature
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size - 1, labels])
        
        loss = self.criterion(similarity_matrix, labels)
        return loss