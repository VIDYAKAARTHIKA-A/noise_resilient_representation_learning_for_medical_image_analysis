import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # inputs: logits
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, num_classes=8):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        
    def forward(self, pred, labels):
        # pred: logits
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        
        ce = -1 * torch.sum(label_one_hot * torch.log(pred), dim=1)
        
        # Reverse CE
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)
        
        loss = self.alpha * ce + self.beta * rce
        return loss.mean()
        
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity matrix
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
        
        similarity_matrix = similarity_matrix / self.temperature
        
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size - 1, labels]) # Offset for masked diagonal
        
        loss = self.criterion(similarity_matrix, labels)
        return loss
