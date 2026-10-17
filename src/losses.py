import torch
import torch.nn as nn
import torch.nn.functional as F


def ce_per_sample(logits, targets):
    return F.cross_entropy(logits, targets, reduction='none')


def gce_per_sample(logits, targets, q=0.7):
    """Generalised cross-entropy (Zhang & Sabuncu 2018): (1 - p_y^q) / q.
    q->0 recovers CE, q=1 is MAE. Bounded, so a wrong label cannot dominate the gradient."""
    p = F.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1).clamp(min=1e-7)
    return (1.0 - p ** q) / q


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
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
        return self.criterion(similarity_matrix, labels)
