import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = resnet18(weights=None)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(inplace=True), nn.Linear(dim_mlp, proj_dim))

    def forward(self, x):
        features = self.backbone(x)
        return features, self.projection_head(features)


class ResNet18Classifier(nn.Module):
    """init: 'imagenet' | 'simclr' | 'scratch'"""
    def __init__(self, num_classes, init='imagenet', simclr_path='checkpoints/simclr_best.pth'):
        super().__init__()
        assert init in ('imagenet', 'simclr', 'scratch'), init
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if init == 'imagenet' else None)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(dim_mlp, num_classes)

        if init == 'simclr':
            ckpt = torch.load(simclr_path, map_location='cpu')
            sd = {k[len('backbone.'):]: v for k, v in ckpt['state_dict'].items()
                  if k.startswith('backbone.')}
            self.backbone.load_state_dict(sd, strict=True)

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.classifier(self.backbone(x))


def build_reference_encoder(kind='imagenet', simclr_path='checkpoints/simclr_best.pth'):
    """Frozen, label-agnostic feature extractor used for kNN noise detection."""
    enc = ResNet18Classifier(2, init=kind, simclr_path=simclr_path).backbone
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval()
