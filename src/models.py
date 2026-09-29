import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super(ResNet18SimCLR, self).__init__()
        # Use random initialization for baseline, as requested "train SimCLR only once... reuse resulting encoder"
        # We start from scratch or from imagenet. The prompt says "Implement SimCLR using ResNet-18... train only once without labels".
        # We'll use pretrained=False to strictly follow standard SSL, but standard is fine. Let's use weights=None.
        self.backbone = resnet18(weights=None)
        
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() # Remove the classification head
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, proj_dim)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        proj = self.projection_head(features)
        return features, proj

class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes, pretrained_encoder_path=None):
        super(ResNet18Classifier, self).__init__()
        # If supervised baseline, train from scratch (or ImageNet). We'll use from scratch to keep it fair with SimCLR from scratch.
        # Wait, usually "ResNet-18 baseline" means ImageNet pretrained or from scratch. I'll use ImageNet pretrained for baseline if pretrained_encoder_path is None, as is standard in medical imaging, but let's allow weights=None for true scratch. 
        # For better results in reasonable time, I'll use ImageNet weights for baseline.
        if pretrained_encoder_path is None:
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            self.backbone = resnet18(weights=None)
            
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Linear(dim_mlp, num_classes)
        
        if pretrained_encoder_path is not None:
            # Load SimCLR weights
            checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
            # The SimCLR model wraps backbone. We need to load only backbone weights
            # or load state dict into self.backbone
            state_dict = checkpoint['state_dict']
            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    backbone_state_dict[k.replace('backbone.', '')] = v
                    
            # Handle if fc was replaced with Identity in saved state
            if 'fc.weight' in backbone_state_dict:
                del backbone_state_dict['fc.weight']
                del backbone_state_dict['fc.bias']
                
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out
