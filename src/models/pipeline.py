"""
BioInspiredVisionPipeline module.

Integrates the full scale-invariant bio-inspired vision pipeline:
Input (B, 3, H, W)
  -> BioFrontEnd f_theta (512 channels)
  -> SpaceVariantFoveatedPool (Early SVFP locked to 56x56 or 42x75 grid)
  -> GroupedAdapter (Grouped 3x3 conv with groups=256, GN, 1x1 conv to 64 channels)
  -> ResNet Backbone (Layer 1 -> Layer 2 -> Layer 3 -> Layer 4 -> Head)
  -> Logits (B, num_classes)
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torchvision.models as models

from .bio_frontend import BioFrontEnd
from .foveated_pooling import SpaceVariantFoveatedPool
from .grouped_adapter import GroupedAdapter


class BioInspiredVisionPipeline(nn.Module):
    """
    End-to-End Scale-Invariant Bio-Inspired Vision Pipeline.
    
    Args:
        backbone_name: Name of torchvision backbone ('resnet18' or 'resnet50').
        num_classes: Number of classification output classes (default: 100 for ImageNet-100).
        pretrained_backbone: Whether to load ImageNet pre-trained weights for the backbone.
        r_rel: Relative receptive field radius for subcortical DoG.
        r_fovea: Relative foveation boundary radius.
        r_norm: Relative normalization pooling radius.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 100,
        pretrained_backbone: bool = False,
        r_rel: float = 0.015,
        r_fovea: float = 0.15,
        r_norm: float = 0.035,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        # 1. Biological Front-End (Fixed / Analytic)
        self.frontend = BioFrontEnd(r_rel=r_rel, r_norm=r_norm)

        # 2. Early Space-Variant Foveated Pooling (SVFP)
        self.foveated_pool = SpaceVariantFoveatedPool(r_fovea=r_fovea)

        # 3. Grouped Spatial-Aware Adapter (Trainable Bridge)
        self.adapter = GroupedAdapter(in_channels=512, out_channels=64)

        # 4. Downstream ResNet Backbone
        self._init_backbone(backbone_name, num_classes, pretrained_backbone)

    def _init_backbone(self, backbone_name: str, num_classes: int, pretrained: bool):
        """Initializes the ResNet backbone layers, bypassing conv1 and maxpool."""
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.resnet18(weights=weights)
            except Exception:
                base_model = models.resnet18(weights=None)
            fc_in_features = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            try:
                base_model = models.resnet50(weights=weights)
            except Exception:
                base_model = models.resnet50(weights=None)
            fc_in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Use 'resnet18' or 'resnet50'.")

        # Extract stages directly
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(fc_in_features, num_classes)

    def forward_features(
        self,
        x: torch.Tensor,
        gaze_center: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts intermediate features across all stages for analysis and delta gating.
        """
        # Step 1: BioFrontEnd (B, 512, H/2, W/2)
        x_v1 = self.frontend(x)

        # Step 2: Early Space-Variant Foveated Pooling -> (B, 512, H_g, W_g)
        x_fov = self.foveated_pool(x_v1, gaze_center=gaze_center)

        # Step 3: Grouped Spatial Adapter -> (B, 64, H_g, W_g)
        x_adapt = self.adapter(x_fov)

        # Step 4: ResNet Stages
        x_l1 = self.layer1(x_adapt)
        x_l2 = self.layer2(x_l1)
        x_l3 = self.layer3(x_l2)
        x_l4 = self.layer4(x_l3)

        return {
            "v1": x_v1,
            "fov": x_fov,
            "adapter": x_adapt,
            "layer1": x_l1,
            "layer2": x_l2,
            "layer3": x_l3,
            "layer4": x_l4
        }

    def forward(
        self,
        x: torch.Tensor,
        gaze_center: Optional[torch.Tensor] = None,
        cached_l4: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Standard forward pass. Supports temporal bypass when cached_l4 is supplied.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
            gaze_center: Optional (B, 2) tensor of normalized gaze coordinates in [-1, 1].
            cached_l4: Optional cached Layer 4 tensor from previous frame (if delta < tau_reuse).
            
        Returns:
            logits: (B, num_classes) classification logits.
        """
        if cached_l4 is not None:
            # Temporal bypass: skip Stage 1-4 computation directly to head
            pooled = self.avgpool(cached_l4)
            flat = torch.flatten(pooled, 1)
            logits = self.fc(flat)
            return logits

        feats = self.forward_features(x, gaze_center=gaze_center)
        pooled = self.avgpool(feats["layer4"])
        flat = torch.flatten(pooled, 1)
        logits = self.fc(flat)
        return logits
