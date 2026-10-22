"""
Discriminative Parameter Groups and Dynamic Unfreezing Manager.

Configures layer-wise learning rate multipliers and handles smooth progressive unfreezing
with 3-epoch dampening to protect pre-trained weights from 1-Cycle peak learning rate shocks.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn

from ..models.pipeline import BioInspiredVisionPipeline


class DynamicUnfreezeManager:
    """
    Manages stage unfreezing and learning rate dampening across epochs.
    
    Unfreezing schedule:
    - Epoch 1..5: Train adapter and fc head only. Layers 1-4 frozen.
    - Epoch 6: Unfreeze Layer 4 (dampening ramp over 3 epochs: 0.33 -> 0.67 -> 1.0).
    - Epoch 8: Unfreeze Layer 3 (dampening ramp over 3 epochs).
    - Epoch 10..50: Unfreeze Layers 1 and 2 (dampening ramp over 3 epochs).
    """
    def __init__(self, dampening_epochs: int = 3):
        self.dampening_epochs = dampening_epochs
        self.schedule = {
            "adapter": {"unfreeze_epoch": 1, "lr_mult": 1.0},
            "fc":      {"unfreeze_epoch": 1, "lr_mult": 10.0},
            "layer4":  {"unfreeze_epoch": 6, "lr_mult": 1.0},
            "layer3":  {"unfreeze_epoch": 8, "lr_mult": 0.1},
            "layer2":  {"unfreeze_epoch": 10, "lr_mult": 0.01},
            "layer1":  {"unfreeze_epoch": 10, "lr_mult": 0.01},
        }

    def build_param_groups(self, model: BioInspiredVisionPipeline) -> List[Dict[str, Any]]:
        """Constructs PyTorch optimizer parameter groups with metadata."""
        module_map = {
            "adapter": model.adapter,
            "fc": model.fc,
            "layer4": model.layer4,
            "layer3": model.layer3,
            "layer2": model.layer2,
            "layer1": model.layer1,
        }

        param_groups: List[Dict[str, Any]] = []
        for name, module in module_map.items():
            cfg = self.schedule[name]
            params = [p for p in module.parameters() if p.requires_grad or True]
            param_groups.append({
                "name": name,
                "params": params,
                "lr_mult": cfg["lr_mult"],
                "unfreeze_epoch": cfg["unfreeze_epoch"],
                "base_lr_mult": cfg["lr_mult"],
            })
        return param_groups

    def update_epoch(self, epoch: int, optimizer: torch.optim.Optimizer):
        """
        Updates requires_grad flags and applies learning rate dampening for freshly unfrozen layers.
        
        Args:
            epoch: Current training epoch (1-indexed).
            optimizer: SGD optimizer containing the parameter groups.
        """
        for group in optimizer.param_groups:
            name = group.get("name", "")
            unfreeze_ep = group.get("unfreeze_epoch", 1)

            if epoch < unfreeze_ep:
                # Stage is frozen
                for p in group["params"]:
                    p.requires_grad = False
            else:
                # Stage is active
                for p in group["params"]:
                    p.requires_grad = True

                # Apply 3-epoch dampening factor to avoid peak 1-cycle shock
                elapsed = epoch - unfreeze_ep + 1
                dampening = min(1.0, elapsed / float(self.dampening_epochs))
                # Scale lr multiplier accordingly
                group["lr_mult"] = group["base_lr_mult"] * dampening
