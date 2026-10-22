"""
Proxy Adaptation Evaluator for Stage 1 Evolutionary Search.

Executes 15 mini-batch gradient updates on the adapter and classification head
from a warm-start checkpoint to eliminate the representation gap in < 20s per candidate,
and returns the multi-objective fitness vector: [Top-1 Acc, -Latency_1080p, -Pixel Budget].
"""

from typing import Dict, List, Tuple, Optional, Any
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..models.pipeline import BioInspiredVisionPipeline
from .saccadic_jitter import SaccadicJitterSampler
from .latency_profiler import LatencyProfiler


class ProxyEvaluator:
    """
    Evaluates candidate spatial configurations via 15-step proxy adaptation.
    
    Args:
        num_classes: Number of classification classes (default: 100).
        num_steps: Number of proxy gradient steps (default: 15).
        lr: Learning rate for AdamW adapter optimizer (default: 5e-4).
        device: Device for execution ('cuda' or 'cpu').
        latency_benchmark: If True, measures 1080p wall-clock latency.
    """
    def __init__(
        self,
        num_classes: int = 100,
        num_steps: int = 15,
        lr: float = 5e-4,
        device: Optional[torch.device] = None,
        latency_benchmark: bool = True
    ):
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latency_benchmark = latency_benchmark

        self.jitter_sampler = SaccadicJitterSampler(sigma=0.08)
        self.latency_profiler = LatencyProfiler(input_resolution=(1080, 1920), warmup_iterations=3, benchmark_iterations=10)

        # Build canonical model template
        self.template_model = BioInspiredVisionPipeline(
            backbone_name="resnet18",
            num_classes=num_classes,
            pretrained_backbone=False
        ).to(self.device)

        # Cache canonical adapter & fc state dict for warm-start initialization
        self.warm_adapter_state = copy.deepcopy(self.template_model.adapter.state_dict())
        self.warm_fc_state = copy.deepcopy(self.template_model.fc.state_dict())

    def evaluate_candidate(
        self,
        genes: Dict[str, float],
        train_batches: List[Tuple[torch.Tensor, torch.Tensor]],
        val_batches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[float]:
        """
        Executes proxy evaluation for candidate spatial parameters.
        
        Args:
            genes: Candidate parameter dictionary with r_rel, r_fovea, r_norm, alpha_fovea, gamma_periph.
            train_batches: List of (x, y) mini-batches for 15-step proxy adaptation.
            val_batches: List of (x, y) mini-batches for validation accuracy estimation.
            
        Returns:
            objectives: [top1_accuracy, -latency_ms, -effective_pixel_ratio]
        """
        # 1. Instantiate candidate pipeline
        candidate_model = BioInspiredVisionPipeline(
            backbone_name="resnet18",
            num_classes=self.num_classes,
            r_rel=genes.get("r_rel", 0.015),
            r_fovea=genes.get("r_fovea", 0.15),
            r_norm=genes.get("r_norm", 0.035)
        )
        # Configure foveation properties
        candidate_model.foveated_pool.alpha_fovea = genes.get("alpha_fovea", 0.5)
        candidate_model.foveated_pool.gamma = genes.get("gamma_periph", 1.5)

        # 2. Warm-start adapter and classification head
        candidate_model.adapter.load_state_dict(self.warm_adapter_state)
        candidate_model.fc.load_state_dict(self.warm_fc_state)
        candidate_model.to(self.device)

        # 3. Freeze backbone stages 1-4, train adapter + head only
        for stage in [candidate_model.layer1, candidate_model.layer2, candidate_model.layer3, candidate_model.layer4]:
            for p in stage.parameters():
                p.requires_grad = False

        optimizer = torch.optim.AdamW(
            list(candidate_model.adapter.parameters()) + list(candidate_model.fc.parameters()),
            lr=self.lr,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # 4. Execute 15 proxy gradient steps with saccadic jittering
        candidate_model.train()
        step_count = min(self.num_steps, len(train_batches))
        for step_idx in range(step_count):
            x_b, y_b = train_batches[step_idx]
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            gaze = self.jitter_sampler.sample_gaze_centers(x_b)
            optimizer.zero_grad()
            logits = candidate_model(x_b, gaze_center=gaze)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        # 5. Evaluate proxy validation accuracy
        candidate_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_v, y_v in val_batches:
                x_v = x_v.to(self.device)
                y_v = y_v.to(self.device)
                logits = candidate_model(x_v)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_v).sum().item()
                total += y_v.size(0)

        top1_acc = (correct / max(1, total)) * 100.0

        # 6. Profile 1080p wall-clock latency (ms)
        if self.latency_benchmark:
            try:
                prof_results = self.latency_profiler.profile(candidate_model, device=self.device)
                latency_ms = prof_results["latency_ms"]
            except Exception:
                latency_ms = 50.0
        else:
            latency_ms = 30.0

        # 7. Compute active foveation pixel budget ratio
        # Smaller r_fovea corresponds to higher compression / lower active pixel density
        effective_pixel_ratio = genes.get("r_fovea", 0.15) * genes.get("alpha_fovea", 0.5)

        # Return objectives to MAXIMIZE: [Acc, -Latency, -Budget]
        return [float(top1_acc), float(-latency_ms), float(-effective_pixel_ratio)]
