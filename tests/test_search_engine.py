"""
Unit tests for NSGA-II evolutionary search engine, saccadic jittering, and latency profiling.
"""

import sys
from pathlib import Path
import unittest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.search import (
    NSGAIIEngine,
    Individual,
    fast_non_dominated_sort,
    SaccadicJitterSampler,
    estimate_image_centroid,
    LatencyProfiler
)


class TestSearchEngine(unittest.TestCase):
    def test_fast_non_dominated_sort(self):
        """Tests Pareto front dominance sorting on synthetic multi-objective individuals."""
        # ind1 strictly dominates ind2
        ind1 = Individual({"x": 1.0})
        ind1.objectives = [10.0, 5.0, 2.0]

        ind2 = Individual({"x": 2.0})
        ind2.objectives = [8.0, 4.0, 1.0]

        # ind3 is trade-off with ind1
        ind3 = Individual({"x": 3.0})
        ind3.objectives = [6.0, 10.0, 3.0]

        pop = [ind1, ind2, ind3]
        fronts = fast_non_dominated_sort(pop)

        # Front 0 should contain ind1 and ind3; Front 1 should contain ind2
        self.assertEqual(len(fronts), 2)
        self.assertIn(ind1, fronts[0])
        self.assertIn(ind3, fronts[0])
        self.assertIn(ind2, fronts[1])
        print("[TEST] Fast non-dominated sorting verified successfully!")

    def test_nsga2_step_generation(self):
        """Verifies NSGA-II generation lifecycle with crossover and polynomial mutation."""
        bounds = {
            "r_rel": (0.005, 0.030),
            "r_fovea": (0.050, 0.300),
            "alpha": (0.2, 0.8)
        }

        # Dummy multi-objective evaluator: maximize sphere functions
        def dummy_eval(genes):
            return [
                genes["r_rel"] * 100.0,
                -genes["r_fovea"] * 50.0,
                genes["alpha"] * 10.0
            ]

        engine = NSGAIIEngine(
            bounds=bounds,
            evaluator_fn=dummy_eval,
            population_size=8,
            generations=2
        )
        engine.initialize_population()
        self.assertEqual(len(engine.population), 8)

        pareto_front = engine.step_generation()
        self.assertEqual(len(engine.population), 8)
        self.assertGreater(len(pareto_front), 0)
        print("[TEST] NSGA-II step generation and elitist selection verified!")

    def test_saccadic_jitter_bounds(self):
        """Verifies saliency estimation and ocular fixation coordinate bounds."""
        sampler = SaccadicJitterSampler(sigma=0.08, clip_bound=0.70)
        dummy_imgs = torch.rand(4, 3, 64, 64)

        centroids = estimate_image_centroid(dummy_imgs)
        self.assertEqual(centroids.shape, (4, 2))
        self.assertTrue(torch.all(centroids >= -1.0) and torch.all(centroids <= 1.0))

        gaze = sampler.sample_gaze_centers(dummy_imgs)
        self.assertEqual(gaze.shape, (4, 2))
        self.assertTrue(torch.all(gaze >= -0.70) and torch.all(gaze <= 0.70))
        print("[TEST] Saccadic jittering coordinates and bounds verified!")

    def test_latency_profiler(self):
        """Verifies LatencyProfiler outputs valid metrics."""
        dummy_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 10)
        )
        profiler = LatencyProfiler(input_resolution=(64, 64), warmup_iterations=2, benchmark_iterations=5)
        results = profiler.profile(dummy_model, device=torch.device("cpu"))

        self.assertIn("latency_ms", results)
        self.assertIn("fps", results)
        self.assertGreater(results["fps"], 0.0)
        print(f"[TEST] LatencyProfiler verified (dummy CPU latency: {results['latency_ms']:.2f} ms)!")


if __name__ == "__main__":
    unittest.main()
