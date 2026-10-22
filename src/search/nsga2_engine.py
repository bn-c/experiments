"""
NSGA-II Evolutionary Search Engine.

Implements the Non-dominated Sorting Genetic Algorithm II (Deb et al., 2002):
- Fast non-dominated sorting into Pareto fronts.
- Crowding distance computation for diversity preservation.
- Crowded comparison operator & binary tournament selection.
- Simulated Binary Crossover (SBX) and Polynomial Mutation within parameter bounds.
"""

from typing import List, Tuple, Dict, Any, Callable, Optional
import random
import copy
import numpy as np


class Individual:
    """Represents a single candidate in the evolutionary population."""
    def __init__(self, genes: Dict[str, float]):
        self.genes = copy.deepcopy(genes)
        self.objectives: List[float] = []  # All objectives formulated to be MAXIMIZED
        self.rank: int = 0
        self.crowding_distance: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        """Returns True if self dominates other (no worse in all, strictly better in at least one)."""
        and_condition = True
        or_condition = False
        for obj_a, obj_b in zip(self.objectives, other.objectives):
            if obj_a < obj_b:
                and_condition = False
                break
            if obj_a > obj_b:
                or_condition = True
        return and_condition and or_condition


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """Sorts population into hierarchical Pareto fronts (Rank 0, 1, 2, ...)."""
    fronts: List[List[Individual]] = [[]]
    domination_counts: Dict[int, int] = {}
    dominated_sets: Dict[int, List[Individual]] = {i: [] for i in range(len(population))}

    for i, p in enumerate(population):
        domination_counts[i] = 0
        for j, q in enumerate(population):
            if p.dominates(q):
                dominated_sets[i].append(q)
            elif q.dominates(p):
                domination_counts[i] += 1
        if domination_counts[i] == 0:
            p.rank = 0
            fronts[0].append(p)

    current_rank = 0
    while len(fronts[current_rank]) > 0:
        next_front: List[Individual] = []
        for p in fronts[current_rank]:
            p_idx = population.index(p)
            for q in dominated_sets[p_idx]:
                q_idx = population.index(q)
                domination_counts[q_idx] -= 1
                if domination_counts[q_idx] == 0:
                    q.rank = current_rank + 1
                    next_front.append(q)
        current_rank += 1
        fronts.append(next_front)

    if len(fronts[-1]) == 0:
        fronts.pop()
    return fronts


def calculate_crowding_distance(front: List[Individual]):
    """Assigns crowding distances to individuals within a single Pareto front."""
    L = len(front)
    if L == 0:
        return
    for ind in front:
        ind.crowding_distance = 0.0

    num_objectives = len(front[0].objectives)
    for m in range(num_objectives):
        front.sort(key=lambda ind: ind.objectives[m])
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")
        obj_range = front[-1].objectives[m] - front[0].objectives[m]
        if obj_range == 0:
            continue
        for i in range(1, L - 1):
            front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / obj_range


def crowded_comparison(ind1: Individual, ind2: Individual) -> int:
    """Crowded comparison operator: lower rank wins; if tied, higher crowding distance wins."""
    if ind1.rank < ind2.rank:
        return -1
    elif ind1.rank > ind2.rank:
        return 1
    elif ind1.crowding_distance > ind2.crowding_distance:
        return -1
    elif ind1.crowding_distance < ind2.crowding_distance:
        return 1
    return 0


def binary_tournament(population: List[Individual]) -> Individual:
    """Picks two individuals at random and returns the winner under crowded comparison."""
    a = random.choice(population)
    b = random.choice(population)
    return a if crowded_comparison(a, b) <= 0 else b


def sbx_crossover(
    parent1: Individual,
    parent2: Individual,
    bounds: Dict[str, Tuple[float, float]],
    eta_c: float = 15.0,
    prob: float = 0.8
) -> Tuple[Individual, Individual]:
    """Simulated Binary Crossover (SBX)."""
    child1_genes = copy.deepcopy(parent1.genes)
    child2_genes = copy.deepcopy(parent2.genes)

    if random.random() <= prob:
        for key in bounds:
            x1, x2 = parent1.genes[key], parent2.genes[key]
            low, high = bounds[key]
            if abs(x1 - x2) > 1e-10:
                y1 = min(x1, x2)
                y2 = max(x1, x2)
                rand = random.random()

                beta = 1.0 + (2.0 * (y1 - low) / (y2 - y1 + 1e-10))
                alpha = 2.0 - beta ** -(eta_c + 1.0)
                if rand <= (1.0 / alpha):
                    beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                c1 = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))

                beta = 1.0 + (2.0 * (high - y2) / (y2 - y1 + 1e-10))
                alpha = 2.0 - beta ** -(eta_c + 1.0)
                if rand <= (1.0 / alpha):
                    beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                else:
                    beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                c2 = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))

                child1_genes[key] = float(np.clip(c1, low, high))
                child2_genes[key] = float(np.clip(c2, low, high))

    return Individual(child1_genes), Individual(child2_genes)


def polynomial_mutation(
    ind: Individual,
    bounds: Dict[str, Tuple[float, float]],
    eta_m: float = 20.0,
    prob: float = 0.2
) -> Individual:
    """Polynomial Mutation operator."""
    genes = copy.deepcopy(ind.genes)
    for key in bounds:
        if random.random() <= prob:
            x = genes[key]
            low, high = bounds[key]
            delta1 = (x - low) / (high - low + 1e-10)
            delta2 = (high - x) / (high - low + 1e-10)
            rand = random.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if rand < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                delta_q = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                delta_q = 1.0 - (val ** mut_pow)
            x = x + delta_q * (high - low)
            genes[key] = float(np.clip(x, low, high))
    return Individual(genes)


class NSGAIIEngine:
    """
    NSGA-II Optimization Orchestrator.
    
    Args:
        bounds: Dictionary mapping parameter names to (min, max) bounds.
        evaluator_fn: Callable taking candidate genes and returning list of objective scores (to maximize).
        population_size: Number of individuals in population (default: 32).
        generations: Number of evolutionary generations (default: 20).
    """
    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        evaluator_fn: Callable[[Dict[str, float]], List[float]],
        population_size: int = 32,
        generations: int = 20,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.2
    ):
        self.bounds = bounds
        self.evaluator_fn = evaluator_fn
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population: List[Individual] = []

    def initialize_population(self):
        """Randomly initializes individuals uniformly across parameter bounds."""
        self.population = []
        for _ in range(self.population_size):
            genes = {
                k: float(random.uniform(b[0], b[1]))
                for k, b in self.bounds.items()
            }
            ind = Individual(genes)
            ind.objectives = self.evaluator_fn(ind.genes)
            self.population.append(ind)

    def step_generation(self) -> List[Individual]:
        """Executes a single generation step: reproduction, evaluation, non-dominated sorting."""
        # 1. Offspring reproduction
        offspring: List[Individual] = []
        while len(offspring) < self.population_size:
            p1 = binary_tournament(self.population)
            p2 = binary_tournament(self.population)
            c1, c2 = sbx_crossover(p1, p2, self.bounds, prob=self.crossover_prob)
            c1 = polynomial_mutation(c1, self.bounds, prob=self.mutation_prob)
            c2 = polynomial_mutation(c2, self.bounds, prob=self.mutation_prob)
            c1.objectives = self.evaluator_fn(c1.genes)
            c2.objectives = self.evaluator_fn(c2.genes)
            offspring.extend([c1, c2])

        offspring = offspring[:self.population_size]

        # 2. Combine parent and offspring (2N individuals)
        combined = self.population + offspring

        # 3. Non-dominated sort
        fronts = fast_non_dominated_sort(combined)

        # 4. Fill next population with elitism and crowding distance
        next_population: List[Individual] = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
            else:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                needed = self.population_size - len(next_population)
                next_population.extend(front[:needed])
                break

        self.population = next_population
        return self.get_pareto_front()

    def get_pareto_front(self) -> List[Individual]:
        """Returns current Rank 0 non-dominated Pareto optimal individuals."""
        fronts = fast_non_dominated_sort(self.population)
        return fronts[0] if len(fronts) > 0 else []
