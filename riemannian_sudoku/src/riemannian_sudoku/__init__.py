"""Riemannian Sudoku spike on S⁸₊ with axial, cross-coupled, or bipartite dynamics."""

from riemannian_sudoku.model import RiemannianSudokuSolver, build_solver, init_sphere_state, init_t2_gauge
from riemannian_sudoku.solver import BipartiteSnTmSolver

__all__ = [
    'RiemannianSudokuSolver',
    'BipartiteSnTmSolver',
    'build_solver',
    'init_sphere_state',
    'init_t2_gauge',
]
