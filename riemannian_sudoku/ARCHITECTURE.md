# Architecture: Product \(S^8 \times S^{54} \leftrightarrow \mathbb{T}^{16}\) (+ global \(\mathbb{T}^{16}\))

**Tip on `main`:** `--mixer bipartite_sn_tm` (`BipartiteSnTmSolver`) with
**`use_global_torus=True`** (default). Desk run W&B **`f7fv0ymk`**.

| Arm | Flag | Widths | ≈params |
|-----|------|--------|---------|
| **Tip (Phase 2)** | default / `--use-global-torus` | `H_Y=304`, `H_Z=96`, `H_X=412` + `w_x_chart` | **~681 877** |
| Phase 1 ablation | `--no-use-global-torus` | `H_Y=320`, `H_X=428` + `w_x_chart` | ~680 565 |

`w_x_chart` = `Linear(9,18)` (+180): learned S^8-only μ/σ chart (replaces rotation-blind Gram).

### Tip metrics (`f7fv0ymk`)

| Split | Metric | Value |
|-------|--------|------:|
| val@10k K=8 | blank | **0.5375** |
| test n=2048 K=8 | blank / exact | 0.536 / 0.026 |
| test K=32 | blank / exact | 0.523 / 0.069 |
| test K=64 | blank / exact | 0.512 / **0.075** (peak) |
| test K=128 | blank / exact | 0.504 / 0.040 (soft dip) |
| test K=512 | blank / exact | 0.504 / 0.061 |

Historical blank@8 bar: `yr1h706t` y0-logit (**0.5479**). Tip success = high-K exact
climb without a K=64 crash; blank@8 is secondary.

### Tip K=16 final-CE ablation (`d6ffqu57`, partial 50k)

Same tip weights recipe, **train K=16**, **final-step CE** (no `--supervise-all-steps`),
user-stopped ~38k. best blank **0.5861** @ step **35500** (K_eval=16). High-K
zeroshot collapses (blank≈0.21 / exact=0 for K≥64). **Not a tip replacement** —
keeps blank@16 local max only. Desk: `scripts/bipartite_sn_tm_tip_k16_finalce_50k.sh`.

### Tip K=16 terminal-window CE + velocity brake (`ca48ruu4`, 10k)

Same tip chart, **K=16**, CE only on last 4 steps (`--supervise-terminal-window 4`;
\(w_k=0\) for \(k=1..12\)), plus
\(\mathcal{L}_{\mathrm{brake}}=\lambda_{\mathrm{brake}}(\|\omega_{y,K}\|^2+\|\omega_{z,K}\|^2+\|v_{x,K}\|^2+\|v_{c,K}\|^2)\)
with \(\lambda_{\mathrm{brake}}=0.01\) via `--gauge-kinetic-weight`. Desk cooled 10k:
`scripts/bipartite_sn_tm_tip_terminal_brake_10k.sh`. best blank **0.5685** @K=16;
K_eval peak blank/exact at K=24 (**0.5726** / **0.098**); K≥64 exact=0 but blank
floor ~0.29 (vs final-CE ~0.21). Softens overshoot vs `d6ffqu57`; **not a tip
promote** (tip `f7fv0ymk` keeps high-K exact).

### Tip K=32 landing-window CE + kinetic (50k)

Same tip chart, **K=32**, CE only on \(k\in\{29..32\}\) (`--supervise-terminal-window 4`),
plus landing-window kinetic
\(\mathcal{L}_{\mathrm{kin}}=\lambda_{\mathrm{kin}}\sum_{k=K-3}^{K}(\|\omega_{y,k}\|^2+\|\omega_{z,k}\|^2+\|v_{c,k}\|^2)\)
with \(\lambda_{\mathrm{kin}}=0.15\) (no \(v_x\); digits settle via CE). Desk cooled 50k
compile-on: `scripts/bipartite_sn_tm_tip_landing_kin_k32_50k.sh`. W&B
`bipartite-sn-tm-tip-landing-kin-k32-s50000`. Undamped K_eval vs `ca48ruu4`.

---

## 1. What the model is

A weight-tied **leapfrog** on a **Riemannian product**:

| Side | Manifold | State | Role |
|------|----------|-------|------|
| Decision | \(S^8 \subset \mathbb{R}^9\) | \(x \in \mathbb{R}^{B\times 81\times 9}\), \(\|x\|_2=1\) | Digits + North-Pole blank |
| Coordination | \(S^{54} \subset \mathbb{R}^{55}\) | \(c \in \mathbb{R}^{B\times 81\times 55}\), \(\|c\|_2=1\) | Scratch / messaging |
| Houses | \(\mathbb{T}^{16}\) | \(y \in \mathbb{R}^{B\times 27\times 16}\) | Row/col/box phase |
| Global (tip) | \(\mathbb{T}^{16}\) | \(z \in \mathbb{R}^{B\times 1\times 16}\) | Board-wide sync |

27 houses = 9 rows + 9 cols + 9 boxes. Each cell belongs to exactly 3 houses.

```
puzzle ──init──► (x₀ on S^8, c₀ on S^{54}, y₀=π/4, z₀)
                    │
        ┌───────────┴───────────┐  ×K
        │  ω_y, ω_z ← vf.omega    │
        │  y ← y+ω_y; z ← z+ω_z   │
        │  u_x, u_c ← vf.ambient_u│
        │  x ← Exp(u_x)  (clues frozen to x₀)
        │  c ← Exp(u_c)  (clues frozen to c₀; blanks free)
        └───────────┬───────────┘
                    ▼
              logits ← scale · (x @ P₉ᵀ)   # x already unit; no slice-norm
```

---

## 2. Geometry & static buffers

[`manifold_embed.py`](src/riemannian_sudoku/manifold_embed.py):

- `DECISION_DIM=9`, `COORD_DIM=55`, `D=64` (= concat ambient for `'s'`).
- Blank North Pole: `e_8` on \(S^8\) (`BLANK_COORD=8`).
- Coordination rest: `e_6` on \(S^{54}\) (`COORD_REST=6`); PE into coords `0..5`.
- `simplex_etf_poles()` → shape **`(9, 9)`** (Householder; `poles[:,8]≈0`).
- Clifford house embed: \(y_{\mathrm{emb}} = M^{-1/2}[\cos y,\sin y]\), \(M=16\).
- `w_x_chart`: `Linear(9,18)` S^8-only chart (never reads \(c\)); house-pooled μ/σ → 36D.

Poles used for **init** (clue \(x\)) and **readout** only — **never** inside `CoupledVectorField`.

---

## 3. State initialization

`init_bipartite_state(..., use_global_torus=True)` (tip):

1. Blanks: \(x \leftarrow e_8\); clues: \(x \leftarrow p_d\).
2. All cells: \(c \leftarrow \mathrm{normalize}(e_6 + \alpha\cdot\mathrm{PE}_{0:5})\).
3. \(y \leftarrow \pi/4\); \(z \leftarrow \pi/4\) shape `(B,1,16)`.
4. Return `(x, c, y, z, is_clue)`.

Clue cells stay frozen on both \(S^8\) (\(x\equiv x_0\)) and \(S^{54}\) (\(c\equiv c_0\)); blank cells move freely on both spheres.

---

## 4. Vector field ([`coupled_vf.py`](src/riemannian_sudoku/coupled_vf.py))

### 4.1 `omega` — product cells → houses (+ global)

Per house: \(\mu/\sigma\) of \(x\) (18D) + \(\mu/\sigma\) of \(c\) (110D) + 36D learned chart
\(\phi=\mathrm{SiLU}(W_{\mathrm{chart}} x)\), then \([\mu(\phi_h),\sigma(\phi_h)]\) + house-type (3)
+ \(y_{\mathrm{emb}}\) (32) + \(z_{\mathrm{emb}}\) (32) → `vf_y` → \(\omega_y\).
`in_y` = **231** tip (199 with `--no-use-global-torus`).

Global: \([\mathrm{mean}_h y_{\mathrm{emb}},\;\mathrm{std}_h y_{\mathrm{emb}},\; z_{\mathrm{emb}}]\) (96D) → `vf_z` → \(\omega_z\).

### 4.2 `ambient_u` — houses → product cells

`cell_in` = \([x,c,\mathrm{pe},\mathrm{is\_clue}]\) (71D); `house_in` = 3×\(y_{\mathrm{emb}}\) + \(z_{\mathrm{emb}}\) (128).
FiLM → body → \(u\in\mathbb{R}^{64}\) split into \(u_x,u_c\).

### 4.3 Geodesic ([`geodesic.py`](src/riemannian_sudoku/geodesic.py))

Hazard-locked Exp; `x0`/`is_clue` optional (skip freeze when omitted). Tip freezes both \(x\) and \(c\) for clues.

---

## 5. Loss & diagnostics ([`solver.py`](src/riemannian_sudoku/solver.py))

- Blank CE (\(\gamma=0.9\)); optional `--supervise-all-steps` linear ramp; optional
  `--supervise-terminal-window N` (uniform CE on last \(N\) steps only; overrides ramp).
- Landing-window kinetic (`N>0`): `--gauge-kinetic-weight` → \(\lambda_{\mathrm{kin}}\) on
  \(\sum_{k=K-N+1}^{K}(\|\omega_{y,k}\|^2+\|\omega_{z,k}\|^2+\|v_{c,k}\|^2)\) (no \(v_x\)).
  Legacy `N=0`: single-step brake including \(\|v_{x,K}\|^2\).
- `scratch_energy` = \((1 - c_{\ldots,6}).\mathrm{mean()}\) (rest-axis displacement; **not** \(\|c\|^2\equiv 1/55\)).
- `blank_np` = mean \(x_8^2\); `claim_energy` = mean \(\mathrm{relu}(x P^\top)^2\) (watch-only).
- `omega_sq_1` / `omega_sq_k` from **\(\omega_y\)**; tip also `omega_z_sq_*`;
  `v_x_sq_k` / `v_c_sq_k` / `brake_loss`; landing also `landing_omega_y_sq` /
  `landing_v_c_sq` (window means).

---

## 6. Desk scripts

- Tip 10k: [`scripts/bipartite_sn_tm_product_chart_global_t16_10k.sh`](scripts/bipartite_sn_tm_product_chart_global_t16_10k.sh)
- Tip K16 final-CE 50k: [`scripts/bipartite_sn_tm_tip_k16_finalce_50k.sh`](scripts/bipartite_sn_tm_tip_k16_finalce_50k.sh)
- Tip K16 terminal-window + brake 10k: [`scripts/bipartite_sn_tm_tip_terminal_brake_10k.sh`](scripts/bipartite_sn_tm_tip_terminal_brake_10k.sh)
- Tip K32 landing-window + kinetic 50k: [`scripts/bipartite_sn_tm_tip_landing_kin_k32_50k.sh`](scripts/bipartite_sn_tm_tip_landing_kin_k32_50k.sh)
- Full train: [`scripts/full_train.sh`](scripts/full_train.sh)
- Post-hoc K_eval: [`scripts/bipartite_sn_tm_keval.py`](scripts/bipartite_sn_tm_keval.py)
- Post-hoc solved strat: [`scripts/bipartite_sn_tm_solved_strat.py`](scripts/bipartite_sn_tm_solved_strat.py)

Superseded A2 / Gram / Phase-1 / geo / cross desks: closed `origin/cursor/*`
branches only (not on `main`). No legacy-mixer resume scripts on tip.
