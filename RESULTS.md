# RESULTS.md — verified numbers (source of truth)

Every number quoted in README.md and post/post_draft.md must trace back to a line in
this file, which in turn traces to a `results_raw/*.json` produced by an actual run
under this repo (see IMPLEMENTATION_PLAN.md). Nothing here is estimated or carried over
from a previous run of a different script version.

<!-- Filled in as Phase 2 tasks complete. See IMPLEMENTATION_PLAN.md section 4. -->

## Task 2.1 — IC formulation decision (exp08_ic_formulation.py)

`px=2.0`, 5 seeds (0-4), 5000 epochs, 2000 collocation points. Raw:
`results_raw/exp08_ic_formulation.json`.

| Formulation | Mean error | Std |
|---|---|---|
| Hard IC (ψ = ψ₀ + t·net) | 77.24% | 0.02% |
| Soft IC, w_ic=1 | **2.17%** | 0.18% |
| Soft IC, w_ic=10 | 2.41% | 0.37% |
| Soft IC, w_ic=100 | 8.02% | 0.53% |
| Soft IC, w_ic=1000 (original 2D weight) | 15.63% | 1.61% |
| Soft IC, w_ic=10000 | 19.98% | 2.26% |

**Decision: soft IC, w_ic=1, adopted for exp06/exp07/exp09 and the Task 2.2 2D re-run.**

This is the opposite of the plan's stated hypothesis (hard IC was expected to be ≤ best
soft IC, since it satisfies the IC exactly and removes a hyperparameter) — and error
*increases* monotonically with more IC weight, meaning the *least*-constrained soft
variant tested is the best one. Per the plan's tripwire ("if some soft-IC weight DOES
substantially rescue the dynamics, stop and flag to the owner"), this was flagged and a
bounded follow-up investigation was run before adopting it (owner-approved). Findings
(raw: `results_raw/diagnostics/ic_investigation.json`, figures:
`figures/diagnostics/loss_curves_hard_vs_soft.png`,
`figures/diagnostics/hard_ic_seed_comparison.png`):

- **Not a px=2-specific bug.** Hard-IC mean error rises smoothly with momentum:
  45.9% (px=1.0) → 63.3% (px=1.5) → 77.2% (px=2.0).
- **Hard IC "solves" its own training objective almost exactly.** Final PDE-residual
  training loss: hard IC 7.7×10⁻⁸ vs soft IC 1.9×10⁻⁴ — four orders of magnitude
  *lower* for hard IC, despite hard IC being four orders of magnitude *worse* on the
  physical accuracy metric.
- **The failure is a reproducible attractor, not seed noise.** All 5 original seeds
  land within 0.02% of each other (77.23–77.25%); a follow-up 3-seed shape comparison
  at t=1.0 shows all three predictions are visually identical.
- **The failure shape is "the packet doesn't move."** At t=1.0 the true wave packet has
  propagated to x≈2; the hard-IC prediction is still centered at x≈0, narrower and
  taller than truth. This is the same qualitative signature flagged for the original
  px=1 exp03 result and cited as evidence for mechanism B (causality violation) in
  IMPLEMENTATION_PLAN.md — i.e. the hard-IC parameterization (ψ=ψ₀+t·net) appears
  susceptible to converging onto a spurious, non-propagating solution branch that still
  satisfies the (linear, so non-unique-under-residual-sampling) PDE almost exactly.
- **Not independently confirmed further** (e.g. by testing whether exp09's causal-training
  fix also rescues hard IC) — that would be additional scope beyond this investigation;
  flagged as a plausible but unverified mechanistic explanation, not a proven one.

## Task 2.2 — 2D re-verification under the adopted formulation

`experiments/exp02_pinn2d_comparison.py`, N=100 grid, 2000 epochs, px=-15 (the original
2D headline setup). Single run each (2D comparison stays single-seed per the plan's
seeds policy — expensive and qualitative). Raw logs: `/tmp/exp02_adopted_log.txt`,
`/tmp/exp02_original_log.txt` (not committed; rerun via the commands below to
reproduce). Figures: `figures/*_soft_w1.png` / `.gif` (adopted) and
`figures/*_soft_w1000.png` / `.gif` (original weight).

| Formulation | Mean rel. error | Final rel. error | PINN train+infer | Slowdown vs classical |
|---|---|---|---|---|
| Soft IC, w_ic=1 (**adopted**) | 99.96% | 99.82% | 267.2s | ~548x |
| Soft IC, w_ic=1000 (original) | 99.78% | 99.37% | 292.6s | ~155x |

**The headline failure holds under the adopted (measured-best) formulation — it fails
under both.** Reproduce:
```
uv run experiments/exp02_pinn2d_comparison.py --formulation soft --w-ic 1.0 --epochs 2000
uv run experiments/exp02_pinn2d_comparison.py --formulation soft --w-ic 1000.0 --epochs 2000
```

This is consistent with, not contradicted by, the 1D momentum sweep (Task 2.3): the 2D
setup uses px=-15, far beyond the px=4 already showing 44% error in 1D, so near-total
failure here is exactly what the spectral-bias trend predicts extrapolated to this
momentum. The `PINNSolverHardIC` variant added to `qbox/pinn2d.py` for a possible
hard-IC 2D re-run was NOT used for the headline claim — the 1D investigation
(Task 2.1) found hard IC converges to a worse, non-propagating attractor, so re-running
the weaker formulation in 2D would not have strengthened anything.

Note on the slowdown factor: the classical solver's wall-clock time was noisy across
runs (0.49s vs 1.88s for the identical deterministic computation) due to background
system load during these runs (see session notes) -- the ~150-550x range should be read
as "two-to-three orders of magnitude slower," not a precise multiplier. PINN training
time itself (267s vs 293s) is the more reliable number.

## Task 2.3 — Momentum sweep (exp06_momentum_sweep.py)

Adopted formulation (soft IC, w_ic=1), 5 seeds, 5000 epochs. Raw:
`results_raw/exp06_momentum_sweep.json`. Figure: `figures/exp06_momentum_sweep.png`.

| px | Mean error | Std |
|---|---|---|
| 0.5 | 0.57% | 0.07% |
| 1.0 | 0.75% | 0.08% |
| 2.0 | 2.17% | 0.18% |
| 4.0 | **44.36%** | 1.95% |

**Monotonic increase: confirmed.** Error stays under 1% through px=1, then grows
steeply — 3x from px=1→2, another 20x from px=2→4. This is a much cleaner and more
dramatic version of the hypothesized spectral-bias trend than the original single-point
(px=1) result suggested, and it's now measured against the strongest (adopted) IC
formulation rather than the confounded one.

## Task 2.4 — Norm/unitarity constraint (exp07_norm_constraint.py)

Adopted formulation (soft IC, w_ic=1), 5 seeds, 5000 epochs, λ=1.0 headline. Raw:
`results_raw/exp07_norm_constraint.json`. Figure: `figures/exp07_norm_constraint.png`.

| px | Baseline | +norm constraint (λ=1) |
|---|---|---|
| 1.0 | 0.75% ± 0.08% | 3.01% ± 0.37% (worse) |
| 4.0 | 44.36% ± 1.95% | 70.28% ± 7.04% (worse) |

λ sensitivity (px=4, 3 seeds): λ=0.1 → 35.40% ± 1.67% (slightly better than baseline);
λ=1.0 → 70.28% (worse); λ=10.0 → 94.78% ± 3.57% (much worse). The conclusion is
λ-dependent, not just a headline-λ artifact.

**Result: the constraint does what it's designed to do, but that's not enough to help
overall.** The N(t) curves (`figures/exp07_norm_constraint.png`, right column) show the
mechanism working exactly as hypothesized at px=4: baseline norm genuinely decays
(N(t): 0.76 → ~0.0 by t=1 — real amplitude collapse), and the constraint holds it much
closer to 1 (0.93–0.99 throughout). But the |ψ|² snapshots (left column) show the
constrained network trades fine oscillatory/interference structure for that norm
compliance — it collapses the true multi-peaked pattern into a single broad hump. Net
effect: worse overall accuracy despite fixing the specific pathology (amplitude decay)
it targeted. At px=1 there's no meaningful amplitude decay to fix under the adopted
formulation (baseline N(t) is already ≈0.997–1.001), so the constraint only adds a
competing loss term and purely hurts.

This revises the original hypothesis ("fixes amplitude, doesn't fix phase — a clean
separation") into something more specific: amplitude decay (mechanism A) is real and
fixable at high momentum, but fixing it via an added penalty term costs representational
capacity elsewhere, so it doesn't rescue accuracy net. A smarter integration (e.g. a
hard normalization layer instead of a soft penalty) might avoid that trade-off — flagged
as future work, not attempted here (out of scope).

## Task 2.5 — Causal training (exp09_causal_training.py)

Adopted formulation (soft IC, w_ic=1), 5 seeds, 5000 epochs, ε=10 (no adjustment
needed). Raw: `results_raw/exp09_causal_training.json`. Figure:
`figures/exp09_causal_training.png`.

| px | Baseline | +causal training | Substantially improved? |
|---|---|---|---|
| 1.0 | 0.75% ± 0.08% | 0.76% ± 0.06% | No |
| 4.0 | 44.36% ± 1.95% | 45.44% ± 2.61% | No |

**Result: causal training has no effect either way, under the adopted formulation.**
This is a different (and cleaner) outcome than the original hypothesis anticipated,
because the premise changed: causal training was meant to fix a "packet doesn't move"
propagation failure at px=1, but that failure was specific to the (rejected) hard-IC
formulation (see Task 2.1) — under soft IC, px=1 already has essentially no error
(0.75%) to fix. At px=4, where real error remains, causal training doesn't move the
needle, consistent with that residual error being spectral bias (mechanism C, confirmed
directly by Task 2.3) rather than a causality artifact.

## Task 2.6 — Multi-seed error bars: exp04 (eigenstates), exp05 (inverse problem)

5 seeds (0-4), `experiments/exp04_eigenstates.py --seed N` / `exp05_inverse_potential.py --seed N`.
Raw: `results_raw/exp04_eigenstates.json`, `results_raw/exp05_inverse_potential.json`.

**exp04 — worst-case PINN energy error across both problems/states, per seed:**
0.59%, 1.31%, 1.24%, 1.51%, 1.39% → **mean 1.21% ± 0.32%** (population std), worst single
seed 1.51%. (Original single-seed claim was "<1.5%" — the worst seed comes in just above
that threshold; reporting honestly rather than rounding down.)

**exp04 — worst-case PINN ψ L2 error, per seed:** 1.84%, 1.78%, 2.09%, 2.07%, 2.17% →
**mean 1.99% ± 0.16%**.

**exp05 — inverse-problem V(x) relative L2 error, per seed:** 13.75%, 12.47%, 21.22%,
17.14%, 10.36% → **mean 14.99% ± 3.82%**. Seed 2 (21.22%) is a real outlier, not
cherry-picked away — noisy-data inverse recovery has real run-to-run variance with only
40 sparse observations. Committed figure (`figures/exp05_inverse_potential.png`) is seed
0 (13.75%), close to the mean.

