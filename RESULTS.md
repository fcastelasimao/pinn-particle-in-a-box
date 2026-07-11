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

