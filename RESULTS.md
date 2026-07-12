# RESULTS.md — verified numbers (source of truth)

Every number quoted in README.md and post/post_draft.md must trace back to a line in
this file, which in turn traces to a `results_raw/*.json` produced by an actual run
under this repo (see IMPLEMENTATION_PLAN.md). Nothing here is estimated or carried over
from a previous run of a different script version.

<!-- Filled in as Phase 2 tasks complete. See IMPLEMENTATION_PLAN.md section 4. -->

## Task 2.1 — IC formulation decision (exp08_ic_formulation.py)

**REVISED.** The original decision below (soft IC, w_ic=1) was caused by a code bug in
the hard-IC residual, not a real optimization pathology. Corrected findings first; the
retracted original analysis is kept at the bottom of this section for the record — it
should not be cited.

### The bug, and how it was found

A later audit (prompted by the suspicious combination of "hard IC's training loss
converges to near machine precision" + "hard IC is catastrophically wrong physically")
found that `PINNSolver1DTD.psi0_at()` — used by the hard-IC path to build
`psi(x,t) = psi0(x) + t*net(x,t)` — computed ψ₀ via a numpy round-trip on a *detached*
copy of `x`, so ψ₀ re-entered the autograd graph as a constant with zero spatial
derivative. The hard-IC PDE residual needs `∂²ψ/∂x²`, and with ψ₀ detached its
`∂²ψ₀/∂x²` term — the actual source term describing the initial wavepacket's curvature —
silently evaluated to zero. Under that broken residual, `net ≡ 0` (a wavefunction frozen
at its t=0 shape forever) is the *exact global minimizer* of the training objective,
which reproduces every symptom that was previously catalogued as a "pathology": near-zero
training loss, a reproducible seed-independent non-propagating attractor, and error
growing with momentum (a faster true packet strays further from a frozen one).

Confirmed three ways: (1) a direct autograd probe — `d(psi)/dx` at t=0 returned all
zeros against a nonzero analytic derivative; (2) a patched hard-IC solver with ψ₀
computed directly in torch (so autograd sees it) trained on the same px=2 setup dropped
from 77.2% to 1.09–1.36% across 2 seeds, before the formal re-run below; (3) the bug is
inherited from the original (pre-reorg) repo code, meaning the original exp03 px=1
headline number (45.9% error) was bug-driven too, not a genuine finding about causality.

Fixed in `experiments/_common1d.py` (new `_psi0_torch`, used by `full_psi` when
`ic="hard"`; the old `psi0_at` is kept only for the soft-IC path's `ic_loss`, where ψ₀ is
a constant MSE target and never needs a spatial derivative — no bug on that path). The
same class of bug was independently present and fixed in `qbox/pinn2d.py`'s
`PINNSolverHardIC` (new `_psi0_torch`, a closed-form Gaussian×plane-wave matching
`QuantumSimulator.initialize_wave_packet`, replacing a nearest-grid-point numpy lookup
that was equally non-differentiable).

A second, independent issue was found and fixed at the same time: `split_operator_1d`'s
original soft-wall ground truth (V=50 starting 0.3 inside the domain edges, a standard
split-operator absorbing-boundary trick) diverges from the true hard-wall
(infinite-square-well) solution by **>80% relative L2 at t=1 for the px=4 case**, once
the packet's probability mass reaches the wall region (~14% of it does by t=1) — a
genuine physics mismatch against what the PINN architecture actually assumes
(`Net1D`'s multiplicative envelope forces ψ≡0 exactly at x=±5, i.e. an infinite well, not
a finite V=50 barrier 0.3 inside it). Replaced with an exact solution via a Dirichlet
sine series (DST-I): eigenmodes evolve analytically (no time-stepping error, no
wall-height/width hyperparameter to tune), and ψ=0 at the boundary is exact by
construction, matching the PINN exactly. Low-momentum ground truth (px≤2) was only
mildly affected by this (<2.5%); it exclusively matters for the px=4 point used in Task
2.3/2.4/2.5.

### Corrected result

Same protocol as the original run (px=2.0, 5 seeds, 5000 epochs, 2000 collocation
points), with both fixes applied. Raw: `results_raw/exp08_ic_formulation.json`.

| Formulation | Mean error | Std |
|---|---|---|
| Hard IC (ψ = ψ₀ + t·net) | **1.09%** | 0.11% |
| Soft IC, w_ic=1 | 2.03% | 0.19% |
| Soft IC, w_ic=10 | 2.10% | 0.35% |
| Soft IC, w_ic=100 | 7.66% | 0.53% |
| Soft IC, w_ic=1000 (original 2D weight) | 15.31% | 1.63% |
| Soft IC, w_ic=10000 | 19.67% | 2.28% |

**Decision: hard IC, adopted for exp06/exp07/exp09.** This matches the plan's original
hypothesis (hard IC ≤ best soft IC, since it satisfies the IC exactly and removes a
hyperparameter). The earlier reversal was entirely a bug artifact, not a real property
of the two formulations, in the simple 1D free-particle setting tested here. **This does
not transfer cleanly to the full 2D problem — see Task 2.2.**

### Retracted original analysis (kept for the record — do not cite)

The original run of this experiment found the opposite (soft IC w=1 at 2.17% vs hard IC
at 77.24%) and included a bounded investigation into *why*, concluding hard IC converges
to a "reproducible non-propagating attractor" and citing it as evidence for causality
violation (mechanism B). That investigation correctly established the failure was
deterministic and reproducible — it misattributed the cause. Reproducibility is exactly
what a deterministic bug (not a stochastic optimization pathology) produces, and the
investigation checked seed-independence and the loss-vs-accuracy mismatch without
checking whether the residual's autograd graph was actually intact. Lesson: a
suspiciously-low training loss on a physically wrong solution should prompt checking the
loss computation itself, not just the optimization dynamics around it.

## Task 2.2 — 2D re-verification under the adopted formulation

`experiments/exp02_pinn2d_comparison.py`, N=100 grid, 2000 epochs, px=-15 (the original
2D headline setup). Single run each (2D comparison stays single-seed per the plan's
seeds policy — expensive and qualitative). Figures: `figures/*_soft_w1.png` / `.gif`,
`figures/*_soft_w1000.png` / `.gif`, `figures/*_hard.png` / `.gif`.

| Formulation | Mean rel. error | Final rel. error | PINN train+infer | Slowdown vs classical |
|---|---|---|---|---|
| Hard IC (**adopted by Task 2.1, fixed ψ₀**) | 272.89% | 470.82% | 331.5s | ~248x |
| Soft IC, w_ic=1 | **99.96%** | 99.82% | 267.2s | ~548x |
| Soft IC, w_ic=1000 (original) | 99.78% | 99.37% | 292.6s | ~155x |

**The formulation that wins decisively in the simple 1D free-particle setting (hard IC,
Task 2.1) is the worst of the three in the full 2D problem** — nearly 3x worse than
either soft-IC variant, and its final-time error (470.82%) is *higher* than its
mean-time error, meaning it gets worse, not better, as training time increases. This
was measured with the psi₀-differentiability bug already fixed (verified: the 2D
wave-packet parameters passed to the traditional solver and to `PINNSolverHardIC`'s
closed-form ψ₀ are identical — `x0=3.75, y0=0, sigma=0.5, px=-15, py=0` — so this is not
a repeat of the Task 2.1 bug). The hard-IC training loss also did not converge cleanly
(oscillating noisily between ~100–180 across 2000 epochs rather than decreasing
smoothly), unlike its clean convergence in 1D — plausibly because the 2D problem has
actual potential structure (confining walls, a circular barrier, a double slit) that the
`Net2DDelta` correction network must learn to navigate, where the 1D setting it won in
was a bare free particle with no potential term at all. Whether more epochs / a tuned
learning rate would close this gap is an open question, not attempted here (out of
scope, and moot for the headline claim below).

**Headline claim:** per this repo's own stated policy of reporting "the strongest
formulation actually measured, not a strawman," and since the 1D-adopted formulation
(hard IC) demonstrably does not carry over to 2D, the headline number is the **best of
the three measured**: soft IC w=1, 99.96% mean relative error. All three formulations
fail badly — the range is 99.78%–272.89%, not a case of one formulation rescuing the
dynamics. Reproduce:
```
uv run experiments/exp02_pinn2d_comparison.py --formulation hard --epochs 2000
uv run experiments/exp02_pinn2d_comparison.py --formulation soft --w-ic 1.0 --epochs 2000
uv run experiments/exp02_pinn2d_comparison.py --formulation soft --w-ic 1000.0 --epochs 2000
```

This is consistent with, not contradicted by, the 1D momentum sweep (Task 2.3): the 2D
setup uses px=-15, far beyond the px=4 already showing ~38% error in 1D, so near-total
failure here is consistent with the spectral-bias trend extrapolated to this momentum —
independent of which IC formulation is used.

Note on the slowdown factor: the classical solver's wall-clock time was noisy across
runs (0.49–1.88s for the identical deterministic computation) due to background system
load during these runs — the ~150–550x range should be read as "two-to-three orders of
magnitude slower," not a precise multiplier. PINN training time itself (267–332s across
all three formulations) is the more reliable number.

## Task 2.3 — Momentum sweep (exp06_momentum_sweep.py)

Adopted formulation (hard IC, fixed), 5 seeds, 5000 epochs, exact sine-series ground
truth (see Task 2.1). Raw: `results_raw/exp06_momentum_sweep.json`. Figure:
`figures/exp06_momentum_sweep.png`.

| px | Mean error | Std |
|---|---|---|
| 0.5 | 0.26% | 0.05% |
| 1.0 | 0.19% | 0.04% |
| 2.0 | 1.09% | 0.11% |
| 4.0 | **37.68%** | 1.35% |

**Monotonic increase: NOT confirmed (script's own tripwire flagged it).** px=0.5→1.0
goes *down* slightly (0.26%→0.19%) before the trend reverses and grows steeply through
px=2 and explosively to px=4. Given both values are well under 0.3% and only ~1.4
combined std apart, this reads as noise at the floor of what this architecture/budget
can achieve on an easy problem, not a real non-monotonicity — but it's reported exactly
as measured rather than smoothed over. The headline trend survives clearly: error is
~flat and near-zero through px=2, then grows roughly 35x from px=2→4. This is a somewhat
less dramatic version of the trend than the (bug-contaminated) original run reported
(44.36% at px=4, now 37.68%, since the ground-truth wall-mismatch was also inflating
that number), but the qualitative spectral-bias story — error explodes as the wave
packet's phase oscillation frequency increases — is unchanged and, if anything, now
rests on a cleaner measurement.

## Task 2.4 — Norm/unitarity constraint (exp07_norm_constraint.py)

Adopted formulation (hard IC, fixed), 5 seeds, 5000 epochs, λ=1.0 headline. Raw:
`results_raw/exp07_norm_constraint.json`. Figure: `figures/exp07_norm_constraint.png`.

| px | Baseline | +norm constraint (λ=1) |
|---|---|---|
| 1.0 | 0.19% ± 0.04% | 1.24% ± 0.56% (worse) |
| 4.0 | 37.68% ± 1.35% | **31.64% ± 1.40% (better)** |

λ sensitivity (px=4, 3 seeds): λ=0.1 → **28.99% ± 1.12%** (best); λ=1.0 → 31.64%
(better than baseline); λ=10.0 → 54.57% ± 6.29% (much worse — over-constrained).

**Result: this reverses the original (bug-contaminated) conclusion. The norm constraint
now genuinely helps where there's a real problem to fix, and hurts where there isn't.**
The N(t) curves (`figures/exp07_norm_constraint.png`) show why: at px=1, baseline N(t)
already fluctuates mildly around 1.0 (no real amplitude decay), so adding the constraint
only adds a competing loss term — net effect purely negative (0.19%→1.24%). At px=4,
baseline N(t) genuinely collapses (1.0 → ~0.5 by t=1, confirming real amplitude decay is
a live mechanism at high momentum), and the constrained network holds N(t) much closer
to 1 (1.0 → ~0.82) *and* the |ψ|² snapshot at t=1 visibly tracks the true multi-peaked
interference structure more closely than baseline. Unlike the original (bug-contaminated)
run, the fix now earns its keep at the momentum where the diagnosed problem is real, as
long as λ isn't pushed too high (λ=10 over-constrains and hurts, same qualitative
shape as before). This is the cleanest "does what it says on the tin" result of the three
mechanism experiments: amplitude decay (mechanism A) is real at high momentum, and a
correctly-weighted penalty term measurably helps.

## Task 2.5 — Causal training (exp09_causal_training.py)

Adopted formulation (hard IC, fixed), 5 seeds, 5000 epochs, ε=10 (no adjustment needed).
Raw: `results_raw/exp09_causal_training.json`. Figure: `figures/exp09_causal_training.png`.

| px | Baseline | +causal training | Substantially improved? |
|---|---|---|---|
| 1.0 | 0.19% ± 0.04% | 0.18% ± 0.05% | No |
| 4.0 | 37.68% ± 1.35% | 40.27% ± 30.15% | Mean says no — see below |

**Result: the summary statistic is misleading. Causal training substantially helps in 4
of 5 seeds and catastrophically diverges in 1.** At px=4, per-seed causal errors were
25.83%, **100.55%**, 25.42%, 24.19%, 25.36%. Excluding the outlier seed, the other four
average **25.20% ± 0.61%** — a large, consistent improvement over the 37.68% baseline
(roughly a third lower). But seed 1 diverged to 100.55% error, more than 2.5x worse than
baseline, which is what drags the naive 5-seed mean (40.27%) and std (30.15%) into
looking like "no effect." At px=1 there's genuinely no effect either way (both near
0.19%), consistent with there being essentially no error left to fix at that momentum
under the adopted formulation.

This is a different and more interesting finding than either the original hypothesis
("fixes a px=1 propagation failure" — moot, since px=1 has no failure to fix under
correct hard IC) or the original run's conclusion ("no effect at either momentum" — an
artifact of the same IC bug that also produced the inflated 44.36%/45.44% baseline/causal
numbers). Causal training is not a reliable fix as currently implemented: it has real
upside at high momentum most of the time, but a training-instability failure mode
(1-in-5 in this small sample) that would need investigating — e.g. gradient clipping, a
smaller `causal_eps`, or a warm-up schedule — before it could be trusted, which is out of
scope for this repo (flagged as future work).

## Task 2.6 — Multi-seed error bars: exp04 (eigenstates), exp05 (inverse problem)

Unaffected by the Task 2.1 fix — exp04 and exp05 use their own independent PINN
solvers (`exp04_eigenstates.py`, `exp05_inverse_potential.py`), neither of which shares
`_common1d.py`'s hard-IC code path. Numbers unchanged from the original run.

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
