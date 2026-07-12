# HANDOFF

**Last updated:** 2026-07-12

## State

All four phases of `IMPLEMENTATION_PLAN.md` are complete: repo reorganized into
`src/qbox/` + `experiments/` and `uv`-packaged, six Phase 2 diagnostic experiments run,
`README.md` rewritten as a research report, and a LinkedIn carousel + post draft
produced.

On top of that, this session ran an independent critical review of the prior session's
work and found a real bug: `PINNSolver1DTD`'s hard-IC path (`ic="hard"` in
`experiments/_common1d.py`) computed ψ₀ via a numpy round-trip on a *detached* tensor, so
its spatial derivative was invisible to autograd. The hard-IC PDE residual needs
`∂²ψ₀/∂x²` as a source term; with it silently zeroed, "predict a frozen wavefunction
forever" became the exact global minimum of the (broken) training loss. This produced
every symptom that the original session catalogued as a genuine "optimization pathology"
tied to causality violation — near-machine-precision loss on a physically wrong answer,
seed-independence, a reproducible non-propagating attractor — and led to an incorrect
headline decision (soft IC adopted over hard IC).

Fixed (`_psi0_torch` added to `PINNSolver1DTD` and to `PINNSolverHardIC` in
`src/qbox/pinn2d.py`), along with a second, independent issue: the 1D ground truth's
soft potential wall diverged from the PINN's actual hard-wall (infinite-well)
architecture once fast wave packets reached the boundary — replaced with an exact
Dirichlet sine-series solution (`split_operator_1d` in `_common1d.py`).

Every affected experiment was re-run under the corrected code: `exp08` (IC formulation
decision — now flips to hard IC), `exp06`/`exp07`/`exp09` (mechanism experiments under
the new adopted formulation), and `exp02` re-verified under all three formulations
(hard, soft w=1, soft w=1000) rather than just two. `RESULTS.md`, `README.md`,
`post/make_carousel.py` (now 7 slides — added "The near-miss," which tells the bug-catch
story), `post/carousel.pdf`, and `post/post_draft.md` were all rewritten to match. All
image links in `README.md` verified to resolve; the regenerated carousel was rendered to
PNG and visually inspected page-by-page (not just "the script ran").

`exp04`/`exp05` (eigenstates control, inverse-problem win) are unaffected by any of this
— they don't share the buggy code path — and their numbers are unchanged.

Nothing has been pushed to any remote at any point in this project. That remains the
owner's manual step.

## Next steps

1. Review the diff and commit (this `/wrap` run — see commit message drafted below).
2. Owner's manual ship checklist (unchanged): push, make the repo public, personalize
   `post/post_draft.md` into your own voice, post the carousel.
3. *(Optional, flagged as future work, not started)* Causal training (`exp09`) has a
   1-in-5-seed catastrophic-divergence failure mode at high momentum that's currently
   just reported honestly, not fixed. Would need gradient clipping, a smaller
   `causal_eps`, or a warm-up schedule to trust it as a real recommendation.
4. *(Optional, flagged as future work, not started)* The 2D hard-IC run doesn't converge
   cleanly (noisy training loss, 272.89% error vs soft IC's ~100%) — observed, not
   root-caused. Whether more epochs or a learning-rate schedule closes the gap is
   untested.

## Open questions

- Whether to keep the "retracted original analysis" subsection in `RESULTS.md` Task 2.1
  long-term (currently kept deliberately, for transparency about the near-miss) or trim
  it once the corrected story has been live for a while.
- Whether to invest in fixing causal training's instability (see Next steps #3) before
  or after publishing, or just ship with it flagged as future work as-is.

## Decisions

- **Adopted formulation for the 1D mechanism experiments (exp06/07/09) is hard IC**
  (bug-fixed), reversing the previous session's soft-IC adoption. This matches the
  original plan's hypothesis before any experiment ran.
- **The 2D headline number still uses soft IC, w=1 (99.96% error)**, even though hard IC
  is the 1D-adopted formulation — because hard IC performs *worse* in the full 2D
  problem (272.89%) than either soft variant. This formulation-dependence is reported as
  a finding in its own right, not smoothed over; the headline uses "best of three
  measured," per this repo's standing policy.
- **1D ground truth now uses an exact Dirichlet sine-series solution**
  (`split_operator_1d` in `experiments/_common1d.py`), replacing the old finite
  soft-wall split-operator method, because the old wall diverged from the PINN's actual
  boundary condition by >80% relative L2 at high momentum once the packet reached it.
  This is a permanent change — any future experiment using this ground truth inherits it.
- **Both `exp07` (norm constraint) and `exp09` (causal training) conclusions reversed**
  from the previous, bug-contaminated writeup: the norm constraint now genuinely helps
  at high momentum (was: "always hurts"); causal training helps in 4/5 seeds but
  diverges in 1/5 (was: "no effect either way"). These are considered the final,
  trustworthy conclusions for this repo's current state.
- User explicitly authorized ("Yes apply the fixes") relaunching every affected
  experiment and rewriting all downstream docs/assets to match — this was not a
  unilateral call.
