# Where physics-informed neural networks fail on the Schrödinger equation — and the one task where they beat a classical solver

A physics-informed neural network (PINN) is trained to solve the 2D time-dependent
Schrödinger equation for a particle diffracting through a double slit, compared against
a classical split-operator solver. The PINN fails badly on the forward dynamics problem.
This repo characterizes *why* — three candidate failure mechanisms, tested separately —
and then shows the one class of problem where the same PINN machinery genuinely beats
the classical solver: recovering an unknown potential from sparse, noisy data.

![Classical double-slit simulation](figures/traditional_solver.gif)

## The question

Can a PINN replace a classical PDE solver for quantum wave-packet dynamics? The
classical split-operator method is fast, exact (to discretisation error), and has been
the standard tool for this class of problem for decades. PINNs are pitched as a
general-purpose alternative that needs no mesh and no numerical scheme, just the PDE
residual as a loss function. Does that generality hold up here?

## The failure

<!-- TODO: numbers from Task 2.2 (results_raw/exp02_summary.json) -->

## Methodology note: which IC formulation is "the" method?

<!-- TODO: numbers from results_raw/exp08_ic_formulation.json -->

## The diagnosis: three candidate mechanisms, tested separately

The time-dependent failure could come from (at least) three places. Each was tested
with a dedicated, falsifiable experiment rather than folded into one "let's try things"
pass.

<!-- TODO: numbers from exp06/exp07/exp09 results_raw -->

## The control: eigenstates work

<!-- TODO: numbers from results_raw/exp04_eigenstates.json -->

## Where PINNs win: the inverse problem

<!-- TODO: numbers from results_raw/exp05_inverse_potential.json -->

## Takeaways

<!-- TODO -->

## Threats to validity

- **Single architecture.** All PINNs here are vanilla MLPs with `tanh` activations.
  Architectures designed for high-frequency functions (Fourier feature embeddings,
  SIREN) are known partial fixes for spectral bias and were deliberately not attempted
  — see "Out of scope" below.
- **Modest, roughly-matched tuning budget.** Both methods got similar iteration counts
  and a small amount of hyperparameter exploration, not an exhaustive search either way.
- **1D diagnosis extrapolated to 2D.** The three mechanism experiments (exp06/07/09) run
  in 1D for cost reasons. The IC-formulation study (exp08) and the 2D re-run under the
  adopted formulation (Task 2.2) mitigate this for the headline failure claim, but the
  mechanism decomposition itself (amplitude / causality / spectral bias) is only
  verified in 1D.
- **Fixed causal-weighting hyperparameters.** `exp09` uses a fixed bin count (M=10) and
  a single recorded adjustment to ε; no further tuning was done (see "Out of scope").
- **Nearest-neighbor potential interpolation** in the 2D PINN (`qbox/pinn2d.py`) is a
  coarse way to hand the potential to autograd; a smoother interpolation might change
  the 2D numbers slightly, though not the qualitative failure.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural
  networks: A deep learning framework for solving forward and inverse problems involving
  nonlinear partial differential equations.* Journal of Computational Physics.
- Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., & Mahoney, M.W. (2021).
  *Characterizing possible failure modes in physics-informed neural networks.* NeurIPS.
- Rahaman, N., et al. (2019). *On the Spectral Bias of Neural Networks.* ICML.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). *Understanding and mitigating gradient
  flow pathologies in physics-informed neural networks.* SIAM Journal on Scientific
  Computing.
- Wang, S., Sankaran, S., & Perdikaris, P. (2022). *Respecting causality is all you need
  for training physics-informed neural networks.* arXiv:2203.07404.

## Reproduce it

```bash
uv sync
uv run experiments/run_all.py          # exp03, exp08, exp04-09 (skips the slow 2D ones)
uv run experiments/exp01_classical_2d.py
uv run experiments/exp02_pinn2d_comparison.py --formulation hard --epochs 2000
```

`exp01`/`exp02` are excluded from `run_all.py` because the 2D comparison is the slowest
part of the pipeline; run them explicitly if you want the 2D figures. Multi-seed error
bars for `exp04`/`exp05` (Task 2.6) are produced by re-running with `--seed 1..4`.
