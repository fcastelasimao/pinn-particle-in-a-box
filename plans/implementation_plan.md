# Implementation Plan: From Current State to Finalized LinkedIn Post

> **Status: complete (historical record).** Written before the repo was reorganized; file
> names below are as they were then. `RESULTS.md` has since been replaced by
> `results.ipynb` (data in `results/`).

> **Audience of this document:** an AI coding assistant (or human) implementing this plan
> in a fresh session, with no prior context. Read this whole file before touching anything.
>
> **Owner:** Francisco Simão. The final LinkedIn post is published manually by the owner;
> everything else below is implementable.

---

## 1. Project context (read first)

This repo is a self-directed research project: solving the Schrödinger equation with a
Physics-Informed Neural Network (PINN) vs a classical split-operator solver, in a
2D particle-in-a-box with a double-slit barrier.

**The scientific arc — this is the product, do not lose it:**

1. **Baseline** (`traditional_solver.py`): classical split-operator solver, 2D double-slit
   simulation. Works. Output: `traditional_solver.gif`.
2. **The hyped method fails** (`pinn_solver.py` + `comparison.py`): a vanilla PINN on the same
   2D time-dependent problem reaches ~100% relative error and is orders of magnitude slower.
   Outputs: `pinn_solver.gif`, `error_evolution.png`, `timing_comparison.png`, `pinn_loss.png`.
3. **Diagnosis** (`03_td_1d_lowmom.py`): controlled 1D time-dependent experiment with hard
   initial condition, low momentum (px=1). Error still grows with time (~46% mean).
   Hypothesis: spectral bias — networks struggle with the fast phase oscillations
   exp(i·px·x) of moving wave packets. Output: `exp03_td_1d_lowmom.png`.
4. **Control experiment** (`04_eigenstates.py`): zero-momentum stationary eigenstates
   (infinite well, harmonic oscillator) via Rayleigh-quotient minimisation. The same PINN
   machinery succeeds (<1.5% energy error vs analytic). This pins the regime boundary.
   Output: `exp04_eigenstates.png`.
5. **Where PINNs structurally win** (`05_inverse_potential.py`): inverse problem — recover
   an unknown potential V(x) from sparse noisy wavefunction samples. Two-stage training
   (fit ψ, freeze, recover V). PINN gets ~13.6% error; the naive ψ″/ψ inversion fails
   catastrophically even with more data. A forward classical solver cannot do this task
   directly at all. Output: `exp05_inverse_potential.png`.

**The diagnosis is a hypothesis SET, not a single claim.** The time-dependent failure has
three candidate mechanisms, and Phase 2 tests each one:

| # | Candidate mechanism | Evidence for it | Tested by |
|---|--------------------|------------------|-----------|
| A | **Amplitude decay** — ψ≡0 trivially satisfies the PDE; amplitude leaks away from the t=0 anchor | general PINN pathology | exp07 (norm/unitarity constraint) |
| B | **Causality violation** — collocation times are sampled uniformly at random; nothing makes the solution at t build on the solution before t. Note the exp03 failure shape: the PINN packet *barely moves* while the truth propagates — at px=1 the wavelength is ~6 on a domain of 10, so spectral bias alone cannot explain this | exp03 figure; Wang et al. 2022; Krishnapriyan et al. 2021 | exp09 (causal training) |
| C | **Spectral bias** — the network cannot represent the fast phase oscillations exp(i·px·x) of high-momentum packets | error grows with px | exp06 (momentum sweep) |

Plus one methodological confound to eliminate: **IC formulation / loss weighting**. The 2D
solver uses a *soft* IC (`loss = loss_pde + 1000.0 * loss_initial` — the 1000 is arbitrary
and untested), while the 1D experiments use a *hard* IC (ψ = ψ₀ + t·net) with no weight at
all. exp08 compares the two head-to-head; the winner is **adopted everywhere**, including a
re-run of the 2D headline experiment, so the failure claim is made against the strongest
formulation of the method ("even with the IC enforced exactly, it fails").

**Scope is FROZEN at experiments 06–09.** These cover the full hypothesis space
(amplitude / causality / weighting / frequency). Do not add further fix attempts.

**Goal of the post:** signal that the owner is motivated, self-directed, has ML experience,
and above all thinks like a researcher (characterizes failure modes, runs controls, states
hypotheses before experiments, reports honestly). The hook is the *failure and its
diagnosis*, NOT "I built a PINN".

**Positioning relative to literature (for the README, keep to 2–3 sentences):** the failure
mode is consistent with known PINN pathologies — spectral bias (Rahaman et al. 2019) and
propagation/temporal failure modes (Krishnapriyan et al. 2021, "Characterizing possible
failure modes in physics-informed neural networks"); PINNs' strength on inverse problems
goes back to Raissi, Perdikaris & Karniadakis 2019. Citing these is a feature: it shows the
owner located their findings in the literature rather than claiming novelty.

---

## 2. Current repo state

```
traditional_solver.py     # classical split-operator solver (2D), imported by others
pinn_solver.py            # vanilla 2D time-dependent PINN, imported by comparison.py
comparison.py             # runs both, produces error/timing figures
03_td_1d_lowmom.py        # 1D time-dependent diagnosis (UNCOMMITTED)
04_eigenstates.py         # eigenstate control experiment (UNCOMMITTED)
05_inverse_potential.py   # inverse problem experiment (UNCOMMITTED)
pinn_control_test.py      # scratch/exploration script (UNCOMMITTED)
First attempt/            # early exploration folder (tracked)
*.png, *.gif              # result artifacts (some uncommitted)
__pycache__/, .DS_Store   # junk
```

No README. No packaging. Roughly half the work is untracked.
Machine: Apple Silicon (MPS backend); scripts auto-detect MPS/CUDA/CPU.

### ⚠️ Global constraints

- **Move code, don't rewrite physics.** The owner has approved a full reorganisation
  (Phase 1), but restructuring means relocating code and fixing imports — NOT refactoring
  solver internals, renaming variables, "improving" docstrings, or changing numerics.
  After every move, regenerate at least one figure and confirm it matches the pre-move
  output (same numbers in titles/legends).
- **Do not fabricate numbers.** Every number in the README and post copy must come from an
  actual run under this plan. If a re-run contradicts an old docstring claim, the claim
  changes, not the data.
- **State hypotheses before running.** Experiments 06 and 07 have explicit predictions
  written below; keep them in the scripts' docstrings and report outcomes against them.
- Long training runs: exp05 takes minutes; the 2D comparison potentially much longer.
  Run with generous timeouts, in the background where sensible.

---

## 3. Phase 1 — Reorganise and package (do this FIRST)

The owner explicitly wants a professional structure. Doing it first means all new
experiment code (06, 07) is written into its final home once.

### Task 1.1 — Target layout

```
QuantumParticle2DBox/
├── pyproject.toml            # uv-managed; deps: numpy, scipy, torch, matplotlib
├── uv.lock                   # committed
├── README.md                 # written in Phase 3
├── RESULTS.md                # verified-numbers scratch file (source of truth for all claims)
├── IMPLEMENTATION_PLAN.md    # this file
├── src/qbox/
│   ├── __init__.py
│   ├── classical.py          # from traditional_solver.py (QuantumSimulator etc.)
│   ├── pinn2d.py             # from pinn_solver.py (SchrodingerPINN, PINNSolver)
│   └── utils.py              # device selection + seed_everything(seed) ONLY
│                             #   (both are currently copy-pasted in every script —
│                             #    this is the one deduplication that is justified)
├── experiments/
│   ├── exp01_classical_2d.py       # thin driver: run classical sim, save gif
│   ├── exp02_pinn2d_comparison.py  # from comparison.py
│   ├── exp03_td_1d_lowmom.py       # from 03_td_1d_lowmom.py
│   ├── exp04_eigenstates.py        # from 04_eigenstates.py
│   ├── exp05_inverse_potential.py  # from 05_inverse_potential.py
│   ├── exp06_momentum_sweep.py     # NEW (Phase 2)
│   ├── exp07_norm_constraint.py    # NEW (Phase 2)
│   ├── exp08_ic_formulation.py     # NEW (Phase 2): soft-IC weight sweep vs hard IC
│   ├── exp09_causal_training.py    # NEW (Phase 2)
│   └── run_all.py                  # runs exp03–exp09 (skips slow 2D ones; README says how)
├── figures/                  # all committed output figures/gifs; scripts save here
└── post/
    ├── make_carousel.py      # Phase 4
    ├── carousel.pdf
    └── post_draft.md
```

Notes:
- Use `uv init` conventions; Python version = whatever is currently installed. Commit
  `uv.lock`. README commands use `uv run experiments/exp04_eigenstates.py` style.
- Experiments import shared code as `from qbox.classical import QuantumSimulator`
  (configure the package so `uv run` from repo root resolves it — hatchling src-layout
  plus `[tool.uv] package = true`, or equivalent).
- Experiments 03–05 are self-contained scripts; leave their internal physics as-is. Only
  swap in `qbox.utils` for the copy-pasted device/seed blocks and update save paths to
  `figures/`.
- Existing figures move to `figures/` (git mv) so their history survives.

### Task 1.2 — Cleanup
- `.gitignore`: add `__pycache__/`, `.DS_Store`, `*.pyc`, `.venv/` (merge with existing).
- Remove tracked junk (`git rm -r --cached` where needed); delete `.DS_Store` files.
- Remove `First attempt/` from the working tree (git rm -r; history keeps it).
  **[Owner default: remove. If the owner objects, move to `archive/` instead.]**
- Delete `pinn_control_test.py` / `pinn_control_test*.png` (scratch), unless they hold the
  only copy of a result.
- Commit the reorganisation as one or few clean commits (reorg separate from new code).

### Task 1.3 — Verify the reorg
- `uv run experiments/exp04_eigenstates.py` and `exp05` run to completion and reproduce
  their figures with the same headline numbers as the pre-move versions.
- `git status` clean; no stray files at repo root except the md files and pyproject/lock.

---

## 4. Phase 2 — Verify and strengthen the evidence

All headline numbers go public. Verify them first. Record everything in `RESULTS.md`.

**Seeds policy:** 1D experiments use **5 seeds** (0–4) and report mean ± std — 3 seeds
gives an unreliable std, and 1D runs are cheap; beyond ~10 is wasted compute for a figure.
The 2D comparison stays **single-seed**: it is expensive and its role is qualitative.

**Order matters:** Task 2.1 decides the IC formulation; every later experiment (including
the 2D re-run) uses the winner. Do 2.1 first.

### Task 2.1 — NEW `exp08_ic_formulation.py` (decide the formulation, kill the confounder)
The repo currently mixes two IC formulations: the 2D solver uses a soft IC with an
arbitrary weight (1000); the 1D experiments use a hard IC with no weight. This is
methodologically inconsistent, and loss-balance pathologies are a documented, *separate*
PINN failure axis (Wang et al. 2021, "Understanding and mitigating gradient flow
pathologies in PINNs"). This experiment settles both questions at once in 1D.

- Build a 1D **soft-IC** variant of the exp03 setup: same architecture/epochs, but the IC
  enforced by penalty (`loss = loss_pde + w_ic * loss_ic`) instead of by construction.
- Sweep `w_ic ∈ {1, 10, 100, 1000, 10000}` at fixed `px = 2` (clear-failure regime where
  reweighting has its best chance to help), 5 seeds each. Run the **hard-IC** version at
  px=2, 5 seeds, in the same script as the head-to-head reference.
- Output `figures/exp08_ic_formulation.png`: error vs w_ic (log x-axis), mean ± std, with
  the hard-IC result as a reference line/band. One panel.
- **Hypothesis (state in docstring):** hard IC ≤ best soft IC in error (it satisfies the
  IC exactly and removes a hyperparameter), and NO formulation/weight rescues the dynamics
  — the failure is not a loss-balance artifact.
- **Decision rule:** the formulation with the lower error (expected: hard IC) is **adopted
  for every subsequent experiment, including the 2D re-run in Task 2.2**. Record the
  decision and numbers in `RESULTS.md`.
- **Verify:** if some soft-IC weight DOES substantially rescue the dynamics, stop and flag
  to the owner — that would redirect the story toward loss balancing and must not be buried.

### Task 2.2 — Re-verify the 2D failure with the ADOPTED formulation (steelman the method)
The headline claim must be made against the strongest version of the method, so a skeptic
cannot say "you used the weak formulation."

- Implement a hard-IC variant of the 2D PINN (assuming hard IC won Task 2.1):
  ψ(x,y,t) = ψ₀(x,y) + t·net(x,y,t), with net outputting real and imaginary channels —
  same construction as exp03, one dimension up. Add it alongside (not replacing) the soft
  version in `qbox/pinn2d.py` with the smallest reasonable diff.
- Run `exp02_pinn2d_comparison.py` with the adopted formulation end to end (slow; run in
  background). Also record the original soft-IC numbers (one run) so the README can say
  "fails under both formulations."
- Record: final relative L2 error, PINN training wall-clock, split-operator wall-clock,
  slowdown factor. Old claim: "~100% error, ~250× slower" — confirm or replace.
- If a run errors or takes >1h, reduce epochs only as far as needed to reproduce the
  qualitative result, and report timings for the configuration actually run.
- **Verify:** concrete numbers for both formulations in `RESULTS.md`; the headline failure
  holds under the adopted (stronger) formulation.

### Task 2.3 — NEW `exp06_momentum_sweep.py` (the "failure law" figure)
Currently "error grows with momentum" rests on one data point (px=1).

- Reuse the exp03 setup (1D, same architecture, same epochs) with the IC formulation
  adopted in Task 2.1. Copy what you need; do not modify exp03.
- Sweep `px ∈ {0.5, 1, 2, 4}` × 5 seeds. Metric: mean relative L2 error of |ψ|² vs
  split-operator ground truth over the time snapshots (same metric as exp03).
- Output `figures/exp06_momentum_sweep.png`: error vs px, mean ± std (error bars or shaded
  band), log y-axis if the spread demands it. One clean panel, minimal title.
- **Hypothesis (state in docstring):** error increases monotonically with px because the
  network must represent phase oscillations exp(i·px·x) of increasing frequency
  (spectral bias).
- **Verify:** trend is monotonic increasing. If NOT, stop and flag to the owner — that
  changes the story and must not be papered over.

### Task 2.4 — NEW `exp07_norm_constraint.py` (test mechanism A: amplitude decay)
ψ ≡ 0 trivially satisfies the PDE residual, so time-dependent PINNs let the amplitude
decay away from the t=0 anchor. Physics says the norm is conserved (unitarity):
∫|ψ(x,t)|²dx = 1 for all t. Test whether enforcing this fixes the failure.

- Reuse the exp03 setup. Add a normalisation loss term:
  - Sample a small set of times per step (e.g. 8 uniform t values, resampled each epoch).
  - For each t, evaluate ψ on a fixed x-grid (e.g. 256 points) and estimate
    N(t) = ∫|ψ|²dx by trapezoid.
  - `loss_norm = mean_t (N(t) − 1)²`, added with weight λ. Headline results at λ=1.0.
- Compare **baseline vs +norm-constraint** at `px ∈ {1, 4}` (one easy, one hard regime),
  5 seeds each, same epochs/architecture.
- **λ sensitivity check (small):** rerun the constrained variant at px=4 with λ ∈ {0.1, 10},
  3 seeds each. Record in `RESULTS.md` whether the conclusion changes; one sentence in the
  README. Do not expand this into a grid search.
- Output `figures/exp07_norm_constraint.png`: 2×2 or 1×4 panel — for each px: |ψ|²
  snapshots (baseline vs constrained vs truth) and N(t) vs t curves showing the constraint
  actually holds. Plus the summary numbers (mean ± std error) into `RESULTS.md`.
- **Hypothesis (state in docstring):** the constraint prevents amplitude collapse
  (N(t) stays ≈1) but does NOT fix the dispersion/phase error at high momentum, because
  spectral bias is about representing fast oscillations, not amplitude. The experiment
  separates the failure into two distinct modes: amplitude decay (fixable by physics
  constraints) and phase/dispersion error (not fixable this way).
- **Verify:** N(t)≈1 in the constrained runs (the mechanism works even if the error
  doesn't drop). Whatever the error outcome, report it against the hypothesis — a
  negative result here is content, not a failure of the plan.

### Task 2.5 — NEW `exp09_causal_training.py` (test mechanism B: causality violation)
The baseline PINN samples collocation times uniformly at random: nothing in training makes
the solution at time t build on the solution before t — only the ∂ψ/∂t coupling and the
t=0 anchor encode temporal order. This is the known "causality violation" pathology (Wang,
Sankaran & Perdikaris 2022, "Respecting causality is all you need for training
physics-informed neural networks"; Krishnapriyan et al. 2021 propose time-marching).
The exp03 failure shape — the PINN packet barely moves while the truth propagates, at a
momentum too low for spectral bias to explain — makes this the prime suspect at low px.

- Reuse the exp03 setup (adopted IC formulation). Implement **causal weighting**:
  - Discretise t into M = 10 bins. Each epoch, compute the mean PDE residual L_j per bin.
  - Weight bin i by w_i = exp(−ε · Σ_{j<i} L_j), with stop-gradient on the w_i
    (weights are treated as constants in backprop).
  - Total PDE loss = Σ_i w_i · L_i / Σ_i w_i. Use ε = 10 as default; if all weights
    saturate at ~1 or collapse to ~0, adjust ε by one order of magnitude and record it.
  - Fallback if causal weighting proves fiddly: sequential **time-marching** (train on
    [0, T/4], then [0, 2T/4], etc., continuing from the same network) — simpler, same
    scientific purpose. Implement ONE of the two, not both.
- Compare **baseline vs causal** at `px ∈ {1, 4}`, 5 seeds each, same epochs/architecture.
- Output `figures/exp09_causal_training.png`: |ψ|² snapshots (baseline vs causal vs truth)
  per px + summary error bars.
- **Hypothesis (state in docstring):** causal training substantially reduces the error at
  px=1 (failure there is propagation-driven — the packet doesn't move) but does NOT rescue
  px=4 (failure there is representation-driven — spectral bias). This would cleanly
  separate mechanisms B and C.
- **Verify:** whatever the outcome, report against the hypothesis. If causal training
  rescues BOTH momenta, stop and flag to the owner: the story becomes "causality, not
  spectral bias, was the dominant failure" — a different (still good) post, and the owner
  must re-approve the narrative before Phases 3–4 proceed.

### Task 2.6 — Multi-seed error bars for the headline experiments
- Run exp04 and exp05 with 5 seeds each (parametrize SEED via CLI arg or env var —
  smallest possible diff to the scripts).
- Record mean ± std of: eigenstate energy errors (exp04), V(x) relative L2 error (exp05)
  into `RESULTS.md`. Report "13.6% ± X%" style in the README.
- Committed PNGs remain the seed-0 run; error bars live in text.
- **Verify:** stds small relative to means. If a seed diverges, report it honestly in the
  README limitations section — do not cherry-pick.

---

## 5. Phase 3 — README as a mini research report

Structure (keep total length ~2 screens; figures do the talking):

1. **Title + one-line abstract.** E.g. "Where physics-informed neural networks fail on the
   Schrödinger equation — and the one task where they beat a classical solver."
2. **Hero visual:** the double-slit GIF.
3. **The question.** Can a PINN replace a classical PDE solver for quantum dynamics?
4. **The failure.** 2D result + verified numbers (Task 2.2), stated for the ADOPTED
   (strongest) formulation, with one line noting it fails under both. Embed
   error-evolution figure.
5. **Methodology note.** One short paragraph: soft vs hard IC compared head-to-head
   (exp08 figure), hard IC adopted everywhere, no soft-IC weight over 4 orders of
   magnitude rescues the dynamics — the failure is not a loss-balancing artifact.
6. **The diagnosis: three candidate mechanisms, tested separately.** This is the
   scientific core of the README — a short table or three tight paragraphs:
   - **Amplitude decay** → exp07: unitarity constraint holds the norm; report what it did
     and didn't fix (+ λ insensitivity sentence).
   - **Causality violation** → exp09: random-time collocation has no arrow of time;
     causal training tested at low and high momentum; report against the hypothesis.
   - **Spectral bias** → exp06: error vs momentum with error bars (the failure-law figure).
   Write the actual outcomes, not the predictions — adapt if results surprised.
7. **The control.** Eigenstates work (numbers ± std). Embed exp04 figure.
8. **Where PINNs win.** Inverse problem (numbers ± std). Embed exp05 figure.
9. **Takeaways.** 3–4 bullets max, incl. a compact "use a PINN when / don't when" verdict.
10. **Threats to validity.** Honest limitations: single vanilla MLP architecture (no
    Fourier features / SIREN / adaptive loss weighting — cite as future work), modest
    tuning budget applied equally to both methods, 1D diagnosis extrapolated to the 2D
    failure (mitigated by exp08 and the hard-IC 2D re-run, but the mechanism experiments
    07–09 are 1D only), fixed ε and bin count in causal weighting, nearest-neighbor
    potential interpolation in the 2D PINN. This section is a researcher signal — keep it.
11. **References.** Raissi et al. 2019; Krishnapriyan et al. 2021; Rahaman et al. 2019;
    Wang, Teng & Perdikaris 2021 (gradient flow pathologies / loss balancing);
    Wang, Sankaran & Perdikaris 2022 (causal training).
12. **Reproduce it.** `uv sync` + `uv run experiments/run_all.py` (note which experiments
    are excluded for cost and how to run them).

- **Verify:** every number traces to `RESULTS.md`; every image path resolves on GitHub;
  reads standalone to someone who never saw the code.

---

## 6. Phase 4 — Post assets

### Task 4.1 — Carousel (primary visual asset)
Produce 6 slides as PNGs, 1080×1350 (LinkedIn portrait), assembled into `post/carousel.pdf`
— LinkedIn renders PDFs as swipeable carousels. Generate with `post/make_carousel.py` so
it's reproducible; big fonts (readable on a phone), minimal text per slide, consistent
style, no dense suptitles.

1. **Hook:** best frame of the double-slit simulation + one line: "I tried to replace this
   classical quantum solver with a neural network. It failed — that was the interesting part."
2. **The failure:** PINN vs ground truth side by side + verified numbers (error, slowdown).
3. **The diagnosis:** error-vs-momentum figure (with error bars). Caption: networks can't
   represent fast phase oscillations — the failure is systematic, not bad luck.
4. **The fixes, tested:** one slide summarising the three-mechanism decomposition —
   unitarity constraint (exp07), causal training (exp09), and what neither fixes
   (spectral bias, exp06). Caption written from ACTUAL outcomes; e.g. "Two of the three
   failure modes are fixable with physics constraints. The third needs a different
   architecture." Adapt if results surprised.
5. **The win:** inverse problem — recovered V(x) vs truth vs failed naive baseline.
   "Classical solvers can't do this task at all." (Eigenstate control goes to the README;
   it doesn't need a slide.)
6. **Decision matrix:** "Should you use a PINN?" — simple table/quadrant:
   forward + smooth/stationary → works but classical is faster; forward + high-frequency
   dynamics → avoid; inverse / sparse noisy data → PINN natively wins; data + physics
   fusion → PINN territory. This slide is the shareable takeaway.

- **Verify:** each slide legible at ~400px wide; numbers match `RESULTS.md`.

### Task 4.2 — Post copy (draft for the owner to edit into their own voice)
Write `post/post_draft.md` with 2 hook variants + one body. Rules:
- Hook on the failure. Variant A: "My neural network failed to solve the Schrödinger
  equation. That turned out to be the interesting part."
- Body ≈ 150–200 words: the arc in plain language — failure → momentum sweep → unitarity
  fix (what it did and didn't repair) → inverse-problem win. One sentence connecting to
  the literature (spectral bias is a known failure mode; I reproduced and measured it).
  Close with repo link.
- Tone: curious and matter-of-fact. No "thrilled to announce", no emoji spam, ≤3 hashtags
  (e.g. #MachineLearning #ComputationalPhysics #PINN).
- End with a question to invite comments (e.g. "Curious if others have gotten
  Fourier-feature PINNs to work on wave-packet dynamics — pointers welcome.")
- **Verify:** every claim matches `RESULTS.md`; reads in <45 seconds.

### Task 4.3 — Ship checklist (owner does the last step)
- [ ] All work committed; pushed to GitHub; repo set to **public**.
- [ ] README renders correctly on GitHub (images, GIF).
- [ ] `post/carousel.pdf` + `post/post_draft.md` delivered to owner.
- [ ] Owner edits copy into own voice and posts (attach carousel PDF, repo link in the
      post body or first comment).

---

## 7. Explicitly out of scope

- Architecture changes (Fourier features, SIREN) — the one fix class deliberately NOT
  attempted; mention once as future work in the README. exp07 (norm constraint) and exp09
  (causal training) are the only fix attempts in scope: both are training/physics changes,
  and together with exp06/exp08 they close the hypothesis set. **Scope is frozen at 06–09;
  do not add further experiments regardless of how interesting the results are.**
- Adaptive/learned loss weighting schemes (NTK-based, GradNorm, annealing) — exp08's fixed
  grid over w_ic is the full extent of weight exploration; anything beyond is future work.
- Sweeping the 2D solver's IC weight directly (too expensive; exp08 covers it in 1D, and
  the 2D headline uses the adopted formulation anyway).
- Tuning exp09's causal-weighting hyperparameters beyond the single recorded ε adjustment.
- Tests, CI, publishing the package, new physics scenarios (3D, other potentials).
- Rewriting solver internals beyond the moves/import fixes of Phase 1.

## 8. Definition of done

1. Repo restructured per Phase 1 layout; uv-packaged; `git status` clean; reorg verified
   by reproducing pre-move figures.
2. `RESULTS.md` holds verified numbers: IC-formulation study + adoption decision
   (per-w_ic mean ± std vs hard IC), 2D failure under the adopted formulation (error +
   slowdown, plus one soft-IC run), momentum sweep (per-px mean ± std, 5 seeds),
   norm-constraint outcome vs hypothesis (px ∈ {1,4}, 5 seeds, incl. λ sensitivity),
   causal-training outcome vs hypothesis (px ∈ {1,4}, 5 seeds, ε recorded),
   eigenstate errors (± std), inverse-problem error (± std).
3. README-as-report per Phase 3, all numbers traceable, all images rendering.
4. Deliverables: `post/carousel.pdf` (6 slides), `post/post_draft.md` (2 hooks + body).
5. Nothing published — publishing is the owner's manual step.
