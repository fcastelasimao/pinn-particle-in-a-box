# Neural operator study: plan

**Status:** not started. Written 2026-11-24. Nothing here has been built or run except
the checks and the pilot marked as such in section 4.
**Owner:** Francisco.
**Audience:** the owner, and the AI agents used through `/start-task`.
This file is the input to the grilling session. It is a starting position, not a spec:
grilling turns it into `.scratch/<slug>/spec.md`, and where the two differ the spec wins.

Scope of the old plan (`plans/implementation_plan.md`, exp01 to exp10) is unchanged. This study
is separate code next to it.

---

## 1. What this is

**Part 1, reproduction.** Reproduce the 1D result of S. Mizera, *Scattering with Neural
Operators*, arXiv:2308.14789 (Phys. Rev. D 108, L101701): a Fourier neural operator (FNO)
that advances a 1D wave function by one time step for an arbitrary potential, trained on
random fields and tested on a wave packet hitting a potential well it never saw.

**Part 2, extension.** When does the surrogate fail, and does it know? Measure error under
controlled shifts away from the training data, and test whether three cheap signals
(ensemble spread, unitarity deviation, conformal intervals) warn about it.

**Why.** It complements the PINN study in this repo: a supervised surrogate with
data, against a method without any. Part 1 shows the ability to reproduce a paper; Part 2
shows evaluation judgement, which is what surrogate-modelling jobs ask about.

### Non-goals

Dropped on purpose. Do not add them without the owner asking.

- Beating or matching a classical solver. A spectral split-step solver is expected to win
  in 1D (section 4) and the write-up says so.
- A speed-up claim. The paper compares with Crank–Nicolson, a weak baseline here (on weak
  baselines: McGreivy & Hakim, arXiv:2407.07218).
- U-Net or other architectures, training-set-size study, multi-fidelity training,
  hyperparameter search, inverse problems.
- 2D. A stretch goal only after Part 2 is written up: one d=2 run at N=64, never the
  double slit.

### Claims this study could support, and those it cannot

| Could support | Cannot support |
|---|---|
| "Reproduced / did not reproduce Mizera's 1D numbers, under stated choices" | "Neural operators are faster than solvers" |
| "Error rises by X under shift Y" | "Neural operators fail at scattering" in general |
| "Signal Z ranks errors with correlation r, and misses these cases" | "Uncertainty is solved" |
| "Conformal coverage holds on familiar data and drops to c% under shift Y" | Any guarantee out of distribution |

---

## 2. What the paper does (d = 1)

Read in full from arXiv v2. No public code or data was found, so everything is generated
here.

| Item | Paper |
|---|---|
| Equation | i ∂ₜΨ = (−½∇² + gV) Ψ, ħ = m = 1, g = 3×10³ |
| Domain | unit torus (periodic), N = 256 points |
| Step | T = 6.3×10⁻⁵; rollouts of M = 256 steps |
| Input | 4 channels: V, Re Ψ(0), Im Ψ(0), x |
| Output | Re Ψ(T), Im Ψ(T) |
| Training fields | V, Re Ψ, Im Ψ independent Gaussian random fields, power ∝ exp(−μ²p²/2), μ = 0.1 |
| Normalisation | ‖Ψ(0)‖ = 1 and ‖V‖ = 1 |
| Labels | Crank–Nicolson, iterated up to M steps, each output renormalised |
| Training set | K = 32 timelines × (M−1) = 8160 pairs |
| Test set | **one** wave packet approaching a potential well, 255 pairs; wave packets never in training |
| Network | linear lifting → 4 Fourier layers → 2-layer projection (hidden 512), width h = 64 |
| Layer | H ← σ(W H + K H + b + s H), GELU, scalar skip weight s |
| Spectral part | F = N/2 modes, real output by conjugate symmetry, Tucker-compressed, ratio r = 10⁻² |
| Parameters | about 9×10⁴ |
| Loss | L2 norm of Ψ(T) − network output |
| Optimiser | AdamW, lr 10⁻³, weight decay 10⁻⁵, batch 32, lr halved on training-loss plateau |
| Epochs | curves run to about 1250–1300; overtraining starts near 500 |
| Cost | 1.8 s per epoch on an A100 |

### Numbers to compare with

| Quantity | Paper, d = 1 | Paper, d = 2 (stretch) |
|---|---|---|
| Train loss | 3×10⁻⁴ | 3×10⁻⁴ |
| Test loss | 5×10⁻⁴ | 1.1×10⁻³ |
| Unitarity deviation | about 10⁻⁴ | 4×10⁻³ |
| Rollout error at step 256 (renormalised each step) | within about 0.02 | about 0.14 |
| Zero-shot super-resolution error, N = 2048 and 16384 | 0.004 | |

The paper calls unitarity deviation a proxy for the operator's confidence. Part 2 tests
that statement.

---

## 3. Ambiguities in the paper

These change results. Each needs an answer in the spec, with the reason. Never present a
choice made here as "the paper's".

| # | Ambiguity | Options | Starting recommendation |
|---|---|---|---|
| A1 | Norm used for ‖Ψ‖ = ‖V‖ = 1 | continuous L2 (with dx), or plain vector norm | **Decide by simulation first** (task T03): draw the test trajectory under both and compare with the paper's figure. It changes the potential phase per step by a factor 16 |
| A2 | Momentum in the random-field spectrum | p = 2πn, or integer n | p = 2πn; check visually against the paper's sample fields |
| A3 | Is exp(−μ²p²/2) the power spectrum or the amplitude | power, amplitude | power, as the text says |
| A4 | Loss | absolute or relative L2; mean or sum over a batch | relative L2 of the complex field, batch mean. Equal to absolute when targets have norm 1 |
| A5 | Label solver sub-steps per T | 1, or more | 1 as "paper-style"; the converged reference is separate (D2) |
| A6 | Wave-packet and well shape, width, momentum | not given | own choices, recorded, and many of them instead of one |
| A7 | Tucker ranks | not given | ranks that give about 1% of the full spectral weights; see section 4 |
| A8 | Epochs and selection | 500 or 1300; last or best | fixed budget, last checkpoint. Never select on the test set |
| A9 | Renormalise in rollout | yes (stated) | yes for the reproduction; report without as well |
| A10 | Unitarity deviation | mean of \|‖N(Ψ)‖ − 1\| or of squared norm | \|‖N(Ψ)‖ − 1\|, stated |

---

## 4. What has already been checked

Throwaway scripts, 2026-11-24, one run each, seed 0. These set expectations. They are not
results, and the numbers have to be recomputed by tested code before being quoted.

### 4.1 One step does very little

Relative L2 error against a converged spectral solver; 64 random-field samples,
μ = 0.1, p = 2πn.

| Predictor | continuous L2 norm, 1 step | vector norm, 1 step |
|---|---|---|
| Persistence (Ψ(T) = Ψ(0)) | 0.184 | 0.014 |
| Potential phase only, exp(−igVT)Ψ | 0.006 | 0.006 |
| One Strang split step | 1.9×10⁻⁵ | 2.8×10⁻⁷ |
| One Crank–Nicolson step (2nd-order differences) | 1.4×10⁻³ | 4.9×10⁻⁶ |
| Paper's FNO (loss scale) | 3 to 5×10⁻⁴ | 3 to 5×10⁻⁴ |

**Consequences.** A per-step error is meaningless without persistence and potential-phase
baselines beside it. One Strang step, which has no parameters, beats the paper's number
under either reading. Every results table has these rows.

### 4.2 The paper's label solver may be far from the true dynamics

After 256 steps, Crank–Nicolson with one sub-step and second-order differences differs
from the spectral solver by:

| Case | continuous L2 norm | vector norm |
|---|---|---|
| Random fields | 0.52 | 0.003 |
| Wave packets on a Gaussian well, momentum 10 / 20 / 40 / 60 | 0.24 / 0.24 / 0.33 / 0.67 | not run |

Under the continuous reading, an FNO that reproduces the labels would be reproducing a
discretisation artefact. Under the vector-norm reading it is negligible. So A1 comes
first, and Part 2 must use the converged solver for truth. One Strang step per T stays
within 0.003 (fields) and 0.005 (packets) over 256 steps under the continuous reading.

### 4.3 Parameter count

Non-spectral weights: 51 270. Full spectral weights at 128 modes: 4 194 304, so 1% is
41 943 and the total 93 213, consistent with the paper's 9×10⁴. Example ranks
(modes, in, out) = (19, 9, 9) with separate real and imaginary factors: total 92 254.

### 4.4 Compute (Apple GPU, this Mac, uncompressed layers with 64 modes)

0.048 s per step of batch 32, so 12 s per epoch of 255 steps.

| Run | Time |
|---|---|
| 500 epochs, one seed | 1.7 h |
| 1300 epochs, one seed | 4.4 h |
| 1300 epochs, five seeds | 22 h |
| CPU instead | 4.5 times slower |

An earlier estimate of about 80 minutes for five runs was wrong. Time again with the real
network (128 modes, Tucker) before committing to a budget; a free Colab GPU is the
fallback. On MPS call `torch.mps.synchronize()` every few steps: without it the queue ran
out of memory.

### 4.5 Pilot (not the paper's setup)

L = 40, N = 256, Δt = 0.25, 8 steps, 2000 trajectories, FNO with 1.78M parameters,
60 epochs, one seed, 16 minutes. Relative L2, 1 step / 8 steps:

| Test | FNO | Strang, 4 sub-steps per step (8 steps) |
|---|---|---|
| In distribution | 0.22% / 1.04% | 0.18% |
| Faster packets (p 4–5) | 2.5% / 9.3% | 0.33% |
| Square barrier | 3.8% / 9.0% | 4.25% |
| Potential twice as strong | 3.9% / 15.3% | 0.42% |

The question in Part 2 is therefore alive: errors grow 9 to 15 times under moderate
shift. The solver is better in every row.

### 4.6 Related results to know

- For one fixed Hamiltonian the map is linear in Ψ, and a linear estimator beat FNO by
  10 to 75 times (Patel et al., arXiv:2505.18288). Here V varies, so the map is linear in
  Ψ but nonlinear in V. A free property test: the trained network's deviation from
  linearity in Ψ.
- The exact map is norm-preserving, linear in Ψ, translation-covariant and
  time-reversible. Each gives a test of the solvers and a diagnostic of the network.

---

## 5. Design

### 5.1 Layout (new files only)

```
src/qbox/operators/
  fields.py      # Gaussian random fields, wave packets, potentials, norms
  solvers.py     # Strang and Crank–Nicolson on the unit torus
  data.py        # timelines, splits, saving and loading with a config hash
  fno.py         # spectral layer (full and Tucker), FNO1d
  train.py       # training loop, checkpoints, run config
  metrics.py     # relative L2, norm deviation, density error, rollout
  uncertainty.py # ensemble spread, conformal quantiles, correlation, AUROC
experiments/
  exp11_reproduction.py
  exp12_shift.py
  exp13_uncertainty.py
tests/operators/  # one file per module
.scratch/neural-operator/  # spec.md, issues/ (tickets, with their review log)
```

Configs are dataclasses written to JSON next to each result in `results/`. No config
framework, no experiment tracker, no CLI library. Dependencies: torch, numpy, scipy,
matplotlib as now, plus pytest for development. Write the Tucker layer by hand (one
`einsum`) instead of adding `neuraloperator` and `tensorly`; use neuraloperator only as a
reading reference.

The box study uses x ∈ [−5, 5] with hard walls. This study uses a unit torus. Nothing is
shared except `seed_everything` and `get_device`, and `_common1d.py` is not imported.

### 5.2 Conventions

- x_j = j/N, dx = 1/N, k = 2π·fftfreq(N, dx).
- Solvers in complex128. Data stored as complex64 or float32. Training in float32.
- MPS has no float64 and incomplete complex support: the network uses real and imaginary
  parts explicitly, and reference solutions are computed on the CPU.
- One function owns the norm. A1 is a named option (`"l2"` or `"vector"`), never a
  factor of N written inline.
- Seeds: data seed and training seed are separate. Train, calibration and test data come
  from different, recorded seeds.

### 5.3 Data

| Set | Content | Use |
|---|---|---|
| train | 32 random-field timelines × 255 pairs, as in the paper | training only |
| calibration | 64 new random-field **timelines** | conformal quantiles only |
| id_test | 64 new random-field timelines | in-distribution test |
| packet_test | 64 packet and well configurations (position, width, momentum, depth) | the paper's test, broadened |
| shift sets | section 6.2 | Part 2 only |

- Split by **timeline**, never by pair: neighbouring pairs are almost identical, so
  splitting pairs leaks.
- Each set is saved with its config hash, seed, solver and convergence check.
- The single configuration closest to the paper's figure is kept as `paper_packet` and
  reported separately.

### 5.4 Solvers and their acceptance tests

Strang splitting is the reference; Crank–Nicolson (periodic, tridiagonal plus corners)
exists for paper-style labels.

Tests, all with analytic or independent oracles:

1. Free plane wave: exact phase exp(−ik²t/2), to 1e-12.
2. Constant potential: exact global phase.
3. Free Gaussian packet, short time, well away from the wrap-around: analytic spreading.
4. Norm conservation over 256 steps to 1e-10.
5. Strang is second order in dt (observed order within 0.1 of 2); Crank–Nicolson second
   order in dt and in dx.
6. Forward then backward returns the input to 1e-10.
7. Translation covariance: shifting Ψ and V by m points shifts the output.
8. Small N: both agree with `scipy.linalg.expm` of the explicit Hamiltonian, where each
   is compared with its own discretised Hamiltonian.
9. Reference convergence: halving dt changes the reference by less than 1e-6 on every
   test set. Computed and stored, not assumed.

### 5.5 Network

As section 2. Build the full spectral layer first and Tucker second, so that the Tucker
layer can be tested against the full one at full rank.

Tests:

1. Output shape and real dtype; complex gradient reaches every parameter (no `None`, no
   exact zeros). This repo has been bitten by a detached tensor once.
2. Spectral layer with identity weights reproduces a band-limited input.
3. Modes above the cut are zero.
4. Translation covariance of the layer without the x channel.
5. Tucker layer equals the full layer when built from the same tensor.
6. Parameter count within 5% of 9×10⁴ for the paper configuration.
7. The same weights run at N = 512 (resolution change).
8. Overfits 8 samples to below 1e-3: a smoke test, marked slow.
9. Save and load gives identical outputs; same seed gives the same loss curve on CPU.

### 5.6 Training

Paper values from section 2. Checkpoint every 50 epochs, log train loss and a held-out
random-field loss. Last checkpoint is used. The packet and shift sets are never
evaluated during training or used to choose anything.

Order: overfit test → 20-epoch timing run → seed 0 complete → **owner looks at it** →
seeds 1 to 4. Agents do not launch runs longer than 10 minutes: they write the command and
the owner runs it in a terminal.

### 5.7 Baselines, in every table

persistence; potential phase only; one Strang step; one Crank–Nicolson step; and, for
Part 2, Strang at a matched cost if wall-clock is shown at all. Report the FNO where it
loses.

---

## 6. Experiments

Write the hypothesis and the pass rule into the spec **before** running. Afterwards the
hypothesis text is not edited; results go in their own section.

### 6.1 exp11: reproduction

- One-step error on id_test, packet_test and `paper_packet`; unitarity deviation; rollout
  to 256 steps with and without renormalisation; super-resolution at N = 512, 2048.
- Five seeds: mean and range. Seeds differ in initialisation and batch order; the
  training data are the same, and the text says so.
- Truth: paper-style labels for "did we reproduce it", converged solver for "is it right".
  Both columns.
- **Pass rule (proposal):** reproduced if the five-seed mean one-step errors (train, test)
  are within a factor of 2 of the paper's and the rollout error at 256 steps is at most
  0.04. Between 2 and 5 times: "partly". Beyond: "not reproduced", with the list of
  ambiguities that could explain it. All three are publishable outcomes.

| | Hypothesis |
|---|---|
| H1 | One-step error lands within a factor of 2 of the paper |
| H2 | One Strang step is more accurate than the FNO on every test set |
| H3 | Rollout error grows faster than linearly on packets, about linearly on random fields |

### 6.2 exp12: where it fails

One factor varied at a time, others at their in-distribution value, 64 configurations
per level, truth from the converged solver. At least one level of each axis must lie
inside the training range so that the curve has an anchor.

| Axis | Levels (proposal) | Probes |
|---|---|---|
| Field roughness μ, in Ψ | 0.1 (train), 0.07, 0.05, 0.03 | higher momentum than seen |
| Field roughness μ, in V | 0.1 (train), 0.07, 0.05, 0.03 | sharper potentials |
| Potential strength | 1 (train), 1.5, 2, 4 | amplitude extrapolation |
| Packet momentum | 4 to 5 levels up to the grid limit | spectral content |
| Potential family | random field, Gaussian well, two wells, square barrier | shape |
| Rollout length | 1, 8, 64, 256 | accumulation |

Report the median and the 10–90% range across configurations, and worst case. For the
momentum axis also plot error against the overlap of the packet's spectrum with the
training spectrum: the claim "fails on unfamiliar frequencies" needs that plot, because
momentum also changes how much one step moves the packet. Normalise by the persistence
error in a second column for the same reason.

| | Hypothesis |
|---|---|
| H4 | Roughness of Ψ beyond training hurts more than roughness of V |
| H5 | Error grows with potential strength faster than the persistence error does |
| H6 | Linearity in Ψ is broken more on shifted inputs than on familiar ones |

### 6.3 exp13: does it know

Three signals, all available without the true answer:

1. **Ensemble spread:** standard deviation over the five seeds' outputs, L2 over x.
2. **Unitarity deviation** of the single step, the paper's proposed proxy.
3. **Input distance:** a trivial control, such as the share of spectral power of Ψ and V
   beyond the training cut-off. If it does as well as the ensemble, say so.

Evaluations:

- Spearman correlation between signal and true error, within each set and pooled. Pooled
  alone is misleading: it can be high because the sets differ, so both are shown.
- AUROC for "error above 5 times the in-distribution median".
- Selective prediction: error after discarding the most uncertain 10, 20, 50%.
- **Confidently wrong cases:** the ten largest errors with spread below the median,
  plotted. This is the figure people remember.
- Ensemble blind spot, checked directly: the share of the error common to all members
  (error of the ensemble mean over the mean member error). Seeds share data and
  architecture, so they can agree and be wrong.

**Conformal intervals.** Split conformal on one scalar per sample, the relative L2 error
of the ensemble mean; score = error divided by (spread + ε), or plain error as control.
Quantile from the calibration set at 90%.

- The exchangeable unit is the **timeline**: pairs inside one are correlated. Use one
  random pair per timeline, or a timeline-level score, and say which.
- With 64 calibration timelines the 90% quantile is the ceil(65 × 0.9) = 59th of 64
  ordered scores. Realised coverage then varies between calibration draws: the
  Beta(59, 6) law gives standard deviation 0.036 and a 5–95% range of 0.84 to 0.96, so
  id_test coverage in that range is compatible with 0.90, before the test set's own
  sampling noise. If that is too coarse, generate more timelines: they cost seconds.
- Coverage on id_test should be near 90%. On shift sets the guarantee does not apply, and
  measuring the drop is the point.

| | Hypothesis |
|---|---|
| H7 | Ensemble spread ranks errors well across sets and noticeably worse within a set |
| H8 | Unitarity deviation is a weaker warning than ensemble spread |
| H9 | Coverage is about 90% in distribution and falls well below it on at least two shift axes |
| H10 | Spread-normalised scores lose less coverage under shift than plain ones |

All ten may be false. A false one is reported with the same prominence as a true one.

### 6.4 Figures (final set, at most five)

1. One packet: density at start, truth, FNO, and error, at 1, 64 and 256 steps.
2. Reproduction table as a dot plot: paper, five seeds, baselines, log scale.
3. Error against each shift axis, with training range shaded.
4. Signal against true error, coloured by set, with confidently wrong cases marked.
5. Conformal coverage per shift level with the 90% line.

---

## 7. Work breakdown

Sized so that each task is one ticket (tests → code → review). `/to-tickets` refines this
list; it does not have to keep it.

| Task | Content | Depends on |
|---|---|---|
| T01 | Norm, grid and field generators (`fields.py`) | |
| T02 | Strang and Crank–Nicolson (`solvers.py`) | T01 |
| T03 | Script that draws the paper's test trajectory under both norms; **owner settles A1, A2, A6** | T02 |
| T04 | Datasets, splits by timeline, hashes (`data.py`) | T03 |
| T05 | Metrics and baselines (`metrics.py`) | T02 |
| T06 | Spectral layer and FNO, full (`fno.py`) | |
| T07 | Tucker layer, parameter count | T06 |
| T08 | Training loop, checkpoints (`train.py`) | T04, T06 |
| T09 | 20-epoch timing run; **owner sets epochs, seeds and hardware** | T07, T08 |
| T10 | exp11 script; owner runs seed 0, then seeds 1–4 | T05, T09 |
| T11 | Reproduction write-up; **owner decides whether to continue** | T10 |
| T12 | Shift sets and exp12 | T11 |
| T13 | `uncertainty.py` with synthetic-data tests | T05 |
| T14 | exp13 | T12, T13 |
| T15 | Figures, results section, README section, CV bullets | T14 |

**Stop points.** After T03 (is the setup understood), T09 (is it affordable) and T11 (is
the reproduction good enough to build on). Stopping after T11 still leaves a complete
piece of work.

Effort guess: T01–T09 two to three days of part-time work with the agents, then one
night per seed or two, then T11–T15 two to three days. This is a guess, not a measurement.

---

## 8. Reporting rules

- Same policy as the rest of the repo: report honestly, all seeds, no best-of.
- Every figure and number comes from a script that reads `results/`, never typed in.
- Paper numbers are labelled as the paper's, and the choices in section 3 are listed next
  to any comparison.
- Baselines in the same table, including where they win.
- A near-perfect loss with odd outputs is a bug until shown otherwise. That pattern was a
  real bug here before: see results.ipynb, Task 2.1 (detached initial condition).
- Say what was done by AI agents. Do not claim a fact in an interview that the owner
  cannot derive or explain.
- No "novel" or "first". Part 2 uses standard methods on a small problem.

Follow-ups for the existing repo, separate from this study and not part of its
tickets: the inverse-problem claims need correcting. (Done 2026-11-26: exp10 committed and
written up in README and `results.ipynb`; HANDOFF removed.)

---

## 9. Risks

| Risk | Sign | Response |
|---|---|---|
| A1 read wrongly | reproduction off by 10 times or more | run both readings on seed 0 |
| Per-step task too easy | FNO near the potential-phase baseline | report it; add a longer-step variant as a stretch, outside the reproduction |
| Training too slow | above 5 h per seed | 500 epochs (paper: overtraining starts there), or Colab |
| Leakage through timelines | test error near train error on "new" data | split test on data module, reviewer check |
| Selection on the test set | repeated looks while tuning | shift and packet sets locked until T10 |
| Agents agree on a wrong physics convention | tests and code share an error | oracles are analytic or `expm`; reviewer injects bugs; owner checks T01–T03 by hand |
| Scope growth | new axis or architecture "while we are here" | non-goals list |
| Owner cannot defend it | cannot answer section 10 | do not put on the CV until possible |

---

## 10. What to be able to answer

Work through these while reviewing. They are the likely interview questions.

1. Why does an FNO layer act in Fourier space, and what does the local path W add?
2. What limits the modes, and why does the same network run at another resolution? When
   is the result wrong anyway?
3. Why Tucker compression, and what would happen without it?
4. Why is Strang second order? What error term does it leave? Why is it unitary?
5. Why does Crank–Nicolson keep the norm and still get the phase wrong?
6. Why is persistence a necessary baseline here?
7. Why split by timeline?
8. The map is linear in Ψ. Why not learn a linear map, and what does the network gain?
9. How does an ensemble estimate uncertainty, and why can it be confidently wrong?
10. What does conformal prediction guarantee, under which assumption, and which step of
    this study breaks it?
11. Why can pooled correlation look better than within-set correlation?
12. If a solver is better, when is a surrogate worth having? (Many evaluations,
    gradients through it, expensive solvers in high dimension or with complex geometry.)
13. Which parts did agents write, how were they checked, and what did the checks miss?

---

## 11. Questions for the grilling session

Asked one at a time, with these starting answers.

| # | Question | Recommended answer |
|---|---|---|
| Q1 | What must exist for this to be worth showing? | Parts 1 and 2 with five figures and an honest README section |
| Q2 | Deadline, and what is cut first? | Cut order: exp13 control signals, super-resolution, rollout without renormalisation |
| Q3 | How are A1–A10 settled? | T03 for A1, A2, A6; the rest as recommended |
| Q4 | What counts as reproduced? | The factor-of-2 rule in 6.1 |
| Q5 | Laptop or cloud GPU? | Decide after T09, laptop first |
| Q6 | Three seeds or five? | Five if under 5 h each, otherwise three and say so |
| Q7 | Which shift axes stay? | All six; drop "two wells" first |
| Q8 | Is one scalar per sample enough for conformal? | Yes; pointwise bands are out of scope |
| Q9 | Where does it live: this repo or its own? | This repo, new subpackage, own README section |
| Q10 | Which files does the owner read line by line? | `solvers.py`, `fno.py`, `uncertainty.py` and their tests |
| Q11 | What happens if it does not reproduce? | Publish that, with the ambiguity list |
| Q12 | What would make the owner stop? | No convergence by T09, or more than two weeks elapsed |
| Q13 | How is the result worded on a CV? | Decide after T11, from the numbers |
