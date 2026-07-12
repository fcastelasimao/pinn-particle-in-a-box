<!--
Draft for the owner to edit into their own voice. Two hook variants, one body.
Every number below traces to RESULTS.md. Verify against RESULTS.md if the
experiments are ever re-run before posting.
-->

# Hook — Variant A

My neural network's training loss hit near-machine precision on a physically wrong
answer. I almost published the wrong reason why.

# Hook — Variant B

I tried to replace a classical quantum solver with a neural network. It lost badly —
and I nearly drew the wrong conclusion about why, until a suspiciously clean loss curve
made me go back and audit the code.

# Body (~230 words)

I built a physics-informed neural network (PINN) to solve the 2D time-dependent
Schrödinger equation — a particle diffracting through a double slit — and compared it
against a classical split-operator solver. The PINN's error against ground truth:
~100%, at ~2-3 orders of magnitude the training cost, and that holds across all three
initial-condition formulations I tested. Not a tuning artifact.

Here's the part I almost got wrong. An early experiment showed one formulation (hard IC)
converging its training loss to near machine precision — while being catastrophically
wrong physically. I initially read that as a real optimization pathology, a "reproducible
attractor" tied to causality violation, and nearly wrote it up as a finding. It wasn't.
A tensor got silently `.detach()`-ed inside the physics-loss residual, zeroing out the
term that made the initial condition's curvature visible to the network's gradient. With
that term gone, "predict a frozen, unmoving wavefunction forever" became the exact global
minimum of the (broken) training objective. Fixed the bug, re-ran: the formulation I'd
rejected actually wins in the simple case, exactly like my original hypothesis said.

With the bug caught, the real mechanism decomposition held up: momentum-swept error
confirms spectral bias as the dominant cause. A unitarity constraint targeting amplitude
decay genuinely helps at high momentum once the norm-decay problem is real. Causal
training helps in 4 of 5 seeds but occasionally diverges — a real but unstable fix.

The twist: the same PINN machinery that fails at forward simulation is genuinely good at
an inverse problem a classical solver can't touch directly — recovering an unknown
potential from sparse, noisy wavefunction data, at ~15% error, where a naive
differentiate-the-noise baseline fails outright.

Full writeup, code, and figures: [repo link]

Curious if others have shipped a "finding" that turned out to be a silent `.detach()` —
feels like a rite of passage for this kind of work.

#MachineLearning #ComputationalPhysics #PINN
