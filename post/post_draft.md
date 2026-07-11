<!--
Draft for the owner to edit into their own voice. Two hook variants, one body.
All numbers below are placeholders (TODO) until Phase 2 results exist in
RESULTS.md -- do not post before replacing them.
-->

# Hook — Variant A

My neural network failed to solve the Schrödinger equation. That turned out to be the
interesting part.

# Hook — Variant B

I tried to replace a classical quantum solver with a neural network. It lost — badly,
and predictably once I understood why. Here's the diagnosis.

# Body (~150-200 words)

<!-- TODO: fill in numbers from RESULTS.md before posting -->

I built a physics-informed neural network (PINN) to solve the 2D time-dependent
Schrödinger equation — a particle diffracting through a double slit — and compared it
against a classical split-operator solver. The PINN's error against ground truth: TODO%,
at TODOx the training cost.

Rather than stop there, I ran it down. A momentum sweep showed the error grows
systematically with how fast the wave packet moves — consistent with "spectral bias," a
documented failure mode where neural networks struggle to represent high-frequency
functions. I tested two candidate fixes separately: a unitarity constraint (the network
kept discarding probability over time) and causal training (respecting that solutions
should be learned in time order, not sampled randomly). TODO: one line per fix's actual
outcome.

The twist: the same PINN machinery that fails at forward simulation is genuinely good at
an inverse problem a classical solver can't touch directly — recovering an unknown
potential from sparse, noisy wavefunction data, at TODO% error, where a naive
differentiate-the-noise baseline fails outright.

Full writeup, code, and figures: [repo link]

Curious if others have gotten Fourier-feature PINNs to work on wave-packet dynamics —
pointers welcome.

#MachineLearning #ComputationalPhysics #PINN
