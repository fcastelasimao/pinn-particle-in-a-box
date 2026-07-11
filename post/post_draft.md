<!--
Draft for the owner to edit into their own voice. Two hook variants, one body.
Every number below traces to RESULTS.md. Verify against RESULTS.md if the
experiments are ever re-run before posting.
-->

# Hook — Variant A

My neural network failed to solve the Schrödinger equation. That turned out to be the
interesting part.

# Hook — Variant B

I tried to replace a classical quantum solver with a neural network. It lost — badly,
and predictably once I understood why. Here's the diagnosis.

# Body (~180 words)

I built a physics-informed neural network (PINN) to solve the 2D time-dependent
Schrödinger equation — a particle diffracting through a double slit — and compared it
against a classical split-operator solver. The PINN's error against ground truth:
~100%, at ~2-3 orders of magnitude the training cost. Not a tuning artifact either —
it fails almost as badly under every initial-condition formulation I tested.

Rather than stop there, I ran it down. A momentum sweep showed error growing sharply
with how fast the wave packet moves — consistent with "spectral bias," a documented
failure mode where neural networks struggle to represent high-frequency functions. I
tested two candidate fixes separately: a unitarity constraint (the network was
discarding probability over time — confirmed, and fixable) and causal training
(respecting that solutions should be learned in time order). The unitarity fix worked
exactly as designed but made overall accuracy *worse* — it traded fine wave structure
for norm conservation. Causal training had no effect once I'd corrected an unrelated
methodology bug.

The twist: the same PINN machinery that fails at forward simulation is genuinely good at
an inverse problem a classical solver can't touch directly — recovering an unknown
potential from sparse, noisy wavefunction data, at ~15% error, where a naive
differentiate-the-noise baseline fails outright.

Full writeup, code, and figures: [repo link]

Curious if others have gotten Fourier-feature PINNs to work on wave-packet dynamics —
pointers welcome.

#MachineLearning #ComputationalPhysics #PINN
