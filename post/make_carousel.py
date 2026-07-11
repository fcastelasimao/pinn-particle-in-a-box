"""
Build the 6-slide LinkedIn carousel (post/carousel.pdf) from committed figures
and results_raw/*.json (the source of truth produced by the experiments).

Each slide is a matplotlib figure sized for LinkedIn portrait (1080x1350 px
at 150 dpi = 7.2x9.0 in), saved into one PDF via PdfPages so LinkedIn renders
it as a swipeable carousel.

Run after all Phase 2 experiments have produced their results_raw/*.json:
    uv run python post/make_carousel.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageSequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "figures")
RESULTS_RAW = os.path.join(ROOT, "results_raw")

SLIDE_SIZE = (7.2, 9.0)  # inches -> 1080x1350 px @ 150 dpi
DPI = 150

BG = "#0d1117"
FG = "#f0f6fc"
ACCENT = "#58a6ff"
ACCENT2 = "#3fb950"
ACCENT3 = "#f85149"


def load_json(name):
    path = os.path.join(RESULTS_RAW, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run the Phase 2 experiments before building the carousel."
        )
    with open(path) as f:
        return json.load(f)


def new_slide():
    fig = plt.figure(figsize=SLIDE_SIZE, dpi=DPI)
    fig.patch.set_facecolor(BG)
    return fig


def style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def best_gif_frame(gif_path, frac=0.55):
    """Extract a representative frame from a gif (default: a bit past midway,
    where the diffraction pattern is visually developed)."""
    im = Image.open(gif_path)
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(im)]
    idx = int(len(frames) * frac)
    return frames[min(idx, len(frames) - 1)]


def slide1_hook(pdf):
    fig = new_slide()

    fig.text(0.5, 0.97, "I tried to replace this quantum solver\nwith a neural network.",
              ha="center", va="top", fontsize=19, fontweight="bold", color=FG)

    ax_img = fig.add_axes([0.06, 0.22, 0.88, 0.52])
    ax_img.axis("off")
    frame = best_gif_frame(os.path.join(FIGURES, "traditional_solver.gif"))
    ax_img.imshow(frame)

    fig.text(0.5, 0.14, "It failed. That was the interesting part.",
              ha="center", va="top", fontsize=17, color=ACCENT, style="italic")
    fig.text(0.5, 0.04, "swipe -->", ha="center", fontsize=11, color="#8b949e")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def failure_comparison_crop():
    """Traditional vs PINN panels only (drop the busy error-map panel), from
    late in the simulation where the PINN's missing fine structure is clearest."""
    im = Image.open(os.path.join(FIGURES, "comparison_soft_w1.gif"))
    frames = [f.copy().convert("RGB") for f in ImageSequence.Iterator(im)]
    frame = frames[int(len(frames) * 0.9)]
    return frame.crop((150, 110, 2380, 1150))


def slide2_failure(pdf, exp02_adopted, exp02_original):
    fig = new_slide()
    fig.text(0.5, 0.95, "The failure", ha="center", fontsize=24, fontweight="bold", color=FG)

    ax_img = fig.add_axes([0.08, 0.42, 0.84, 0.46])
    ax_img.axis("off")
    img = failure_comparison_crop()
    ax_img.imshow(img)
    ax_img.text(0.5, -0.06, "Truth (left) has fine interference fringes the PINN (right) can't represent.",
                transform=ax_img.transAxes, ha="center", va="top", fontsize=9, color="#8b949e")

    err = exp02_adopted["mean_relative_error_pct"]
    slowdown = exp02_adopted["slowdown_factor"]

    fig.text(0.5, 0.32, f"{err:.0f}%", ha="center", fontsize=48, fontweight="bold", color=ACCENT3)
    fig.text(0.5, 0.24, "mean relative error vs the classical solver", ha="center", fontsize=13, color=FG)
    fig.text(0.5, 0.16, f"~{slowdown:.0f}x slower to train, for a wrong answer",
              ha="center", fontsize=13, color=FG)
    fig.text(0.5, 0.07, "Fails under the best initial-condition formulation tested,\nnot just a weak strawman.",
              ha="center", fontsize=11, color="#8b949e", style="italic")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide3_diagnosis(pdf, exp06):
    fig = new_slide()
    fig.text(0.5, 0.95, "The diagnosis", ha="center", fontsize=24, fontweight="bold", color=FG)
    ax = fig.add_axes([0.14, 0.30, 0.75, 0.56])
    style_ax(ax)
    px = exp06["px_values"]
    means = exp06["means"]
    stds = exp06["stds"]
    ax.errorbar(px, means, yerr=stds, fmt="o-", color=ACCENT, lw=2.5, ms=9, capsize=5)
    ax.set_xlabel("Momentum px", fontsize=12)
    ax.set_ylabel("Mean relative error (%)", fontsize=12)
    ax.grid(True, alpha=0.2, color="#30363d")

    fig.text(0.5, 0.16, "Error grows systematically with momentum.", ha="center",
              fontsize=15, color=FG, fontweight="bold")
    fig.text(0.5, 0.09, "Networks can't represent fast phase oscillations --\n"
                        "this is a known failure mode (spectral bias), not bad luck.",
              ha="center", fontsize=12, color="#8b949e")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide4_fixes(pdf, exp07, exp09, exp06):
    fig = new_slide()
    fig.text(0.5, 0.95, "The fixes, tested", ha="center", fontsize=24, fontweight="bold", color=FG)

    rows = [
        ("Amplitude decay", "unitarity constraint", exp07),
        ("Causality violation", "causal training", exp09),
        ("Spectral bias", "diagnosed, not fixed", None),
    ]

    y0 = 0.80
    for i, (mech, fix, data) in enumerate(rows):
        y = y0 - i * 0.22
        fig.text(0.08, y, mech, fontsize=15, fontweight="bold", color=ACCENT)
        fig.text(0.08, y - 0.045, fix, fontsize=11, color="#8b949e", style="italic")
        if data is not None:
            verdict = data.get("verdict", "")
            fig.text(0.55, y - 0.01, verdict, fontsize=10.5, color=FG, va="top")
        else:
            fig.text(0.55, y - 0.01, "Confirmed dominant cause\n(momentum sweep). Needs a\n"
                                       "different architecture --\nfuture work, not attempted.",
                     fontsize=10.5, color=FG, va="top")

    fig.text(0.5, 0.05, "Neither physics-constraint fix rescued accuracy once the\n"
                        "IC-formulation confound was corrected. Spectral bias is the story.",
              ha="center", fontsize=11, color="#8b949e", style="italic")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide5_win(pdf, exp05):
    fig = new_slide()
    fig.text(0.5, 0.95, "Where PINNs win", ha="center", fontsize=24, fontweight="bold", color=FG)
    ax_img = fig.add_axes([0.10, 0.42, 0.80, 0.46])
    ax_img.axis("off")
    full = Image.open(os.path.join(FIGURES, "exp05_inverse_potential.png"))
    img = full.crop((0, 90, 900, full.size[1]))
    ax_img.imshow(img)

    mean_err = exp05["mean_pct"]
    fig.text(0.5, 0.32, f"{mean_err:.1f}%", ha="center", fontsize=44, fontweight="bold", color=ACCENT2)
    fig.text(0.5, 0.24, "error recovering an UNKNOWN potential\nfrom sparse, noisy data",
              ha="center", fontsize=13, color=FG)
    fig.text(0.5, 0.10, "A classical forward solver can't do this task directly.\n"
                        "The PINN does it in one training run.",
              ha="center", fontsize=12, color="#8b949e", style="italic")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def slide6_decision(pdf):
    fig = new_slide()
    fig.text(0.5, 0.95, "Should you use a PINN?", ha="center", fontsize=22, fontweight="bold", color=FG)

    rows = [
        ("Forward + smooth / stationary", "Works, but classical is faster", ACCENT2),
        ("Forward + high-frequency dynamics", "Avoid -- spectral bias dominates", ACCENT3),
        ("Inverse / sparse noisy data", "PINN natively wins", ACCENT2),
        ("Data + physics fusion", "PINN territory", ACCENT2),
    ]
    y0 = 0.74
    for i, (case, verdict, color) in enumerate(rows):
        y = y0 - i * 0.15
        fig.text(0.08, y, case, fontsize=13, fontweight="bold", color=FG)
        fig.text(0.08, y - 0.05, verdict, fontsize=12, color=color)

    fig.text(0.5, 0.06, "Full writeup + code: link in post",
              ha="center", fontsize=11, color="#8b949e")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def main():
    exp06 = load_json("exp06_momentum_sweep.json")
    exp07_raw = load_json("exp07_norm_constraint.json")
    exp09_raw = load_json("exp09_causal_training.json")
    exp05_raw = load_json("exp05_inverse_potential.json")

    exp05_errs = [v["V_rel_err_pct"] for v in exp05_raw.values()]
    exp05 = {"mean_pct": sum(exp05_errs) / len(exp05_errs)}

    def norm_verdict(d):
        s = d["summary"]
        lines = ["Fixes amplitude decay (N(t)\nconfirms it), but costs detail:"]
        for px, row in s.items():
            delta = f"{row['baseline_mean']:.1f}%->{row['constrained_mean']:.1f}%"
            lines.append(f"px={px}: {delta} (worse)")
        return "\n".join(lines)

    def causal_verdict(d):
        s = d["summary"]
        lines = []
        for px, row in s.items():
            lines.append(f"px={px}: {'rescued' if row['improved'] else 'no effect'}")
        return "\n".join(lines)

    exp07 = {"verdict": norm_verdict(exp07_raw)}
    exp09 = {"verdict": causal_verdict(exp09_raw)}

    # exp02 headline numbers, written from the Task 2.2 runs into
    # results_raw/exp02_summary.json (see RESULTS.md Task 2.2).
    exp02 = load_json("exp02_summary.json")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "carousel.pdf")
    with PdfPages(out_path) as pdf:
        slide1_hook(pdf)
        slide2_failure(pdf, exp02["adopted"], exp02["original_weight"])
        slide3_diagnosis(pdf, exp06)
        slide4_fixes(pdf, exp07, exp09, exp06)
        slide5_win(pdf, exp05)
        slide6_decision(pdf)

    print(f"Carousel saved to {out_path}")


if __name__ == "__main__":
    main()
