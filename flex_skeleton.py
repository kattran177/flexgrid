#!/usr/bin/env python3
"""
flex_skeleton.py: walking skeleton for "How firm is citizen flexibility?"

A deliberately tiny Monte Carlo model of ONE event (e.g. an evening-peak request)
asking EV owners to pause home charging. It answers one question:

    Out of the flexibility that looks available on paper (the HEADLINE),
    how much could a grid operator count on (the DEPENDABLE level)?

EVERY NUMBER BELOW IS A PLACEHOLDER, NOT A FINDING. Do not quote this script's
output in your report. Your job is to replace each placeholder with a sourced
value and record it in your assumption register.

RUN
    python flex_skeleton.py
    python flex_skeleton.py --p-respond 0.6 --shock-sd 1.0 --trials 5000
    python flex_skeleton.py --no-plot

Needs numpy. matplotlib is optional (a text histogram is printed if it can't load).

DEFINITIONS
    Headline potential  = every EV owner delivers full charger power
    Expected delivery   = mean delivered MW over all trials
    Dependable (P90)    = the delivered MW met or exceeded in 90% of trials
                          (the 10th percentile of the distribution)
    Reliability ratio   = dependable / headline

THE MODEL (per trial, per household)
    participates : signed up to the scheme                (p_participate)
    plugged_in   : car is home and plugged in at event time (p_plugged_in)
    responds     : doesn't override on the day            (p_respond)
    delivered    = charger_kw  if all three are true, else 0

    Independent run : every household's response is its own coin flip.
    Correlated run  : one shared "shock" per event shifts everyone's chance of
                      responding together (think: a very cold evening makes many
                      households override at once). shock_sd sets its size.

NEXT STEPS (do these yourself, in this order)
    1. Replace each placeholder with a sourced value; log it in your assumption register.
    2. Make participation FIXED per portfolio and vary only availability/response
       across events (currently it is redrawn every trial, which blurs two kinds of uncertainty).
    3. Give households different charger powers and arrival/departure patterns.
    4. Add a second device type (battery or heat pump).
    5. Define the event from your EirGrid data (the real evening peak, a windy overnight surplus).
    6. Compare design scenarios (A: owners only, B: stronger incentives, C: plus low-tech participation).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# PLACEHOLDER ASSUMPTIONS (replace with sourced values; one row each in your register)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "households": 1000,     # EV-owning households in the portfolio
    "charger_kw": 7.0,      # power each household can shift (kW)
    "p_participate": 0.40,  # share of households signed up
    "p_plugged_in": 0.65,   # share plugged in at event time
    "p_respond": 0.70,      # chance a participating, plugged-in household doesn't override
    "shock_sd": 1.0,        # size of the shared event-level shock for the correlated run (logit scale)
    "trials": 5000,
    "seed": 42,
}


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def simulate(households, charger_kw, p_participate, p_plugged_in, p_respond, shock_sd, trials, rng) -> np.ndarray:
    """Return delivered MW for each trial."""
    participates = rng.random((trials, households)) < p_participate
    plugged_in = rng.random((trials, households)) < p_plugged_in

    # One shared shock per trial (event). shock_sd = 0 gives fully independent responses.
    shock = rng.normal(0.0, shock_sd, size=(trials, 1)) if shock_sd > 0 else np.zeros((trials, 1))
    p_event = 1.0 / (1.0 + np.exp(-(logit(p_respond) + shock)))
    responds = rng.random((trials, households)) < p_event

    delivering = participates & plugged_in & responds
    return delivering.sum(axis=1) * charger_kw / 1000.0  # MW


def summarise(delivered_mw: np.ndarray, headline_mw: float) -> dict:
    dependable = float(np.percentile(delivered_mw, 10))  # met or exceeded in 90% of trials
    return {
        "headline_mw": headline_mw,
        "expected_mw": float(delivered_mw.mean()),
        "median_mw": float(np.percentile(delivered_mw, 50)),
        "dependable_p90_mw": dependable,
        "reliability_ratio": dependable / headline_mw,
    }


def print_table(results: dict[str, dict]) -> None:
    names = list(results)
    print(f"\n{'':<24}" + "".join(f"{n:>16}" for n in names))
    rows = [
        ("Headline (MW)", "headline_mw", "{:.2f}"),
        ("Expected (MW)", "expected_mw", "{:.3f}"),
        ("Median (MW)", "median_mw", "{:.3f}"),
        ("Dependable P90 (MW)", "dependable_p90_mw", "{:.3f}"),
        ("Reliability ratio", "reliability_ratio", "{:.1%}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<24}" + "".join(f"{fmt.format(results[n][key]):>16}" for n in names))


def text_histogram(delivered_mw: np.ndarray, bins: int = 15, width: int = 40) -> None:
    counts, edges = np.histogram(delivered_mw, bins=bins)
    peak = counts.max()
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  {lo:6.3f}-{hi:6.3f} MW | {'#' * int(width * c / peak)}")


def make_plot(runs: dict[str, np.ndarray], summaries: dict[str, dict], headline_mw: float, out_path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colours = {"Independent": "#1f77b4", "Correlated": "#d62728"}
    for name, delivered in runs.items():
        ax.hist(delivered, bins=40, alpha=0.5, color=colours.get(name), label=f"{name} responses")
        ax.axvline(summaries[name]["dependable_p90_mw"], color=colours.get(name), linestyle="--",
                   label=f"{name} dependable (P90)")
    ax.axvline(headline_mw, color="black", linestyle="-", label="Headline potential")
    ax.set_xlabel("Delivered flexibility in one event (MW)")
    ax.set_ylabel("Number of simulated events")
    ax.set_title("Headline vs dependable flexibility (PLACEHOLDER INPUTS)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Walking skeleton: headline vs dependable flexibility.")
    for key, default in DEFAULTS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=type(default), default=default)
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument("--out", default="figures/skeleton_dependable_flex.png")
    args = parser.parse_args(argv)

    params = dict(
        households=args.households, charger_kw=args.charger_kw, p_participate=args.p_participate,
        p_plugged_in=args.p_plugged_in, p_respond=args.p_respond, trials=args.trials,
    )
    headline_mw = args.households * args.charger_kw / 1000.0

    # Same seed for both runs so the only difference is the shared shock.
    runs = {
        "Independent": simulate(**params, shock_sd=0.0, rng=np.random.default_rng(args.seed)),
        "Correlated": simulate(**params, shock_sd=args.shock_sd, rng=np.random.default_rng(args.seed)),
    }
    summaries = {name: summarise(d, headline_mw) for name, d in runs.items()}

    print("PLACEHOLDER INPUTS, NOT FINDINGS")
    print(f"  {args.households} households x {args.charger_kw} kW | participate {args.p_participate:.0%} | "
          f"plugged in {args.p_plugged_in:.0%} | respond {args.p_respond:.0%} | "
          f"shock sd {args.shock_sd} | {args.trials} trials | seed {args.seed}")
    print_table(summaries)

    expected_share = args.p_participate * args.p_plugged_in * args.p_respond
    print(f"\nSanity check: expected share of headline = {args.p_participate} x {args.p_plugged_in} x "
          f"{args.p_respond} = {expected_share:.1%}  (simulated mean: "
          f"{summaries['Independent']['expected_mw'] / headline_mw:.1%})")

    if not args.no_plot:
        out = Path(args.out)
        if make_plot(runs, summaries, headline_mw, out):
            print(f"\nFigure saved to {out}")
        else:
            print("\nmatplotlib is not available here, so here is a text histogram (correlated run):")
            text_histogram(runs["Correlated"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
