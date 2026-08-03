#!/usr/bin/env python3
"""Performance plots for the HZa mass regression estimator.

Produces:
  - 1D Per-Sample Normalized Predicted Mass Distributions (Signal points + Bkg)
  - 1D Per-Sample Normalized Relative Mass Residuals (Signal points + Bkg)

Usage
-----
    python analysis/scripts/plots_regression.py \
        --scores data/test_scores.h5 \
        --outdir analysis/plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="H5 file with evaluated datasets")
    p.add_argument("--outdir", default="analysis/plots")
    p.add_argument('--eval', type=str, default="atlas_regression_a_mass", help='what regression name to store results in')
    return p.parse_args()


def _load(scores_path: str, compare: str):
    import h5py
    import numpy as np
    from common.io import JETS_DATASET

    with h5py.File(scores_path, "r") as f:
        ds_name = JETS_DATASET if JETS_DATASET in f else "jets"
        jets = f[ds_name][:]

        # Attempt to dynamically extract signal vs background label column
        a_jet = None
        if "a_jet" in jets.dtype.names:
            a_jet = jets["a_jet"]
        elif "labels" in f and "a_jet" in f["labels"]:
            a_jet = f["labels"]["a_jet"][:]
        elif "labels" in f and ds_name in f["labels"] and "a_jet" in f["labels"][ds_name]:
            a_jet = f["labels"][ds_name]["a_jet"][:]

    pt = jets["pt"]
    eta = jets["eta"]

    if "truth_a_mass" in jets.dtype.names:
        true_mass = jets["truth_a_mass"]
    else:
        raise KeyError("Could not find true mass target 'truth_a_mass' in the H5 file.")

    if compare in jets.dtype.names:
        pred_mass = jets[compare]
    else:
        raise KeyError("Could not find predicted mass 'atlas_regression_a_mass' in the H5 file.")

    if "atlas_valid" in jets.dtype.names:
        valid_mask = jets["atlas_valid"].astype(bool)
    else:
        valid_mask = np.ones(len(jets), dtype=bool)

    if a_jet is not None:
        return pt[valid_mask], eta[valid_mask], true_mass[valid_mask], pred_mass[valid_mask], a_jet[valid_mask]
    return pt[valid_mask], eta[valid_mask], true_mass[valid_mask], pred_mass[valid_mask], None


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    regress_name = args.eval

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency: {e}. pip install matplotlib numpy")
        sys.exit(1)

    # Load entries that passed validation selections
    pt, eta, true_mass, pred_mass, a_jet = _load(args.scores, regress_name)

    if len(true_mass) == 0:
        print("ERROR: No valid entries found after applying selection mask.")
        sys.exit(1)

    # Define the discrete target mass spectrum
    atlas_paper_masses = [0.5, 2.0, 3.5]
    atlas_colors = ["Red", "Lime", "Blue", "Black"]
    all_masses = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0]

    # Isolate sample indices into matching dictionary categories
    atlas_paper_masks = {}
    all_mass_masks = {}
    for m in atlas_paper_masses:
        # Floating point tolerance window around discrete points
        atlas_paper_masks[f"m = {m} GeV"] = np.isclose(true_mass, m, atol=0.1)
    for m in all_masses:
        all_mass_masks[f"m = {m} GeV"] = np.isclose(true_mass, m, atol=0.05)

    if a_jet is not None:
        atlas_paper_masks["Background"] = (a_jet == 0)
        all_mass_masks["Background"] = (a_jet == 0)
    else:
        # Fallback if label column is missing: classify anything far from signal points as background
        atlas_matched_signal = np.any([atlas_paper_masks[f"m = {m} GeV"] for m in atlas_paper_masses], axis=0)
        matched_signal = np.any([all_mass_masks[f"m = {m} GeV"] for m in all_mass_masks], axis=0)
        atlas_paper_masks["Background"] = ~atlas_matched_signal
        all_mass_masks["Background"] = ~matched_signal

    # Use a distinct 20-color map so all 11 overlaid lines remain clearly distinguishable
    colors = plt.cm.tab20.colors

    # ── 1. Separate Predicted Mass Distributions Overlay ────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bins_dist = np.linspace(0.0, 4.0, 100)

    for idx, (label, mask) in enumerate(atlas_paper_masks.items()):
        pred_sub = pred_mass[mask]
        if len(pred_sub) == 0:
            continue

        # density=True normalizes each curve independently by its own event count
        ax.hist(
            pred_sub,
            bins=bins_dist,
            histtype="step",
            linewidth=2,
            density=True,
            label=label,
            color=atlas_colors[idx]
        )

    ax.set_xlabel("Predicted Mass ($m_{pred}$) [GeV]")
    ax.set_ylabel("Probability Density (Normalized per sample)")
    ax.set_title("Predicted Mass Distribution by Category")

    ax.legend()
    fig.savefig(outdir / "atlas_paper_mass_dist_overlay.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved atlas_paper_mass_dist_overlay.pdf")

    # ── 1. Separate Predicted Mass Distributions Overlay ────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bins_dist = np.linspace(0.0, 8.0, 100)

    for idx, (label, mask) in enumerate(all_mass_masks.items()):
        pred_sub = pred_mass[mask]
        if len(pred_sub) == 0:
            continue

        # density=True normalizes each curve independently by its own event count
        ax.hist(
            pred_sub,
            bins=bins_dist,
            histtype="step",
            linewidth=2,
            density=True,
            label=label,
            color=colors[idx % len(colors)]
        )

    ax.set_xlabel("Predicted Mass ($m_{pred}$) [GeV]")
    ax.set_ylabel("Probability Density (Normalized per sample)")
    ax.set_title("Predicted Mass Distribution by Category")

    ax.legend()
    fig.savefig(outdir / "all_mass_dist_overlay.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved all_mass_dist_overlay.pdf")

    # ── 2. Overlaid Relative Mass Residuals ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bins_res = np.linspace(-1.5, 1.5, 100)

    for idx, (label, mask) in enumerate(atlas_paper_masks.items()):
        pred_sub = pred_mass[mask]
        true_sub = true_mass[mask]
        if len(pred_sub) == 0:
            continue

        # Safe residual computation
        if "Background" in label: continue # skip bkg
        else:
            res_sub = (pred_sub - true_sub) / true_sub
            display_label = label

        ax.hist(
            res_sub,
            bins=bins_res,
            histtype="step",
            linewidth=2,
            density=True,
            label=display_label,
            color=atlas_colors[idx]
        )

    ax.axvline(0, color="black", linestyle=":", alpha=0.7)
    ax.set_xlabel("Error Metric: $\Delta m / m_{true}$")
    ax.set_ylabel("Probability Density (Normalized per sample)")
    ax.set_title("Mass Resolution Residuals by Category")

    ax.legend()
    fig.savefig(outdir / "atlas_paper_mass_residuals.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved atlas_paper_mass_residuals.pdf")

    # ── 2. Overlaid Relative Mass Residuals ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    bins_res = np.linspace(-1.5, 1.5, 100)

    for idx, (label, mask) in enumerate(all_mass_masks.items()):
        pred_sub = pred_mass[mask]
        true_sub = true_mass[mask]
        if len(pred_sub) == 0:
            continue

        # Safe residual computation
        if "Background" in label: continue # skip bkg
        else:
            res_sub = (pred_sub - true_sub) / true_sub
            display_label = label

        ax.hist(
            res_sub,
            bins=bins_res,
            histtype="step",
            linewidth=2,
            density=True,
            label=display_label,
            color=colors[idx % len(colors)]
        )

    ax.axvline(0, color="black", linestyle=":", alpha=0.7)
    ax.set_xlabel("Error Metric: $\Delta m / m_{true}$")
    ax.set_ylabel("Probability Density (Normalized per sample)")
    ax.set_title("Mass Resolution Residuals by Category")

    ax.legend()
    fig.savefig(outdir / "all_mass_residuals.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved all_mass_residuals.pdf")

if __name__ == "__main__":
    main()


