#!/usr/bin/env python3
"""Make plots for a classifier that was trained on ALL mass points

Produces:
  - Score distributions (signal vs background) for the ATLAS paper masses and all masses

Usage
-----
    python analysis/scripts/plots.py \\
        --scores data/test_scores.h5 \\
        --outdir analysis/plots/
        --atlas  #specified if using ATLAS classifier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dask.array import store

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="H5 file with scores datasets")
    p.add_argument("--outdir", default="analysis/plots_mass_specific_classifier", help="Output directory")
    p.add_argument('--atlas', action='store_true', default=False, help='include if evaluating ATLAS classifier')
    return p.parse_args()

def _load(scores_path: str, atlas_flag: bool):
    import h5py
    from common.io import JETS_DATASET, LABELS_DATASET

    with h5py.File(scores_path, "r") as f:
        jets   = f[JETS_DATASET][:]
        labels = f[LABELS_DATASET]["a_jet"][:]
        true_masses = f[JETS_DATASET]["truth_a_mass"][:]
        scores = f["scores"][:, 0] if atlas_flag else f["scores"][:, 1] # P(a_jet)

    pt   = jets["pt"]
    eta  = jets["eta"]
    return pt, eta, true_masses, labels, scores

def plot_scores(to_plot, labels, plot_name, outdir):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Missing dependency: {e}.  pip install matplotlib scikit-learn numpy")
        sys.exit(1)

    colors = ["Red", "Lime", "Blue", "Black"] if len(to_plot) == 4 else [plt.colormaps["tab10"](i) for i in range(len(to_plot))] + ["Black"] # to align w ATLAS
    
    fig, ax = plt.subplots()
    bins = np.linspace(0, 1, 50)
    for p, l, c in zip(to_plot, labels, colors):
        ax.hist(p, bins=bins, density=True, alpha=0.6, label=l, color=c)

    ax.set_xlabel("P(a-jet)")
    ax.set_ylabel("Normalised counts")
    ax.legend()
    fig.savefig(outdir / f"{plot_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir}/{plot_name}.pdf")

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    atlas = args.atlas

    pt, eta, truth_mass, labels, scores = _load(args.file, atlas)

    sig = labels == 1
    bkg = labels == 0

    masses = [0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 8] # labels for all
    smpl_labels = [f"mA={m} GeV" for m in masses]
    smpl_labels.append("ATLAS Bkg")

    bkg_sample = scores[bkg] # get bkg sample
    mass_samples = []
    for m in masses: mass_samples.append(scores[sig & (truth_mass.astype(float) == m)]) # get sig samples for each mass
    all_samples = mass_samples + [bkg_sample] # all samples

    plot_scores(all_samples, smpl_labels, plot_name="mass_specific_classifier_scores", outdir=outdir)
    plot_scores([all_samples[0], all_samples[4], all_samples[7], bkg_sample],
                [smpl_labels[0], smpl_labels[4], smpl_labels[7], "ATLAS Bkg"],
                plot_name="atlas_bkg_score", outdir=outdir)


if __name__ == "__main__":
    main()

