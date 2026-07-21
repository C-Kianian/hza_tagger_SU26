#!/usr/bin/env python3
"""Make plots from > 1 scores files for classifiers that were trained on ALL mass points

Produces:
  - Overlaid ROC curve for all the files

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

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Missing dependency: {e}.  pip install matplotlib scikit-learn numpy")
    sys.exit(1)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs='+', required=True, help="H5 files with scores datasets")
    p.add_argument("--outdir", default="analysis/overlaid_classifier_plots", help="Output directory")
    return p.parse_args()

def _load(scores_path: str):
    import h5py
    from common.io import JETS_DATASET, LABELS_DATASET

    with h5py.File(scores_path, "r") as f:
        jets   = f[JETS_DATASET][:]
        labels = f[LABELS_DATASET]["a_jet"][:]
        true_masses = f[JETS_DATASET]["truth_a_mass"][:]
        name = (Path(scores_path).stem.removeprefix("test_all_samples_filtered_w_atlas_valid_").removesuffix("_scores"))
        try:
            scores = f["scores"][:, 1] # P(a_jet)
        except Exception as e:
            print(f"Allowed {e} to pass, assuming this model is an ATLAS classifier")
            scores = f["scores"][:, 0] # P(signal)

    pt   = jets["pt"]
    eta  = jets["eta"]
    return pt, eta, true_masses, name, labels, scores

def plot_roc(file_dicts, outdir):
    try:
        from sklearn.metrics import roc_curve, auc
    except ImportError as e:
        print(f"Missing dependency: {e}.  pip install matplotlib scikit-learn numpy")
        sys.exit(1)

    fig, ax = plt.subplots()

    for info in file_dicts.values():
        labels = info["labels"] # get info from file dict
        scores = info["scores"]
        name = info["name"]

        fpr, tpr, thr = roc_curve(labels, scores) # calc auc and roc
        roc_auc = auc(fpr, tpr)

        info.update({ # save info
            "fpr": fpr,
            "tpr": tpr,
            "thr": thr,
            "auc": roc_auc,
        })

        ax.plot(tpr, 1 / (fpr + 1e-9), label=f"{name} AUC={roc_auc:.4f}") # plot info

    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("1 / Background efficiency")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title("ROC — a-jet vs other")
    fig.savefig(outdir / "overlaid_roc.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir}/overlaid_roc.pdf")

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    file_dicts = {} # store info from each file
    for i, f in enumerate(args.files):
        pt, eta, truth_mass, name, labels, scores = _load(f)
        file_dicts[f"file_{i}"] = {"name": name, "pt": pt, "eta": eta, "truth_mass": truth_mass, "labels": labels, "scores": scores}

    # =========================
    # plot overlaid ROC curves
    plot_roc(file_dicts, outdir)

if __name__ == "__main__":
    main()

