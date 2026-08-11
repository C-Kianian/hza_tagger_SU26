#!/usr/bin/env python3
"""Make plots from a scores file for a classifier that was trained on ALL mass points

Produces:
  - Score distributions (signal vs background) for the ATLAS paper masses and all masses
  - ATLAS comparison table; matches the fpr of the ATLAS 2025 paper and compares the efficiency for the ATLAS mass points

Usage
-----
    python analysis/scripts/model_plotting/mass_specific_classifier_plots.py \\
        --file data/test_scores.h5 \\
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="H5 file with scores datasets")
    p.add_argument("--outdir", default="analysis/plots_mass_specific_classifier", help="Output directory")
    p.add_argument('--config', required=True, help='Path to training config file')
    return p.parse_args()

def _load(scores_path: str, score_ds_name: str):
    import h5py
    from common.io import JETS_DATASET, LABELS_DATASET

    with h5py.File(scores_path, "r") as f:
        jets   = f[JETS_DATASET][:]
        labels = f[LABELS_DATASET]["a_jet"][:]
        true_masses = f[JETS_DATASET]["truth_a_mass"][:]
        scores = f[score_ds_name][:, 0] if 'atlas' in score_ds_name else f[score_ds_name][:, 1] # P(a_jet)

    pt   = jets["pt"]
    eta  = jets["eta"]
    return {
        "pt": pt,
        "eta": eta,
        "true_mass": true_masses,
        "labels": labels,
        "scores": scores,
    }

def plot_scores(to_plot, labels, plot_name, outdir):
    if len(to_plot) == 4: colors = ["Red", "Lime", "Blue", "Black"] # to align w ATLAS
    else: colors = [plt.colormaps["tab10"](i) for i in range(10)] + ["Black"] # for all 10 masses + bkg

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

def tradeoff_table(rows, outdir, thresh):
    fig, ax = plt.subplots(figsize=(4, 0.5*len(rows)+1))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Mass [GeV]", "AUC", r"$\epsilon_{\mathrm{sig}}$@~0.7%FPR"],
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    plt.savefig(outdir / f"tradeoff_table_thresh={thresh:.2f}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outdir}/roc_summary_thresh={thresh:.2f}.pdf")

def main():
    try:
        from sklearn.metrics import roc_curve, auc
        import yaml
        from common.parse_yaml import get_tasks
    except ImportError as e:
        print(f"Missing dependency: {e}.  pip install scikit-learn yaml")
        sys.exit(1)

    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the YAML Config
    print(f"Loading configuration: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    names, tasks, _ = get_tasks(config)
    # get only classification task info
    class_scores_names = [f"{name}_scores" for name, task in zip(names, tasks) if task == "ClassificationTask"]
    # gets per (classification) task info into a dict has form of {"task1_scores": {"pt":, "eta":, "true_mass":, ""labels":, "scores":,} ...}
    per_task_dict = {score_name: _load(args.scores, score_name) for score_name in class_scores_names}

    for task, info in per_task_dict.items(): # makes plots per classification task
        pt, eta, truth_mass, labels, scores = info["pt"], info["eta"], info["true_mass"], info["labels"], info["scores"] # get task info
        taskdir = outdir / task
        taskdir.mkdir(parents=True, exist_ok=True)

        sig = labels == 1
        bkg = labels == 0

        masses = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0] # all masses

        sample_info = {}
        for mass in masses: # store info for each mass
            truth_mass = truth_mass.astype(float) # get truth mass
            mass_filter = (sig & (truth_mass == mass)) # setup filter for this mass

            score = scores[mass_filter] # get scores ONLY for this mass

            roc_condition = mass_filter | bkg # get roc info for ONLY this mass vs bkg
            roc_idxs = np.where(roc_condition)
            fpr, tpr, thr = roc_curve(labels[roc_idxs], scores[roc_idxs])

            sample_info[mass] = {
                "label": f"mA={mass} GeV",
                "score": score,
                "fpr":   fpr,
                "tpr":   tpr,
                "thr":   thr,
                "auc":   auc(fpr, tpr),
            }
        sample_info["bkg"] = {
            "label": f"ATLAS Bkg",
            "score": scores[bkg],
        }

        # =============================
        # plot classifier scores
        plot_scores([info["score"] for info in sample_info.values()],
                    [info["label"] for info in sample_info.values()],
                    plot_name="all_classifier_scores", outdir=taskdir)

        plot_scores([sample_info[0.5]["score"], sample_info[2.0]["score"], sample_info[3.5]["score"], sample_info["bkg"]["score"]],
                    [sample_info[0.5]["label"], sample_info[2.0]["label"], sample_info[3.5]["label"], sample_info["bkg"]["label"]],
                    plot_name="mass_specific_classifier_scores", outdir=taskdir)

        # =============================
        # make tpr, fpr, tradeoff table
        wp_bkg = 0.007 # replicate atlas paper
        idx_wp = np.argmin(np.abs(sample_info[3.5]["fpr"] - wp_bkg))
        cut = sample_info[3.5]["thr"][idx_wp]

        rows = [] # get info at this model threshold for ATLAS masses
        for mass in [0.5, 2.0, 3.5]:
            score = sample_info[mass]["score"]

            tpr = np.mean(score > cut)

            rows.append([
                mass,
                f"{sample_info[mass]['auc']:.2f}",
                f"{tpr:.2f}",
            ])
        tradeoff_table(rows, outdir, cut)


if __name__ == "__main__":
    main()

