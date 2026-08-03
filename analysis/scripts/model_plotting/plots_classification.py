#!/usr/bin/env python3
"""Performance plots for a jet classifier.

Produces:
  - ROC curve (a-jet vs other) overall and in pt bins
  - Score distributions (signal vs background)
  - Efficiency vs jet pT and eta

Usage
-----
    python analysis/scripts/model_plotting/plots_classification.py \\
        --scores data/test_scores.h5 \\
        --outdir analysis/plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True, help="H5 file with jets, labels, scores datasets")
    p.add_argument("--outdir", default="analysis/plots")
    p.add_argument("--wp",     type=float, nargs="+", default=[0.7, 0.85], help="Signal efficiency working points for WP lines")
    p.add_argument('--config', required=True, help='Path to config file')
    return p.parse_args()


def _load(scores_path: str, score_ds_name: str):
    import h5py
    from common.io import JETS_DATASET, LABELS_DATASET

    with h5py.File(scores_path, "r") as f:
        jets   = f[JETS_DATASET][:]
        labels = f[LABELS_DATASET]["a_jet"][:]
        scores = f[score_ds_name][:, 0] if 'atlas' in score_ds_name else f[score_ds_name][:, 1] # P(a_jet)

    pt   = jets["pt"]
    eta  = jets["eta"]
    return {
        "pt": pt,
        "eta": eta,
        "labels": labels,
        "scores": scores,
    }


def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        import yaml
        from common.parse_yaml import get_tasks
    except ImportError as e:
        print(f"Missing dependency: {e}.  pip install matplotlib scikit-learn numpy shap")
        sys.exit(1)

    # Load the YAML Config
    print(f"Loading configuration: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    names, tasks, _ = get_tasks(config)
    # get only classification task info
    class_scores_names = [f"{name}_scores" for name, task in zip(names, tasks) if task == "ClassificationTask"]
    # gets per (classification) task info into a dict has form of {"task1_scores": {"pt":, "eta":, "labels":, "scores":,} ...}
    per_task_dict = {score_name: _load(args.scores, score_name) for score_name in class_scores_names}

    for task, info in per_task_dict.items(): # makes plots per classification task
        pt, eta, labels, scores = info["pt"], info["eta"], info["labels"], info["scores"] # get task info
        taskdir = outdir / task
        taskdir.mkdir(parents=True, exist_ok=True)

        sig = labels == 1
        bkg = labels == 0

        # ── ROC curve ──────────────────────────────────────────────────────────
        fpr, tpr, thr = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(tpr, 1 / (fpr + 1e-9), label=f"HZa tagger  AUC={roc_auc:.4f}")
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("1 / Background efficiency")
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.legend()
        ax.set_title("ROC — a-jet vs other")
        fig.savefig(taskdir / "roc.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved roc.pdf")

        # ── Score distributions ─────────────────────────────────────────────────
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 50)
        ax.hist(scores[sig], bins=bins, density=True, alpha=0.6, label="a-jet (signal)")
        ax.hist(scores[bkg], bins=bins, density=True, alpha=0.6, label="other (background)")
        ax.set_xlabel("P(a-jet)")
        ax.set_ylabel("Normalised counts")
        ax.legend()
        fig.savefig(taskdir / "score_dist.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved score_dist.pdf")

        # ── Working-point lines on ROC ──────────────────────────────────────────
        print("\nWorking points:")
        for wp in args.wp:
            idx = np.searchsorted(tpr, wp)
            if idx < len(tpr):
                score_cut = thr[idx]
                bkg_eff   = fpr[idx]
                print(f"  sig_eff={wp:.0%}  →  score>{score_cut:.4f}  bkg_eff={bkg_eff:.4f}"
                      f"  (1/bkg_eff={1/max(bkg_eff,1e-9):.1f})")

        # ── Efficiency vs pT ────────────────────────────────────────────────────
        wp_main = args.wp[0]
        idx_wp  = np.searchsorted(tpr, wp_main)
        cut     = thr[idx_wp] if idx_wp < len(thr) else 0.5

        pt_bins  = np.array([20, 40, 60, 80, 100, 150, 200, 300, 500])
        pt_cents = 0.5 * (pt_bins[:-1] + pt_bins[1:])
        sig_eff  = []
        bkg_eff  = []

        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            mask_s = sig & (pt >= lo) & (pt < hi)
            mask_b = bkg & (pt >= lo) & (pt < hi)
            sig_eff.append(np.mean(scores[mask_s] > cut) if mask_s.sum() else np.nan)
            bkg_eff.append(np.mean(scores[mask_b] > cut) if mask_b.sum() else np.nan)

        fig, ax = plt.subplots()
        ax.plot(pt_cents, sig_eff, "o-", label="Signal efficiency")
        ax.plot(pt_cents, bkg_eff, "s--", label="Background efficiency")
        ax.axhline(wp_main, ls=":", color="grey", label=f"Target sig eff {wp_main:.0%}")
        ax.set_xlabel("Jet $p_T$ [GeV]")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Efficiency vs $p_T$ (cut at WP {wp_main:.0%})")
        fig.savefig(taskdir / "eff_vs_pt.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved eff_vs_pt.pdf")

        # ── Efficiency vs eta ────────────────────────────────────────────────────
        wp_main = args.wp[0]
        idx_wp  = np.searchsorted(tpr, wp_main)
        cut     = thr[idx_wp] if idx_wp < len(thr) else 0.5

        eta_bins  = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        eta_cents = 0.5 * (eta_bins[:-1] + eta_bins[1:])
        sig_eff  = []
        bkg_eff  = []

        for lo, hi in zip(eta_bins[:-1], eta_bins[1:]):
            mask_s = sig & (eta >= lo) & (eta < hi)
            mask_b = bkg & (eta >= lo) & (eta < hi)
            sig_eff.append(np.mean(scores[mask_s] > cut) if mask_s.sum() else np.nan)
            bkg_eff.append(np.mean(scores[mask_b] > cut) if mask_b.sum() else np.nan)

        fig, ax = plt.subplots()
        ax.plot(eta_cents, sig_eff, "o-", label="Signal efficiency")
        ax.plot(eta_cents, bkg_eff, "s--", label="Background efficiency")
        ax.axhline(wp_main, ls=":", color="grey", label=f"Target sig eff {wp_main:.0%}")
        ax.set_xlabel("Jet $\eta$")
        ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title(f"Efficiency vs $\eta$ (cut at WP {wp_main:.0%})")
        fig.savefig(taskdir / "eff_vs_eta.pdf", bbox_inches="tight")
        plt.close(fig)
        print("Saved eff_vs_eta.pdf")

if __name__ == "__main__":
    main()

