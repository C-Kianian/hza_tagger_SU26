#!/usr/bin/env python3
"""Calculate SHAP feature importances for the ATLAS Jet Classification model."""

import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys

try:
    import torch
    import h5py
    import numpy as np
    import yaml
    import shap
except ImportError as e:
    print(f"Missing dependency: {e}; If Shap, to stay consistent with Salt, run: pip install \"shap\" \"numpy<2\"")
    sys.exit(1)

# Ensure repository root is in python path (matching your eval script)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Path to input test H5 file")
    p.add_argument("--ckpt",   required=True, help="Path to trained SALT checkpoint (.ckpt)")
    p.add_argument("--config", required=True, help="Path to model training YAML config")
    p.add_argument("--output", default="atlas_shap_summary.png", help="Filename for output plot")
    p.add_argument("--nsamples", type=int, default=10000, help="Number of jets to explain (keep small, "
                                                               "10k takes about a minute for ATLAS classifier)")
    return p.parse_args()

def main():
    args = parse_args()

    # load the SALT model
    try:
        from salt.modelwrapper import ModelWrapper
    except ImportError:
        print("SALT not installed. Ensure your virtual environment is active.")
        sys.exit(1)

    print(f"Loading checkpoint: {args.ckpt}")
    model = ModelWrapper.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device) # load model checkpoint on gpu if available

    # get model features from config
    print(f"Loading configuration: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    jet_vars = config["data"]["variables"]["jets"] # look only at jets, ATLAS classifier only uses jet inputs
    input_map = config["data"]["input_map"]        # doing this for other models with tracks/edges becomes trickier
    primary_jet_ds = input_map.get("jets", "jets")

    # load testing dataset
    print(f"Loading data from: {args.input}")
    with h5py.File(args.input, "r") as fin:
        raw_jets = fin[primary_jet_ds][:]

        # ensure this is filtered with ATLAS criteria if evaluating an ATLAS model
        if "atlas_valid" in raw_jets.dtype.names:
            selection_mask = raw_jets["atlas_valid"].astype(bool)
            filtered_jets = raw_jets[selection_mask]
            print(f"Applied atlas_valid selection: {len(filtered_jets)} / {len(raw_jets)} jets passed.")
        else:
            filtered_jets = raw_jets

    # matrix of features per sample (N_samples, 8_features)
    X_matrix = np.stack([filtered_jets[v].astype(np.float32) for v in jet_vars], axis=-1)
    X_matrix = np.nan_to_num(X_matrix, nan=-1.0, posinf=-1.0, neginf=-1.0)

    # prediction function required by shap
    def predict_for_shap(changed_matrix):
        """
        SHAP perturbs feature matrices and passes them here.
        We convert it to a PyTorch dictionary format for SALT.
        """
        with torch.no_grad():
            # make everything float32, avoid mismatches
            tensor_input = torch.as_tensor(changed_matrix, dtype=torch.float32, device=device)

            # make inputs into a dict as salt expects
            inputs = {"jets": tensor_input}
            pad_masks = {} # no tracks = no padding

            # pass inputs to model
            preds, *_ = model(inputs, pad_masks)

            # get the output numbers
            logits = preds["jets"]["jets_classification"] # (Batch, 1)

            # make raw logits into probabilities using sigmoid
            probs = torch.sigmoid(logits).cpu().numpy().squeeze(-1) # (Batch,)
            return probs

    # shap part, initialize explainer
    print(f"Initializing Explainer...")
    # background dataset, the reference state the features are compared against
    background_data = shap.kmeans(X_matrix, 50)

    explainer = shap.KernelExplainer(predict_for_shap, background_data)

    # select the n jets to explain
    X_explain = X_matrix[:args.nsamples]

    print(f"Calculating SHAP values for {args.nsamples} jets. This might take a few minutes...")
    shap_values = explainer.shap_values(X_explain)

    # plot summary
    print(f"Generating summary plot...")
    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=jet_vars,
        show=False
    )

    plt.title(f"ATLAS Classifier Feature Importance (Sample size: {args.nsamples})", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Success! SHAP Plot saved to: {args.output}")

if __name__ == "__main__":
    main()

