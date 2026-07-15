#!/usr/bin/env python3
"""Run a trained SALT checkpoint over a test H5 file and write tagger scores.

The output H5 mirrors the input but adds a dataset "scores" with shape (N, 2):
  column 0 → P(background)
  column 1 → P(a_jet)

Usage
-----
    python analysis/scripts/eval_to_h5.py \\
        --input  data/test.h5 \\
        --ckpt   logs/hza_tagger/.../ckpts/epoch=080-val_loss=0.06297.ckpt \\
        --config tagger/configs/hza_train.yaml \\
        --output data/test_scores.h5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Expected true or false")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument('--atlas', type=str2bool, default=False, help='include if evaluating ATLAS models')
    p.add_argument('--regression', type=str2bool, default=False, help='include if evaluating a regression model')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        import h5py
        import numpy as np
        import yaml
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    # Load SALT ModelWrapper via Lightning's standard checkpoint loading
    try:
        from salt.modelwrapper import ModelWrapper
    except ImportError:
        print("SALT not installed.  Run: bash tagger/scripts/setup_salt.sh")
        sys.exit(1)

    print(f"Loading checkpoint: {args.ckpt}")
    model = ModelWrapper.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load the YAML Config
    print(f"Loading configuration: {args.config}")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # get vars from the model training config
    variables_config = config["data"]["variables"]
    input_map = config["data"]["input_map"]

    # Copy input to output (preserves jets/tracks/labels).
    # Remove any stale output file first to avoid HDF5 lock errors on re-runs.
    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()
    shutil.copy2(args.input, args.output)

    from common.io import JETS_DATASET, TRACKS_DATASET

    with (h5py.File(args.input, "r") as fin, h5py.File(args.output, "a") as fout):
        # use jet dataset for count of n events
        primary_jet_ds = input_map.get("jets", "jets")
        raw_jets = fin[primary_jet_ds]

        n_jets = fin[primary_jet_ds].shape[0] # n events

        ds_name = "predictions" if args.regression else "scores"
        ds_shape = (n_jets, 1) if args.regression or args.atlas else (n_jets, 2) # only our classifier outputs two predictions

        if ds_name in fout: # make empty scores dataset
            del fout[ds_name]
        output_ds = fout.create_dataset(ds_name, shape=ds_shape, dtype=np.float32, compression="gzip")


        # batch processing
        for start in range(0, n_jets, args.batch_size):
            stop = min(start + args.batch_size, n_jets)

            batch_jets = raw_jets[start:stop] # jets for this batch

            # if ATLAS apply ATLAS selection criteria
            if args.atlas and "atlas_valid" in raw_jets.dtype.names:
                selection_mask = batch_jets["atlas_valid"].astype(bool)
            else:
                selection_mask = np.ones(len(batch_jets), dtype=bool) # keep all if not specified

            # if the entire batch fails, skip to the next batch
            if not np.any(selection_mask):
                continue

            inputs = {}
            pad_masks = {}

            # loop over all input streams defined in YAML
            for input_name, var_list in variables_config.items():
                # find the H5 dataset this group maps to (ie EDGE maps to tracks)
                h5_dataset_name = input_map[input_name]
                batch_data = fin[h5_dataset_name][start:stop]

                filtered_batch = batch_data[selection_mask] # in case selection criteria applied

                # stack only the variables listed in YAML
                np_arr = np.stack([filtered_batch[v].astype(np.float32) for v in var_list], axis=-1)
                np_arr = np.nan_to_num(np_arr, nan=-1.0, posinf=-1.0, neginf=-1.0) # remove placeholders
                inputs[input_name] = torch.from_numpy(np_arr).to(device)

                # apply padding if applicable
                if "valid" in batch_data.dtype.names:
                    valid_t = torch.from_numpy(batch_data["valid"]).to(device)
                    pad_masks[input_name] = ~valid_t  # True = padded/ignored

            # ── Forward pass ─────────────────────────────────────────────────
            # SALT 0.11 ModelWrapper.forward(inputs, pad_masks) → (preds, loss, ...)
            with torch.no_grad():
                preds, *_ = model(inputs, pad_masks)

            if args.regression: # mass regression task
                reg_scores = preds["jets"]["jet_regression"]
                processed_outputs = reg_scores.cpu().numpy().squeeze(-1) # (num_selected_jets,)

                batch_mass = fout[primary_jet_ds]['regression_a_mass', start:stop] # get mass preds for the batch
                batch_mass[selection_mask] = processed_outputs # update the jets passing selection criteria
                if args.atlas: fout[primary_jet_ds]['atlas_regression_a_mass', start:stop] = batch_mass # write info to file
                else: fout[primary_jet_ds]['regression_a_mass', start:stop] = batch_mass # write info to file
                processed_outputs = processed_outputs[:, np.newaxis]
            else: # classification task
                # preds is a dict: {"jets": {"jets_classification": logits}}
                logits = preds["jets"]["jets_classification"] # (B, out_dim), account for different loss funcs
                processed_outputs = torch.sigmoid(logits).cpu().numpy() if args.atlas else torch.softmax(logits, dim=-1).cpu().numpy() # make into probs (B, out_dim)

            output_ds[start:stop] = processed_outputs # write batch predictions
            print(f"  {stop}/{n_jets} jets scored")

    print(f"\nScores written to: {args.output}")


if __name__ == "__main__":
    main()

