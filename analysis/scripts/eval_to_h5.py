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
        --config tagger/configs/v1_v2_hza_train.yaml \\
        --output data/test_scores.h5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
import numpy as np

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

def make_dataset(fout, name, shape):
    ds_name = name
    ds_shape = shape

    if ds_name in fout: # make empty scores dataset
        del fout[ds_name]
    output_ds = fout.create_dataset(ds_name, shape=ds_shape, dtype=np.float32, compression="gzip")
    return output_ds


def main():
    args = parse_args()

    try:
        import torch
        import h5py
        import yaml
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    # Load SALT ModelWrapper via Lightning's standard checkpoint loading
    try:
        from salt.modelwrapper import ModelWrapper
    except ImportError as e:
        print("SALT not installed.  Run: bash tagger/scripts/setup_salt.sh")
        print(f"Full error: {e}")
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
    from common.parse_yaml import get_tasks

    with (h5py.File(args.input, "r") as fin, h5py.File(args.output, "a") as fout):
        # use jet dataset for count of n events
        primary_jet_ds = input_map.get("jets", "jets")
        raw_jets = fin[primary_jet_ds]

        n_jets = fin[primary_jet_ds].shape[0] # n events

        output_datasets_dict = {} # dict to store output datasets, needed for multitasking
        names, task_types, losses = get_tasks(config) # get the task info, needed for multitasking
        task_lookup = dict(zip(names, task_types)) # lookup to ensure what is returned matches yaml cfg

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

                np_arr = np.stack([filtered_batch[v].astype(np.float32) for v in var_list], axis=-1)
                np_arr = np.nan_to_num(np_arr, nan=-1.0, posinf=-1.0, neginf=-1.0) # remove placeholders

                inputs[input_name] = torch.from_numpy(np_arr).to(device)

                # apply padding if applicable
                if input_name == 'tracks' and "valid" in filtered_batch.dtype.names:
                    valid_t = torch.from_numpy(filtered_batch["valid"]).to(device)
                    pad_masks[input_name] = ~valid_t  # True = padded/ignored

            # ── Forward pass ─────────────────────────────────────────────────
            # SALT 0.13 ModelWrapper.forward(inputs, pad_masks) → (preds, loss, ...)
            with torch.no_grad():
                preds, *_ = model(inputs, pad_masks) # preds is a dict: {"jets": {"jets_classification": logits}}

            jet_level_preds = preds["jets"] # if predictions are made on the track level this script does nothing

            for name, score in jet_level_preds.items():
                # create dataset based on the results from preds
                score_name = f"{name}_preds" if 'regress' in name else f"{name}_scores" # name of dataset to store in
                if score_name not in output_datasets_dict:
                    # score is a torch tensor with shape (B,) or (B, out_dim)
                    if score.ndim == 1: shape = (n_jets,) # get shapes of model outputs
                    else: shape = (n_jets, score.shape[1])
                    output_datasets_dict[score_name] = make_dataset(fout, score_name, shape) # create datasets w/ correct shapes


                task_type = task_lookup[name] # predictions for classifier/regressor
                if task_type is None:
                    print(f"Task '{name}' not found in config, skipping...")
                    continue

                if task_type == "RegressionTask": processed_outputs = score.cpu().numpy().squeeze(-1)
                elif task_type == "ClassificationTask":
                    if "atlas" in name:
                        processed_outputs = torch.sigmoid(score).cpu().numpy() # ATLAS uses BCE requires different handling
                    else:
                        processed_outputs = torch.softmax(score, dim=-1).cpu().numpy()
                else: continue

                out_ds = output_datasets_dict[score_name] # store in specific dataset

                batch_size = stop - start
                out_shape = out_ds.shape[1:]      # () or (2,)

                if len(out_shape) == 0: batch_output = np.full(batch_size, np.nan, dtype=np.float32)
                else: batch_output = np.full((batch_size, *out_shape), np.nan, dtype=np.float32)

                if task_type == "RegressionTask":
                    # write info to file, for regression tasks where the output may be used as input to another model
                    if "atlas" in name: fout[primary_jet_ds]["atlas_regression_a_mass", start:stop] = processed_outputs
                    else: fout[primary_jet_ds]["regression_a_mass", start:stop] = processed_outputs
                    processed_outputs = processed_outputs[:, None]

                batch_output[selection_mask] = processed_outputs
                out_ds[start:stop] = batch_output # write batch predictions

            print(f"  {stop}/{n_jets} jets scored")

    print(f"\nScores written to: {args.output}")


if __name__ == "__main__":
    main()

