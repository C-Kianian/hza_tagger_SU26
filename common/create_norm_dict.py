#!/usr/bin/env python3
"""Compute per-variable mean and std from the training H5 file and write norm_dict.yaml.

This replaces `salt preprocess` which is not available in SALT 0.11.

Usage
-----
    python tagger/scripts/create_norm_dict.py \
        --input  data/train.h5 \
        --config tagger/configs/hza_variables.yaml \
        --output tagger/configs/norm_dict.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import h5py
import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Path to training H5 file")
    p.add_argument("--config", required=True, help="Path to hza_variables.yaml")
    p.add_argument("--output", required=True, help="Output norm_dict.yaml path")
    p.add_argument("--max-jets", type=int, default=None,
                   help="Cap number of jets used for statistics (default: all)")
    return p.parse_args()


def compute_stats(values: np.ndarray) -> dict:
    """Return mean and std, guarding against zero std."""
    mean = float(np.nanmean(values))
    std  = float(np.nanstd(values))
    if std == 0.0 or not np.isfinite(std):
        std = 1.0
    return {"mean": round(mean, 6), "std": round(std, 6)}


def main():
    args = parse_args()

    variables = yaml.safe_load(Path(args.config).read_text())
    jet_vars   = [v["name"] for v in variables.get("jets",   [])]
    track_vars = [v["name"] for v in variables.get("tracks", [])]

    norm: dict = {}

    with h5py.File(args.input, "r") as f:
        # ── jets ──────────────────────────────────────────────────────────────
        jets_ds = f["jets"]
        n = len(jets_ds) if args.max_jets is None else min(len(jets_ds), args.max_jets)
        print(f"Computing jet stats from {n} jets …")
        norm["jets"] = {}
        for var in jet_vars:
            if var not in jets_ds.dtype.names:
                print(f"  WARNING: jet variable '{var}' not in H5, using mean=0 std=1")
                norm["jets"][var] = {"mean": 0.0, "std": 1.0}
                continue
            vals = jets_ds[var][:n].astype(np.float32)
            vals = vals[np.isfinite(vals)]
            norm["jets"][var] = compute_stats(vals)
            print(f"  jets/{var}: mean={norm['jets'][var]['mean']:.4f}  std={norm['jets'][var]['std']:.4f}")

        # ── tracks ────────────────────────────────────────────────────────────
        tracks_ds = f["tracks"]
        n_trk = len(tracks_ds) if args.max_jets is None else min(len(tracks_ds), args.max_jets)
        print(f"\nComputing track stats from {n_trk} jets ({tracks_ds.shape[1]} tracks/jet) …")
        norm["tracks"] = {}
        for var in track_vars:
            if var == "valid":
                continue
            if var not in tracks_ds.dtype.names:
                print(f"  WARNING: track variable '{var}' not in H5, using mean=0 std=1")
                norm["tracks"][var] = {"mean": 0.0, "std": 1.0}
                continue
            vals = tracks_ds[var][:n_trk].astype(np.float32).ravel()
            # Mask padded (zero-filled) entries using the "valid" flag if available
            if "valid" in tracks_ds.dtype.names:
                valid_mask = tracks_ds["valid"][:n_trk].ravel()
                vals = vals[valid_mask]
            vals = vals[np.isfinite(vals)]
            norm["tracks"][var] = compute_stats(vals)
            print(f"  tracks/{var}:         mean={norm['tracks'][var]['mean']:.4f}  std={norm['tracks'][var]['std']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(norm, f, sort_keys=False)

    print(f"\nNorm dict written to {out_path}")


if __name__ == "__main__":
    main()
