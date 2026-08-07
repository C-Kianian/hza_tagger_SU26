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
import time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Path to training H5 file")
    p.add_argument("--config", required=True, help="Path to hza_variables.yaml")
    p.add_argument("--output", required=True, help="Output norm_dict.yaml path")
    p.add_argument("--max-jets", type=int, default=None, help="Cap number of jets used for stats (default: 25 million)")
    p.add_argument("--chunk-size", type=int, default=131072, help="Rows to process at a time")
    return p.parse_args()


def compute_stats_chunked(dataset, variables: list[str], max_rows: int | None, chunk_size: int, use_valid_flag: bool = False) -> dict:
    """Computes mean and std in a single pass over the dataset to minimize disk I/O."""
    n_total = len(dataset)
    n = n_total if max_rows is None else min(n_total, max_rows)

    # Separate variables that actually exist in the dataset
    valid_vars = [v for v in variables if v in dataset.dtype.names]
    missing_vars = [v for v in variables if v not in dataset.dtype.names]

    # Accumulators (using float64 to prevent catastrophic cancellation on large datasets)
    counts = {v: 0 for v in valid_vars}
    sums = {v: 0.0 for v in valid_vars}
    sq_sums = {v: 0.0 for v in valid_vars}

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        # Read the block of data ONCE from disk
        chunk = dataset[start:end]

        # Compute the valid mask ONCE per chunk for tracks
        if use_valid_flag and "valid" in chunk.dtype.names:
            base_mask = chunk["valid"].astype(bool)
        else:
            base_mask = np.ones(chunk.shape, dtype=bool)

        for var in valid_vars:
            # Extract data, apply valid mask, and cast to float64
            vals = chunk[var][base_mask].astype(np.float64)

            # Filter finite values
            finites = np.isfinite(vals)
            vals = vals[finites]

            counts[var] += len(vals)
            sums[var] += np.sum(vals)
            sq_sums[var] += np.sum(vals**2)

    # Finalize stats
    results = {}
    for var in valid_vars:
        c = counts[var]
        if c == 0:
            results[var] = {"mean": 0.0, "std": 1.0}
        else:
            mean = sums[var] / c
            variance = (sq_sums[var] / c) - (mean ** 2)
            std = np.sqrt(max(0.0, variance))  # max() guards against negative zeros from floating point math

            if std == 0.0 or not np.isfinite(std):
                std = 1.0

            results[var] = {"mean": round(float(mean), 6), "std": round(float(std), 6)}
            print(f"  {var:<20} mean={results[var]['mean']:>8.4f}  std={results[var]['std']:>8.4f}")

    for var in missing_vars:
        print(f"  WARNING: variable '{var}' not in H5, using mean=0 std=1")
        results[var] = {"mean": 0.0, "std": 1.0}

    return results


def main():
    t1 = time.time()
    args = parse_args()

    variables = yaml.safe_load(Path(args.config).read_text())
    jet_vars   = [v["name"] for v in variables.get("jets",   [])] # lists of vars
    track_vars = [v["name"] for v in variables.get("tracks", []) if v["name"] != "valid"]

    norm: dict = {}

    with h5py.File(args.input, "r") as f:
        print(f"Computing jet stats from up to {args.max_jets or 'all'} jets...")
        norm["jets"] = compute_stats_chunked(
            f["jets"],
            jet_vars,
            args.max_jets,
            args.chunk_size,
            use_valid_flag=False
        )

        print(f"\nComputing track stats from up to {args.max_jets or 'all'} jets...")
        norm["tracks"] = compute_stats_chunked(
            f["tracks"],
            track_vars,
            args.max_jets,
            args.chunk_size,
            use_valid_flag=True
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(norm, f, sort_keys=False)

    print(f"\nNorm dict written to {out_path}")
    print("total time", round(time.time() - t1, 2))


if __name__ == "__main__":
    main()
