#!/usr/bin/env python3
"""Run a trained SALT checkpoint over a test H5 file and write tagger scores.

The output H5 mirrors the input but adds a dataset "scores" with shape (N, 2):
  column 0 → P(other)
  column 1 → P(a_jet)

Usage
-----
    python analysis/scripts/eval_to_h5.py \\
        --input  data/test.h5 \\
        --ckpt   logs/hza_tagger/checkpoints/best.ckpt \\
        --config tagger/configs/hza_train.yaml \\
        --output data/test_scores.h5
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import torch
        import yaml
        import h5py
        import numpy as np
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    # Copy input to output (preserves jets/tracks/labels)
    shutil.copy2(args.input, args.output)

    # Load SALT model
    try:
        from salt.utils.loading import load_checkpoint
    except ImportError:
        print("SALT not installed.  Run: bash tagger/scripts/setup_salt.sh")
        sys.exit(1)

    model = load_checkpoint(args.ckpt, args.config)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    from common.io import JETS_DATASET, TRACKS_DATASET, LABELS_DATASET
    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "a") as fout:
        n_jets  = fin[JETS_DATASET].shape[0]
        n_tracks = fin[TRACKS_DATASET].shape[1]

        scores_ds = fout.require_dataset(
            "scores", shape=(n_jets, 2), dtype=np.float32, compression="gzip"
        )

        for start in range(0, n_jets, args.batch_size):
            stop   = min(start + args.batch_size, n_jets)
            jets   = fin[JETS_DATASET][start:stop]
            tracks = fin[TRACKS_DATASET][start:stop]

            # Convert structured arrays to float tensors
            jet_t = torch.tensor(
                np.stack([jets[f] for f in jets.dtype.names], axis=-1),
                dtype=torch.float32, device=device
            )
            track_fields = [f for f in tracks.dtype.names if f != "valid"]
            track_t = torch.tensor(
                np.stack([tracks[f] for f in track_fields], axis=-1),
                dtype=torch.float32, device=device
            )
            valid_t = torch.tensor(tracks["valid"], dtype=torch.bool, device=device)

            with torch.no_grad():
                logits = model(jet_t, track_t, valid_t)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            scores_ds[start:stop] = probs
            print(f"  {stop}/{n_jets} jets scored")

    print(f"\nScores written to: {args.output}")


if __name__ == "__main__":
    main()
