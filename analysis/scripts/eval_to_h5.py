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
        import h5py
        import numpy as np
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

    # Copy input to output (preserves jets/tracks/labels).
    # Remove any stale output file first to avoid HDF5 lock errors on re-runs.
    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()
    shutil.copy2(args.input, args.output)

    from common.io import JETS_DATASET, TRACKS_DATASET

    with h5py.File(args.input, "r") as fin, h5py.File(args.output, "a") as fout:
        n_jets = fin[JETS_DATASET].shape[0]

        # Remove existing scores dataset if present (e.g. from a previous run)
        if "scores" in fout:
            del fout["scores"]
        scores_ds = fout.create_dataset(
            "scores", shape=(n_jets, 2), dtype=np.float32, compression="gzip"
        )

        for start in range(0, n_jets, args.batch_size):
            stop  = min(start + args.batch_size, n_jets)
            jets_batch   = fin[JETS_DATASET][start:stop]
            tracks_batch = fin[TRACKS_DATASET][start:stop]

            # ── Jets: drop a_jet label, keep only kinematic input features ────
            #jet_input_fields = [f for f in jets_batch.dtype.names if f != "a_jet"]
            #jets_np = np.stack([jets_batch[f] for f in jet_input_fields], axis=-1).astype(np.float32)
            #jets_t  = torch.from_numpy(jets_np).to(device)           # (B, n_jet_vars)

            # ── Tracks ────────────────────────────────────────────────────────
            #track_input_fields = [f for f in tracks_batch.dtype.names if f != "valid"]
            #tracks_np = np.stack(
            #    [tracks_batch[f].astype(np.float32) for f in track_input_fields], axis=-1
            #)                                                         # (B, T, n_track_vars)
            #tracks_t  = torch.from_numpy(tracks_np).to(device)
            #valid_t   = torch.from_numpy(tracks_batch["valid"]).to(device)  # (B, T) bool
            #pad_mask  = ~valid_t                                      # True = padded


            # ======================= VIBECODED FIX ==========================
            # ── Jets: Explicitly select the 4 features the model expects ────
            jet_input_fields = ["pt", "eta", "phi", "mass"]
            jets_np = np.stack([jets_batch[f] for f in jet_input_fields], axis=-1).astype(np.float32)
            jets_t  = torch.from_numpy(jets_np).to(device)           # (B, 4)

            # ── Tracks: Explicitly select the features used during training ──
            # (Explicitly listing these prevents a secondary crash if your track dataset has extra fields)
            track_input_fields = [
                "pt", "eta_rel", "phi_rel", "mass", "charge", 
                "pdgId", "dxy", "dz", "dxySig", "dzSig", 
                "trkQuality", "puppiWeight"
            ]
            tracks_np = np.stack(
                [tracks_batch[f].astype(np.float32) for f in track_input_fields], axis=-1
            )                                                                 # (B, T, n_track_vars)
            tracks_t  = torch.from_numpy(tracks_np).to(device)
            valid_t   = torch.from_numpy(tracks_batch["valid"]).to(device)  # (B, T) bool
            pad_mask  = ~valid_t                                              # True = padded

            # ── Forward pass ─────────────────────────────────────────────────
            # SALT 0.11 ModelWrapper.forward(inputs, pad_masks) → (preds, loss, ...)
            inputs    = {"jets": jets_t, "tracks": tracks_t}
            pad_masks = {"tracks": pad_mask}

            with torch.no_grad():
                preds, *_ = model(inputs, pad_masks)

            # preds is a dict: {"jets": {"jets_classification": logits}}
            logits = preds["jets"]["jets_classification"]             # (B, 2)
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

            scores_ds[start:stop] = probs
            print(f"  {stop}/{n_jets} jets scored")

    print(f"\nScores written to: {args.output}")


if __name__ == "__main__":
    main()
