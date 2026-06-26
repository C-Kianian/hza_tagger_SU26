#!/usr/bin/env python3
"""Split merged/processed H5 files into train/val/test datasets.

Reads chunks from multiple input H5 files and randomly scatters the jets
into separate train, validation, and test H5 files for model training.

Usage
-----
    python converter/split_h5.py -i "data/background/*.h5" "data/signal/*.h5" \
                                 --out-dir data/ml_inputs/ \
                                 --train 0.7 --val 0.15 --test 0.15
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Ensure local imports work for the writer
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np

from converter.processors.writer import H5Writer


def parse_args():
    p = argparse.ArgumentParser(description="Split processed H5 files into train/val/test splits.")
    p.add_argument("-i", "--inputs", nargs="+", required=True, help="Input H5 file(s) or wildcards")
    p.add_argument("--out-dir", required=True, type=str, help="Directory to save train.h5, val.h5, test.h5")
    p.add_argument("--name", type=str, default='', help="name to be added to the files")

    # Split fractions
    p.add_argument("--train", type=float, default=0.70, help="Training fraction")
    p.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    p.add_argument("--test", type=float, default=0.15, help="Testing fraction")
    
    p.add_argument("--chunk-size", type=int, default=50_000, help="Rows to read at once to save RAM")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible splitting")
    return p.parse_args()


def main():
    args = parse_args()

    # ─── 1. EXPAND WILDCARDS & VALIDATE INPUTS ───────────────────────────────
    expanded_files = []
    for path in args.inputs:
        if "*" in path:
            expanded_files.extend(glob.glob(path))
        else:
            expanded_files.append(path)

    if not expanded_files:
        raise FileNotFoundError("Error: No H5 files found matching the provided inputs.")

    print(f"Found {len(expanded_files)} input H5 files to split.")

    # ─── 2. SETUP SPLIT MATH & DIRECTORIES ───────────────────────────────────
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    split_names = [f"train", f"val", f"test"]
    split_probs = np.array([args.train, args.val, args.test], dtype=float)
    
    # Normalize probabilities in case they don't perfectly sum to 1.0
    split_probs /= split_probs.sum()

    # ─── 3. INITIALIZE WRITERS ───────────────────────────────────────────────
    n = args.name
    if n != '': n = '_' + n

    writers = {
        "train": H5Writer(str(outdir / f"train{n}.h5")),
        "val": H5Writer(str(outdir / f"val{n}.h5")),
        "test": H5Writer(str(outdir / f"test{n}.h5"))
    }

    split_counts = {name: 0 for name in split_names}
    total_jets = 0

    # ─── 4. CORE PROCESSING LOOP ─────────────────────────────────────────────
    try:
        for file_path in expanded_files:
            print(f"\nReading: {file_path}")
            
            try:
                # Open the H5 file in read-only mode
                with h5py.File(file_path, "r") as fin:
                    # Look up the keys dynamically (assuming standard jets, tracks, labels)
                    # We use "jets" to determine the total length of the dataset
                    dataset_keys = list(fin.keys())
                    if "jets" not in dataset_keys:
                        print(f"  WARNING: 'jets' dataset missing in {file_path}. Skipping.")
                        continue
                    
                    n_entries = fin["jets"].shape[0]
                    if n_entries == 0:
                        continue

                    # Stream through the file in chunks so we don't blow up RAM
                    for start in range(0, n_entries, args.chunk_size):
                        stop = min(start + args.chunk_size, n_entries)
                        
                        # Eagerly load the chunk into numpy arrays
                        jets_chunk   = fin["jets"][start:stop]
                        tracks_chunk = fin["tracks"][start:stop]
                        labels_chunk = fin["labels"][start:stop]
                        
                        n_chunk = len(jets_chunk)
                        if n_chunk == 0:
                            continue

                        # Generate a random integer array (0=train, 1=val, 2=test) mapped to the chunk
                        ids = rng.choice(len(split_names), size=n_chunk, p=split_probs)

                        # Mask and route the data to the respective writers
                        for i, name in enumerate(split_names):
                            mask = (ids == i)
                            if not mask.any():
                                continue
                            
                            writers[name].write_chunk(
                                jets_chunk[mask],
                                tracks_chunk[mask],
                                labels_chunk[mask]
                            )
                            split_counts[name] += int(mask.sum())

                        total_jets += n_chunk
                        print(f"  Processed {stop}/{n_entries} events...")

            except (OSError, BlockingIOError) as e:
                print(f"  FAILED to open {file_path}: {e}")

    finally:
        # Guarantee all files are cleanly finalized and closed
        for w in writers.values():
            w.finalize()

    # ─── 5. SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "="*40)
    print("SPLIT COMPLETE")
    print("="*40)
    print(f"Total jets processed: {total_jets}")
    for name in split_names:
        print(f"  {name:5s}: {split_counts[name]:8d} jets -> {outdir / f'{name}{n}.h5'}")


if __name__ == "__main__":
    main()
