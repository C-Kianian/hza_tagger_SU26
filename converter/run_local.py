#!/usr/bin/env python3
"""Run the converter locally (single process, iterative executor).

By default reads split_fractions from the config and writes separate
train / val / test H5 files.  Pass --out to override and write a single file.

Usage
-----
    python converter/run_local.py --config converter/configs/hza_signal.yaml
    python converter/run_local.py --config converter/configs/hza_signal.yaml \\
                                  --out data/all.h5 --max-events 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
import uproot
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

import warnings
warnings.filterwarnings("ignore", message="Missing cross-reference index", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="coffea.nanoevents.methods.vector will be removed", category=FutureWarning)

from converter.processors.jet_dumper import process_events
from converter.processors.writer import H5Writer
from common.variables import REQUIRED_BRANCHES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="Write single file (disables train/val/test split)")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split shuffle")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())

    single_out = args.out  # None → use split mode
    chunk_size = cfg.get("chunk_size", 10_000)
    max_events = args.max_events
    max_events_per_file = cfg.get("max_events_per_file", None)  # per-file cap from config
    tree_name  = cfg.get("tree", "Events")
    rng        = np.random.default_rng(args.seed)

    fracs = cfg.get("split_fractions", {"train": 0.70, "val": 0.15, "test": 0.15})
    split_names = list(fracs.keys())           # ["train", "val", "test"]
    split_probs = np.array([fracs[k] for k in split_names], dtype=float)
    split_probs /= split_probs.sum()           # normalise in case of rounding

    # ── open output writers up front, stream chunks directly ─────────────────
    # This keeps peak RAM at O(one chunk) rather than O(entire dataset).
    if single_out:
        writers = {None: H5Writer(single_out)}
    else:
        out = cfg["output"]
        writers = {name: H5Writer(out[name]) for name in split_names}

    n_processed  = 0
    total_jets   = 0
    total_a      = 0
    split_counts = {k: 0 for k in (split_names if not single_out else [None])}

    try:
        for file_path in cfg["files"]:
            print(f"\nProcessing: {file_path}")
            try:
                f    = uproot.open(file_path)
                tree = f[tree_name]
            except FileNotFoundError:
                print(f"  WARNING: file not found, skipping")
                continue

            n_entries = tree.num_entries
            if max_events_per_file is not None:
                n_entries = min(n_entries, max_events_per_file)
            if max_events is not None:
                n_entries = min(n_entries, max_events - n_processed)
            if n_entries <= 0:
                break

            for start in range(0, n_entries, chunk_size):
                stop = min(start + chunk_size, n_entries)
                chunk = NanoEventsFactory.from_root(
                    {file_path: tree_name},
                    entry_start=start,
                    entry_stop=stop,
                    schemaclass=NanoAODSchema,
                    uproot_options={"filter_name": REQUIRED_BRANCHES},
                ).events()
                chunk = ak.Array(chunk.compute())
                arrays_out = process_events(chunk)

                n_events_in_chunk = stop - start

                n_chunk = len(arrays_out["jets"])
                if n_chunk == 0:
                    continue

                n_a = int(arrays_out["labels"]["a_jet"].sum())
                print(f"  events {start}\u2013{stop}: {n_chunk} jets  ({n_a} a-jets)")
                total_jets += n_chunk
                total_a    += n_a

                if single_out:
                    writers[None].write_chunk(
                        arrays_out["jets"], arrays_out["tracks"], arrays_out["labels"]
                    )
                    split_counts[None] += n_chunk
                else:
                    # Assign each jet in this chunk to a split randomly
                    ids = rng.choice(len(split_names), size=n_chunk, p=split_probs)
                    for i, name in enumerate(split_names):
                        mask = ids == i
                        if not mask.any():
                            continue
                        writers[name].write_chunk(
                            arrays_out["jets"][mask],
                            arrays_out["tracks"][mask],
                            arrays_out["labels"][mask],
                        )
                        split_counts[name] += int(mask.sum())

                # Release chunk memory explicitly
                del chunk, arrays_out

                n_processed += n_events_in_chunk
                if max_events is not None and n_processed >= max_events:
                    break

            if max_events is not None and n_processed >= max_events:
                break

    finally:
        for w in writers.values():
            w.finalize()

    if total_jets == 0:
        print("No jets found — check your config files.")
        return

    print(f"\nTotal jets: {total_jets}  a-jets: {total_a}  ({100*total_a/max(total_jets,1):.1f}%)")

    if single_out:
        print(f"Written to: {single_out}")
    else:
        out = cfg["output"]
        for name in split_names:
            print(f"  {name:5s}: {split_counts[name]:6d} jets  → {out[name]}")


if __name__ == "__main__":
    main()
