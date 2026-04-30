#!/usr/bin/env python3
"""Run the converter locally (single process, iterative executor).

Usage
-----
    python converter/run_local.py --config converter/configs/hza_signal.yaml \\
                                  --out data/train.h5 \\
                                  [--max-events 5000]

If --out is given it overrides the train/val/test split and writes a single
combined file — useful for quick tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make repo root importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
import uproot
import awkward as ak

from converter.processors.jet_dumper import process_events
from converter.processors.writer import H5Writer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None, help="Override output path (single file)")
    p.add_argument("--max-events", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = yaml.safe_load(Path(args.config).read_text())

    out_path   = args.out or cfg["output"]["train"]
    chunk_size = cfg.get("chunk_size", 10_000)
    max_events = args.max_events
    tree_name  = cfg.get("tree", "Events")

    print(f"Output: {out_path}")
    print(f"Chunk size: {chunk_size}")

    with H5Writer(out_path) as writer:
        total_jets = 0
        total_a    = 0
        n_processed = 0

        for file_path in cfg["files"]:
            print(f"\nProcessing: {file_path}")
            try:
                f    = uproot.open(file_path)
                tree = f[tree_name]
            except FileNotFoundError:
                print(f"  WARNING: file not found, skipping")
                continue

            n_entries = tree.num_entries
            if max_events is not None:
                n_entries = min(n_entries, max_events - n_processed)
            if n_entries <= 0:
                break

            # Lazy iteration over chunks
            for start in range(0, n_entries, chunk_size):
                stop = min(start + chunk_size, n_entries)
                chunk = tree.arrays(entry_start=start, entry_stop=stop, library="ak")
                arrays = process_events(chunk)

                if len(arrays["jets"]) == 0:
                    continue

                writer.write_chunk(arrays["jets"], arrays["tracks"], arrays["labels"])
                n_a   = int(arrays["labels"]["a_jet"].sum())
                total_jets += len(arrays["jets"])
                total_a    += n_a
                print(f"  events {start}–{stop}: {len(arrays['jets'])} jets  ({n_a} a-jets)")

                n_processed += stop - start
                if max_events is not None and n_processed >= max_events:
                    break

            if max_events is not None and n_processed >= max_events:
                break

        writer.finalize()

    print(f"\nDone. Total jets: {total_jets}  a-jets: {total_a}  ({100*total_a/max(total_jets,1):.1f}%)")
    print(f"Written to: {out_path}")


if __name__ == "__main__":
    main()
