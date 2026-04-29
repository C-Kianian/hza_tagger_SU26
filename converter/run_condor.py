#!/usr/bin/env python3
"""Submit converter jobs to HTCondor via dask_jobqueue.

Each input file becomes one condor job. Results are merged afterwards.

Usage
-----
    python converter/run_condor.py --config converter/configs/hza_signal.yaml \\
                                   --outdir data/chunks/ \\
                                   [--merge]

Requirements
------------
    pip install dask dask-jobqueue

DESY NAF example
----------------
The HTCondorCluster settings below are tuned for DESY NAF.
Adjust memory/disk/cores for your site.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True, help="Directory for per-file H5 outputs")
    p.add_argument("--merge", action="store_true", help="Merge chunk files after all jobs complete")
    p.add_argument("--max-workers", type=int, default=20)
    return p.parse_args()


def convert_one_file(file_path: str, out_path: str, cfg: dict):
    """Worker function executed on the condor node."""
    import uproot
    from converter.processors.jet_dumper import process_events
    from converter.processors.writer import H5Writer

    tree_name  = cfg.get("tree", "Events")
    chunk_size = cfg.get("chunk_size", 10_000)

    with H5Writer(out_path) as writer:
        f    = uproot.open(file_path)
        tree = f[tree_name]
        for start in range(0, tree.num_entries, chunk_size):
            stop   = min(start + chunk_size, tree.num_entries)
            chunk  = tree.arrays(entry_start=start, entry_stop=stop, library="ak")
            arrays = process_events(chunk)
            if len(arrays["jets"]) > 0:
                writer.write_chunk(arrays["jets"], arrays["tracks"], arrays["labels"])
        writer.finalize()


def merge_files(outdir: Path, merged_path: Path):
    """Concatenate per-file H5 chunks into a single file."""
    import h5py
    import numpy as np
    from common.io import JETS_DATASET, TRACKS_DATASET, LABELS_DATASET

    chunks = sorted(outdir.glob("chunk_*.h5"))
    print(f"Merging {len(chunks)} chunk files → {merged_path}")

    with h5py.File(merged_path, "w") as fout:
        first = True
        for chunk_path in chunks:
            with h5py.File(chunk_path, "r") as fin:
                jets   = fin[JETS_DATASET][:]
                tracks = fin[TRACKS_DATASET][:]
                labels = fin[LABELS_DATASET][:]
            if first:
                fout.create_dataset(JETS_DATASET,   data=jets,   maxshape=(None,),          compression="gzip")
                fout.create_dataset(TRACKS_DATASET, data=tracks, maxshape=(None, tracks.shape[1]), compression="gzip")
                fout.create_dataset(LABELS_DATASET, data=labels, maxshape=(None,),          compression="gzip")
                first = False
            else:
                for ds, arr in [(JETS_DATASET, jets), (TRACKS_DATASET, tracks), (LABELS_DATASET, labels)]:
                    old = fout[ds].shape[0]
                    fout[ds].resize(old + len(arr), axis=0)
                    fout[ds][old:] = arr


def main():
    args   = parse_args()
    cfg    = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        from dask_jobqueue import HTCondorCluster
        from dask.distributed import Client
    except ImportError:
        print("dask-jobqueue not installed.  Run: pip install dask dask-jobqueue")
        sys.exit(1)

    cluster = HTCondorCluster(
        cores=1,
        memory="4GB",
        disk="10GB",
        log_directory=str(outdir / "logs"),
        # DESY NAF flavour — adjust for your site
        job_extra_directives={
            "+RequestRuntime": 3600,
            "universe": "vanilla",
        },
    )
    cluster.scale(min(args.max_workers, len(cfg["files"])))
    client = Client(cluster)

    futures = []
    for i, file_path in enumerate(cfg["files"]):
        out_path = str(outdir / f"chunk_{i:04d}.h5")
        fut = client.submit(convert_one_file, file_path, out_path, cfg)
        futures.append(fut)

    print(f"Submitted {len(futures)} jobs — watching …")
    client.gather(futures)
    print("All jobs complete.")

    if args.merge:
        merge_files(outdir, outdir.parent / "merged.h5")


if __name__ == "__main__":
    main()
