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

import os
# Tells HDF5 to ignore strict network filesystem locks for this merge process, suggested by Gemini
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import glob
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
    import os
    # Tells HDF5 to ignore strict network filesystem locks for this merge process, suggested by Gemini
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    import warnings
    import awkward as ak
    import uproot
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
    from converter.processors.jet_dumper import process_events
    from converter.processors.writer import H5Writer
    from common.variables import REQUIRED_BRANCHES


    warnings.filterwarnings("ignore", message="Missing cross-reference index", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="coffea.nanoevents.methods.vector will be removed", category=FutureWarning)

    schema_map = {"NanoAODSchema": NanoAODSchema, "PFNanoAODSchema": PFNanoAODSchema}
    schema     = schema_map.get(cfg.get("nano_schema", "NanoAODSchema"), NanoAODSchema)

    tree_name           = cfg.get("tree", "Events")
    chunk_size          = cfg.get("chunk_size", 10_000)
    max_events_per_file = cfg.get("max_events_per_file", None)

    tree      = uproot.open(f"{file_path}:{tree_name}")
    n_entries = tree.num_entries
    if max_events_per_file is not None:
        n_entries = min(n_entries, max_events_per_file)

    writer = H5Writer(out_path)
    try:
        for start in range(0, n_entries, chunk_size):
            stop  = min(start + chunk_size, n_entries)
            chunk = NanoEventsFactory.from_root(
                {file_path: tree_name},
                entry_start=start,
                entry_stop=stop,
                schemaclass=schema,
                uproot_options={"filter_name": REQUIRED_BRANCHES},
            ).events()
            chunk  = ak.Array(chunk.compute())
            arrays = process_events(chunk)
            if len(arrays["jets"]) > 0:
                writer.write_chunk(arrays["jets"], arrays["tracks"], arrays["labels"])
            del chunk, arrays
    finally:
        writer.finalize()


def merge_files(outdir: Path, merged_path: Path):
    """Concatenate per-file H5 chunks into a single file."""
    import h5py
    import numpy as np
    import time
    from common.io import JETS_DATASET, TRACKS_DATASET, LABELS_DATASET
	
    chunks = sorted(outdir.glob("chunk_*.h5"))
    print(f"Merging {len(chunks)} chunk files → {merged_path}")

    with h5py.File(merged_path, "w") as fout:
        first = True
        for chunk_path in chunks:
            
            # VIBECODED BLOCK START
            
            # ─── EXTRA CHECK: Skip if file is empty/corrupted at the OS level
            #if not chunk_path.exists() or chunk_path.stat().st_size == 0:
            #    print(f"Skipping corrupted/empty file (0 bytes): {chunk_path.name}")
            #    continue
			# ─── ADD RETRY LOGIC FOR NETWORK FILE LOCKS ──────────────────────
            fin = None
            retries = 6
            wait = 10
            while retries > 0:
                try:
                    fin = h5py.File(chunk_path, "r")
                    break # Success! Break the retry loop
                except (BlockingIOError, OSError):
                    retries -= 1
                    print(f"File {chunk_path.name} is locked by network filesystem. Retrying in {wait}s... ({retries} left)")
                    time.sleep(wait)
            if fin is None:
                print(f"WARNING: Skipping permanently dead/locked file: {chunk_path.name}")
                continue
                #raise RuntimeError(f"Could not open {chunk_path} after multiple retries due to network filesystem locks.")
            # ─────────────────────────────────────────────────────────────────

            try:
                if JETS_DATASET not in fin:
                    print(f"Skipping empty file: {chunk_path.name} (No jets passed selection)")
                    continue
                jets   = fin[JETS_DATASET][:]
                tracks = fin[TRACKS_DATASET][:]
                labels = fin[LABELS_DATASET][:]
            finally:
                fin.close() # Always ensure it gets closed explicitly
			
            # VIBECODED BLOCK END 
            
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

    # ───----- VIBECODED WILDCARD EXPANSION BLOCK ──────────────────────────────
    raw_files = cfg.get("files", [])
    expanded_files = []
    for path in raw_files:
        if "*" in path:
            matched_files = glob.glob(path)
            expanded_files.extend(matched_files)
        else:
            expanded_files.append(path)

    if not expanded_files:
        raise FileNotFoundError(
            f"Error: No ROOT files found matching the configuration paths in {args.config}."
        )
    
    cfg["files"] = expanded_files
    # ─────────────────────────────────────────────────────────────────────────


    try:
        from dask_jobqueue import HTCondorCluster
        from dask.distributed import Client
    except ImportError:
        print("dask-jobqueue not installed.  Run: pip install dask dask-jobqueue")
        sys.exit(1)

    cluster = HTCondorCluster(
        cores=1,
        memory="16GB",
        disk="32GB",
        log_directory=str(outdir / "logs"),
        python="/afs/desy.de/user/k/kianianc/.conda/envs/hza_tagger/bin/python",
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

	# ─── VIBECODED  COOLDOWN BLOCK ─────────────────────────────────────────────
    import gc
    import time
    
    print("Cleaning up Dask worker objects and flushing network filesystem buffers...")
    del futures # Clear the memory tracking references to the worker nodes
    gc.collect() # Force Python to run garbage collection immediately
    
    print("Waiting 10 seconds for cluster file handles to cleanly release...")
    time.sleep(10) # Gives DESY AFS and Dask time to completely release file streams
    # ─────────────────────────────────────────────────────────────────────────

    if args.merge:
        merge_files(outdir, outdir.parent / "merged.h5")


if __name__ == "__main__":
    main()
