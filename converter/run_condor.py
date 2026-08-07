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
The HTCondorCluster settings below are tuned for DESY NAF. naf info: https://docs.desy.de/naf/documentation/must_know/
Adjust memory/disk/cores for your site.
"""
from __future__ import annotations
import os

# Tells HDF5 to ignore strict network filesystem locks for this merge process, suggested by Gemini
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ─── SILENCE COFFEA WARNINGS GLOBALLY AT LAUNCH ───────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*")
# ───────────────────────────────────────────────────────────────────────

import argparse
import glob
import sys
import gc
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True, help="Directory for per-file H5 outputs") # often best to do path/to/data/chunks
    p.add_argument("--merge", action="store_true", help="Merge chunk files after all jobs complete")
    p.add_argument("--max-workers", type=int, default=1000, help="Max number of workers") # one could ask for more, but this is already a lot
    p.add_argument("--files-per-worker", type=int, default=5, help="Number of files processed sequentially by each worker")
    p.add_argument("--name", type=str, default="")
    return p.parse_args()


def convert_files(file_paths: list[str], out_path: str, cfg: dict):
    """Worker function executed on the condor node. Converts multiple root files to a single h5 file."""
    import os
    # Tells HDF5 to ignore strict network filesystem locks for this merge process, suggested by Gemini
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    import warnings
    import uproot
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, PFNanoAODSchema
    from converter.processors.jet_dumper import process_events
    from converter.processors.writer import H5Writer
    from common.variables import REQUIRED_BRANCHES


    warnings.filterwarnings("ignore", message="Missing cross-reference index", category=RuntimeWarning)
    #warnings.filterwarnings("ignore", message="coffea.nanoevents.methods.vector will be removed", category=FutureWarning)
    warnings.filterwarnings("ignore", category=FutureWarning, module="coffea.*") # the above misses the warning, so I added this

    schema_map = {"NanoAODSchema": NanoAODSchema, "PFNanoAODSchema": PFNanoAODSchema}
    schema     = schema_map.get(cfg.get("nano_schema", "NanoAODSchema"), NanoAODSchema)

    tree_name           = cfg.get("tree", "Events")
    chunk_size          = cfg.get("chunk_size", 10_000)
    max_events_per_file = cfg.get("max_events_per_file", None)

    writer = H5Writer(out_path)
    try:
        # check all root files in the batch
        for file_path in file_paths:
            try:
                tree = uproot.open(f"{file_path}:{tree_name}")
            except Exception as e:
                print(f"[WORKER WARNING] Skipping unopenable file {file_path}: {e}")
                continue

            n_entries = tree.num_entries # determine how many events from this file to run over
            if max_events_per_file is not None: n_entries = min(n_entries, max_events_per_file)

            for start in range(0, n_entries, chunk_size): # run converting
                stop  = min(start + chunk_size, n_entries)
                chunk = NanoEventsFactory.from_root(
                    {file_path: tree_name},
                    entry_start=start,
                    entry_stop=stop,
                    schemaclass=schema,
                    uproot_options={"filter_name": REQUIRED_BRANCHES},
                ).events()

                arrays = process_events(chunk)
                if len(arrays["jets"]) > 0:
                    writer.write_chunk(arrays["jets"], arrays["tracks"], arrays["labels"])
                del chunk, arrays
    finally:
        writer.finalize()

        # flush data before job closes, prevents files from appearing to be processing
        try:
            fd = os.open(out_path, os.O_RDONLY)
            os.fsync(fd) # Forces the worker node to wait until the file is written to network disk
            os.close(fd)
        except Exception as e:
            print(f"[WORKER WARNING]: fsync failed for {os.path.basename(out_path)}: {e}")
            pass

def merge_files(outdir: Path, merged_path: Path, name: str = "", max_workers: int = 8, compress: bool = True):
    """Concatenate per-file H5 chunks into a single file, there is room to optimize this."""
    import h5py
    from common.io import JETS_DATASET, TRACKS_DATASET, LABELS_DATASET
    from common.variables import H5_COMPRESSION, H5_COMPRESSION_OPTS, H5_SHUFFLE
    from concurrent.futures import ThreadPoolExecutor

    pattern = f"chunk_{name}*.h5" if name else "chunk*.h5"
    chunks = sorted(outdir.glob(pattern))

    if not chunks:
        print(f"WARNING: No chunk files found matching pattern: {pattern}")
        return

    print(f"Merging {len(chunks)} files ({pattern}) → {merged_path}")

    # get sizes for pre allocating datasets
    def inspect_chunk(chunk_path):
        # wait if files are still being processed, otherwise get sizes
        retries = 6
        while retries > 0:
            try:
                with h5py.File(chunk_path, "r") as fin:
                    if JETS_DATASET not in fin or fin[JETS_DATASET].shape[0] == 0: return None

                    # store shapes & dtypes
                    n_jets = fin[JETS_DATASET].shape[0]
                    jet_dtype = fin[JETS_DATASET].dtype
                    track_subshape = fin[TRACKS_DATASET].shape[1:]
                    track_dtype = fin[TRACKS_DATASET].dtype
                    label_dtype = fin[LABELS_DATASET].dtype

                    return chunk_path, n_jets, jet_dtype, track_subshape, track_dtype, label_dtype
            except (BlockingIOError, OSError):
                retries -= 1
                time.sleep(2)
        print(f"WARNING: Skipping inaccessible chunk: {chunk_path.name}")
        return None

    print("Inspecting chunk info...") # check the chunks in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        meta_results = list(executor.map(inspect_chunk, chunks))

    valid_meta = [m for m in meta_results if m is not None]

    if not valid_meta:
        print("WARNING: No valid chunks found to merge.")
        return

    # calc total out array size
    total_jets = sum(m[1] for m in valid_meta)
    _, _, jet_dtype, track_subshape, track_dtype, label_dtype = valid_meta[0]

    print(f"Found {total_jets} total jets across {len(valid_meta)} valid files.")

    # write info in factors of 2, easy on the bits
    hdf5_chunk_size = min(131072, total_jets)

    create_kwargs = {} if not compress else {"compression": H5_COMPRESSION, "compression_opts": H5_COMPRESSION_OPTS, "shuffle": H5_SHUFFLE} # compress if specified

    with h5py.File(merged_path, "w") as fout: # create datasets
        ds_jets = fout.create_dataset(
            JETS_DATASET,
            shape=(total_jets,),
            dtype=jet_dtype,
            chunks=(hdf5_chunk_size,),
            **create_kwargs,
        )
        ds_tracks = fout.create_dataset(
            TRACKS_DATASET,
            shape=(total_jets, *track_subshape),
            dtype=track_dtype,
            chunks=(hdf5_chunk_size, *track_subshape),
            **create_kwargs,
        )
        ds_labels = fout.create_dataset(LABELS_DATASET,
            shape=(total_jets,),
            dtype=label_dtype,
            chunks=(hdf5_chunk_size,),
            **create_kwargs,
        )
        print("Datasets created, writing jets...")

        # write chunks
        current_idx = 0
        next_report = 0
        for chunk_path, n_jets, _, _, _, _ in valid_meta:
            next_idx = current_idx + n_jets

            with h5py.File(chunk_path, "r") as fin:
                ds_jets[current_idx:next_idx] = fin[JETS_DATASET][:]
                ds_tracks[current_idx:next_idx] = fin[TRACKS_DATASET][:]
                ds_labels[current_idx:next_idx] = fin[LABELS_DATASET][:]

            current_idx = next_idx

            percent = 100 * (current_idx / total_jets) # print progress reports
            while percent >= next_report: # while loop used in case one chunk crosses multiple thresholds
                print(f"Percent of jets written: {next_report:.0f}%")
                next_report += 10

        # flush anything 'hanging' to the output file
        fout.flush()

    print(f"Successfully merged data into {merged_path}")

def main():
    args   = parse_args()
    cfg    = yaml.safe_load(Path(args.config).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # wildcard input expansion
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

    cfg.pop("files", None)

    # ensure condor can run
    try:
        from dask_jobqueue import HTCondorCluster
        from dask.distributed import Client
    except ImportError:
        print("dask-jobqueue not installed.  Run: pip install dask dask-jobqueue")
        sys.exit(1)

    # worker requirements
    cluster = HTCondorCluster( # this setup allows use to take advantage of 'lite' jobs on naf
        cores=1,
        memory="1.5GB", # keep <=1.5 to take advantage of lite jobs
        disk="1.5GB",   # keep  <=1.5 to take advantage of lite jobs
        log_directory=str(outdir / "logs"),
        python="/data/dust/user/kianianc/.conda/envs/hza_tagger/bin/python", #CHANGE TO WHERE YOUR ENV IS
        # DESY NAF flavor — adjust for your site
        job_extra_directives={
            "+RequestRuntime": 10800, # 3600 = 1hr, keep <=3hrs to take advantage of 'lite' jobs
            "universe": "vanilla",
        },
    )

    batch_size = args.files_per_worker # get the files to process in each worker
    file_batches = [expanded_files[i : i + batch_size] for i in range(0, len(expanded_files), batch_size)]
    # to take advantage of 'lite' jobs ensure 'time for one file' * batch_size < 3hrs

    cluster.scale(min(args.max_workers, len(file_batches))) # scale to # files or workers
    client = Client(cluster)
    
    n = args.name # name to add to files
    if n != '': n = '_' + n
    futures = []

    for i, batch in enumerate(file_batches): # submit jobs
        out_path = str(outdir / f"chunk{n}_num={i:05d}.h5")
        fut = client.submit(convert_files, batch, out_path, cfg)
        futures.append(fut)

    print(f"Submitted {len(futures)} jobs — watching …")
    client.gather(futures)
    print("All jobs complete.")
    
    # cooldown block
    print("Cleaning up Dask worker objects and flushing network filesystem buffers...")
    del futures
    gc.collect() # force garbage collection
    print("Waiting 10 seconds for cluster file handles to cleanly release...")
    time.sleep(10) # let files be released

    client.close()
    cluster.close() # tells workers to exit, ideally this frees up the workers while our merge takes place

    # merge chunks 
    if args.merge:
        merge_files(outdir, outdir.parent / f"merged{n}.h5", name=args.name, max_workers=16)
        print("Done merging!")


if __name__ == "__main__":
    main()
