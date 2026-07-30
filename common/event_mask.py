"""
Script for filtering events based on a mask, this can be the ATLAS valid mask, which indicates which events pass the
additional ATLAS 2025 criteria. This can be the labels of signal/background, filtering to only include one of them

Originally used for training the ATLAS models and subsequent comparisons with the v1-3 Salt models, here the original
h5 files were filtered to include only ATLAS jets, then only for the regressor were pure signal labeled h5 files used,
ie. filtered the already ATLAS filtered jets by requiring them to be labeled as signal
"""
#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import h5py
import numpy as np
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--infile", required=True, help="Input H5 file to filter")
    p.add_argument("--outdir", help="Output directory")
    p.add_argument("--mask", default="atlas_valid", help="Name of the mask variable")
    p.add_argument("--dataset", default='jets', help="Which dataset in the h5 to search for mask, ie. jets, tracks, labels")
    return p.parse_args()

def main():
    args = parse_args()
    infile = Path(args.infile)
    mask_name = args.mask
    out_dir = args.outdir
    dataset = args.dataset

    if not infile.exists():
        print(infile)
        return

    out_filename = f"{infile.stem}_filtered_w_{mask_name}{infile.suffix}"
    outfile = (infile.parent / out_filename) if out_dir is None else (Path(out_dir) / out_filename)

    # Cache check
    if outfile.exists():
        print(outfile)
        return

    print(f"[Preprocessing] Slicing {infile.name} completely in memory...", file=sys.stderr, flush=True)

    with h5py.File(infile, 'r') as fin:
        if 'jets' not in fin or mask_name not in fin['jets'].dtype.names:
            shutil.copy2(infile, outfile)
            print(outfile)
            return
        
        # 1. Read the boolean mask completely into RAM
        bool_mask = fin[dataset][mask_name][:]

        if bool_mask.dtype != np.bool_: # if mask is 0,1 turn into bool. useful for labels ie sig=1 bkg=0
            unique = np.unique(bool_mask)
            if np.all(np.isin(unique, [0, 1])):
                bool_mask = bool_mask.astype(bool)
            else:
                raise ValueError(
                f"{mask_name} is not a boolean mask or 0/1 labels. "
                f"Found values: {unique}"
                )


        with h5py.File(outfile, 'w') as fout:
            for key in fin.keys():
                # Ensure we are dealing with a dataset, not a group attribute
                if isinstance(fin[key], h5py.Dataset) and fin[key].ndim >= 1 and len(fin[key]) == len(bool_mask):
                    
                    print(f"  -> Processing '{key}'...", file=sys.stderr, flush=True)
                    
                    # 2. THE ULTIMATE SPEEDUP: Pull the ENTIRE unmasked array into RAM
                    # This uses an optimized contiguous block read from your hard drive
                    data_in_ram = fin[key][:]
                    
                    # 3. Use native NumPy to mask inside RAM (Takes milliseconds)
                    filtered_data = data_in_ram[bool_mask]
                    
                    # 4. Stream the finalized block out to disk
                    fout.create_dataset(
                        key, 
                        data=filtered_data, 
                        chunks=True, 
                        compression='gzip', 
                        compression_opts=1  # Level 1 gives maximum write speed
                    )
                else:
                    # Safely copy over meta collections or text groups
                    fin.copy(key, fout)

    print(outfile)

if __name__ == "__main__":
    main()
