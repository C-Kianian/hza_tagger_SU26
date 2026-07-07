#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path
import h5py
import numpy as np
import sys

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Input H5 file to filter")
    p.add_argument("--out-dir", help="Output directory")
    p.add_argument("--mask-name", default="atlas_valid", help="Name of the mask variable")
    return p.parse_args()

def main():
    args = parse_args()
    infile = Path(args.file)
    mask_name = args.mask_name
    out_dir = args.out_dir

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
        bool_mask = fin['jets'][mask_name][:]
        
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
