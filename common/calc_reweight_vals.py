#!/usr/bin/env python3
"""
File: calc_reweight_vals.py
Optimized for 300M+ events: vectorizes mass reweighting and minimizes RAM overhead.
files arg, add if you want the weights calculated from the default data to be applied to other files ie.
python calc_reweight_vals.py \
    --default data/train.h5 \
    --files data/val.h5 data/test.h5 \
    --towrite \
"""
from argparse import ArgumentParser
import h5py
import numpy as np

parser = ArgumentParser()
parser.add_argument('--default', type=str, required=True, help='Path to HDF5 file to calculate weights')
parser.add_argument('--files', nargs="+", type=str, help='Path to other HDF5 files to apply train weights to')
parser.add_argument('--bce', action='store_true', help='Set for BCE weight calculation')
parser.add_argument('--towrite', action='store_true', help='Set to write weights into HDF5 dataset')

args = parser.parse_args()
DEFAULT = args.default
FILES = args.files or []
BCE  = args.bce
WRITE = args.towrite


def write_h5_weights(tocalc, files=None):
    """Calculates and writes 3 weight arrays efficiently for 300M+ events."""
    if files is None:
        files = []
    with h5py.File(tocalc, "a") as f:
        jets_dataset = f["jets"]

        # Load required arrays once
        labels = f["labels"]["a_jet"][:]             # 300M elements
        truth_mass = jets_dataset["truth_a_mass"][:] # 300M elements

        sig_mask = (labels == 1)
        n_events = len(labels)
        n_sig = np.count_nonzero(sig_mask)
        n_bkg = n_events - n_sig
        if n_sig == 0 or n_bkg == 0: raise ValueError(f"Reference file needs both classes: n_sig={n_sig}, n_bkg={n_bkg}")

        # =========================================================================
        # Weight 1: Signal-only weight (1.0 for sig, 0.0 for bkg)
        # =========================================================================
        print("Calculating Weight 1 (signal_only_weight)...")
        signal_only_weight = sig_mask.astype(np.float32)
        jets_dataset["signal_only_weight"] = signal_only_weight
        del signal_only_weight  # Free memory immediately
        # =========================================================================
        # Weight 2: Vectorized Regression Mass Weight
        # =========================================================================
        print("Calculating Weight 2 (regression_mass_weight)...")
        sig_masses = truth_mass[sig_mask]
        unique_masses, counts = np.unique(sig_masses, return_counts=True)

        # Calculate per-mass weight lookup table
        weights_per_mass = n_sig / (len(unique_masses) * counts)

        # Map truth_mass directly to weight using searchsorted (O(N log U) instead of O(N * U))
        idx = np.searchsorted(unique_masses, truth_mass)
        # Ensure indices stay in bounds for background events (which may have mass=0 not in unique_masses)
        np.clip(idx, 0, len(unique_masses) - 1, out=idx)

        # Match mask to ensure truth_mass actually equals the unique mass at index
        valid_mass_match = (truth_mass == unique_masses[idx])

        # Combine signal mask and valid mass match
        regression_mass_weight = np.where(sig_mask & valid_mass_match, weights_per_mass[idx], 0.0).astype(np.float32)

        jets_dataset["regression_mass_weight"] = regression_mass_weight

        del regression_mass_weight, idx, valid_mass_match
        # =========================================================================
        # Weight 3: Binary Classification Weight
        # =========================================================================
        print("Calculating Weight 3 (binary_classification_weight)...")
        w_sig = n_events / (2.0 * n_sig)
        w_bkg = n_events / (2.0 * n_bkg)

        # Use np.where to assign signal and background weights in a single pass
        binary_classification_weight = np.where(sig_mask, w_sig, w_bkg).astype(np.float32)
        jets_dataset["binary_classification_weight"] = binary_classification_weight

        # =========================================================================
        # Apply weights from default file to the other files
        # =========================================================================
        for file in files: # write the stored weights to the files
            if file == DEFAULT: continue # skip the default file
            with h5py.File(file, "a") as f2: # apply per file
                # prelim
                jets_dataset2 = f2["jets"]
                labels2 = f2["labels"]["a_jet"][:]
                sig_mask2 = (labels2 == 1)

                # weight 1
                signal_only_weight2 = sig_mask2.astype(np.float32)
                jets_dataset2["signal_only_weight"] = signal_only_weight2

                # weight 2
                truth_mass2 = jets_dataset2["truth_a_mass"][:]
                idx2 = np.searchsorted(unique_masses, truth_mass2)
                np.clip(idx2, 0, len(unique_masses) - 1, out=idx2)
                valid_mass_match2 = (truth_mass2 == unique_masses[idx2])
                regression_mass_weight2 = np.where(sig_mask2 & valid_mass_match2, weights_per_mass[idx2], 0.0).astype(np.float32)
                jets_dataset2["regression_mass_weight"] = regression_mass_weight2

                # weight 3
                binary_classification_weight2 = np.where(sig_mask2, w_sig, w_bkg).astype(np.float32)
                jets_dataset2["binary_classification_weight"] = binary_classification_weight2

                del labels2, sig_mask2, signal_only_weight2, idx2, valid_mass_match2, regression_mass_weight2, binary_classification_weight2

        del binary_classification_weight # w3

        n_files = 1 + sum(file != DEFAULT for file in files) # get total number of files written to
        print(f"Successfully wrote weights to {n_files} file(s)")


def calculate_classifier_vals(n_sig, n_bkg):
    """Calculates weights for classifier-only tasks."""
    if BCE:
        return n_bkg / n_sig, 0.0
    n_events = n_sig + n_bkg
    w_bkg = n_events / (2.0 * n_bkg)
    w_sig = n_events / (2.0 * n_sig)
    return w_bkg, w_sig


def main():
    if WRITE:
        write_h5_weights(DEFAULT, files=FILES)
        return

    with h5py.File(DEFAULT, 'r') as f:
        # Load labels into memory once
        labels = f["labels"]["a_jet"][:]
        n_sig = np.count_nonzero(labels == 1)
        n_bkg = len(labels) - n_sig

        if BCE:
            w_pos, _ = calculate_classifier_vals(n_sig, n_bkg)
            print(f"{w_pos}")
        else:
            w_bkg, w_sig = calculate_classifier_vals(n_sig, n_bkg)
            print(f"{w_bkg} {w_sig}")


if __name__ == '__main__':
    main()
