"""
File: calc_reweight_vals.py
This script was made to calculate reweight values and returns them one of two ways
1. for a simple classifier task the weights are returned to train.sh which passes them to the training config
2. for a regressor/multi task model the weights are written to the train.h5 file in its pre allocated weight arrays
"""
from argparse import ArgumentParser
import h5py
import numpy as np

parser = ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='space separated list of file(s) to analyze')
parser.add_argument('--bce', action='store_true', help='set for BCE weight calculation')
parser.add_argument('--towrite', action='store_true', help='set if you want to write the weights to file, useful for regression or multi task models')

args = parser.parse_args()
FILE = args.file
BCE  = args.bce
WRITE = args.towrite

def write_h5_weights(file):
    """calculates 3 different kinds of weights for regression and multi task models; writes results to pre allocated
        weight arrays in the h5"""
    with h5py.File(file, "a") as f:
        # load
        labels_dataset = f["labels"]
        labels = f["labels"]["a_jet"][:]             # 1 for signal, 0 for bkg
        truth_mass = f["jets"]["truth_a_mass"][:]    # truth masses

        # setup
        sig_mask = (labels == 1)
        n_events = len(labels)
        n_sig    = np.count_nonzero(sig_mask)

        # ============
        # weight 1: set 1 for sig 0 for bkg, useful for regression tasks where we just want to learn the signal as is
        signal_only_weight = np.zeros_like(labels, dtype=np.float32)
        signal_only_weight[sig_mask] = 1.0
        labels_dataset["signal_only_weight"][:] = signal_only_weight.astype(np.float32) # write

        # ============
        # weight 2: set 0 for bkg weight by masses, useful for regression tasks where we want each mass point learned equally
        regression_mass_weight = np.zeros_like(labels, dtype=np.float32)
        unique_masses, counts = np.unique(truth_mass[sig_mask], return_counts=True)

        for m, count in zip(unique_masses, counts):
            m_mask = sig_mask & (truth_mass == m)
            # all masses get weighted except 0, which are the background labeled events, get set to 0
            regression_mass_weight[m_mask] = len(truth_mass[sig_mask]) / (len(unique_masses) * count)

        labels_dataset["regression_mass_weight"][:] = regression_mass_weight.astype(np.float32) # write

        # ============
        # weight 3: weight both sig and bkg to be learned equally, useful for classification tasks
        binary_classification_weight = np.zeros_like(labels, dtype=np.float32)
        binary_classification_weight[sig_mask] = n_events / (2 * n_sig)
        binary_classification_weight[~sig_mask] = n_events / (2 * (n_events - n_sig))

        labels_dataset["binary_classification_weight"][:] = binary_classification_weight.astype(np.float32)  # write
        # ============

        print(f"Successfully wrote weights to {FILE}")

def calculate_classifier_vals(n_sig, n_bkg):
    """calculates for classifier ONLY tasks"""
    if BCE: return n_bkg/n_sig, 0
    n_events = n_sig + n_bkg

    # calc reweight vals
    w_bkg  = n_events/(2 * n_bkg)
    w_sig = n_events/(2 * n_sig)
    #print("===============================================================================================")
    #print(f"Reweight Values [Background, Signal]: [{w_bkg:.2f},{w_sig:.2f}]")
    #print("===============================================================================================")
    return w_bkg, w_sig

def main():
    if WRITE: # for our other regression/multi task training cases
        write_h5_weights(FILE)
        return

    with (h5py.File(FILE, 'r') as f):
        labels = f["labels"]["a_jet"]
        n_sig = np.count_nonzero(labels == 1)
        n_bkg = len(labels) - n_sig

        # calc reweight vals
        if BCE: w_sig, _ = calculate_classifier_vals(n_sig, n_bkg)
        else: w_bkg, w_sig = calculate_classifier_vals(n_sig, n_bkg)
        print(f"{w_bkg} {w_sig}") if not BCE else print(f"{w_sig}")

if __name__ == '__main__':
    main()

