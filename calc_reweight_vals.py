from argparse import ArgumentParser
import h5py

parser = ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='space separated list of file(s) to analyze')
args = parser.parse_args()

FILE = args.file

def calculate_reweight_vals(sig, bkg):
    n_bkg = len(bkg)
    n_sig = len(sig)
    N_events = n_sig + n_bkg

    # calc reweight vals
    w_bg  = N_events/(2 * n_bkg)
    w_sig = N_events/(2 * n_sig)
    print("===============================================================================================")
    print(f"Reweight Values [Background, Signal]: [{w_bg:.2f},{w_sig:.2f}]")
    print("===============================================================================================")

def main():
    with h5py.File(FILE, 'r') as f:
        labels = f["labels"]["a_jet"]
        is_sig = (labels == 1)

        # calc reweight vals
        sig_jets = f["jets"][is_sig]
        bkg_jets = f["jets"][~is_sig]
        calculate_reweight_vals(sig_jets, bkg_jets)

if __name__ == '__main__':
    main()

