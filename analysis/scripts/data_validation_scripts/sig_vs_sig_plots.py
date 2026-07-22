"""
Plots signal vs signal for the different mass points from an h5 file on different jet/track/edge variables

Produces:
  - A directory (analysis/default_plot_outdir by default) with subdirectories containing plots on the readily available
  jet/track/edge variables and some derived feature.
  - Additionally, --atlas can be specified to plot the features used in the ATLAS 2025 paper
  - Similarly, --edg can be specified to plot the edge features that salt calculates on the fly

Usage
-----
    python analysis/data_validation_scripts/sig_vs_bkg.py \\
        --file   data/merged.h5 \\
        --outdir analysis/plots/ \\
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
from argparse import ArgumentParser
from pathlib import Path
import hist

hep.style.use("CMS")

parser = ArgumentParser()
parser.add_argument('--files', type=str, required=True, nargs='+', help='space separated list of file(s) to analyze')
parser.add_argument('--maxEvents', type=int, default=None, help='max amount of events, per file, to analyze')
parser.add_argument("--outdir", default="analysis/default_sig_vs_sig_outdir")
args = parser.parse_args()

FILES = [f for f in args.files]
MAX_EVENTS = args.maxEvents
OUTDIR = Path(args.outdir) # make output dir
OUTDIR.mkdir(parents=True, exist_ok=True)

file_names = [Path(f).stem for f in FILES]

def calculate_bin_edges(sig,):
    # Filter out empty dictionaries
    active_samples = {k: v for k, v in sig.items() if len(v) > 0}
    if not active_samples:
        return

    # Combine all active samples to calculate robust global boundaries
    global_data = np.concatenate(list(active_samples.values()))

    # Handle Discrete vs Continuous variables to prevent outlier squishing
    is_discrete = np.issubdtype(global_data.dtype, np.integer)

    if is_discrete:
        global_min = np.min(global_data) - 0.5
        global_max = np.max(global_data) + 0.5
        n_bins = int(global_max - global_min)
        return n_bins, global_min, global_max
    else:
        # Strip the extreme 0.5% tails on both ends for a clean zoom
        global_min = np.percentile(global_data, 0.5)
        global_max = np.percentile(global_data, 99.5)

        # Apply 1% margin to the active range
        padding = (global_max - global_min) * 0.01
        global_min -= padding
        global_max += padding

        # Guardrails: Don't let naturally positive variables drop below zero
        if np.min(global_data) >= 0 > global_min:
            global_min = 0.0

        if (global_max - global_min) < 0.01:
            global_max = global_min + 0.01

        return 40, global_min, global_max

def plot_sig_vs_sig(var_dict, var, xtitle, name):
    fig, ax = plt.subplots()

    # Filter out empty dictionaries
    active_samples = {k: v for k, v in var_dict.items() if len(v) > 0}
    if not active_samples:
        return

    nbins, global_min, global_max = calculate_bin_edges(var_dict)
    colors = list(plt.cm.tab10.colors)
    # Loop over each mass signal stored for this specific variable
    for i, (mass_name, sig_data) in enumerate(active_samples.items()):
        if len(sig_data) == 0:
            continue

        # Create, fill, and plot the histogram for this mass point
        h = hist.Hist(hist.axis.Regular(nbins, global_min, global_max, label=xtitle))
        h.fill(sig_data)

        # norm
        h = h / h.sum()

        hep.histplot(h, ax=ax, label=f"Signal {mass_name}", color=colors[i])

    # Decorate the canvas once all masses are drawn
    ax.set_xlabel(xtitle)
    ax.set_ylabel("Normalized Entries")
    ax.legend()
    hep.cms.label("Preliminary", data=False, ax=ax, com=13.6)
    # save pdfs to outdir
    fig.savefig(f"{OUTDIR}/compare_{name}_{var}.pdf", bbox_inches="tight")
    print(f"Finished plot: {OUTDIR}/{name} {var}")
    plt.close(fig)

def main():
    # x-axis latex and names for each var, jets and tracks
    jet_labels = {
        "pt": r"$p_{T}^{\mathrm{jet}} [GeV/c]$",
        "eta":  r"$\eta^{jet}$",
        "phi":  r"$\phi^{jet}$",
        "mass": "$m^{jet} [GeV/c^{2}]$",
    }

    trk_labels = {
        "pt": r"$p_T^{\mathrm{track}} [GeV/c]$",
        "eta_rel": r"$\eta_{\mathrm{rel}}^{\mathrm{track}}$",
        "phi_rel": r"$\phi_{\mathrm{rel}}^{\mathrm{track}}$",
        "mass": r"$m^{track} [GeV/c^{2}]$",
        "charge": "Charge",
        'pdgId': "PDG Id",
        'dxy': r"$\Delta xy$",
        'dz': r"$\Delta z$",
        'dxySig': r"$\delta xy \mathrm{Z}$",
        'dzSig': r"$\delta z \mathrm{Z}$",
        'trkQuality': "Track quality",
        'puppiWeight': "PUPPI weight",
    }

    # Create empty containers for each variable
    jet_signal_storage = {var: {} for var in jet_labels}
    trk_signal_storage = {var: {} for var in trk_labels}

    # pt, mass, eta, phi for all jets
    for file_path, mass_name in zip(FILES, file_names):
        with h5py.File(file_path, 'r') as f:
            idx = slice(None) if MAX_EVENTS is None else slice(0, MAX_EVENTS)

            labels = f["labels"]["a_jet"][idx]
            is_sig = (labels == 1)

            # 1. Extract Jet Signals
            jets = f["jets"][idx]
            for jet_var in jet_labels:
                if jet_var in jets.dtype.names:
                    jet_signal_storage[jet_var][mass_name] = jets[jet_var][is_sig]

            # 2. Extract Track Signals (using the flattened mask setup)
            tracks = f["tracks"][idx]
            valid_flat = tracks["valid"].ravel()
            label_flat = np.repeat(labels, tracks.shape[1])
            sig_mask_flat = valid_flat & (label_flat == 1)

            for trk_var in trk_labels:
                if trk_var in tracks.dtype.names:
                    # Store the flattened, track-validated signal array
                    flat_vals = tracks[trk_var].ravel()
                    trk_signal_storage[trk_var][mass_name] = flat_vals[sig_mask_flat]


    # Loop and plot each variable
    print("===============================================================================================")
    print("Plotting Jet Info:")
    print("===============================================================================================")
    for jet_var, jet_xtitle in jet_labels.items():
        plot_sig_vs_sig(jet_signal_storage[jet_var], jet_var, jet_xtitle, "jet")

    print("===============================================================================================")
    print("Plotting Track Info:")
    print("===============================================================================================")
    # Loop and plot each variable
    for trk_var, trk_xtitle in trk_labels.items():
        plot_sig_vs_sig(trk_signal_storage[trk_var], trk_var, trk_xtitle, "trk")


if __name__ == "__main__":
    main()


