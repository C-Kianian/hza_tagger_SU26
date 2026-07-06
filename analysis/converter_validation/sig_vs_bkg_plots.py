import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import gc
import hist
from argparse import ArgumentParser

hep.style.use("CMS")

parser = ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='path to the file for analysis')
parser.add_argument('--maxEvents', type=int, default=None, help='max amount of events to analyze')
parser.add_argument('--plot', action='store_true', help='set true for all the plots to be made')
parser.add_argument('--atlas', action='store_true', help='set true for plots with ATLAS variables')
args = parser.parse_args()

# get file path and set up histogram tracker
FILE = args.file
MAX_EVENTS = args.maxEvents
PLOT = args.plot
ATLAS = args.atlas
hists = []

def print_val(file):
    # display the content of the file
    def print_keys(name, _):
        print(name)
    file.visititems(print_keys) # https://docs.h5py.org/en/stable/high/group.html#h5py.Group.visititems

    jets = file["jets"]
    tracks = file["tracks"]
    labels = file["labels"]
    print("Jets shape:", jets.shape)
    print("Tracks shape:", tracks.shape)
    print("Labels shape:", labels.shape, "| dtype:", labels.dtype)
    print("Jets variables:", jets.dtype.names)
    print("Tracks variables:", tracks.dtype.names)
    print("Jets dtype:", jets.dtype)
    print("Tracks dtype:", tracks.dtype)
    gc.collect()

def calculate_bin_edges(plots, n_bins_float=50):
    """
    Vibecoded:
    Generates robust bin edges for both discrete and continuous HEP variables.
    """
    # Combine to find the shared data envelope
    comb = np.concatenate(plots)

    # 1. Safely check if the array contains integers (handles np.int32, np.int64)
    is_discrete = np.issubdtype(comb.dtype, np.integer)

    if is_discrete:
        # For discrete variables (pdgId, charge, trkQuality), we want bins
        # centered exactly on the integers.
        ll = np.min(comb)
        ul = np.max(comb)
        bins = int(ul - ll + 1)
        # Shift edges by 0.5 so integer values fall in the center of the bin
        return ll - 0.5, ul + 0.5, bins + 1

    # 2. For continuous variables (pt, eta, phi, mass, etc.)
    # Use percentiles to ignore massive outliers that squash the histogram
    ll = np.percentile(comb, 0.2)
    ul = np.percentile(comb, 99.8)

    if ul == ll:  # Fallback if percentiles are identical (e.g., very sparse data)
        ll, ul = np.min(comb), np.max(comb)

    # 3. Apply a 5% margin based on the *total range*, not the absolute value
    padding = (ul - ll) * 0.05
    ll -= padding
    ul += padding

    if ul == ll:  # Extreme edge case: all values are identical and 0
        ul += 0.01
        ll -= 0.01

    # 4. Prevent purely positive variables (pt, mass) from dipping below 0 due to padding
    if np.min(comb) >= 0 and ll < 0:
        ll = 0.0

    return ll, ul, n_bins_float + 1

def plot_hist(to_plot, labels, x, name, y="Entries", norm=False, logy=False, logx=False):
    # quick checks, ensure same lengths, less than color length, and non empty
    if len(to_plot) != len(labels): raise ValueError("to_plot and labels must have the same length")
    if len(to_plot) > 10: raise ValueError("Not enough colors for this many plots.")
    for plot, label in zip(to_plot, labels, strict=True):
        if len(plot) == 0: raise ValueError(f"Empty for {label}")

    ll, ul, n_bins = calculate_bin_edges(to_plot, n_bins_float=50) # calc bin edges

    # hists
    hists = []
    for plot in to_plot:
        h = hist.Hist(hist.axis.Regular(n_bins, ll, ul, label=""))
        h.fill(plot)
        hists.append(h)
        del h

    # norm if specified
    if norm:
        norm_hists = []
        for h in hists:
            if h.sum() == 0: norm_hists.append(h) # add even if empty
            else: norm_hists.append(h / h.sum())
            del h
        hists = norm_hists
        y = "Normalised entries"

    # plot the overlaid hist
    fig, ax = plt.subplots()
    if logy: ax.set_yscale("log") # log if specified
    if logx: ax.set_xscale("log")
    colors = list(plt.cm.tab10.colors)
    for l, h, c, plot in zip(labels, hists, colors, to_plot):
        hep.histplot(h, ax=ax, label=f"{l} (N = {len(plot)})", color=c, flow="show")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    hep.cms.label("Preliminary", data=False, ax=ax, com=13.6)
    plt.tight_layout()
    # save as PDF output
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    print(f"Finished plot: {name}")

    plt.close(fig)

def main():
    with h5py.File(FILE, "r") as f:
        # print the file content
        print("===============================================================================================")
        print("File info:")
        print("===============================================================================================")
        print_val(f)

        if PLOT:
            print("===============================================================================================")
            print("Plots (Signal vs Background):")
            print("===============================================================================================")
            # get jet track info for analysis
            idx = slice(None) if MAX_EVENTS is None else slice(0, MAX_EVENTS)

            jets = f["jets"][idx]
            tracks = f["tracks"][idx]
            labels = f["labels"]["a_jet"][idx] # 1 = signal (a-jet), 0 = background

            # x-axis latex and names for each var, jets and tracks
            jet_labels = {
                "pt": r"$p_{T}^{\mathrm{jet}} [GeV/c]$",
                "eta":  r"$\eta^{jet}$",
                "phi":  r"$\phi^{jet}$",
                "mass": "$m^{jet} [GeV/c^{2}]$",
            }
            if ATLAS:
                atlas_labels = {
                    "atlas_valid": "ATLAS valid",
                    "trk_multi": "Track multiplicity",
                    "lead_trk_rel_system_pt": r"$\frac{p_T^{\mathrm{leading\ track}}}{\sum_{\text{tracks}} p_T}$",
                    "lead_trk_dr": r"$\Delta R^{\mathrm{leading\ track}}$",
                    "angularity_n2": "Angularity (n2)",
                    "U1_0p7": r"$U_1^{0.7}$",
                    "M2_0p3": r"$M_2^{0.3}$",
                    "tau2": r"$\tau_2$ (N-Subjettiness)",
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

            ############ JET INFO ############
            # pt, mass, eta, phi for all jets
            true_jet_names = jets.dtype.names
            for jet_var, jet_xtitle in jet_labels.items():
                if jet_var not in true_jet_names:
                    print(f"Missing jet variable: {jet_var}")
                    continue

                # plot
                sig_jet_var = jets[jet_var][labels == 1]
                bkg_jet_var = jets[jet_var][labels == 0]
                plot_hist(to_plot=[sig_jet_var, bkg_jet_var], labels=["Signal", "Background"], x=jet_xtitle, name=f"jet_{jet_var}", norm=True)

            ####### ATLAS JET INFO ############
            if ATLAS:
                atlas_mask = jets["atlas_valid"].ravel()
                atlas_sig_mask = atlas_mask & (labels == 1)
                atlas_bkg_mask = atlas_mask & (labels == 0)

                for atlas_var, atlas_xtitle in atlas_labels.items():
                    if atlas_var not in true_jet_names:
                        print(f"Missing atlas variable: {atlas_var}")
                        continue

                    sig_vals = jets[atlas_var][labels == 1]
                    bkg_vals = jets[atlas_var][labels == 0]
                    atlas_sig_vals = jets[atlas_var][atlas_sig_mask]
                    atlas_bkg_vals = jets[atlas_var][atlas_bkg_mask]
                    plots = [sig_vals, bkg_vals, atlas_sig_vals, atlas_bkg_vals]
                    lbls = ["Signal", "Background", "ATLAS Sig", "ATLAS Bkg"]

                    if atlas_sig_vals.dtype == bool: # just to see the events which are atlas valid
                        plots = [jets[atlas_var][labels == 1].astype(int), jets[atlas_var][labels == 0].astype(int)]
                        lbls = ["Signal", "Background"]

                    if atlas_var == "angularity_n2": # remove high values of angularity
                        plots = [sig_vals[sig_vals < 1000], bkg_vals[bkg_vals < 1000],
                                 atlas_sig_vals[atlas_sig_vals < 1000], atlas_bkg_vals[atlas_bkg_vals < 1000]]

                    plot_hist(to_plot=plots, labels=lbls, x=atlas_xtitle, name=f"atlas_{atlas_var}", norm=True)

            ############# TRACK INFO ############
            # masks
            valid_mask = tracks["valid"].ravel()
            label_flat   = np.repeat(labels, tracks.shape[1])
            sig_mask = valid_mask & (label_flat == 1)
            bkg_mask = valid_mask & (label_flat == 0)

            true_track_names = tracks.dtype.names

            # all tracks vars, pt, eta_rel, phi_rel, mass, charge, pdgId, dxy, dz, dxySig, dzSig, trkQuality, puppiWeight
            for trk_var, trk_xtitle in trk_labels.items():

                if trk_var not in true_track_names:
                    print(f"Missing track variable: {trk_var}")
                    continue

                # plot
                trk_vals = tracks[trk_var].ravel().astype(float)
                sig_trk_vals = trk_vals[sig_mask]
                bkg_trk_vals = trk_vals[bkg_mask]
                plot_hist(to_plot=[sig_trk_vals, bkg_trk_vals], labels=["Signal", "Background"], x=trk_xtitle, name=f"trk_{trk_var}", norm=True)

            ############ MULTIPLICITY INFO ###########

            # track multiplicity
            trk_valid = tracks['valid']

            multi = trk_valid.sum(axis=1)
            sig_trk_multi = multi[labels == 1]
            bkg_trk_multi = multi[labels == 0]

            plot_hist(to_plot=[sig_trk_multi, bkg_trk_multi], labels=["Signal", "Background"], x="Track multiplicity", name="trk_multi", norm=True)

            ############ (RELATIVE) SUB AND LEADING INFO ###########
            # get valid track pt values
            valid_pts = np.where(tracks["valid"], tracks["pt"], 0.0)

            # sum and sort pts
            sum_pt = valid_pts.sum(axis=1)
            sorted_pts = np.sort(valid_pts, axis=1)[:, ::-1]

            # get the leading and subleading track pts
            lead_pt = sorted_pts[:, 0]
            sublead_pt = sorted_pts[:, 1] if sorted_pts.shape[1] > 1 else np.zeros_like(lead_pt) # if only one track, set to 0

            is_sig = labels == 1 # signal mask

            # plot lead, sub, and sum pts
            plot_hist(to_plot=[lead_pt[is_sig], lead_pt[~is_sig]], labels=["Signal", "Background"], x=r"$p_T^{\mathrm{leading\ track}}$", name="trk_lead_pt", norm=True)
            plot_hist(to_plot=[sublead_pt[is_sig], sublead_pt[~is_sig]], labels=["Signal", "Background"], x=r"$p_T^{\mathrm{sub-leading\ track}}$", name="trk_sub_lead_pt", norm=True)
            plot_hist(to_plot=[sum_pt[is_sig], sum_pt[~is_sig]], labels=["Signal", "Background"], x=r"$\sum p_T^{\mathrm{track}}$", name="trk_sum_pt", norm=True)

            jet_pt = jets["pt"]

            rel_lead_pt = lead_pt / jet_pt
            rel_sublead_pt = sublead_pt / jet_pt
            rel_sum_pt = sum_pt / jet_pt

            # plots lead and sub lead wrt jet pt
            plot_hist(to_plot=[rel_lead_pt[is_sig], rel_lead_pt[~is_sig]], labels=["Signal", "Background"], x=r"$p_T^{\mathrm{leading\ track}} / p_T^{\mathrm{jet}}$", name="trk_rel_lead_pt", norm=True)
            plot_hist(to_plot=[rel_sublead_pt[is_sig], rel_sublead_pt[~is_sig]], labels=["Signal", "Background"], x=r"$p_T^{\mathrm{sub-leading\ track}} / p_T^{\mathrm{jet}}$", name="trk_rel_sub_lead_pt", norm=True)
            plot_hist(to_plot=[rel_sum_pt[is_sig], rel_sum_pt[~is_sig]], labels=["Signal", "Background"], x=r"$\sum p_T^{\mathrm{track}} / p_T^{\mathrm{jet}}$", name="trk_rel_sum_pt", norm=True)

            ############ DELTA R INFO ###########
            dR = np.sqrt(tracks["eta_rel"]**2 + tracks["phi_rel"]**2)

            dR = np.where(trk_valid, dR, np.nan)

            mean_dR = np.nanmean(dR, axis=1)
            max_dR  = np.nanmax(dR, axis=1)

            sig_mean_dR = mean_dR[labels == 1]
            bkg_mean_dR = mean_dR[labels == 0]

            sig_max_dR = max_dR[labels == 1]
            bkg_max_dR = max_dR[labels == 0]

            plot_hist(to_plot=[sig_mean_dR, bkg_mean_dR], labels=["Signal", "Background"], x=r"Mean $\Delta R$", name="trk_dR_mean", norm=True)
            plot_hist(to_plot=[sig_max_dR, bkg_max_dR], labels=["Signal", "Background"], x=r"Max $\Delta R$", name="trk_dR_max", norm=True)



if __name__ == "__main__":
    main()


