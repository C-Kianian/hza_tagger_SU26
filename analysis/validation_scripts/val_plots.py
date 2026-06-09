import h5py
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import hist
import gc
import math
from argparse import ArgumentParser

hep.style.use("CMS")

parser = ArgumentParser()
parser.add_argument('--file', type=str, required=True, help='path to the file for analysis')
parser.add_argument('--maxEvents', type=int, default=None, help='max amount of events to analyze')
parser.add_argument('--plot', type=bool, default=False, help='set true for all the plots to be made')
args = parser.parse_args()

# get file path and setup histogram tracker
FILE = args.file
MAX_EVENTS = args.maxEvents
PLOT = args.plot
hists = []

def print_val(file):
    # display the content of the file
    def print_keys(name, obj):
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

def plot_hist(sig, bkg, x, name, binSize = 1.0, y="Entries", norm=False, logy=False, logx=False):
    # quick check
    if len(sig) == 0 or len(bkg) == 0:
        raise ValueError(f"Sig or bkg is empty for {name}")

    # upper and lower limits
    ll = min(0, np.min(sig), np.min(bkg))
    ul = max(np.max(sig), np.max(bkg))
    # zoom out a little
    if ll < 0: ll = ll - (ll * 0.05)
    ul = ul + (ul * 0.05)
    if ul == ll == 0: 
        ul += 0.1
        ll -= 0.1

    bins = max(1, int(np.ceil((ul - ll) / binSize))) # calc the number of bins

    # hists for sig and bkg
    h_sig = hist.Hist(hist.axis.Regular(bins, ll, ul, label=x))
    h_bkg = hist.Hist(hist.axis.Regular(bins, ll, ul, label=x))
    h_sig.fill(sig)
    h_bkg.fill(bkg)

    # norm if specified
    if norm:
        y = "Normalised entries"
        h_sig = h_sig / h_sig.sum()
        h_bkg = h_bkg / h_bkg.sum()

    # plot the overlaid hist
    fig, ax = plt.subplots()
    if logy: ax.set_yscale("log") # log if specified
    if logx: ax.set_xscale("log")
    hep.histplot(h_sig, ax=ax, label=f"Signal (a-jet, N={len(sig)})", color="tab:red")
    hep.histplot(h_bkg, ax=ax, label=f"Background (N={len(bkg)})", color="tab:blue", linestyle="--")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    hep.cms.label("Preliminary", data=False, ax=ax, com=13.6)
    plt.tight_layout()

    # save as pdf output
    fig.savefig(f"{name}.pdf", bbox_inches="tight")

    plt.close(fig)


def main():
    with h5py.File(FILE, "r") as f:
        # print the file content
        print_val(f)

        if PLOT:
            # get jet track info for analysis
            idx = slice(None) if MAX_EVENTS is None else slice(0, MAX_EVENTS)

            jets = f["jets"][idx]
            tracks = f["tracks"][idx]
            labels = f["labels"]["a_jet"][idx] # 1 = signal (a-jet), 0 = background

            # x axis latex and names for each var, jets and tracks
            jet_labels = {
                "pt": r"$p_{T}^{\mathrm{jet}}$",
                "eta":  r"$\eta$",
                "phi":  r"$\phi$",
                "mass": "$m$",
            }

            trk_labels = {
                "pt": r"$p_T^{\mathrm{track}}$",
                "eta_rel": r"$\eta_{\mathrm{rel}}$",
                "phi_rel": r"$\phi_{\mathrm{rel}}$",
                "mass": r"$m$",
                "charge": "Charge",
                'pdgId': "PDG Id",
                'dxy': r"$\Delta xy$",
                'dz': r"$\Delta z$",
                'dxySig': r"$\Delta xy \mathrm{Z}$",
                'dzSig': r"$\Delta z \mathrm{Z}$",
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
                plot_hist(sig_jet_var, bkg_jet_var, x=jet_xtitle, name=f"jet_{jet_var}", norm=True)

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
                plot_hist(sig_trk_vals, bkg_trk_vals, x=trk_xtitle, name=f"trk_{trk_var}", norm=True)

            ############ MULTIPLICITY INFO ###########

            # track multiplicity
            trk_valid = trk_valid = tracks['valid']

            multi = trk_valid.sum(axis=1)
            sig_trk_multi = multi[labels == 1]
            bkg_trk_multi = multi[labels == 0]

            plot_hist(sig_trk_multi, bkg_trk_multi, x="Track multiplicity", name="trk_multi", norm=True)


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
            plot_hist(lead_pt[is_sig], lead_pt[~is_sig], x=r"$p_T^{\mathrm{leading\ track}}$", name="trk_lead_pT", norm=True)
            plot_hist(sublead_pt[is_sig], sublead_pt[~is_sig], x=r"$p_T^{\mathrm{sub-leading\ track}}$", name="trk_sub_lead_pT", norm=True)
            plot_hist(sum_pt[is_sig], sum_pt[~is_sig], x=r"$\sum p_T^{\mathrm{track}}$", name="trk_sum_pT", norm=True)

            jet_pt = jets["pt"]

            rel_lead_pt = lead_pt / jet_pt
            rel_sublead_pt = sublead_pt / jet_pt
            rel_sum_pt = sum_pt / jet_pt

            # plots lead and sub lead wrt jet pt
            plot_hist(rel_lead_pt[is_sig], rel_lead_pt[~is_sig], x=r"$p_T^{\mathrm{leading\ track}} / p_T^{\mathrm{jet}}$", name="trk_rel_lead_pT", norm=True)
            plot_hist(rel_sublead_pt[is_sig], rel_sublead_pt[~is_sig], x=r"$p_T^{\mathrm{sub-leading\ track}} / p_T^{\mathrm{jet}}$", name="trk_rel_sub_lead_pT", norm=True)
            plot_hist(rel_sum_pt[is_sig], rel_sum_pt[~is_sig], x=r"$\sum p_T^{\mathrm{track}} / p_T^{\mathrm{jet}}$", name="trk_rel_sum_pT", norm=True)

            ############ DELTA R INFO ###########
            dR = np.sqrt(
                tracks["eta_rel"]**2 +
                tracks["phi_rel"]**2
            )

            dR = np.where(trk_valid, dR, np.nan)

            mean_dR = np.nanmean(dR, axis=1)
            max_dR  = np.nanmax(dR, axis=1)

            sig_mean_dR = mean_dR[labels == 1]
            bkg_mean_dR = mean_dR[labels == 0]

            sig_max_dR = max_dR[labels == 1]
            bkg_max_dR = max_dR[labels == 0]

            plot_hist(sig_mean_dR, bkg_mean_dR, x=r"$\Delta R$", name="trk_dR", norm=True)
            plot_hist(sig_max_dR, bkg_max_dR, x=r"$\Delta R$", name="trk_dR_max", norm=True)



if __name__ == "__main__":
    main()


