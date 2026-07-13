"""Coffea processor: btvNanoAOD → structured H5 arrays for HZa tagger.

For each chunk of events the processor:
1. Selects AK4 PUPPI jets (pt/eta/jetId cuts).
2. Labels each jet as a-jet (1) or other (0) via dR matching to the
   a boson (PDG 36) and its hadronic daughters in GenPart. Also dR-matches
   each jet to the nearest GenJet to obtain truth_pt and truth_mass.
3. Gathers PFCands associated to each jet (via JetPFCands index arrays),
   computes relative kinematics w.r.t. the jet axis, and zero-pads to
   N_TRACKS constituents.
4. Returns a dict of numpy structured arrays ready for H5Writer.

"""

from __future__ import annotations

import numpy as np
import awkward as ak

from common.truth_matching import label_jets
from common.variables import (
    JET_PT_MIN,
    JET_ETA_MAX,
    JET_ID_MIN,
    N_TRACKS,
    DR_MATCH,
)
from common.io import JET_DTYPE, TRACK_DTYPE, LABEL_DTYPE


# ── helpers ───────────────────────────────────────────────────────────────────

def _flat(arr) -> np.ndarray:
    """Flatten a ragged awkward array to a 1-D numpy array."""
    return ak.to_numpy(ak.flatten(arr))


def _flat_pf_field(pf, field: str, default: float = 0.0) -> np.ndarray:
    """Return flat PFCand field array; fill with *default* if absent."""
    try:
        return _flat(pf[field])
    except (ValueError, ak.errors.FieldNotFoundError):
        return np.full(int(ak.sum(ak.num(pf.pt))), default, dtype=np.float32)

def _sel_remap(flat_sel: np.ndarray) -> np.ndarray:
    """Global selected-jet index for each original jet slot (-1 if not selected).

    Fully vectorised: the global selected-jet index of jet j is simply
    cumsum(flat_sel)[j] - 1 when flat_sel[j] is True.
    """
    cs = np.cumsum(flat_sel.astype(np.int64))
    return np.where(flat_sel, cs - 1, -1)

def compute_ecfs(tracks_pt, tracks_eta, tracks_phi, jet_pt, beta, calc_e3=False):
    """
    Computes 2-point and 3-point energy correlation functions.
    Inputs are expected to be 2D arrays of shape (N_jets, N_tracks),
    except jet_pt which is (N_jets,).
    """
    # if jet pt < 0 set to 1
    safe_jet_pt = np.where(jet_pt > 0, jet_pt, 1.0)

    # momentum fractions z_i, (N_jets, N_tracks)
    z = tracks_pt / safe_jet_pt[:, np.newaxis]

    # dR matrix between track pairs (N_jets, N_tracks, N_tracks)
    # actual calculation uses R which swaps eta for y
    deta = tracks_eta[:, :, np.newaxis] - tracks_eta[:, np.newaxis, :]
    dphi = tracks_phi[:, :, np.newaxis] - tracks_phi[:, np.newaxis, :]
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi,
                    np.where(dphi < -np.pi, dphi + 2*np.pi, dphi))

    dr = np.sqrt(deta**2 + dphi**2)
    dr_beta = dr ** beta

    # calc e_2, 2-point ECF
    # z_i * z_j * dR_ij^beta
    e2_matrix = z[:, :, np.newaxis] * z[:, np.newaxis, :] * dr_beta
    # sum all i, j, divide by 2! to remove double counting
    e2_beta = np.sum(e2_matrix, axis=(1, 2)) / 2.0

    # for jets that had 0 pt at the start set to 0
    e2_beta = np.where(jet_pt > 0, e2_beta, 0.0)

    if not calc_e3: return e2_beta, None

    # calc e_3, 3-point ECF
    # construct z_i * z_j * z_k. (N_jets, N_tracks, N_tracks, N_tracks)
    z_ijk = z[:, :, np.newaxis, np.newaxis] * z[:, np.newaxis, :, np.newaxis] * z[:, np.newaxis, np.newaxis, :]

    # Broadcast the 2D pairwise angles into the 4D triplet space
    dr_ij = dr_beta[:, :, :, np.newaxis]  # dR_ij (add k dimension)
    dr_jk = dr_beta[:, np.newaxis, :, :]  # dR_jk (add i dimension)
    dr_ik = dr_beta[:, :, np.newaxis, :]  # dR_ki (add j dimension)

    min_angle = np.minimum(np.minimum(dr_ij, dr_ik), dr_jk)

    e3_matrix = z_ijk * min_angle

    # sum all i, j, k, divide by 3! (6) to double counting
    e3_beta = np.sum(e3_matrix, axis=(1, 2, 3)) / 6.0

    # for jets that had 0 pt at the start set to 0
    e3_beta = np.where(jet_pt > 0, e3_beta, 0.0)

    return e2_beta, e3_beta

def find_kt_subjets_numpy(eta, phi, pt, n_subjets, subjet_radius):
    """
    Find subjet axes using kT-algorithm for ONE jet.
    Inputs are 1D arrays without padded/invalid tracks.
    """
    # clustering impossible with fewer particles than subjets
    if len(pt) < n_subjets:
        return None

    # start with individual trks, will shrink as subjets merge
    curr_eta = eta.copy()
    curr_phi = phi.copy()
    curr_pt = pt.copy()

    while len(curr_pt) > n_subjets: # repeat until less trks than num subjets requested, ie. no more merging possible
        # particle to beam distance, d_iB = pT^2
        # treats each particle as its own jet beam
        # later particles can merge into this beam or particles can merge into a combined particle
        d_iB = curr_pt ** 2

        # trk to trk dr distances
        deta = curr_eta[:, np.newaxis] - curr_eta[np.newaxis, :]
        dphi = curr_phi[:, np.newaxis] - curr_phi[np.newaxis, :]
        dphi = np.where(dphi > np.pi, dphi - 2 * np.pi,
                        np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi))
        dr = np.sqrt(deta**2 + dphi**2)

        # get min(pT_i^2, pT_j^2) for each track pair
        min_pt_sq = np.minimum(curr_pt[:, np.newaxis], curr_pt[np.newaxis, :]) ** 2
        # calc distances between particle pairs
        d_ij = min_pt_sq * (dr / subjet_radius) ** 2

        # set the diagonal (i == j) to inf so a particle won't cluster with itself
        np.fill_diagonal(d_ij, np.inf)

        # find the pair with min distance, these particles will be merged first
        min_ij_idx = np.argmin(d_ij) # returns inx of min if the array was 1D
        i, j = np.unravel_index(min_ij_idx, d_ij.shape) # converts 1D idx into coordinate of d_ij matrix
        min_d_ij = d_ij[i, j] # gets the min

        min_iB_idx = np.argmin(d_iB) # get trk with min beam distance
        min_d_iB = d_iB[min_iB_idx]

        # merge step
        if min_d_iB < min_d_ij: # case one: beam is closer to one of the particles than they are to each other
            # merge with beam, remove the particle close to the beam
            curr_eta = np.delete(curr_eta, min_iB_idx)
            curr_phi = np.delete(curr_phi, min_iB_idx)
            curr_pt = np.delete(curr_pt, min_iB_idx)
        else: # case 2: particles are closer to each other than the beam
            # merge particle i and j
            pt_i, pt_j = curr_pt[i], curr_pt[j]
            pt_tot = pt_i + pt_j # combined pt

            eta_new = (curr_eta[i] * pt_i + curr_eta[j] * pt_j) / pt_tot # combined eta

            dphi_ij = curr_phi[j] - curr_phi[i]
            if dphi_ij > np.pi: dphi_ij -= 2 * np.pi
            elif dphi_ij < -np.pi: dphi_ij += 2 * np.pi
            phi_new = curr_phi[i] + dphi_ij * pt_j / pt_tot   # combined phi
            phi_new = (phi_new + np.pi) % (2 * np.pi) - np.pi # ensure within [-pi, pi]

            # delete old particle indices, higher then lower, otherwise higher would shift
            first, second = sorted([i, j], reverse=True)
            curr_eta = np.delete(curr_eta, [first, second])
            curr_phi = np.delete(curr_phi, [first, second])
            curr_pt = np.delete(curr_pt, [first, second])

            # add the combined pseudo-jet
            curr_eta = np.append(curr_eta, eta_new)
            curr_phi = np.append(curr_phi, phi_new)
            curr_pt = np.append(curr_pt, pt_tot)

    return np.stack([curr_eta, curr_phi], axis=-1) # return subjet info, each row is subjet

def compute_batch_nsubjettiness(trks_arr, subjet_axes_arr, jet_radius):
    """
    Computes N-subjettiness vectorized across passing jets.

    Inputs:
        trks_arr: Structured array of shape (N_jets, N_TRACKS)
        subjet_axes_arr: Array of shape (N_jets, n_subjets, 2) containing [eta, phi]
        jet_radius: float (e.g., 0.4)
    """
    t_pt = trks_arr["pt"] # (N_jets, N_TRACKS)
    t_eta = trks_arr["eta"]
    t_phi = trks_arr["phi"]

    # makes shapes compatible: tracks (N_jets, N_TRACKS, 1)  -  subjet axes (N_jets, 1, n_subjets)
    # so for every jet we get deta, dphi values between each track subject pair
    deta = t_eta[:, :, np.newaxis] - subjet_axes_arr[:, np.newaxis, :, 0] # (N_jets, N_TRACKS, n_subjets)
    dphi = t_phi[:, :, np.newaxis] - subjet_axes_arr[:, np.newaxis, :, 1] # (N_jets, N_TRACKS, n_subjets)

    # boundary wrapping
    dphi = np.where(dphi > np.pi, dphi - 2*np.pi,
                    np.where(dphi < -np.pi, dphi + 2*np.pi, dphi))

    dr = np.sqrt(deta**2 + dphi**2) # calc dr for each track, subjet pair (N_jets, N_TRACKS, n_subjets)

    # for each track, find min distance to a subjet axis (axis=2)
    min_dr = np.min(dr, axis=2) # (N_jets, N_TRACKS)

    # calc subjettiness parts across the tracks (axis=1)
    numerator = np.sum(t_pt * min_dr, axis=1)
    denominator = np.sum(t_pt * jet_radius, axis=1)

    # final ratios
    return np.where(denominator > 0, numerator / denominator, -999999.0)

# ── main processor function ────────────────────────────────────────────────────

def process_events(events) -> dict[str, np.ndarray]:
    """Convert one chunk of NanoAOD events to structured numpy arrays.

    Parameters
    ----------
    events : coffea NanoEvents
        One chunk from NanoAODSchema.

    Returns
    -------
    dict with keys "jets", "tracks", "labels" — numpy structured arrays.
    Returns empty arrays (len 0) if no jets survive selection.
    """
    # ── 1. Jet selection ─────────────────────────────────────────────────────
    jets = events.Jet
    sel = (jets.pt > JET_PT_MIN) & (abs(jets.eta) < JET_ETA_MAX)
    try:
        sel = sel & (jets.jetId >= JET_ID_MIN)
    except (AttributeError, ak.errors.FieldNotFoundError):
        pass

    jets_sel    = jets[sel]
    n_sel_total = int(ak.sum(ak.num(jets_sel)))
    if n_sel_total == 0:
        return _empty_arrays()

    # ── 2. Truth labeling ────────────────────────────────────────────────────
    gp = events.GenPart
    truth = label_jets(
        jet_eta=jets.eta,
        jet_phi=jets.phi,
        gen_eta=gp.eta,
        gen_phi=gp.phi,
        gen_pdgid=gp.pdgId,
        gen_mother_idx=gp.genPartIdxMother,
        gen_status_flags=gp.statusFlags,
        gen_mass=gp.mass,
    )
    labels_sel      = _flat(truth["labels"][sel]).astype(np.int32)
    truth_amass_sel = _flat(truth["truth_a_mass"][sel]).astype(np.float32)
    failed_dau_dr_sel = _flat(truth["failed_dau_dr"][sel]).astype(np.float32)
    n_dau_failed_sel = _flat(truth["n_dau_failed"][sel]).astype(np.int32)
    n_dau_sel = _flat(truth["n_dau"][sel]).astype(np.int32)


    # ── 2b. GenJet dR-matching → truth_pt and truth_mass ────────────────────
    # Match each selected reco jet to the nearest GenJet within DR_MATCH.
    # Fall back to reco pt/mass when no GenJet is found (e.g. background data).
    flat_jet_eta = ak.to_numpy(ak.flatten(jets_sel.eta)).astype(np.float32)
    flat_jet_phi = ak.to_numpy(ak.flatten(jets_sel.phi)).astype(np.float32)
    try:
        gj_eta  = ak.to_numpy(ak.flatten(events.GenJet.eta)).astype(np.float32)
        gj_phi  = ak.to_numpy(ak.flatten(events.GenJet.phi)).astype(np.float32)
        gj_pt   = ak.to_numpy(ak.flatten(events.GenJet.pt)).astype(np.float32)
        gj_mass = ak.to_numpy(ak.flatten(events.GenJet.mass)).astype(np.float32)
        n_gj_per_evt = ak.to_numpy(ak.num(events.GenJet.pt)).astype(np.int64)
        gj_offsets   = np.concatenate([[0], np.cumsum(n_gj_per_evt)])

        n_jets_per_evt = ak.to_numpy(ak.num(jets_sel.pt)).astype(np.int64)
        jet_offsets    = np.concatenate([[0], np.cumsum(n_jets_per_evt)])

        truth_pt_sel   = _flat(jets_sel.pt).astype(np.float32)   # fallback = reco
        truth_mass_sel = _flat(jets_sel.mass).astype(np.float32)

        for ievt in range(len(n_jets_per_evt)):
            js = jet_offsets[ievt];  je = jet_offsets[ievt + 1]
            gs = gj_offsets[ievt];   ge = gj_offsets[ievt + 1]
            if js == je or gs == ge:
                continue
            j_eta = flat_jet_eta[js:je, None]   # (nj, 1)
            j_phi = flat_jet_phi[js:je, None]
            g_eta = gj_eta[gs:ge][None, :]           # (1, ng)
            g_phi = gj_phi[gs:ge][None, :]
            dphi  = j_phi - g_phi
            dphi  = np.where(dphi >  np.pi, dphi - 2 * np.pi,
                             np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi))
            dr    = np.sqrt((j_eta - g_eta)**2 + dphi**2)  # (nj, ng)
            best  = np.argmin(dr, axis=1)                  # (nj,)
            matched = dr[np.arange(len(best)), best] < DR_MATCH
            idxs  = js + np.where(matched)[0]
            truth_pt_sel  [idxs] = gj_pt  [gs:ge][best[matched]]
            truth_mass_sel[idxs] = gj_mass[gs:ge][best[matched]]

    except (AttributeError, ak.errors.FieldNotFoundError):
        # No GenJet collection (data) — fall back to reco
        truth_pt_sel   = _flat(jets_sel.pt).astype(np.float32)
        truth_mass_sel = _flat(jets_sel.mass).astype(np.float32)

    # ── 3. Vectorised PFCand gathering ───────────────────────────────────────
    n_events = len(jets)

    n_jets_orig      = ak.to_numpy(ak.num(jets)).astype(np.int64)
    jet_orig_offsets = np.concatenate([[0], np.cumsum(n_jets_orig)])
    flat_sel         = _flat(sel).astype(bool)
    sel_remap        = _sel_remap(flat_sel)

    jpc          = events.JetPFCands
    n_jpc        = ak.to_numpy(ak.num(jpc.jetIdx)).astype(np.int64)
    flat_jpc_jet = _flat(jpc.jetIdx).astype(np.int64)
    flat_jpc_pfc = _flat(jpc.pFCandsIdx).astype(np.int64)
    evt_of_jpc   = np.repeat(np.arange(n_events, dtype=np.int64), n_jpc)

    global_orig_jet = jet_orig_offsets[evt_of_jpc] + flat_jpc_jet
    keep            = flat_sel[global_orig_jet]

    global_sel_jet = sel_remap[global_orig_jet[keep]]
    flat_jpc_pfc_k = flat_jpc_pfc[keep]
    evt_of_jpc_k   = evt_of_jpc[keep]

    pf         = events.PFCands
    n_pf       = ak.to_numpy(ak.num(pf.pt)).astype(np.int64)
    pf_offsets = np.concatenate([[0], np.cumsum(n_pf)])
    global_pfc = pf_offsets[evt_of_jpc_k] + flat_jpc_pfc_k

    fp_pt     = _flat_pf_field(pf, "pt")
    fp_eta    = _flat_pf_field(pf, "eta")
    fp_phi    = _flat_pf_field(pf, "phi")
    fp_mass   = _flat_pf_field(pf, "mass")
    fp_charge = _flat_pf_field(pf, "charge")
    fp_pdgid  = _flat_pf_field(pf, "pdgId")
    fp_dxy    = _flat_pf_field(pf, "dxy")
    fp_dz     = _flat_pf_field(pf, "dz")
    fp_dxySig = _flat_pf_field(pf, "dxySig")
    fp_dzSig  = _flat_pf_field(pf, "dzSig")
    fp_trkQ   = _flat_pf_field(pf, "trkQuality")
    fp_puppi  = _flat_pf_field(pf, "puppiWeight", default=1.0)

    # Sort by (global_sel_jet, -pt) → pT-descending within each jet
    order            = np.lexsort((-fp_pt[global_pfc], global_sel_jet))
    global_sel_jet_s = global_sel_jet[order]
    global_pfc_s     = global_pfc[order]

    # Within-jet position index
    boundary    = np.empty(len(global_sel_jet_s), dtype=bool)
    boundary[0] = True
    boundary[1:] = global_sel_jet_s[1:] != global_sel_jet_s[:-1]
    group_start  = np.where(boundary)[0]
    group_id     = np.searchsorted(group_start,
                                   np.arange(len(global_sel_jet_s)), side="right") - 1
    within_pos   = np.arange(len(global_sel_jet_s)) - group_start[group_id]

    mask  = within_pos < N_TRACKS
    jet_f = global_sel_jet_s[mask]
    pos_f = within_pos[mask].astype(np.int64)
    pfc_f = global_pfc_s[mask]

    # ── 4. Assemble output arrays ─────────────────────────────────────────────
    jets_arr                        = np.zeros(n_sel_total, dtype=JET_DTYPE)
    jets_arr["pt"]                  = _flat(jets_sel.pt)
    jets_arr["eta"]                 = flat_jet_eta
    jets_arr["phi"]                 = flat_jet_phi
    jets_arr["mass"]                = _flat(jets_sel.mass)
    jets_arr["regression_a_mass"]   = np.zeros(n_sel_total, dtype=np.float32)
    jets_arr["a_jet"]               = labels_sel
    jets_arr["truth_pt"]            = truth_pt_sel
    jets_arr["truth_mass"]          = truth_mass_sel
    jets_arr["truth_a_mass"]        = truth_amass_sel
    jets_arr["failed_dau_dr"]       = failed_dau_dr_sel # track info about the daughter particles, to test matching criteria
    jets_arr["n_dau_failed"]        = n_dau_failed_sel
    jets_arr["n_dau"]               = n_dau_sel

    labels_arr          = np.zeros(n_sel_total, dtype=LABEL_DTYPE)
    labels_arr["a_jet"] = labels_sel

    # Vectorized scatter into (n_jets, N_TRACKS) track array
    tracks_arr = np.zeros((n_sel_total, N_TRACKS), dtype=TRACK_DTYPE)

    dphi = fp_phi[pfc_f] - flat_jet_phi[jet_f]
    dphi = np.where(dphi >  np.pi, dphi - 2 * np.pi,
                    np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi))

    tracks_arr["pt"]         [jet_f, pos_f] = fp_pt[pfc_f]
    tracks_arr["eta"]        [jet_f, pos_f] = fp_eta[pfc_f]
    tracks_arr["phi"]        [jet_f, pos_f] = fp_phi[pfc_f]
    tracks_arr["eta_rel"]    [jet_f, pos_f] = fp_eta[pfc_f] - flat_jet_eta[jet_f]
    tracks_arr["phi_rel"]    [jet_f, pos_f] = dphi
    tracks_arr["mass"]       [jet_f, pos_f] = fp_mass[pfc_f]
    tracks_arr["charge"]     [jet_f, pos_f] = fp_charge[pfc_f].astype(np.int8)
    tracks_arr["pdgId"]      [jet_f, pos_f] = fp_pdgid[pfc_f].astype(np.int32)
    tracks_arr["dxy"]        [jet_f, pos_f] = fp_dxy[pfc_f]
    tracks_arr["dz"]         [jet_f, pos_f] = fp_dz[pfc_f]
    tracks_arr["dxySig"]     [jet_f, pos_f] = fp_dxySig[pfc_f]
    tracks_arr["dzSig"]      [jet_f, pos_f] = fp_dzSig[pfc_f]
    tracks_arr["trkQuality"] [jet_f, pos_f] = fp_trkQ[pfc_f].astype(np.int8)
    tracks_arr["puppiWeight"][jet_f, pos_f] = fp_puppi[pfc_f]
    tracks_arr["valid"]      [jet_f, pos_f] = True
    # trk to trk dR
    #tracks_arr["trk_trk_dr"]                =
    # trk to jet dR
    tracks_arr["jet_trk_dr"]                = np.sqrt(tracks_arr["eta_rel"] ** 2 + tracks_arr["phi_rel"] ** 2)


    # rel, lead, sub, sum pts
    #jets_arr["sum_trk_pt"]                 = np.sum(tracks_arr["pt"], axis=1)
    #jets_arr["lead_trk_pt"]                = np.max(tracks_arr["pt"], axis=1)
    #jets_arr["sublead_trk_pt"]             = np.partition(tracks_arr["pt"], -2, axis=1)[:, -2]
    #jets_arr["lead_trk_rel_jet_pt"]        = jets_arr["lead_trk_pt"] / jets_arr["pt"]
    #jets_arr["lead_trk_rel_system_pt"]     = jets_arr["lead_trk_pt"] / jets_arr["sum_trk_pt"]
    #jets_arr["sublead_trk_rel_jet_pt"]     = jets_arr["sublead_trk_pt"] / jets_arr["pt"]
    #jets_arr["sublead_trk_rel_system_pt"]  = jets_arr["sublead_trk_pt"] / jets_arr["sum_trk_pt"]
    # trk multi and trk to jet dR
    #jets_arr["mean_trk_jet_dr"]  = np.sum(tracks_arr["trk_jet_dR"], axis=1) / np.maximum(n_valid_trks, 1)
    #jets_arr["max_trk_jet_dr"]   = np.max(tracks_arr["trk_jet_dR"], axis=1)
    #lead_trk_idx                 = np.argmax(tracks_arr["pt"], axis=1)
    #jets_arr["lead_trk_dr"]      = tracks_arr["trk_jet_dR"][lead_trk_pt_idx]

    # ── 5. Assemble ATLAS specific output arrays ──────────────────────────────
    # setup for getting ATLAS vars, apply ATLAS 2025 event selection
    jets_arr["atlas_regression_a_mass"] = np.zeros(n_sel_total, dtype=np.float32)
    atlas_trk_mask = (
            (tracks_arr["valid"] == True) &             # only look at valid tracks
            (tracks_arr["jet_trk_dr"] < DR_MATCH) &     # within jet dR
            (tracks_arr["pt"] > 0.5) &                  # with pt > 500 MeV
            (np.abs(tracks_arr["eta"]) < JET_ETA_MAX)   # and abs(eta) < 2.5
    )                                                   # (N jets, N tracks)

    # ATLAS paper requires jets to have >= 2 trks
    n_atlas_trks = np.sum(atlas_trk_mask, axis=1)    # Column of N jets (N jets, 1)
    atlas_jet_mask = (n_atlas_trks >= 2)             # make >= 2 mask   (N jets, 1)

    jets_arr["atlas_valid"] = atlas_jet_mask         # flag events valid under ATLAS criteria

    # zero out pt of trks that fail ATLAS trk criteria
    filtered_tracks = tracks_arr.copy()                                            # (N jets, N tracks)
    filtered_tracks["pt"] = np.where(atlas_trk_mask, filtered_tracks["pt"], 0.0)   # (N jets, N tracks),

    #=================
    # ATLAS HZa 2025 variables
    #=================
    # trk multiplicity, zero out ATLAS invalid events
    jets_arr["trk_multi"] = np.where(atlas_jet_mask, n_atlas_trks, 0)              # (N jets,)

    #=================
    # lead jet to trk dr
    #=================
    # sum and lead trk info
    sum_trk_pt = np.sum(filtered_tracks["pt"], axis=1) # lead and sum trk pt info,    (N jets,)
    lead_trk_pt = np.max(filtered_tracks["pt"], axis=1)                             # (N jets,)
    lead_trk_idxs = np.argmax(filtered_tracks["pt"], axis=1)                        # (N jets,)

    # get lead dr info
    row_indices = np.arange(n_sel_total)
    lead_trk_dr = filtered_tracks["jet_trk_dr"][row_indices, lead_trk_idxs]

    # final lead trk dr and lead trk pt rel system pt, zero out ATLAS invalid events
    jets_arr["lead_trk_rel_system_pt"] = np.where(atlas_jet_mask & (sum_trk_pt > 0), lead_trk_pt / sum_trk_pt, 0.0)  # (N jets,)
    jets_arr["lead_trk_dr"] = np.where(atlas_jet_mask, lead_trk_dr, 0.0)                                             # (N jets,)

    #=================
    # angularity
    #=================
    valid_trk_mask = filtered_tracks["pt"] > 0 # filter for invalid tracks, recall above these were set to 0

    angles = np.zeros_like(filtered_tracks["jet_trk_dr"]) # column of 0s (N jets, 1)
    np.divide(
        np.pi * filtered_tracks["jet_trk_dr"],  # numerator
        (2.0 * DR_MATCH),                      # denominator
        out=angles,                            # save to angles var
        where=valid_trk_mask                   # only perform on valid tracks
    )

    sin_terms = np.where(valid_trk_mask, np.sin(angles), 1.0) # only perform on valid tracks, else set to 1.0
    sin2s = sin_terms ** -2
    cos_terms = (1.0 - np.cos(angles)) ** 3
    trk_angularity_contributions = filtered_tracks["pt"] * sin2s * cos_terms # multiply sum terms

    angularities = np.sum(trk_angularity_contributions, axis=1) # sum trk angularities, (N jets,)
    jets_arr["angularity_n2"] = np.where(
        atlas_jet_mask & (jets_arr["mass"] > 0), # divide by jet mass if > 0 and ATLAS valid, else set to 0
        angularities / jets_arr["mass"],
        0
    )

    #=================
    # energy correlation functions ( U_1(0.7), M_2(0.3) )
    #=================
    u1_0p7, _              = compute_ecfs(filtered_tracks["pt"],
                                          filtered_tracks["eta"],
                                          filtered_tracks["phi"],
                                          jets_arr["pt"],  beta=0.7, calc_e3=False)
    ecf2_1_0p3, ecf3_1_0p3 = compute_ecfs(filtered_tracks["pt"],
                                          filtered_tracks["eta"],
                                          filtered_tracks["phi"],
                                          jets_arr["pt"], beta=0.3, calc_e3=True)

    jets_arr["U1_0p7"] = np.where(atlas_jet_mask, u1_0p7, 0.0)
    m2 = np.where(ecf2_1_0p3 > 0, ecf3_1_0p3 / ecf2_1_0p3, 0.0) # avoid divide by zero
    jets_arr["M2_0p3"] = np.where(atlas_jet_mask, m2, 0.0)      # zero if not ATLAS valid

    #=================
    # N-subjettiness tau2
    #=================
    n_subjets = 2
    subjet_radius = 0.2
    jet_radius = 0.4

    # get subjet axes row by row
    axes_list = []
    invalid_subjet_mask = ~atlas_jet_mask.copy()
    # set all the jets that pass to false, and the fails to true, used in tau_2_results to set failed jets to -999999.0

    for i in range(n_sel_total):
        if not atlas_jet_mask[i]:
            # skip jets that failed ATLAS checks
            axes_list.append(np.zeros((n_subjets, 2)))
            continue

        # get valid trks for this jet (row)
        mask = filtered_tracks["pt"][i] > 0
        eta = filtered_tracks["eta"][i][mask]
        phi = filtered_tracks["phi"][i][mask]
        pt  = filtered_tracks["pt"][i][mask]

        # find subjet axis from this jet's trks
        axes = find_kt_subjets_numpy(eta, phi, pt, n_subjets, subjet_radius)

        if axes is None:
            # jet didn't have enough tracks to satisfy n_subjets, happens when num trks < 2
            axes_list.append(np.zeros((n_subjets, 2)))
            invalid_subjet_mask[i] = True
        else:
            axes_list.append(axes)

    # make axis list into matrix of shape (N_atlas_jets, n_subjets, 2)
    subjet_axes_arr = np.array(axes_list)

    # vectorized batch calculator for N = 2 subjettiness
    tau_2_results = compute_batch_nsubjettiness(filtered_tracks, subjet_axes_arr, jet_radius)

    # replace events where clustering was impossible
    tau_2_results = np.where(invalid_subjet_mask, -999999.0, tau_2_results)

    # add to output array
    jets_arr["tau2"] = tau_2_results

    # ── 5. PFCand → GenCands truth labels (absent in data → silently skipped) ──
    try:
        gc = events.GenCands
        n_gc       = ak.to_numpy(ak.num(gc.pdgId)).astype(np.int64)
        gc_offsets = np.concatenate([[0], np.cumsum(n_gc)])
        gc_pdgid   = _flat(gc.pdgId).astype(np.int32)
        gc_isFromB = _flat(gc.isFromB).astype(np.int8)
        gc_isFromC = _flat(gc.isFromC).astype(np.int8)

        fp_gcIdx = _flat_pf_field(pf, "genCandIdx", default=-1).astype(np.int64)

        # Determine which event each selected global PFCand belongs to
        evt_of_pfc_f = np.searchsorted(pf_offsets[1:], pfc_f, side="right")
        local_gc     = fp_gcIdx[pfc_f]
        has_match    = local_gc >= 0
        global_gc    = np.where(has_match, gc_offsets[evt_of_pfc_f] + local_gc, 0)

        tracks_arr["truth_pdgId"][jet_f, pos_f] = np.where(has_match, gc_pdgid[global_gc],   0)
        tracks_arr["isFromB"]    [jet_f, pos_f] = np.where(has_match, gc_isFromB[global_gc], 0)
        tracks_arr["isFromC"]    [jet_f, pos_f] = np.where(has_match, gc_isFromC[global_gc], 0)
    except (AttributeError, ak.errors.FieldNotFoundError):
        pass  # background (data) files have no GenCands — leave fields at 0

    return {"jets": jets_arr, "tracks": tracks_arr, "labels": labels_arr}

def _empty_arrays():
    return {
        "jets":   np.zeros(0, dtype=JET_DTYPE),
        "tracks": np.zeros((0, N_TRACKS), dtype=TRACK_DTYPE),
        "labels": np.zeros(0, dtype=LABEL_DTYPE),
    }

