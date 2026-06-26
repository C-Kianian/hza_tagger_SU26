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
    #tracks_arr["trk_trk_dR"]                =
    # trk to jet dR
    #tracks_arr["trk_jet_dR"]                = np.sqrt(tracks_arr["eta_rel"] ** 2 + tracks_arr["phi_rel"] ** 2)


    # rel, lead, sub, sum pts
    #jets_arr["sum_trk_pt"]                 = np.sum(tracks_arr["pt"], axis=1)
    #jets_arr["lead_trk_pt"]                = np.max(tracks_arr["pt"], axis=1)
    #jets_arr["sublead_trk_pt"]             = np.partition(tracks_arr["pt"], -2, axis=1)[:, -2]
    #jets_arr["lead_trk_rel_jet_pt"]        = jets_arr["lead_trk_pt"] / jets_arr["pt"]
    #jets_arr["sublead_trk_rel_jet_pt"]     = jets_arr["sublead_trk_pt"] / jets_arr["pt"]
    #jets_arr["lead_trk_rel_system_pt"]     = jets_arr["lead_trk_pt"] / jets_arr["sum_trk_pt"]
    #jets_arr["sublead_trk_rel_system_pt"]  = jets_arr["sublead_trk_pt"] / jets_arr["sum_trk_pt"]
    # trk multi and trk to jet dR
    n_valid_trks = np.sum(tracks_arr["valid"], axis=1)
    jets_arr["trk_multi"] = n_valid_trks
    #jets_arr["mean_trk_jet_dR"]  = np.sum(tracks_arr["trk_jet_dR"], axis=1) / np.maximum(n_valid_trks, 1)
    #jets_arr["max_trk_jet_dR"]   = np.max(tracks_arr["trk_jet_dR"], axis=1)
    #lead_trk_pt_idx = np.argmax(tracks_arr["pt"], axis=1)
    #jets_arr["lead_trk_dR"]  = tracks_arr["trk_jet_dR"][lead_trk_pt_idx]

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

