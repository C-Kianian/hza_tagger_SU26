"""Coffea processor: btvNanoAOD → structured H5 arrays for HZa tagger.

For each event the processor:
1. Selects AK4 PUPPI jets (pt/eta/jetId cuts).
2. Labels each jet as a-jet (1) or other (0) via dR matching to the
   a boson (PDG 36) and its hadronic daughters in GenPart.
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
)
from common.io import JET_DTYPE, TRACK_DTYPE, LABEL_DTYPE


# ── helpers ───────────────────────────────────────────────────────────────────

def _phi_diff(phi1, phi2):
    """Compute phi1 - phi2 wrapped to [-π, π]."""
    diff = phi1 - phi2
    return ak.where(diff > np.pi, diff - 2 * np.pi,
           ak.where(diff < -np.pi, diff + 2 * np.pi, diff))


def _safe_field(cands, field: str, default: float = 0.0):
    """Return cands[field] if available, else a zero-filled array of same shape."""
    try:
        return cands[field]
    except (ValueError, ak.errors.FieldNotFoundError):
        return ak.zeros_like(cands["pt"]) + default


def _pad_tracks(arr: ak.Array, n: int) -> np.ndarray:
    """Pad/truncate ragged (n_jets, var) → dense (n_jets, n) numpy array."""
    padded = ak.pad_none(arr[:, :n], n, clip=True)
    return ak.to_numpy(ak.fill_none(padded, 0))


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
    sel = (
        (jets.pt > JET_PT_MIN)
        & (abs(jets.eta) < JET_ETA_MAX)
        & (jets.jetId >= JET_ID_MIN)
    )
    jets = jets[sel]

    n_jets_total = ak.sum(ak.num(jets))
    if n_jets_total == 0:
        return _empty_arrays()

    # ── 2. Truth labeling ────────────────────────────────────────────────────
    gp = events.GenPart
    labels_ak = label_jets(
        jet_eta=jets.eta,
        jet_phi=jets.phi,
        gen_eta=gp.eta,
        gen_phi=gp.phi,
        gen_pdgid=gp.pdgId,
        gen_mother_idx=gp.genPartIdxMother,
        gen_status_flags=gp.statusFlags,
    )

    # ── 3. PFCand gathering ──────────────────────────────────────────────────
    # JetPFCands gives (flat) arrays with jetIdx and pFCandsIdx.
    # We need to group pfcand indices by jet, then look up PFCands fields.
    pf = events.PFCands
    jpc = events.JetPFCands  # flat: jetIdx, pFCandsIdx per event

    # Build a per-jet list of PFCand indices
    # sorted by pT descending (proxy for importance)
    pfcand_idx = jpc.pFCandsIdx  # ragged over events (flat within event)
    jet_idx    = jpc.jetIdx

    # Map each JetPFCands entry to its reco jet's new index after selection.
    # We need to account for the jet selection mask: remap original jet indices.
    # Build an index map: original jet index → selected jet position (-1 if removed)
    n_jets_orig = ak.num(events.Jet)  # before selection
    # Selection mask per event (bool, ragged)
    sel_mask = sel

    # Remap: for each event build array of length n_jets_orig mapping orig→selected
    # selected index = cumsum of mask up to that point - 1
    sel_cumsum = ak.Array([np.cumsum(np.asarray(m, dtype=int)) - 1 for m in sel_mask])
    sel_flat = ak.Array([np.asarray(m, dtype=bool) for m in sel_mask])

    # For each JetPFCands entry: keep only if its jet passed selection
    jet_passed = ak.Array([
        np.asarray(sf, dtype=bool)[np.asarray(ji, dtype=int)]
        if len(ji) > 0 else np.array([], dtype=bool)
        for sf, ji in zip(sel_flat, jet_idx)
    ])
    pfcand_idx_sel = pfcand_idx[jet_passed]
    jet_idx_sel    = jet_idx[jet_passed]

    # Remap jet indices to post-selection numbering
    jet_idx_remapped = ak.Array([
        np.asarray(sc, dtype=int)[np.asarray(ji, dtype=int)]
        if len(ji) > 0 else np.array([], dtype=int)
        for sc, ji in zip(sel_cumsum, jet_idx_sel)
    ])

    # Gather PFCand fields per event
    pf_pt     = pf.pt
    pf_eta    = pf.eta
    pf_phi    = pf.phi
    pf_mass   = pf.mass
    pf_charge = pf.charge
    pf_pdgid  = pf.pdgId
    pf_dxy       = _safe_field(pf, "dxy")
    pf_dz        = _safe_field(pf, "dz")
    pf_dxySig    = _safe_field(pf, "dxySig")
    pf_dzSig     = _safe_field(pf, "dzSig")
    pf_trkQual   = _safe_field(pf, "trkQuality")
    pf_puppi     = _safe_field(pf, "puppiWeight", default=1.0)

    # Per-event: build (n_sel_jets, var_n_cands) ragged arrays
    # We accumulate flat arrays then split by sel-jet index
    all_jet_arrays = []  # list of dicts, one per event

    for ievt in range(len(jets)):
        n_sel = ak.num(jets)[ievt]
        if n_sel == 0:
            all_jet_arrays.append(None)
            continue

        ji  = np.asarray(jet_idx_remapped[ievt], dtype=int)
        pci = np.asarray(pfcand_idx_sel[ievt], dtype=int)

        # Sort by pT descending within each jet
        pt_vals = np.asarray(pf_pt[ievt])[pci]
        order = np.argsort(-pt_vals)
        ji  = ji[order]
        pci = pci[order]

        # Build per-jet constituent lists
        jet_cands = [[] for _ in range(n_sel)]
        for j, p in zip(ji, pci):
            jet_cands[j].append(p)

        jet_eta_vals = np.asarray(jets[ievt].eta)
        jet_phi_vals = np.asarray(jets[ievt].phi)

        # Retrieve all needed PFCand arrays for this event
        get = lambda arr: np.asarray(arr[ievt])
        pt_a       = get(pf_pt);    eta_a   = get(pf_eta);    phi_a   = get(pf_phi)
        mass_a     = get(pf_mass);  chg_a   = get(pf_charge); pdg_a   = get(pf_pdgid)
        dxy_a      = get(pf_dxy);   dz_a    = get(pf_dz)
        dxySig_a   = get(pf_dxySig); dzSig_a = get(pf_dzSig)
        trkQ_a     = get(pf_trkQual); puppi_a = get(pf_puppi)

        evt_data = {
            "jet_cands": jet_cands,
            "jet_eta":   jet_eta_vals,
            "jet_phi":   jet_phi_vals,
            "pt":        pt_a,   "eta":  eta_a,   "phi":  phi_a,
            "mass":      mass_a, "charge": chg_a, "pdgId": pdg_a,
            "dxy":       dxy_a,  "dz":   dz_a,
            "dxySig":    dxySig_a, "dzSig": dzSig_a,
            "trkQuality": trkQ_a,  "puppiWeight": puppi_a,
        }
        all_jet_arrays.append(evt_data)

    # ── 4. Flatten to numpy structured arrays ─────────────────────────────────
    jet_pt_flat  = np.concatenate([np.asarray(jets[i].pt)   for i in range(len(jets))])
    jet_eta_flat = np.concatenate([np.asarray(jets[i].eta)  for i in range(len(jets))])
    jet_phi_flat = np.concatenate([np.asarray(jets[i].phi)  for i in range(len(jets))])
    jet_mass_flat= np.concatenate([np.asarray(jets[i].mass) for i in range(len(jets))])
    labels_flat  = np.concatenate([np.asarray(labels_ak[i]) for i in range(len(labels_ak))])

    n_jets = len(jet_pt_flat)

    jets_arr = np.zeros(n_jets, dtype=JET_DTYPE)
    jets_arr["pt"]   = jet_pt_flat
    jets_arr["eta"]  = jet_eta_flat
    jets_arr["phi"]  = jet_phi_flat
    jets_arr["mass"] = jet_mass_flat

    labels_arr = np.zeros(n_jets, dtype=LABEL_DTYPE)
    labels_arr["a_jet"] = labels_flat

    # Build track array: (n_jets, N_TRACKS)
    tracks_arr = np.zeros((n_jets, N_TRACKS), dtype=TRACK_DTYPE)
    tracks_arr["valid"] = False

    jet_global_idx = 0
    for ievt, evt_data in enumerate(all_jet_arrays):
        if evt_data is None:
            continue
        n_sel = len(evt_data["jet_eta"])
        for j in range(n_sel):
            cand_idxs = evt_data["jet_cands"][j]
            if not cand_idxs:
                jet_global_idx += 1
                continue
            c = np.array(cand_idxs[:N_TRACKS])
            nc = len(c)
            j_eta = evt_data["jet_eta"][j]
            j_phi = evt_data["jet_phi"][j]
            trk = tracks_arr[jet_global_idx, :nc]
            trk["pt"]          = evt_data["pt"][c]
            trk["eta_rel"]     = evt_data["eta"][c] - j_eta
            # wrap phi difference
            dphi = evt_data["phi"][c] - j_phi
            dphi = np.where(dphi > np.pi, dphi - 2*np.pi,
                   np.where(dphi < -np.pi, dphi + 2*np.pi, dphi))
            trk["phi_rel"]     = dphi
            trk["mass"]        = evt_data["mass"][c]
            trk["charge"]      = evt_data["charge"][c].astype(np.int8)
            trk["pdgId"]       = evt_data["pdgId"][c].astype(np.int32)
            trk["dxy"]         = evt_data["dxy"][c]
            trk["dz"]          = evt_data["dz"][c]
            trk["dxySig"]      = evt_data["dxySig"][c]
            trk["dzSig"]       = evt_data["dzSig"][c]
            trk["trkQuality"]  = evt_data["trkQuality"][c].astype(np.int8)
            trk["puppiWeight"] = evt_data["puppiWeight"][c]
            trk["valid"]       = True
            tracks_arr[jet_global_idx, :nc] = trk
            jet_global_idx += 1

    return {"jets": jets_arr, "tracks": tracks_arr, "labels": labels_arr}


def _empty_arrays():
    return {
        "jets":   np.zeros(0, dtype=JET_DTYPE),
        "tracks": np.zeros((0, N_TRACKS), dtype=TRACK_DTYPE),
        "labels": np.zeros(0, dtype=LABEL_DTYPE),
    }
