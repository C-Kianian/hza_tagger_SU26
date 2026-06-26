"""Truth-level jet labeling for H→Z(ll)+a(had).

Strategy
--------
For each reco AK4 jet we check two conditions:

1. **a-proximity**: the jet axis is within DR_MATCH of any a (PDG 36) that is
   the last copy (NanoAOD statusFlags bit 13).
2. **daughter containment**: ALL prompt hadronic daughters of the a lie within
   the same cone. This naturally handles the merged topology (both daughters
   inside one AK4 jet) without any special-casing.

If both hold → label = 1 (a-jet), else → label = 0.

Inputs are awkward arrays with the standard NanoAOD ragged shape.
The function loops over events explicitly to keep the logic transparent.
"""

from __future__ import annotations

import awkward as ak
import numpy as np

from common.variables import A_PDG_ID, DR_MATCH

_LEPTON_IDS = frozenset({11, 12, 13, 14, 15, 16})


def find_a_bosons(gen_pdgid, gen_status_flags):
    """Boolean mask of GenParts that are the a (PDG 36) and last copy.

    NanoAOD statusFlags bit 13 = isLastCopy.
    """
    is_a = abs(gen_pdgid) == A_PDG_ID
    is_last_copy = (gen_status_flags >> 13) & 1 == 1
    return is_a & is_last_copy


def label_jets(
        jet_eta,
        jet_phi,
        gen_eta,
        gen_phi,
        gen_pdgid,
        gen_mother_idx,
        gen_status_flags,
        gen_mass=None,
        dr_match: float = DR_MATCH,
) -> dict:
    """Assign binary a-jet labels and truth a-boson pole mass to AK4 PUPPI jets.

    Returns a dict with awkward arrays of shape (n_events, n_jets):
      ``labels``       int32   — 1 = a-jet, 0 = background
      ``truth_a_mass`` float32 — pole mass of matched a boson from GenPart (0 for background)

    Parameters
    ----------
    jet_eta, jet_phi : ak.Array, shape (n_events, n_jets)
    gen_eta, gen_phi, gen_pdgid, gen_mother_idx, gen_status_flags :
        ak.Array, shape (n_events, n_genparts)
    gen_mass : ak.Array or None
        GenPart masses; required to fill ``truth_a_mass``.
    dr_match : float
        Cone size for matching jet axis to a boson and daughters.
    """
    labels_out      = []
    truth_amass_out = []
    n_dau_out        = []
    n_dau_failed_out = []
    failed_dau_max_dr_out = []

    for ievt in range(len(jet_eta)):
        n_jets     = len(jet_eta[ievt])
        jet_labels = np.zeros(n_jets, dtype=np.int32)
        tamass     = np.zeros(n_jets, dtype=np.float32)

        evt_n_dau_final             = np.zeros(n_jets, dtype=np.int32)
        evt_n_dau_failed_final      = np.zeros(n_jets, dtype=np.int32)
        evt_failed_dau_max_dr_final = np.zeros(n_jets, dtype=np.int32)

        if n_jets == 0:
            labels_out.append(jet_labels)
            truth_amass_out.append(tamass)
            n_dau_out.append(evt_n_dau_final)
            n_dau_failed_out.append(evt_n_dau_failed_final)
            failed_dau_max_dr_out.append(evt_failed_dau_max_dr_final)
            continue

        pdg     = np.asarray(gen_pdgid[ievt])
        flags   = np.asarray(gen_status_flags[ievt])
        g_eta   = np.asarray(gen_eta[ievt])
        g_phi   = np.asarray(gen_phi[ievt])
        mothers = np.asarray(gen_mother_idx[ievt])
        g_mass  = np.asarray(gen_mass[ievt], dtype=np.float32) if gen_mass is not None else None

        # Indices of last-copy a bosons in this event
        a_mask    = (np.abs(pdg) == A_PDG_ID) & ((flags >> 13) & 1 == 1)
        a_indices = np.where(a_mask)[0]

        if len(a_indices) == 0:
            labels_out.append(jet_labels)
            truth_amass_out.append(tamass)
            n_dau_out.append(evt_n_dau_final)
            n_dau_failed_out.append(evt_n_dau_failed_final)
            failed_dau_max_dr_out.append(evt_failed_dau_max_dr_final)
            continue

        j_etas = np.asarray(jet_eta[ievt])
        j_phis = np.asarray(jet_phi[ievt])

        # ── Vectorised dR: (n_jets, n_candidates) matrices ───────────────────
        def _dr_matrix(eta_a, phi_a):
            deta = j_etas[:, None] - eta_a[None, :]
            dphi = j_phis[:, None] - phi_a[None, :]
            dphi = np.where(dphi >  np.pi, dphi - 2 * np.pi,
                            np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi))
            return np.sqrt(deta ** 2 + dphi ** 2)

        dr_to_a = _dr_matrix(g_eta[a_mask], g_phi[a_mask])  # (n_jets, n_a)

        # Per-boson matching: both conditions must hold for the SAME boson.
        matched_by        = []  # one (n_jets,) bool array per boson
        has_had_daughters = []

        # daughter info
        evt_n_dau    = []
        evt_n_failed = []
        evt_max_dr   = []

        for i_a, a_idx in enumerate(a_indices):
            is_child    = mothers == a_idx
            is_hadronic = ~np.isin(np.abs(pdg), list(_LEPTON_IDS))
            dau_mask_i  = is_child & is_hadronic

            dau_etas_i = g_eta[dau_mask_i]
            dau_phis_i = g_phi[dau_mask_i]

            n_dau_i = len(dau_etas_i)

            if n_dau_i == 0:
                has_had_daughters.append(False)
                matched_by.append(np.zeros(len(j_etas), dtype=bool))

                evt_n_dau.append(0)
                evt_n_failed.append(np.zeros(n_jets, dtype=np.float32))
                evt_max_dr.append(np.zeros(n_jets, dtype=np.float32))
                continue

            has_had_daughters.append(True)
            close_this_a = dr_to_a[:, i_a] < dr_match           # (n_jets,)
            dr_dau_i     = _dr_matrix(dau_etas_i, dau_phis_i)   # (n_jets, n_dau_i)
            all_dau_in_i = np.all(dr_dau_i < dr_match, axis=1)  # (n_jets,)
            matched_by.append(close_this_a & all_dau_in_i)

            # daughter info
            evt_n_dau.append(len(dau_etas_i))
            evt_n_failed.append(np.sum(dr_dau_i >= dr_match, axis=1))
            evt_max_dr.append(np.max(dr_dau_i, axis=1))

        if not any(has_had_daughters):
            labels_out.append(jet_labels)
            truth_amass_out.append(tamass)
            n_dau_out.append(evt_n_dau_final)
            n_dau_failed_out.append(evt_n_dau_failed_final)
            failed_dau_max_dr_out.append(evt_failed_dau_max_dr_final)
            continue

        matched_matrix = np.column_stack(matched_by)             # (n_jets, n_a)
        jet_labels     = np.any(matched_matrix, axis=1).astype(np.int32)

        # truth_a_mass: pole mass of the closest boson that triggered label=1
        if g_mass is not None and np.any(jet_labels):
            a_jet_idxs = np.where(jet_labels)[0]
            dr_masked  = np.where(
                matched_matrix[a_jet_idxs],
                dr_to_a[a_jet_idxs],
                np.inf,
            )
            closest_a_col      = np.argmin(dr_masked, axis=1)
            tamass[a_jet_idxs] = g_mass[a_indices[closest_a_col]]

        labels_out.append(jet_labels)
        truth_amass_out.append(tamass)

        # ─── Vectorized Filtering ────────────────────────────────────────────
        evt_n_dau    = np.array(evt_n_dau)
        evt_n_failed = np.column_stack(evt_n_failed)
        evt_max_dr   = np.column_stack(evt_max_dr)

        closest_a = np.argmin(dr_to_a, axis=1)
        jet_idx   = np.arange(n_jets)

        closest_dr       = dr_to_a[jet_idx, closest_a]
        closest_n_dau    = evt_n_dau[closest_a]
        closest_n_failed = evt_n_failed[jet_idx, closest_a]
        closest_max_dr   = evt_max_dr[jet_idx, closest_a]

        fail_mask = (closest_dr < dr_match) & (closest_n_failed > 0)

        # Place values only on your non-colliding output target arrays
        evt_n_dau_final[fail_mask]              = closest_n_dau[fail_mask]
        evt_n_dau_failed_final[fail_mask]       = closest_n_failed[fail_mask]
        evt_failed_dau_max_dr_final[fail_mask]  = closest_max_dr[fail_mask]

        n_dau_out.append(evt_n_dau_final)
        n_dau_failed_out.append(evt_n_dau_failed_final)
        failed_dau_max_dr_out.append(evt_failed_dau_max_dr_final)

    return {
        "labels":       ak.Array(labels_out),
        "truth_a_mass": ak.Array(truth_amass_out),
        "failed_dau_dr": ak.Array(failed_dau_max_dr_out),
        "n_dau_failed": ak.Array(n_dau_failed_out),
        "n_dau":        ak.Array(n_dau_out),
    }

