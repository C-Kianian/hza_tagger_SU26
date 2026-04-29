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


def _dr(eta1: float, phi1: float, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    deta = eta1 - eta2
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi,
           np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi))
    return np.sqrt(deta**2 + dphi**2)


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
    dr_match: float = DR_MATCH,
) -> ak.Array:
    """Assign binary a-jet labels to AK4 PUPPI jets.

    Returns an awkward array of shape (n_events, n_jets) with dtype int32:
      1  → a-jet
      0  → other

    Parameters
    ----------
    jet_eta, jet_phi : ak.Array, shape (n_events, n_jets)
    gen_eta, gen_phi, gen_pdgid, gen_mother_idx, gen_status_flags :
        ak.Array, shape (n_events, n_genparts)
    dr_match : float
        Cone size for matching jet axis to a boson and daughters.
    """
    result = []

    for ievt in range(len(jet_eta)):
        n_jets      = len(jet_eta[ievt])
        jet_labels  = np.zeros(n_jets, dtype=np.int32)

        if n_jets == 0:
            result.append(jet_labels)
            continue

        pdg     = np.asarray(gen_pdgid[ievt])
        flags   = np.asarray(gen_status_flags[ievt])
        g_eta   = np.asarray(gen_eta[ievt])
        g_phi   = np.asarray(gen_phi[ievt])
        mothers = np.asarray(gen_mother_idx[ievt])

        # Indices of last-copy a bosons in this event
        a_mask    = (np.abs(pdg) == A_PDG_ID) & ((flags >> 13) & 1 == 1)
        a_indices = np.where(a_mask)[0]

        if len(a_indices) == 0:
            result.append(jet_labels)
            continue

        # Hadronic daughters of any a boson (direct, non-leptonic children)
        dau_mask = np.zeros(len(pdg), dtype=bool)
        for a_idx in a_indices:
            is_child = mothers == a_idx
            is_hadronic = ~np.isin(np.abs(pdg), list(_LEPTON_IDS))
            dau_mask |= is_child & is_hadronic

        dau_etas = g_eta[dau_mask]
        dau_phis = g_phi[dau_mask]
        n_dau    = len(dau_etas)

        j_etas = np.asarray(jet_eta[ievt])
        j_phis = np.asarray(jet_phi[ievt])

        for j in range(n_jets):
            # Condition 1: jet within DR of any a boson
            dr_to_a = _dr(j_etas[j], j_phis[j], g_eta[a_mask], g_phi[a_mask])
            if not np.any(dr_to_a < dr_match):
                continue

            # Condition 2: all hadronic daughters inside the cone
            # (n_dau == 0 means we found no daughters → skip)
            if n_dau == 0:
                continue

            dr_to_dau = _dr(j_etas[j], j_phis[j], dau_etas, dau_phis)
            if np.all(dr_to_dau < dr_match):
                jet_labels[j] = 1

        result.append(jet_labels)

    return ak.Array(result)
