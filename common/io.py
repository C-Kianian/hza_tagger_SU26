"""H5 schema helpers shared by converter and analysis."""

from __future__ import annotations

import numpy as np

# ── Expected H5 dataset names ─────────────────────────────────────────────────

JETS_DATASET = "jets"
TRACKS_DATASET = "tracks"
LABELS_DATASET = "labels"

# dtype of the jets structured array stored in H5
# Note: a_jet label is included here so SALT's SaltDataset can find it
# when processing labels for the "jets" input.
JET_DTYPE = np.dtype([
    # Reco kinematics
    ("pt",                                  np.float32),
    ("eta",                                 np.float32),
    ("phi",                                 np.float32),
    ("mass",                                np.float32),

    # Truth labels for jet-classification task
    ("a_jet",                               np.int32),    # truth label: 1=a-jet, 0=background
    # Truth kinematics — filled from GenJet dR-matching (falls back to reco when unavailable)
    ("truth_pt",                            np.float32),  # pT of the matched GenJet (or reco jet if unmatched)
    ("truth_mass",                          np.float32),  # mass of the matched GenJet (or reco jet if unmatched)
    ("truth_a_mass",                        np.float32),  # pole mass of the a boson from GenPart (0 for background)

    # Info about the jets failing daughter criteria but passing a-boson dr criteria
    ("failed_dau_dr",                       np.float32),
    ("n_dau_failed",                        np.int32),
    ("n_dau",                               np.int32),

    # ATLAS specific features (available for all jets, set = 0 for those that fail ATLAS criteria)
    ("atlas_valid",                         np.bool_),    # flag for events passing ATLAS criteria
    ("trk_multi",                           np.int32),    # number of tracks in jet
    ("lead_trk_rel_system_pt",              np.float32),  # lead trk pt / sum trk pt
    ("lead_trk_dr",                         np.float32),  # dR of leading trk to jet
    ("angularity_n2",                       np.float32),  # angularity with n = 2, see arXiv:0807.0234
    ("U1_0p7",                              np.float32),  # Energy correlation function U, i=1, beta=0.7, see arXiv:1609.07483
    ("M2_0p3",                              np.float32),  # Energy correlation function M, i=2, beta=0.3, see arXiv:1609.07483
    ("tau2",                                np.int32),    # N(=2) subjettiness, (subjet_radius=0.2, jet_radius=0.4) see arXiv:1011.2268

    # Other features, coded but unused
    #("sum_trk_pt",                         np.float32), # relative, sum, sub, and leading pTs
    #("lead_trk_pt",                        np.float32),
    #("sublead_trk_pt",                     np.float32),
    #("lead_trk_rel_jet_pt",                np.float32),
    #("sublead_trk_rel_jet_pt",             np.float32),
    #("sublead_trk_rel_system_pt",          np.float32),
    #("trk_multi",                          np.int32),   # number of tracks in jet
    #("mean_trk_jet_dR",                    np.float32), # mean dR of trk to jet
    #("max_trk_jet_dR",                     np.float32), # max dR of trk to jet
])

# dtype of the tracks structured array stored in H5
# shape: (n_jets, N_TRACKS)
TRACK_DTYPE = np.dtype([
    ("pt",          np.float32),
    ("eta",         np.float32),
    ("phi",         np.float32),
    ("eta_rel",     np.float32),
    ("phi_rel",     np.float32),
    ("mass",        np.float32),
    ("charge",      np.int8),
    ("pdgId",       np.int32),
    ("dxy",         np.float32),
    ("dz",          np.float32),
    ("dxySig",      np.float32),
    ("dzSig",       np.float32),
    ("trkQuality",  np.int8),
    ("puppiWeight", np.float32),
    ("jet_trk_dr",  np.float32),  #jet to trk dR, added for ATLAS criteria
    #("trk_trk_dR",  np.float32), #trk to trk dR
    ("valid",       np.bool_),   # False for padding slots
    # Truth labels for node-classification auxiliary task (0 when unavailable)
    ("truth_pdgId", np.int32),
    ("isFromB",     np.int8),
    ("isFromC",     np.int8),
])

LABEL_DTYPE = np.dtype([
    ("a_jet", np.int32),
])

