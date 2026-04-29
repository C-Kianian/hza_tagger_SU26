"""Single source of truth for branch names and feature variable lists.

Converter and tagger configs both import from here so renaming a branch
only requires a change in one place.
"""

# ── NanoAOD branch names ─────────────────────────────────────────────────────

# AK4 PUPPI jet branches (standard NanoAOD)
JET_BRANCHES = [
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
    "Jet_jetId",
    "Jet_puId",
    "Jet_nConstituents",
]

# Jet ↔ PFCand index branches (btvNanoAOD)
JET_PFCAND_IDX_BRANCHES = [
    "JetPFCands_jetIdx",
    "JetPFCands_pFCandsIdx",
]

# PFCand kinematic branches
PFCAND_KIN_BRANCHES = [
    "PFCands_pt",
    "PFCands_eta",
    "PFCands_phi",
    "PFCands_mass",
    "PFCands_charge",
    "PFCands_pdgId",
]

# PFCand track / IP branches (may be absent in pheno nanos — handled gracefully)
PFCAND_TRACK_BRANCHES = [
    "PFCands_dxy",
    "PFCands_dz",
    "PFCands_dxySig",
    "PFCands_dzSig",
    "PFCands_trkQuality",
    "PFCands_puppiWeight",
]

# GenPart branches
GENPART_BRANCHES = [
    "GenPart_pt",
    "GenPart_eta",
    "GenPart_phi",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_genPartIdxMother",
    "GenPart_statusFlags",
]

# ── Feature variable lists (used in SALT configs) ────────────────────────────

JET_FEATURES = ["pt", "eta", "phi", "mass"]

TRACK_FEATURES = [
    "pt",
    "eta_rel",
    "phi_rel",
    "mass",
    "charge",
    "pdgId",
    # IP features — filled with 0 when unavailable
    "dxy",
    "dz",
    "dxySig",
    "dzSig",
    "trkQuality",
    "puppiWeight",
]

# Maximum number of tracks (PFCands) per jet stored in H5
N_TRACKS = 40

# PDG ID of the BSM pseudoscalar a
A_PDG_ID = 36

# Jet selection defaults
JET_PT_MIN = 20.0   # GeV
JET_ETA_MAX = 2.5
JET_ID_MIN = 2      # tight jet ID bit

# Truth-matching cone
DR_MATCH = 0.4
