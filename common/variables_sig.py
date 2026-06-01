"""Single source of truth for branch names and feature variable lists.

Converter and tagger configs both import from here so renaming a branch
only requires a change in one place.
"""

# ── NanoAOD branch names ─────────────────────────────────────────────────────

# AK4 PUPPI jet branches (standard NanoAOD)
Jet = [
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_mass",
#    "Jet_jetId",
    "Jet_puIdDisc",
    "Jet_nConstituents",
]

# Jet ↔ PFCand index branches (btvNanoAOD)
JetPFCands = [
    "JetPFCands_jetIdx",
    "JetPFCands_pFCandsIdx",
]

# PFCand kinematic branches
PFCands = [
    "PFCands_pt",
    "PFCands_eta",
    "PFCands_phi",
    "PFCands_mass",
    "PFCands_charge",
    "PFCands_pdgId",
    "PFCands_genCandIdx",   # index into GenCands (−1 = unmatched); absent in data → handled gracefully
]

# PFCand track / IP branches (may be absent in pheno nanos — handled gracefully)
# Not found in either signal or background files I was given
PFCAND_TRACK_BRANCHES = [
    "PFCands_dxy",
    "PFCands_dz",
    "PFCands_dxySig",
    "PFCands_dzSig",
    "PFCands_trkQuality",
    "PFCands_puppiWeight",
]

# GenJet branches (AK4, particle-level jets for truth_pt / truth_mass)
GenJet = [
    "GenJet_pt",
    "GenJet_eta",
    "GenJet_phi",
    "GenJet_mass",
]

# GenCands branches (truth particles matched 1-to-1 to PFCands; pheno nanos only)
GenCands = [
    "GenCands_pdgId",
    "GenCands_isFromB",
    "GenCands_isFromC",
]

# GenPart branches
GenPart = [
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
    # Truth labels for node-classification auxiliary task
    "truth_pdgId",   # PDG ID of matched GenCand (0 = unmatched)
    "isFromB",       # 1 if the GenCand originates from a b hadron
    "isFromC",       # 1 if the GenCand originates from a c hadron
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

# ── Minimal branch filter for uproot/coffea ──────────────────────────────────
# Pass to NanoEventsFactory via uproot_options={"filter_name": REQUIRED_BRANCHES}
# The n* counter branches are required by NanoAODSchema to build jagged arrays.
REQUIRED_BRANCHES = (
    # event ID scalars (required by NanoAODSchema — cannot be dropped)
    ["run", "luminosityBlock", "event"]
    # counters (required by NanoAODSchema jagged builder)
    + ["nJet", "nJetPFCands", "nPFCands", "nGenPart", "nGenCands", "nGenJet"]
    + JET_BRANCHES
    + JET_PFCAND_IDX_BRANCHES
    + PFCAND_KIN_BRANCHES
    + PFCAND_TRACK_BRANCHES
    + GENPART_BRANCHES
    + GENCANDS_BRANCHES
    + GENJET_BRANCHES
)
