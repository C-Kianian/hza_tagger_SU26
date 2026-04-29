"""Label definitions for the H→Z(ll)+a(had) tagger.

Binary scheme:
  class 0 — other jet  (background)
  class 1 — a-jet      (signal: AK4 jet from hadronic a decay, PDG 36)
"""

from __future__ import annotations

CLASS_NAMES = {0: "other", 1: "a_jet"}
CLASS_SIGNAL = 1
CLASS_BACKGROUND = 0

N_CLASSES = len(CLASS_NAMES)
