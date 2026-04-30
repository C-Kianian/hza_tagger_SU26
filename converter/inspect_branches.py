#!/usr/bin/env python3
"""Print all branch names in a NanoAOD ROOT file.

Run this locally against your actual file to verify branch names before
editing converter/configs/ and common/variables.py.

Usage
-----
    python converter/inspect_branches.py /path/to/hzanano_output_1.root
    python converter/inspect_branches.py root://cms-xrd-global.cern.ch//store/...
"""

from __future__ import annotations

import sys
import uproot


def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_branches.py <file.root> [tree_name]")
        sys.exit(1)

    path      = sys.argv[1]
    tree_name = sys.argv[2] if len(sys.argv) > 2 else "Events"

    f    = uproot.open(path)
    tree = f[tree_name]

    print(f"\nFile : {path}")
    print(f"Tree : {tree_name}  ({tree.num_entries} entries)\n")

    groups = {}
    for name in sorted(tree.keys()):
        prefix = name.split("_")[0] if "_" in name else name
        groups.setdefault(prefix, []).append(name)

    for prefix, names in sorted(groups.items()):
        print(f"  [{prefix}]")
        for n in names:
            print(f"    {n}")
        print()


if __name__ == "__main__":
    main()
