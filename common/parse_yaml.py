#!/usr/bin/env python3

import argparse
from operator import contains
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file")
    group = p.add_mutually_exclusive_group(required=True) # only one action can and must be specified
    group.add_argument("--get", help="Dot-separated path")
    group.add_argument("--contains", help="Substring to search in config name")
    return p.parse_args()


def get_value(cfg, path):
    # function to get the specified value from the yaml
    value = cfg
    for key in path.split("."):
        if isinstance(value, list): value = value[int(key)]
        else: value = value[key]
    return value

def in_name(cfg, substring):
    # check if substring exists in the config name
    name = cfg.get("name", "")
    return substring.lower() in name.lower()


def main():
    args = parse_args()

    config = Path(args.config)
    if not config.exists(): raise FileNotFoundError(config) # yaml DNE

    with config.open() as f: # open config
        cfg = yaml.safe_load(f)

    to_get = args.get if not None else None
    if to_get:
        value = get_value(cfg, args.get) # parse for value in confid

        if isinstance(value, bool): print(str(value).lower())
        elif value is None: print("null")
        else: print(value)
        return

    name_contains = args.contains if not None else None
    if name_contains: # search for substring in config name
        print(str(in_name(cfg, args.contains)).lower())
        return


if __name__ == "__main__":
    main()

