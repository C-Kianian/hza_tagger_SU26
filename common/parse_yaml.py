#!/usr/bin/env python3
"""
File: parse_yaml.py
This script was made to parse a yaml config, in particular a training config file, used in evaluate.sh
"""
import argparse
from pathlib import Path
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML config file")
    group = p.add_mutually_exclusive_group(required=True) # only one action can and must be specified
    group.add_argument("--get", help="Dot-separated path")
    group.add_argument("--in-name", help="Substring to search in config name")
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

def get_tasks(cfg):
    # get a list of tasks that the model performs, returns [task_names], [task_types]
    try:
        modules = get_value(cfg, "model.model.init_args.tasks.init_args.modules") # get modules list
    except (KeyError, IndexError, TypeError):
        return [], []

    names = []
    tasks = []
    losses = []

    for module in modules: # for each module
        class_path = module.get("class_path", "") # get the salt task name
        init_args = module.get("init_args", {}) # get init args which will contain the user defined name

        # get the name set by user
        name = init_args.get("name", "")
        # get loss path
        loss_path = init_args.get("loss", "").get("class_path", "")

        # get only the task and loss names
        task_type = class_path.split(".")[-1] if class_path else "UnknownTask"
        loss = loss_path.split(".")[-1] if loss_path else "UnknownLoss"

        names.append(name)
        tasks.append(task_type)
        losses.append(loss)

    return names, tasks, losses # names of the tasks, the SALT task name, the loss used

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

    name_contains = args.in_name if not None else None
    if name_contains: # search for substring in config name
        print(str(in_name(cfg, args.in_name)).lower())
        return


if __name__ == "__main__":
    main()

