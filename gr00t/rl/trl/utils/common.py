# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import importlib

import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import


def recursive_set_struct(cfg, struct_value: bool):
    OmegaConf.set_struct(cfg, struct_value)
    if isinstance(cfg, DictConfig):
        for key in cfg.keys():
            try:
                value = cfg[key]
                if isinstance(value, (DictConfig, ListConfig)):
                    recursive_set_struct(value, struct_value)
            except Exception:
                # print(e)
                pass
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                recursive_set_struct(item, struct_value)


def custom_instantiate(d, _resolve=True, **add_kwargs):
    d = d.copy()
    if isinstance(d, DictConfig):
        if _resolve:
            d = OmegaConf.to_container(d, resolve=_resolve)
        else:
            recursive_set_struct(d, False)
    if d.get("_recursive_", None) == True:
        assert False, "recursive is not supported"
    d.pop("_recursive_", None)  # remove _recursive_
    d.pop("_convert_", None)  # remove _convert_
    d.pop("_partial_", None)  # remove _partial_
    _type = import_type_from_str(d.pop("_target_"))
    return _type(**d, **add_kwargs)
