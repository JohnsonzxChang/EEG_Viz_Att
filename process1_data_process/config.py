"""Tiny YAML config loader with `extends:` inheritance + dict deep-merge.

Usage
-----
    from process1_data_process.config import load_config
    cfg = load_config("process1_data_process/configs/process1_clean.yaml")

The returned object behaves like a dotted-attribute namespace (and a dict).
Inheritance: a child file may have a top-level key  `extends: "<path>"`
(relative to the child file).  The child's dict is deep-merged onto the
parent's dict (child wins on scalar/leaf, lists are replaced wholesale).

A copy of the final fully-resolved config is saved alongside the run
output (use `dump_resolved_config(cfg, path)`).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class DotDict(dict):
    """A dict that also allows attribute access  (cfg.foo.bar)."""
    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(v, dict) and not isinstance(v, DotDict):
            v = DotDict(v); self[key] = v
        return v
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _deep_merge(parent: dict, child: dict) -> dict:
    """Recursive merge: child overrides parent. Lists are replaced."""
    out = copy.deepcopy(parent)
    for k, v in child.items():
        if (k in out and isinstance(out[k], dict)
                and isinstance(v, dict)):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(path: str | Path) -> DotDict:
    """Resolve `extends:` chain and return a DotDict."""
    path = Path(path).resolve()
    raw = _load_yaml(path)
    if "extends" in raw:
        parent_rel = raw.pop("extends")
        parent_path = (path.parent / parent_rel).resolve()
        parent = load_config(parent_path)
        merged = _deep_merge(parent, raw)
    else:
        merged = raw
    # convert nested dicts to DotDicts lazily on access
    return DotDict(merged)


def dump_resolved_config(cfg: dict, out_path: str | Path) -> None:
    """Persist the fully-merged config next to results, for reproducibility."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plain = json.loads(json.dumps(cfg, default=str))  # remove DotDict wrappers
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(plain, f, sort_keys=False, allow_unicode=True)
