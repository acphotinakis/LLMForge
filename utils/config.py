"""
Configuration management for Research LLM.

Supports YAML configs with dot-notation access, deep merging,
and command-line override via `key=value` strings.
"""

from __future__ import annotations

import copy
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class Config:
    """
    Hierarchical configuration object with dot-notation access.

    Wraps a nested dict and provides:
      - cfg.model.n_layers         (attribute access)
      - cfg["model"]["n_layers"]   (dict access)
      - cfg.get("missing", default)
      - repr / str for debugging
    """

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            if isinstance(value, dict):
                self._data[key] = Config(value)
            else:
                self._data[key] = value

    # ---- access ----

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, dict):
            self._data[key] = Config(value)
        else:
            self._data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.__setattr__(key, value)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ---- iteration ----

    def items(self) -> Iterator[Tuple[str, Any]]:
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    # ---- serialization ----

    def to_dict(self) -> Dict[str, Any]:
        """Recursively convert to plain dict."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({yaml.dump(self.to_dict(), default_flow_style=False).strip()})"

    def __str__(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


# ---- Loading ----

def load_config(path: Union[str, Path]) -> Config:
    """Load a YAML config file and return a Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return Config(data)


def merge_configs(base: Config, override: Config) -> Config:
    """
    Deep-merge two Config objects.  Values in `override` take precedence.
    Nested dicts are merged recursively; scalars are replaced.
    """
    base_dict = base.to_dict()
    override_dict = override.to_dict()
    merged = _deep_merge(base_dict, override_dict)
    return Config(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def apply_overrides(cfg: Config, overrides: List[str]) -> Config:
    """
    Apply command-line overrides of the form ``key.subkey=value``.

    Example::

        overrides = ["training.learning_rate=1e-4", "model.n_layers=24"]
        cfg = apply_overrides(cfg, overrides)
    """
    data = cfg.to_dict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be in 'key=value' format, got: '{override}'")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = _parse_value(raw_value.strip())
        _set_nested(data, keys, value)
    return Config(data)


def _parse_value(raw: str) -> Any:
    """Try to parse a string into int, float, bool, None, or keep as string."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in ("null", "none", "~"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _set_nested(data: dict, keys: List[str], value: Any) -> None:
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


# ---- Preset model sizes ----

MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "nano": dict(n_layers=4,  n_heads=4,  d_model=256,  d_ff=1024,  context_length=512),
    "small": dict(n_layers=12, n_heads=12, d_model=768,  d_ff=3072,  context_length=1024),
    "medium": dict(n_layers=24, n_heads=16, d_model=1024, d_ff=4096,  context_length=2048),
    "large": dict(n_layers=36, n_heads=20, d_model=1280, d_ff=5120,  context_length=2048),
    "xl":    dict(n_layers=48, n_heads=25, d_model=1600, d_ff=6400,  context_length=2048),
}


def resolve_model_config(cfg: Config) -> Config:
    """
    If ``cfg.model.preset`` is set and explicit architecture values are null/absent,
    fill in values from the preset table.
    """
    preset_name = cfg.model.get("preset", None)
    if preset_name and preset_name in MODEL_PRESETS:
        preset = MODEL_PRESETS[preset_name]
        data = cfg.to_dict()
        for key, val in preset.items():
            # Only override if the field is null/missing
            if data["model"].get(key) is None:
                data["model"][key] = val
        return Config(data)
    return cfg
