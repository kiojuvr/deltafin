"""Read-only bootstrap for an official local Kimi-K3 checkout.

The direct-shard runtime must not require ``tools/setup_k3.py`` to copy model
metadata into ``k3-meta`` or ``tools/k3pkg``.  This module mounts the official
model directory as a Python package in memory and instantiates Moonshot's
configuration, modeling, and tokenizer classes from those source files.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import types


_PACKAGE_NAME = "deltafin_official_k3_runtime"


def direct_local_requested() -> bool:
    return (
        os.environ.get("K3_EXPERT_SOURCE") == "direct-shards"
        or os.environ.get("K3_RESIDENT_SOURCE") == "direct-shards"
    )


def model_dir() -> pathlib.Path:
    text = os.environ.get("K3_MODEL_DIR")
    if not text:
        raise RuntimeError(
            "local official K3 bootstrap requires K3_MODEL_DIR"
        )
    path = pathlib.Path(text).expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors.index.json",
        "modeling_kimi_linear.py",
        "configuration_kimi_k3.py",
        "tokenization_kimi.py",
        "tiktoken.model",
    )
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(
            f"K3_MODEL_DIR {path} is missing official files: {missing}"
        )
    return path


def metadata_dir(root: os.PathLike[str] | str) -> pathlib.Path:
    if direct_local_requested():
        return model_dir()
    return pathlib.Path(root) / "k3-meta"


def _official_package(path: pathlib.Path):
    # Import directly from the official checkpoint without creating
    # ``__pycache__`` entries beside Moonshot's source files.
    sys.dont_write_bytecode = True
    package = sys.modules.get(_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(_PACKAGE_NAME)
        package.__path__ = [str(path)]
        package.__package__ = _PACKAGE_NAME
        sys.modules[_PACKAGE_NAME] = package
    elif list(package.__path__) != [str(path)]:
        raise RuntimeError(
            "official K3 package is already mounted from another directory"
        )
    return package


def load_runtime(root: os.PathLike[str] | str):
    """Return ``(modeling_module, config, metadata_path)``."""
    path = metadata_dir(root)
    config_data = json.loads((path / "config.json").read_text())["text_config"]
    if direct_local_requested():
        _official_package(path)
        modeling = importlib.import_module(
            f"{_PACKAGE_NAME}.modeling_kimi_linear"
        )
        configuration = importlib.import_module(
            f"{_PACKAGE_NAME}.configuration_kimi_k3"
        )
        config_class = configuration.KimiLinearConfig
    else:
        modeling = importlib.import_module("k3pkg.modeling_kimi_linear")
        config_class = getattr(modeling, "KimiLinearConfig", None)
        if config_class is None:
            configuration = importlib.import_module(
                "k3pkg.configuration_kimi_k3"
            )
            config_class = configuration.KimiLinearConfig
    config = config_class(**config_data)
    config._attn_implementation = "eager"
    return modeling, config, path


def load_tokenizer(root: os.PathLike[str] | str):
    path = metadata_dir(root)
    if direct_local_requested():
        _official_package(path)
        tokenizer_class = importlib.import_module(
            f"{_PACKAGE_NAME}.tokenization_kimi"
        ).TikTokenTokenizer
        return tokenizer_class.from_pretrained(
            str(path), local_files_only=True
        )

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path), trust_remote_code=True
    )
