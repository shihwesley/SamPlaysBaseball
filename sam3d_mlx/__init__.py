"""SAM 3D Body — thin facade over mlx_vlm.models.sam3d_body.

The model code lives upstream in mlx-vlm (PR 922, branch feat/sam-3d-body,
installed editable via `pip install -e /Users/quartershots/Source/mlx-vlm`).
This package keeps only baseball-specific glue:

    __main__.py        — CLI entrypoint (python -m sam3d_mlx ...)
    sam31_detector.py  — SAM 3.1 pitcher detector wrapper
    video.py           — per-pitch video I/O and skeleton rendering

Every other submodule (config, estimator, model, backbone, batch_prep,
mhr_body, mhr_head, mhr_utils, decoder, vision, language, layers, camera,
convert_weights, generate, prompt_encoder, rope, transformer) is aliased to
the corresponding upstream module via sys.modules, so existing callsites
like `from sam3d_mlx.estimator import SAM3DBodyEstimator` keep working
without modification.

When PR 922 merges, swap the editable install for a pinned release (see
requirements.txt / pyproject.toml).
"""

import sys as _sys
from importlib import import_module as _import_module

# Every upstream submodule we want to expose under the sam3d_mlx.* namespace.
# KEEP SORTED. If you add a file upstream, add it here.
_UPSTREAM_MODULES = (
    "backbone",
    "batch_prep",
    "camera",
    "config",
    "convert_weights",
    "decoder",
    "estimator",
    "generate",
    "language",
    "layers",
    "mhr_body",
    "mhr_head",
    "mhr_utils",
    "model",
    "prompt_encoder",
    "rope",
    "transformer",
    "vision",
)

for _name in _UPSTREAM_MODULES:
    _mod = _import_module(f"mlx_vlm.models.sam3d_body.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _mod

# Re-export the top-level names the old __init__ surfaced, so code doing
# `from sam3d_mlx import SAM3DBody` keeps working.
from mlx_vlm.models.sam3d_body.config import (  # noqa: E402
    ModelConfig,
    SAM3DConfig,
    TextConfig,
    VisionConfig,
)
from mlx_vlm.models.sam3d_body.language import LanguageModel  # noqa: E402
from mlx_vlm.models.sam3d_body.model import Model, SAM3DBody  # noqa: E402
from mlx_vlm.models.sam3d_body.vision import VisionModel  # noqa: E402

__all__ = [
    "LanguageModel",
    "Model",
    "ModelConfig",
    "SAM3DBody",
    "SAM3DConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
