"""The Protenix checkpoints this port targets, and their architecture contracts.

A Protenix checkpoint is ``{"model": state_dict, "model_version": str}`` and nothing
else -- no hyperparameters, no dimensions, no block counts. The graph shape lives in
upstream's config tree, keyed by model name. So unlike a Lightning checkpoint, the
architecture cannot be recovered from the file: it must be resolved here and frozen
into the artifact's ``config.json``.
"""

from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Verbatim copies of upstream's config tree. Vendored rather than depended on
#: because installing Protenix proper pulls rdkit, biotite, deepspeed and a CUDA
#: toolchain, none of which an export needs -- the config subsystem itself imports
#: only `yaml` and `ml_collections`.
_UPSTREAM_ROOT = Path(__file__).resolve().parent.parent / "_protenix_upstream"


@dataclass(frozen=True)
class Variant:
    """One released checkpoint and the packs built from it."""

    #: Upstream model name. Also the config key and the checkpoint's basename.
    name: str
    #: Short identifier used in artifact and bundle names.
    slug: str
    url: str
    #: Parameter count in millions, as published in upstream's model table. Checked
    #: against the checkpoint at export time -- a mismatch means the name resolved to
    #: the wrong config, which would otherwise only surface as a silent bad pack.
    published_params_m: float
    #: Tolerance on the above. Upstream's table counts `model.parameters()` after a
    #: strict load; a raw state dict also carries buffers, so the two differ slightly.
    params_tolerance_m: float = 1.5

    @property
    def checkpoint_filename(self) -> str:
        """Basename upstream's own downloader writes, and what we expect locally."""
        return f"{self.name}.pt"


#: The four variants in scope. `base_20250630` is deliberately absent: it is the same
#: graph as `base_default_v1.0.0` with different weights, so it costs no new porting
#: work and can be added as a pack whenever it is wanted. ESM/ISM variants are absent
#: because they require ESM2-3B alongside, which does not fit the on-device budget.
VARIANTS: tuple[Variant, ...] = (
    Variant(
        name="protenix_tiny_default_v0.5.0",
        slug="tiny",
        url="https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_tiny_default_v0.5.0.pt",
        published_params_m=109.50,
    ),
    Variant(
        name="protenix_mini_default_v0.5.0",
        slug="mini",
        url="https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_default_v0.5.0.pt",
        published_params_m=134.06,
    ),
    Variant(
        name="protenix_base_default_v1.0.0",
        slug="base",
        url="https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_base_default_v1.0.0.pt",
        published_params_m=368.48,
    ),
    # Kept because the export path is complete for it -- only the checkpoint is out of
    # reach. Upstream lists this URL in `protenix/web_service/dependency_url.py`, but
    # as of 2026-08-16 the bucket answers 403 AccessDenied for it while every other
    # released checkpoint answers 200. Third-party Hugging Face mirrors exist; none is
    # used here, because a pack's whole contract is a digest over weights of known
    # provenance.
    Variant(
        name="protenix-v2",
        slug="v2",
        url="https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix-v2.pt",
        published_params_m=464.44,
    ),
)

_BY_NAME = {variant.name: variant for variant in VARIANTS}


def get_variant(name: str) -> Variant:
    """Look one target variant up by its upstream model name."""
    try:
        return _BY_NAME[name]
    except KeyError:
        supported = ", ".join(sorted(_BY_NAME))
        message = f"unknown Protenix variant {name!r}; supported: {supported}"
        raise KeyError(message) from None


def variant_names() -> tuple[str, ...]:
    """Upstream model names this exporter can build packs for."""
    return tuple(variant.name for variant in VARIANTS)


def resolve_upstream_config(model_name: str) -> Any:
    """Resolve one model name to upstream's fully-populated config tree.

    Mirrors the two-pass merge in upstream ``runner/inference.py``: base defaults are
    deep-updated with the model-specific overrides *before* parsing, because
    ``GlobalConfigValue`` references (``c_z`` in a dozen submodules, say) must resolve
    against the overridden globals rather than the defaults.
    """
    upstream = str(_UPSTREAM_ROOT)
    # Prepended, not appended: the pinned tree must win over any real `protenix`
    # install, or the same model name could resolve to different dimensions on a
    # different machine and silently produce an incompatible pack.
    inserted = upstream not in sys.path
    if inserted:
        sys.path.insert(0, upstream)
    try:
        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config import parse_configs

        if model_name not in model_configs:
            supported = ", ".join(sorted(model_configs))
            message = (
                f"model {model_name!r} is not in the pinned upstream config tree; "
                f"available: {supported}"
            )
            raise KeyError(message)

        # Deep-copied, not spread. `{**configs_base}` copies only the top level, so a
        # deep merge would write THROUGH the shared nested dicts into the imported
        # module and leak one variant's block counts into the next export in the same
        # process -- upstream's runner has the same shape but only ever resolves one
        # model per process, so it never notices.
        merged = copy.deepcopy(
            {**configs_base, **{"data": data_configs}, **inference_configs}
        )
        _deep_update(merged, copy.deepcopy(model_configs[model_name]))
        # `arg_str=""` keeps resolution independent of this process's argv; upstream's
        # parser reads sys.argv by default and would otherwise pick up our own flags.
        return parse_configs(
            configs=merged,
            arg_str="",
            fill_required_with_null=True,
        )
    finally:
        if inserted:
            sys.path.remove(upstream)


def _deep_update(target: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge `updates` into `target`, as upstream's runner does."""
    for key, value in updates.items():
        existing = target.get(key)
        if isinstance(value, dict) and isinstance(existing, dict):
            _deep_update(existing, value)
        else:
            target[key] = value
    return target
