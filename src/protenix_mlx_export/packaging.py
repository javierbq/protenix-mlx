"""Pack exported artifacts for distribution, and describe them to consumers.

The distribution contract is deliberately the same one boltz-mlx established, because
RayMol's weight cache already implements the consumer side of it: a flat ZIP whose
root entries are exactly the artifact files, published as a release asset, and pinned
downstream by ``(url, sha256, size, members)``. The digest is over the ZIP's bytes --
what the CDN actually serves back -- not over a local re-export.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Files an artifact directory must contain to be publishable. A pack missing
#: config.json is not merely incomplete: Protenix architecture is not recoverable
#: from weights alone, so a runtime could not rebuild the graph at all.
REQUIRED_MEMBERS = ("config.json", "manifest.json", "model.safetensors")

_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class PackedArtifact:
    """A published-ready ZIP and the facts a consumer must pin it by."""

    bundle_id: str
    path: Path
    sha256: str
    size: int
    members: tuple[str, ...]

    def weight_bundle_snippet(self, *, version: str, url: str) -> str:
        """Render the RayMol ``WeightBundle`` literal for this pack.

        Emitted rather than hand-written because all four fields must agree with the
        uploaded bytes, and a transcription slip in any one of them fails only at
        download time on a user's machine.
        """
        members = ", ".join(f"'{member}'" for member in self.members)
        return (
            "    weight_bundle = WeightBundle(\n"
            f"        id='{self.bundle_id}',\n"
            f"        version='{version}',\n"
            f"        url='{url}',\n"
            f"        sha256='{self.sha256}',\n"
            f"        size={self.size:_},\n"
            f"        members=({members},),\n"
            "    )\n"
        )


def pack_artifact(directory: Path, *, output: Path, bundle_id: str) -> PackedArtifact:
    """Zip one artifact directory into a distributable bundle.

    Written with fixed timestamps and sorted entries so the same artifact directory
    always produces byte-identical ZIPs. That makes the published digest reproducible
    for dense packs; int8 packs still differ across machines because `mx.quantize`
    runs on Metal, which is why the digest is always taken from the uploaded asset.
    """
    missing = [
        member
        for member in REQUIRED_MEMBERS
        if not (directory / member).is_file()
    ]
    if missing:
        message = f"artifact directory {directory} is missing: {', '.join(missing)}"
        raise FileNotFoundError(message)

    members = tuple(
        sorted(entry.name for entry in directory.iterdir() if entry.is_file())
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for member in members:
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with (directory / member).open("rb") as source:
                archive.writestr(info, source.read())

    return PackedArtifact(
        bundle_id=bundle_id,
        path=output,
        sha256=sha256_file(output),
        size=output.stat().st_size,
        members=members,
    )


def sha256_file(path: Path) -> str:
    """Digest a file's bytes without buffering it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_id(slug: str, precision: str) -> str:
    """Canonical bundle identifier, e.g. ``protenix-mini-mlx-int8``."""
    return f"protenix-{slug}-mlx-{precision}"
