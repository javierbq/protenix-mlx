"""Command-line interface for offline Protenix MLX artifact generation."""

from __future__ import annotations

from pathlib import Path

import click

from protenix_mlx_export.schema import SCHEMA_VERSION
from protenix_mlx_export.variants import VARIANTS, get_variant, variant_names

_PRECISIONS = ("int8", "float16", "bfloat16")


@click.group()
@click.version_option(version=f"schema {SCHEMA_VERSION}", prog_name="protenix-mlx")
def cli() -> None:
    """Export Protenix model artifacts for MLX Swift."""


@cli.command("list-variants")
def list_variants_command() -> None:
    """Show the Protenix checkpoints this exporter targets."""
    click.echo(f"{'model name':32s} {'slug':6s} {'params':>9s}  checkpoint")
    for variant in VARIANTS:
        click.echo(
            f"{variant.name:32s} {variant.slug:6s} "
            f"{variant.published_params_m:8.2f}M  {variant.checkpoint_filename}"
        )


@cli.command("export-model")
@click.option(
    "--checkpoint",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to the downloaded upstream .pt checkpoint.",
)
@click.option(
    "--model-name",
    type=click.Choice(variant_names()),
    required=True,
    help="Upstream model name. Protenix checkpoints carry no architecture metadata, "
    "so this selects the config the artifact is built against.",
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--precision",
    type=click.Choice(_PRECISIONS),
    default="int8",
    show_default=True,
    help="int8 quantizes every matrix; float16/bfloat16 store them dense (~2x larger).",
)
def export_model_command(
    checkpoint: Path,
    model_name: str,
    output: Path,
    precision: str,
) -> None:
    """Export one checkpoint into an MLX artifact directory."""
    from protenix_mlx_export.model_export import (  # noqa: PLC0415
        Precision,
        export_checkpoint,
    )

    manifest = export_checkpoint(
        checkpoint=checkpoint,
        output=output,
        model_name=model_name,
        precision=Precision(precision),
    )
    written = (output / "model.safetensors").stat().st_size
    click.echo(
        f"wrote {len(manifest.tensors)} tensors to {output} "
        f"({written / (1 << 20):.1f} MiB, precision={precision})"
    )


@cli.command("make-fixtures")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--source",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Path to a Protenix checkout. Needed unless it is already importable.",
)
@click.option("--seed", type=int, default=0, show_default=True)
def make_fixtures_command(output: Path, source: Path | None, seed: int) -> None:
    """Record PyTorch module-boundary fixtures for the Swift parity tests.

    Needs upstream's runtime dependencies (torch, scipy, rdkit, biotite). The Swift
    tests skip loudly when the fixture tree is absent, so this is only required when
    changing a layer.
    """
    from protenix_mlx_export.fixtures import make_fixtures  # noqa: PLC0415

    written = make_fixtures(output=output, source=source, seed=seed)
    for path in written:
        click.echo(f"  {path.name}")
    click.echo(f"wrote {len(written)} fixtures to {output}")


@cli.command("export-features")
@click.option(
    "--sequence",
    required=True,
    help='Protein sequence. Separate chains of a complex with "/".',
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--name", default="prediction", show_default=True)
@click.option(
    "--source",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="A Protenix checkout, if not already importable.",
)
@click.option(
    "--augment/--no-augment",
    default=False,
    show_default=True,
    help="Apply upstream's random rotation/translation per reference conformer. "
         "Off makes the bundle a function of the sequence alone.",
)
@click.option(
    "--seed", type=int, default=0, show_default=True,
    help="Seeds the augmentation RNG. Ignored without --augment.",
)
def export_features_command(
    sequence: str, output: Path, name: str, source: Path | None,
    augment: bool, seed: int,
) -> None:
    """Featurize one or more chains into a Swift-loadable feature bundle.

    Needs upstream Protenix importable and the CCD cache reachable via
    PROTENIX_ROOT_DIR (a tree with common/components.cif and its rdkit pickle).
    Runs a single-sequence dummy MSA -- no alignment or templates.

    Identical sequences are folded as copies of one entity, which is what makes a
    homodimer's relp say the two chains are related.
    """
    from protenix_mlx_export.feature_export import export_features  # noqa: PLC0415

    features = export_features(
        sequence=sequence, output=output, name=name, source=source,
        augment=augment, seed=seed)
    chains = ", ".join(
        f"{chain_id}:{len(chain_sequence)}" for chain_id, chain_sequence in features.chains
    )
    click.echo(
        f"wrote {features.token_count} tokens / {features.atom_count} atoms "
        f"({chains}) to {output}")


@cli.command("export-residue-templates")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=Path("Sources/ProtenixMLX/Resources/residue_templates.json"),
    show_default=True,
    help="Where to write the table. The default is where the Swift package reads it.",
)
@click.option(
    "--source",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="A Protenix checkout, if not already importable.",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    show_default=True,
    help="Rebuild sample sequences from the table and require upstream to agree bitwise.",
)
def export_residue_templates_command(
    output: Path, source: Path | None, verify: bool
) -> None:
    """Freeze the canonical-20 reference conformers into a table Swift can ship.

    This is what removes torch, rdkit and a 624 MB components.cif from the machine doing
    the fold: for a standard amino acid, everything the featurizer looks up in the CCD is
    a constant. Derived by running upstream's own featurizer, never by reimplementing it.
    """
    from protenix_mlx_export.residue_templates import (  # noqa: PLC0415
        load_templates,
        write_templates,
    )

    commit = ""
    if source is not None:
        head = source / ".git" / "HEAD"
        if head.is_file():
            commit = _resolve_git_head(source)
    templates = write_templates(output, source=source, upstream_commit=commit)
    atoms = sum(len(t.atoms) + len(t.terminal_atoms) for t in templates)
    click.echo(
        f"wrote {len(templates)} residues / {atoms} reference atoms to {output} "
        f"({output.stat().st_size / 1024:.1f} KiB)"
    )
    if verify:
        from protenix_mlx_export.residue_templates import (  # noqa: PLC0415
            verify_templates,
        )

        sequences = [
            "GSHM",
            "ACDEFGHIKLMNPQRSTVWY",
            "M",
            "GSHM/AWKD",
            "GSHM/GSHM",
            "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        ]
        verify_templates(load_templates(output), sequences=sequences, source=source)
        click.echo(f"verified bitwise against upstream on {len(sequences)} sequences")


def _resolve_git_head(repository: Path) -> str:
    """The commit a checkout is on, for the table's provenance field."""
    head = (repository / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        reference = repository / ".git" / head.removeprefix("ref: ")
        if reference.is_file():
            return reference.read_text(encoding="utf-8").strip()
        packed = repository / ".git" / "packed-refs"
        if packed.is_file():
            target = head.removeprefix("ref: ")
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {target}"):
                    return line.split(" ", 1)[0]
        return ""
    return head


@cli.command("pack")
@click.option(
    "--artifact",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    required=True,
    help="An artifact directory produced by export-model.",
)
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option(
    "--model-name",
    type=click.Choice(variant_names()),
    required=True,
)
@click.option("--precision", type=click.Choice(_PRECISIONS), required=True)
@click.option(
    "--bundle-version",
    default="v1",
    show_default=True,
    help="Distribution version, bumped when republishing the same model.",
)
@click.option(
    "--release-url-prefix",
    default=None,
    help="Release download prefix; when given, prints the RayMol WeightBundle literal.",
)
def pack_command(
    artifact: Path,
    output: Path,
    model_name: str,
    precision: str,
    bundle_version: str,
    release_url_prefix: str | None,
) -> None:
    """Zip an artifact directory and report the digest consumers must pin."""
    from protenix_mlx_export.packaging import (  # noqa: PLC0415
        bundle_id,
        pack_artifact,
    )

    variant = get_variant(model_name)
    identifier = bundle_id(variant.slug, precision)
    packed = pack_artifact(artifact, output=output, bundle_id=identifier)
    click.echo(f"{packed.path}")
    click.echo(f"  id      {packed.bundle_id}")
    click.echo(f"  sha256  {packed.sha256}")
    click.echo(f"  size    {packed.size:_} bytes")
    click.echo(f"  members {', '.join(packed.members)}")
    if release_url_prefix is not None:
        url = f"{release_url_prefix.rstrip('/')}/{output.name}"
        click.echo("\n" + packed.weight_bundle_snippet(version=bundle_version, url=url))
