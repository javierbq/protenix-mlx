#!/usr/bin/env bash
# Export the feature bundles FeaturizerParityTests diffs the Swift featurizer against.
#
# These are the ONLY thing standing between "the Swift featurizer compiles" and "the
# Swift featurizer produces what upstream's data pipeline produces", so they are
# generated from the real thing: upstream Protenix + the CCD, via export-features.
# They are not committed (a few MB, and reproducible from this script); the Swift tests
# skip loudly when they are absent.
#
#   PROTENIX_ROOT_DIR=... scripts/build_feature_fixtures.sh [upstream-checkout]
#
# The cases are chosen for what breaks a featurizer, not for coverage of the alphabet:
# a single residue, every canonical residue, the 32-atom window boundaries, a
# heterodimer, a homodimer (which must share an entity id), a trimer, and the two
# residues whose reference conformers are least like the rest.
set -euo pipefail

UPSTREAM="${1:-$PWD/.artifacts/upstream}"
OUTPUT="${OUTPUT:-$PWD/.artifacts/feature_bundles}"
PYTHON="${PYTHON:-$PWD/.venv/bin/protenix-mlx}"

if [[ -z "${PROTENIX_ROOT_DIR:-}" ]]; then
  echo "PROTENIX_ROOT_DIR must point at a tree with common/components.cif" >&2
  exit 2
fi

# name:sequence. The name is what the Swift test reports, so it should say what broke.
CASES=(
  "single_residue:M"
  "tetrapeptide:GSHM"
  "all_twenty:ACDEFGHIKLMNPQRSTVWY"
  "window_exact:AAAAAAAAAAAAAAAAAAA"
  "window_plus_one:AAAAAAAAAAAAAAAAAAAA"
  "medium_chain:MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
  "heterodimer:GSHM/AWKD"
  "homodimer:GSHM/GSHM"
  "trimer:GG/GG/GG"
  "polyproline:PPPP"
  "polycysteine:CCCC"
)

mkdir -p "$OUTPUT"
for entry in "${CASES[@]}"; do
  name="${entry%%:*}"
  sequence="${entry#*:}"
  "$PYTHON" export-features \
    --sequence "$sequence" \
    --name "$name" \
    --output "$OUTPUT/$name" \
    --source "$UPSTREAM" \
    >/dev/null
  echo "  $name ($sequence)"
done
echo "wrote ${#CASES[@]} feature bundles to $OUTPUT"
