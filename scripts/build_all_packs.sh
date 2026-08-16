#!/usr/bin/env bash
# Build every published pack: 4 variants x 3 precisions.
#
# Downloads any missing checkpoint, exports int8/float16/bfloat16 artifacts, verifies
# each against the checkpoint it came from, and zips it for release. Re-running is
# cheap: existing checkpoints and packs are left alone unless --force is passed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINTS="$ROOT/.artifacts/checkpoints"
PACKS="$ROOT/.artifacts/packs"
DIST="$ROOT/.artifacts/dist"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BUNDLE_VERSION="${BUNDLE_VERSION:-v1}"
BASE_URL="${BASE_URL:-https://github.com/javierbq/protenix-mlx/releases/download}"

# name:slug pairs, matching src/protenix_mlx_export/variants.py.
VARIANTS=(
  "protenix_tiny_default_v0.5.0:tiny"
  "protenix_mini_default_v0.5.0:mini"
  "protenix_base_default_v1.0.0:base"
  "protenix-v2:v2"
)
PRECISIONS=(int8 float16 bfloat16)

FORCE=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --only) ONLY="$2"; shift 2 ;;
    *) echo "usage: $0 [--force] [--only SLUG]" >&2; exit 2 ;;
  esac
done

mkdir -p "$CHECKPOINTS" "$PACKS" "$DIST"

for entry in "${VARIANTS[@]}"; do
  name="${entry%%:*}"
  slug="${entry##*:}"
  [[ -n "$ONLY" && "$ONLY" != "$slug" ]] && continue

  checkpoint="$CHECKPOINTS/$name.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "==> fetching $name"
    # Downloaded to .part first so an interrupted transfer is never mistaken for a
    # complete checkpoint on the next run.
    curl -fL --retry 3 -o "$checkpoint.part" \
      "https://protenix.tos-cn-beijing.volces.com/checkpoint/$name.pt"
    mv "$checkpoint.part" "$checkpoint"
  fi

  for precision in "${PRECISIONS[@]}"; do
    pack="$PACKS/protenix-$slug-mlx-$precision"
    zip="$DIST/protenix-$slug-mlx-$precision-$BUNDLE_VERSION.zip"

    if [[ -d "$pack" && $FORCE -eq 0 ]]; then
      echo "==> $slug/$precision already exported"
    else
      echo "==> exporting $slug/$precision"
      rm -rf "$pack"
      "$PYTHON" -m protenix_mlx_export export-model \
        --checkpoint "$checkpoint" \
        --model-name "$name" \
        --output "$pack" \
        --precision "$precision"
    fi

    echo "==> verifying $slug/$precision"
    "$PYTHON" "$ROOT/scripts/verify_pack.py" \
      --checkpoint "$checkpoint" --pack "$pack" --worst 3

    echo "==> packing $slug/$precision"
    "$PYTHON" -m protenix_mlx_export pack \
      --artifact "$pack" \
      --output "$zip" \
      --model-name "$name" \
      --precision "$precision" \
      --bundle-version "$BUNDLE_VERSION" \
      --release-url-prefix "$BASE_URL/weights-$slug-$BUNDLE_VERSION"
  done
done

echo
echo "packs:"
du -sh "$DIST"/*.zip
