# protenix-mlx

MLX artifacts and a native **[MLX](https://github.com/ml-explore/mlx) Swift** runtime for
[Protenix](https://github.com/bytedance/Protenix), ByteDance's open AlphaFold3-class
biomolecular structure predictor — packed at **native precision (float16 / bfloat16)**
and **int8**, for macOS and iOS.

Built to the same shape as [boltz-mlx](https://github.com/javierbq/boltz-mlx): an offline
Python exporter produces versioned artifact directories, a Swift package consumes them,
and packs are distributed as release ZIPs pinned by digest — the contract RayMol's weight
cache already speaks.

## Status

| Layer | State |
| :--- | :--- |
| Python exporter (int8 / fp16 / bf16) | **working, verified against real checkpoints** |
| Artifact schema, manifest, config contract | **working** |
| Packing + distribution (ZIP, digest, `WeightBundle`) | **working** |
| Swift artifact loading + quantized/dense matrix layers | **working** |
| Swift core layers, verified against PyTorch | **working** — Transition, AdaLN, AttentionPairBias, TriangleMultiplication ×2, TriangleAttention ×2, OuterProductMean, MSAPairWeightedAveraging |
| Swift trunk (input embedder, relpos, MSA, Pairformer, recycling) | **working, verified** |
| Swift diffusion (atom encoder/decoder, transformer, conditioning, sampler) | **working, verified** |
| Swift end-to-end fold (sequence → coordinates → PDB) | **working** |
| Swift featurizer (canonical-20 protein, multi-chain) | **working** — bitwise-identical to upstream's data pipeline |
| Swift confidence head (pLDDT / PAE / PDE / resolved) | **working, verified** |
| Distogram head, templates, real MSA, ligands, nucleic acids | not implemented |

**55 Swift tests, 92 Python tests.** Every learned component of the structure path —
trunk, diffusion and the confidence head — reproduces upstream PyTorch to within 2e-4,
verified by fixtures recorded from the real modules.

**A fold needs nothing but a sequence and a pack.** Featurization runs in Swift, so
there is no Python, no torch, no rdkit and no 624 MB `components.cif` on the machine
doing the fold — for the canonical 20, everything upstream looks up in the CCD is a
constant, and those constants ship in the package (~90 KB). A 4-residue peptide folds
with correct backbone bond lengths (N–CA ≈ 1.5 Å) in ~1 s on an M-series Mac through the
int8 tiny pack.

> Accuracy is bounded by what is wired up: single-sequence (no MSA search) and no
> templates. The tiny/mini v0.5.0 models at 5–20 diffusion steps produce chemically
> sensible local geometry, not production-grade folds. The point proven here is that the
> *network* runs faithfully in MLX Swift, not that this replaces a full Protenix
> inference stack.

### Folding a sequence

```bash
# Fold, and score the fold. Nothing but the pack is required.
ProtenixMLXCLI predict --model artifacts/protenix-base-mlx-int8 \
  --sequence GSHM --output GSHM.pdb --confidence

# A complex: separate chains with "/". Identical sequences fold as copies of one
# entity, which is what tells the model a homodimer's chains are related.
ProtenixMLXCLI predict --model artifacts/protenix-base-mlx-int8 \
  --sequence "GSHM/GSHM" --output dimer.pdb
```

`--confidence` runs the confidence head and writes per-atom pLDDT into the B-factor
column, which is what viewers colour by. Without it the column is `0.00` rather than a
fabricated value.

The Python featurizer is still there for inputs the table cannot serve — anything that
genuinely needs the CCD — and is what the Swift featurizer is tested against:

```bash
export PROTENIX_ROOT_DIR=/path/to/tree-with-common   # components.cif + its rdkit pickle
protenix-mlx export-features --sequence GSHM --output bundles/GSHM
ProtenixMLXCLI predict --model artifacts/protenix-tiny-mlx-int8 \
  --features bundles/GSHM --output GSHM.pdb
```

Both paths produce byte-identical structures for a canonical-20 protein. The bundle it
writes is now a function of its sequences alone: upstream applies a random rotation and
translation to every reference conformer from the global numpy RNG, which makes a fold
irreproducible even at a fixed diffusion seed, so that augmentation is off by default
here (`--augment` restores it, `--seed` makes even that reproducible).

### The residue table

`protenix-mlx export-residue-templates` freezes the canonical-20 reference conformers
into `Sources/ProtenixMLX/Resources/residue_templates.json`. It **derives** the table by
running upstream's own featurizer and slicing per residue — never by reimplementing the
CCD lookup — and refuses to write one unless two properties hold, both checked rather
than assumed:

- a residue's conformer does not depend on its neighbours (built in several sequence
  contexts, required to agree bitwise);
- position changes a residue in exactly one way, the C-terminal `OXT`.

Coordinates are stored already centred, so `ref_pos` is a copy rather than a sum and the
Swift featurizer agrees with Python *bitwise* rather than within a tolerance.

### Parity against PyTorch

A port of a 464M-parameter network cannot be validated by reading it. `protenix-mlx
make-fixtures` instantiates each upstream module at small dimensions with seeded random
weights, runs it, and records weights/inputs/outputs; the Swift tests replay the same
weights and inputs and assert the same outputs to within 2e-4.

Two properties this harness is built to have:

- **Fixtures reproduce bit-identically from their seed**, so a stale tree cannot drift
  into agreement.
- **Missing fixtures skip loudly**, printing `SKIP` per case rather than reporting green.
  A parity suite that silently passes when its reference data is absent is worse than no
  suite at all.

Verified by mutation: swapping the outgoing/incoming permutation in
`TriangleMultiplication` — the subtlest line in the port — moves the deviation from below
2e-4 to 1.09, and the corresponding test fails by name.

The featurizer has no PyTorch module to imitate, so it is checked differently: bundles
exported by the real upstream pipeline (`scripts/build_feature_fixtures.sh`) are compared
**exactly**, all 17 tensors, over single chains, every canonical residue, both atom-window
boundaries, hetero- and homodimers and a trimer. Also mutation-verified — dropping the
C-terminal `OXT`, forgetting that equal residue indices in different chains are different
residues, and an off-by-one in the atom window each fail by name.

Upstream imports with torch alone provided `LAYERNORM_TYPE` is set to anything other than
`fast_layernorm`; the default tries to build a CUDA extension. `fixtures.py` sets it
automatically.

## Models

Four variants are targeted. `protenix_base_20250630_v1.0.0` is deliberately excluded: it
is the same graph as `base_default` with a later data cutoff, so it costs no porting work
and can be added as a pack whenever wanted. The ESM/ISM variants are excluded because
they require ESM2-3B alongside, which does not fit an on-device budget.

| Variant | Params | Pairformer | Diffusion | Template | Recycles / steps | Checkpoint |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `protenix_tiny_default_v0.5.0` | 109.5 M | 8 | 8 | inert (0 blocks) | 4 / 5 | official |
| `protenix_mini_default_v0.5.0` | 134.1 M | 16 | 8 | inert (0 blocks) | 4 / 5 | official |
| `protenix_base_default_v1.0.0` | 368.5 M | 48 | 24 | 2 blocks | 10 / 200 | official |
| `protenix-v2` | 464.4 M | 48 | 24 | 2 blocks, `c_z`=256 | 10 / 200 | **mirror** |

### protenix-v2 provenance

Three of the four checkpoints come from ByteDance's own bucket. **`protenix-v2` does
not.** Its official URL has answered `403 AccessDenied` since April 2026, deliberately —
a ByteDance collaborator stated in
[issue #296](https://github.com/bytedance/Protenix/issues/296) that "accessibility of the
protenix-v2 checkpoint is currently under review as part of our company-level internal
evaluation process". Every other released checkpoint answers 200 from the same network.

Packs for it are built from `TMF001/protenix-v2-weights` on Hugging Face. What that rests
on:

- The mirror is **byte-faithful where it can be checked**: its copy of
  `protenix_mini_default_v0.5.0.pt` hashes to
  `3803340c…791f81`, identical to the file downloaded straight from ByteDance's CDN.
- The v2 file **audits clean** (`scripts/audit_checkpoint.py`): exactly 464,442,431
  parameters, `c_z` 256, block depths 48/4/24/4, a module tree matching the
  officially-sourced base checkpoint on all 4174 tensor names, and weight statistics of a
  trained network (LayerNorm gains mean 0.43 / std 0.55, none left at exactly 1.0).

What it does **not** rest on:

- **No official checksum exists for any Protenix checkpoint.** All 30 GitHub releases
  carry zero binary assets, the repo publishes no hashes, and `download_from_url`'s only
  integrity check is that `torch.load` succeeds. So nothing authoritative can confirm the
  weight *values*; that needs an accuracy benchmark.
- The uploader states they have no affiliation with the Protenix team, and has not said
  how they obtained the file.

Accordingly the v2 checkpoint is pinned by digest
(`8f931f97…0d599` — the exporter refuses to build from anything else), and every v2
artifact records `"checkpoint_provenance": "mirror"` plus its source URL in `config.json`,
so a pack cannot later be mistaken for one built from an official download. Consider
whether redistributing it is appropriate for your setting before publishing those packs.

### Built packs

Twelve packs, all verified against the checkpoint they came from. Release ZIP sizes:

| Variant | Quantized matrices | int8 | float16 | bfloat16 |
| :--- | ---: | ---: | ---: | ---: |
| tiny (110.65 M) | 733 | **87.5 MB** | 173.2 MB | 151.6 MB |
| mini (134.07 M) | 1020 | **96.2 MB** | 193.5 MB | 171.7 MB |
| base (368.48 M) | 2645 | **224.7 MB** | 468.9 MB | 417.7 MB |
| v2 (464.44 M) *mirror* | 2645 | **299.4 MB** | 610.5 MB | 560.6 MB |

Weight fidelity, worst tensor per pack (tiny):

| Precision | Pearson r (min / mean) | Margin to tolerance |
| :--- | :--- | ---: |
| int8 (affine, group 64) | 0.999942 / 0.999989 | 6.4× |
| float16 | 0.999990 / 1.000000 | 104.9× |
| bfloat16 | 0.999308 / 0.999986 | 13.5× |

bfloat16 is the *less* faithful dense pack despite being the larger-range format — the
released checkpoints are float32 throughout, and float16 is the strictly closer narrowing
of float32. bfloat16 is offered because it is the width the model trains in.

### Verifying a pack: why not just correlation?

`scripts/verify_pack.py` gates on `atol + rtol`, not on Pearson r, because r alone is
misleading in both directions. `protenix_mini` ships a Pairformer pair-bias projection
that training collapsed to the noise floor —
`pairformer_stack.blocks.0.attention_pair_bias.linear_nobias_z` has std `1e-5` against
tiny's `0.33`, with some 64-wide quantization groups spanning `1e-36`. Quantizing that
reconstructs it with an absolute error around `7e-6` but a correlation of only 0.98:
numerically fine, yet it fails any naive r ≥ 0.999 gate. The tolerance is relative to
each tensor's own magnitude with a floor set by the model's *median* peak weight — a
median, because normalizing by the maximum would let `upper_bins`' `1e6` sentinel make
every real weight in the model look negligible.

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

# 1. Export a pack (downloads nothing; point at an upstream checkpoint)
.venv/bin/protenix-mlx export-model \
  --checkpoint protenix_tiny_default_v0.5.0.pt \
  --model-name protenix_tiny_default_v0.5.0 \
  --output artifacts/protenix-tiny-mlx-int8 \
  --precision int8

# 2. Prove the pack reconstructs the checkpoint
.venv/bin/python scripts/verify_pack.py \
  --checkpoint protenix_tiny_default_v0.5.0.pt \
  --pack artifacts/protenix-tiny-mlx-int8

# 3. Zip it and print the RayMol WeightBundle literal to paste downstream
.venv/bin/protenix-mlx pack \
  --artifact artifacts/protenix-tiny-mlx-int8 \
  --output dist/protenix-tiny-mlx-int8-v1.zip \
  --model-name protenix_tiny_default_v0.5.0 --precision int8 \
  --release-url-prefix https://github.com/javierbq/protenix-mlx/releases/download/weights-tiny-v1
```

`scripts/build_all_packs.sh` does all of the above for every variant and precision.

### Swift

> `swift build` compiles the package but **cannot build MLX's Metal shaders**, so a
> binary built that way dies with `Failed to load the default metallib`. Use
> `xcodebuild`, which can.

```bash
xcodebuild build -scheme ProtenixMLXCLI -destination 'platform=OS X' \
  -derivedDataPath .build/xcode -skipPackagePluginValidation -skipMacroValidation

.build/xcode/Build/Products/Debug/ProtenixMLXCLI inspect \
  --model artifacts/protenix-tiny-mlx-int8
```

`inspect` validates a pack against its own manifest, prints the architecture it
declares, and runs a matrix from each major stack to prove the weights are usable and
not merely present.

## How an artifact is laid out

```
protenix-tiny-mlx-int8/
├── manifest.json       # schema version, per-tensor shape/dtype, quantization block
├── config.json         # the architecture contract (see below)
└── model.safetensors   # every array the manifest declares
```

### Why `config.json` is mandatory

A Protenix checkpoint is `{"model": state_dict, "model_version": str}` and nothing else —
no dimensions, no block counts. The four variants share tensor *names* while differing in
depth and width, so weights alone cannot tell a runtime whether it holds an 8-block or a
48-block Pairformer. The architecture is resolved at export time from upstream's config
tree (vendored, pinned by commit, under `src/_protenix_upstream/`) and frozen into
`config.json`. A pack without it is unusable, and the loader treats it as fatal.

This is the main structural difference from boltz-mlx, whose Lightning checkpoints carry
their own hyperparameters.

### int8 packing

Every rank-2 `.weight` is quantized with MLX affine int8 at group size 64, its contracted
axis zero-padded up to a multiple of the group, and stored as three arrays
(`weight` / `scales` / `biases`). The manifest records both the padded `physical_shape`
and the true `logical_shape`; the runtime pads activations to match, so the padding
cancels. Everything else — LayerNorm gains, the confidence head's rank-3 `plddt_weight`
and `resolved_weight` stacks — stays dense at float16.

Two details worth knowing:

- **`confidence_head.upper_bins` is stored at float32** in the int8 and float16 packs. Its
  last bin edge is a `1e6` sentinel that becomes `inf` in float16. Rather than rewriting a
  released model's constant or relying on `inf` comparing correctly, any tensor that
  cannot be represented at the pack width is kept at float32 and declared as such. It
  stays bfloat16 in a bf16 pack, because bfloat16 keeps float32's exponent range.
- **`.bias` and `.biases` are different things.** The former is a layer's additive term
  (Protenix mixes `nn.Linear` and `LinearNoBias` freely); the latter is the quantizer's
  zero-point. The runtime looks up exactly `.bias` and never confuses the two.

## Published weights

All four variants are published as GitHub release assets, three precisions each, pinned by
digest in **[WEIGHTS.md](WEIGHTS.md)**:

| Variant | int8 | Release |
| :--- | ---: | :--- |
| tiny (110.65 M) | 83.5 MB | [weights-tiny-v1](https://github.com/javierbq/protenix-mlx/releases/tag/weights-tiny-v1) |
| mini (134.07 M) | 91.8 MB | [weights-mini-v1](https://github.com/javierbq/protenix-mlx/releases/tag/weights-mini-v1) |
| base (368.48 M) | 214.3 MB | [weights-base-v1](https://github.com/javierbq/protenix-mlx/releases/tag/weights-base-v1) |
| v2 (464.44 M) *mirror* | 285.5 MB | [weights-v2-v1](https://github.com/javierbq/protenix-mlx/releases/tag/weights-v2-v1) |

```bash
# grab a pack and unzip it into an artifact directory
curl -L -o tiny-int8.zip \
  https://github.com/javierbq/protenix-mlx/releases/download/weights-tiny-v1/protenix-tiny-mlx-int8-v1.zip
unzip tiny-int8.zip -d artifacts/protenix-tiny-mlx-int8
```

## Distribution

Packs are published as release ZIPs and pinned downstream by `(url, sha256, size,
members)` — the same contract RayMol's `WeightBundle` / weight cache implements, so
integration is a predictor class and a bundle literal rather than new download code.
`protenix-mlx pack --release-url-prefix ...` emits that literal directly, because all four
fields have to agree with the uploaded bytes and a transcription slip in any one of them
fails only later, on a user's machine.

The digest is always taken from the uploaded asset. Dense packs *are* byte-reproducible
from a checkpoint (the ZIP is written with fixed timestamps and sorted entries), but int8
packs are not: `mx.quantize` runs on Metal and is not guaranteed identical across machines.

## Credits & license

Derived from **Protenix** (ByteDance) — https://github.com/bytedance/Protenix — which is
released under the **Apache License 2.0**, as is this port. `src/_protenix_upstream/`
contains verbatim copies of upstream's config subsystem, pinned at commit
`4c355be4553512f72453ecbfb65e69f4c35d1413`. Uses
[mlx-swift](https://github.com/ml-explore/mlx-swift).
