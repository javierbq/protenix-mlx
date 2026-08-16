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
| Swift artifact loading + quantized/dense matrix layers | **working, 13 tests** |
| Swift forward graph (Pairformer, MSA, diffusion, confidence) | **not yet implemented** |
| Featurizer (MSA, templates, ligands) | **not yet implemented** |

So: packs can be built, verified, published, downloaded and loaded on device today, and
individual matrices run. **Nothing folds a sequence yet** — the trunk and diffusion
modules are still to be written.

## Models

Four variants are targeted. `protenix_base_20250630_v1.0.0` is deliberately excluded: it
is the same graph as `base_default` with a later data cutoff, so it costs no porting work
and can be added as a pack whenever wanted. The ESM/ISM variants are excluded because
they require ESM2-3B alongside, which does not fit an on-device budget.

| Variant | Params | Pairformer | Diffusion | Template | Recycles / steps | Checkpoint |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| `protenix_tiny_default_v0.5.0` | 109.5 M | 8 | 8 | inert (0 blocks) | 4 / 5 | available |
| `protenix_mini_default_v0.5.0` | 134.1 M | 16 | 8 | inert (0 blocks) | 4 / 5 | available |
| `protenix_base_default_v1.0.0` | 368.5 M | 48 | 24 | 2 blocks | 10 / 200 | available |
| `protenix-v2` | 464.4 M | 48 | 24 | 2 blocks, `c_z`=256 | 10 / 200 | **403** |

**`protenix-v2` cannot currently be packed.** Upstream lists its checkpoint URL in
`protenix/web_service/dependency_url.py`, but as of 2026-08-16 the bucket answers
`403 AccessDenied` for that one file while every other released checkpoint answers 200.
The exporter supports the variant in full — its architecture resolves, and a pack would
build the moment the file is reachable. Third-party Hugging Face mirrors exist
(`TMF001/protenix-v2-weights`, `nabbo/protenix_v2`); none is used here, because a pack's
entire contract is a digest over weights of known provenance.

### Built packs

Nine packs, all verified against the checkpoint they came from. Release ZIP sizes:

| Variant | Quantized matrices | int8 | float16 | bfloat16 |
| :--- | ---: | ---: | ---: | ---: |
| tiny (110.65 M) | 733 | **83.5 MB** | 165.2 MB | 144.6 MB |
| mini (134.07 M) | 1020 | **91.8 MB** | 184.5 MB | 163.7 MB |
| base (368.48 M) | 2645 | **214.3 MB** | 447.2 MB | 398.4 MB |

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
