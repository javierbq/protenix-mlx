# Published MLX weight packs

Every pack is a flat ZIP (`config.json` + `manifest.json` + `model.safetensors`), pinned
below by the sha256 of the ZIP GitHub serves. `tiny`/`mini`/`base` are exported from
ByteDance's public checkpoints (Apache 2.0); `v2` is mirror-sourced — see its release notes
and `src/protenix_mlx_export/variants.py`.

| Variant | Precision | Size | Provenance | sha256 |
| :--- | :--- | ---: | :--- | :--- |
| tiny | int8 | 83.5 MB | official | `11716a7c69d10c0b…` |
| tiny | float16 | 165.2 MB | official | `73fdf86436646138…` |
| tiny | bfloat16 | 144.6 MB | official | `77777d1a40dc45b5…` |
| mini | int8 | 91.8 MB | official | `ea3a8f81ad8ce055…` |
| mini | float16 | 184.5 MB | official | `159206251babea99…` |
| mini | bfloat16 | 163.7 MB | official | `9dc3a7397d054421…` |
| base | int8 | 214.3 MB | official | `6a405fbfb0f3b331…` |
| base | float16 | 447.2 MB | official | `b849de70134b8b64…` |
| base | bfloat16 | 398.4 MB | official | `ee2d1d8e2070c232…` |
| v2 | int8 | 285.5 MB | mirror | `1eef8793e18be9f4…` |
| v2 | float16 | 582.2 MB | mirror | `130018dec413a2c8…` |
| v2 | bfloat16 | 534.6 MB | mirror | `80232df8dfb6b04d…` |

## Full URLs and digests

```json
[
  {
    "variant": "tiny",
    "precision": "int8",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-tiny-v1/protenix-tiny-mlx-int8-v1.zip",
    "sha256": "11716a7c69d10c0b9c90410503bc2b4b05a3c83f8b39c572bf4d962a56094858",
    "size": 87543789
  },
  {
    "variant": "tiny",
    "precision": "float16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-tiny-v1/protenix-tiny-mlx-float16-v1.zip",
    "sha256": "73fdf864366461380cee4b316b354018118e55f5d0ce6ee7be698d30b64274a1",
    "size": 173209981
  },
  {
    "variant": "tiny",
    "precision": "bfloat16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-tiny-v1/protenix-tiny-mlx-bfloat16-v1.zip",
    "sha256": "77777d1a40dc45b5de900978e2d0018fd361e93fdebb4384c4928c07866b99b9",
    "size": 151635809
  },
  {
    "variant": "mini",
    "precision": "int8",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-mini-v1/protenix-mini-mlx-int8-v1.zip",
    "sha256": "ea3a8f81ad8ce055b5b3d1dba92f595b5284dfc2c13b8eedd0458827babc0cea",
    "size": 96248656
  },
  {
    "variant": "mini",
    "precision": "float16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-mini-v1/protenix-mini-mlx-float16-v1.zip",
    "sha256": "159206251babea99824110da45546707ed1428f0cf1b5a38025cc2db76f7337d",
    "size": 193494769
  },
  {
    "variant": "mini",
    "precision": "bfloat16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-mini-v1/protenix-mini-mlx-bfloat16-v1.zip",
    "sha256": "9dc3a7397d054421d1b9a64959477c927a86b56446b1f24fa642a57d4a5a8837",
    "size": 171699965
  },
  {
    "variant": "base",
    "precision": "int8",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-base-v1/protenix-base-mlx-int8-v1.zip",
    "sha256": "6a405fbfb0f3b331315bc317106f22ed5daf10eb7b9b1122c4eae5db77b26977",
    "size": 224688268
  },
  {
    "variant": "base",
    "precision": "float16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-base-v1/protenix-base-mlx-float16-v1.zip",
    "sha256": "b849de70134b8b64f63274c7ec6b578b8ab83d22960d22fd76226ad76c17873d",
    "size": 468919887
  },
  {
    "variant": "base",
    "precision": "bfloat16",
    "provenance": "official",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-base-v1/protenix-base-mlx-bfloat16-v1.zip",
    "sha256": "ee2d1d8e2070c2325eff3c2f5803ebe1266e8c182ea4e58fa14a1f6d280907e9",
    "size": 417731102
  },
  {
    "variant": "v2",
    "precision": "int8",
    "provenance": "mirror",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-v2-v1/protenix-v2-mlx-int8-v1.zip",
    "sha256": "1eef8793e18be9f4a5dd040392e4358fbac1bfbf1d63dee33b7a5bc9398d4432",
    "size": 299351203
  },
  {
    "variant": "v2",
    "precision": "float16",
    "provenance": "mirror",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-v2-v1/protenix-v2-mlx-float16-v1.zip",
    "sha256": "130018dec413a2c8054fa3f89c99bde7158a403b58462f5c429144098a06d860",
    "size": 610453479
  },
  {
    "variant": "v2",
    "precision": "bfloat16",
    "provenance": "mirror",
    "url": "https://github.com/javierbq/protenix-mlx/releases/download/weights-v2-v1/protenix-v2-mlx-bfloat16-v1.zip",
    "sha256": "80232df8dfb6b04db6163a6c845598d45f3bef9239f909fcd66e23688fb2c00e",
    "size": 560585456
  }
]
```
