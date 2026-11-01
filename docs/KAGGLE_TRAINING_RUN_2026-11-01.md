# Kaggle single-frame segmentation training — 2026-11-01

**State:** private kernel `guzzbr/draw-segformer-synthetic-only-20261101`, version 2, completed successfully. Version 1 failed before completing an epoch and produced no model artifacts. Version 2's metrics, ONNX validation, data fingerprint, and artifact hashes were downloaded and checked.

## Scope and label authority

This is single-frame semantic segmentation for visible ink (`class 3`) on synthetic 512×512 scenes. It is not temporal new-ink detection. No temporal pairs or temporal masks were attached. The real review pool and the 719-frame pilot were excluded because they contain only unapproved pseudo-labels. No human-reviewed real pixel masks or independent real holdout are available. Therefore every result from this run is exploratory synthetic evidence only; it cannot establish camera quality or authorize live integration.

## Pre-kernel data audit

Kaggle status and metadata queries confirmed these inputs are `ready`, current version 1, and `isPrivate=true`:

| Input | Files | Bytes | Role |
|---|---:|---:|---|
| `guzzbr/draw-synthetic-v1-part1` | 1,201 | 225,467,028 | 400 train scenes + split manifest |
| `guzzbr/draw-synthetic-v1-part2` | 1,200 | 225,765,256 | 400 train scenes |
| `guzzbr/draw-synthetic-v1-part3` | 1,200 | 224,131,842 | 400 train scenes |
| `guzzbr/draw-synthetic-v1-part4` | 1,200 | 224,969,894 | 400 train scenes |
| `guzzbr/draw-synthetic-v1-part5` | 1,200 | 223,240,028 | 400 train scenes |
| `guzzbr/draw-synthetic-v1-part6` | 1,200 | 225,831,624 | 400 train scenes |
| `guzzbr/draw-synthetic-v1-part7` | 900 | 168,844,840 | 300 validation scenes |
| `guzzbr/draw-synthetic-v1-part8` | 900 | 167,192,146 | 300 test scenes |
| **Total** | **9,001** | **1,685,442,658** | **2,400 train / 300 val / 300 test** |

Each scene has an image, a four-class pixel mask, and an `ink_full` auxiliary mask. All images decode at 512×512. Masks decode as single channel and use only `{0,1,2,3}`; there are no ignored pixels in this synthetic release. All four classes have nonzero support in every split. Manifest seed is 42. Background, hand, and paper source IDs have zero pairwise overlap across train/val/test. The manifest says no hand is a placeholder in these 3,000 scenes.

All eight ZIP downloads were compared against the complete remote file listings by file name and size, and every payload file was SHA-256 hashed. ZIP integrity passed; there were no missing, extra, or size-mismatched files.

- Full downloaded-file SHA-256 inventory digest: `ebcdccbf51e12418cfe82847d7224de29eb07b7f223f2a5c7c2c93dd2604f341`.
- Trainer's sorted merged-input fingerprint, including `split_manifest.json`: `4d98f653506439d6cb059be6c1e94883a6f6a394ff916b1c10300f68cbe16dfa`.
- `split_manifest.json` SHA-256: `9ba05ab08ad5bf276a96b527777424b8e143958a31d192e7862d458af105c931`.
- Audit JSON SHA-256: `c1939217444d8cc8c6097e415bd5cd14228c36d5a63c3a48c4695effcb9be4ef` (`.kaggle-staging/remote-audit-20261101/current/synthetic-data-audit.json`, ignored local artifact).

The only pretrained-weight input is `guzzbr/mit-b0-cache`, version 1, private, ready; all seven remote file names and sizes matched the downloaded archive. Archive SHA-256: `c3cd17f8b2a4417a69ae4adff69981f6e4a85e9caa25f461aca39152c2e47b25`. The weight blob SHA-256 is `4af8348c2a802bf76115d34797ff5ce1d9f110bb8593c22d9b66d8bd7fa227fc` (Hugging Face snapshot `80983a413c30d36a39c20203974ae7807835e2b4`).

No Kaggle datasets were created or updated. `guzzbr/draw-ink-synth-v1` was not used because it has a different document ink/print contract; `draw-realbg-v2` remains unopened; capture and pseudo datasets were not attached.

## Baseline and gates fixed before training

The frozen baseline is the local `synthetic_only` run, trained on 2,400 scenes and evaluated on the same manifest. The baseline manifest and remote manifest are equal after removing the run-only `training_sources` field. Baseline validation: `mIoU=0.9177034868`, ink IoU `0.7433429512`; ONNX validation passed with `max_abs_diff=5.72e-05` and per-class argmax IoU 1.0. Baseline metrics SHA-256: `b7135c498215f2ce1d70b66c8459c78392957347b8800eb4f397bda3433f605d`.

Selection uses validation `mean_iou` only. Acceptance requires validation `mIoU >= 0.9127034868`, ink IoU `>= 0.7333429512`, plus ONNX Runtime validation with `max_abs_diff <= 1e-3` and minimum per-class argmax IoU `>= 0.99`. The test split is evaluated once after checkpoint selection and is never used for training, early stopping, thresholds, or model selection. Per-class IoU, Dice, precision, recall and ink tolerance at 2/3 px are reported for val and test.

The existing Kaggle kernel `guzzbr/draw-segformer-v1-4class` is not the baseline: its output used 719 real pseudo-labels, reports validation only (`mIoU=0.9136732182`, ink IoU `0.7268709406`), has no test metrics, and its ONNX validation is `null`. It is exploratory and excluded from the new comparison.

## Kernel package

- Kernel: `guzzbr/draw-segformer-synthetic-only-20261101`, version 2, private, GPU enabled, internet enabled only so the notebook can install pinned ONNX validation packages if absent.
- Inputs: the eight private synthetic chunks and the private `mit-b0` cache only.
- Notebook for version 2: `notebooks/kaggle-segmentation-synthetic-20261101/train_segformer_synthetic_only.ipynb`, SHA-256 `151dbc99d6db578ef7db69bd84e3673719e3cc2a797014b2b639c87a6b3e935c`.
- Embedded current branch training code SHA-256: `0f08df81b926429f6ca614ef3d2afa6a2d1f81490c3657510216e0175cd961cb`; config SHA-256: `e1f049a8812c3679d68530523e6938dc925f1090d56c5fb1a943b897730054b5`.
- Version 1 was private and did pass the mounted-data fingerprint, split and label checks on a Tesla T4. It failed at the end of the first training epoch because `train.py` called `require_valid_evaluation` without defining it. No epoch checkpoint or model output was saved. That missing fail-closed validation helper was added, covered by the local smoke test, embedded in version 2, and the new private version was pushed. Version 2's first status is `RUNNING`.

## Version 2 result

- Final Kaggle status: `KernelWorkerStatus.COMPLETE`; runtime was `3,841.15 s` (about 64 min). Kernel metadata confirms `is_private=true`, GPU enabled, and Tesla T4 machine shape. Runtime reported `GPU Tesla T4`.
- Runtime packages: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1, NumPy 2.1.3, OpenCV 4.14.0.94, ONNX 1.22.0, ONNX Runtime 1.20.1.
- Full training completed all 30 epochs. Best checkpoint selected using validation only. Training source counts: 2,400 synthetic / 0 real; split counts: 2,400 train / 300 val / 300 test. The test split was evaluated once after checkpoint selection.
- The runtime recomputed the input inventory: 9,001 files / 1,685,442,658 bytes, fingerprint `4d98f653506439d6cb059be6c1e94883a6f6a394ff916b1c10300f68cbe16dfa`; it matches the pre-kernel audit. Split manifest SHA-256 remained `9ba05ab08ad5bf276a96b527777424b8e143958a31d192e7862d458af105c931`.

### Metrics

All metrics below are from procedural synthetic labels. They are not human-reviewed and do not measure camera performance.

| Split | Class | IoU | Dice | Precision | Recall |
|---|---|---:|---:|---:|---:|
| Val | background | 0.996586 | 0.998290 | 0.999802 | 0.996783 |
| Val | paper | 0.985865 | 0.992882 | 0.992508 | 0.993257 |
| Val | hand | 0.942276 | 0.970280 | 0.956242 | 0.984737 |
| Val | ink | 0.747902 | 0.855771 | 0.825282 | 0.888600 |
| Test | background | 0.998097 | 0.999047 | 0.999728 | 0.998368 |
| Test | paper | 0.987072 | 0.993494 | 0.995675 | 0.991322 |
| Test | hand | 0.951911 | 0.975363 | 0.965495 | 0.985436 |
| Test | ink | 0.741492 | 0.851559 | 0.790154 | 0.923312 |

Validation mIoU is `0.9181574974`, a delta of `+0.0004540106` against baseline `0.9177034868`. Validation ink IoU is `0.7479022100`, a delta of `+0.0045592588` against baseline `0.7433429512`. Both pre-fixed validation criteria passed. Synthetic test mIoU is `0.9196427878` and test ink IoU is `0.7414916808`. The baseline artifact has no stored test metrics, so only validation scores were compared between baseline and candidate; the test score above is candidate-only.

The trainer currently records ink tolerance for the first validation/test batch (`ink_tolerance_sample`), not a full-split aggregate. The recorded samples are: val radius 2 px P/R `0.973748/0.999511`, radius 3 px `0.973748/1.000000`; test radius 2 px `0.999876/0.998782`, radius 3 px `1.000000/0.999865`. Treat these as examples, not aggregate split metrics. A next-cycle improvement is to aggregate tolerance over every image.

### ONNX and artifacts

ONNX validation passed on the same validation batch: `status=ok`, `max_abs_diff=6.484985e-05` (limit `1e-3`), minimum argmax IoU `1.0` (limit `0.99`), argmax agreement `1.0`.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/best.pt` | 14,932,509 | `beb7d6cb8ed529f8584180dcc57684fc9fc72fa86ce0be8d733621061c17f048` |
| `checkpoints/last.pt` | 14,932,509 | `c934304761807b5f7992ed6189e9f881540a4eeef7f3dde507b0a25ef35e1c5d` |
| `draw_segmentation_v1.onnx` | 15,181,516 | `5baa56e272df146d67e63983f576508e06d4e1fb3fd97339f3f622bfd8a5da1c` |
| `classes.json` | 68 | `9a1211b27d7ec6af386aefc87def44453cebc86aba43ad045b4ded345f527c8c` |
| `metrics.json` | 45,066 | `b085259c99a35c5866aaa222e2b5fbb6e7b6995fcd4e8fac8c7ab4b3290f3c4a` |
| `training_config.json` | 1,710,863 | `babe75c054c1f6df8cd7e8c9a11a8b0c9ece0aeed7e0f54e008ca1c49f22fc40` |
| `split_manifest.json` | 1,608,553 | `f065bf351e1dbcbb0e178773e55b31ed4704a71b1d5b11aeadb2534810a6e3a7` |
| `artifact_hashes.json` | 936 | `e7807f34f1d12fff075f0b65e1d506b0c303cd9a819afdf6b422fe4b68318760` |

All seven output artifacts listed in `artifact_hashes.json` were independently re-hashed locally and matched the recorded sizes and SHA-256 values. The downloaded private kernel outputs are retained under ignored `.kaggle-staging/remote-audit-20261101/current/new-kernel-output-v2/`.

### Post-training ONNX inference smoke

An additional local CPU inference smoke used ONNX Runtime 1.20.1 on eight synthetic **training** images (no labels loaded, no metrics, no holdout). The model hash matched the artifact hash above. It accepted input `image` with shape `N×3×512×512` and produced finite `logits` with shape `8×4×128×128`; argmax returned all four class IDs across the batch. Batch inference took 2.50 s (about 0.313 s/frame at batch size 8 on this CPU; not a single-frame latency benchmark). The ephemeral environment and JSON result are under ignored `.kaggle-staging/model-inference-test-20261101*` paths.

This smoke verifies that the exported ONNX runs locally, not that the new segmentation model is a drop-in replacement in the current app. `draw/neural.py` still requests `mask_logits` plus `presence_logit` and interprets two channels for paper detection; this model exports one four-class `logits` output for ink segmentation. Integration needs an explicit adapter/product change and separate validation, and was not attempted here.

## Limitations and next cycle

This successful run is an exploratory synthetic pretraining result only. There are still no human-reviewed real pixel masks and no independent real-camera holdout. Do not present the result as real-world quality, mix it with temporal-model metrics, or activate it in the live camera. Next cycle: annotate and adjudicate pixel-level real masks, reserve a new unseen capture session as real holdout, aggregate tolerance metrics over full splits, and evaluate baseline and candidate once on the frozen synthetic test without using it for model selection.

The locally available `guzzbr/draw-training-code` version 3 was audited and found older than this branch's trainer. To avoid changing any Kaggle dataset version, the private notebook embeds and hashes the exact current branch `train.py` and `config.json`.

## Pending result fields

Final report fields are recorded above. Version 1 remains a failed code attempt; only version 2 passed the pre-fixed validation and ONNX gates.
