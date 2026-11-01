# Kaggle single-frame segmentation training — 2026-11-01

**State:** private kernel `guzzbr/draw-segformer-synthetic-only-20261101`, version 2, was accepted and reported `RUNNING` by Kaggle. Version 1 failed before completing an epoch and produced no model artifacts. No result is claimed until version 2 reaches a terminal state and its metrics, ONNX validation, and artifact hashes are checked.

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

The locally available `guzzbr/draw-training-code` version 3 was audited and found older than this branch's trainer. To avoid changing any Kaggle dataset version, the private notebook embeds and hashes the exact current branch `train.py` and `config.json`.

## Pending result fields

Update this report after version 2 reaches a terminal state with: final kernel status and run time; runtime GPU and package versions; actual input fingerprint; train/val/test counts; per-class val/test metrics; baseline deltas and gates; ONNX validation; SHA-256 for `best.pt`, `last.pt`, ONNX, metrics/config/manifest and output hash manifest. If the run fails or validation is skipped, report it as failed/unvalidated and do not describe it as a successful training result.
