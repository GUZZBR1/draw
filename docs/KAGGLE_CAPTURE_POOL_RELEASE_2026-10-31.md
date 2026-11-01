# Kaggle capture pool — audit and release record

**State:** local preflight approved; Kaggle create was accepted for asynchronous private creation, but the exact slug is absent from `datasets list --mine`; remote publication remains unverified.
**Branch:** `training-zone`
**Candidate:** `guzzbr/draw-real-capture-pool-v1`
**Local release ID:** `20261031-v4-preflight`
**Privacy:** private only (`isPrivate: true`).
**Training status:** exploratory review/pseudo-label data; zero ground-truth masks and zero training-approved frames.

## Authorization and scope

The user confirmed that the partially visible person and captured document content are theirs and explicitly authorized private Kaggle upload. The candidate contains only original JPEGs from the user's `captures/training/session_*` folders and derived pseudo-masks for those same source frames. It contains no third-party document corpus, external-source file, model proposal mask, or temporal pair. Kaggle metadata declares `isPrivate: true`; the planned create command must omit `--public`.

The approved `kaggle datasets create` command was run without `-u/--public`; it used `-t` (`--keep-tabular`, not a privacy flag). Kaggle responded “Your private Dataset is being created.” A subsequent exact `kaggle datasets list --mine --format json` check did not contain `guzzbr/draw-real-capture-pool-v1`. The `status`, `files`, and `metadata` endpoints returned HTTP 403, and exact search also returned no result. Therefore completion, remote privacy, version, files, and hashes could not be independently verified. No second create/update was attempted. No Kaggle kernel or training job was started. No model or app configuration was changed. No Git push or merge was performed.

## Source and annotation inventory

The source crosswalk and canonical manifest reconcile 5,086 original JPEGs across eight sessions. The preflight checked each image hash, decode, native dimensions, session `frames.csv` hash and row, canonical manifest ID/hash/status, and duplicate source hashes. The eight session counts are recorded in the package manifest.

| Annotation status | Frames | Included pixel mask | Meaning |
|---|---:|---:|---|
| `pseudo_supervised` | 286 | 286 | Exploratory heuristic pseudo-label; thumbnail-level triage; fine pixels unverified |
| `pseudo_all_unknown` | 11 | 11 | All pixels are 255; contributes no supervised pixels |
| `unlabeled` | 4,789 | 0 | No mask; never infer an empty/background mask |
| **Total** | **5,086** | **297** | `training_approved=false` for every frame |

There are zero human ground-truth masks. The mask contract is `0=background`, `1=visible paper (including pre-existing print/text)`, `2=visible hand`, `3=user-added visible ink`, `255=uncertain/ignore`. Missing-mask count is 4,789; invalid-mask count is 0. Scene tags are explicitly `not_provided` for all frames and never act as pixel masks. Proposal queue/snapshot status is kept in separate fields; proposal masks are not included.

Each pseudo-mask is single-channel, image-sized, limited to `{0,1,2,3,255}`, and checked by SHA-256. The 297 masks are method/version tagged `mask proposal reviewed_v3 + thumbnail-level visual triage`; their fine boundaries remain unverified. No pseudo-label or agent consensus is described as ground truth.

## Lineage and split boundaries

- Every sample is assigned `review_only`; there are no train, validation, test, or holdout splits.
- The complete session `session_20261029_130013_453419` was used in prior pseudo-label training. It remains marked as prior-training data and must not be used as holdout.
- 297 image and mask pairs have exact, hash-confirmed overlap with `guzzbr/draw-real-seg-pilot-v1`. Each is cross-referenced by frame ID and hash; this preserves the complete review inventory and is not evidence of independent evaluation.
- Previous/next source-frame IDs and timestamp deltas, nearest source-session relationships, and per-frame derived-copy references are recorded. The derived-copy manifest has 90 records; none references the 5,086 frame IDs in this pool. Only original JPEGs are packaged.
- Single-frame review, temporal new-ink pairs, and future stroke-vectorization data remain separate tasks.

## Candidate files and hashes

Staging path: `.kaggle-staging/release-v4-preflight/` (local, ignored by Git; preserve for audit).

| File | Bytes | SHA-256 |
|---|---:|---|
| `images.zip` | 208,751,342 | `db05512b565c25c083180ef43bd98249149e9400c35cc4b59c9ce52d6c585ea5` |
| `pseudo_masks.zip` | 1,214,709 | `42c1cb6e894dfb9da8cf682a3977f67c98da3a8d7a867eb857cb84b10e5e3aa2` |
| `frames.csv` | 4,449,925 | `9e237829c4f2400294bae1020f9ad772775ecddacf9677d7dddc2074b7735c36` |
| `privacy_exclusions.csv` | 478 | `87747eaa658dc98098463a285e1596c6eda85c6863996bd3ab943579a7739db5` |
| `dataset_manifest.json` | 10,110 | `590d54d2946959f14d729045245e5619a64be14a6bf6a1fd9c95068c2fd8981e` |
| `dataset-metadata.json` | 359 | `756fc96456c0ec52591fcc1d57a2b95fc6817f43df8e8be3bb286b0332ea2fa4` |

Both ZIPs passed archive integrity checks and contain 5,086 JPEGs and 297 masks, respectively. The source manifest, crosswalk, derived-copy manifest, remote pilot sample manifest, and per-session CSV hashes are recorded in `dataset_manifest.json`.

## Pre-upload gates

- [x] Branch/worktree is `training-zone`; worktree was clean before this task. Main checkout and original captures were not modified.
- [x] Local source, IDs, hashes, dimensions, decode, session manifests, annotation status, duplicate source hashes, pseudo-mask pixels, and remote pilot overlaps checked.
- [x] Privacy ownership and private-use authorization explicitly confirmed by the user.
- [x] Package metadata explicitly says `isPrivate: true`; no external data files included.
- [x] No training approval, holdout, temporal labels, or model proposal masks.
- [x] Adversarial review of release v4: **PASS**; no external/third-party content found.
- [x] Independent verification of release v4: **PASS**; archive hashes, counts, pixel masks, and lineage reconfirmed.
- [x] Final architecture signoff after incorporating the PASS findings: **PASS / APPROVE**.
- [x] Manifest records all three review passes and upload eligibility.
- [x] Kaggle title shortened to satisfy the platform's 6–50 character limit; `isPrivate: true` retained.
- [x] Submit one private Kaggle create request after all reviews passed; no public flag was used.
- [x] Perform the final exact account-list check: slug absent. `status/files/metadata` returned 403.
- [ ] Remote existence, privacy, version, file sizes, and downloaded hashes remain **unverified**; stop without retrying create/update.
- [ ] After upload, verify remote privacy, version, file list/sizes, and downloaded archive hashes; record remote version here.

## Known limitations

The package is exploratory. Pseudo-masks have only thumbnail-level triage, fine ink boundaries are not pixel-verified, and 255 is ignored. The user-provided scene descriptions/tags are absent for these frames. The previous training session is not a holdout. This release does not support real-world quality claims or supervised ground-truth evaluation.
