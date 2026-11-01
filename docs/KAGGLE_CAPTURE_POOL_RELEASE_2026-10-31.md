# Kaggle capture pool — audit and release record

**State:** Kaggle release verified. `guzzbr/draw-real-capture-pool-v1` exists as dataset ID `13001954`, version `1`, status `ready`, and `isPrivate: true`. The complete remote file set (5,387 files; 213,839,598 bytes) matches the local release by name, size, and SHA-256.
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
- [x] Initial exact account-list check recorded the slug as absent; after refreshing OAuth, `kaggle datasets list --mine --sort-by updated` shows it.
- [x] Remote existence, privacy (`isPrivate: true`), version (`1`), status (`ready`), file list, sizes, and downloaded SHA-256 values verified.
- [x] After-upload verification completed against all 5,387 remote files; detailed per-file hashes are in `.kaggle-staging/remote-audit-20261101/verified-file-hashes.json`.

## Known limitations

The package is exploratory. Pseudo-masks have only thumbnail-level triage, fine ink boundaries are not pixel-verified, and 255 is ignored. The user-provided scene descriptions/tags are absent for these frames. The previous training session is not a holdout. This release does not support real-world quality claims or supervised ground-truth evaluation.

## Initial blocked audit — 2026-10-31 (superseded)

**Historical result:** at this stage the earlier asynchronous create response could not be resolved. The final verification below supersedes the blocked state. No duplicate create was attempted.

Fresh Kaggle CLI 2.2.4 queries from `training-zone` returned:

| Query | Result |
|---|---|
| `kaggle datasets list --mine --format json` | `No datasets found`; does not establish absence because the status endpoint reports missing authentication in this environment. |
| `kaggle datasets list -s 'draw real capture pool v1' --format json` | `No datasets found`. |
| `kaggle datasets list --user guzzbr --format json` | Only the public `guzzbr/drawink-corpus` appeared; private datasets are not confirmed by this listing. |
| `kaggle datasets status guzzbr/draw-real-capture-pool-v1` | Authentication required. |
| `kaggle datasets files guzzbr/draw-real-capture-pool-v1` | HTTP 403 Forbidden. |
| `kaggle datasets metadata guzzbr/draw-real-capture-pool-v1` | HTTP 403 Forbidden. |

Consequently, available queries do not verify that the candidate is absent, nor do they verify remote privacy, version, files, sizes, or downloaded hashes. Per the no-duplicate rule, do not retry `create` or `update` until the exact candidate can be queried successfully. Do not report the private release as published. The same access block prevents safe review/update of other private Kaggle datasets from the plan.

Local read-only checks reconfirmed the six staged package-file SHA-256 values and both ZIP integrity checks. The staged `dataset-metadata.json` still declares `isPrivate: true`; this is local intent only, not evidence of remote privacy. Contents remain 5,086 original JPEGs, 297 separate exploratory pseudo-masks, and 4,789 unlabeled frames. Scene labels are not pixel masks, there are no human ground-truth masks, and all samples remain review-only. No kernel or training job was started.

This audit was recorded on local `training-zone` HEAD `670031da361cee01abbc62f183347984ba77b98a` before the follow-up commit. The worktree was clean before this documentation change; it was already two commits ahead of `origin/training-zone`. No push was performed.

## Additional recheck — 2026-10-31 22:09 BRT (before credential-cache inspection)

This historical check found no `KAGGLE_API_TOKEN`, `~/.kaggle/access_token`, or `~/.kaggle/kaggle.json`; it did not inspect the modern `~/.kaggle/credentials.json` file. `kaggle auth login` reported an existing CLI-managed session for `guzzbr`, but the cached access token was stale. At that time, list/search had no result, `status` requested authentication, and files/metadata returned 403. An unauthenticated browser returned 404, which did not establish absence of a private dataset. The final resolution below supersedes these observations.

The plan's metadata query was also attempted for all listed datasets. `guzzbr/drawink-corpus` metadata and its first page of files were readable; metadata requests for the other listed slugs returned HTTP 403. File-list requests for `guzzbr/draw-real-seg-pilot-v1` and `guzzbr/draw-realbg-v2` also returned HTTP 403. The public corpus is unchanged and is not eligible for a re-upload; the capture pool has a distinct review-only contract and must remain private. No Kaggle dataset version was confirmed, and no create/update was attempted.

The local release hashes, byte sizes, ZIP integrity, and `isPrivate: true` metadata were rechecked again. The main checkout remains on `main` with its existing local changes; this recheck changed only this report on `training-zone`. The remaining blocker is inconsistent Kaggle API access: some catalog reads work, while the candidate and other planned datasets cannot be listed with usable metadata/files, and status reports authentication-required despite the CLI session. This still prevents verifying candidate existence/privacy/files or safely updating another dataset.

## CLI-auth follow-up — 2026-10-31

`kaggle auth login` reports an existing logged-in account `guzzbr`, and `kaggle auth print-access-token` succeeds (the token was captured only inside a subprocess and was not emitted or saved). This supersedes any interpretation that no Kaggle session exists: the session is available to the CLI, but dataset API access remains inconsistent. Re-running the candidate's `list --mine`, exact search, `status`, `files`, and `metadata` produced the same results: no list/search match, authentication-required status, and HTTP 403 for files/metadata. The public `drawink-corpus` metadata and first file-list page remain readable.

An OAuth refresh with `kaggle auth login --force --no-launch-browser` could not be completed non-interactively; the CLI requests a verification code and exits with `EOFError` when no interactive input is available. No auth code or token was recorded. Without a successful exact account listing and private metadata/file access, absence cannot be safely distinguished from a hidden private object, and no create/update is permitted by the release gate.

## CLI 2.2.5 recheck — 2026-10-31

To rule out the installed CLI patch-level warning, the same candidate queries were run with an isolated `pipx run --spec kaggle==2.2.5` invocation. `datasets list --mine` and exact search again returned no results; `status` again required authentication; `files` and `metadata` returned HTTP 403 with “You don't have access to this resource.” The CLI upgrade did not change the result. The OAuth access token and any verification code were not written to the repository or this report. Candidate absence, privacy, and remote file contents remain unverified; do not create a duplicate or update any dataset until Kaggle account/API access is corrected.

## Final resolution and remote verification — 2026-11-01

This section supersedes the earlier blocked findings.

### Cause and fix

The working OAuth credentials were in `~/.kaggle/credentials.json` (mode `0600`), owned by `guzzbr` with scope `resources.admin:*`. Its cached access token expired at `2026-11-01T01:00:50Z`. At the time of the first failing requests it was about 16 minutes past expiration. The installed Kaggle CLI's `KaggleCredentials.access_token_has_expired()` keeps using a token until its expiry is more than 30 minutes old, so the CLI continued sending a stale token and private dataset calls failed.

I refreshed the cached access token through the CLI's installed `KaggleCredentials.refresh_access_token()` routine. The cache now expires at `2026-11-01T13:22:13Z`. The token value was not saved in the repository or report. Normal CLI queries now work, and `kaggle datasets list --mine --sort-by updated` includes the capture pool. Exact text search remains unreliable, but the direct status, metadata, and file endpoints confirm the dataset.

### Capture pool

| Check | Verified result |
|---|---|
| Slug / dataset ID | `guzzbr/draw-real-capture-pool-v1` / `13001954` |
| Current version / status | `1` / `ready` |
| Privacy / owner | `isPrivate: true` / `guzzbr` |
| Remote file count / total file bytes | `5,387` / `213,839,598` |
| File types | 5,086 JPEGs, 297 PNG pseudo-masks, 2 CSVs, 1 JSON, 1 Markdown file |
| Local comparison | All 5,387 names, sizes, and SHA-256 hashes match the staged release |

The full remote dataset was downloaded and compared against `images.zip`, `pseudo_masks.zip`, `README.md`, `dataset_manifest.json`, `frames.csv`, and `privacy_exclusions.csv` in `.kaggle-staging/release-v4-preflight/`. There were zero missing files, extra files, size mismatches, or SHA-256 mismatches. The per-file hash inventory is `.kaggle-staging/remote-audit-20261101/verified-file-hashes.json` (SHA-256 `ed2fb83d2d2ae5c65e8f20ca830610d251205dfca0235b00e9e6956df9dd2709`); the paginated file-list inventory is `files.json` (SHA-256 `eece4344526d5b1cc7eeded58f88dcb4fc2b48076a31a97d3c9da192aed48a6c`); the summary audit is `dataset-audit.json` (SHA-256 `cbded7e6757f2d69e017c21f05c9badf777fdc8508760936f974f2a61aa959ed`). These local ignored audit artifacts contain no credentials.

The remote files preserve the intended contract: 5,086 original JPEGs; 297 separate exploratory pseudo-masks; 4,789 frames without masks; no human ground truth; no train/validation/test or holdout split. Scene descriptions remain metadata and are not pixel masks.

### Other planned datasets

Authenticated metadata and status queries confirmed these versions and privacy states. No other update was eligible: the review pool has a distinct contract; 297 of its image/mask pairs overlap the pilot, so the sets must not be merged automatically; the background holdout remains unchanged; and no new synthetic data or reviewed temporal labels were available.

| Slug | Current version | Privacy | Action |
|---|---:|---|---|
| `guzzbr/draw-real-seg-pilot-v1` | 1 | Private | Unchanged; exploratory pseudo-label contract |
| `guzzbr/draw-ink-real-v1` | 1 | Private | Unchanged; no new compatible package |
| `guzzbr/draw-realbg-v2` | 1 | Private | Unchanged; holdout preserved |
| `guzzbr/draw-ink-synth-v1` | 2 | Private | Unchanged; no material new synthetic data |
| `guzzbr/draw-ink-pairs-v1` | 1 | Private | Unchanged; no reviewed temporal delta labels |
| `guzzbr/draw-segformer-expert-pilot-v1` | 1 | Private | Unchanged; no new adjudicated samples |
| `guzzbr/draw-drawfmt-v1` | 1 | Private | Unchanged; source licensing constraints preserved |
| `guzzbr/drawink-corpus` | 1 | Public | Unchanged; no new redistributable corpus content |
| `guzzbr/drawink-corpus-full` | 1 | Private | Unchanged; no redistribution |
| `guzzbr/drawink-corpus-pilot` | 1 | Private | Unchanged; no compatible new package |

The capture-pool release was already created and verified as version 1, so no duplicate create or content-identical update was submitted. No kernel or training job was started. No push was performed.
