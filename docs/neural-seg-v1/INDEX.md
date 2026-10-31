# Neural model docs index

This folder contains two separate tracks and several dated execution records.
For current application behavior, check the root [README](../../README.md) and
[`models/README.md`](../../models/README.md) first.

## Model contracts

- [`INTEGRATION.md`](INTEGRATION.md): single-frame SegFormer output and the
  page/hand/ink class contract.
- [`SENADO_EXECUCAO_2026-10-30.md`](SENADO_EXECUCAO_2026-10-30.md): review of
  the single-frame pilot and pairwise ink v3 artifacts.
- [`SENADO_VALIDACAO_KAGGLE_2026-10-30.md`](SENADO_VALIDACAO_KAGGLE_2026-10-30.md):
  follow-up validation record and current limits.

The pairwise model is a distinct temporal task. It is not the page detector;
the app exposes it through an opt-in experimental adapter. CPU smoke inference
passes; checkpoint parity and webcam validation remain pending.

## Training and data

- [`TRAINING_READY.md`](TRAINING_READY.md): training readiness and unresolved
  gates.
- [`KAGGLE_RUNBOOK.md`](KAGGLE_RUNBOOK.md): Kaggle workflow.
- [`KAGGLE_ENV.md`](KAGGLE_ENV.md): runtime and dependency findings.
- [`DATASET_FREIHAND.md`](DATASET_FREIHAND.md) and
  [`HAND_SOURCES_ALT.md`](HAND_SOURCES_ALT.md): source and license research.

## Capture and review

- [`REAL_FRAMES_PLAN.md`](REAL_FRAMES_PLAN.md): plan for an independent real
  capture set.
- [`REAL_CAPTURE_PUBLISHING.md`](REAL_CAPTURE_PUBLISHING.md): capture
  publication rules.
- [`REAL_MASK_ANNOTATION.md`](REAL_MASK_ANNOTATION.md): mask review contract.
- [`FRAME_TRIAGE.md`](FRAME_TRIAGE.md) and [`REVIEW_QA.md`](REVIEW_QA.md):
  selection and quality checks.

## Dated execution notes

- [`RESTANTES_KAGGLE_2026-10-30.md`](RESTANTES_KAGGLE_2026-10-30.md) records
  remaining Kaggle work at that point.
- The two `SENADO_*.md` files are dated audit/validation records, not a live
  status feed. Read their dates and evidence limits before reusing conclusions.
