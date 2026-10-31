# Pairwise ink model v3

This is the experimental `draw_ink_pairs_v3.onnx` exported by the Kaggle
training run. Its matching checkpoint and raw training log are stored beside
it locally. The original recovery bundle remains under
`captures/triage/kaggle_pair_v4_recovery_20261030/v3_existing_output/`.

## Contract

- Input: `pair`, three float channels `[after_gray, before_gray, signed_delta]`
  built from aligned grayscale page frames.
- Output: `logits`, three classes: rest, old content, and newly drawn ink.
- It detects changes between two frames. It is not the single-frame page
  detector consumed by `draw/neural.py`.

## Status

The app exposes an opt-in pairwise adapter and combines its new-ink mask with
the existing heuristic. The Kaggle run ended before its ONNX Runtime check
because `onnxruntime` was not installed. A local CPU smoke inference later
passed at 512×512 and at the app's 850×1200 sheet size; the adapter pads
dimensions to a multiple of 8 and crops the output back. Checkpoint parity and
webcam quality remain unvalidated. The sheet-size inference took about 1.3 s
on this environment, so this model is not yet suitable for fluid live drawing.
Do not set `DRAW_NEURAL_MODEL` to this artifact; use `--pair-ink-model` or
`DRAW_PAIR_INK_MODEL` only for an experimental trial.

The binaries/log are ignored by Git; this README records the local artifact
contract and provenance. SHA-256 of the ONNX export:

```text
1ccc83f22aa9f0c3d68b58b47578fba381c0d3923e7bb9ce1b4ab88cbb9353fe
```
