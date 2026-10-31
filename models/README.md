# Model artifacts

The Draw has two model contracts. Keep artifacts in a task-named directory;
do not choose a model by scanning for an arbitrary `.onnx` file.

- **Page detection:** a single-frame ONNX model loaded by `--neural` or
  `DRAW_NEURAL_MODEL`. It must follow `draw/neural.py`'s input/output contract.
- **Pairwise ink detection:** an experimental before/after model. It is
  available through the app's opt-in `--pair-ink-model` /
  `DRAW_PAIR_INK_MODEL` path.
  It cannot be passed as `DRAW_NEURAL_MODEL`. CPU smoke inference passes at
  512×512 and at the 850×1200 sheet size; checkpoint parity and webcam quality
  remain unvalidated.

See [`experimental/ink-pairs-v3/`](experimental/ink-pairs-v3/README.md) for
the model currently produced by the ink-pair training run.
