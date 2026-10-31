# Pixel-mask micro-engine

`tools/evaluate_pixelmask_proposals.py` runs the same source image through:

- the existing OpenCV proposal for background, paper, hand, and ink;
- an optional four-class ONNX model (`--segformer-model`);
- the optional ink ONNX specialist already shipped in this branch.

It writes one PNG mask and one overlay per method and frame, a side-by-side
review sheet, per-frame metrics, and a JSON summary. The `agreement` mask keeps
only pixels where OpenCV and the four-class model predict the same class. It
marks disagreements as `255` (unknown), so it can reduce the area that needs
review. Agreement is still only a proposal; correlated model errors can agree.

## Run

```bash
python tools/evaluate_pixelmask_proposals.py \
  --pack /path/to/reviewed-pilot \
  --segformer-model /path/to/four-class-model.onnx \
  --output-dir captures/triage/pixelmask-pilot
```

The package must contain `samples.csv`, plus the image and mask files named by
that manifest. If the four-class model is unavailable, pass `--no-segformer`;
the OpenCV and ink-specialist proposals still run. `--reuse-predictions` is an
explicit speed option for a previously completed run in the same output folder.
Use a fresh output folder after changing source images or model files.

## Reading the report

The pilot reference masks have three-agent visual review, but the stored pixels
are byte-for-byte copies of the earlier OpenCV proposals. They are not
independently drawn pixel ground truth, and there is no approved real holdout.
Therefore IoU and Dice here measure agreement with masks derived from that
earlier proposal. They do not establish real-world accuracy or prove that one
method is better. The current OpenCV implementation also differs from the
earlier one, so even its own score is not a self-comparison.
Reference pixels labeled `255` are excluded. Proposal pixels labeled `255` are
also excluded from class scores, and their effect is shown as `mean_coverage`.
Always inspect the overlays and unknown regions before accepting a mask.

The first local 19-frame run showed that the four-class SegFormer was less
aligned with those masks than the current OpenCV proposal, especially for hand
and ink. Adding the separate ink specialist did not improve agreement in this
pilot. The agreement mask improved measured agreement while reducing coverage
slightly, but this can reflect shared errors in the weak reference and
proposal methods. These results do not justify promoting any generated mask to
training data without per-frame review and a separately labeled evaluation
set.

No raw captures, generated masks, model files, or pilot reports belong in Git.
Keep them in the private local/Kaggle data store; commit only the tool and this
method note.
