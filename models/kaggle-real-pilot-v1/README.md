# Kaggle real-data pilot model

SegFormer-B0, four-class ONNX exported by Kaggle kernel
`guzzbr/draw-segformer-v1-4class` on 2026-10-30.

## Labels

| ID | Class |
|---:|---|
| 0 | background |
| 1 | visible paper |
| 2 | visible hand |
| 3 | user-added visible ink |

The model was trained with 2,400 synthetic images and 719 real pseudo-labeled
frames. The real masks were reviewed as contact-sheet proposals and had clear
mistakes corrected, but were not verified pixel by pixel. Treat this as an
experimental pilot, not a validated production model.

## Reported validation

Kaggle reported mean IoU `0.913673` on the synthetic validation split. Per-class
IoU was background `0.9981`, paper `0.9858`, hand `0.9440`, ink `0.7269`.
These are synthetic validation results, not a measurement on an independent
real-camera holdout. ONNX-vs-PyTorch parity was not run in Kaggle
(`onnx_validation: null`).

## Artifact integrity

`model.onnx` SHA-256:

```text
2ca5ea38210f8f1c7b4a6e981de89c0c434437341a18b7317c6733daba11c908
```

The ONNX input is `image`, `[N,3,512,512]`, ImageNet-normalized float32. It has
one output, `logits`, with four class channels at one-quarter input resolution.
The app adapter is responsible for deriving the page mask from paper and ink
classes while excluding hand pixels.
