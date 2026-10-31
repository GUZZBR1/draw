# Real-frame masks for training

## Current status

This branch does not contain a reviewed, dense real-frame mask package. Its
`255` ignore-index path is implemented in the trainer and metrics, but this
environment has no PyTorch installation, so the real PyTorch loss test is
skipped. Run that test and a small training-data preflight in a PyTorch
environment before using partial masks for training.

The two capture sessions must remain intact when assigning data splits. Record
one split role per complete session before expanding annotations. With only two
sessions, using one for real training and the other for real evaluation is a
limited near-domain comparison, not a broad independent validation and test
set.

## Pixel-label contract

Each mask is a single-channel PNG at the source image's original dimensions:

| Value | Meaning |
|---:|---|
| 0 | Visible background, including the pen and other non-hand objects |
| 1 | Visible paper, including printed text |
| 2 | Visible hand; this overrides ink when a hand covers a stroke |
| 3 | Clearly visible user-added ink only |
| 255 | Deliberately unknown pixel, excluded from loss and metrics |

Never map unknown pixels to background or paper. A mask with no supervised
pixels is not a usable example. Image-level tags such as `user_added_ink` only
guide mask proposals; they do not define pixel labels. SAM, OpenCV,
color-threshold, and interpolated masks remain drafts until a reviewer checks
each frame against its source image.

## Dataset acceptance

Keep experimental masks and generated packages under ignored
`captures/triage/`. For every candidate pair, record the session/frame ID,
source path and checksum, mask path and checksum, original dimensions, split,
annotation method/version, reviewer status, and per-class/unknown pixel counts.

Do not call a mask package training-ready until all of these checks pass:

1. Every selected ID has exactly one decodable source image and one mask.
2. Image and mask dimensions match; masks are single-channel and contain only
   `0, 1, 2, 3, 255`.
3. Every released mask has been reviewed at full resolution, with targeted
   review of ink, printed text, hand-over-ink, and uncertain boundaries.
4. No mask is entirely ignored. Report class support and unknown-pixel counts.
5. Tests for the PyTorch loss, evaluation metrics, and dataset preflight pass in
   the actual training environment before a GPU run.

Passing these data checks establishes usable training input; it does not prove
real-world model accuracy. Report real-session results separately from
synthetic results.
