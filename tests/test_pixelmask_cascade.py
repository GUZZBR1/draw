import numpy as np

from tools.evaluate_pixelmask_proposals import cascade_refine


def test_cascade_preserves_classical_labels_and_fills_only_confident_unknowns():
    classical = np.array([[0, 1, 2, 3, 255, 255, 255]], dtype=np.uint8)
    neural = np.array([[3, 3, 0, 0, 1, 1, 1]], dtype=np.uint8)
    confidence = np.array([[1, 1, 1, 1, 0.95, 0.95, 0.80]], dtype=np.float32)
    context = {
        "paper_region": np.array([[0, 1, 1, 1, 1, 1, 1]], dtype=bool),
        "paper_candidate": np.array([[0, 1, 1, 1, 1, 0, 1]], dtype=bool),
        "paper_core": np.array([[0, 1, 1, 1, 1, 0, 1]], dtype=bool),
        "likely_skin": np.array([[0, 0, 1, 0, 0, 0, 0]], dtype=bool),
        "paper_detected": True,
    }

    result = cascade_refine(classical, neural, confidence, context, {"user_added_ink"}, 0.90)

    # First-stage labels remain untouched, even when the model disagrees.
    assert result[0, :4].tolist() == classical[0, :4].tolist()
    # The confident paper proposal fills an unknown pixel inside the sheet.
    assert result[0, 4] == 1
    # Low confidence remains unknown.
    assert result[0, 5] == 255  # inside the polygon, but failed appearance check
    assert result[0, 6] == 255


def test_cascade_does_not_call_printed_text_ink_or_guess_without_sheet():
    classical = np.full((1, 3), 255, dtype=np.uint8)
    neural = np.array([[3, 0, 1]], dtype=np.uint8)
    confidence = np.ones((1, 3), dtype=np.float32)
    context = {
        "paper_region": np.zeros((1, 3), dtype=bool),
        "paper_candidate": np.zeros((1, 3), dtype=bool),
        "paper_core": np.zeros((1, 3), dtype=bool),
        "likely_skin": np.zeros((1, 3), dtype=bool),
        "paper_detected": False,
    }

    result = cascade_refine(
        classical, neural, confidence, context,
        {"printed_text", "user_added_ink"}, 0.90,
    )

    # No polygon was found; preserve the unknowns instead of hallucinating
    # paper/background. Printed text is never promoted to class 3.
    assert result.tolist() == [[255, 255, 255]]
