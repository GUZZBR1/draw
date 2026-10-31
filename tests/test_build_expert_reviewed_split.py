"""Regression tests for session-isolated expert-review splits."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.build_expert_reviewed_split import make_freeze


TRAIN_640 = "session_20261029_130013_453419"
TRAIN_720 = "session_20261029_151126_089557"
NEW_HOLDOUT = "session_20261031_100000_000001"


def _sample_rows():
    samples = {}
    for index in range(15):
        samples[(TRAIN_640, index)] = {"visible_tags_hint_only": ""}
        samples[(TRAIN_720, index)] = {"visible_tags_hint_only": ""}
    samples[(NEW_HOLDOUT, 0)] = {"visible_tags_hint_only": ""}
    return samples


def test_freeze_requires_an_unseen_holdout_session():
    samples = _sample_rows()
    with pytest.raises(ValueError, match="already been used for training"):
        make_freeze(
            samples,
            "samples-hash",
            {TRAIN_720},
            TRAIN_640,
        )

    samples[NEW_HOLDOUT, 1] = {"visible_tags_hint_only": ""}
    with pytest.raises(ValueError, match="already been used for training"):
        make_freeze(
            samples,
            "samples-hash",
            {TRAIN_640, TRAIN_720},
            NEW_HOLDOUT,
            {NEW_HOLDOUT},
        )


def test_freeze_keeps_new_session_wholly_out_of_training():
    samples = _sample_rows()
    freeze = make_freeze(
        samples,
        "samples-hash",
        {TRAIN_640, TRAIN_720},
        NEW_HOLDOUT,
    )

    assert freeze["train_session_ids"] == sorted([TRAIN_640, TRAIN_720])
    assert freeze["holdout_session_id"] == NEW_HOLDOUT
    assert {row["session_id"] for row in freeze["train_source_ids"]} == {
        TRAIN_640,
        TRAIN_720,
    }
    assert {row["session_id"] for row in freeze["holdout_source_ids"]} == {
        NEW_HOLDOUT
    }
    assert all(
        row["session_id"] != NEW_HOLDOUT
        for row in freeze["calibration_source_ids"]
    )
