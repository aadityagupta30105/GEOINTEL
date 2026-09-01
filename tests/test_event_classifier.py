"""Tests for the event classifier, its fallback path and corpus construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from models.event_classifier import (
    LABEL_NAMES,
    NUM_LABELS,
    GeopoliticalEventClassifier,
    TrainingConfig,
    build_training_dataset_from_gdelt,
    create_synthetic_training_data,
)


class TestLabels:
    """Label space contract."""

    def test_five_labels_in_documented_order(self) -> None:
        assert NUM_LABELS == 5
        assert LABEL_NAMES == (
            "Military/Conflict",
            "Trade/Economic",
            "Diplomatic",
            "Humanitarian/Aid",
            "Political",
        )


class TestImportContract:
    """The module must import without initialising the deep-learning stack."""

    def test_import_does_not_touch_torch(self, project_root: Path) -> None:
        script = (
            "import sys; "
            "import models.event_classifier as m; "
            "print('torch' in sys.modules)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip().endswith("False")


class TestSyntheticCorpus:
    """Shape and determinism of the bootstrap corpus."""

    def test_shape_and_columns(self) -> None:
        frame = create_synthetic_training_data(100, seed=1)
        assert len(frame) == 100
        assert list(frame.columns) == ["text", "label"]

    def test_labels_are_within_the_label_space(self) -> None:
        frame = create_synthetic_training_data(200, seed=2)
        assert frame["label"].between(0, NUM_LABELS - 1).all()

    def test_seed_makes_output_reproducible(self) -> None:
        pd.testing.assert_frame_equal(
            create_synthetic_training_data(50, seed=9),
            create_synthetic_training_data(50, seed=9),
        )

    def test_all_classes_are_represented(self) -> None:
        frame = create_synthetic_training_data(500, seed=4)
        assert set(frame["label"].unique()) == set(range(NUM_LABELS))


class TestRuleBasedFallback:
    """Keyword classification used when no checkpoint is available."""

    @pytest.fixture
    def classifier(self, tmp_path: Path) -> GeopoliticalEventClassifier:
        """A classifier pointed at an empty checkpoint directory."""
        return GeopoliticalEventClassifier(
            TrainingConfig(save_path=str(tmp_path / "absent"))
        )

    def test_predictions_use_the_fallback(
        self, classifier: GeopoliticalEventClassifier
    ) -> None:
        predictions = classifier.predict(["USA and China signed a trade agreement"])
        assert predictions[0]["method"] == "rule_based"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Airstrikes and missile attacks caused casualties", "Military/Conflict"),
            ("Tariff and export market commerce talks", "Trade/Economic"),
            ("The ambassador and foreign minister will consult", "Diplomatic"),
            ("Humanitarian relief for refugees and medical aid", "Humanitarian/Aid"),
            ("Election vote and opposition government protest", "Political"),
        ],
    )
    def test_keyword_evidence_selects_the_right_class(
        self, classifier: GeopoliticalEventClassifier, text: str, expected: str
    ) -> None:
        assert classifier.predict([text])[0]["label_name"] == expected

    def test_unmatched_text_defaults_to_political_not_military(
        self, classifier: GeopoliticalEventClassifier
    ) -> None:
        """Regression: an all-zero keyword score previously selected label 0."""
        prediction = classifier.predict(["zzz qqq"])[0]
        assert prediction["label_name"] == "Political"
        assert prediction["confidence"] == 0.0

    def test_scores_cover_every_label(
        self, classifier: GeopoliticalEventClassifier
    ) -> None:
        prediction = classifier.predict(["a trade agreement"])[0]
        assert set(prediction["all_scores"]) == set(LABEL_NAMES)

    def test_confidence_is_bounded(
        self, classifier: GeopoliticalEventClassifier
    ) -> None:
        for prediction in classifier.predict(["war", "aid", "trade", ""]):
            assert 0.0 <= prediction["confidence"] <= 1.0

    def test_empty_input_returns_empty_output(
        self, classifier: GeopoliticalEventClassifier
    ) -> None:
        assert classifier.predict([]) == []


class TestPredictDataframe:
    """Frame-level inference plumbing."""

    @pytest.fixture
    def classifier(self, tmp_path: Path) -> GeopoliticalEventClassifier:
        return GeopoliticalEventClassifier(
            TrainingConfig(save_path=str(tmp_path / "absent"))
        )

    def test_appends_prediction_columns(
        self, classifier: GeopoliticalEventClassifier, events: pd.DataFrame
    ) -> None:
        sample = events.head(50)
        result = classifier.predict_dataframe(sample)
        assert {"ml_event_type", "ml_confidence", "ml_method"} <= set(result.columns)
        assert len(result) == len(sample)

    def test_does_not_mutate_the_input(
        self, classifier: GeopoliticalEventClassifier, events: pd.DataFrame
    ) -> None:
        sample = events.head(10)
        before = list(sample.columns)
        classifier.predict_dataframe(sample)
        assert list(sample.columns) == before

    def test_missing_column_falls_back_to_synthesised_text(
        self, classifier: GeopoliticalEventClassifier, events: pd.DataFrame
    ) -> None:
        result = classifier.predict_dataframe(events.head(5), text_col="absent_column")
        assert result["ml_event_type"].notna().all()

    def test_empty_frame_returns_typed_columns(
        self, classifier: GeopoliticalEventClassifier, events: pd.DataFrame
    ) -> None:
        result = classifier.predict_dataframe(events.iloc[0:0])
        assert result.empty
        assert "ml_event_type" in result.columns


class TestGdeltCorpus:
    """Distant supervision from CAMEO root codes."""

    def test_columns_and_length(self, events: pd.DataFrame) -> None:
        corpus = build_training_dataset_from_gdelt(events)
        assert list(corpus.columns) == ["text", "label"]
        assert len(corpus) == len(events)

    def test_labels_are_within_the_label_space(self, events: pd.DataFrame) -> None:
        corpus = build_training_dataset_from_gdelt(events)
        assert corpus["label"].between(0, NUM_LABELS - 1).all()

    @pytest.mark.parametrize(
        ("root_code", "expected_label"),
        [("19", 0), ("06", 1), ("04", 2), ("07", 3), ("01", 4)],
    )
    def test_cameo_mapping(self, root_code: str, expected_label: int) -> None:
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "EventRootCode": root_code, "event_label": "Example",
        }])
        assert build_training_dataset_from_gdelt(frame).iloc[0]["label"] == expected_label

    def test_unknown_code_defaults_to_political(self) -> None:
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "EventRootCode": "99", "event_label": "Example",
        }])
        assert build_training_dataset_from_gdelt(frame).iloc[0]["label"] == 4

    def test_text_contains_both_actors(self) -> None:
        frame = pd.DataFrame([{
            "Actor1CountryCode": "USA", "Actor2CountryCode": "CHN",
            "EventRootCode": "04", "event_label": "Consult",
        }])
        text = build_training_dataset_from_gdelt(frame).iloc[0]["text"]
        assert "USA" in text and "CHN" in text and "Consult" in text

    def test_empty_frame_returns_typed_empty_corpus(self) -> None:
        corpus = build_training_dataset_from_gdelt(pd.DataFrame())
        assert corpus.empty
        assert list(corpus.columns) == ["text", "label"]
