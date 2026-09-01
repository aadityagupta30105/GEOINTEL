"""Tests for GDELT ingestion, schema alignment and preprocessing."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd
import pytest
import requests

from data.gdelt_collector import (
    EVENT_ROOT_MAP,
    GDELT_COLS,
    REQUIRED_COLUMNS,
    collect_gdelt_range,
    fetch_gdelt_day,
    generate_mock_data,
    get_gdelt_url,
    preprocess,
)


class TestSchema:
    """Guards on the GDELT 1.0 column contract."""

    def test_column_count_is_58(self) -> None:
        assert len(GDELT_COLS) == 58

    def test_column_names_are_unique(self) -> None:
        assert len(set(GDELT_COLS)) == len(GDELT_COLS)

    def test_actor_country_codes_sit_at_indices_7_and_17(self) -> None:
        """The historical misalignment bug mapped Actor2CountryCode to index 10."""
        assert GDELT_COLS.index("Actor1CountryCode") == 7
        assert GDELT_COLS.index("Actor2CountryCode") == 17
        assert GDELT_COLS[10] == "Actor1Religion1Code"

    def test_actor_blocks_are_ten_fields_each(self) -> None:
        assert GDELT_COLS[5:15] == [
            "Actor1Code", "Actor1Name", "Actor1CountryCode",
            "Actor1KnownGroupCode", "Actor1EthnicCode",
            "Actor1Religion1Code", "Actor1Religion2Code",
            "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
        ]

    def test_url_format(self) -> None:
        url = get_gdelt_url(datetime(2024, 3, 15))
        assert url.endswith("/20240315.export.CSV.zip")


def _build_gdelt_zip(rows: list[list[str]]) -> bytes:
    """Encode rows as a Latin-1 tab-separated GDELT export inside a ZIP.

    Parameters
    ----------
    rows : list of list of str
        Row values, each of length 58.

    Returns
    -------
    bytes
        In-memory ZIP archive.
    """
    body = "\n".join("\t".join(row) for row in rows)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("20240315.export.CSV", body.encode("latin-1"))
    return buffer.getvalue()


def _gdelt_row(actor1: str, actor2: str, name: str = "Beyonce") -> list[str]:
    """Build one 58-field GDELT row.

    Parameters
    ----------
    actor1, actor2 : str
        Country codes placed at their correct schema indices.
    name : str
        Actor name, used to carry a non-ASCII payload in encoding tests.

    Returns
    -------
    list of str
        A complete 58-field row.
    """
    row = [""] * 58
    row[0] = "1000001"
    row[1] = "20240315"
    row[5], row[6], row[7] = f"{actor1}GOV", name, actor1
    row[15], row[16], row[17] = f"{actor2}GOV", "Counterpart", actor2
    row[28] = "04"          # EventRootCode
    row[29] = "1"           # QuadClass
    row[30] = "2.5"         # GoldsteinScale
    row[31] = "12"          # NumMentions
    row[34] = "3.75"        # AvgTone
    row[57] = "https://example.invalid/article"
    return row


class _StubResponse:
    """Minimal stand-in for a streamed ``requests`` response."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int = 1) -> Any:
        yield self._payload


class TestFetch:
    """Behaviour of the daily fetch path under stubbed transport."""

    def test_parses_latin1_and_filters_bilateral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _build_gdelt_zip([
            _gdelt_row("USA", "CHN", name="Beyonc\xe9"),   # bilateral, non-ASCII
            _gdelt_row("RUS", "UKR"),                       # bilateral
            _gdelt_row("USA", "USA"),                       # self-loop, dropped
            _gdelt_row("USA", ""),                          # missing actor, dropped
        ])
        monkeypatch.setattr(
            requests, "get", lambda *args, **kwargs: _StubResponse(payload)
        )

        frame = fetch_gdelt_day(datetime(2024, 3, 15))

        assert frame is not None
        assert len(frame) == 2
        assert set(frame["Actor1CountryCode"]) == {"USA", "RUS"}
        # Correct schema alignment: actor codes are not blank religion fields.
        assert set(frame["Actor2CountryCode"]) == {"CHN", "UKR"}
        assert "Beyonc\xe9" in set(frame["Actor1Name"])

    def test_returns_none_when_no_bilateral_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _build_gdelt_zip([_gdelt_row("USA", "USA")])
        monkeypatch.setattr(
            requests, "get", lambda *args, **kwargs: _StubResponse(payload)
        )
        assert fetch_gdelt_day(datetime(2024, 3, 15)) is None

    def test_transport_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def explode(*args: Any, **kwargs: Any) -> None:
            raise requests.ConnectionError("network unreachable")

        monkeypatch.setattr(requests, "get", explode)
        assert fetch_gdelt_day(datetime(2024, 3, 15)) is None

    def test_range_falls_back_to_synthetic_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: Any, **kwargs: Any) -> None:
            raise requests.Timeout("timed out")

        monkeypatch.setattr(requests, "get", explode)
        frame = collect_gdelt_range(datetime(2024, 3, 15), datetime(2024, 3, 16))

        assert not frame.empty
        assert {"Actor1CountryCode", "Actor2CountryCode"} <= set(frame.columns)


class TestSyntheticGenerator:
    """Determinism and shape of the offline generator."""

    def test_seed_makes_output_reproducible(self) -> None:
        first = generate_mock_data(datetime(2024, 1, 1), datetime(2024, 1, 31), 200, seed=7)
        second = generate_mock_data(datetime(2024, 1, 1), datetime(2024, 1, 31), 200, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_actors_are_always_distinct_iso3_codes(self) -> None:
        frame = generate_mock_data(datetime(2024, 1, 1), datetime(2024, 1, 31), 300, seed=3)
        assert (frame["Actor1CountryCode"].str.len() == 3).all()
        assert (frame["Actor2CountryCode"].str.len() == 3).all()
        assert (frame["Actor1CountryCode"] != frame["Actor2CountryCode"]).all()

    def test_single_day_window_does_not_raise(self) -> None:
        frame = generate_mock_data(datetime(2024, 1, 1), datetime(2024, 1, 1), 10, seed=1)
        assert len(frame) == 10
        assert set(frame["date"]) == {"2024-01-01"}

    def test_tension_dyads_skew_negative(self) -> None:
        """Curated tension pairs must resolve to negative mean tone."""
        frame = preprocess(
            generate_mock_data(datetime(2024, 1, 1), datetime(2024, 6, 30), 40000, seed=5)
        )
        pair = frame[
            frame["Actor1CountryCode"].isin(["IND", "PAK"])
            & frame["Actor2CountryCode"].isin(["IND", "PAK"])
        ]
        assert len(pair) > 0
        assert pair["tone_norm"].mean() < 0

    def test_referenced_tension_codes_are_generatable(self) -> None:
        """Codes named in the tension table must exist in the country pool."""
        frame = generate_mock_data(datetime(2024, 1, 1), datetime(2024, 6, 30), 30000, seed=8)
        generated = set(frame["Actor1CountryCode"]) | set(frame["Actor2CountryCode"])
        assert {"ERI", "SSD", "XKX"} <= generated


class TestPreprocess:
    """Normalisation, filtering and derived-column behaviour."""

    def test_adds_derived_columns(self, events: pd.DataFrame) -> None:
        assert {"tone_norm", "event_label", "event_type"} <= set(events.columns)
        assert set(REQUIRED_COLUMNS) <= set(events.columns)

    def test_tone_is_normalised_to_unit_range(self, events: pd.DataFrame) -> None:
        assert events["tone_norm"].between(-1.0, 1.0).all()

    def test_drops_self_loops_and_invalid_codes(self) -> None:
        frame = pd.DataFrame({
            "Actor1CountryCode": ["USA", "USA", "US", "CHN", None],
            "Actor2CountryCode": ["CHN", "USA", "CHN", "RU", "CHN"],
            "AvgTone": [1.0, 1.0, 1.0, 1.0, 1.0],
            "QuadClass": [1, 1, 1, 1, 1],
            "EventRootCode": ["04", "04", "04", "04", "04"],
            "date": ["2024-01-01"] * 5,
            "NumMentions": [1] * 5,
            "GoldsteinScale": [0.0] * 5,
        })
        result = preprocess(frame)
        assert len(result) == 1
        assert result.iloc[0]["Actor1CountryCode"] == "USA"

    def test_normalises_variant_column_names(self) -> None:
        frame = pd.DataFrame({
            "actor1_country_code": ["USA"],
            "ACTOR2COUNTRYCODE": ["CHN"],
            "AvgTone": [2.0],
            "QuadClass": [1],
            "EventRootCode": ["4"],
            "date": ["2024-01-01"],
            "NumMentions": [3],
            "GoldsteinScale": [1.0],
        })
        result = preprocess(frame)
        assert list(result["Actor1CountryCode"]) == ["USA"]
        assert list(result["Actor2CountryCode"]) == ["CHN"]

    def test_zero_pads_single_digit_event_codes(self) -> None:
        frame = pd.DataFrame({
            "Actor1CountryCode": ["USA"], "Actor2CountryCode": ["CHN"],
            "AvgTone": [1.0], "QuadClass": [1], "EventRootCode": ["4"],
            "date": ["2024-01-01"], "NumMentions": [1], "GoldsteinScale": [0.0],
        })
        result = preprocess(frame)
        assert result.iloc[0]["EventRootCode"] == "04"
        assert result.iloc[0]["event_label"] == EVENT_ROOT_MAP["04"]

    @pytest.mark.parametrize(
        ("root_code", "quad_class", "expected"),
        [
            ("06", 1, "Trade/Aid"),
            ("07", 4, "Trade/Aid"),          # aid code overrides quad class
            ("19", 1, "Military/Conflict"),  # violence code overrides quad class
            ("04", 1, "Cooperation"),
            ("04", 2, "Cooperation"),
            ("11", 3, "Conflict"),
            ("11", 4, "Conflict"),
            ("04", 9, "Diplomatic"),         # unknown quad class
        ],
    )
    def test_event_type_precedence(
        self, root_code: str, quad_class: int, expected: str
    ) -> None:
        frame = pd.DataFrame({
            "Actor1CountryCode": ["USA"], "Actor2CountryCode": ["CHN"],
            "AvgTone": [0.0], "QuadClass": [quad_class], "EventRootCode": [root_code],
            "date": ["2024-01-01"], "NumMentions": [1], "GoldsteinScale": [0.0],
        })
        assert preprocess(frame).iloc[0]["event_type"] == expected

    def test_empty_input_returns_empty_frame(self) -> None:
        result = preprocess(pd.DataFrame({
            "Actor1CountryCode": [], "Actor2CountryCode": [],
        }))
        assert result.empty

    def test_all_rows_filtered_returns_empty_frame(self) -> None:
        frame = pd.DataFrame({
            "Actor1CountryCode": ["XX"], "Actor2CountryCode": ["YY"],
            "AvgTone": [1.0], "QuadClass": [1], "EventRootCode": ["04"],
            "date": ["2024-01-01"], "NumMentions": [1], "GoldsteinScale": [0.0],
        })
        assert preprocess(frame).empty

    def test_non_numeric_values_coerce_without_raising(self) -> None:
        frame = pd.DataFrame({
            "Actor1CountryCode": ["USA"], "Actor2CountryCode": ["CHN"],
            "AvgTone": ["not-a-number"], "QuadClass": ["x"], "EventRootCode": ["04"],
            "date": ["2024-01-01"], "NumMentions": ["y"], "GoldsteinScale": ["z"],
        })
        result = preprocess(frame)
        assert result.iloc[0]["AvgTone"] == 0.0
        assert result.iloc[0]["QuadClass"] == 1
