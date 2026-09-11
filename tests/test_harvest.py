"""Contract tests for the year-scale harvester.

Nothing here touches the network. The collector is stubbed at
``data.harvest.fetch_gdelt_day``, which is the single seam through which the
harvester reaches GDELT, so resume, retry and failure handling are all
exercised deterministically.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from data import harvest
from data.gdelt_collector import generate_mock_data
from data.store import EventStore


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    """An initialised, empty store on a temporary path."""
    with EventStore(tmp_path / "harvest.db") as handle:
        yield handle


def _day_frame(day: str, rows: int = 20) -> pd.DataFrame:
    """Build a plausible one-day GDELT frame for the stubbed collector.

    Parameters
    ----------
    day : str
        Day as ``YYYY-MM-DD``.
    rows : int, optional
        Events to synthesise.

    Returns
    -------
    pandas.DataFrame
        Raw events carrying a ``download_bytes`` attribute, as
        :func:`data.gdelt_collector.fetch_gdelt_day` returns.
    """
    date = datetime.strptime(day, "%Y-%m-%d")
    frame = generate_mock_data(date, date, n_events=rows, seed=abs(hash(day)) % 10_000)
    frame.attrs["download_bytes"] = 1_000 * rows
    return frame


class _Collector:
    """A stub collector that records calls and replays scripted outcomes.

    Parameters
    ----------
    failures : set of str or None, optional
        Days for which the fetch returns ``None``, standing in for a
        transport failure or an unpublished export.
    """

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[str] = []

    def __call__(self, date: datetime, target_rows: Any = None) -> pd.DataFrame | None:
        """Return a day's frame, or ``None`` for a scripted failure.

        Parameters
        ----------
        date : datetime
            Day requested by the harvester.
        target_rows : Any, optional
            Retention cap, recorded but unused.

        Returns
        -------
        pandas.DataFrame or None
            Scripted outcome for the day.
        """
        day = date.strftime("%Y-%m-%d")
        self.calls.append(day)
        return None if day in self.failures else _day_frame(day)

    @property
    def days(self) -> set[str]:
        """Distinct days the harvester asked for."""
        return set(self.calls)


@pytest.fixture
def collector(monkeypatch: pytest.MonkeyPatch) -> _Collector:
    """Replace the network collector with a deterministic stub."""
    stub = _Collector()
    monkeypatch.setattr(harvest, "fetch_gdelt_day", stub)
    monkeypatch.setattr(harvest, "_RETRY_BACKOFF_SECONDS", 0.0)
    monkeypatch.setattr(harvest, "_export_exists", lambda day: False)
    return stub


class TestDayRange:
    """Window enumeration is inclusive at both ends."""

    def test_inclusive_bounds(self) -> None:
        days = harvest.day_range(datetime(2024, 1, 1), datetime(2024, 1, 3))
        assert days == ["2024-01-01", "2024-01-02", "2024-01-03"]

    def test_single_day(self) -> None:
        assert harvest.day_range(datetime(2024, 1, 1), datetime(2024, 1, 1)) == [
            "2024-01-01"
        ]

    def test_reversed_window_is_empty(self) -> None:
        assert harvest.day_range(datetime(2024, 1, 3), datetime(2024, 1, 1)) == []

    def test_a_year_is_365_or_366_days(self) -> None:
        days = harvest.day_range(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert len(days) == 366  # 2024 is a leap year


class TestWindowResolution:
    """The CLI resolves windows the way the help text promises."""

    def test_year_flag_spans_365_days(self) -> None:
        args = harvest.parse_args(["--year", "--end", "2024-12-31"])
        start, end = harvest.resolve_window(args)
        assert len(harvest.day_range(start, end)) == 365
        assert end.date() == datetime(2024, 12, 31).date()

    def test_explicit_window_is_respected(self) -> None:
        args = harvest.parse_args(["--start", "2024-03-01", "--end", "2024-03-05"])
        start, end = harvest.resolve_window(args)
        assert (start.date().isoformat(), end.date().isoformat()) == (
            "2024-03-01", "2024-03-05"
        )

    def test_default_end_is_yesterday(self) -> None:
        """GDELT publishes a day's export the following morning.

        Defaulting to today would make every run's final day a guaranteed
        failure, which then lingers in the harvest log as a retryable gap.
        """
        args = harvest.parse_args([])
        _, end = harvest.resolve_window(args)
        assert end.date() < datetime.now().date()

    def test_reversed_window_is_rejected(self) -> None:
        args = harvest.parse_args(["--start", "2024-05-01", "--end", "2024-04-01"])
        with pytest.raises(ValueError, match="after end date"):
            harvest.resolve_window(args)


class TestResume:
    """A rerun continues rather than restarting."""

    def test_pending_excludes_completed_days(self, store: EventStore) -> None:
        days = ["2024-01-01", "2024-01-02", "2024-01-03"]
        store.record_harvest("2024-01-02", "ok", rows=5)
        assert harvest.pending_days(store, days) == ["2024-01-01", "2024-01-03"]

    def test_pending_retains_failed_days(self, store: EventStore) -> None:
        store.record_harvest("2024-01-01", "failed", message="timeout")
        assert harvest.pending_days(store, ["2024-01-01"]) == ["2024-01-01"]

    def test_second_run_fetches_nothing(
        self, store: EventStore, collector: _Collector
    ) -> None:
        window = (datetime(2024, 1, 1), datetime(2024, 1, 3))
        first = harvest.harvest_range(store, *window, workers=2)
        assert first.collected == 3
        assert collector.days == {"2024-01-01", "2024-01-02", "2024-01-03"}

        collector.calls.clear()
        second = harvest.harvest_range(store, *window, workers=2)
        assert second.skipped == 3
        assert second.collected == 0
        assert collector.calls == []

    def test_extending_the_window_fetches_only_the_new_days(
        self, store: EventStore, collector: _Collector
    ) -> None:
        harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 2))
        collector.calls.clear()
        harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 4))
        assert collector.days == {"2024-01-03", "2024-01-04"}


class TestCollection:
    """Collected days land in the store with their rollups rebuilt."""

    def test_events_and_rollups_are_written(
        self, store: EventStore, collector: _Collector
    ) -> None:
        report = harvest.harvest_range(
            store, datetime(2024, 1, 1), datetime(2024, 1, 3), workers=2
        )
        stats = store.stats()
        assert report.events == stats["events"] > 0
        assert stats["dyad_rows"] > 0
        assert stats["days_held"] == 3

    def test_download_volume_is_reported(
        self, store: EventStore, collector: _Collector
    ) -> None:
        report = harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 2))
        assert report.bytes_downloaded == 2 * 20 * 1_000

    def test_each_day_is_stamped_with_its_own_date(
        self, store: EventStore, collector: _Collector
    ) -> None:
        """The harvester overrides the frame's dates with the requested day.

        GDELT's SQLDATE is the date of the event, not of the export, so rows
        in one day's file can carry neighbouring dates. Stamping the export
        day is what makes the harvest log a truthful index of the store.
        """
        harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 3))
        assert store.date_bounds() == ("2024-01-01", "2024-01-03")

    def test_report_totals_are_self_consistent(
        self, store: EventStore, collector: _Collector
    ) -> None:
        report = harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 5))
        assert report.collected + report.empty + report.failed == report.requested
        assert "Days collected" in report.summary()


class TestFailureHandling:
    """Unreachable and unpublished days are told apart."""

    def test_unpublished_day_is_terminal(
        self, store: EventStore, collector: _Collector
    ) -> None:
        collector.failures = {"2024-01-02"}
        report = harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 3))

        assert report.empty == 1
        assert report.failed == 0
        # Terminal: a rerun must not spend attempts on it again.
        assert "2024-01-02" in store.completed_days()

    def test_unreachable_day_stays_retryable(
        self,
        store: EventStore,
        collector: _Collector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(harvest, "_export_exists", lambda day: True)
        collector.failures = {"2024-01-02"}
        report = harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 3))

        assert report.failed == 1
        assert store.failed_days() == ["2024-01-02"]
        assert harvest.pending_days(store, ["2024-01-02"]) == ["2024-01-02"]

    def test_a_failure_is_retried_before_being_recorded(
        self,
        store: EventStore,
        collector: _Collector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(harvest, "_export_exists", lambda day: True)
        collector.failures = {"2024-01-01"}
        harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 1))
        assert collector.calls.count("2024-01-01") == harvest._MAX_ATTEMPTS

    def test_retry_failed_targets_only_the_failures(
        self,
        store: EventStore,
        collector: _Collector,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(harvest, "_export_exists", lambda day: True)
        collector.failures = {"2024-01-02"}
        harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 3))

        collector.failures = set()
        collector.calls.clear()
        report = harvest.harvest_range(
            store, datetime(2024, 1, 1), datetime(2024, 1, 3), retry_failed=True
        )

        assert collector.days == {"2024-01-02"}
        assert report.collected == 1
        assert store.failed_days() == []

    def test_one_bad_day_does_not_stop_the_run(
        self, store: EventStore, collector: _Collector
    ) -> None:
        collector.failures = {"2024-01-02"}
        report = harvest.harvest_range(store, datetime(2024, 1, 1), datetime(2024, 1, 4))
        assert report.collected == 3


class TestSyntheticHarvest:
    """The offline path fills a window without touching the network."""

    def test_a_short_window_is_filled(self, store: EventStore) -> None:
        report = harvest.harvest_mock(
            store, datetime(2024, 1, 1), datetime(2024, 1, 10), events_per_day=25
        )
        assert report.requested == 10
        assert report.events > 0
        assert store.stats()["days_held"] == 10

    def test_batching_spans_a_long_window(self, store: EventStore) -> None:
        """Generation batches monthly, so batch boundaries must not drop days."""
        report = harvest.harvest_mock(
            store, datetime(2024, 1, 1), datetime(2024, 3, 31), events_per_day=8
        )
        assert report.requested == 91
        assert report.collected + report.empty == 91
        assert store.date_bounds() == ("2024-01-01", "2024-03-31")

    def test_rollups_are_built(self, store: EventStore) -> None:
        harvest.harvest_mock(
            store, datetime(2024, 1, 1), datetime(2024, 1, 5), events_per_day=20
        )
        assert not store.dyad_aggregates().empty


class TestCostProjection:
    """The pre-flight projection scales with the work requested."""

    def test_more_days_cost_more(self) -> None:
        small = harvest.project_cost(10, workers=6)
        large = harvest.project_cost(365, workers=6)
        assert "10" in small and "365" in large

    def test_more_workers_reduce_the_estimate(self) -> None:
        assert harvest.project_cost(365, 1) != harvest.project_cost(365, 12)

    def test_projection_names_resumability(self) -> None:
        assert "resumable" in harvest.project_cost(365, 6)


class TestDurationFormat:
    """Durations render without empty leading units."""

    @pytest.mark.parametrize("seconds,expected", [
        (0, "0s"),
        (45, "45s"),
        (90, "1m 30s"),
        (3661, "1h 01m 01s"),
    ])
    def test_rendering(self, seconds: int, expected: str) -> None:
        assert harvest._format_duration(seconds) == expected
