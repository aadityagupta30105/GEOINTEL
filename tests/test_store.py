"""Contract tests for the SQLite event store.

The store's central promise is that a graph built from a database window is
indistinguishable from one built from the equivalent event frame. That
equivalence is asserted directly here rather than inferred, because the two
paths compute the same aggregate through completely different machinery: one
in SQL, one in pandas.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from analysis.graph_builder import (
    aggregate_events,
    build_graph,
    build_graph_from_aggregates,
    build_temporal_graphs,
    build_temporal_graphs_from_aggregates,
)
from data.gdelt_collector import generate_mock_data, preprocess
from data.store import (
    AGGREGATE_COLUMNS,
    EVENT_COLUMNS,
    SCHEMA_VERSION,
    EventStore,
    is_read_only_sql,
)


@pytest.fixture
def store(tmp_path: Path) -> EventStore:
    """An initialised, empty store on a temporary path."""
    with EventStore(tmp_path / "test.db") as handle:
        yield handle


@pytest.fixture
def loaded(store: EventStore, events: pd.DataFrame) -> EventStore:
    """A store holding the session's deterministic event frame."""
    store.upsert_events(events)
    return store


class TestSchema:
    """The store creates a complete, versioned schema."""

    def test_tables_and_views_exist(self, store: EventStore) -> None:
        names = set(store.schema()["object"])
        assert {"events", "dyad_daily", "harvest_log", "meta"} <= names
        assert {"v_edges", "v_country_activity", "v_monthly"} <= names

    def test_schema_version_is_stamped(self, store: EventStore) -> None:
        version = store.query("SELECT value FROM meta WHERE key = 'schema_version'")
        assert int(version.iloc[0, 0]) == SCHEMA_VERSION

    def test_initialise_is_idempotent(self, store: EventStore) -> None:
        store.initialise()
        store.initialise()
        assert store.stats()["events"] == 0

    def test_empty_store_reports_no_bounds(self, store: EventStore) -> None:
        assert store.date_bounds() == (None, None)
        assert store.countries() == []
        assert store.coverage().empty


class TestIngestion:
    """Events round-trip into the store without loss or duplication."""

    def test_upsert_stores_every_row(
        self, store: EventStore, events: pd.DataFrame
    ) -> None:
        assert store.upsert_events(events) == len(events)
        assert store.stats()["events"] == len(events)

    def test_upsert_is_idempotent(
        self, store: EventStore, events: pd.DataFrame
    ) -> None:
        store.upsert_events(events)
        assert store.upsert_events(events) == 0
        assert store.stats()["events"] == len(events)

    def test_gdelt_ids_are_used_verbatim(self, store: EventStore) -> None:
        """A real GLOBALEVENTID becomes the primary key.

        This is what makes re-harvesting a day free rather than duplicative,
        so it is asserted on the identifier itself, not on a row count.
        """
        frame = preprocess(
            generate_mock_data(datetime(2024, 1, 1), datetime(2024, 1, 2), 10, seed=1)
        ).assign(GLOBALEVENTID=range(900_000, 900_010))

        store.upsert_events(frame)
        stored = store.query("SELECT event_id FROM events ORDER BY event_id")
        assert stored["event_id"].tolist() == list(range(900_000, 900_010))

    def test_synthetic_ids_never_collide_with_gdelt_ids(
        self, loaded: EventStore
    ) -> None:
        largest = loaded.query("SELECT MAX(event_id) AS m FROM events").iloc[0, 0]
        assert largest < 0

    def test_empty_frame_stores_nothing(self, store: EventStore) -> None:
        assert store.upsert_events(pd.DataFrame()) == 0

    def test_rows_without_a_parseable_date_are_dropped(
        self, store: EventStore, events: pd.DataFrame
    ) -> None:
        broken = events.head(5).copy()
        broken["date"] = "not-a-date"
        assert store.upsert_events(broken) == 0

    def test_events_read_back_in_pipeline_columns(self, loaded: EventStore) -> None:
        frame = loaded.events(limit=10)
        assert list(frame.columns) == list(EVENT_COLUMNS)
        assert len(frame) == 10


class TestAggregateEquivalence:
    """A graph from the store equals a graph from the event frame."""

    def test_aggregate_columns_match_the_in_memory_path(
        self, loaded: EventStore, events: pd.DataFrame
    ) -> None:
        from_sql = loaded.dyad_aggregates()
        from_pandas = aggregate_events(events)
        assert set(from_sql.columns) == set(AGGREGATE_COLUMNS)
        assert set(from_pandas.columns) == set(AGGREGATE_COLUMNS)
        assert len(from_sql) == len(from_pandas)

    def test_graphs_are_identical(
        self, loaded: EventStore, events: pd.DataFrame
    ) -> None:
        expected = build_graph(events)
        actual = build_graph_from_aggregates(loaded.dyad_aggregates())

        assert actual.number_of_nodes() == expected.number_of_nodes()
        assert set(actual.edges()) == set(expected.edges())

        for source, target, data in expected.edges(data=True):
            stored = actual[source][target]
            assert stored["num_events"] == data["num_events"]
            assert stored["mentions"] == data["mentions"]
            assert stored["tone"] == pytest.approx(data["tone"], abs=1e-9)
            assert stored["event_types"] == data["event_types"]
            assert stored["dominant_type"] == data["dominant_type"]
            assert stored["conflict_count"] == data["conflict_count"]
            assert stored["coop_count"] == data["coop_count"]

    def test_temporal_snapshots_match(
        self, loaded: EventStore, events: pd.DataFrame
    ) -> None:
        expected = build_temporal_graphs(events, period="month")
        actual = build_temporal_graphs_from_aggregates(
            loaded.dyad_aggregates(period="month")
        )

        assert set(actual) == set(expected)
        for period, graph in expected.items():
            assert set(actual[period].edges()) == set(graph.edges())

    @pytest.mark.parametrize("period", ["month", "quarter", "year"])
    def test_period_labels_match_pandas(
        self, loaded: EventStore, events: pd.DataFrame, period: str
    ) -> None:
        """SQL period labels must match the ones pandas produces.

        The quarter label in particular is computed arithmetically in SQL and
        would silently diverge from ``Period.to_period('Q')`` if the month
        boundaries were off by one.
        """
        sql_labels = set(loaded.dyad_aggregates(period=period)["period"])
        pandas_labels = set(build_temporal_graphs(events, period=period))
        assert sql_labels == pandas_labels


class TestFilters:
    """Filters are applied in SQL and restrict what is returned."""

    def test_window_narrows_the_result(self, loaded: EventStore) -> None:
        first, last = loaded.date_bounds()
        full = loaded.dyad_aggregates()
        narrow = loaded.dyad_aggregates(start=first, end=first)

        assert first is not None and last is not None
        assert 0 < narrow["num_events"].sum() < full["num_events"].sum()

    def test_event_type_filter(self, loaded: EventStore) -> None:
        frame = loaded.dyad_aggregates(event_types=["Conflict"])
        assert set(frame["event_type"]) == {"Conflict"}

    def test_country_filter_matches_either_actor(self, loaded: EventStore) -> None:
        frame = loaded.dyad_aggregates(countries=["USA"])
        assert not frame.empty
        involved = (
            (frame["Actor1CountryCode"] == "USA") | (frame["Actor2CountryCode"] == "USA")
        )
        assert involved.all()

    def test_impossible_filter_returns_typed_empty_frame(
        self, loaded: EventStore
    ) -> None:
        frame = loaded.dyad_aggregates(event_types=["No Such Type"])
        assert frame.empty
        assert set(frame.columns) == set(AGGREGATE_COLUMNS)

    def test_row_limit_is_honoured(self, loaded: EventStore) -> None:
        assert len(loaded.events(limit=25)) == 25

    def test_count_events_matches_an_uncapped_read(self, loaded: EventStore) -> None:
        assert loaded.count_events() == len(loaded.events())

    def test_count_events_respects_filters(self, loaded: EventStore) -> None:
        conflict = loaded.count_events(event_types=["Conflict"])
        assert 0 < conflict < loaded.count_events()

    def test_a_capped_read_still_spans_the_whole_window(
        self, loaded: EventStore
    ) -> None:
        """A cap must thin the window, never truncate it.

        ``ORDER BY date LIMIT n`` returns a chronological prefix, so a capped
        read would silently cover only the earliest part of the requested
        range while every downstream artefact described it as the whole. The
        capped read must reach the same last date as the uncapped one.
        """
        full = loaded.events()
        capped = loaded.events(limit=len(full) // 4)

        assert len(capped) <= len(full) // 4
        assert capped["date"].min() == full["date"].min()
        assert capped["date"].max() == full["date"].max()

    def test_a_capped_read_preserves_the_windows_shape(
        self, loaded: EventStore
    ) -> None:
        """Thinning must not distort the distribution across the window.

        A prefix truncation would put every retained row in the first months.
        Checking that each half of the window keeps roughly its original share
        is what distinguishes a sample from a truncation.
        """
        full = loaded.events()
        midpoint = full["date"].iloc[len(full) // 2]

        capped = loaded.events(limit=len(full) // 4)
        expected = (full["date"] < midpoint).mean()
        actual = (capped["date"] < midpoint).mean()

        assert actual == pytest.approx(expected, abs=0.10)

    def test_an_uncapped_read_is_not_thinned(self, loaded: EventStore) -> None:
        generous = loaded.events(limit=loaded.count_events() * 2)
        assert len(generous) == loaded.count_events()


class TestHarvestLog:
    """The harvest log is what makes collection resumable."""

    def test_terminal_days_are_reported_complete(self, store: EventStore) -> None:
        store.record_harvest("2024-01-01", "ok", rows=100)
        store.record_harvest("2024-01-02", "empty")
        assert store.completed_days() == {"2024-01-01", "2024-01-02"}

    def test_failed_days_stay_pending(self, store: EventStore) -> None:
        store.record_harvest("2024-01-03", "failed", message="timeout")
        assert store.completed_days() == set()
        assert store.failed_days() == ["2024-01-03"]

    def test_a_retry_replaces_the_previous_outcome(self, store: EventStore) -> None:
        store.record_harvest("2024-01-04", "failed", message="timeout")
        store.record_harvest("2024-01-04", "ok", rows=42)
        assert store.failed_days() == []
        assert store.completed_days() == {"2024-01-04"}

    def test_coverage_groups_by_month(self, store: EventStore) -> None:
        store.record_harvest("2024-01-01", "ok", rows=10)
        store.record_harvest("2024-01-02", "failed")
        store.record_harvest("2024-02-01", "empty")

        coverage = store.coverage().set_index("month")
        assert int(coverage.loc["2024-01", "days_ok"]) == 1
        assert int(coverage.loc["2024-01", "days_failed"]) == 1
        assert int(coverage.loc["2024-02", "days_empty"]) == 1


class TestRollups:
    """The rollup is derived state and must be rebuildable."""

    def test_rebuild_reproduces_the_aggregate(self, loaded: EventStore) -> None:
        before = loaded.dyad_aggregates()
        loaded.rebuild_rollups()
        after = loaded.dyad_aggregates()
        assert len(after) == len(before)
        assert after["num_events"].sum() == before["num_events"].sum()

    def test_partial_rebuild_touches_only_the_named_dates(
        self, loaded: EventStore
    ) -> None:
        first, _ = loaded.date_bounds()
        total_before = loaded.dyad_aggregates()["num_events"].sum()
        loaded.rebuild_rollups(dates=[str(first)])
        assert loaded.dyad_aggregates()["num_events"].sum() == total_before

    def test_deferred_rebuild_leaves_the_rollup_stale(
        self, store: EventStore, events: pd.DataFrame
    ) -> None:
        """A deferred rebuild is the harvester's fast path, and is explicit.

        Bulk loaders pass ``rebuild=False`` and rebuild once at the end. The
        rollup is genuinely stale until then, which is safe only because it is
        never the source of truth.
        """
        store.upsert_events(events, rebuild=False)
        assert store.dyad_aggregates().empty
        store.rebuild_rollups()
        assert not store.dyad_aggregates().empty


class TestViews:
    """The convenience views agree with the graph layer."""

    def test_v_edges_matches_the_graph(
        self, loaded: EventStore, events: pd.DataFrame
    ) -> None:
        graph = build_graph(events)
        edges = loaded.query("SELECT * FROM v_edges").set_index(["source", "target"])

        assert len(edges) == graph.number_of_edges()
        for (source, target), row in edges.head(50).iterrows():
            data = graph[source][target]
            assert int(row["num_events"]) == data["num_events"]
            assert row["tone"] == pytest.approx(data["tone"], abs=1e-4)
            assert int(row["conflict_count"]) == data["conflict_count"]

    def test_v_country_activity_counts_both_directions(
        self, loaded: EventStore
    ) -> None:
        activity = loaded.query(
            "SELECT * FROM v_country_activity WHERE country = 'USA'"
        )
        assert int(activity.iloc[0]["total_events"]) > 0

    def test_v_monthly_covers_every_stored_month(self, loaded: EventStore) -> None:
        months = set(loaded.query("SELECT month FROM v_monthly")["month"])
        stored = set(loaded.dyad_aggregates(period="month")["period"])
        assert months == stored


class TestReadOnlyAccess:
    """Read-only access is enforced by the connection, not only by parsing."""

    def test_missing_store_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            EventStore(tmp_path / "absent.db", read_only=True).connect()

    def test_writes_are_refused(self, tmp_path: Path, events: pd.DataFrame) -> None:
        path = tmp_path / "ro.db"
        with EventStore(path) as writable:
            writable.upsert_events(events.head(50))

        with EventStore(path, read_only=True) as reader:
            assert not reader.dyad_aggregates().empty
            with pytest.raises(sqlite3.OperationalError):
                reader.connect().execute("DELETE FROM events")


class TestReadOnlySqlGuard:
    """The console guard admits reads and nothing else."""

    @pytest.mark.parametrize("statement", [
        "SELECT * FROM v_edges",
        "select source from v_edges limit 5;",
        "WITH t AS (SELECT 1 AS a) SELECT * FROM t",
        "  SELECT 1  ",
    ])
    def test_reads_are_admitted(self, statement: str) -> None:
        assert is_read_only_sql(statement)

    @pytest.mark.parametrize("statement", [
        "DELETE FROM events",
        "DROP TABLE events",
        "INSERT INTO events VALUES (1)",
        "UPDATE events SET actor1 = 'XXX'",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA table_info(events)",
        "VACUUM",
        "SELECT 1; DROP TABLE events",
        "SELECT * FROM events; DELETE FROM events;",
        "",
        "   ",
    ])
    def test_writes_and_chains_are_rejected(self, statement: str) -> None:
        assert not is_read_only_sql(statement)


class TestConflictTypeAgreement:
    """The store's conflict taxonomy tracks the graph layer's.

    ``data.store`` restates the conflict and cooperation labels rather than
    importing them, so that storage carries no dependency on analysis. This
    test is the seam that keeps the two copies honest.
    """

    def test_labels_match_the_graph_builder(self) -> None:
        from analysis import graph_builder
        from data import store as store_module

        assert set(store_module._CONFLICT_TYPES) == graph_builder._CONFLICT_TYPES
        assert set(store_module._COOPERATION_TYPES) == graph_builder._COOPERATION_TYPES
