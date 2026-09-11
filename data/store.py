"""
Event Store
===========
Structured, queryable persistence for the GeoIntel event stream.

The store is a single SQLite file. It replaces the flat CSV export as the
authoritative home for collected events and gives every consumer a real query
surface instead of a whole-file read.

Why SQLite scales to a full year here
-------------------------------------
A year of full-fidelity GDELT bilateral events is on the order of several
million rows, which is more than a dashboard should ever load into memory. No
consumer reads the event table directly. Every aggregate the graph layer needs
is served from ``dyad_daily``, a rollup keyed by
``(date, actor1, actor2, event_type)`` that collapses the event stream by one
to two orders of magnitude while preserving exactly the four quantities
:func:`analysis.graph_builder.build_graph_from_aggregates` consumes: event
count, tone sum, mentions and the event-type histogram. Graph construction
over an arbitrary window therefore costs one indexed ``GROUP BY`` rather than a
full scan.

Schema
------
``events``
    One row per collected event, keyed by GDELT ``GLOBALEVENTID`` where
    available. Re-harvesting a day is idempotent.
``dyad_daily``
    Rollup of ``events``. Derived, and rebuildable at any time from
    :meth:`EventStore.rebuild_rollups`.
``harvest_log``
    One row per calendar day attempted, carrying the outcome. This is what
    makes collection resumable.
``meta``
    Key-value store holding the schema version.

Views ``v_edges``, ``v_country_activity`` and ``v_monthly`` exist for
hand-written SQL; nothing in the platform depends on them.

Usage
-----
::

    with EventStore() as store:
        store.upsert_events(preprocess(raw))
        edges = store.dyad_aggregates("2024-01-01", "2024-12-31")
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import pandas as pd

from utils.logging_config import OK, WARN, get_logger

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_DB_PATH",
    "AGGREGATE_COLUMNS",
    "EVENT_COLUMNS",
    "HarvestStatus",
    "StoreStats",
    "EventStore",
    "open_store",
    "is_read_only_sql",
]

_log = get_logger(__name__)

SCHEMA_VERSION: Final[int] = 1

DEFAULT_DB_PATH: Final[Path] = Path(__file__).resolve().parent / "geointel.db"

# Columns returned by :meth:`EventStore.dyad_aggregates`, in the pipeline's
# own naming so that the graph layer consumes store output unchanged.
AGGREGATE_COLUMNS: Final[tuple[str, ...]] = (
    "Actor1CountryCode",
    "Actor2CountryCode",
    "event_type",
    "num_events",
    "tone_sum",
    "mentions",
)

# Columns returned by :meth:`EventStore.events`, matching the frame that
# :func:`data.gdelt_collector.preprocess` produces.
EVENT_COLUMNS: Final[tuple[str, ...]] = (
    "Actor1CountryCode",
    "Actor2CountryCode",
    "date",
    "EventRootCode",
    "event_label",
    "event_type",
    "QuadClass",
    "GoldsteinScale",
    "AvgTone",
    "tone_norm",
    "NumMentions",
    "NumArticles",
    "SOURCEURL",
)

# Outcome recorded in ``harvest_log`` for one calendar day.
#
# ``ok``      The export was retrieved and yielded bilateral events.
# ``empty``   The export was retrieved and contained no bilateral events.
#             Terminal: re-fetching cannot change the answer.
# ``failed``  Transport or parse failure. Retryable.
HarvestStatus = str

_TERMINAL_STATUSES: Final[frozenset[str]] = frozenset({"ok", "empty"})

# Event types contributing to the conflict and cooperation counters. Kept in
# sync with analysis.graph_builder by the test suite rather than imported, so
# that the storage layer carries no dependency on the analysis layer.
_CONFLICT_TYPES: Final[tuple[str, ...]] = ("Conflict", "Military/Conflict")
_COOPERATION_TYPES: Final[tuple[str, ...]] = ("Cooperation", "Trade/Aid")

_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS events (
    event_id      INTEGER PRIMARY KEY,
    date          TEXT    NOT NULL,
    actor1        TEXT    NOT NULL,
    actor2        TEXT    NOT NULL,
    event_root    TEXT    NOT NULL,
    event_label   TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    quad_class    INTEGER NOT NULL,
    goldstein     REAL    NOT NULL,
    avg_tone      REAL    NOT NULL,
    tone_norm     REAL    NOT NULL,
    num_mentions  INTEGER NOT NULL,
    num_articles  INTEGER NOT NULL,
    source_url    TEXT
);

CREATE INDEX IF NOT EXISTS ix_events_date ON events(date);
CREATE INDEX IF NOT EXISTS ix_events_dyad ON events(actor1, actor2);
CREATE INDEX IF NOT EXISTS ix_events_type ON events(event_type);

CREATE TABLE IF NOT EXISTS dyad_daily (
    date        TEXT    NOT NULL,
    actor1      TEXT    NOT NULL,
    actor2      TEXT    NOT NULL,
    event_type  TEXT    NOT NULL,
    num_events  INTEGER NOT NULL,
    tone_sum    REAL    NOT NULL,
    mentions    INTEGER NOT NULL,
    PRIMARY KEY (date, actor1, actor2, event_type)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ix_dyad_daily_date ON dyad_daily(date);

CREATE TABLE IF NOT EXISTS harvest_log (
    day        TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    rows       INTEGER NOT NULL DEFAULT 0,
    bytes      INTEGER NOT NULL DEFAULT 0,
    fetched_at TEXT    NOT NULL,
    message    TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Convenience views for hand-written SQL. Rebuilt on every open so that a
# definition change ships with the code rather than requiring a migration.
_VIEWS: Final[str] = f"""
DROP VIEW IF EXISTS v_edges;
CREATE VIEW v_edges AS
SELECT
    actor1                                   AS source,
    actor2                                   AS target,
    SUM(num_events)                          AS num_events,
    SUM(tone_sum) / SUM(num_events)          AS tone,
    SUM(mentions)                            AS mentions,
    SUM(CASE WHEN event_type IN {_CONFLICT_TYPES} THEN num_events ELSE 0 END)
                                             AS conflict_count,
    SUM(CASE WHEN event_type IN {_COOPERATION_TYPES} THEN num_events ELSE 0 END)
                                             AS coop_count,
    MIN(date)                                AS first_seen,
    MAX(date)                                AS last_seen
FROM dyad_daily
GROUP BY actor1, actor2;

DROP VIEW IF EXISTS v_country_activity;
CREATE VIEW v_country_activity AS
SELECT
    country,
    SUM(num_events)                 AS total_events,
    SUM(tone_sum) / SUM(num_events) AS avg_tone,
    COUNT(DISTINCT counterpart)     AS partners
FROM (
    SELECT actor1 AS country, actor2 AS counterpart, num_events, tone_sum
      FROM dyad_daily
    UNION ALL
    SELECT actor2 AS country, actor1 AS counterpart, num_events, tone_sum
      FROM dyad_daily
)
GROUP BY country;

DROP VIEW IF EXISTS v_monthly;
CREATE VIEW v_monthly AS
SELECT
    substr(date, 1, 7)              AS month,
    SUM(num_events)                 AS num_events,
    SUM(tone_sum) / SUM(num_events) AS avg_tone,
    COUNT(DISTINCT actor1 || actor2) AS dyads
FROM dyad_daily
GROUP BY month;
"""

# Statements that make a connection perform. WAL keeps readers unblocked while
# the harvester writes; the negative cache size is expressed in kibibytes.
_PRAGMAS: Final[tuple[str, ...]] = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA cache_size=-131072",
    "PRAGMA temp_store=MEMORY",
    "PRAGMA foreign_keys=ON",
)

# SQL keywords that make a statement something other than a read.
_MUTATING_KEYWORDS: Final[frozenset[str]] = frozenset({
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "begin", "commit",
    "rollback", "savepoint", "release", "analyze",
})

_SYNTHETIC_ID_MASK: Final[int] = (1 << 62) - 1


class StoreStats(dict[str, Any]):
    """Summary statistics for a store.

    Keys
    ----
    events : int
        Total rows in the event table.
    dyad_rows : int
        Rows in the rollup table.
    countries : int
        Distinct country codes appearing as either actor.
    first_date, last_date : str or None
        Inclusive bounds of the stored window.
    days_held : int
        Calendar days with a terminal harvest outcome.
    size_bytes : int
        On-disk size of the database file.
    """


class EventStore:
    """SQLite-backed store for preprocessed geopolitical events.

    The instance is a thin handle: a connection is opened on first use and
    held until :meth:`close`. Use as a context manager to guarantee closure.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Database file. Created along with its parent directory when absent.
    read_only : bool, optional
        Open the file in SQLite read-only mode. Writes raise
        :class:`sqlite3.OperationalError`. Requires an existing file.

    Attributes
    ----------
    path : pathlib.Path
        Resolved database path.
    read_only : bool
        Whether the connection refuses writes.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_DB_PATH,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._connection: sqlite3.Connection | None = None

    # --- Lifecycle ----------------------------------------------------------

    def __enter__(self) -> EventStore:
        """Open the store and return it.

        Returns
        -------
        EventStore
            This instance, with the schema applied.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying connection.

        Parameters
        ----------
        exc_type, exc, traceback
            Standard context-manager exception triple, unused.
        """
        self.close()

    def connect(self) -> sqlite3.Connection:
        """Return the live connection, opening and initialising it if needed.

        Returns
        -------
        sqlite3.Connection
            Connection with the platform pragmas applied.

        Raises
        ------
        FileNotFoundError
            When ``read_only`` is set and the database file does not exist.
        """
        if self._connection is not None:
            return self._connection

        if self.read_only:
            if not self.path.exists():
                raise FileNotFoundError(f"No store at {self.path}")
            connection = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=30.0
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=60.0)

        connection.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            try:
                connection.execute(pragma)
            except sqlite3.OperationalError:
                # A read-only connection refuses journal_mode changes. The
                # remaining pragmas are advisory, so a refusal is not fatal.
                continue

        self._connection = connection
        if not self.read_only:
            self.initialise()
        return connection

    def close(self) -> None:
        """Close the connection if one is open."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def initialise(self) -> None:
        """Create the schema, indexes and views, and stamp the version.

        Idempotent: safe to call on an existing store.
        """
        connection = self.connect()
        with connection:
            connection.executescript(_SCHEMA)
            connection.executescript(_VIEWS)
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside a single committed transaction.

        Yields
        ------
        sqlite3.Connection
            The live connection. The transaction commits on clean exit and
            rolls back on exception.
        """
        connection = self.connect()
        with connection:
            yield connection

    # --- Writes -------------------------------------------------------------

    def upsert_events(self, frame: pd.DataFrame, rebuild: bool = True) -> int:
        """Insert preprocessed events, ignoring rows already held.

        Idempotency rests on the primary key. GDELT rows carry
        ``GLOBALEVENTID``, so re-harvesting a day inserts nothing new.
        Synthetic rows have no natural identifier and are keyed by a content
        digest, which makes a repeated generator run with the same seed
        equally idempotent.

        Parameters
        ----------
        frame : pandas.DataFrame
            Output of :func:`data.gdelt_collector.preprocess`.
        rebuild : bool, optional
            Refresh the rollup for the affected dates. Pass ``False`` when
            loading many batches, then call :meth:`rebuild_rollups` once.

        Returns
        -------
        int
            Number of rows actually inserted.
        """
        if frame.empty:
            return 0

        prepared = self._to_rows(frame)
        if not prepared:
            return 0

        connection = self.connect()
        before = self._scalar("SELECT COUNT(*) FROM events")
        with connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO events (
                    event_id, date, actor1, actor2, event_root, event_label,
                    event_type, quad_class, goldstein, avg_tone, tone_norm,
                    num_mentions, num_articles, source_url
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                prepared,
            )
        inserted = self._scalar("SELECT COUNT(*) FROM events") - before

        if rebuild and inserted:
            self.rebuild_rollups(dates=sorted({str(row[1]) for row in prepared}))

        return inserted

    def rebuild_rollups(self, dates: Sequence[str] | None = None) -> int:
        """Recompute the ``dyad_daily`` rollup from the event table.

        Parameters
        ----------
        dates : sequence of str or None, optional
            Restrict the rebuild to these ``YYYY-MM-DD`` days. ``None``
            rebuilds the whole table, which is the correct action after a bulk
            import or a schema change.

        Returns
        -------
        int
            Number of rollup rows written.
        """
        connection = self.connect()
        aggregate = """
            SELECT date, actor1, actor2, event_type,
                   COUNT(*)         AS num_events,
                   SUM(tone_norm)   AS tone_sum,
                   SUM(num_mentions) AS mentions
              FROM events
        """
        with connection:
            if dates is None:
                connection.execute("DELETE FROM dyad_daily")
                connection.execute(
                    "INSERT INTO dyad_daily "
                    f"{aggregate} GROUP BY date, actor1, actor2, event_type"
                )
            else:
                unique = sorted(set(dates))
                placeholders = ",".join("?" * len(unique))
                connection.execute(
                    f"DELETE FROM dyad_daily WHERE date IN ({placeholders})", unique
                )
                connection.execute(
                    "INSERT INTO dyad_daily "
                    f"{aggregate} WHERE date IN ({placeholders}) "
                    "GROUP BY date, actor1, actor2, event_type",
                    unique,
                )
        return self._scalar("SELECT COUNT(*) FROM dyad_daily")

    def record_harvest(
        self,
        day: str,
        status: HarvestStatus,
        rows: int = 0,
        size_bytes: int = 0,
        message: str | None = None,
    ) -> None:
        """Record the outcome of one calendar day's collection.

        Parameters
        ----------
        day : str
            Day as ``YYYY-MM-DD``.
        status : str
            One of ``ok``, ``empty`` or ``failed``.
        rows : int, optional
            Bilateral events retained for the day.
        size_bytes : int, optional
            Compressed bytes transferred.
        message : str or None, optional
            Failure detail, retained for diagnosis.
        """
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO harvest_log
                    (day, status, rows, bytes, fetched_at, message)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    day,
                    status,
                    int(rows),
                    int(size_bytes),
                    datetime.now().isoformat(timespec="seconds"),
                    message,
                ),
            )

    def completed_days(self) -> set[str]:
        """Return days with a terminal harvest outcome.

        A day is terminal when it was collected successfully or was confirmed
        to hold no bilateral events. Failures are excluded so that a rerun
        retries them.

        Returns
        -------
        set of str
            Days as ``YYYY-MM-DD``.
        """
        placeholders = ",".join("?" * len(_TERMINAL_STATUSES))
        rows = self.connect().execute(
            f"SELECT day FROM harvest_log WHERE status IN ({placeholders})",
            tuple(sorted(_TERMINAL_STATUSES)),
        )
        return {str(row["day"]) for row in rows}

    def failed_days(self) -> list[str]:
        """Return days whose last collection attempt failed.

        Returns
        -------
        list of str
            Days as ``YYYY-MM-DD``, chronologically ordered.
        """
        rows = self.connect().execute(
            "SELECT day FROM harvest_log WHERE status = 'failed' ORDER BY day"
        )
        return [str(row["day"]) for row in rows]

    # --- Reads --------------------------------------------------------------

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        """Run an arbitrary SQL statement and return the result frame.

        Parameters
        ----------
        sql : str
            Statement to execute.
        params : sequence, optional
            Bound parameters.

        Returns
        -------
        pandas.DataFrame
            Result set, empty when the statement returned no rows.
        """
        return pd.read_sql_query(sql, self.connect(), params=tuple(params))

    def dyad_aggregates(
        self,
        start: str | None = None,
        end: str | None = None,
        event_types: Sequence[str] | None = None,
        countries: Sequence[str] | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        """Aggregate the rollup into the frame the graph layer consumes.

        This is the hot path. The window, event-type and country filters are
        applied in SQL against the indexed rollup, so the amount of data that
        reaches pandas is proportional to the number of active dyads rather
        than to the number of events.

        Parameters
        ----------
        start, end : str or None, optional
            Inclusive window bounds as ``YYYY-MM-DD``.
        event_types : sequence of str or None, optional
            Restrict to these event types.
        countries : sequence of str or None, optional
            Restrict to dyads where either actor is in this set.
        period : {"month", "quarter", "year"} or None, optional
            When given, a ``period`` column is added and the aggregation is
            grouped by it as well, which yields temporal snapshots in one
            query instead of one query per period.

        Returns
        -------
        pandas.DataFrame
            Columns :data:`AGGREGATE_COLUMNS`, plus ``period`` when requested.
        """
        period_expression = _period_expression(period)
        select_period = f"{period_expression} AS period, " if period_expression else ""
        group_period = f"{period_expression}, " if period_expression else ""

        clause, params = self._filter_clause(start, end, event_types, countries)

        frame = self.query(
            f"""
            SELECT {select_period}
                   actor1 AS Actor1CountryCode,
                   actor2 AS Actor2CountryCode,
                   event_type,
                   SUM(num_events) AS num_events,
                   SUM(tone_sum)   AS tone_sum,
                   SUM(mentions)   AS mentions
              FROM dyad_daily
              {clause}
             GROUP BY {group_period} actor1, actor2, event_type
            """,
            params,
        )

        if frame.empty:
            columns = (["period"] if period_expression else []) + list(AGGREGATE_COLUMNS)
            return pd.DataFrame(columns=columns)
        return frame

    def events(
        self,
        start: str | None = None,
        end: str | None = None,
        event_types: Sequence[str] | None = None,
        countries: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Read raw events back in the pipeline's own column naming.

        Materialising events is expensive over a wide window; prefer
        :meth:`dyad_aggregates` for anything the graph layer consumes. This
        method exists for the pipeline's CSV export and for event-level
        inspection.

        Parameters
        ----------
        start, end : str or None, optional
            Inclusive window bounds as ``YYYY-MM-DD``.
        event_types : sequence of str or None, optional
            Restrict to these event types.
        countries : sequence of str or None, optional
            Restrict to dyads where either actor is in this set.
        limit : int or None, optional
            Cap the number of rows returned. When the window holds more rows
            than the cap, the result is thinned across the whole window rather
            than truncated at the cap. See the note below.

        Returns
        -------
        pandas.DataFrame
            Columns :data:`EVENT_COLUMNS`.

        Notes
        -----
        A plain ``ORDER BY date LIMIT n`` would return a chronological prefix,
        which silently changes the window the caller asked for: requesting six
        months of a large store would yield the first ten weeks of it, and
        every downstream artefact would describe that shorter window as though
        it were the requested one. Instead the cap is applied as a
        deterministic thinning across the whole window, so a capped read still
        spans the dates it was asked for and preserves their relative
        proportions. Counts are reduced; shape is not.

        The stride is taken on ``event_id``, which GDELT allocates as an
        ascending counter, so consecutive ids are close in time and a modulo
        stride distributes evenly over the window.
        """
        clause, params = self._filter_clause(
            start, end, event_types, countries, table_is_rollup=False
        )

        stride_clause = ""
        if limit:
            matching = self._scalar(
                f"SELECT COUNT(*) FROM events {clause}", params  # noqa: S608
            )
            if matching > limit:
                stride = -(-matching // int(limit))  # ceiling division
                connector = "AND" if clause else "WHERE"
                stride_clause = f" {connector} (abs(event_id) % {stride}) = 0"

        limit_clause = f" LIMIT {int(limit)}" if limit else ""

        frame = self.query(
            f"""
            SELECT actor1       AS Actor1CountryCode,
                   actor2       AS Actor2CountryCode,
                   date,
                   event_root   AS EventRootCode,
                   event_label,
                   event_type,
                   quad_class   AS QuadClass,
                   goldstein    AS GoldsteinScale,
                   avg_tone     AS AvgTone,
                   tone_norm,
                   num_mentions AS NumMentions,
                   num_articles AS NumArticles,
                   source_url   AS SOURCEURL
              FROM events
              {clause}{stride_clause}
             ORDER BY date{limit_clause}
            """,
            params,
        )
        return frame if not frame.empty else pd.DataFrame(columns=list(EVENT_COLUMNS))

    def count_events(
        self,
        start: str | None = None,
        end: str | None = None,
        event_types: Sequence[str] | None = None,
        countries: Sequence[str] | None = None,
    ) -> int:
        """Count the events matching a filter without materialising them.

        Lets a caller learn how much a window holds before deciding to read
        it, and lets a capped read report the population it was drawn from.

        Parameters
        ----------
        start, end : str or None, optional
            Inclusive window bounds as ``YYYY-MM-DD``.
        event_types : sequence of str or None, optional
            Restrict to these event types.
        countries : sequence of str or None, optional
            Restrict to dyads where either actor is in this set.

        Returns
        -------
        int
            Matching row count.
        """
        clause, params = self._filter_clause(
            start, end, event_types, countries, table_is_rollup=False
        )
        return self._scalar(f"SELECT COUNT(*) FROM events {clause}", params)  # noqa: S608

    def date_bounds(self) -> tuple[str | None, str | None]:
        """Return the inclusive date range held by the rollup.

        Returns
        -------
        tuple of (str or None, str or None)
            First and last stored day, both ``None`` for an empty store.
        """
        row = self.connect().execute(
            "SELECT MIN(date) AS lo, MAX(date) AS hi FROM dyad_daily"
        ).fetchone()
        return (row["lo"], row["hi"]) if row else (None, None)

    def event_types(self) -> list[str]:
        """Return the distinct event types present in the store.

        Returns
        -------
        list of str
            Alphabetically ordered labels.
        """
        rows = self.connect().execute(
            "SELECT DISTINCT event_type FROM dyad_daily ORDER BY event_type"
        )
        return [str(row["event_type"]) for row in rows]

    def countries(self) -> list[str]:
        """Return the distinct country codes present in the store.

        Returns
        -------
        list of str
            Alphabetically ordered ISO-3 codes.
        """
        rows = self.connect().execute(
            "SELECT actor1 AS c FROM dyad_daily "
            "UNION SELECT actor2 FROM dyad_daily ORDER BY c"
        )
        return [str(row["c"]) for row in rows]

    def coverage(self) -> pd.DataFrame:
        """Summarise collection completeness by month.

        Returns
        -------
        pandas.DataFrame
            One row per month with the count of days by outcome and the total
            events retained. Empty when nothing has been harvested.
        """
        return self.query(
            """
            SELECT substr(day, 1, 7) AS month,
                   SUM(status = 'ok')     AS days_ok,
                   SUM(status = 'empty')  AS days_empty,
                   SUM(status = 'failed') AS days_failed,
                   SUM(rows)              AS events,
                   SUM(bytes)             AS bytes
              FROM harvest_log
             GROUP BY month
             ORDER BY month
            """
        )

    def stats(self) -> StoreStats:
        """Return summary statistics for the store.

        Returns
        -------
        StoreStats
            Row counts, window bounds, day coverage and file size.
        """
        first, last = self.date_bounds()
        return StoreStats(
            events=self._scalar("SELECT COUNT(*) FROM events"),
            dyad_rows=self._scalar("SELECT COUNT(*) FROM dyad_daily"),
            countries=len(self.countries()),
            first_date=first,
            last_date=last,
            days_held=len(self.completed_days()),
            size_bytes=self._size_bytes(),
        )

    def _size_bytes(self) -> int:
        """Return the on-disk footprint of the store.

        Under write-ahead logging a large share of recently written data lives
        in the sidecar files until a checkpoint, so reporting the main file
        alone understates the store immediately after a harvest.

        Returns
        -------
        int
            Combined size of the database and its WAL sidecars, in bytes.
        """
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self.path}{suffix}")
            if candidate.exists():
                total += candidate.stat().st_size
        return total

    def schema(self) -> pd.DataFrame:
        """Describe every table and view in the store.

        Returns
        -------
        pandas.DataFrame
            One row per column, with the owning object and its kind.
        """
        objects = self.connect().execute(
            "SELECT name, type FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type DESC, name"
        ).fetchall()

        records: list[dict[str, str]] = []
        for entry in objects:
            for column in self.connect().execute(
                f'PRAGMA table_info("{entry["name"]}")'
            ):
                records.append({
                    "object": str(entry["name"]),
                    "kind": str(entry["type"]),
                    "column": str(column["name"]),
                    "type": str(column["type"]) or "ANY",
                })
        return pd.DataFrame(records)

    # --- Internals ----------------------------------------------------------

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Execute a statement returning a single integer.

        Parameters
        ----------
        sql : str
            Statement yielding one column in one row.
        params : sequence, optional
            Bound parameters.

        Returns
        -------
        int
            The scalar result, or ``0`` when the statement returned no row.
        """
        row = self.connect().execute(sql, tuple(params)).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    @staticmethod
    def _filter_clause(
        start: str | None,
        end: str | None,
        event_types: Sequence[str] | None,
        countries: Sequence[str] | None,
        table_is_rollup: bool = True,
    ) -> tuple[str, list[Any]]:
        """Build a parameterised ``WHERE`` clause for the read methods.

        Parameters
        ----------
        start, end : str or None
            Inclusive window bounds.
        event_types : sequence of str or None
            Event-type restriction.
        countries : sequence of str or None
            Country restriction, matched against either actor.
        table_is_rollup : bool, optional
            Unused for column naming, since both tables share the filtered
            column names; retained so the caller states its intent.

        Returns
        -------
        tuple of (str, list)
            The clause (possibly empty) and its bound parameters.
        """
        del table_is_rollup

        conditions: list[str] = []
        params: list[Any] = []

        if start:
            conditions.append("date >= ?")
            params.append(start)
        if end:
            conditions.append("date <= ?")
            params.append(end)
        if event_types:
            placeholders = ",".join("?" * len(event_types))
            conditions.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
        if countries:
            placeholders = ",".join("?" * len(countries))
            conditions.append(
                f"(actor1 IN ({placeholders}) OR actor2 IN ({placeholders}))"
            )
            params.extend(countries)
            params.extend(countries)

        return ("WHERE " + " AND ".join(conditions) if conditions else "", params)

    @staticmethod
    def _to_rows(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
        """Convert a preprocessed event frame into insertable tuples.

        Parameters
        ----------
        frame : pandas.DataFrame
            Output of :func:`data.gdelt_collector.preprocess`.

        Returns
        -------
        list of tuple
            One tuple per row, ordered to match the ``events`` insert.
        """
        working = frame.copy()

        for column, default in (
            ("NumArticles", 1),
            ("SOURCEURL", ""),
            ("event_label", "Unknown"),
        ):
            if column not in working.columns:
                working[column] = default

        # Dates reaching the store are ISO by construction: both the GDELT
        # collector and the synthetic generator emit YYYY-MM-DD. Parsing under
        # an explicit format rather than by inference makes anything else a
        # dropped row instead of a guess, and keeps pandas from warning that
        # it fell back to per-element parsing.
        working["date"] = pd.to_datetime(
            working["date"], format="ISO8601", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        working = working[working["date"].notna()]
        if working.empty:
            _log.warning("%s No rows carried a parseable date; nothing stored", WARN)
            return []

        identifiers = _event_ids(working)

        numeric = {
            "QuadClass": 1,
            "NumMentions": 1,
            "NumArticles": 1,
            "GoldsteinScale": 0.0,
            "AvgTone": 0.0,
            "tone_norm": 0.0,
        }
        for column, default in numeric.items():
            working[column] = pd.to_numeric(
                working.get(column), errors="coerce"
            ).fillna(default)

        return [
            (
                int(identifier),
                str(row.date),
                str(row.Actor1CountryCode),
                str(row.Actor2CountryCode),
                str(row.EventRootCode),
                str(row.event_label),
                str(row.event_type),
                int(row.QuadClass),
                float(row.GoldsteinScale),
                float(row.AvgTone),
                float(row.tone_norm),
                int(row.NumMentions),
                int(row.NumArticles),
                str(row.SOURCEURL) if row.SOURCEURL else None,
            )
            for identifier, row in zip(identifiers, working.itertuples(index=False))
        ]


def _period_expression(period: str | None) -> str:
    """Return the SQL expression labelling a date with its period.

    Quarters are computed arithmetically from the month substring so that the
    labels match the ``YYYYQn`` form pandas produces for
    :meth:`pandas.Series.dt.to_period`.

    Parameters
    ----------
    period : {"month", "quarter", "year"} or None
        Requested granularity.

    Returns
    -------
    str
        SQL expression, or the empty string when no period was requested.
    """
    if period == "month":
        return "substr(date, 1, 7)"
    if period == "year":
        return "substr(date, 1, 4)"
    if period == "quarter":
        return (
            "substr(date, 1, 4) || 'Q' || "
            "CAST((CAST(substr(date, 6, 2) AS INTEGER) + 2) / 3 AS TEXT)"
        )
    return ""


def _event_ids(frame: pd.DataFrame) -> list[int]:
    """Derive a stable primary key for each event row.

    GDELT's ``GLOBALEVENTID`` is used verbatim where present, which makes
    re-harvesting a day a no-op. Rows without one, notably synthetic events,
    are keyed by a digest over their content and ordinal position, so a
    repeated generator run with the same seed is equally idempotent while two
    genuinely distinct rows with identical content remain distinct.

    Parameters
    ----------
    frame : pandas.DataFrame
        Event frame carrying at least the actor, date and tone columns.

    Returns
    -------
    list of int
        One non-null identifier per row.
    """
    natural = (
        pd.to_numeric(frame["GLOBALEVENTID"], errors="coerce")
        if "GLOBALEVENTID" in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype="Float64")
    )

    identifiers: list[int] = []
    for position, (value, row) in enumerate(zip(natural, frame.itertuples(index=False))):
        if pd.notna(value):
            identifiers.append(int(value))
            continue
        payload = (
            f"{position}|{row.date}|{row.Actor1CountryCode}|{row.Actor2CountryCode}"
            f"|{row.EventRootCode}|{row.AvgTone}|{row.NumMentions}"
        )
        digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
        # Synthetic keys are negative so they can never collide with a real
        # GLOBALEVENTID, which GDELT allocates as a positive counter.
        identifiers.append(-(int.from_bytes(digest, "big") & _SYNTHETIC_ID_MASK) - 1)
    return identifiers


def is_read_only_sql(sql: str) -> bool:
    """Report whether a statement is a single read-only query.

    Used to guard the dashboard's SQL console. The check is deliberately
    conservative: a statement must be one ``SELECT`` or ``WITH`` with no
    embedded mutating keyword and no statement separator. It complements,
    rather than replaces, opening the connection read-only.

    Parameters
    ----------
    sql : str
        Candidate statement.

    Returns
    -------
    bool
        ``True`` when the statement is safe to run.
    """
    stripped = sql.strip().rstrip(";").strip()
    if not stripped or ";" in stripped:
        return False

    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return False

    tokens = {
        token.strip("(),")
        for token in lowered.replace("\n", " ").split()
    }
    return not (tokens & _MUTATING_KEYWORDS)


def open_store(
    path: str | Path = DEFAULT_DB_PATH,
    read_only: bool = False,
) -> EventStore:
    """Open a store and log a one-line summary of what it holds.

    Parameters
    ----------
    path : str or pathlib.Path, optional
        Database file.
    read_only : bool, optional
        Open without write access.

    Returns
    -------
    EventStore
        Connected store.
    """
    store = EventStore(path, read_only=read_only)
    stats = store.stats()
    _log.info(
        "%s Store %s | %s events | %s dyad rows | %s to %s",
        OK,
        store.path.name,
        f"{stats['events']:,}",
        f"{stats['dyad_rows']:,}",
        stats["first_date"] or "empty",
        stats["last_date"] or "empty",
    )
    return store
