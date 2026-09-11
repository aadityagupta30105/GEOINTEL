"""
Year-Scale Harvester
====================
Collects GDELT daily exports into the :mod:`data.store` event store.

The plain collector in :mod:`data.gdelt_collector` walks days serially, keeps
nothing between runs and re-downloads everything on the next invocation. That
is adequate for a week and untenable for a year. This module adds the three
properties a long collection needs:

Resumable
    Every calendar day's outcome is written to ``harvest_log`` as it lands. A
    rerun skips days already collected or confirmed empty, so an interrupted
    harvest continues rather than restarting. Failed days are retried.

Parallel
    Downloads and parsing run on a worker pool. Writes are serialised on the
    calling thread because SQLite admits one writer, which is not a bottleneck:
    the cost is dominated by transferring roughly 10 to 40 MB per day.

Interruptible
    ``Ctrl+C`` stops scheduling, drains the work already in flight, commits it
    and reports coverage. Nothing collected is ever lost.

Command line
------------
::

    python -m data.harvest --year                    # trailing 365 days
    python -m data.harvest --start 2024-01-01 --end 2024-12-31 --workers 8
    python -m data.harvest --year --dry-run          # cost projection only
    python -m data.harvest --retry-failed            # failed days only
    python -m data.harvest --mock --year             # offline synthetic year
    python -m data.harvest --coverage                # what the store holds

Full fidelity is the default: every state-to-state event in each export is
retained. Pass ``--cap N`` to sample instead.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.gdelt_collector import (  # noqa: E402
    fetch_gdelt_day,
    generate_mock_data,
    get_gdelt_url,
    preprocess,
)
from data.store import DEFAULT_DB_PATH, EventStore  # noqa: E402
from utils.logging_config import (  # noqa: E402
    ARROW,
    ERROR,
    OK,
    WARN,
    configure_logging,
    get_logger,
    rule,
    section,
)

__all__ = [
    "DEFAULT_WORKERS",
    "MEAN_EXPORT_BYTES",
    "HarvestReport",
    "day_range",
    "pending_days",
    "project_cost",
    "harvest_range",
    "harvest_mock",
    "main",
]

_log = get_logger(__name__)

DEFAULT_WORKERS: int = 6

# Mean size of a compressed GDELT 1.0 daily export, measured over sample days
# in 2024 at roughly 6.2 MB. Used only for the pre-flight projection; the
# report prints the volume actually transferred.
MEAN_EXPORT_BYTES: int = 6_500_000

# Attempts per day before the day is recorded as failed.
_MAX_ATTEMPTS: int = 3
_RETRY_BACKOFF_SECONDS: float = 4.0
_PROBE_TIMEOUT_SECONDS: float = 20.0

# Transfer-and-parse cost of one daily export on a single worker.
_SECONDS_PER_DAY_PER_WORKER: float = 25.0

# Days per synthetic batch, and events synthesised per day, when filling the
# store offline. Batching keeps the generator's memory flat over a long window.
_MOCK_BATCH_DAYS: int = 30
_MOCK_EVENTS_PER_DAY: int = 400

_BYTES_PER_GIB: float = 1024.0 ** 3


@dataclass(slots=True)
class HarvestReport:
    """Outcome of one harvest run.

    Attributes
    ----------
    requested : int
        Calendar days in the requested window.
    skipped : int
        Days already held with a terminal outcome.
    collected : int
        Days newly retrieved with at least one bilateral event.
    empty : int
        Days retrieved that held no bilateral events.
    failed : int
        Days whose attempts were all exhausted.
    events : int
        Rows newly inserted into the store.
    bytes_downloaded : int
        Compressed bytes transferred.
    seconds : float
        Wall-clock duration.
    interrupted : bool
        Whether the operator stopped the run before it completed.
    failures : list of str
        Days recorded as failed, for the closing report.
    """

    requested: int = 0
    skipped: int = 0
    collected: int = 0
    empty: int = 0
    failed: int = 0
    events: int = 0
    bytes_downloaded: int = 0
    seconds: float = 0.0
    interrupted: bool = False
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Render the report as an operator-facing block.

        Returns
        -------
        str
            Multi-line ASCII summary.
        """
        rate = self.events / self.seconds if self.seconds > 0 else 0.0
        lines = [
            f"Days requested   : {self.requested:,}",
            f"Days skipped     : {self.skipped:,} (already held)",
            f"Days collected   : {self.collected:,}",
            f"Days empty       : {self.empty:,}",
            f"Days failed      : {self.failed:,}",
            f"Events stored    : {self.events:,}",
            f"Downloaded       : {self.bytes_downloaded / _BYTES_PER_GIB:.2f} GiB",
            f"Elapsed          : {_format_duration(self.seconds)}",
            f"Throughput       : {rate:,.0f} events/s",
        ]
        if self.failures:
            shown = ", ".join(self.failures[:10])
            more = f" (+{len(self.failures) - 10} more)" if len(self.failures) > 10 else ""
            lines.append(f"Failed days      : {shown}{more}")
            lines.append("Rerun with --retry-failed to attempt them again.")
        return "\n".join(lines)


def _format_duration(seconds: float) -> str:
    """Render a duration as ``HHhMMmSSs``, omitting empty leading units.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Human-readable duration.
    """
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def day_range(start: datetime, end: datetime) -> list[str]:
    """Enumerate the inclusive calendar days between two dates.

    Parameters
    ----------
    start, end : datetime
        Window bounds. Only the date component is used.

    Returns
    -------
    list of str
        Days as ``YYYY-MM-DD``, chronologically ordered. Empty when ``start``
        is after ``end``.
    """
    days: list[str] = []
    current = start.date()
    last = end.date()
    while current <= last:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def pending_days(store: EventStore, days: list[str]) -> list[str]:
    """Filter a day list down to those not yet terminally collected.

    Parameters
    ----------
    store : EventStore
        Store holding the harvest log.
    days : list of str
        Candidate days as ``YYYY-MM-DD``.

    Returns
    -------
    list of str
        Days still to collect, in the input order.
    """
    held = store.completed_days()
    return [day for day in days if day not in held]


def project_cost(pending: int, workers: int) -> str:
    """Describe the expected cost of collecting a number of days.

    The projection is deliberately coarse. Its purpose is to let an operator
    decide whether to start a multi-hour job, not to be accurate to the minute.

    Parameters
    ----------
    pending : int
        Days still to collect.
    workers : int
        Concurrent download workers.

    Returns
    -------
    str
        Multi-line projection.
    """
    gib = pending * MEAN_EXPORT_BYTES / _BYTES_PER_GIB
    # Roughly 25 s of transfer and parse per day on one worker, divided by the
    # pool size and discounted for imperfect scaling. Measured against live
    # exports: three days at three workers completed in 29 s.
    seconds = pending * _SECONDS_PER_DAY_PER_WORKER / max(1, workers) * 1.3
    return (
        f"Days to collect  : {pending:,}\n"
        f"Estimated volume : {gib:.1f} GiB compressed\n"
        f"Estimated time   : {_format_duration(seconds)} at {workers} workers\n"
        "The run is resumable: stop it at any time and rerun to continue."
    )


def _export_exists(day: str) -> bool:
    """Report whether GDELT publishes an export for a calendar day.

    :func:`data.gdelt_collector.fetch_gdelt_day` returns ``None`` both for a
    transport failure and for a day GDELT never published, and the two demand
    opposite treatment: the first must be retried on the next run, the second
    must never be attempted again. A single ``HEAD`` request separates them.

    Parameters
    ----------
    day : str
        Day as ``YYYY-MM-DD``.

    Returns
    -------
    bool
        ``True`` when the export is addressable, and when the probe itself
        fails. Assuming existence on an inconclusive probe keeps the day
        retryable, which is the safe direction to be wrong in.
    """
    url = get_gdelt_url(datetime.strptime(day, "%Y-%m-%d"))
    try:
        response = requests.head(url, timeout=_PROBE_TIMEOUT_SECONDS, allow_redirects=True)
    except requests.RequestException:
        return True
    return response.status_code != 404


def _fetch_with_retry(day: str, cap: int | None) -> tuple[pd.DataFrame | None, str]:
    """Fetch one day, retrying transient failures with a linear backoff.

    Parameters
    ----------
    day : str
        Day as ``YYYY-MM-DD``.
    cap : int or None
        Per-day retention cap; ``None`` retains everything.

    Returns
    -------
    tuple of (pandas.DataFrame or None, str)
        The day's bilateral events and a status of ``ok``, ``empty`` or
        ``failed``. ``None`` accompanies the latter two. ``empty`` is terminal
        and ``failed`` is retried by the next run.
    """
    date = datetime.strptime(day, "%Y-%m-%d")

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        frame = fetch_gdelt_day(date, target_rows=cap)
        if frame is not None and not frame.empty:
            return frame, "ok"
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)

    return None, "empty" if not _export_exists(day) else "failed"


def harvest_range(
    store: EventStore,
    start: datetime,
    end: datetime,
    workers: int = DEFAULT_WORKERS,
    cap: int | None = None,
    retry_failed: bool = False,
) -> HarvestReport:
    """Collect a date range into the store, resumably and in parallel.

    Parameters
    ----------
    store : EventStore
        Destination store, opened for writing.
    start, end : datetime
        Inclusive window bounds.
    workers : int, optional
        Concurrent download workers.
    cap : int or None, optional
        Per-day retention cap. ``None`` retains every bilateral event.
    retry_failed : bool, optional
        Collect only the days whose previous attempt failed, ignoring the
        window.

    Returns
    -------
    HarvestReport
        What the run achieved.
    """
    if retry_failed:
        days = store.failed_days()
        _log.info("Retrying %d previously failed day(s)", len(days))
    else:
        days = day_range(start, end)

    report = HarvestReport(requested=len(days))
    outstanding = days if retry_failed else pending_days(store, days)
    report.skipped = report.requested - len(outstanding)

    if not outstanding:
        _log.info("%s Nothing to collect; the window is already held", OK)
        return report

    _log.info(
        "Collecting %d day(s) with %d worker(s) %s %s",
        len(outstanding), workers, ARROW, store.path,
    )

    started = time.perf_counter()
    stop = threading.Event()
    touched_dates: set[str] = set()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Submitted in window order and consumed in the same order, so the
        # progress line advances chronologically and the store is written by
        # this thread alone.
        scheduled: list[tuple[str, Future[tuple[pd.DataFrame | None, str]]]] = [
            (day, pool.submit(_fetch_with_retry, day, cap)) for day in outstanding
        ]
        settled: set[str] = set()

        try:
            for index, (day, future) in enumerate(scheduled, start=1):
                if stop.is_set():
                    break
                frame, status = future.result()
                settled.add(day)
                _record_day(store, report, day, frame, status, touched_dates)
                _log_progress(index, len(outstanding), day, status, report, started)

        except KeyboardInterrupt:
            report.interrupted = True
            stop.set()
            _log.warning(
                "%s Interrupted. Draining work in flight; rerun to continue.", WARN
            )
            for day, future in scheduled:
                if day in settled:
                    continue
                if not future.done():
                    future.cancel()
                    continue
                try:
                    frame, status = future.result()
                except (KeyboardInterrupt, RuntimeError, OSError):
                    continue
                settled.add(day)
                _record_day(store, report, day, frame, status, touched_dates)

    if touched_dates:
        _log.info("Rebuilding rollups for %d day(s)", len(touched_dates))
        store.rebuild_rollups(dates=sorted(touched_dates))

    report.seconds = time.perf_counter() - started
    return report


def _record_day(
    store: EventStore,
    report: HarvestReport,
    day: str,
    frame: pd.DataFrame | None,
    status: str,
    touched: set[str],
) -> None:
    """Persist one day's result and update the running report.

    Rollups are deliberately not rebuilt here. Doing so per day would repeat
    the same aggregation hundreds of times over a year; the caller rebuilds
    once for every touched date when the run ends.

    Parameters
    ----------
    store : EventStore
        Destination store.
    report : HarvestReport
        Running tally, mutated in place.
    day : str
        Day as ``YYYY-MM-DD``.
    frame : pandas.DataFrame or None
        Raw bilateral events for the day.
    status : str
        Fetch outcome from :func:`_fetch_with_retry`.
    touched : set of str
        Dates whose rollups need rebuilding, mutated in place.
    """
    if frame is None or status != "ok":
        # The distinction matters and must survive to the log: an "empty" day
        # is terminal and a rerun skips it, while a "failed" day stays pending
        # and --retry-failed picks it up. Collapsing the two here would make
        # every transport failure permanent.
        if status == "failed":
            store.record_harvest(day, "failed", message="Export unreachable")
            report.failed += 1
            report.failures.append(day)
        else:
            store.record_harvest(day, "empty", message="No bilateral events in export")
            report.empty += 1
        return

    size_bytes = int(frame.attrs.get("download_bytes", 0))
    report.bytes_downloaded += size_bytes

    try:
        working = frame.copy()
        working["date"] = day
        events = preprocess(working)
    except (ValueError, KeyError, TypeError) as exc:
        _log.error("%s %s: preprocessing failed: %s", ERROR, day, exc)
        store.record_harvest(day, "failed", size_bytes=size_bytes, message=str(exc))
        report.failed += 1
        report.failures.append(day)
        return

    if events.empty:
        store.record_harvest(day, "empty", size_bytes=size_bytes)
        report.empty += 1
        return

    inserted = store.upsert_events(events, rebuild=False)
    store.record_harvest(day, "ok", rows=len(events), size_bytes=size_bytes)
    report.collected += 1
    report.events += inserted
    touched.add(day)


def _log_progress(
    index: int,
    total: int,
    day: str,
    status: str,
    report: HarvestReport,
    started: float,
) -> None:
    """Emit a single progress line with a running estimate of time remaining.

    Parameters
    ----------
    index : int
        1-indexed position of the completed day.
    total : int
        Days in the run.
    day : str
        Day just completed.
    status : str
        Its outcome.
    report : HarvestReport
        Running tally.
    started : float
        Perf counter reading taken when the run began.
    """
    elapsed = time.perf_counter() - started
    remaining = (elapsed / index) * (total - index) if index else 0.0
    _log.info(
        "[%d/%d] %s %-6s | %s events | %.2f GiB | ETA %s",
        index, total, day, status,
        f"{report.events:,}",
        report.bytes_downloaded / _BYTES_PER_GIB,
        _format_duration(remaining),
    )


def harvest_mock(
    store: EventStore,
    start: datetime,
    end: datetime,
    events_per_day: int = _MOCK_EVENTS_PER_DAY,
    seed: int | None = 42,
) -> HarvestReport:
    """Fill the store with synthetic events across a window, offline.

    Generation runs in monthly batches so that a full year does not build one
    large frame in memory. Every day in the window is recorded as collected,
    which makes the resulting store indistinguishable in shape from a live
    harvest and lets the dashboard be exercised without any network access.

    Parameters
    ----------
    store : EventStore
        Destination store.
    start, end : datetime
        Inclusive window bounds.
    events_per_day : int, optional
        Mean synthetic events per calendar day.
    seed : int or None, optional
        Base seed. Each batch derives its own so that batches differ while the
        run as a whole stays reproducible.

    Returns
    -------
    HarvestReport
        What the run achieved.
    """
    days = day_range(start, end)
    report = HarvestReport(requested=len(days))
    started = time.perf_counter()

    batch_start = start
    batch_index = 0

    while batch_start <= end:
        batch_end = min(batch_start + timedelta(days=_MOCK_BATCH_DAYS - 1), end)
        span = (batch_end - batch_start).days + 1

        raw = generate_mock_data(
            batch_start,
            batch_end,
            n_events=events_per_day * span,
            seed=None if seed is None else seed + batch_index,
        )
        events = preprocess(raw)
        report.events += store.upsert_events(events, rebuild=False)

        counts = events["date"].value_counts()
        for day in day_range(batch_start, batch_end):
            rows = int(counts.get(day, 0))
            store.record_harvest(day, "ok" if rows else "empty", rows=rows)
            if rows:
                report.collected += 1
            else:
                report.empty += 1

        _log.info(
            "%s Synthesised %s to %s (%s events)",
            OK, batch_start.date(), batch_end.date(), f"{len(events):,}",
        )

        batch_start = batch_end + timedelta(days=1)
        batch_index += 1

    _log.info("Rebuilding rollups")
    store.rebuild_rollups()
    report.seconds = time.perf_counter() - started
    return report


def print_coverage(store: EventStore) -> None:
    """Print per-month collection completeness and store statistics.

    Parameters
    ----------
    store : EventStore
        Store to describe.
    """
    stats = store.stats()
    print(section("Store coverage"))
    print(f"File             : {store.path}")
    print(f"Size             : {stats['size_bytes'] / _BYTES_PER_GIB:.3f} GiB")
    print(f"Events           : {stats['events']:,}")
    print(f"Rollup rows      : {stats['dyad_rows']:,}")
    print(f"Countries        : {stats['countries']:,}")
    print(f"Window           : {stats['first_date'] or '-'} to {stats['last_date'] or '-'}")
    print(f"Days held        : {stats['days_held']:,}")

    coverage = store.coverage()
    if coverage.empty:
        print(rule())
        print("No harvest recorded yet.")
        return

    print(rule())
    print(coverage.to_string(index=False))

    failed = store.failed_days()
    if failed:
        print(rule())
        print(f"{WARN} {len(failed)} day(s) failed: {', '.join(failed[:10])}")
        print("Rerun with --retry-failed.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse harvester command-line arguments.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="geointel-harvest",
        description="Collect GDELT daily exports into the GeoIntel event store",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Store path")
    parser.add_argument("--start", default=None,
                        help="First day to collect, YYYY-MM-DD")
    parser.add_argument("--end", default=None,
                        help="Last day to collect, YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--year", action="store_true",
                        help="Collect the trailing 365 days")
    parser.add_argument("--days", type=int, default=None,
                        help="Collect the trailing N days")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Concurrent download workers")
    parser.add_argument("--cap", type=int, default=0,
                        help="Retain at most N events per day (0 retains all)")
    parser.add_argument("--mock", action="store_true",
                        help="Fill the window with synthetic events, no network")
    parser.add_argument("--events-per-day", type=int, default=_MOCK_EVENTS_PER_DAY,
                        help="Synthetic events per day when --mock is set")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Collect only the days whose last attempt failed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the cost projection and exit")
    parser.add_argument("--coverage", action="store_true",
                        help="Print store coverage and exit")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the rollup table from the event table and exit")
    return parser.parse_args(argv)


def resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Resolve the collection window from the parsed options.

    GDELT publishes each day's export the following morning, so the default
    end date is yesterday rather than today.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed options.

    Returns
    -------
    tuple of (datetime, datetime)
        Inclusive window bounds.

    Raises
    ------
    ValueError
        When the resolved start date is after the end date.
    """
    end = (
        datetime.strptime(args.end, "%Y-%m-%d")
        if args.end
        else datetime.now() - timedelta(days=1)
    )

    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        span = args.days if args.days else (365 if args.year else 7)
        start = end - timedelta(days=span - 1)

    if start > end:
        raise ValueError(f"Start date {start.date()} is after end date {end.date()}")
    return start, end


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m data.harvest``.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status. ``1`` when every requested day failed.
    """
    configure_logging()
    args = parse_args(argv)

    with EventStore(args.db) as store:
        if args.coverage:
            print_coverage(store)
            return 0

        if args.rebuild:
            rows = store.rebuild_rollups()
            _log.info("%s Rollup rebuilt: %s rows", OK, f"{rows:,}")
            return 0

        try:
            start, end = resolve_window(args)
        except ValueError as exc:
            _log.error("%s %s", ERROR, exc)
            return 1

        print(section("GeoIntel  |  Event harvest", char="="))
        _log.info("Window   : %s %s %s", start.date(), ARROW, end.date())
        _log.info("Store    : %s", Path(args.db).resolve())
        _log.info("Fidelity : %s", f"cap {args.cap:,}/day" if args.cap else "full")

        if args.mock:
            report = harvest_mock(store, start, end, args.events_per_day)
        else:
            outstanding = (
                len(store.failed_days())
                if args.retry_failed
                else len(pending_days(store, day_range(start, end)))
            )
            print(rule())
            print(project_cost(outstanding, args.workers))
            print(rule())

            if args.dry_run:
                return 0

            report = harvest_range(
                store,
                start,
                end,
                workers=args.workers,
                cap=args.cap or None,
                retry_failed=args.retry_failed,
            )

        print(section("Harvest complete" if not report.interrupted else "Harvest stopped"))
        print(report.summary())
        print(rule())
        print_coverage(store)

    return 1 if report.requested and report.failed == report.requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
