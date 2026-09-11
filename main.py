"""
Geopolitical Intelligence Pipeline
==================================
Command-line orchestrator for the GeoIntel platform.

Stages
------
1. Data collection (GDELT daily exports or the synthetic generator)
2. Preprocessing and normalisation
3. Graph construction
4. Network analysis
5. Temporal analysis
6. Bilateral analysis (optional)
7. Event classification (optional)
8. Narrative generation (optional)
9. Report export

Usage
-----
Full pipeline against synthetic data, no credentials required::

    python main.py

Against GDELT for an explicit window::

    python main.py --source gdelt --start 2024-01-01 --end 2024-03-31

With narrative generation and ML classification::

    python main.py --llm --classify

Dashboard only::

    python main.py --dashboard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from analysis.graph_builder import (  # noqa: E402
    NetworkStats,
    build_graph,
    build_temporal_graphs,
    compute_metrics,
    compute_network_stats,
    get_bilateral_summary,
)
from analysis.narrator import GeopoliticalNarrator  # noqa: E402
from data.gdelt_collector import (  # noqa: E402
    collect_gdelt_range,
    generate_mock_data,
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

_log = get_logger(__name__)

DEFAULT_WINDOW_DAYS = 90


@dataclass(slots=True)
class PipelineResult:
    """Artefacts produced by a completed pipeline run.

    Attributes
    ----------
    events : pandas.DataFrame
        Preprocessed event frame.
    graph : networkx.DiGraph
        Static interaction graph over the full window.
    metrics : pandas.DataFrame
        Node-level metrics indexed by country code.
    stats : NetworkStats
        Global topology statistics.
    output_dir : pathlib.Path
        Directory holding the exported artefacts.
    """

    events: pd.DataFrame
    graph: nx.DiGraph
    metrics: pd.DataFrame
    stats: NetworkStats
    output_dir: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

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
        prog="geointel",
        description="Geopolitical Intelligence Pipeline",
    )
    parser.add_argument("--source", choices=["gdelt", "mock", "store"], default="mock",
                        help="Event data source. 'store' reads the event store")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Event store path, used by --source store and --save-store")
    parser.add_argument("--save-store", action="store_true",
                        help="Also write the collected events into the event store")
    parser.add_argument("--max-rows", type=int, default=2_000_000,
                        help="Row cap when reading from the store")
    parser.add_argument("--start", default=None,
                        help="Start date YYYY-MM-DD (default: 90 days before end)")
    parser.add_argument("--end", default=None,
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--events", type=int, default=5000,
                        help="Number of synthetic events to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for the synthetic generator (reproducible runs)")
    parser.add_argument("--llm", action="store_true",
                        help="Enable narrative generation")
    parser.add_argument("--llm-provider", default="auto",
                        choices=["auto", "anthropic", "openai", "offline"],
                        help="Narrative provider")
    parser.add_argument("--classify", action="store_true",
                        help="Run ML event classification over the event stream")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch the Streamlit dashboard and exit")
    parser.add_argument("--output", default="output",
                        help="Output directory")
    parser.add_argument("--bilateral", nargs=2, metavar=("COUNTRY_A", "COUNTRY_B"),
                        help="Analyse a bilateral relationship, e.g. --bilateral USA CHN")
    parser.add_argument("--temporal", choices=["month", "quarter", "year"],
                        default="month", help="Temporal granularity")
    return parser.parse_args(argv)


def safe_save_csv(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Write a frame to CSV, falling back to a suffixed name when locked.

    On Windows a spreadsheet application may hold an exclusive lock on the
    target file, raising ``PermissionError``. The fallback appends a timestamp
    so the run still completes with a durable artefact.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to persist.
    path : str or pathlib.Path
        Preferred destination.
    **kwargs
        Forwarded to :meth:`pandas.DataFrame.to_csv`.

    Returns
    -------
    pathlib.Path
        The path actually written.
    """
    destination = Path(path)
    kwargs.setdefault("encoding", "utf-8")

    try:
        df.to_csv(destination, **kwargs)
        _log.info("%s Saved: %s", OK, destination)
        return destination
    except PermissionError:
        alternate = destination.with_stem(f"{destination.stem}_{int(time.time())}")
        _log.warning(
            "%s %s is locked by another process %s writing %s",
            WARN, destination.name, ARROW, alternate.name,
        )
        df.to_csv(alternate, **kwargs)
        _log.info("%s Saved: %s", OK, alternate)
        return alternate


def resolve_window(start: str | None, end: str | None) -> tuple[datetime, datetime]:
    """Resolve the analysis window from optional date strings.

    Parameters
    ----------
    start : str or None
        Start date as ``YYYY-MM-DD``. Defaults to
        :data:`DEFAULT_WINDOW_DAYS` before ``end``.
    end : str or None
        End date as ``YYYY-MM-DD``. Defaults to the current date.

    Returns
    -------
    tuple of (datetime, datetime)
        Inclusive window bounds.

    Raises
    ------
    ValueError
        When the resolved start date is after the end date.
    """
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()
    start_dt = (
        datetime.strptime(start, "%Y-%m-%d")
        if start
        else end_dt - timedelta(days=DEFAULT_WINDOW_DAYS)
    )

    if start_dt > end_dt:
        raise ValueError(
            f"Start date {start_dt.date()} is after end date {end_dt.date()}"
        )
    return start_dt, end_dt


def read_from_store(
    db_path: str | Path,
    start: datetime,
    end: datetime,
    max_rows: int,
) -> pd.DataFrame:
    """Read a window of stored events for analysis.

    Materialising events is the expensive way to use the store, and the
    pipeline needs it: the CSV export and the report are both event-level. The
    row cap guards against a wide window quietly pulling millions of rows into
    memory, and truncation is reported rather than silent.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Store path.
    start, end : datetime
        Inclusive window bounds.
    max_rows : int
        Row cap.

    Returns
    -------
    pandas.DataFrame
        Stored events, empty when the store is absent or the window matched
        nothing.
    """
    if not Path(db_path).exists():
        _log.error("%s No event store at %s", ERROR, db_path)
        _log.error("   Create one with: python -m data.harvest --year")
        return pd.DataFrame()

    window = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    with EventStore(db_path, read_only=True) as store:
        held = store.stats()
        _log.info(
            "Store    : %s events, %s to %s",
            f"{held['events']:,}", held["first_date"], held["last_date"],
        )
        matching = store.count_events(start=window[0], end=window[1])
        events = store.events(start=window[0], end=window[1], limit=max_rows)

    if matching > max_rows:
        # The read is thinned across the window rather than truncated, so the
        # analysis still covers the requested dates. Record that on the frame:
        # the report renders it, so a capped run cannot produce an artefact
        # that reads as a complete census of the window.
        events.attrs["sampled_from"] = matching
        events.attrs["window"] = window
        _log.warning(
            "%s Window holds %s events, above the %s row cap %s thinned to a "
            "%.1f%% sample spanning the full window",
            WARN, f"{matching:,}", f"{max_rows:,}", ARROW,
            100.0 * len(events) / matching,
        )
        _log.warning("   Raise --max-rows for a complete read.")

    _log.info("%s Read %s events from the store", OK, f"{len(events):,}")
    return events


def save_to_store(db_path: str | Path, events: pd.DataFrame) -> int:
    """Persist a run's events into the event store.

    Parameters
    ----------
    db_path : str or pathlib.Path
        Store path, created when absent.
    events : pandas.DataFrame
        Preprocessed events.

    Returns
    -------
    int
        Rows newly inserted. Zero when every row was already held.
    """
    with EventStore(db_path) as store:
        inserted = store.upsert_events(events)
    _log.info(
        "%s Store updated: %s new row(s) of %s (%s)",
        OK, f"{inserted:,}", f"{len(events):,}", db_path,
    )
    return inserted


def run_pipeline(args: argparse.Namespace) -> PipelineResult | None:
    """Execute the full analysis pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.

    Returns
    -------
    PipelineResult or None
        Run artefacts, or ``None`` when preprocessing yielded no events.
    """
    print(section("GeoIntel  |  Geopolitical Intelligence Pipeline", char="="))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        start, end = resolve_window(args.start, args.end)
    except ValueError as exc:
        _log.error("%s %s", ERROR, exc)
        return None

    _log.info("Window   : %s %s %s", start.date(), ARROW, end.date())
    _log.info("Source   : %s", args.source)
    _log.info("Output   : %s", output_dir.resolve())

    # --- Stage 1: collection ------------------------------------------------
    print(section("Stage 1: data collection"))
    if args.source == "store":
        # Reading from the store skips preprocessing: rows were normalised on
        # the way in, so a rerun over a stored window costs a query and no
        # network access at all.
        events = read_from_store(args.db, start, end, args.max_rows)
        print(section("Stage 2: preprocessing skipped (store rows are normalised)"))
    else:
        if args.source == "gdelt":
            raw_events = collect_gdelt_range(start, end)
        else:
            raw_events = generate_mock_data(
                start, end, n_events=args.events, seed=args.seed
            )

        # --- Stage 2: preprocessing ----------------------------------------
        print(section("Stage 2: preprocessing"))
        events = preprocess(raw_events)

        if args.save_store and not events.empty:
            save_to_store(args.db, events)

    if events.empty:
        _log.error("%s Preprocessing returned zero valid events.", ERROR)
        _log.error(
            "   The exports were retrieved but contain no rows with valid "
            "3-letter country codes in both actor fields."
        )
        _log.error("   Narrow the date range, or use --source mock to exercise the pipeline.")
        return None

    # --- Stage 3: classification (optional) --------------------------------
    if args.classify:
        print(section("Stage 3: event classification"))
        events = classify_events(events)
    else:
        _log.info("Stage 3: event classification skipped (enable with --classify)")

    events_path = safe_save_csv(events, output_dir / "events_clean.csv", index=False)
    _log.info("Event stream persisted: %s", events_path)

    # --- Stage 4: graph construction ---------------------------------------
    print(section("Stage 4: graph construction"))
    graph = build_graph(events)

    edge_records = [
        {
            "source": source,
            "target": target,
            **{key: value for key, value in data.items() if key != "event_types"},
        }
        for source, target, data in graph.edges(data=True)
    ]
    safe_save_csv(pd.DataFrame(edge_records), output_dir / "edges.csv", index=False)

    # --- Stage 5: network analysis -----------------------------------------
    print(section("Stage 5: network analysis"))
    metrics = compute_metrics(graph)
    stats = compute_network_stats(graph)

    safe_save_csv(metrics, output_dir / "country_metrics.csv")

    stats_path = output_dir / "network_stats.json"
    serialisable = {key: value for key, value in stats.items() if key != "community_map"}
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(serialisable, handle, indent=2)
    _log.info("%s Saved: %s", OK, stats_path)

    print(rule())
    print(metrics.head(10)[["pagerank", "betweenness", "conflict_ratio"]].to_string())
    print(rule())

    # --- Stage 6: temporal analysis ----------------------------------------
    print(section("Stage 6: temporal analysis"))
    temporal = build_temporal_graphs(events, period=args.temporal)
    temporal_frame = summarise_temporal(temporal)

    if not temporal_frame.empty:
        safe_save_csv(temporal_frame, output_dir / "temporal_metrics.csv", index=False)
        print(temporal_frame[["period", "avg_tone", "ggpi", "modularity"]].to_string(index=False))
    else:
        _log.warning("%s No temporal snapshots produced", WARN)

    # --- Stage 7: bilateral analysis ---------------------------------------
    if args.bilateral:
        country_a, country_b = args.bilateral
        print(section(f"Stage 7: bilateral analysis {country_a} {ARROW} {country_b}"))
        summary = get_bilateral_summary(graph, country_a, country_b)
        print(json.dumps(
            {
                "country_a": summary["country_a"],
                "country_b": summary["country_b"],
                "relationship_type": summary["relationship_type"],
                "dominant_tone": summary["dominant_tone"],
                "a_to_b_events": (summary["a_to_b"] or {}).get("num_events", 0),
                "b_to_a_events": (summary["b_to_a"] or {}).get("num_events", 0),
            },
            indent=2,
        ))

    # --- Stage 8: narratives -----------------------------------------------
    if args.llm:
        print(section("Stage 8: narrative generation"))
        generate_narratives(graph, metrics, stats, args, output_dir)

    # --- Stage 9: report ----------------------------------------------------
    print(section("Stage 9: report export"))
    generate_report(events, metrics, stats, output_dir)

    print(section("Pipeline complete", char="="))
    _log.info("%s Artefacts written to %s", OK, output_dir.resolve())

    return PipelineResult(
        events=events,
        graph=graph,
        metrics=metrics,
        stats=stats,
        output_dir=output_dir,
    )


def classify_events(events: pd.DataFrame) -> pd.DataFrame:
    """Attach ML event-type predictions to the event stream.

    The classifier is imported lazily so that a missing deep-learning stack
    degrades this stage rather than the whole pipeline.

    Parameters
    ----------
    events : pandas.DataFrame
        Preprocessed event frame.

    Returns
    -------
    pandas.DataFrame
        The frame with ``ml_event_type``, ``ml_confidence`` and ``ml_method``
        columns, or the input unchanged when classification is unavailable.
    """
    try:
        from models.event_classifier import GeopoliticalEventClassifier
    except ImportError as exc:
        _log.error("%s Event classifier unavailable: %s", ERROR, exc)
        return events

    classifier = GeopoliticalEventClassifier()
    classified = classifier.predict_dataframe(events)

    distribution = classified["ml_event_type"].value_counts()
    method = classified["ml_method"].iloc[0] if len(classified) else "none"
    _log.info("%s Classified %s events (method=%s)", OK, f"{len(classified):,}", method)
    for label, count in distribution.items():
        _log.info("   %-20s %s", label, f"{count:,}")

    return classified


def summarise_temporal(temporal: dict[str, nx.DiGraph]) -> pd.DataFrame:
    """Reduce temporal snapshots to a per-period statistics frame.

    Parameters
    ----------
    temporal : dict of str to networkx.DiGraph
        Snapshot graphs keyed by period label.

    Returns
    -------
    pandas.DataFrame
        One row per period, sorted chronologically by label.
    """
    if not temporal:
        return pd.DataFrame()

    records = []
    for period in sorted(temporal):
        snapshot_stats = compute_network_stats(temporal[period])
        records.append({
            "period": period,
            "nodes": snapshot_stats["nodes"],
            "edges": snapshot_stats["edges"],
            "avg_tone": snapshot_stats["avg_tone"],
            "neg_ratio": snapshot_stats["negative_edge_ratio"],
            "ggpi": snapshot_stats["ggpi"],
            "modularity": snapshot_stats["modularity"],
        })

    return pd.DataFrame(records)


def generate_narratives(
    graph: nx.DiGraph,
    metrics: pd.DataFrame,
    stats: NetworkStats,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate and persist narrative summaries.

    Parameters
    ----------
    graph : networkx.DiGraph
        Static interaction graph.
    metrics : pandas.DataFrame
        Node-level metrics.
    stats : NetworkStats
        Global topology statistics.
    args : argparse.Namespace
        Parsed command-line options.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    dict
        The persisted summary payload.
    """
    narrator = GeopoliticalNarrator(provider=args.llm_provider)
    summaries: dict[str, Any] = {}

    _log.info("Generating global executive summary")
    summaries["network"] = narrator.summarize_network(graph, stats, metrics)
    print(rule())
    print(summaries["network"])
    print(rule())

    _log.info("Generating country profiles (top 10)")
    summaries["countries"] = narrator.batch_summarize_countries(graph, metrics, top_n=10)

    if args.bilateral:
        country_a, country_b = args.bilateral
        key = f"bilateral_{country_a}_{country_b}"
        summaries[key] = narrator.summarize_bilateral(graph, country_a, country_b)
        print(rule())
        print(f"{country_a} {ARROW} {country_b}")
        print(summaries[key])
        print(rule())

    narrator.save_summaries(summaries, output_dir / "summaries.json")
    return summaries


def _sampling_note(events: pd.DataFrame) -> list[str]:
    """Render the provenance line for a capped read, if there was one.

    A report drawn from a thinned sample must say so. Without this the
    document states an event count and a date range that read as a complete
    census of the window, and nothing in the artefact contradicts that.

    Parameters
    ----------
    events : pandas.DataFrame
        Event frame, possibly carrying ``sampled_from`` in
        :attr:`pandas.DataFrame.attrs`.

    Returns
    -------
    list of str
        One Markdown line when the frame was sampled, otherwise empty.
    """
    population = events.attrs.get("sampled_from")
    if not population:
        return []

    share = 100.0 * len(events) / population
    return [
        f"- **Sampling**: {share:.1f}% sample of {population:,} events "
        f"spanning the full window. Counts below are scaled accordingly; "
        f"rates and rankings are not. Raise `--max-rows` for a complete read."
    ]


def generate_report(
    events: pd.DataFrame,
    metrics: pd.DataFrame,
    stats: NetworkStats,
    output_dir: Path,
) -> Path:
    """Render the Markdown intelligence report.

    Parameters
    ----------
    events : pandas.DataFrame
        Preprocessed event frame.
    metrics : pandas.DataFrame
        Node-level metrics.
    stats : NetworkStats
        Global topology statistics.
    output_dir : pathlib.Path
        Destination directory.

    Returns
    -------
    pathlib.Path
        Path of the written report.
    """
    lines: list[str] = [
        "# Geopolitical Intelligence Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Dataset Summary",
        f"- **Total events**: {len(events):,}",
        f"- **Countries**: {events['Actor1CountryCode'].nunique()}",
        f"- **Date range**: {events['date'].min()} to {events['date'].max()}",
        *_sampling_note(events),
        "",
        "## Network Statistics",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Nodes | {stats['nodes']} |",
        f"| Edges | {stats['edges']} |",
        f"| Density | {stats['density']:.4f} |",
        f"| Avg Sentiment | {stats['avg_tone']:+.4f} |",
        f"| Conflict Rate | {stats['negative_edge_ratio']:.1%} |",
        f"| Reciprocity | {stats['reciprocity']:.4f} |",
        f"| Modularity | {stats['modularity']:.4f} |",
        f"| Geopolitical Blocs | {stats['num_communities']} |",
        f"| **GGPI** | **{stats['ggpi']:.4f}** |",
        "",
        "## Top 10 Most Influential Countries",
        "| Country | PageRank | Betweenness | Conflict% | Events |",
        "|---------|----------|-------------|-----------|--------|",
    ]

    for country, row in metrics.head(10).iterrows():
        lines.append(
            f"| {country} | {row['pagerank']:.6f} | {row['betweenness']:.6f} | "
            f"{row['conflict_ratio']:.1%} | {int(row['total_events'])} |"
        )

    lines += ["", "## Event Type Distribution"]
    for event_type, count in events["event_type"].value_counts().items():
        lines.append(f"- **{event_type}**: {count:,} ({count / len(events):.1%})")

    if "ml_event_type" in events.columns:
        lines += ["", "## ML Classification Distribution"]
        for event_type, count in events["ml_event_type"].value_counts().items():
            lines.append(f"- **{event_type}**: {count:,} ({count / len(events):.1%})")

    lines += [
        "",
        "## Global Geopolitical Polarization Index",
        "",
        f"**GGPI = {stats['ggpi']:.4f}** (scale 0 to 1; higher indicates greater polarization)",
        "",
        "Computed as:",
        f"- 40% x Network Modularity ({stats['modularity']:.4f})",
        f"- 40% x Negative Edge Ratio ({stats['negative_edge_ratio']:.4f})",
        f"- 20% x Negative Average Tone ({max(0.0, -stats['avg_tone']):.4f})",
        "",
        "---",
        "Generated by the GeoIntel pipeline.",
        "",
    ]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("%s Report written: %s", OK, report_path)
    return report_path


def launch_dashboard() -> int:
    """Launch the Streamlit dashboard.

    Invokes the console script first, then ``python -m streamlit`` so that the
    dashboard starts even when the entry point is absent from ``PATH``.

    Returns
    -------
    int
        Process exit status, or ``1`` when Streamlit could not be launched.
    """
    dashboard_path = ROOT / "dashboard" / "app.py"
    _log.info("Launching dashboard: %s", dashboard_path)
    _log.info("Open http://localhost:8501 once the server reports ready")

    for command in (
        ["streamlit", "run", str(dashboard_path)],
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
    ):
        try:
            completed = subprocess.run(command, check=False)
            return completed.returncode
        except FileNotFoundError:
            continue

    _log.error("%s Streamlit could not be launched. Install it with:", ERROR)
    _log.error("   pip install streamlit")
    _log.error("Then run: python -m streamlit run %s", dashboard_path)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Parameters
    ----------
    argv : list of str or None, optional
        Argument vector. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    configure_logging()
    args = parse_args(argv)

    if args.dashboard:
        return launch_dashboard()

    return 0 if run_pipeline(args) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
