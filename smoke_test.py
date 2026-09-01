"""
GeoIntel Smoke Test
===================
End-to-end operational check. Exercises every subsystem against the running
interpreter and reports a single pass or fail verdict.

Unlike the pytest suite, which isolates units, this script verifies that the
platform works as an installed whole: imports resolve, the pipeline produces
artefacts on disk, the dashboard module loads under a real Streamlit runtime,
and every documented fallback engages without credentials or network access.

Usage
-----
    python smoke_test.py
    python smoke_test.py --keep    # retain the temporary output directory

Exit status is 0 when every check passes and 1 otherwise, so the script is
usable as a deployment gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.logging_config import ERROR, OK, WARN, configure_logging, rule, section

_BANNER_WIDTH = 68


@dataclass(slots=True)
class CheckResult:
    """Outcome of a single smoke check.

    Attributes
    ----------
    name : str
        Human-readable check name.
    passed : bool
        Whether the check succeeded.
    detail : str
        One-line summary shown in the report.
    duration : float
        Wall-clock seconds spent on the check.
    error : str
        Formatted traceback when the check raised.
    """

    name: str
    passed: bool
    detail: str = ""
    duration: float = 0.0
    error: str = field(default="", repr=False)


class SmokeTest:
    """Runs the smoke checks and accumulates their results.

    Parameters
    ----------
    workdir : pathlib.Path
        Scratch directory for generated artefacts.
    """

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.results: list[CheckResult] = []

    # --- Harness ------------------------------------------------------------

    def run(self, name: str, check: Callable[[], str]) -> None:
        """Execute one check and record its result.

        Parameters
        ----------
        name : str
            Check name.
        check : callable
            Zero-argument callable returning a one-line detail string. Any
            exception is recorded as a failure.
        """
        started = time.perf_counter()
        try:
            detail = check()
            elapsed = time.perf_counter() - started
            self.results.append(CheckResult(name, True, detail, elapsed))
            print(f"  {OK:<8} {name:<44} {detail}")
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.results.append(
                CheckResult(name, False, str(exc), elapsed, traceback.format_exc())
            )
            print(f"  {ERROR:<8} {name:<44} {type(exc).__name__}: {exc}")

    # --- Checks -------------------------------------------------------------

    def check_imports(self) -> str:
        """Import every first-party module.

        Returns
        -------
        str
            Count of modules imported.
        """
        modules = [
            "utils.logging_config",
            "data.gdelt_collector",
            "analysis.graph_builder",
            "analysis.narrator",
            "models.event_classifier",
            "dashboard.theme",
            "dashboard.geodata",
            "dashboard.blocs",
            "dashboard.figures",
            "main",
        ]
        for module in modules:
            importlib.import_module(module)
        return f"{len(modules)} modules"

    def check_classifier_import_is_lazy(self) -> str:
        """Confirm importing the classifier does not initialise PyTorch.

        Returns
        -------
        str
            Confirmation detail.
        """
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, models.event_classifier; print('torch' in sys.modules)"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        if result.stdout.strip().splitlines()[-1] != "False":
            raise AssertionError("importing the classifier loaded torch")
        return "torch not loaded at import"

    def check_synthetic_generator(self) -> str:
        """Generate and preprocess synthetic events.

        Returns
        -------
        str
            Event and country counts.
        """
        from data.gdelt_collector import generate_mock_data, preprocess

        end = datetime(2024, 6, 30)
        raw = generate_mock_data(end - timedelta(days=120), end, 3000, seed=99)
        events = preprocess(raw)

        if events.empty:
            raise AssertionError("preprocessing produced no events")
        if not events["tone_norm"].between(-1, 1).all():
            raise AssertionError("tone_norm outside [-1, 1]")

        self._events = events
        return f"{len(events):,} events, {events['Actor1CountryCode'].nunique()} countries"

    def check_gdelt_fallback(self) -> str:
        """Confirm an unreachable GDELT endpoint degrades to synthetic data.

        Returns
        -------
        str
            Fallback confirmation.
        """
        import requests

        from data import gdelt_collector

        original = requests.get

        def unreachable(*args: object, **kwargs: object) -> None:
            raise requests.ConnectionError("smoke test: network disabled")

        requests.get = unreachable  # type: ignore[assignment]
        try:
            frame = gdelt_collector.collect_gdelt_range(
                datetime(2024, 1, 1), datetime(2024, 1, 2)
            )
        finally:
            requests.get = original  # type: ignore[assignment]

        if frame.empty:
            raise AssertionError("fallback returned no events")
        return f"degraded to synthetic ({len(frame):,} events)"

    def check_graph_analytics(self) -> str:
        """Build the graph and validate the metric and GGPI contracts.

        Returns
        -------
        str
            Graph size and GGPI.
        """
        from analysis.graph_builder import (
            GGPI_MODULARITY_WEIGHT,
            GGPI_NEGATIVE_EDGE_WEIGHT,
            GGPI_NEGATIVE_TONE_WEIGHT,
            build_graph,
            compute_metrics,
            compute_network_stats,
        )

        graph = build_graph(self._events)
        metrics = compute_metrics(graph)
        stats = compute_network_stats(graph)

        if not metrics["pagerank"].is_monotonic_decreasing:
            raise AssertionError("metrics not sorted by PageRank")
        if not 0.0 <= stats["ggpi"] <= 1.0:
            raise AssertionError(f"GGPI out of range: {stats['ggpi']}")

        expected = (
            GGPI_MODULARITY_WEIGHT * stats["modularity"]
            + GGPI_NEGATIVE_EDGE_WEIGHT * stats["negative_edge_ratio"]
            + GGPI_NEGATIVE_TONE_WEIGHT * max(0.0, -stats["avg_tone"])
        )
        if abs(expected - stats["ggpi"]) > 5e-4:
            raise AssertionError("GGPI does not match its documented formula")

        self._graph, self._metrics, self._stats = graph, metrics, stats
        return f"{stats['nodes']} nodes, {stats['edges']:,} edges, GGPI {stats['ggpi']:.4f}"

    def check_degenerate_graphs(self) -> str:
        """Confirm empty and single-node graphs do not raise.

        Returns
        -------
        str
            Confirmation detail.
        """
        import networkx as nx

        from analysis.graph_builder import compute_metrics, compute_network_stats

        empty_stats = compute_network_stats(nx.DiGraph())
        if empty_stats["ggpi"] != 0.0 or not compute_metrics(nx.DiGraph()).empty:
            raise AssertionError("empty graph handling is incorrect")

        single = nx.DiGraph()
        single.add_node("USA")
        compute_network_stats(single)
        return "empty and single-node graphs handled"

    def check_temporal_slicing(self) -> str:
        """Slice the event stream and confirm the slices partition it.

        Returns
        -------
        str
            Snapshot count.
        """
        from analysis.graph_builder import build_temporal_graphs

        snapshots = build_temporal_graphs(self._events, period="month")
        if len(snapshots) < 2:
            raise AssertionError("expected at least two monthly snapshots")

        total = sum(
            data["num_events"]
            for graph in snapshots.values()
            for _, _, data in graph.edges(data=True)
        )
        if total != len(self._events):
            raise AssertionError(
                f"snapshots cover {total} events, expected {len(self._events)}"
            )
        return f"{len(snapshots)} snapshots partition the stream"

    def check_classifier_fallback(self) -> str:
        """Classify events without a checkpoint.

        Returns
        -------
        str
            Method and label count.
        """
        from models.event_classifier import GeopoliticalEventClassifier, TrainingConfig

        classifier = GeopoliticalEventClassifier(
            TrainingConfig(save_path=str(self.workdir / "no-checkpoint"))
        )
        classified = classifier.predict_dataframe(self._events.head(200))

        if "ml_event_type" not in classified.columns:
            raise AssertionError("classification columns missing")
        if classified["ml_method"].iloc[0] != "rule_based":
            raise AssertionError("expected the keyword fallback")
        return f"{classified['ml_event_type'].nunique()} labels via rule_based"

    def check_narrator_offline(self) -> str:
        """Generate narratives without credentials and verify the figures.

        Returns
        -------
        str
            Narrative length.
        """
        import os

        from analysis.narrator import GeopoliticalNarrator

        saved = {
            key: os.environ.pop(key, None)
            for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        }
        try:
            narrator = GeopoliticalNarrator(provider="auto")
            if narrator.llm.provider != "offline":
                raise AssertionError("expected the offline provider")

            summary = narrator.summarize_network(self._graph, self._stats, self._metrics)
            if f"{self._stats['ggpi']:.3f}" not in summary:
                raise AssertionError("narrative does not report the computed GGPI")
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value

        return f"offline narrative, {len(summary)} chars, GGPI quoted"

    def check_figures(self) -> str:
        """Build every dashboard figure.

        Returns
        -------
        str
            Figure count.
        """
        from dashboard.blocs import compute_blocs
        from dashboard.figures import (
            build_bilateral_chart,
            build_bloc_figure,
            build_network_figure,
            build_pagerank_bar,
            build_radar_chart,
            build_tone_heatmap,
        )

        poles = [p for p in ("USA", "CHN") if p in self._graph]
        assignments, affinity = compute_blocs(self._graph, poles)

        countries = list(self._graph.nodes())[:2]
        figures = [
            build_network_figure(self._graph, self._metrics, self._stats, 1, 25, "PageRank"),
            build_bloc_figure(self._graph, self._metrics, assignments, affinity, poles, 1),
            build_radar_chart(self._metrics, self._metrics.index[0]),
            build_bilateral_chart(self._graph, countries[0], countries[1]),
            build_tone_heatmap(self._graph, self._metrics, top_n=10),
            build_pagerank_bar(self._metrics, top_n=10),
        ]
        if any(figure is None for figure in figures):
            raise AssertionError("a figure builder returned None")
        return f"{len(figures)} figures built"

    def check_pipeline_cli(self) -> str:
        """Run the pipeline as a subprocess and verify its artefacts.

        Returns
        -------
        str
            Artefact count and GGPI from disk.
        """
        output = self.workdir / "pipeline"
        result = subprocess.run(
            [
                sys.executable, "main.py",
                "--events", "2000", "--seed", "7",
                "--start", "2024-01-01", "--end", "2024-03-31",
                "--classify", "--llm", "--llm-provider", "offline",
                "--bilateral", "USA", "CHN",
                "--output", str(output),
            ],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"pipeline exited {result.returncode}: {result.stderr[-400:]}"
            )

        expected = [
            "events_clean.csv", "edges.csv", "country_metrics.csv",
            "temporal_metrics.csv", "network_stats.json", "summaries.json",
            "report.md",
        ]
        missing = [name for name in expected if not (output / name).exists()]
        if missing:
            raise AssertionError(f"missing artefacts: {', '.join(missing)}")

        stats = json.loads((output / "network_stats.json").read_text(encoding="utf-8"))
        if "ggpi" not in stats:
            raise AssertionError("network_stats.json has no ggpi field")

        self._pipeline_output = output
        return f"{len(expected)} artefacts, GGPI {stats['ggpi']:.4f}"

    def check_artefacts_are_ascii(self) -> str:
        """Confirm no emoji reached the generated artefacts.

        Returns
        -------
        str
            Confirmation detail.
        """
        import unicodedata

        allowed = {chr(code) for code in range(0x2500, 0x2580)}
        offenders: list[str] = []

        for path in sorted(self._pipeline_output.glob("*")):
            if path.suffix not in {".md", ".csv", ".json"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for character in text:
                if character.isascii() or character in allowed:
                    continue
                if unicodedata.category(character) in {"So", "Sk"}:
                    offenders.append(f"{path.name} U+{ord(character):04X}")

        if offenders:
            raise AssertionError(f"emoji found: {', '.join(offenders[:5])}")
        return "no emoji in generated artefacts"

    def check_dashboard_boots(self) -> str:
        """Start the dashboard under Streamlit and confirm it serves.

        Returns
        -------
        str
            Port and startup time.
        """
        port = 8599
        log_path = self.workdir / "streamlit.log"

        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
                    "--server.headless", "true",
                    "--server.port", str(port),
                    "--browser.gatherUsageStats", "false",
                ],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            )

        try:
            deadline = time.time() + 90
            while time.time() < deadline:
                if process.poll() is not None:
                    raise AssertionError(
                        f"streamlit exited {process.returncode}: "
                        f"{log_path.read_text(encoding='utf-8')[-400:]}"
                    )
                contents = log_path.read_text(encoding="utf-8", errors="replace")
                if "You can now view" in contents:
                    break
                if "Traceback" in contents:
                    raise AssertionError(f"startup error: {contents[-400:]}")
                time.sleep(1)
            else:
                raise AssertionError("streamlit did not report ready within 90s")

            import urllib.request

            with urllib.request.urlopen(
                f"http://localhost:{port}/healthz", timeout=15
            ) as response:
                if response.status != 200:
                    raise AssertionError(f"healthz returned {response.status}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()

        return f"served on port {port}, healthz 200"

    # --- Report -------------------------------------------------------------

    def report(self) -> bool:
        """Print the summary table.

        Returns
        -------
        bool
            ``True`` when every check passed.
        """
        passed = sum(1 for result in self.results if result.passed)
        failed = len(self.results) - passed
        elapsed = sum(result.duration for result in self.results)

        print()
        print(rule("=", _BANNER_WIDTH))
        print(f"SMOKE TEST: {passed} passed, {failed} failed, {elapsed:.1f}s")
        print(rule("=", _BANNER_WIDTH))

        if failed:
            print()
            for result in self.results:
                if not result.passed:
                    print(f"{ERROR} {result.name}")
                    print(result.error)
        return failed == 0


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
    parser = argparse.ArgumentParser(description="GeoIntel end-to-end smoke test")
    parser.add_argument(
        "--keep", action="store_true", help="retain the temporary output directory"
    )
    args = parser.parse_args(argv)

    configure_logging()
    workdir = Path(tempfile.mkdtemp(prefix="geointel-smoke-"))

    print(section("GeoIntel  |  Smoke Test", char="=", width=_BANNER_WIDTH))
    print(f"Interpreter : {sys.executable}")
    print(f"Version     : {sys.version.split()[0]}")
    print(f"Workdir     : {workdir}")
    print(rule("-", _BANNER_WIDTH))

    suite = SmokeTest(workdir)
    checks: list[tuple[str, Callable[[], str]]] = [
        ("Module imports", suite.check_imports),
        ("Classifier import is lazy", suite.check_classifier_import_is_lazy),
        ("Synthetic generator", suite.check_synthetic_generator),
        ("GDELT network fallback", suite.check_gdelt_fallback),
        ("Graph analytics and GGPI", suite.check_graph_analytics),
        ("Degenerate graph handling", suite.check_degenerate_graphs),
        ("Temporal slicing", suite.check_temporal_slicing),
        ("Classifier fallback", suite.check_classifier_fallback),
        ("Offline narrative generation", suite.check_narrator_offline),
        ("Dashboard figure builders", suite.check_figures),
        ("Pipeline CLI end to end", suite.check_pipeline_cli),
        ("Artefacts are emoji-free", suite.check_artefacts_are_ascii),
        ("Dashboard boots under Streamlit", suite.check_dashboard_boots),
    ]

    for name, check in checks:
        suite.run(name, check)

    succeeded = suite.report()

    if args.keep:
        print(f"\n{WARN} Workdir retained: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0 if succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
