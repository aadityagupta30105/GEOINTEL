"""Tests for narrative context assembly, provider resolution and fallbacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import pytest

from analysis.graph_builder import build_graph, compute_metrics, compute_network_stats
from analysis.narrator import (
    BILATERAL_PROMPT,
    COUNTRY_PROFILE_PROMPT,
    NETWORK_SUMMARY_PROMPT,
    GeopoliticalNarrator,
    LLMClient,
    OfflineNarrator,
)


@pytest.fixture(autouse=True)
def clear_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider credentials so tests never reach a live API."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


class TestProviderResolution:
    """Credential-driven provider selection."""

    def test_auto_without_credentials_resolves_offline(self) -> None:
        client = LLMClient(provider="auto")
        assert client.provider == "offline"
        assert client.is_offline

    def test_explicit_offline_is_honoured(self) -> None:
        assert LLMClient(provider="offline").provider == "offline"

    def test_anthropic_without_key_degrades_to_offline(self) -> None:
        assert LLMClient(provider="anthropic").provider == "offline"

    def test_openai_without_key_degrades_to_offline(self) -> None:
        assert LLMClient(provider="openai").provider == "offline"

    def test_offline_client_uses_the_supplied_fallback(self) -> None:
        client = LLMClient(provider="offline")
        assert client.complete("ignored", fallback=lambda: "FALLBACK") == "FALLBACK"

    def test_provider_exception_degrades_to_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A runtime API failure must not propagate into the caller."""
        client = LLMClient(provider="offline")

        class _Exploding:
            @property
            def messages(self) -> Any:
                raise RuntimeError("upstream unavailable")

        monkeypatch.setattr(client, "provider", "anthropic")
        monkeypatch.setattr(client, "_client", _Exploding())

        assert client.complete("prompt", fallback=lambda: "RECOVERED") == "RECOVERED"


class TestContextAssembly:
    """Facts extracted from the graph must match the graph."""

    def test_bilateral_context_is_numerically_faithful(
        self, tiny_events: pd.DataFrame
    ) -> None:
        graph = build_graph(tiny_events)
        context = GeopoliticalNarrator.build_bilateral_context(graph, "USA", "CHN")

        assert context.total_events == 4
        assert context.conflict_ratio == pytest.approx(2 / 4)
        assert context.coop_ratio == pytest.approx(2 / 4)
        assert context.avg_tone == pytest.approx(
            (graph["USA"]["CHN"]["tone"] + graph["CHN"]["USA"]["tone"]) / 2
        )

    def test_bilateral_context_for_absent_dyad(self, graph: nx.DiGraph) -> None:
        context = GeopoliticalNarrator.build_bilateral_context(graph, "ZZZ", "QQQ")
        assert context.total_events == 0
        assert context.avg_tone == 0.0
        assert context.types_a_to_b == "no direct interactions recorded"

    def test_country_context_reports_frame_values(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        country = metrics.index[0]
        context = GeopoliticalNarrator.build_country_context(graph, country, metrics)

        assert context.country == country
        assert context.pagerank == pytest.approx(metrics.loc[country, "pagerank"])
        assert context.total_events == int(metrics.loc[country, "total_events"])
        assert context.top_partners

    def test_network_context_reports_stats_values(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        stats = compute_network_stats(graph)
        context = GeopoliticalNarrator.build_network_context(graph, stats, metrics)

        assert context.nodes == stats["nodes"]
        assert context.ggpi == stats["ggpi"]
        assert context.top5.split(", ")[0] == metrics.index[0]


class TestPromptRendering:
    """Prompt templates must accept exactly the context fields."""

    def test_bilateral_prompt_renders(self, tiny_events: pd.DataFrame) -> None:
        context = GeopoliticalNarrator.build_bilateral_context(
            build_graph(tiny_events), "USA", "CHN"
        )
        rendered = BILATERAL_PROMPT.format(**context.as_prompt_fields())
        assert "USA" in rendered and "CHN" in rendered

    def test_country_prompt_renders(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        context = GeopoliticalNarrator.build_country_context(
            graph, metrics.index[0], metrics
        )
        assert metrics.index[0] in COUNTRY_PROFILE_PROMPT.format(
            **context.as_prompt_fields()
        )

    def test_network_prompt_renders(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        context = GeopoliticalNarrator.build_network_context(
            graph, compute_network_stats(graph), metrics
        )
        assert "GGPI" in NETWORK_SUMMARY_PROMPT.format(**context.as_prompt_fields())


class TestOfflineNarratives:
    """The offline generator must report the real figures."""

    def test_bilateral_narrative_quotes_the_actual_tone(
        self, tiny_events: pd.DataFrame
    ) -> None:
        """Regression: the previous implementation always reported 0.000."""
        graph = build_graph(tiny_events)
        context = GeopoliticalNarrator.build_bilateral_context(graph, "USA", "CHN")
        narrative = OfflineNarrator.bilateral(context)

        assert f"{context.avg_tone:+.3f}" in narrative
        assert "+0.000" not in narrative
        assert str(context.total_events) in narrative

    def test_bilateral_narrative_for_absent_dyad(self, graph: nx.DiGraph) -> None:
        context = GeopoliticalNarrator.build_bilateral_context(graph, "ZZZ", "QQQ")
        assert "No direct interactions" in OfflineNarrator.bilateral(context)

    def test_country_narrative_quotes_metrics(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        country = metrics.index[0]
        context = GeopoliticalNarrator.build_country_context(graph, country, metrics)
        narrative = OfflineNarrator.country(context)

        assert country in narrative
        assert f"{context.pagerank:.4f}" in narrative

    def test_network_narrative_quotes_ggpi(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        stats = compute_network_stats(graph)
        context = GeopoliticalNarrator.build_network_context(graph, stats, metrics)
        narrative = OfflineNarrator.network(context)

        assert f"{stats['ggpi']:.3f}" in narrative
        assert str(stats["nodes"]) in narrative


class TestNarratorIntegration:
    """End-to-end behaviour without credentials."""

    def test_summaries_are_generated_offline(self, graph: nx.DiGraph) -> None:
        metrics = compute_metrics(graph)
        stats = compute_network_stats(graph)
        narrator = GeopoliticalNarrator(provider="offline")

        assert len(narrator.summarize_network(graph, stats, metrics)) > 100
        assert len(narrator.summarize_country(graph, metrics.index[0], metrics)) > 100

    def test_unknown_country_reports_insufficient_data(self, graph: nx.DiGraph) -> None:
        narrator = GeopoliticalNarrator(provider="offline")
        result = narrator.summarize_country(graph, "ZZZ", compute_metrics(graph))
        assert "Insufficient data" in result

    def test_results_are_memoised(self, graph: nx.DiGraph) -> None:
        narrator = GeopoliticalNarrator(provider="offline")
        first = narrator.summarize_bilateral(graph, "USA", "CHN")
        assert narrator.summarize_bilateral(graph, "USA", "CHN") is first

    def test_batch_profiles_cover_the_requested_countries(
        self, graph: nx.DiGraph
    ) -> None:
        metrics = compute_metrics(graph)
        narrator = GeopoliticalNarrator(provider="offline")
        summaries = narrator.batch_summarize_countries(graph, metrics, top_n=3)
        assert list(summaries) == metrics.head(3).index.tolist()


class TestPersistence:
    """Summary round-tripping."""

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        narrator = GeopoliticalNarrator(provider="offline")
        destination = tmp_path / "nested" / "summaries.json"
        narrator.save_summaries({"network": "text", "countries": {"USA": "x"}}, destination)

        assert destination.exists()
        assert json.loads(destination.read_text(encoding="utf-8"))["network"] == "text"

        loaded = GeopoliticalNarrator(provider="offline").load_summaries(destination)
        assert loaded["network"] == "text"

    def test_load_missing_file_is_not_an_error(self, tmp_path: Path) -> None:
        narrator = GeopoliticalNarrator(provider="offline")
        assert narrator.load_summaries(tmp_path / "absent.json") == {}

    def test_load_corrupt_file_is_not_an_error(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not json", encoding="utf-8")
        assert GeopoliticalNarrator(provider="offline").load_summaries(corrupt) == {}
