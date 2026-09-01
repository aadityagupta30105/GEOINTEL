"""
LLM Narrative Generation
========================
Generates intelligence-style prose from quantitative network output.

Three execution modes are supported and resolved automatically:

- Anthropic Claude, when ``ANTHROPIC_API_KEY`` is present.
- OpenAI, when ``OPENAI_API_KEY`` is present.
- A deterministic offline generator requiring no credentials.

Design note
-----------
Narrative inputs are modelled as explicit context dataclasses. Both the LLM
path and the offline path consume the same structures: the LLM path renders
them into a prompt template, the offline path renders them into a fixed
narrative. No component recovers facts by parsing a prompt back into fields,
which keeps the offline output numerically faithful to the graph.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import networkx as nx
import numpy as np
import pandas as pd

from analysis.graph_builder import NetworkStats
from utils.logging_config import ERROR, OK, WARN, get_logger

__all__ = [
    "BILATERAL_PROMPT",
    "COUNTRY_PROFILE_PROMPT",
    "NETWORK_SUMMARY_PROMPT",
    "BilateralContext",
    "CountryContext",
    "NetworkContext",
    "LLMClient",
    "GeopoliticalNarrator",
]

_log = get_logger(__name__)

Provider = Literal["auto", "anthropic", "openai", "offline"]

# Model identifiers. Overridable so deployments can pin a specific revision.
DEFAULT_ANTHROPIC_MODEL: Final[str] = os.getenv(
    "GEOINTEL_ANTHROPIC_MODEL", "claude-opus-5"
)
DEFAULT_OPENAI_MODEL: Final[str] = os.getenv("GEOINTEL_OPENAI_MODEL", "gpt-4o-mini")

_COOPERATIVE_THRESHOLD: Final[float] = 0.10
_CONFLICTUAL_THRESHOLD: Final[float] = -0.10
_SENTIMENT_EPSILON: Final[float] = 0.05


# --- Prompt templates -------------------------------------------------------

BILATERAL_PROMPT: Final[str] = """You are a geopolitical intelligence analyst. Based on the following quantitative relationship data between {country_a} and {country_b}, write a concise analytical summary (3-5 sentences).

Relationship Data:
- Total events analyzed: {total_events}
- Average sentiment tone: {avg_tone:.3f} (range: -1 very negative to +1 very positive)
- Relationship type: {relationship_type}
- Dominant event types from {country_a} to {country_b}: {types_a_to_b}
- Dominant event types from {country_b} to {country_a}: {types_b_to_a}
- Conflict event ratio: {conflict_ratio:.1%}
- Cooperation event ratio: {coop_ratio:.1%}

Write a professional intelligence-style summary focusing on:
1. Overall relationship character
2. Key interaction patterns
3. Potential drivers of tension or cooperation

Keep it factual, analytical, and 3-5 sentences."""


COUNTRY_PROFILE_PROMPT: Final[str] = """You are a geopolitical intelligence analyst. Based on the network metrics for {country}, write a country geopolitical profile (4-6 sentences).

Network Metrics:
- Global Influence (PageRank): {pagerank:.4f} (higher = more influential)
- Betweenness Centrality: {betweenness:.4f} (bridge role in global network)
- Eigenvector Centrality: {eigenvector:.4f} (connected to influential countries)
- Conflict Ratio: {conflict_ratio:.1%} (fraction of interactions that are conflictual)
- Average Outgoing Tone: {avg_out_tone:.3f}
- Total Interactions: {total_events}
- Top Interaction Partners: {top_partners}

Write a professional intelligence-style profile covering:
1. The role and influence of {country} in the global network
2. Whether it plays a bridging or peripheral role
3. Overall relationship posture (cooperative vs conflictual)
4. Key geopolitical relationships"""


NETWORK_SUMMARY_PROMPT: Final[str] = """You are a geopolitical intelligence analyst. Summarize the current state of the global geopolitical network based on these metrics:

Network Statistics:
- Countries (nodes): {nodes}
- Diplomatic interactions (edges): {edges}
- Network Density: {density:.4f}
- Average Sentiment: {avg_tone:.3f}
- Negative Interaction Ratio: {negative_ratio:.1%}
- Modularity (bloc formation): {modularity:.3f}
- Number of Geopolitical Blocs: {num_communities}
- Global Geopolitical Polarization Index (GGPI): {ggpi:.3f}

Top 5 Most Influential Countries: {top5}
Most Conflictual Pairs: {conflict_pairs}

Write a 5-7 sentence executive summary of global geopolitical dynamics, identifying key trends, power centers, and risk areas."""


# --- Narrative contexts -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class BilateralContext:
    """Quantitative facts describing one country pair.

    Attributes
    ----------
    country_a, country_b : str
        ISO-3 country codes.
    total_events : int
        Events summed across both directions.
    avg_tone : float
        Mean normalised tone across the realised directions.
    relationship_type : str
        Categorical relationship label.
    types_a_to_b, types_b_to_a : str
        Rendered event-type histograms per direction.
    conflict_ratio, coop_ratio : float
        Shares of classified events in ``[0, 1]``.
    """

    country_a: str
    country_b: str
    total_events: int
    avg_tone: float
    relationship_type: str
    types_a_to_b: str
    types_b_to_a: str
    conflict_ratio: float
    coop_ratio: float

    def as_prompt_fields(self) -> dict[str, Any]:
        """Return the mapping consumed by :data:`BILATERAL_PROMPT`."""
        return {
            "country_a": self.country_a,
            "country_b": self.country_b,
            "total_events": self.total_events,
            "avg_tone": self.avg_tone,
            "relationship_type": self.relationship_type,
            "types_a_to_b": self.types_a_to_b,
            "types_b_to_a": self.types_b_to_a,
            "conflict_ratio": self.conflict_ratio,
            "coop_ratio": self.coop_ratio,
        }


@dataclass(frozen=True, slots=True)
class CountryContext:
    """Network metrics describing a single country."""

    country: str
    pagerank: float
    betweenness: float
    eigenvector: float
    conflict_ratio: float
    avg_out_tone: float
    total_events: int
    top_partners: str

    def as_prompt_fields(self) -> dict[str, Any]:
        """Return the mapping consumed by :data:`COUNTRY_PROFILE_PROMPT`."""
        return {
            "country": self.country,
            "pagerank": self.pagerank,
            "betweenness": self.betweenness,
            "eigenvector": self.eigenvector,
            "conflict_ratio": self.conflict_ratio,
            "avg_out_tone": self.avg_out_tone,
            "total_events": self.total_events,
            "top_partners": self.top_partners,
        }


@dataclass(frozen=True, slots=True)
class NetworkContext:
    """Global topology facts describing one graph snapshot."""

    nodes: int
    edges: int
    density: float
    avg_tone: float
    negative_ratio: float
    modularity: float
    num_communities: int
    ggpi: float
    top5: str
    conflict_pairs: str

    def as_prompt_fields(self) -> dict[str, Any]:
        """Return the mapping consumed by :data:`NETWORK_SUMMARY_PROMPT`."""
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "density": self.density,
            "avg_tone": self.avg_tone,
            "negative_ratio": self.negative_ratio,
            "modularity": self.modularity,
            "num_communities": self.num_communities,
            "ggpi": self.ggpi,
            "top5": self.top5,
            "conflict_pairs": self.conflict_pairs,
        }


# --- Deterministic offline generator ---------------------------------------

def _sentiment_label(tone: float) -> str:
    """Describe a normalised tone value in prose.

    Parameters
    ----------
    tone : float
        Normalised tone in ``[-1, 1]``.

    Returns
    -------
    str
        One of ``predominantly positive``, ``predominantly negative`` or
        ``mixed``.
    """
    if tone > _SENTIMENT_EPSILON:
        return "predominantly positive"
    if tone < -_SENTIMENT_EPSILON:
        return "predominantly negative"
    return "mixed"


class OfflineNarrator:
    """Deterministic narrative generator used when no LLM is reachable.

    Output is derived entirely from the supplied context objects, so the
    resulting prose reports the same figures as the dashboard and the report
    export.
    """

    @staticmethod
    def bilateral(context: BilateralContext) -> str:
        """Render a bilateral relationship summary.

        Parameters
        ----------
        context : BilateralContext
            Facts describing the country pair.

        Returns
        -------
        str
            A four-sentence analytical summary.
        """
        if context.total_events == 0:
            return (
                f"No direct interactions between {context.country_a} and "
                f"{context.country_b} were recorded in the observation window. "
                f"The dyad is either dormant or mediated entirely through third "
                f"parties over this period."
            )

        posture = (
            "cooperation outweighs friction"
            if context.coop_ratio > context.conflict_ratio
            else "friction outweighs cooperation"
            if context.conflict_ratio > context.coop_ratio
            else "cooperative and conflictual activity are balanced"
        )
        return (
            f"The {context.country_a}-{context.country_b} relationship registers "
            f"{context.total_events} recorded interactions with "
            f"{_sentiment_label(context.avg_tone)} sentiment "
            f"(mean tone {context.avg_tone:+.3f}), classifying the dyad as "
            f"{context.relationship_type.lower()}. Across classified events "
            f"{posture}, at {context.conflict_ratio:.1%} conflictual against "
            f"{context.coop_ratio:.1%} cooperative. Outbound activity from "
            f"{context.country_a} is dominated by {context.types_a_to_b}, while "
            f"{context.country_b} responds with {context.types_b_to_a}. The "
            f"asymmetry between the two directions is the primary indicator to "
            f"monitor for a shift in relationship character."
        )

    @staticmethod
    def country(context: CountryContext) -> str:
        """Render a country geopolitical profile.

        Parameters
        ----------
        context : CountryContext
            Network metrics for the country.

        Returns
        -------
        str
            A four-sentence analytical profile.
        """
        bridging = (
            "a structural bridge between otherwise weakly connected regions"
            if context.betweenness > 0.05
            else "a participant within established clusters rather than a bridge"
        )
        posture = (
            "predominantly conflictual"
            if context.conflict_ratio > 0.55
            else "predominantly cooperative"
            if context.conflict_ratio < 0.45
            else "evenly balanced"
        )
        return (
            f"{context.country} holds a PageRank influence score of "
            f"{context.pagerank:.4f} across {context.total_events} recorded "
            f"interactions, with an eigenvector centrality of "
            f"{context.eigenvector:.4f} indicating its proximity to other "
            f"high-influence states. Its betweenness of {context.betweenness:.4f} "
            f"positions it as {bridging}. Outbound posture is {posture} at a "
            f"{context.conflict_ratio:.1%} conflict ratio and a mean outgoing "
            f"tone of {context.avg_out_tone:+.3f}. Principal counterparties are "
            f"{context.top_partners}, which together define its immediate "
            f"strategic environment."
        )

    @staticmethod
    def network(context: NetworkContext) -> str:
        """Render a global executive summary.

        Parameters
        ----------
        context : NetworkContext
            Global topology facts.

        Returns
        -------
        str
            A five-sentence executive summary.
        """
        polarization = (
            "severe"
            if context.ggpi > 0.60
            else "elevated"
            if context.ggpi > 0.40
            else "moderate"
        )
        return (
            f"The global interaction network spans {context.nodes} states linked "
            f"by {context.edges:,} directed relationships at a density of "
            f"{context.density:.4f}. Community detection resolves "
            f"{context.num_communities} distinct blocs with a modularity of "
            f"{context.modularity:.3f}, indicating measurable structural "
            f"separation between alignment groups. Sentiment across the network "
            f"averages {context.avg_tone:+.3f}, with {context.negative_ratio:.1%} "
            f"of relationships carrying negative tone. The resulting "
            f"Polarization Index of {context.ggpi:.3f} places the system in a "
            f"{polarization} band. Influence concentrates in {context.top5}, "
            f"while the sharpest friction is recorded on {context.conflict_pairs}."
        )


# --- LLM client -------------------------------------------------------------

class LLMClient:
    """Provider-resolving completion client with a deterministic fallback.

    Resolution order for ``provider="auto"`` is Anthropic, then OpenAI, then
    offline. A missing package, a missing credential or a runtime API failure
    degrades to the offline path rather than raising.

    Parameters
    ----------
    provider : {"auto", "anthropic", "openai", "offline"}, optional
        Requested provider. The resolved provider is exposed as
        :attr:`provider` after construction.
    """

    def __init__(self, provider: Provider = "auto") -> None:
        self.provider: Provider = provider
        self._client: Any = None
        self._resolve_client()

    @property
    def is_offline(self) -> bool:
        """Whether the client resolved to the offline generator."""
        return self.provider == "offline"

    def _resolve_client(self) -> None:
        """Instantiate the first available provider client.

        Sets :attr:`provider` to the resolved provider and :attr:`_client` to
        the corresponding SDK client, or to ``None`` in offline mode.
        """
        requested = self.provider

        if requested in ("anthropic", "auto") and self._try_anthropic():
            return
        if requested in ("openai", "auto") and self._try_openai():
            return

        self._client = None
        self.provider = "offline"
        if requested == "offline":
            _log.info("%s Offline narrative generation selected", OK)
        else:
            _log.warning(
                "%s No LLM credentials available; using deterministic offline "
                "narrative generation", WARN,
            )

    def _try_anthropic(self) -> bool:
        """Attempt to construct an Anthropic client.

        Returns
        -------
        bool
            ``True`` when the client was constructed successfully.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return False
        try:
            import anthropic
        except ImportError:
            _log.warning("%s ANTHROPIC_API_KEY set but the SDK is not installed", WARN)
            return False

        self._client = anthropic.Anthropic(api_key=api_key)
        self.provider = "anthropic"
        _log.info("%s Narrative provider: Anthropic (%s)", OK, DEFAULT_ANTHROPIC_MODEL)
        return True

    def _try_openai(self) -> bool:
        """Attempt to construct an OpenAI client.

        Returns
        -------
        bool
            ``True`` when the client was constructed successfully.
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False
        try:
            import openai
        except ImportError:
            _log.warning("%s OPENAI_API_KEY set but the SDK is not installed", WARN)
            return False

        self._client = openai.OpenAI(api_key=api_key)
        self.provider = "openai"
        _log.info("%s Narrative provider: OpenAI (%s)", OK, DEFAULT_OPENAI_MODEL)
        return True

    def complete(
        self,
        prompt: str,
        fallback: Callable[[], str],
        max_tokens: int = 500,
    ) -> str:
        """Generate a completion, degrading to ``fallback`` on any failure.

        Parameters
        ----------
        prompt : str
            Fully rendered prompt.
        fallback : callable
            Zero-argument callable producing the deterministic narrative used
            when no provider is available or the API call fails.
        max_tokens : int, optional
            Response length cap forwarded to the provider.

        Returns
        -------
        str
            Generated narrative text.
        """
        if self.provider == "offline" or self._client is None:
            return fallback()

        try:
            if self.provider == "anthropic":
                message = self._client.messages.create(
                    model=DEFAULT_ANTHROPIC_MODEL,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text

            response = self._client.chat.completions.create(
                model=DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or fallback()

        except Exception as exc:  # SDK exception hierarchies vary by provider.
            _log.error(
                "%s %s completion failed (%s); using offline narrative",
                ERROR, self.provider, exc,
            )
            return fallback()


# --- Narrator ---------------------------------------------------------------

def _format_event_types(edge: dict[str, Any] | None, limit: int = 3) -> str:
    """Render an edge event-type histogram as a readable clause.

    Parameters
    ----------
    edge : dict or None
        Edge attribute mapping, or ``None`` when the direction is absent.
    limit : int, optional
        Maximum number of event types to list.

    Returns
    -------
    str
        Comma-separated ``label (count)`` pairs, ordered by descending count.
    """
    if not edge:
        return "no direct interactions recorded"

    histogram: dict[str, int] = edge.get("event_types", {})
    if not histogram:
        return str(edge.get("dominant_type", "unclassified activity"))

    ranked = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return ", ".join(f"{label} ({count})" for label, count in ranked)


class GeopoliticalNarrator:
    """Produces narrative summaries for the pipeline and the dashboard.

    Results are memoised per instance so repeated dashboard interactions do
    not re-issue provider calls.

    Parameters
    ----------
    provider : {"auto", "anthropic", "openai", "offline"}, optional
        Requested narrative provider.
    """

    def __init__(self, provider: Provider = "auto") -> None:
        self.llm = LLMClient(provider=provider)
        self._cache: dict[str, str] = {}

    # --- Context builders ---------------------------------------------------

    @staticmethod
    def build_bilateral_context(
        G: nx.DiGraph,
        country_a: str,
        country_b: str,
    ) -> BilateralContext:
        """Assemble the facts describing one country pair.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        country_a, country_b : str
            ISO-3 country codes.

        Returns
        -------
        BilateralContext
            Populated context, zeroed when neither direction exists.
        """
        forward = dict(G[country_a][country_b]) if G.has_edge(country_a, country_b) else None
        reverse = dict(G[country_b][country_a]) if G.has_edge(country_b, country_a) else None

        total_events = sum(
            edge.get("num_events", 0) for edge in (forward, reverse) if edge
        )
        tones = [edge["tone"] for edge in (forward, reverse) if edge and "tone" in edge]
        avg_tone = float(np.mean(tones)) if tones else 0.0

        conflict = sum(edge.get("conflict_count", 0) for edge in (forward, reverse) if edge)
        cooperation = sum(edge.get("coop_count", 0) for edge in (forward, reverse) if edge)
        classified = conflict + cooperation

        if avg_tone > _COOPERATIVE_THRESHOLD:
            relationship = "Cooperative"
        elif avg_tone < _CONFLICTUAL_THRESHOLD:
            relationship = "Conflictual"
        else:
            relationship = "Neutral/Mixed"

        return BilateralContext(
            country_a=country_a,
            country_b=country_b,
            total_events=int(total_events),
            avg_tone=avg_tone,
            relationship_type=relationship,
            types_a_to_b=_format_event_types(forward),
            types_b_to_a=_format_event_types(reverse),
            conflict_ratio=conflict / classified if classified else 0.0,
            coop_ratio=cooperation / classified if classified else 0.0,
        )

    @staticmethod
    def build_country_context(
        G: nx.DiGraph,
        country: str,
        metrics_df: pd.DataFrame,
        top_partners: int = 5,
    ) -> CountryContext:
        """Assemble the facts describing one country.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        country : str
            ISO-3 country code, expected in ``metrics_df.index``.
        metrics_df : pandas.DataFrame
            Frame produced by ``compute_metrics``.
        top_partners : int, optional
            Number of counterparties to list.

        Returns
        -------
        CountryContext
            Populated context.
        """
        metrics = metrics_df.loc[country]

        partner_volume: dict[str, int] = {}
        for neighbour in set(G.successors(country)) | set(G.predecessors(country)):
            volume = 0
            if G.has_edge(country, neighbour):
                volume += G[country][neighbour].get("num_events", 0)
            if G.has_edge(neighbour, country):
                volume += G[neighbour][country].get("num_events", 0)
            partner_volume[neighbour] = volume

        ranked = sorted(partner_volume.items(), key=lambda item: (-item[1], item[0]))
        partners = ", ".join(code for code, _ in ranked[:top_partners]) or "none recorded"

        return CountryContext(
            country=country,
            pagerank=float(metrics.get("pagerank", 0.0)),
            betweenness=float(metrics.get("betweenness", 0.0)),
            eigenvector=float(metrics.get("eigenvector", 0.0)),
            conflict_ratio=float(metrics.get("conflict_ratio", 0.0)),
            avg_out_tone=float(metrics.get("avg_out_tone", 0.0)),
            total_events=int(metrics.get("total_events", 0)),
            top_partners=partners,
        )

    @staticmethod
    def build_network_context(
        G: nx.DiGraph,
        stats: NetworkStats,
        metrics_df: pd.DataFrame,
        conflict_threshold: float = -0.15,
        max_pairs: int = 5,
    ) -> NetworkContext:
        """Assemble the facts describing the global network.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        stats : NetworkStats
            Output of ``compute_network_stats``.
        metrics_df : pandas.DataFrame
            Frame produced by ``compute_metrics``.
        conflict_threshold : float, optional
            Tone below which a dyad is reported as conflictual.
        max_pairs : int, optional
            Number of conflictual dyads to list.

        Returns
        -------
        NetworkContext
            Populated context.
        """
        top5 = ", ".join(metrics_df.head(5).index.tolist()) or "none ranked"

        conflictual = sorted(
            (
                (data.get("tone", 0.0), source, target)
                for source, target, data in G.edges(data=True)
                if data.get("tone", 0.0) < conflict_threshold
            )
        )[:max_pairs]
        pairs = ", ".join(
            f"{source}-{target} (tone {tone:.2f})" for tone, source, target in conflictual
        ) or "none identified"

        return NetworkContext(
            nodes=stats["nodes"],
            edges=stats["edges"],
            density=stats["density"],
            avg_tone=stats["avg_tone"],
            negative_ratio=stats["negative_edge_ratio"],
            modularity=stats["modularity"],
            num_communities=stats["num_communities"],
            ggpi=stats["ggpi"],
            top5=top5,
            conflict_pairs=pairs,
        )

    # --- Summaries ----------------------------------------------------------

    def summarize_bilateral(self, G: nx.DiGraph, country_a: str, country_b: str) -> str:
        """Generate a bilateral relationship summary.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        country_a, country_b : str
            ISO-3 country codes.

        Returns
        -------
        str
            Narrative text, memoised per country pair.
        """
        cache_key = f"bilateral_{country_a}_{country_b}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        context = self.build_bilateral_context(G, country_a, country_b)
        result = self.llm.complete(
            BILATERAL_PROMPT.format(**context.as_prompt_fields()),
            fallback=lambda: OfflineNarrator.bilateral(context),
            max_tokens=400,
        )
        self._cache[cache_key] = result
        return result

    def summarize_country(
        self,
        G: nx.DiGraph,
        country: str,
        metrics_df: pd.DataFrame,
    ) -> str:
        """Generate a country geopolitical profile.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        country : str
            ISO-3 country code.
        metrics_df : pandas.DataFrame
            Frame produced by ``compute_metrics``.

        Returns
        -------
        str
            Narrative text, or a notice when the country is absent from the
            metrics frame.
        """
        cache_key = f"country_{country}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if country not in metrics_df.index:
            return f"Insufficient data for {country}."

        context = self.build_country_context(G, country, metrics_df)
        result = self.llm.complete(
            COUNTRY_PROFILE_PROMPT.format(**context.as_prompt_fields()),
            fallback=lambda: OfflineNarrator.country(context),
            max_tokens=450,
        )
        self._cache[cache_key] = result
        return result

    def summarize_network(
        self,
        G: nx.DiGraph,
        stats: NetworkStats,
        metrics_df: pd.DataFrame,
    ) -> str:
        """Generate a global executive summary.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        stats : NetworkStats
            Output of ``compute_network_stats``.
        metrics_df : pandas.DataFrame
            Frame produced by ``compute_metrics``.

        Returns
        -------
        str
            Narrative text.
        """
        context = self.build_network_context(G, stats, metrics_df)
        return self.llm.complete(
            NETWORK_SUMMARY_PROMPT.format(**context.as_prompt_fields()),
            fallback=lambda: OfflineNarrator.network(context),
            max_tokens=600,
        )

    def batch_summarize_countries(
        self,
        G: nx.DiGraph,
        metrics_df: pd.DataFrame,
        top_n: int = 10,
    ) -> dict[str, str]:
        """Generate profiles for the highest-ranked countries.

        Parameters
        ----------
        G : networkx.DiGraph
            Interaction graph.
        metrics_df : pandas.DataFrame
            Frame produced by ``compute_metrics``.
        top_n : int, optional
            Number of leading countries to profile.

        Returns
        -------
        dict of str to str
            Country code to narrative mapping.
        """
        countries = metrics_df.head(top_n).index.tolist()
        summaries: dict[str, str] = {}
        for position, country in enumerate(countries, start=1):
            _log.info("Profiling %s (%d/%d)", country, position, len(countries))
            summaries[country] = self.summarize_country(G, country, metrics_df)
        return summaries

    # --- Persistence --------------------------------------------------------

    def save_summaries(
        self,
        summaries: dict[str, Any],
        path: str | Path = "summaries.json",
    ) -> Path:
        """Persist generated summaries to JSON.

        Parameters
        ----------
        summaries : dict
            Summary payload to serialise.
        path : str or pathlib.Path, optional
            Destination file.

        Returns
        -------
        pathlib.Path
            The resolved destination path.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(summaries, handle, indent=2, ensure_ascii=False)
        _log.info("%s Summaries written: %s", OK, destination)
        return destination

    def load_summaries(self, path: str | Path = "summaries.json") -> dict[str, str]:
        """Load previously generated summaries into the instance cache.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Source file. A missing file is not an error.

        Returns
        -------
        dict of str to str
            The updated cache.
        """
        source = Path(path)
        if not source.exists():
            _log.warning("%s No cached summaries at %s", WARN, source)
            return self._cache

        try:
            with source.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            self._cache.update(
                {key: value for key, value in payload.items() if isinstance(value, str)}
            )
            _log.info("%s Loaded %d cached summaries", OK, len(self._cache))
        except (json.JSONDecodeError, OSError) as exc:
            _log.error("%s Failed to read summaries from %s: %s", ERROR, source, exc)

        return self._cache
