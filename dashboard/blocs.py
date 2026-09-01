"""
Geopolitical Bloc Analysis
==========================
Assigns each state to the power pole it leans toward, by blending observed
network sentiment with a curated prior over established alignments.

Scoring model
-------------
For a country ``c`` and a pole ``p``::

    network(c, p) = mean bilateral tone on the direct c-p edges
    friend(c, p)  = weight-averaged bilateral tone of the neighbours of c
                    toward p, propagated one hop
    data(c, p)    = 0.70 * network(c, p) + 0.30 * friend(c, p)
    score(c, p)   = (1 - PRIOR_WEIGHT) * data(c, p)
                    + PRIOR_WEIGHT * prior(c, p)

The prior anchors the assignment so that sparse or synthetic observation
windows cannot invert well-established alignments. Countries absent from the
prior table are scored on observed data alone.
"""

from __future__ import annotations

from typing import Final

import networkx as nx

__all__ = [
    "GEOPOLITICAL_PRIORS",
    "PRIOR_WEIGHT",
    "POLE_PRESETS",
    "compute_blocs",
]

# Baseline affinities reflecting treaties, alliances and standing alignments.
# Format: {country: {pole: prior in [-1, 1]}}.
GEOPOLITICAL_PRIORS: Final[dict[str, dict[str, float]]] = {
    # Western bloc
    "GBR": {"USA": +0.80, "CHN": -0.35, "RUS": -0.50},
    "CAN": {"USA": +0.85, "CHN": -0.25, "RUS": -0.40},
    "AUS": {"USA": +0.80, "CHN": -0.30, "RUS": -0.35},
    "DEU": {"USA": +0.65, "CHN": -0.15, "RUS": -0.55},
    "FRA": {"USA": +0.60, "CHN": -0.10, "RUS": -0.40},
    "JPN": {"USA": +0.75, "CHN": -0.40, "RUS": -0.35},
    "KOR": {"USA": +0.70, "CHN": -0.20, "RUS": -0.30},
    "NLD": {"USA": +0.65, "CHN": -0.10, "RUS": -0.45},
    "NOR": {"USA": +0.65, "CHN": -0.10, "RUS": -0.45},
    "SWE": {"USA": +0.65, "CHN": -0.10, "RUS": -0.50},
    "POL": {"USA": +0.70, "CHN": -0.15, "RUS": -0.70},
    "ITA": {"USA": +0.55, "CHN": -0.05, "RUS": -0.30},
    # Aligned despite geography
    "ISR": {"USA": +0.85, "CHN": -0.10, "RUS": -0.10, "IRN": -0.90},
    "UKR": {"USA": +0.80, "CHN": -0.10, "RUS": -0.95},
    # Contested and swing states
    "IND": {"USA": +0.20, "CHN": -0.30, "RUS": +0.15},
    "TUR": {"USA": +0.10, "CHN": -0.05, "RUS": -0.15},
    "SAU": {"USA": +0.35, "CHN": +0.15, "RUS": +0.10},
    "EGY": {"USA": +0.25, "CHN": +0.10, "RUS": +0.05},
    "ZAF": {"USA": +0.05, "CHN": +0.10, "RUS": +0.05},
    "NGA": {"USA": +0.10, "CHN": +0.05, "RUS": -0.05},
    "IDN": {"USA": +0.15, "CHN": +0.05, "RUS": -0.05},
    "MEX": {"USA": +0.40, "CHN": -0.10, "RUS": -0.10},
    "BRA": {"USA": +0.10, "CHN": +0.15, "RUS": +0.05},
    "ARG": {"USA": +0.05, "CHN": +0.20, "RUS": +0.10},
    "CHL": {"USA": +0.25, "CHN": +0.10, "RUS": -0.05},
    # Eastern bloc
    "RUS": {"USA": -0.80, "CHN": +0.65},
    "IRN": {"USA": -0.85, "CHN": +0.40, "RUS": +0.35},
    "PAK": {"USA": -0.10, "CHN": +0.65, "RUS": +0.10},
    "CHN": {"USA": -0.65, "RUS": +0.50},
}

# Blend ratio between curated priors and observed network data.
PRIOR_WEIGHT: Final[float] = 0.40

# Relative contribution of direct edges against one-hop propagation.
_DIRECT_WEIGHT: Final[float] = 0.70
_FRIEND_WEIGHT: Final[float] = 0.30

POLE_PRESETS: Final[dict[str, tuple[str, ...]]] = {
    "USA / CHN": ("USA", "CHN"),
    "USA / CHN / RUS": ("USA", "CHN", "RUS"),
    "USA / CHN / EU": ("USA", "CHN", "DEU"),
    "USA / CHN / RUS / IND": ("USA", "CHN", "RUS", "IND"),
    "Custom": (),
}


def bilateral_tone(G: nx.DiGraph, country_a: str, country_b: str) -> float:
    """Mean tone across the realised directions of a dyad.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    country_a, country_b : str
        ISO-3 country codes.

    Returns
    -------
    float
        Mean tone, or ``0.0`` when neither direction exists.
    """
    tones: list[float] = []
    if G.has_edge(country_a, country_b):
        tones.append(G[country_a][country_b].get("tone", 0.0))
    if G.has_edge(country_b, country_a):
        tones.append(G[country_b][country_a].get("tone", 0.0))
    return float(sum(tones) / len(tones)) if tones else 0.0


def _neighbour_weight(G: nx.DiGraph, country: str, neighbour: str) -> float:
    """Interaction weight between a country and one neighbour.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph.
    country, neighbour : str
        ISO-3 country codes.

    Returns
    -------
    float
        Summed weight across the realised directions, minimum ``1.0``.
    """
    weight = 0.0
    if G.has_edge(country, neighbour):
        weight += float(G[country][neighbour].get("weight", 0.0))
    if G.has_edge(neighbour, country):
        weight += float(G[neighbour][country].get("weight", 0.0))
    return weight or 1.0


def compute_blocs(
    G: nx.DiGraph,
    poles: list[str],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """Assign every country to the pole it leans toward.

    Parameters
    ----------
    G : networkx.DiGraph
        Interaction graph produced by ``build_graph``.
    poles : list of str
        Two to four ISO-3 codes acting as power poles. Each pole is assigned
        to its own bloc.

    Returns
    -------
    tuple of (dict, dict)
        Country to pole assignment, and the full affinity table mapping each
        non-pole country to its score against every pole.
    """
    affinity: dict[str, dict[str, float]] = {}

    # Precompute pole tone for every node once; the friend term reuses it.
    tone_to_pole: dict[str, dict[str, float]] = {
        node: {pole: bilateral_tone(G, node, pole) for pole in poles}
        for node in G.nodes()
    }

    for node in G.nodes():
        if node in poles:
            continue

        neighbours = (set(G.successors(node)) | set(G.predecessors(node))) - {node}
        neighbour_weights = {
            neighbour: _neighbour_weight(G, node, neighbour)
            for neighbour in neighbours
        }

        scores: dict[str, float] = {}
        for pole in poles:
            direct = tone_to_pole[node][pole]

            relevant = {
                neighbour: weight
                for neighbour, weight in neighbour_weights.items()
                if neighbour != pole
            }
            total_weight = sum(relevant.values())
            friend = (
                sum(
                    tone_to_pole[neighbour][pole] * weight
                    for neighbour, weight in relevant.items()
                )
                / total_weight
                if total_weight > 0
                else 0.0
            )

            observed = _DIRECT_WEIGHT * direct + _FRIEND_WEIGHT * friend
            prior = GEOPOLITICAL_PRIORS.get(node, {}).get(pole, 0.0)
            scores[pole] = (1 - PRIOR_WEIGHT) * observed + PRIOR_WEIGHT * prior

        affinity[node] = scores

    # Deterministic assignment: highest score, ties broken alphabetically.
    assignments: dict[str, str] = {
        node: min(scores.items(), key=lambda item: (-item[1], item[0]))[0]
        for node, scores in affinity.items()
    }
    for pole in poles:
        if pole in G:
            assignments[pole] = pole

    return assignments, affinity
