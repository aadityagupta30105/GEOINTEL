# GeoIntel

Geopolitical intelligence platform. Collects bilateral event data from GDELT into a queryable SQLite store, models international relations as a directed weighted graph, computes network centrality and a polarization index, classifies events with DistilBERT, and generates analytical narratives. Ships with an interactive Streamlit dashboard.

Runs fully offline: no API keys, no network, no model checkpoint required.

---

## Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Optional extras:

```bash
pip install -r requirements-ml.txt
pip install -r requirements-dev.txt
```

Without `requirements-ml.txt`, classification falls back to keyword matching; nothing else changes.

---

## Quick start

Fill the event store with a full year, then open the dashboard:

```bash
python -m data.harvest --mock --year
```

```bash
python main.py --dashboard
```

`--mock` synthesises a realistic year in about fifteen seconds and needs no network. Swap it for a real collection when you are ready:

```bash
python -m data.harvest --year --workers 6
```

---

## Collecting a year

`python -m data.harvest` is the collector. It is built for long runs.

**Resumable.** Every calendar day's outcome is written to the store as it lands. A rerun skips days already held, so an interrupted harvest continues where it stopped rather than starting over. `Ctrl+C` drains work in flight, commits it, and reports coverage.

**Parallel.** Downloads and parsing run on a worker pool; writes are serialised because SQLite admits one writer. That is not the bottleneck — the cost is transferring roughly 6 MB per day.

**Full fidelity.** Every state-to-state event in each export is retained by default. Pass `--cap N` to sample instead.

**Honest about cost.** A projection prints before anything is downloaded:

```bash
python -m data.harvest --year --dry-run
```

```
Days to collect  : 365
Estimated volume : 2.2 GiB compressed
Estimated time   : 32m 57s at 6 workers
The run is resumable: stop it at any time and rerun to continue.
```

| Option | Description |
|---|---|
| `--year` | Collect the trailing 365 days |
| `--days N` | Collect the trailing N days |
| `--start`, `--end` | Explicit window as `YYYY-MM-DD` |
| `--workers N` | Concurrent download workers. Default 6 |
| `--cap N` | Retain at most N events per day. Default 0, meaning all |
| `--mock` | Fill the window with synthetic events, no network |
| `--retry-failed` | Collect only the days whose last attempt failed |
| `--dry-run` | Print the cost projection and exit |
| `--coverage` | Print what the store holds and exit |
| `--rebuild` | Rebuild the rollup table from the event table |
| `--db PATH` | Store path. Default `data/geointel.db` |

Check what you have at any time:

```bash
python -m data.harvest --coverage
```

A day GDELT never published is recorded as `empty` and never retried. A day that failed in transit is recorded as `failed` and stays pending until `--retry-failed` or the next run picks it up.

---

## The event store

Events live in a single SQLite file at `data/geointel.db`. No server, no extra dependency — `sqlite3` is in the standard library.

### Why it stays fast over a year

A year of full-fidelity GDELT is millions of rows, which is more than a dashboard should ever load. Nothing reads the event table directly. Every aggregate the graph layer needs comes from `dyad_daily`, a rollup keyed by `(date, actor1, actor2, event_type)` that collapses the stream by one to two orders of magnitude while preserving exactly what graph construction consumes: event count, tone sum, mentions and the event-type histogram. Building a graph over any window is one indexed `GROUP BY`.

Measured on a real 3,538,518-event store (183 days of GDELT, 221 countries, 578,528 rollup rows): the full-window aggregate returns 45,306 dyad rows in 0.64 s, a one-month window in 0.22 s, and the whole dashboard first render - aggregate, graph, centralities and statistics - completes in 4.2 s, cached thereafter.

### Schema

| Object | Contents |
|---|---|
| `events` | One row per event, keyed by GDELT `GLOBALEVENTID`. Re-harvesting a day inserts nothing |
| `dyad_daily` | Rollup of `events`. Derived, rebuildable at any time |
| `harvest_log` | One row per calendar day attempted, with its outcome. This is what makes collection resumable |
| `meta` | Schema version |
| `v_edges` | Directed dyad aggregate over all time |
| `v_country_activity` | Per-country volume, mean tone and partner count |
| `v_monthly` | Monthly volume and mean tone |

### Querying it

From the dashboard, the **SQL console** tab: the schema is on screen, the query box is free-form, results export to CSV. Read-only, guarded twice — only a single `SELECT` or `WITH` statement is accepted, and it runs on a connection opened in SQLite read-only mode.

From Python:

```python
from data.store import EventStore

with EventStore() as store:
    top = store.query("""
        SELECT source, target, num_events, ROUND(tone, 3) AS tone
          FROM v_edges
         ORDER BY num_events DESC
         LIMIT 10
    """)

    # The hot path: filters are pushed into SQL, not applied in pandas.
    agg = store.dyad_aggregates(
        start="2024-01-01", end="2024-12-31",
        event_types=["Conflict"], countries=["USA", "CHN"],
    )
```

From the shell:

```bash
sqlite3 data/geointel.db "SELECT * FROM v_monthly"
```

---

## Pipeline

```bash
python main.py [options]
```

| Option | Description |
|---|---|
| `--source {mock,gdelt,store}` | Event source. Default `mock`. `store` reads the event store |
| `--db PATH` | Store path for `--source store` and `--save-store` |
| `--save-store` | Also write the run's events into the store |
| `--max-rows N` | Row cap when reading from the store. Default 2,000,000 |
| `--start`, `--end` | Window as `YYYY-MM-DD`. Default: last 90 days |
| `--events N` | Synthetic event count. Default 5000 |
| `--seed N` | Seed the generator for reproducible runs |
| `--classify` | Run ML event classification |
| `--llm` | Generate narratives |
| `--llm-provider {auto,anthropic,openai,offline}` | Narrative provider. Default `auto` |
| `--bilateral A B` | Analyse a country pair, e.g. `--bilateral USA CHN` |
| `--temporal {month,quarter,year}` | Snapshot granularity. Default `month` |
| `--output DIR` | Output directory. Default `output` |
| `--dashboard` | Launch the dashboard and exit |

Examples:

```bash
python main.py --source store --start 2024-01-01 --end 2024-12-31
python main.py --source gdelt --start 2024-01-01 --end 2024-03-31 --save-store
python main.py --classify --llm --bilateral USA CHN
```

Analysis over stored data costs a query and no network access, so reruns are free.

### Reading more than the row cap

`--source store` materialises events, because the CSV export and the report are event-level. When a window holds more rows than `--max-rows`, the read is **thinned across the whole window rather than truncated at the cap**, and the report declares the sample it was drawn from:

```
- **Date range**: 2026-03-12 to 2026-09-10
- **Sampling**: 50.0% sample of 3,538,518 events spanning the full window.
```

This matters: a plain `LIMIT` would return a chronological prefix, so a six-month request would silently yield the first ten weeks and the report would present that shorter range as the whole. Raise the cap for a complete read — 3.5M rows takes 21 s and 0.82 GiB:

```bash
python main.py --source store --max-rows 4000000
```

The dashboard is unaffected either way; it never materialises events.

---

## Dashboard

```bash
python main.py --dashboard
```

Six tabs: network graph, influence rankings, bilateral analysis, temporal trends, network overview, and the SQL console. The store is the default source when one exists; the pipeline export, a live GDELT fetch and synthetic generation remain available.

On the store source, the window, event-type and country filters are pushed into SQL. Widening the window costs a query, not a reload.

---

## Output

`main.py` writes these to `output/` regardless of source:

| File | Contents |
|---|---|
| `events_clean.csv` | Preprocessed bilateral events |
| `edges.csv` | Graph edge list with weights, tone, event counts |
| `country_metrics.csv` | PageRank, betweenness, eigenvector, conflict ratio |
| `temporal_metrics.csv` | Per-period network statistics |
| `network_stats.json` | Global statistics including GGPI |
| `summaries.json` | Generated narratives |
| `report.md` | Markdown intelligence report |

---

## GGPI

The Global Geopolitical Polarization Index scores network fragmentation on `[0, 1]`:

```
GGPI = 0.40 * modularity
     + 0.40 * negative_edge_ratio
     + 0.20 * max(0, -avg_tone)
```

---

## Structure

```
main.py                     Pipeline orchestrator (CLI)
smoke_test.py               End-to-end operational check
utils/logging_config.py     ASCII-safe console logging
data/gdelt_collector.py     GDELT fetch, synthetic generator, preprocessing
data/store.py               SQLite event store, schema, views, query API
data/harvest.py             Parallel resumable harvester (CLI)
analysis/graph_builder.py   Graph construction, centralities, GGPI
analysis/narrator.py        Narrative generation
models/event_classifier.py  DistilBERT classifier and keyword fallback
dashboard/app.py            Streamlit shell: data, caching, layout, SQL console
dashboard/theme.py          Palette, typography, Plotly defaults
dashboard/geodata.py        Country centroids and GDP (PPP)
dashboard/blocs.py          Bloc affinity scoring
dashboard/figures.py        Plotly figure builders
.streamlit/config.toml      Streamlit chrome, kept in step with the palette
tests/                      pytest suite
```

---

## Design

The dashboard follows the Automobili Lamborghini visual language: an unbroken black ground, a single gold accent, and hard geometry. Chroma is rationed on purpose — gold carries emphasis, the tone pair carries cooperation and conflict, and nothing else is coloured, so a red edge on the network map reads as a finding rather than as decoration.

Panels are cut at 45 degrees at two opposing corners rather than rounded. Display type is Chakra Petch, set uppercase with wide tracking; running text is Barlow; figures are JetBrains Mono so digits align in columns.

Every colour comes from `dashboard/theme.py`. The test suite fails on a literal hex in the stylesheet and on any drift between the palette and `.streamlit/config.toml`.

---

## Testing

```bash
pytest
```

```bash
python smoke_test.py
```

305 unit and contract tests plus 15 operational checks, all offline. `tests/test_standards.py` enforces the project conventions: no emoji in source or output, full type annotations, documented public definitions, and a stylesheet built entirely from palette constants.

The store's central contract — that a graph built from a database window is identical to one built from the equivalent event frame — is asserted edge by edge in both suites, because the two paths compute the same aggregate through entirely different machinery.

---

## Notes

- GDELT 1.0 exports are Latin-1 encoded with a 58-column schema; `Actor2CountryCode` is at index 17. A truncated column list silently yields zero bilateral events.
- GDELT publishes a day's export the following morning, so the harvester's default end date is yesterday.
- The ML stack (`torch`, `transformers`) is kept out of `requirements.txt` so the dashboard deploys within hosted size limits. Install `requirements-ml.txt` for classification.
- `matplotlib` is deliberately not a dependency; table shading is rendered from the platform palette.
- The store is gitignored. Rebuild it with `python -m data.harvest --year`.

---

## License

MIT. GDELT data is provided by [The GDELT Project](https://www.gdeltproject.org/) under its own terms.
