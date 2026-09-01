# GeoIntel

Geopolitical intelligence pipeline. Ingests bilateral event data from GDELT, models international relations as a directed weighted graph, computes network centrality and a polarization index, classifies events with DistilBERT, and generates analytical narratives. Ships with an interactive Streamlit dashboard.

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

```bash
python main.py
```

Generates synthetic events, builds the graph, and writes all artefacts to `output/`.

Then launch the dashboard:

```bash
python main.py --dashboard
```

---

## Usage

```bash
python main.py [options]
```

| Option | Description |
|---|---|
| `--source {mock,gdelt}` | Event source. Default `mock`. |
| `--start`, `--end` | Window as `YYYY-MM-DD`. Default: last 90 days. |
| `--events N` | Synthetic event count. Default 5000. |
| `--seed N` | Seed the generator for reproducible runs. |
| `--classify` | Run ML event classification. |
| `--llm` | Generate narratives. |
| `--llm-provider {auto,anthropic,openai,offline}` | Narrative provider. Default `auto`. |
| `--bilateral A B` | Analyse a country pair, e.g. `--bilateral USA CHN`. |
| `--temporal {month,quarter,year}` | Snapshot granularity. Default `month`. |
| `--output DIR` | Output directory. Default `output`. |
| `--dashboard` | Launch the dashboard and exit. |

Examples:

```bash
python main.py --source gdelt --start 2024-01-01 --end 2024-03-31
python main.py --classify --llm --bilateral USA CHN
python main.py --events 10000 --seed 42
```

---

## Output

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
analysis/graph_builder.py   Graph construction, centralities, GGPI
analysis/narrator.py        Narrative generation
models/event_classifier.py  DistilBERT classifier and keyword fallback
dashboard/app.py            Streamlit shell: data, caching, layout
dashboard/theme.py          Palette, typography, Plotly defaults
dashboard/geodata.py        Country centroids and GDP (PPP)
dashboard/blocs.py          Bloc affinity scoring
dashboard/figures.py        Plotly figure builders
tests/                      pytest suite
```

---

## Testing

```bash
pytest              # unit and contract tests
python smoke_test.py   # end-to-end operational check
```

Both run offline. `tests/test_standards.py` enforces the project conventions: no emoji in source or output, full type annotations, documented public definitions.

---


## Notes

- GDELT 1.0 exports are Latin-1 encoded with a 58-column schema; `Actor2CountryCode` is at index 17. A truncated column list silently yields zero bilateral events.
- The ML stack (`torch`, `transformers`) is kept out of `requirements.txt` so the dashboard deploys within hosted size limits. Install `requirements-ml.txt` for classification.
- `matplotlib` is deliberately not a dependency; table shading is rendered from the platform palette.

---

## License

MIT. GDELT data is provided by [The GDELT Project](https://www.gdeltproject.org/) under its own terms.
