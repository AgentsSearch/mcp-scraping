# mcp-scraping - Agent Search Engine

A data pipeline that scrapes AI agent (MCP server) metadata from the official [Model Context Protocol registry](https://registry.modelcontextprotocol.io), probes live servers for tool definitions, enriches with pricing detection and documentation, and optionally performs LLM-powered analysis. Outputs a structured JSON dataset ready for embedding and similarity search.

## What It Does

1. **Scrapes** the MCP registry API with cursor-based pagination
2. **Captures remote endpoints** (`remotes`) — live MCP server URLs with transport type and auth requirements
3. **Probes live servers** via MCP protocol (`initialize` + `tools/list`) to get ground-truth tool definitions (name, description, input schema)
4. **Checks Smithery config** (when `--smithery`) — queries Smithery registry to determine which hosted servers need only a Smithery API key vs external service credentials
5. **Fetches documentation** for each agent (GitHub README → registry detail page → API description)
6. **Detects pricing** (free / freemium / paid / open_source) via pricing pages, keyword analysis, LICENSE files, and NPM/PyPI metadata
7. **Checks availability** — tests remote endpoints first, falls back to source URL; only keeps working remotes
8. **Classifies** whether entries are true AI agents or API wrappers
9. **LLM analysis (fallback)** — for agents without live remotes or where probing fails, uses an LLM to extract capabilities, limitations, requirements, and a calibrated quality score. Skipped for successfully probed agents.

Output is saved to `mcp_agents.json`.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/AgentsSearch/Agent-Search-Engine.git
cd Agent-Search-Engine

# Install dependencies
pip install -r requirements.txt

# Run the scraper
python3 web_scraper_v2.py
```

### Probeable-Only Mode

To output only free AI agents with accessible remote endpoints and no required auth:

```bash
python3 web_scraper_v2.py --probeable
```

### Smithery Mode

Like `--probeable`, but also includes Smithery-hosted AI agents that only need a Smithery API key (no external service credentials):

```bash
python3 web_scraper_v2.py --smithery
```

### LLM Analysis (Optional)

LLM enrichment is used as a fallback for agents that can't be probed via MCP protocol. To enable:

```bash
export OPENAI_API_KEY="sk-..."
```

Then set `ENABLE_LLM = True` in the `main()` function of `web_scraper_v2.py`.

## Configuration

Configuration is set via constants in `main()` of `web_scraper_v2.py`:

| Variable | Default | Description |
|---|---|---|
| `LIMIT` | `500` | Max agents to fetch (`None` = all) |
| `ENABLE_LLM` | `True` | Toggle LLM analysis |
| `LLM_MODEL` | `"gpt-4.1-mini"` | OpenAI model for analysis |
| `LLM_WORKERS` | `10` | Concurrent threads for LLM API calls |

CLI flags:

| Flag | Description |
|---|---|
| `--probeable` | Only output free AI agents with accessible remotes and no required auth |
| `--smithery` | Like `--probeable`, plus Smithery-hosted agents needing only a Smithery API key |

## Pipeline

```
Scrape registry API
        │
        ▼
Probe remotes (MCP initialize + tools/list)
        │
        ▼
Check Smithery config (--smithery only)
        │
        ▼
Fetch documentation (README / detail page)
        │
        ▼
LLM analysis (only for agents not successfully probed)
        │
        ▼
Filter (--probeable / --smithery)
        │
        ▼
Save to mcp_agents.json
```

## Output Schema

Each agent in `mcp_agents.json` includes:

- **Identity** — `agent_id`, `name`, `source`, `source_url`
- **Remotes** — live MCP server endpoints with transport type (`streamable-http` / `sse`), URL, and auth header requirements
- **Tools** — from MCP `tools/list` (probed) or documentation (LLM-extracted): name, description, input schema
- **Pricing** — detected pricing model
- **Documentation** — raw README and detail page text, chunked into ~512-token segments
- **Availability** — `is_available`, `availability_status`, with dead remotes filtered out
- **Probe status** — `probe_status` (`success` / `failed` / `skipped`), `probed_tool_count`
- **Smithery config** — `smithery_config` (`none` / `optional` / `required` / `unknown`) for Smithery-hosted servers
- **Classification** — whether it's a true AI agent, with rationale
- **LLM Extracted** (fallback) — `capabilities`, `limitations`, `requirements`, `documentation_quality` score (0.0–1.0)

Full schema definition: [`schema.json`](schema.json)

## Cost & Runtime Estimates

Using `gpt-4.1-mini` ($0.40/1M input tokens, $1.60/1M output tokens). All pipeline phases run in parallel using thread pools. GitHub requests include 429 retry with exponential backoff, and doc-fetching concurrency is capped at 5 workers to stay within rate limits.

| LIMIT | Agents processed* | LLM Cost | Estimated Runtime |
|---|---|---|---|
| 500 | ~400 | ~$0.45 | ~10–12 min |
| 1000 | ~800 | ~$0.90 | ~20–25 min |
| 1500 | ~1200 | ~$1.35 | ~30–38 min |
| None | ~2000–2500 | ~$2.25–$3.30 | ~50–65 min |

*\*After dedup and 404 filtering. ~15% of agents are successfully probed via MCP protocol and skip LLM capability extraction (classification still runs). Runtime increases at higher counts due to GitHub rate-limit retries.*

**Breakdown by phase** (for LIMIT=500):

| Phase | Time | Notes |
|---|---|---|
| Pagination | ~10s | Serial API calls |
| Dedup + 404 check | ~60s | Parallel (10 workers) |
| Doc fetch + pricing | ~360–420s | Parallel (5 workers) — capped at 5 workers + 429 retry backoff |
| MCP probing | ~50s | Parallel (10 workers) |
| LLM analysis | ~80s | Parallel (LLM_WORKERS=10) |

## Dependencies

- `requests` — HTTP client
- `beautifulsoup4` — HTML parsing
- `python-dotenv` — Environment variable loading
- OpenAI API key (only if LLM analysis is enabled)
