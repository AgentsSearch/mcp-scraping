# mcp-scraping - Agent Search Engine

A data pipeline that scrapes AI agent (MCP server) metadata from the official [Model Context Protocol registry](https://registry.modelcontextprotocol.io), enriches it with pricing detection, documentation fetching, and optional LLM-powered analysis, then outputs a structured JSON dataset.

## What It Does

1. **Scrapes** the MCP registry API with cursor-based pagination
2. **Fetches documentation** for each agent (GitHub README → registry detail page → API description)
3. **Detects pricing** (free / freemium / paid / open_source) via pricing pages, keyword analysis, LICENSE files, and NPM/PyPI metadata
4. **Classifies** whether entries are true AI agents and checks availability
5. **Optionally analyzes** documentation with an LLM to extract capabilities, limitations, requirements, and a calibrated quality score

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

### LLM Analysis (Optional)

To enable LLM-powered capability extraction and quality scoring:

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
| `LLM_DELAY` | `0` | Seconds between LLM API calls |

## Output Schema

Each agent in `mcp_agents.json` includes:

- **Identity** — `agent_id`, `name`, `source`, `source_url`
- **Tools** — list of tools with names and descriptions
- **Pricing** — detected pricing model
- **Documentation** — raw README and detail page text, chunked into ~512-token segments
- **Classification** — whether it's a true AI agent, with rationale
- **LLM Extracted** (when enabled) — `capabilities`, `limitations`, `requirements`, `documentation_quality` score (0.0–1.0)

Full schema definition: [`schema.json`](schema.json)

## Dependencies

- `requests` — HTTP client
- `beautifulsoup4` — HTML parsing
- `python-dotenv` — Environment variable loading
- OpenAI API key (only if LLM analysis is enabled)
