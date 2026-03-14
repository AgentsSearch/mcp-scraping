# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Search Engine is a Python data pipeline that scrapes MCP (Model Context Protocol) server metadata from the official registry, probes live MCP servers to extract real tool lists, checks availability of remote endpoints, detects pricing models, fetches documentation, and optionally performs LLM-powered analysis as a fallback. Output is stored in `mcp_agents.json`.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run the scraper (primary entry point)
python3 web_scraper_v2.py

# CLI flags
python3 web_scraper_v2.py --probeable   # Only output agents with live MCP endpoints
python3 web_scraper_v2.py --smithery     # Only output agents with Smithery configs

# Configuration is set via constants in main() of web_scraper_v2.py:
#   LIMIT = 500          # Max agents (None = all)
#   ENABLE_LLM = True    # Toggle LLM analysis (requires OPENAI_API_KEY env var)
#   LLM_MODEL = "gpt-4.1-mini"
```

There are no tests, linter, or build system configured.

## Architecture

The pipeline runs in sequential steps: **scrape agents (dedup + parallel 404 check) → probe MCP servers → check Smithery configs → process documentation + LLM analysis (parallel) → save JSON**.

### Key Classes (all in `web_scraper_v2.py`)

- **MCPRegistryScraper** — Fetches agent list from `registry.modelcontextprotocol.io` API with cursor-based pagination. Deduplicates by URL, parallel 404-checks, then concurrently fetches documentation (GitHub README preferred, registry detail page fallback) and extracts pricing.
- **PricingExtractor** — Detects pricing model (free/freemium/paid/open_source/unknown) using a priority chain: explicit pricing page (parallel URL checks) → keyword analysis → LICENSE file (fast-path + parallel fallback) → NPM/PyPI registry metadata.
- **MCPProber** — Probes live MCP servers using the MCP protocol (sends `initialize` + `tools/list` JSON-RPC requests). Extracts real tool names and counts from running servers. When probing succeeds, LLM capability extraction is skipped for that agent.
- **SmitheryConfigChecker** — Queries the Smithery registry API to detect Smithery-hosted servers and extract their configuration requirements (e.g., required API keys).
- **RegistryPageScraper** — Extracts prose content from registry HTML pages, stripping noise elements.
- **LLMAnalyser** — Calls OpenAI API to extract capabilities, limitations, requirements, quality score, and agent classification (ai_agent/api_wrapper) in a single combined API call via `analyse_and_classify()`. Used as a **fallback** — capability extraction is skipped for agents successfully probed via MCP protocol (classification still runs).
- **DocumentationProcessor** — Chunks documentation into ~512-token segments with sentence-boundary awareness and overlap, then orchestrates parallel LLM analysis across agents.

### Design Patterns

- **Multi-layer fallback** throughout: documentation sources, pricing detection, URL extraction all cascade through multiple strategies.
- **Probe-first, LLM-fallback** — MCP probing provides ground-truth tool data; LLM analysis only runs when probing fails or is unavailable.
- **Parallel execution** — 404 checking, scraping, MCP probing, pricing detection, and LLM analysis all use ThreadPoolExecutor for concurrent processing.
- **Deduplication** via URL tracking to prevent duplicate agents.
- **Graceful degradation** — Missing API keys, failed requests, and unavailable dependencies are handled without crashing.

### Data Schema

The unified agent format is defined in `schema.json` (with annotated version in `schema_comments.json`). Key fields: `agent_id` (MD5 hash), `tools[]`, `pricing`, `documentation`, `remotes[]` (live MCP server endpoints with transport type and auth), `probe_status`, `probed_tool_count`, `smithery_config`, `agent_classification` (ai_agent/api_wrapper/unknown), and `llm_extracted` (capabilities/limitations/requirements).

### Files

- `web_scraper_v2.py` — Primary scraper (~2500 lines) with all classes
- `web_scraper.py` — Original v1 scraper (superseded by v2)
- `mcp_agents.json` — Output data file
- `mcp_ai_agents_remote.json` — Full dataset with remote endpoints
- `schema.json` / `schema_comments.json` — Data schema definitions
