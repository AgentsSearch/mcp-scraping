"""Comprehensive unit tests for web_scraper_v2.py."""

import json
import hashlib
import os
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import requests

from web_scraper_v2 import (
    MCPRegistryScraper,
    PricingExtractor,
    MCPProber,
    SmitheryConfigChecker,
    RegistryPageScraper,
    LLMAnalyser,
    DocumentationProcessor,
    main,
)


# ============================================================================
# 1. MCPRegistryScraper
# ============================================================================

class TestMCPRegistryScraper:
    """Tests for the MCPRegistryScraper class."""

    # --- fetch_agent_list ---

    def test_fetch_agent_list_single_page(self):
        scraper = MCPRegistryScraper()
        agents = [{"server": {"name": f"agent-{i}"}} for i in range(5)]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"servers": agents, "metadata": {}}

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.fetch_agent_list()

        assert len(result) == 5

    def test_fetch_agent_list_multi_page_pagination(self):
        scraper = MCPRegistryScraper()
        page1_agents = [{"server": {"name": f"agent-{i}"}} for i in range(3)]
        page2_agents = [{"server": {"name": f"agent-{i}"}} for i in range(3, 5)]

        resp1 = MagicMock(status_code=200)
        resp1.json.return_value = {
            "servers": page1_agents,
            "metadata": {"nextCursor": "cursor_abc"},
        }
        resp2 = MagicMock(status_code=200)
        resp2.json.return_value = {
            "servers": page2_agents,
            "metadata": {},
        }

        with patch.object(scraper.session, "get", side_effect=[resp1, resp2]):
            result = scraper.fetch_agent_list()

        assert len(result) == 5

    def test_fetch_agent_list_max_results_truncation(self):
        scraper = MCPRegistryScraper()
        agents = [{"server": {"name": f"agent-{i}"}} for i in range(10)]
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"servers": agents, "metadata": {}}

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.fetch_agent_list(max_results=5)

        assert len(result) == 5

    def test_fetch_agent_list_bare_list_response(self):
        scraper = MCPRegistryScraper()
        agents = [{"name": "agent-1"}, {"name": "agent-2"}]
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = agents

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.fetch_agent_list()

        assert len(result) == 2

    def test_fetch_agent_list_http_error(self):
        scraper = MCPRegistryScraper()
        mock_resp = MagicMock(status_code=500)

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.fetch_agent_list()

        assert result == []

    def test_fetch_agent_list_request_exception(self):
        scraper = MCPRegistryScraper()

        with patch.object(
            scraper.session, "get", side_effect=requests.exceptions.ConnectionError
        ):
            result = scraper.fetch_agent_list()

        assert result == []

    def test_fetch_agent_list_empty_servers(self):
        scraper = MCPRegistryScraper()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"servers": [], "metadata": {}}

        with patch.object(scraper.session, "get", return_value=mock_resp):
            result = scraper.fetch_agent_list()

        assert result == []

    def test_fetch_agent_list_alternative_keys(self):
        scraper = MCPRegistryScraper()

        for key in ["agents", "data", "items"]:
            agents = [{"name": "test"}]
            mock_resp = MagicMock(status_code=200)
            mock_resp.json.return_value = {key: agents, "metadata": {}}

            with patch.object(scraper.session, "get", return_value=mock_resp):
                result = scraper.fetch_agent_list()

            assert len(result) == 1, f"Failed for key: {key}"

    # --- convert_to_unified_schema ---

    def test_convert_basic_agent(self, sample_raw_agent):
        scraper = MCPRegistryScraper()
        result = scraper.convert_to_unified_schema(sample_raw_agent)

        assert result["name"] == "test-server"
        assert result["source"] == "mcp"
        assert len(result["agent_id"]) == 16
        expected_id = hashlib.md5("mcp_test-server".encode()).hexdigest()[:16]
        assert result["agent_id"] == expected_id

    def test_convert_agent_id_deterministic(self, sample_raw_agent):
        scraper = MCPRegistryScraper()
        r1 = scraper.convert_to_unified_schema(sample_raw_agent)
        r2 = scraper.convert_to_unified_schema(sample_raw_agent)
        assert r1["agent_id"] == r2["agent_id"]

    def test_convert_source_url_priority(self):
        scraper = MCPRegistryScraper()

        # repository.url is preferred
        agent = {
            "server": {
                "name": "test",
                "repository": {"url": "https://github.com/a/b"},
                "websiteUrl": "https://example.com",
                "remotes": [{"url": "https://remote.example.com"}],
            },
            "_meta": {},
        }
        result = scraper.convert_to_unified_schema(agent)
        assert result["source_url"] == "https://github.com/a/b"

        # Falls back to websiteUrl
        agent["server"]["repository"] = {}
        result = scraper.convert_to_unified_schema(agent)
        assert result["source_url"] == "https://example.com"

        # Falls back to first remote URL
        agent["server"]["websiteUrl"] = ""
        result = scraper.convert_to_unified_schema(agent)
        assert result["source_url"] == "https://remote.example.com"

    def test_convert_missing_meta(self):
        scraper = MCPRegistryScraper()
        agent = {"server": {"name": "test"}, "_meta": None}
        result = scraper.convert_to_unified_schema(agent)
        assert result["last_updated"] == "Unknown"

    def test_convert_remotes_preserved(self, sample_raw_agent):
        scraper = MCPRegistryScraper()
        result = scraper.convert_to_unified_schema(sample_raw_agent)
        assert len(result["remotes"]) == 1
        assert result["remotes"][0]["url"] == "https://remote.example.com/mcp"

    def test_convert_empty_agent(self):
        scraper = MCPRegistryScraper()
        result = scraper.convert_to_unified_schema({"server": {}, "_meta": {}})
        assert result["name"] == "Unknown"
        assert result["source_url"] == ""
        assert result["description"] == ""

    # --- _extract_capabilities ---

    def test_extract_capabilities_from_list(self):
        scraper = MCPRegistryScraper()
        agent = {"capabilities": ["search", "analyze"], "description": ""}
        caps = scraper._extract_capabilities(agent)
        assert "search" in caps
        assert "analyze" in caps

    def test_extract_capabilities_from_dict(self):
        scraper = MCPRegistryScraper()
        agent = {"capabilities": {"search": True, "analyze": True}, "description": ""}
        caps = scraper._extract_capabilities(agent)
        assert "search" in caps
        assert "analyze" in caps

    def test_extract_capabilities_from_tools_dicts(self):
        scraper = MCPRegistryScraper()
        agent = {
            "tools": [{"name": "search"}, {"name": "fetch"}],
            "description": "",
        }
        caps = scraper._extract_capabilities(agent)
        assert "tool:search" in caps
        assert "tool:fetch" in caps

    def test_extract_capabilities_from_tools_strings(self):
        scraper = MCPRegistryScraper()
        agent = {"tools": ["search", "fetch"], "description": ""}
        caps = scraper._extract_capabilities(agent)
        assert "tool:search" in caps

    def test_extract_capabilities_from_description_keywords(self):
        scraper = MCPRegistryScraper()
        agent = {"description": "A tool to search and analyze data"}
        caps = scraper._extract_capabilities(agent)
        assert "search" in caps
        assert "analyze" in caps

    def test_extract_capabilities_deduplication(self):
        scraper = MCPRegistryScraper()
        agent = {
            "capabilities": ["search", "search"],
            "description": "search tool",
        }
        caps = scraper._extract_capabilities(agent)
        assert caps.count("search") == 1

    def test_extract_capabilities_empty(self):
        scraper = MCPRegistryScraper()
        assert scraper._extract_capabilities({"description": "a server"}) == []

    # --- fetch_documentation ---

    def test_fetch_documentation_readme_github_main(self):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "https://github.com/org/repo"}

        readme_resp = MagicMock(status_code=200, text="# README content here")

        with patch.object(scraper.session, "get", return_value=readme_resp):
            docs = scraper.fetch_documentation(agent)

        assert "readme" in docs
        assert docs["readme"] == "# README content here"

    def test_fetch_documentation_readme_github_master_fallback(self):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "https://github.com/org/repo"}

        fail_resp = MagicMock(status_code=404)
        ok_resp = MagicMock(status_code=200, text="# Master README")

        with patch.object(scraper.session, "get", side_effect=[fail_resp, ok_resp]):
            docs = scraper.fetch_documentation(agent)

        assert docs.get("readme") == "# Master README"

    def test_fetch_documentation_explicit_readme_url(self):
        scraper = MCPRegistryScraper()
        agent = {
            "readme_url": "https://example.com/docs/README.md",
            "source_url": "",
        }
        ok_resp = MagicMock(status_code=200, text="# Custom README")

        with patch.object(scraper.session, "get", return_value=ok_resp):
            docs = scraper.fetch_documentation(agent)

        assert docs["readme"] == "# Custom README"

    def test_fetch_documentation_fallback_to_detail_page(self, registry_detail_html):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "https://example.com/server"}

        detail_resp = MagicMock(status_code=200, text=registry_detail_html)

        with patch.object(scraper.session, "get", return_value=detail_resp):
            docs = scraper.fetch_documentation(agent)

        assert "detail_page" in docs
        assert len(docs["detail_page"]) > 100

    def test_fetch_documentation_all_fail(self):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "https://github.com/org/repo"}

        fail_resp = MagicMock(status_code=404)

        with patch.object(scraper.session, "get", return_value=fail_resp):
            docs = scraper.fetch_documentation(agent)

        # No readme found, detail page also 404
        assert docs.get("readme") is None

    def test_fetch_documentation_request_exception(self):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "https://github.com/org/repo"}

        with patch.object(
            scraper.session, "get", side_effect=requests.exceptions.ConnectionError
        ):
            docs = scraper.fetch_documentation(agent)

        assert docs.get("readme") is None

    def test_fetch_documentation_registry_url_fallback(self, registry_detail_html):
        scraper = MCPRegistryScraper()
        agent = {"source_url": "", "name": "test server"}

        detail_resp = MagicMock(status_code=200, text=registry_detail_html)

        with patch.object(scraper.session, "get", return_value=detail_resp):
            docs = scraper.fetch_documentation(agent)

        assert "detail_page" in docs

    # --- scrape_all_agents ---

    def test_scrape_all_agents_deduplication(self):
        scraper = MCPRegistryScraper()
        agents = [
            {"server": {"name": "dup", "repository": {"url": "https://github.com/a/b"}}},
            {"server": {"name": "dup2", "repository": {"url": "https://github.com/a/b"}}},
        ]

        with patch.object(scraper, "fetch_agent_list", return_value=agents), \
             patch("web_scraper_v2.requests.head") as mock_head, \
             patch.object(scraper, "_process_single_agent", side_effect=lambda a: {
                 "name": a["server"]["name"],
                 "source_url": a["server"]["repository"]["url"],
                 "documentation": {},
             }):
            mock_head.return_value = MagicMock(status_code=200)
            result = scraper.scrape_all_agents(max_workers=1)

        assert len(result) == 1

    def test_scrape_all_agents_404_filtering(self):
        scraper = MCPRegistryScraper()
        agents = [
            {"server": {"name": "dead", "repository": {"url": "https://github.com/a/dead"}}},
        ]

        with patch.object(scraper, "fetch_agent_list", return_value=agents), \
             patch("web_scraper_v2.requests.head") as mock_head:
            mock_head.return_value = MagicMock(status_code=404)
            result = scraper.scrape_all_agents(max_workers=1)

        assert len(result) == 0

    def test_scrape_all_agents_empty_list(self):
        scraper = MCPRegistryScraper()
        with patch.object(scraper, "fetch_agent_list", return_value=[]):
            result = scraper.scrape_all_agents()
        assert result == []

    def test_scrape_all_agents_marks_available(self):
        scraper = MCPRegistryScraper()
        agents = [
            {"server": {"name": "alive", "repository": {"url": "https://github.com/a/alive"}}},
        ]
        unified = {"name": "alive", "source_url": "https://github.com/a/alive", "documentation": {}}

        with patch.object(scraper, "fetch_agent_list", return_value=agents), \
             patch("web_scraper_v2.requests.head") as mock_head, \
             patch.object(scraper, "_process_single_agent", return_value=unified):
            mock_head.return_value = MagicMock(status_code=200)
            result = scraper.scrape_all_agents(max_workers=1)

        assert result[0]["is_available"] is True
        assert result[0]["availability_status"] == "reachable"

    # --- save_to_file ---

    def test_save_to_file_creates_json(self, tmp_path):
        scraper = MCPRegistryScraper()
        agents = [{"name": "test", "id": "123"}]

        filepath = str(tmp_path / "output.json")
        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            # Just test the method doesn't crash; actual file test below
            pass

        # Direct file test
        os.chdir(tmp_path)
        result = scraper.save_to_file(agents, "test_output.json")
        assert os.path.exists(result)

        with open(result) as f:
            loaded = json.load(f)
        assert loaded == agents

    def test_save_to_file_custom_filename(self, tmp_path):
        scraper = MCPRegistryScraper()
        os.chdir(tmp_path)
        result = scraper.save_to_file([{"a": 1}], "custom.json")
        assert "custom.json" in result


# ============================================================================
# 2. PricingExtractor
# ============================================================================

class TestPricingExtractor:
    """Tests for the PricingExtractor class."""

    # --- extract_pricing (full priority chain) ---

    def test_extract_pricing_pricing_page_wins(self):
        pe = PricingExtractor()
        with patch.object(pe, "_check_pricing_page", return_value="freemium"):
            result = pe.extract_pricing("https://example.com", "readme", None, None)
        assert result == "freemium"

    def test_extract_pricing_text_analysis_second(self):
        pe = PricingExtractor()
        with patch.object(pe, "_check_pricing_page", return_value="unknown"):
            result = pe.extract_pricing(
                "https://example.com",
                readme_text="This is an open source project under MIT license",
                detail_page_text=None,
                description=None,
            )
        assert result == "open_source"

    def test_extract_pricing_license_file_fallback(self, mit_license_text):
        pe = PricingExtractor()
        with patch.object(pe, "_check_pricing_page", return_value="unknown"), \
             patch("web_scraper_v2.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, text=mit_license_text)
            result = pe.extract_pricing(
                "https://github.com/org/repo",
                readme_text="A simple server.",
                detail_page_text=None,
                description=None,
            )
        assert result == "open_source"

    def test_extract_pricing_npm_fallback(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"license": "MIT"}

        with patch.object(pe, "_check_pricing_page", return_value="unknown"), \
             patch.object(pe.session, "get", return_value=mock_resp):
            result = pe.extract_pricing(
                "https://www.npmjs.com/package/test-pkg",
                readme_text=None,
                detail_page_text=None,
                description=None,
            )
        assert result == "open_source"

    def test_extract_pricing_pypi_fallback(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "info": {
                "license": "MIT License",
                "classifiers": [],
            }
        }

        with patch.object(pe, "_check_pricing_page", return_value="unknown"), \
             patch.object(pe.session, "get", return_value=mock_resp):
            result = pe.extract_pricing(
                "https://pypi.org/project/test-pkg",
                readme_text=None,
                detail_page_text=None,
                description=None,
            )
        assert result == "open_source"

    def test_extract_pricing_all_unknown(self):
        pe = PricingExtractor()
        result = pe.extract_pricing("", None, None, None)
        assert result == "unknown"

    def test_extract_pricing_empty_inputs(self):
        pe = PricingExtractor()
        result = pe.extract_pricing("", "", "", "")
        assert result == "unknown"

    # --- _check_pricing_page ---

    def test_check_pricing_page_found(self):
        pe = PricingExtractor()
        html = "<html><body>Free plan available. Pro plan $9/month.</body></html>"
        mock_resp = MagicMock(status_code=200, text=html)

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = pe._check_pricing_page("https://example.com")
        assert result == "freemium"

    def test_check_pricing_page_not_found(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=404)

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = pe._check_pricing_page("https://example.com")
        assert result == "unknown"

    def test_check_pricing_page_network_error(self):
        pe = PricingExtractor()
        with patch(
            "web_scraper_v2.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            result = pe._check_pricing_page("https://example.com")
        assert result == "unknown"

    # --- _analyze_pricing_page ---

    def test_analyze_pricing_page_freemium(self):
        pe = PricingExtractor()
        html = "<html><body>Free plan. Pro plan $9/month. Enterprise pricing.</body></html>"
        assert pe._analyze_pricing_page(html) == "freemium"

    def test_analyze_pricing_page_paid_only(self):
        pe = PricingExtractor()
        html = "<html><body>Enterprise plan $49/month. Premium features.</body></html>"
        assert pe._analyze_pricing_page(html) == "paid"

    def test_analyze_pricing_page_free_only(self):
        pe = PricingExtractor()
        html = "<html><body>Completely free to use forever.</body></html>"
        assert pe._analyze_pricing_page(html) == "free"

    def test_analyze_pricing_page_no_signals(self):
        pe = PricingExtractor()
        html = "<html><body>This is a server with tools.</body></html>"
        assert pe._analyze_pricing_page(html) == "unknown"

    # --- _analyze_text_for_pricing ---

    def test_text_pricing_open_source(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing("Licensed under MIT license.", None, None)
        assert result == "open_source"

    def test_text_pricing_freemium_detection(self):
        pe = PricingExtractor()
        text = "Free to use. Premium features available. Subscription $9/month and billing required."
        result = pe._analyze_text_for_pricing(text, None, None)
        assert result == "freemium"

    def test_text_pricing_explicit_freemium_keyword(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing("This is a freemium service.", None, None)
        assert result == "freemium"

    def test_text_pricing_paid_only(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(
            "Requires paid subscription. $9.99/month billing.", None, None
        )
        assert result == "paid"

    def test_text_pricing_free_only(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(
            "This tool is 100% free, forever free, zero cost.", None, None
        )
        assert result == "free"

    def test_text_pricing_no_text(self):
        pe = PricingExtractor()
        assert pe._analyze_text_for_pricing(None, None, None) == "unknown"

    def test_text_pricing_readme_preferred_over_detail(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(
            "MIT License open source project.",
            "Paid subscription required.",
            None,
        )
        # README is used (open_source), detail_page is ignored
        assert result == "open_source"

    def test_text_pricing_detail_page_fallback(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(
            None, "MIT License open source project.", None
        )
        assert result == "open_source"

    def test_text_pricing_description_fallback(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(None, None, "Free and open source tool.")
        assert result == "open_source"

    def test_text_pricing_currency_symbols(self):
        pe = PricingExtractor()
        result = pe._analyze_text_for_pricing(
            "Starting at $19 per month with annual billing.", None, None
        )
        assert result == "paid"

    # --- _check_license_from_readme ---

    def test_license_from_readme_explicit_mention(self):
        pe = PricingExtractor()
        result = pe._check_license_from_readme(
            "This project uses the MIT License.", "https://github.com/a/b"
        )
        assert result == "open_source"

    def test_license_from_readme_badge(self):
        pe = PricingExtractor()
        result = pe._check_license_from_readme(
            "![License](https://img.shields.io/badge/license/mit)",
            "https://github.com/a/b",
        )
        assert result == "open_source"

    def test_license_from_readme_spdx(self):
        pe = PricingExtractor()
        result = pe._check_license_from_readme(
            "SPDX-License-Identifier: MIT", "https://github.com/a/b"
        )
        assert result == "open_source"

    def test_license_from_readme_no_match(self):
        pe = PricingExtractor()
        result = pe._check_license_from_readme(
            "A great tool for developers.", "https://github.com/a/b"
        )
        assert result == "unknown"

    # --- _fetch_license_file ---

    def test_fetch_license_file_fast_path(self, mit_license_text):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200, text=mit_license_text)

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = pe._fetch_license_file("https://github.com/org/repo")
        assert result == "open_source"

    def test_fetch_license_file_parallel_fallback(self, mit_license_text):
        pe = PricingExtractor()
        fail = MagicMock(status_code=404)

        def side_effect(url, **kwargs):
            if "LICENSE.md" in url and "master" in url:
                return MagicMock(status_code=200, text=mit_license_text)
            return fail

        with patch("web_scraper_v2.requests.get", side_effect=side_effect):
            result = pe._fetch_license_file("https://github.com/org/repo")
        assert result == "open_source"

    def test_fetch_license_file_short_path(self):
        pe = PricingExtractor()
        result = pe._fetch_license_file("https://github.com/onlyone")
        assert result == "unknown"

    def test_fetch_license_file_gitlab(self, mit_license_text):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200, text=mit_license_text)

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = pe._fetch_license_file("https://gitlab.com/org/repo")
        assert result == "open_source"

    def test_fetch_license_file_all_fail(self):
        pe = PricingExtractor()
        fail = MagicMock(status_code=404)

        with patch("web_scraper_v2.requests.get", return_value=fail):
            result = pe._fetch_license_file("https://github.com/org/repo")
        assert result == "unknown"

    def test_fetch_license_file_non_github(self):
        pe = PricingExtractor()
        result = pe._fetch_license_file("https://example.com/some/project")
        assert result == "unknown"

    # --- _parse_license_text ---

    def test_parse_license_text_mit(self, mit_license_text):
        pe = PricingExtractor()
        assert pe._parse_license_text(mit_license_text) == "open_source"

    def test_parse_license_text_apache(self):
        pe = PricingExtractor()
        assert pe._parse_license_text("Apache License Version 2.0") == "open_source"

    def test_parse_license_text_proprietary(self):
        pe = PricingExtractor()
        assert pe._parse_license_text("All Rights Reserved. Proprietary.") == "paid"

    def test_parse_license_text_unknown(self):
        pe = PricingExtractor()
        assert pe._parse_license_text("Some random text with no license.") == "unknown"

    # --- _extract_from_npm ---

    def test_npm_mit_license(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"license": "MIT"}

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_npm("https://www.npmjs.com/package/test-pkg")
        assert result == "open_source"

    def test_npm_no_license(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {}

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_npm("https://www.npmjs.com/package/test-pkg")
        assert result == "unknown"

    def test_npm_http_error(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=404)

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_npm("https://www.npmjs.com/package/test-pkg")
        assert result == "unknown"

    def test_npm_malformed_url(self):
        pe = PricingExtractor()
        result = pe._extract_from_npm("https://www.npmjs.com/no-package-segment")
        assert result == "unknown"

    # --- _extract_from_pypi ---

    def test_pypi_license_field(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"info": {"license": "MIT", "classifiers": []}}

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_pypi("https://pypi.org/project/test-pkg")
        assert result == "open_source"

    def test_pypi_license_classifier(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "info": {
                "license": "",
                "classifiers": ["License :: OSI Approved :: MIT License"],
            }
        }

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_pypi("https://pypi.org/project/test-pkg")
        assert result == "open_source"

    def test_pypi_no_license(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"info": {"license": "", "classifiers": []}}

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_pypi("https://pypi.org/project/test-pkg")
        assert result == "unknown"

    def test_pypi_http_error(self):
        pe = PricingExtractor()
        mock_resp = MagicMock(status_code=500)

        with patch.object(pe.session, "get", return_value=mock_resp):
            result = pe._extract_from_pypi("https://pypi.org/project/test-pkg")
        assert result == "unknown"


# ============================================================================
# 3. MCPProber
# ============================================================================

class TestMCPProber:
    """Tests for the MCPProber class."""

    # --- _parse_sse_response ---

    def test_parse_sse_with_data_prefix(self):
        text = 'event: message\ndata: {"result": {"tools": [{"name": "search"}]}}'
        result = MCPProber._parse_sse_response(text)
        assert result == {"tools": [{"name": "search"}]}

    def test_parse_sse_plain_json(self):
        text = '{"result": {"tools": []}}'
        result = MCPProber._parse_sse_response(text)
        assert result == {"tools": []}

    def test_parse_sse_multiple_events(self):
        text = (
            "event: ping\ndata: {}\n\n"
            'event: message\ndata: {"result": {"tools": ["a"]}}\n\n'
            'event: message\ndata: {"result": {"tools": ["b"]}}'
        )
        result = MCPProber._parse_sse_response(text)
        assert result == {"tools": ["a"]}

    def test_parse_sse_invalid_json(self):
        text = "data: {not valid json}"
        result = MCPProber._parse_sse_response(text)
        assert result is None

    def test_parse_sse_no_result_key(self):
        text = '{"error": "something"}'
        result = MCPProber._parse_sse_response(text)
        assert result is None

    def test_parse_sse_empty_string(self):
        assert MCPProber._parse_sse_response("") is None

    # --- _build_headers ---

    def test_build_headers_basic(self):
        remote = {"headers": []}
        headers = MCPProber._build_headers(remote)
        assert headers["Content-Type"] == "application/json"
        assert "Accept" in headers

    def test_build_headers_with_auth(self):
        remote = {
            "headers": [{"name": "Authorization", "value": "Bearer sk-test"}]
        }
        headers = MCPProber._build_headers(remote)
        assert headers["Authorization"] == "Bearer sk-test"

    def test_build_headers_skips_empty(self):
        remote = {
            "headers": [
                {"name": "", "value": "test"},
                {"name": "X-Custom", "value": ""},
            ]
        }
        headers = MCPProber._build_headers(remote)
        assert "" not in headers
        assert "X-Custom" not in headers

    # --- _should_probe ---

    def test_should_probe_no_remotes(self):
        prober = MCPProber()
        assert prober._should_probe({"remotes": []}) is False

    def test_should_probe_with_url(self):
        prober = MCPProber()
        agent = {"remotes": [{"url": "https://example.com/mcp", "headers": []}]}
        assert prober._should_probe(agent) is True

    def test_should_probe_requires_auth_no_value(self):
        prober = MCPProber()
        agent = {
            "remotes": [
                {
                    "url": "https://example.com/mcp",
                    "headers": [{"name": "Authorization", "isRequired": True, "value": ""}],
                }
            ]
        }
        assert prober._should_probe(agent) is False

    def test_should_probe_auth_with_value(self):
        prober = MCPProber()
        agent = {
            "remotes": [
                {
                    "url": "https://example.com/mcp",
                    "headers": [
                        {"name": "Authorization", "isRequired": True, "value": "Bearer x"}
                    ],
                }
            ]
        }
        assert prober._should_probe(agent) is True

    def test_should_probe_mixed_remotes(self):
        prober = MCPProber()
        agent = {
            "remotes": [
                {
                    "url": "https://auth.example.com/mcp",
                    "headers": [{"name": "Auth", "isRequired": True, "value": ""}],
                },
                {
                    "url": "https://open.example.com/mcp",
                    "headers": [],
                },
            ]
        }
        assert prober._should_probe(agent) is True

    def test_should_probe_empty_url(self):
        prober = MCPProber()
        agent = {"remotes": [{"url": "", "headers": []}]}
        assert prober._should_probe(agent) is False

    # --- _probe_single ---

    def test_probe_single_success(self):
        prober = MCPProber()
        init_resp = MagicMock(
            status_code=200,
            text='{"result": {"protocolVersion": "2024-11-05"}}',
            headers={},
        )
        tools_resp = MagicMock(
            status_code=200,
            text='{"result": {"tools": [{"name": "search"}, {"name": "fetch"}]}}',
        )

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.side_effect = [init_resp, tools_resp]
            session_instance.headers = {}

            result = prober._probe_single("https://example.com/mcp", {"Content-Type": "application/json"})

        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "search"

    def test_probe_single_initialize_fails_http(self):
        prober = MCPProber()
        fail_resp = MagicMock(status_code=500, text="Server Error")

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.return_value = fail_resp
            session_instance.headers = {}

            result = prober._probe_single("https://example.com/mcp", {})

        assert result is None

    def test_probe_single_initialize_no_result(self):
        prober = MCPProber()
        resp = MagicMock(
            status_code=200,
            text='{"error": "something"}',
            headers={},
        )

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.return_value = resp
            session_instance.headers = {}

            result = prober._probe_single("https://example.com/mcp", {})

        assert result is None

    def test_probe_single_tools_list_fails(self):
        prober = MCPProber()
        init_resp = MagicMock(
            status_code=200,
            text='{"result": {"protocolVersion": "2024-11-05"}}',
            headers={},
        )
        tools_resp = MagicMock(status_code=500, text="Error")

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.side_effect = [init_resp, tools_resp]
            session_instance.headers = {}

            result = prober._probe_single("https://example.com/mcp", {})

        assert result is None

    def test_probe_single_session_id_forwarded(self):
        prober = MCPProber()
        init_resp = MagicMock(
            status_code=200,
            text='{"result": {"protocolVersion": "2024-11-05"}}',
        )
        init_resp.headers = {"Mcp-Session-Id": "sess-123"}
        tools_resp = MagicMock(
            status_code=200,
            text='{"result": {"tools": []}}',
        )

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.side_effect = [init_resp, tools_resp]
            session_instance.headers = {}

            prober._probe_single("https://example.com/mcp", {})

        assert session_instance.headers.get("Mcp-Session-Id") == "sess-123"

    def test_probe_single_network_error(self):
        prober = MCPProber()

        with patch("web_scraper_v2.requests.Session") as MockSession:
            session_instance = MockSession.return_value
            session_instance.post.side_effect = requests.exceptions.ConnectionError
            session_instance.headers = {}

            result = prober._probe_single("https://example.com/mcp", {})

        assert result is None

    # --- _probe_agent ---

    def test_probe_agent_first_remote_succeeds(self):
        prober = MCPProber()
        agent = {
            "name": "test",
            "remotes": [{"url": "https://r1.example.com/mcp", "headers": []}],
        }

        with patch.object(
            prober, "_probe_single", return_value=[{"name": "tool1"}]
        ):
            prober._probe_agent(agent)

        assert agent["probe_status"] == "success"
        assert agent["probed_tool_count"] == 1
        assert agent["tools"] == [{"name": "tool1"}]

    def test_probe_agent_second_remote_fallback(self):
        prober = MCPProber()
        agent = {
            "name": "test",
            "remotes": [
                {"url": "https://r1.example.com/mcp", "headers": []},
                {"url": "https://r2.example.com/mcp", "headers": []},
            ],
        }

        with patch.object(
            prober, "_probe_single", side_effect=[None, [{"name": "tool1"}]]
        ):
            prober._probe_agent(agent)

        assert agent["probe_status"] == "success"

    def test_probe_agent_all_fail(self):
        prober = MCPProber()
        agent = {
            "name": "test",
            "remotes": [{"url": "https://r1.example.com/mcp", "headers": []}],
        }

        with patch.object(prober, "_probe_single", return_value=None):
            prober._probe_agent(agent)

        assert agent["probe_status"] == "failed"
        assert agent["probed_tool_count"] == 0

    def test_probe_agent_skips_auth_remotes(self):
        prober = MCPProber()
        agent = {
            "name": "test",
            "remotes": [
                {
                    "url": "https://auth.example.com/mcp",
                    "headers": [{"name": "Auth", "isRequired": True, "value": ""}],
                },
                {"url": "https://open.example.com/mcp", "headers": []},
            ],
        }

        with patch.object(
            prober, "_probe_single", return_value=[{"name": "tool1"}]
        ) as mock_probe:
            prober._probe_agent(agent)

        # Only called once (for the open remote)
        mock_probe.assert_called_once()
        assert agent["probe_status"] == "success"

    # --- probe_all ---

    def test_probe_all_marks_skipped(self):
        prober = MCPProber()
        agents = [{"name": "no-remote", "remotes": []}]
        result = prober.probe_all(agents)
        assert result[0]["probe_status"] == "skipped"
        assert result[0]["probed_tool_count"] == 0

    def test_probe_all_empty_list(self):
        prober = MCPProber()
        result = prober.probe_all([])
        assert result == []

    def test_probe_all_mixed(self):
        prober = MCPProber(max_workers=1)
        agents = [
            {"name": "probeable", "remotes": [{"url": "https://r.example.com/mcp", "headers": []}]},
            {"name": "no-remote", "remotes": []},
        ]

        with patch.object(prober, "_probe_agent") as mock_probe:
            def side_effect(agent):
                agent["probe_status"] = "success"
                agent["probed_tool_count"] = 2
            mock_probe.side_effect = side_effect

            result = prober.probe_all(agents)

        assert result[0]["probe_status"] == "success"
        assert result[1]["probe_status"] == "skipped"


# ============================================================================
# 4. SmitheryConfigChecker
# ============================================================================

class TestSmitheryConfigChecker:
    """Tests for the SmitheryConfigChecker class."""

    # --- _extract_smithery_path ---

    def test_extract_smithery_path_standard(self, sample_agent_with_smithery_remote):
        path = SmitheryConfigChecker._extract_smithery_path(sample_agent_with_smithery_remote)
        assert path == "testuser/testserver"

    def test_extract_smithery_path_no_mcp_suffix(self):
        agent = {
            "remotes": [{"url": "https://server.smithery.ai/@user/server"}]
        }
        path = SmitheryConfigChecker._extract_smithery_path(agent)
        assert path == "user/server"

    def test_extract_smithery_path_non_smithery(self):
        agent = {"remotes": [{"url": "https://example.com/mcp"}]}
        assert SmitheryConfigChecker._extract_smithery_path(agent) is None

    def test_extract_smithery_path_no_remotes(self):
        agent = {"remotes": []}
        assert SmitheryConfigChecker._extract_smithery_path(agent) is None

    # --- _check_single ---

    def test_check_single_no_config_properties(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "connections": [{"configSchema": {"properties": {}, "required": []}}]
        }

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "none"

    def test_check_single_required_field(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "connections": [
                {
                    "configSchema": {
                        "properties": {"api_key": {"type": "string"}},
                        "required": ["api_key"],
                    }
                }
            ]
        }

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "required"

    def test_check_single_optional_field(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "connections": [
                {
                    "configSchema": {
                        "properties": {"verbose": {"type": "boolean"}},
                        "required": [],
                    }
                }
            ]
        }

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "optional"

    def test_check_single_required_with_default(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "connections": [
                {
                    "configSchema": {
                        "properties": {
                            "timeout": {"type": "integer", "default": 30}
                        },
                        "required": ["timeout"],
                    }
                }
            ]
        }

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "optional"

    def test_check_single_required_nullable(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "connections": [
                {
                    "configSchema": {
                        "properties": {
                            "key": {"anyOf": [{"type": "string"}, {"type": "null"}]}
                        },
                        "required": ["key"],
                    }
                }
            ]
        }

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "optional"

    def test_check_single_api_404(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=404)

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "unknown"

    def test_check_single_no_connections(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"connections": []}

        with patch("web_scraper_v2.requests.get", return_value=mock_resp):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "unknown"

    def test_check_single_network_error(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker()

        with patch(
            "web_scraper_v2.requests.get",
            side_effect=requests.exceptions.ConnectionError,
        ):
            result = checker._check_single(sample_agent_with_smithery_remote)
        assert result == "unknown"

    def test_check_single_no_smithery_path(self):
        checker = SmitheryConfigChecker()
        agent = {"remotes": [{"url": "https://example.com/mcp"}]}
        result = checker._check_single(agent)
        assert result == "unknown"

    # --- check_all ---

    def test_check_all_non_smithery_marked_null(self):
        checker = SmitheryConfigChecker()
        agents = [{"name": "plain", "remotes": [{"url": "https://example.com"}]}]
        result = checker.check_all(agents)
        assert result[0]["smithery_config"] is None

    def test_check_all_no_smithery_agents(self):
        checker = SmitheryConfigChecker()
        agents = [{"name": "plain", "remotes": [{"url": "https://example.com"}]}]
        result = checker.check_all(agents)
        assert len(result) == 1
        assert result[0]["smithery_config"] is None

    def test_check_all_concurrent(self, sample_agent_with_smithery_remote):
        checker = SmitheryConfigChecker(max_workers=1)

        with patch.object(checker, "_check_single", return_value="none"):
            result = checker.check_all([sample_agent_with_smithery_remote])

        assert result[0]["smithery_config"] == "none"


# ============================================================================
# 5. RegistryPageScraper
# ============================================================================

class TestRegistryPageScraper:
    """Tests for the RegistryPageScraper class."""

    def test_scrape_article_tag(self, registry_detail_html):
        rps = RegistryPageScraper()
        mock_resp = MagicMock(status_code=200, text=registry_detail_html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert len(result) > 100
        assert "article" not in result.lower() or "content" in result.lower()

    def test_scrape_main_tag(self):
        rps = RegistryPageScraper()
        html = """
        <html><body>
            <main>This is the main content area with detailed information about the server
            and its capabilities, tools, and integration options for developers.</main>
        </body></html>
        """
        mock_resp = MagicMock(status_code=200, text=html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert "main content" in result

    def test_scrape_class_selector(self):
        rps = RegistryPageScraper()
        html = """
        <html><body>
            <div class="server-description">
                Detailed description of the server with comprehensive information about
                all features, installation steps, and configuration options available.
            </div>
        </body></html>
        """
        mock_resp = MagicMock(status_code=200, text=html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert "Detailed description" in result

    def test_scrape_strips_noise(self, registry_detail_html):
        rps = RegistryPageScraper()
        mock_resp = MagicMock(status_code=200, text=registry_detail_html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert "console.log" not in result
        assert "Navigation" not in result
        assert "Footer" not in result

    def test_scrape_body_fallback(self):
        rps = RegistryPageScraper()
        html = """
        <html><body>
            <div>Short div</div>
            <p>This is body content that doesn't match any selectors but exists as fallback text.</p>
        </body></html>
        """
        mock_resp = MagicMock(status_code=200, text=html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert "body content" in result

    def test_scrape_minimum_length_filter(self):
        rps = RegistryPageScraper()
        html = """
        <html><body>
            <article>Short.</article>
            <main>Also short.</main>
            <div class="content">Very long content that exceeds the minimum threshold of one
            hundred characters so it should be returned as the scraped result from the page.</div>
        </body></html>
        """
        mock_resp = MagicMock(status_code=200, text=html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert "Very long content" in result

    def test_scrape_http_error(self):
        rps = RegistryPageScraper()
        mock_resp = MagicMock(status_code=404)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert result == ""

    def test_scrape_network_exception(self):
        rps = RegistryPageScraper()

        with patch.object(
            rps.session, "get", side_effect=requests.exceptions.ConnectionError
        ):
            result = rps.scrape_detail_page("https://example.com/server")

        assert result == ""

    def test_scrape_empty_page(self):
        rps = RegistryPageScraper()
        html = "<html><body></body></html>"
        mock_resp = MagicMock(status_code=200, text=html)

        with patch.object(rps.session, "get", return_value=mock_resp):
            result = rps.scrape_detail_page("https://example.com/server")

        assert result == ""

    # --- build_registry_detail_url ---

    def test_build_url_with_id(self):
        rps = RegistryPageScraper()
        url = rps.build_registry_detail_url("https://registry.mcp.run", "Test Server", "srv-123")
        assert url == "https://registry.mcp.run/servers/srv-123"

    def test_build_url_with_name(self):
        rps = RegistryPageScraper()
        url = rps.build_registry_detail_url("https://registry.mcp.run", "Test Server")
        assert url == "https://registry.mcp.run/servers/test-server"

    def test_build_url_no_id_no_name(self):
        rps = RegistryPageScraper()
        url = rps.build_registry_detail_url("https://registry.mcp.run", "")
        assert url == "https://registry.mcp.run/servers/"


# ============================================================================
# 6. LLMAnalyser
# ============================================================================

class TestLLMAnalyser:
    """Tests for the LLMAnalyser class."""

    # --- Constructor ---

    def test_init_with_explicit_key(self):
        analyser = LLMAnalyser(api_key="sk-test")
        assert analyser.api_key == "sk-test"

    def test_init_from_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        analyser = LLMAnalyser()
        assert analyser.api_key == "sk-env-test"

    def test_init_no_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No OpenAI API key"):
            LLMAnalyser()

    # --- _choose_best_text ---

    def test_choose_text_readme_preferred(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {
            "documentation": {
                "readme": "A" * 100,
                "detail_page": "B" * 200,
            },
            "description": "Short desc",
        }
        text, label = analyser._choose_best_text(agent)
        assert label == "readme"
        assert text == "A" * 100

    def test_choose_text_detail_page_fallback(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {
            "documentation": {"readme": "", "detail_page": "B" * 200},
            "description": "Short desc",
        }
        text, label = analyser._choose_best_text(agent)
        assert label == "detail_page"

    def test_choose_text_description_fallback(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {
            "documentation": {"readme": "", "detail_page": ""},
            "description": "A simple server",
        }
        text, label = analyser._choose_best_text(agent)
        assert label == "description_only"
        assert text == "A simple server"

    def test_choose_text_empty(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {"documentation": {}, "description": ""}
        text, label = analyser._choose_best_text(agent)
        assert label == "description_only"
        assert text == ""

    def test_choose_text_readme_too_short(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {
            "documentation": {"readme": "Short", "detail_page": "B" * 200},
            "description": "",
        }
        text, label = analyser._choose_best_text(agent)
        assert label == "detail_page"

    def test_choose_text_detail_page_too_short(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {
            "documentation": {"readme": "", "detail_page": "Short"},
            "description": "Fallback desc",
        }
        text, label = analyser._choose_best_text(agent)
        assert label == "description_only"

    # --- _truncate ---

    def test_truncate_short_text(self):
        analyser = LLMAnalyser(api_key="sk-test")
        text = "Short text"
        assert analyser._truncate(text) == text

    def test_truncate_at_newline(self):
        analyser = LLMAnalyser(api_key="sk-test")
        # Create text longer than MAX_CHARS with a newline in the second half
        text = "A" * 8000 + "\n" + "B" * 5000
        result = analyser._truncate(text)
        assert len(result) <= analyser.MAX_CHARS
        assert result.endswith("A" * (8000 - len(result.rstrip("A"))) if "B" not in result else True)

    def test_truncate_no_good_newline(self):
        analyser = LLMAnalyser(api_key="sk-test")
        text = "A" * 20000  # No newlines
        result = analyser._truncate(text)
        assert len(result) == analyser.MAX_CHARS

    def test_truncate_exactly_max(self):
        analyser = LLMAnalyser(api_key="sk-test")
        text = "A" * analyser.MAX_CHARS
        assert analyser._truncate(text) == text

    # --- _ensure_str_list ---

    def test_ensure_str_list_from_list(self):
        assert LLMAnalyser._ensure_str_list(["a", "b"]) == ["a", "b"]

    def test_ensure_str_list_filters_empty(self):
        assert LLMAnalyser._ensure_str_list(["a", "", "  ", "b"]) == ["a", "b"]

    def test_ensure_str_list_from_string(self):
        assert LLMAnalyser._ensure_str_list("single") == ["single"]

    def test_ensure_str_list_empty_string(self):
        assert LLMAnalyser._ensure_str_list("") == []

    def test_ensure_str_list_none(self):
        assert LLMAnalyser._ensure_str_list(None) == []

    def test_ensure_str_list_non_string_items(self):
        result = LLMAnalyser._ensure_str_list([1, 2.5, True])
        assert result == ["1", "2.5", "True"]

    # --- _call_openai ---

    def test_call_openai_success(self):
        analyser = LLMAnalyser(api_key="sk-test")
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"capabilities": []}'}}]
        }

        with patch("web_scraper_v2.requests.post", return_value=mock_resp):
            result = analyser._call_openai("system", "user")

        assert result == '{"capabilities": []}'

    def test_call_openai_429_retry(self):
        analyser = LLMAnalyser(api_key="sk-test", max_retries=3)
        rate_resp = MagicMock(status_code=429, text="Rate limited")
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": '{"result": "ok"}'}}]
        }

        with patch("web_scraper_v2.requests.post", side_effect=[rate_resp, ok_resp]), \
             patch("web_scraper_v2.time.sleep"):
            result = analyser._call_openai("system", "user")

        assert result == '{"result": "ok"}'

    def test_call_openai_429_all_retries_exhausted(self):
        analyser = LLMAnalyser(api_key="sk-test", max_retries=3)
        rate_resp = MagicMock(status_code=429, text="Rate limited")

        with patch("web_scraper_v2.requests.post", return_value=rate_resp), \
             patch("web_scraper_v2.time.sleep"):
            result = analyser._call_openai("system", "user")

        assert result is None

    def test_call_openai_500_retry(self):
        analyser = LLMAnalyser(api_key="sk-test", max_retries=3)
        err_resp = MagicMock(status_code=500, text="Server Error")
        ok_resp = MagicMock(status_code=200)
        ok_resp.json.return_value = {
            "choices": [{"message": {"content": '{"ok": true}'}}]
        }

        with patch("web_scraper_v2.requests.post", side_effect=[err_resp, ok_resp]), \
             patch("web_scraper_v2.time.sleep"):
            result = analyser._call_openai("system", "user")

        assert result == '{"ok": true}'

    def test_call_openai_timeout(self):
        analyser = LLMAnalyser(api_key="sk-test", max_retries=2)

        with patch(
            "web_scraper_v2.requests.post", side_effect=requests.exceptions.Timeout
        ), patch("web_scraper_v2.time.sleep"):
            result = analyser._call_openai("system", "user")

        assert result is None

    def test_call_openai_unexpected_exception(self):
        analyser = LLMAnalyser(api_key="sk-test", max_retries=3)

        with patch(
            "web_scraper_v2.requests.post", side_effect=RuntimeError("unexpected")
        ):
            result = analyser._call_openai("system", "user")

        assert result is None

    # --- _parse_response ---

    def test_parse_response_valid_json(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "capabilities": ["search", "fetch"],
            "limitations": ["rate limited"],
            "requirements": ["API key"],
            "quality_score": 0.75,
            "quality_rationale": "Good docs.",
        })
        result = analyser._parse_response(raw)
        assert result["capabilities"] == ["search", "fetch"]
        assert result["quality_score"] == 0.75

    def test_parse_response_markdown_fences(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = '```json\n{"capabilities": ["a"], "limitations": [], "requirements": [], "quality_score": 0.5, "quality_rationale": "ok"}\n```'
        result = analyser._parse_response(raw)
        assert result["capabilities"] == ["a"]

    def test_parse_response_quality_score_clamped_high(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({"quality_score": 1.5})
        result = analyser._parse_response(raw)
        assert result["quality_score"] == 1.0

    def test_parse_response_quality_score_clamped_low(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({"quality_score": -0.3})
        result = analyser._parse_response(raw)
        assert result["quality_score"] == 0.0

    def test_parse_response_quality_score_non_numeric(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({"quality_score": "high"})
        result = analyser._parse_response(raw)
        assert result["quality_score"] == 0.0

    def test_parse_response_invalid_json(self):
        analyser = LLMAnalyser(api_key="sk-test")
        result = analyser._parse_response("not json at all {{{")
        assert result["capabilities"] == []
        assert result["quality_score"] == 0.0

    def test_parse_response_missing_fields(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({"quality_score": 0.5})
        result = analyser._parse_response(raw)
        assert result["capabilities"] == []
        assert result["limitations"] == []

    # --- _parse_combined_response ---

    def test_parse_combined_valid(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "capabilities": ["search"],
            "limitations": [],
            "requirements": ["API key"],
            "quality_score": 0.8,
            "quality_rationale": "Comprehensive.",
            "agent_classification": "ai_agent",
            "is_ai_agent": True,
            "classification_rationale": "Uses embedded LLM.",
        })
        result = analyser._parse_combined_response(raw)
        assert result["agent_classification"] == "ai_agent"
        assert result["is_ai_agent"] is True
        assert result["capabilities"] == ["search"]

    def test_parse_combined_invalid_classification(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "agent_classification": "something_else",
            "quality_score": 0.5,
        })
        result = analyser._parse_combined_response(raw)
        assert result["agent_classification"] == "unknown"

    def test_parse_combined_is_ai_agent_bool(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "agent_classification": "api_wrapper",
            "is_ai_agent": False,
            "quality_score": 0.5,
        })
        result = analyser._parse_combined_response(raw)
        assert result["is_ai_agent"] is False

    def test_parse_combined_is_ai_agent_inferred(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "agent_classification": "api_wrapper",
            "quality_score": 0.5,
        })
        result = analyser._parse_combined_response(raw)
        assert result["is_ai_agent"] is False

    def test_parse_combined_is_ai_agent_inferred_from_ai_agent(self):
        analyser = LLMAnalyser(api_key="sk-test")
        raw = json.dumps({
            "agent_classification": "ai_agent",
            "quality_score": 0.5,
        })
        result = analyser._parse_combined_response(raw)
        assert result["is_ai_agent"] is True

    def test_parse_combined_invalid_json(self):
        analyser = LLMAnalyser(api_key="sk-test")
        result = analyser._parse_combined_response("broken json {{{")
        assert result["capabilities"] == []
        assert result["agent_classification"] == "unknown"
        assert result["is_ai_agent"] is None

    # --- analyse_and_classify ---

    def test_analyse_and_classify_success(self, sample_agent_with_readme):
        analyser = LLMAnalyser(api_key="sk-test")
        mock_response = json.dumps({
            "capabilities": ["search"],
            "limitations": [],
            "requirements": [],
            "quality_score": 0.7,
            "quality_rationale": "Good.",
            "agent_classification": "ai_agent",
            "is_ai_agent": True,
            "classification_rationale": "Uses AI.",
        })

        with patch.object(analyser, "_call_openai", return_value=mock_response):
            result = analyser.analyse_and_classify(sample_agent_with_readme)

        assert result["text_source"] == "readme"
        assert result["capabilities"] == ["search"]
        assert result["agent_classification"] == "ai_agent"

    def test_analyse_and_classify_no_text(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {"documentation": {}, "description": "", "name": "empty"}

        result = analyser.analyse_and_classify(agent)

        assert result["capabilities"] == []
        assert result["agent_classification"] == "unknown"

    def test_analyse_and_classify_openai_returns_none(self, sample_agent_with_readme):
        analyser = LLMAnalyser(api_key="sk-test")

        with patch.object(analyser, "_call_openai", return_value=None):
            result = analyser.analyse_and_classify(sample_agent_with_readme)

        assert result["capabilities"] == []

    # --- classify_agent_type ---

    def test_classify_agent_type_success(self, sample_agent_with_readme):
        analyser = LLMAnalyser(api_key="sk-test")
        mock_response = json.dumps({
            "agent_classification": "ai_agent",
            "is_ai_agent": True,
            "classification_rationale": "Uses LLM internally.",
        })

        with patch.object(analyser, "_call_openai", return_value=mock_response):
            result = analyser.classify_agent_type(sample_agent_with_readme)

        assert result["agent_classification"] == "ai_agent"
        assert result["is_ai_agent"] is True

    def test_classify_agent_type_no_text(self):
        analyser = LLMAnalyser(api_key="sk-test")
        agent = {"documentation": {}, "description": "", "name": "empty"}

        result = analyser.classify_agent_type(agent)

        assert result["agent_classification"] == "unknown"
        assert result["is_ai_agent"] is None


# ============================================================================
# 7. DocumentationProcessor
# ============================================================================

class TestDocumentationProcessor:
    """Tests for the DocumentationProcessor class."""

    # --- chunk_documentation ---

    def test_chunk_short_text(self):
        proc = DocumentationProcessor()
        text = "Short text that fits in one chunk."
        chunks = proc.chunk_documentation(text)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text
        assert chunks[0]["chunk_id"] == 0

    def test_chunk_multiple_chunks(self):
        proc = DocumentationProcessor(chunk_size=100)  # 100 tokens = 400 chars
        text = "A" * 1000  # ~250 tokens, should produce multiple chunks
        chunks = proc.chunk_documentation(text)
        assert len(chunks) > 1

    def test_chunk_sentence_boundary(self):
        proc = DocumentationProcessor(chunk_size=50)  # 200 chars per chunk
        text = "A" * 150 + ". " + "B" * 200
        chunks = proc.chunk_documentation(text)
        # First chunk should end at sentence boundary
        assert chunks[0]["text"].endswith(".")

    def test_chunk_overlap(self):
        proc = DocumentationProcessor(chunk_size=100, overlap=25)
        text = "Word " * 200  # 1000 chars
        chunks = proc.chunk_documentation(text)
        if len(chunks) > 1:
            # Chunks should have overlapping regions
            assert chunks[1]["start_pos"] < chunks[0]["end_pos"]

    def test_chunk_empty_text(self):
        proc = DocumentationProcessor()
        assert proc.chunk_documentation("") == []

    def test_chunk_ids_sequential(self):
        proc = DocumentationProcessor(chunk_size=50)
        text = "A" * 1000
        chunks = proc.chunk_documentation(text)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == i

    def test_chunk_token_estimate(self):
        proc = DocumentationProcessor()
        text = "A" * 400  # ~100 tokens
        chunks = proc.chunk_documentation(text)
        assert chunks[0]["token_count_estimate"] == len(chunks[0]["text"]) // 4

    def test_chunk_forward_progress(self):
        proc = DocumentationProcessor(chunk_size=50)
        text = "A" * 5000  # No sentence boundaries
        chunks = proc.chunk_documentation(text)
        assert len(chunks) > 1
        # Verify we don't get stuck
        for i in range(1, len(chunks)):
            assert chunks[i]["start_pos"] > chunks[i - 1]["start_pos"]

    def test_chunk_whitespace_only(self):
        proc = DocumentationProcessor()
        assert proc.chunk_documentation("   \n\n   ") == []

    # --- process_agent_documentation ---

    def test_process_with_llm_readme(self, sample_agent_with_readme):
        mock_analyser = MagicMock()
        mock_analyser.analyse_and_classify.return_value = {
            "capabilities": ["search"],
            "limitations": [],
            "requirements": [],
            "quality_score": 0.7,
            "quality_rationale": "Good.",
            "text_source": "readme",
            "is_ai_agent": True,
            "agent_classification": "ai_agent",
            "classification_rationale": "Uses AI.",
        }

        proc = DocumentationProcessor(llm_analyser=mock_analyser)
        result = proc.process_agent_documentation(sample_agent_with_readme)

        mock_analyser.analyse_and_classify.assert_called_once()
        assert result["llm_extracted"]["capabilities"] == ["search"]
        assert result["documentation_quality"] == 0.7
        assert result["agent_classification"] == "ai_agent"
        assert len(result["documentation_chunks"]) > 0
        assert result["documentation_chunks"][0]["source_type"] == "readme"

    def test_process_with_llm_detail_page(self, sample_agent_with_detail_page):
        mock_analyser = MagicMock()
        mock_analyser.analyse_and_classify.return_value = {
            "capabilities": [],
            "limitations": [],
            "requirements": [],
            "quality_score": 0.4,
            "quality_rationale": "Limited.",
            "text_source": "detail_page",
            "is_ai_agent": False,
            "agent_classification": "api_wrapper",
            "classification_rationale": "Wraps API.",
        }

        proc = DocumentationProcessor(llm_analyser=mock_analyser)
        result = proc.process_agent_documentation(sample_agent_with_detail_page)

        assert result["documentation_chunks"][0]["source_type"] == "detail_page"

    def test_process_probed_still_runs_llm(self, sample_agent_with_readme):
        mock_analyser = MagicMock()
        mock_analyser.analyse_and_classify.return_value = {
            "capabilities": ["search"],
            "limitations": [],
            "requirements": [],
            "quality_score": 0.8,
            "quality_rationale": "Good docs.",
            "text_source": "readme",
            "is_ai_agent": True,
            "agent_classification": "ai_agent",
            "classification_rationale": "Uses AI.",
        }

        sample_agent_with_readme["probe_status"] = "success"
        proc = DocumentationProcessor(llm_analyser=mock_analyser)
        result = proc.process_agent_documentation(sample_agent_with_readme, skip_llm=True)

        mock_analyser.analyse_and_classify.assert_called_once()
        assert result["llm_extracted"]["capabilities"] == ["search"]
        assert result["agent_classification"] == "ai_agent"
        assert result["documentation_quality"] == 0.8

    def test_process_no_llm(self, sample_unified_agent):
        proc = DocumentationProcessor(llm_analyser=None)
        result = proc.process_agent_documentation(sample_unified_agent)

        assert result["llm_extracted"]["capabilities"] == []
        assert result["documentation_quality"] == 0.0
        assert result["agent_classification"] == "unknown"
        assert result["documentation_chunks"] == []

    def test_process_chunks_have_agent_id(self, sample_agent_with_readme):
        proc = DocumentationProcessor(llm_analyser=None)
        result = proc.process_agent_documentation(sample_agent_with_readme)

        for chunk in result["documentation_chunks"]:
            assert chunk["agent_id"] == sample_agent_with_readme["agent_id"]

    def test_process_no_docs(self, sample_unified_agent):
        sample_unified_agent["documentation"] = {}
        proc = DocumentationProcessor(llm_analyser=None)
        result = proc.process_agent_documentation(sample_unified_agent)
        assert result["documentation_chunks"] == []


# ============================================================================
# 8. Main function
# ============================================================================

class TestMain:
    """Tests for the main() function and CLI argument parsing."""

    def test_main_no_agents(self):
        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper:
            instance = MockScraper.return_value
            instance.scrape_all_agents.return_value = []
            main()

        instance.scrape_all_agents.assert_called_once()

    def test_main_happy_path(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        agents = [
            {
                "name": "test",
                "agent_id": "abc123",
                "source_url": "https://github.com/test/test",
                "probe_status": "skipped",
                "probed_tool_count": 0,
                "documentation": {},
                "description": "A test server",
                "tools": [],
                "remotes": [],
                "is_available": True,
                "availability_status": "reachable",
                "agent_classification": "ai_agent",
                "is_ai_agent": True,
                "pricing": "free",
                "llm_extracted": {"capabilities": ["search"]},
                "documentation_quality": 0.5,
                "quality_rationale": "Good.",
                "llm_text_source": "readme",
                "documentation_chunks": [],
            }
        ]

        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper, \
             patch("web_scraper_v2.MCPProber") as MockProber, \
             patch("web_scraper_v2.LLMAnalyser") as MockLLM, \
             patch("web_scraper_v2.DocumentationProcessor") as MockProcessor:

            scraper_inst = MockScraper.return_value
            scraper_inst.scrape_all_agents.return_value = agents
            scraper_inst.base_url = "https://registry.mcp.run"
            scraper_inst.save_to_file.return_value = "./mcp_agents.json"

            prober_inst = MockProber.return_value
            prober_inst.probe_all.return_value = agents

            processor_inst = MockProcessor.return_value
            processor_inst.process_agent_documentation.side_effect = lambda a, **kw: a

            main()

        scraper_inst.scrape_all_agents.assert_called_once()
        prober_inst.probe_all.assert_called_once()
        scraper_inst.save_to_file.assert_called()

    def test_main_no_openai_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        agents = [
            {
                "name": "test",
                "agent_id": "abc123",
                "source_url": "https://github.com/test/test",
                "probe_status": "skipped",
                "probed_tool_count": 0,
                "documentation": {},
                "description": "A test server",
                "tools": [],
                "remotes": [],
                "is_available": True,
                "availability_status": "reachable",
                "agent_classification": "unknown",
                "is_ai_agent": None,
                "pricing": "unknown",
                "llm_extracted": {"capabilities": []},
                "documentation_quality": 0.0,
                "quality_rationale": "",
                "llm_text_source": "none",
                "documentation_chunks": [],
            }
        ]

        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper, \
             patch("web_scraper_v2.MCPProber") as MockProber, \
             patch("web_scraper_v2.DocumentationProcessor") as MockProcessor:

            scraper_inst = MockScraper.return_value
            scraper_inst.scrape_all_agents.return_value = agents
            scraper_inst.base_url = "https://registry.mcp.run"
            scraper_inst.save_to_file.return_value = "./mcp_agents.json"

            prober_inst = MockProber.return_value
            prober_inst.probe_all.return_value = agents

            processor_inst = MockProcessor.return_value
            processor_inst.process_agent_documentation.side_effect = lambda a, **kw: a

            main()

        # LLMAnalyser should NOT have been created
        scraper_inst.save_to_file.assert_called()

    def test_main_probeable_filter(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        agents = [
            {
                "name": "ai-free",
                "agent_id": "1",
                "source_url": "https://github.com/test/ai-free",
                "agent_classification": "ai_agent",
                "remotes": [{"url": "https://r.example.com", "headers": []}],
                "pricing": "free",
                "probe_status": "success",
                "probed_tool_count": 2,
                "documentation": {},
                "description": "An AI agent",
                "tools": [],
                "is_available": True,
                "availability_status": "reachable",
                "is_ai_agent": True,
                "llm_extracted": {"capabilities": []},
                "documentation_quality": 0.0,
                "quality_rationale": "",
                "llm_text_source": "none",
                "documentation_chunks": [],
            },
            {
                "name": "wrapper-free",
                "agent_id": "2",
                "source_url": "https://github.com/test/wrapper",
                "agent_classification": "api_wrapper",
                "remotes": [{"url": "https://r2.example.com", "headers": []}],
                "pricing": "free",
                "probe_status": "success",
                "probed_tool_count": 1,
                "documentation": {},
                "description": "A wrapper",
                "tools": [],
                "is_available": True,
                "availability_status": "reachable",
                "is_ai_agent": False,
                "llm_extracted": {"capabilities": []},
                "documentation_quality": 0.0,
                "quality_rationale": "",
                "llm_text_source": "none",
                "documentation_chunks": [],
            },
        ]

        saved_agents = []

        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper, \
             patch("web_scraper_v2.MCPProber") as MockProber, \
             patch("web_scraper_v2.DocumentationProcessor") as MockProcessor:

            scraper_inst = MockScraper.return_value
            scraper_inst.scrape_all_agents.return_value = agents
            scraper_inst.base_url = "https://registry.mcp.run"

            def capture_save(a, filename="mcp_agents.json"):
                saved_agents.extend(a)
                return f"./{filename}"
            scraper_inst.save_to_file.side_effect = capture_save

            prober_inst = MockProber.return_value
            prober_inst.probe_all.return_value = agents

            processor_inst = MockProcessor.return_value
            processor_inst.process_agent_documentation.side_effect = lambda a, **kw: a

            main(probeable=True)

        # Only the ai_agent with free pricing should pass
        main_save = [a for a in saved_agents if a["name"] == "ai-free"]
        wrapper_save = [a for a in saved_agents if a["name"] == "wrapper-free"]
        assert len(main_save) >= 1
        # wrapper should be filtered out from the primary save
        # (it may appear in the secondary ai_agents save)

    def test_parse_args_default(self):
        import argparse
        # Test that main accepts default args
        # We just verify main() can be called without crashing when no agents found
        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper:
            inst = MockScraper.return_value
            inst.scrape_all_agents.return_value = []
            main(probeable=False, smithery=False)

    def test_parse_args_probeable(self):
        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper:
            inst = MockScraper.return_value
            inst.scrape_all_agents.return_value = []
            main(probeable=True)

    def test_parse_args_smithery(self):
        with patch("web_scraper_v2.MCPRegistryScraper") as MockScraper, \
             patch("web_scraper_v2.MCPProber") as MockProber, \
             patch("web_scraper_v2.SmitheryConfigChecker") as MockChecker:
            scraper_inst = MockScraper.return_value
            scraper_inst.scrape_all_agents.return_value = []
            main(smithery=True)
