"""Shared fixtures for web_scraper_v2 tests."""

import sys
import os
import pytest

# Ensure the project root is on sys.path so we can import web_scraper_v2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_raw_agent():
    """Minimal raw agent dict as returned by the registry API."""
    return {
        "server": {
            "name": "test-server",
            "description": "A test MCP server for search and analysis",
            "repository": {"url": "https://github.com/test-org/test-server"},
            "websiteUrl": "https://test-server.example.com",
            "tools": [
                {"name": "search", "description": "Search for items"},
                {"name": "analyze", "description": "Analyze data"},
            ],
            "remotes": [
                {
                    "url": "https://remote.example.com/mcp",
                    "transportType": "streamable-http",
                    "headers": [],
                }
            ],
            "capabilities": ["search", "analysis"],
        },
        "_meta": {
            "io.modelcontextprotocol.registry/official": {
                "updatedAt": "2025-01-15T10:00:00Z"
            }
        },
    }


@pytest.fixture
def sample_unified_agent():
    """A fully populated unified-schema agent dict."""
    return {
        "agent_id": "abc123def4567890",
        "name": "test-server",
        "source": "mcp",
        "source_url": "https://github.com/test-org/test-server",
        "description": "A test MCP server for search and analysis",
        "tools": [
            {"name": "search", "description": "Search for items"},
            {"name": "analyze", "description": "Analyze data"},
        ],
        "detected_capabilities": ["search", "tool:search", "tool:analyze"],
        "pricing": "unknown",
        "last_updated": "2025-01-15T10:00:00Z",
        "is_available": None,
        "availability_status": "unknown",
        "is_ai_agent": None,
        "agent_classification": "unknown",
        "classification_rationale": "",
        "remotes": [
            {
                "url": "https://remote.example.com/mcp",
                "transportType": "streamable-http",
                "headers": [],
            }
        ],
        "documentation": {},
    }


@pytest.fixture
def sample_agent_with_readme(sample_unified_agent):
    """Unified agent with documentation.readme populated."""
    sample_unified_agent["documentation"] = {
        "readme": "# Test Server\n\nThis is a comprehensive README with installation "
        "instructions, usage examples, and detailed tool descriptions.\n\n"
        "## Installation\n\n```bash\nnpm install test-server\n```\n\n"
        "## Usage\n\nRun the server with `npx test-server`.\n\n"
        "## Tools\n\n- search: Search for items\n- analyze: Analyze data\n"
    }
    return sample_unified_agent


@pytest.fixture
def sample_agent_with_detail_page(sample_unified_agent):
    """Unified agent with documentation.detail_page populated."""
    sample_unified_agent["documentation"] = {
        "detail_page": "Test Server - An MCP server that provides search and analysis "
        "capabilities. It supports multiple data sources and can handle "
        "concurrent requests efficiently. The server exposes two main tools: "
        "search and analyze."
    }
    return sample_unified_agent


@pytest.fixture
def sample_agent_with_remotes():
    """Unified agent with remotes[] containing a probeable URL."""
    return {
        "agent_id": "probe123",
        "name": "probeable-server",
        "source": "mcp",
        "source_url": "https://github.com/test-org/probeable",
        "description": "A probeable server",
        "tools": [],
        "remotes": [
            {
                "url": "https://remote.example.com/mcp",
                "transportType": "streamable-http",
                "headers": [],
            }
        ],
        "documentation": {},
    }


@pytest.fixture
def sample_agent_with_smithery_remote():
    """Unified agent with a server.smithery.ai remote URL."""
    return {
        "agent_id": "smithery123",
        "name": "smithery-server",
        "source": "mcp",
        "source_url": "",
        "description": "A Smithery-hosted server",
        "tools": [],
        "remotes": [
            {
                "url": "https://server.smithery.ai/@testuser/testserver/mcp",
                "transportType": "streamable-http",
                "headers": [],
            }
        ],
        "documentation": {},
    }


@pytest.fixture
def registry_detail_html():
    """Sample HTML for a registry detail page with article/main/section tags."""
    return """
    <html>
    <head><title>Test Server</title></head>
    <body>
        <nav>Navigation</nav>
        <header>Header content</header>
        <article>
            This is the main article content for the test server. It contains
            a detailed description of the server's capabilities, including search
            functionality, data analysis tools, and integration options.
            The server supports multiple protocols and can handle concurrent requests.
        </article>
        <footer>Footer content</footer>
        <script>console.log('noise')</script>
        <style>.hidden { display: none; }</style>
    </body>
    </html>
    """


@pytest.fixture
def mit_license_text():
    """Full text of an MIT license file."""
    return """MIT License

Copyright (c) 2025 Test Organization

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
"""
