"""
MCP Registry Web Scraper
Fetches AI agent metadata from MCP registry and converts to unified schema format.

LLM Analysis Pipeline
---------------------
This module uses an LLM (GPT-4 / GPT-5 via OpenAI API) to:
  1. Extract capabilities, limitations, and requirements from server documentation.
  2. Assign a documentation quality score (0.0 – 1.0).

Text source priority (best signal → least signal):
  1. README file fetched from GitHub (richest, structured, written for humans)
  2. Registry detail page scraped text (if no GitHub link exists)
  3. Plain `description` field from the registry API (last resort, often one-liner)

The README is strongly preferred because it contains installation steps, usage
examples, tool tables, and limitations that the registry description almost never
includes.  When only the description is available the LLM prompt is adjusted so
the model knows it is working with limited information and adjusts the quality
score accordingly.

Set the environment variable OPENAI_API_KEY before running:
    export OPENAI_API_KEY="sk-..."
"""

import os
import re
import time
import argparse
import requests
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv

load_dotenv()


class MCPRegistryScraper:
    """
    Scraper for MCP (Model Context Protocol) Registry to fetch AI agent metadata.
    Converts MCP schema to unified agent format for indexing.
    """
    
    def __init__(self, base_url: str = "https://registry.mcp.run"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MCP-Agent-Search-Engine/1.0',
            'Accept': 'application/json'
        })
        self.pricing_extractor = PricingExtractor(session=self.session)
    
    def fetch_agent_list(self, max_results: Optional[int] = None) -> List[Dict]:
        """
        Fetch the list of available agents from the MCP registry with pagination.
        
        Args:
            max_results: Maximum number of results to fetch (None = fetch all)
        
        Returns:
            List of agent metadata dictionaries
        """
        try:
            all_agents: List[Dict] = []
            cursor: Optional[str] = None
            per_page = 100  # API max page size
            base_endpoint = 'https://registry.modelcontextprotocol.io/v0.1/servers'

            print(f"  🔄 Starting cursor pagination (limit per page: {per_page})")

            page = 1
            while True:
                params = {"limit": per_page}
                if cursor:
                    params["cursor"] = cursor

                try:
                    print(f"  📡 Fetching page {page}...")
                    response = self.session.get(base_endpoint, params=params, timeout=10)

                    if response.status_code != 200:
                        print(f"  ⚠ HTTP {response.status_code}, stopping pagination")
                        break

                    data = response.json()

                    # Expected shape: {"servers": [...], "metadata": {"count": N, "nextCursor": "..."}}
                    if isinstance(data, dict):
                        page_results = data.get("servers") or data.get("agents") or data.get("data") or data.get("items") or []
                        if not isinstance(page_results, list):
                            page_results = []

                        all_agents.extend(page_results)
                        print(f"  ✓ Retrieved {len(page_results)} agents (total so far: {len(all_agents)})")

                        if max_results and len(all_agents) >= max_results:
                            print(f"  🎯 Reached requested limit of {max_results} agents")
                            return all_agents[:max_results]

                        meta = data.get("metadata", {}) or {}
                        next_cursor = meta.get("nextCursor") or meta.get("next_cursor")

                        if not next_cursor:
                            print("  ✅ Reached end of available data")
                            break

                        cursor = next_cursor
                        page += 1
                        continue

                    # If the API ever returns a bare list, handle it
                    if isinstance(data, list):
                        all_agents.extend(data)
                        print(f"  ✓ Retrieved {len(data)} agents (total so far: {len(all_agents)})")
                        break

                    print("  ⚠ Unexpected response shape, stopping pagination")
                    break

                except requests.exceptions.RequestException as e:
                    print(f"  ✗ Request failed: {e}")
                    break

        except Exception as e:
            print(f"Error fetching agent list: {e}")
            return []
    
    def _scrape_html_listing(self) -> List[Dict]:
        """
        Fallback: Scrape HTML page if API is not available.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("BeautifulSoup not installed. Install with: pip install beautifulsoup4 --break-system-packages")
            return []
        
        try:
            response = self.session.get(self.base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            agents = []
            # Look for common patterns in agent listings
            for card in soup.find_all(['div', 'article', 'li'], class_=lambda x: x and any(
                term in str(x).lower() for term in ['agent', 'card', 'server', 'item']
            )):
                agent_data = self._extract_agent_from_html(card)
                if agent_data:
                    agents.append(agent_data)
            
            print(f"Found {len(agents)} agents from HTML scraping")
            return agents
            
        except Exception as e:
            print(f"Error scraping HTML: {e}")
            return []
    
    def _extract_agent_from_html(self, element) -> Optional[Dict]:
        """Extract agent data from HTML element."""
        try:
            # Template extraction - adjust selectors based on actual HTML
            name_elem = element.find(['h2', 'h3', 'h4', 'a', 'strong'])
            name = name_elem.get_text(strip=True) if name_elem else "Unknown"
            
            description_elem = element.find(['p', 'div', 'span'], class_=lambda x: x and 'description' in str(x).lower())
            description = description_elem.get_text(strip=True) if description_elem else ""
            
            link = element.find('a')
            url = link['href'] if link and link.get('href') else ""
            
            return {
                'name': name,
                'description': description,
                'url': urljoin(self.base_url, url) if url else "",
                'source': 'mcp_html'
            }
        except:
            return None
    
    def fetch_agent_details(self, agent_identifier: str) -> Dict:
        """
        Fetch detailed information for a specific agent.
        
        Args:
            agent_identifier: Agent name, ID, or URL path
            
        Returns:
            Detailed agent metadata
        """
        try:
            # Try different endpoint patterns
            endpoints = [
                f"{self.base_url}/api/agents/{agent_identifier}",
                f"{self.base_url}/api/v1/agents/{agent_identifier}",
                f"{self.base_url}/api/servers/{agent_identifier}",
            ]
            
            for endpoint in endpoints:
                try:
                    response = self.session.get(endpoint, timeout=10)
                    if response.status_code == 200:
                        return response.json()
                except:
                    continue
                    
            return {}
                
        except Exception as e:
            print(f"  Error fetching details: {e}")
            return {}
    
    def convert_to_unified_schema(self, mcp_agent: Dict) -> Dict:
        """
        Convert MCP schema to unified agent schema.
        
        Args:
            mcp_agent: Agent data in MCP format
            
        Returns:
            Agent data in unified schema format
        """
        # Generate unique agent_id from source and name
        mcp_agent_meta = mcp_agent.get('_meta')
        mcp_agent = mcp_agent.get('server')
        source_name = mcp_agent.get('name', 'unknown')
        agent_id = hashlib.md5(f"mcp_{source_name}".encode()).hexdigest()[:16]
        
        # Extract source URL
        source_url = (
            mcp_agent.get('repository', {}).get('url') or 
            mcp_agent.get('websiteUrl') or
            mcp_agent.get('remotes', [{}])[0].get('url') or 
            ''
        )
        
        unified = {
            # Identity
            'agent_id': agent_id,
            'name': mcp_agent.get('name', 'Unknown'),
            'source': 'mcp',
            'source_url': source_url,
            
            # Description
            'description': mcp_agent.get('description', ''),
            # 'short_description': (mcp_agent.get('description', '')[:200] + '...') if len(mcp_agent.get('description', '')) > 200 else mcp_agent.get('description', ''),
            
            # Capabilities
            'tools': mcp_agent.get('tools', []),
            'detected_capabilities': self._extract_capabilities(mcp_agent),
            'llm_backbone': mcp_agent.get('framework') or mcp_agent.get('llm_backbone') or 'Unknown',
            
            # Evaluation Data
            'arena_elo': mcp_agent.get('arena_elo'),
            'arena_battles': mcp_agent.get('arena_battles'),
            'community_rating': mcp_agent.get('rating'),
            'rating_count': mcp_agent.get('rating_count', 0),
            
            # Metadata
            'pricing': mcp_agent.get('pricing', 'unknown'),
            'last_updated': mcp_agent_meta.get('io.modelcontextprotocol.registry/official').get('updatedAt', 'Unknown'),
            'indexed_at': datetime.utcnow().isoformat(),
            
            # Computed
            'description_embedding': None,
            'testability_tier': 'n/a',

            # Availability & classification (populated by later pipeline steps)
            'is_available': None,
            'availability_status': 'unknown',
            'is_ai_agent': None,
            'agent_classification': 'unknown',
            'classification_rationale': '',
            
            # Remote endpoints (live MCP server URLs)
            'remotes': mcp_agent.get('remotes', []),

            # Raw data
            # '_raw_mcp_data': mcp_agent
        }
        
        return unified
    
    def _extract_capabilities(self, mcp_agent: Dict) -> List[str]:
        """Extract structured capability list from MCP agent data."""
        capabilities = []
        
        # Direct capabilities field
        if 'capabilities' in mcp_agent:
            caps = mcp_agent['capabilities']
            if isinstance(caps, list):
                capabilities.extend([str(c) for c in caps])
            elif isinstance(caps, dict):
                capabilities.extend([str(k) for k in caps.keys()])
        
        # Infer from tools
        if 'tools' in mcp_agent:
            for tool in mcp_agent['tools']:
                if isinstance(tool, dict):
                    tool_name = tool.get('name') or tool.get('type') or str(tool)
                    capabilities.append(f"tool:{tool_name}")
                elif isinstance(tool, str):
                    capabilities.append(f"tool:{tool}")
        
        # Extract from description
        desc = mcp_agent.get('description', '').lower()
        capability_keywords = ['search', 'generate', 'analyze', 'process', 'create', 'manage', 'monitor']
        for keyword in capability_keywords:
            if keyword in desc:
                capabilities.append(keyword)
        
        return list(set(capabilities))
    
    def fetch_documentation(self, agent: Dict) -> Dict[str, str]:
        """
        Fetch documentation for an agent.
        
        Priority order:
          1. README from GitHub (if available)
          2. Registry HTML detail page (fallback when no GitHub/README)
        
        Args:
            agent: Agent metadata (raw or unified)
            
        Returns:
            Dictionary with documentation sources. Keys can be:
              - 'readme': GitHub README text
              - 'detail_page': scraped registry HTML detail page text
        """
        docs = {}
        
        # -----------------------------------------------------------------
        # Step 1: Try to fetch README from GitHub
        # -----------------------------------------------------------------
        readme_urls = []
        
        # Check for explicit documentation URLs
        for key in ['readme_url', 'documentation_url', 'docs_url']:
            if key in agent and agent[key]:
                readme_urls.append(agent[key])
        
        # Try to construct GitHub README URL if repository is available
        repo_url = agent.get('source_url') or agent.get('websiteUrl', '')
        if 'github.com' in repo_url:
            # Clean up URL
            repo_url = repo_url.rstrip('/')
            readme_urls.append(f"{repo_url}/raw/main/README.md")
            readme_urls.append(f"{repo_url}/raw/master/README.md")
        
        # Fetch README
        for url in readme_urls:
            if not url:
                continue
            try:
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    docs['readme'] = response.text
                    print(f"    ✓ Fetched README from {url}")
                    break
            except Exception as e:
                print(f"    ✗ Failed to fetch from {url}: {e}")
                continue
        
        # -----------------------------------------------------------------
        # Step 2: If no README found, scrape the registry detail page HTML
        # -----------------------------------------------------------------
        if not docs.get('readme'):
            print(f"    ℹ  No README found — attempting to scrape registry detail page...")
            
            # Import BeautifulSoup here (lazy import)
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                print(f"    ⚠  beautifulsoup4 not installed — cannot scrape detail page.")
                return docs
            
            # Determine which URL to scrape for documentation
            source_url = agent.get('source_url', '')
            detail_page_url = None
            
            # Priority: Always try to scrape source_url if it exists
            # This works for:
            # - Regular websites (e.g., https://example.com/mcp-server)
            # - GitHub repos without README (e.g., https://github.com/org/repo)
            # - NPM packages (e.g., https://npmjs.com/package/name)
            # - Any other valid URL
            
            if source_url and source_url.startswith('http'):
                detail_page_url = source_url
                print(f"    → Scraping source URL: {detail_page_url}")
            else:
                # Only fall back to registry URL if there's no source_url at all
                name = agent.get('name', '')
                raw_id = (agent.get('_raw_mcp_data') or {}).get('id') or name
                slug = (raw_id or name).lower().replace(' ', '-')
                detail_page_url = f"{self.base_url}/servers/{slug}"
                print(f"    → No source URL, trying registry: {detail_page_url}")
            
            # Scrape the detail page
            if detail_page_url:
                try:
                    response = self.session.get(detail_page_url, timeout=12)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Remove noise
                        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                            tag.decompose()
                        
                        # Try to find the main content area with priority selectors
                        content = None
                        selectors = [
                            ('tag',   'article'),
                            ('tag',   'main'),
                            ('class', 'description'),
                            ('class', 'readme'),
                            ('class', 'content'),
                            ('class', 'detail'),
                            ('class', 'server-detail'),
                            ('class', 'prose'),
                            ('tag',   'section'),
                        ]
                        
                        for kind, value in selectors:
                            if kind == 'tag':
                                element = soup.find(value)
                            else:
                                element = soup.find(class_=lambda c: c and value in c.lower())
                            
                            if element:
                                text = element.get_text(separator='\n', strip=True)
                                if len(text) > 100:   # Only accept if substantial
                                    content = text
                                    break
                        
                        # Fallback: use full body
                        if not content:
                            body = soup.find('body')
                            if body:
                                content = body.get_text(separator='\n', strip=True)
                        
                        if content:
                            docs['detail_page'] = content
                            print(f"    ✓ Scraped registry detail page ({len(content)} chars)")
                        else:
                            print(f"    ✗ No content found on detail page")
                    else:
                        print(f"    ✗ Detail page returned HTTP {response.status_code}")
                        
                except Exception as e:
                    print(f"    ✗ Failed to scrape detail page: {e}")
        
        return docs
    
    def _process_single_agent(self, agent_data: Dict) -> Optional[Dict]:
        """
        Process a single agent: convert schema, fetch docs, extract pricing.

        Returns the unified agent dict, or None if it should be skipped.
        """
        agent_summary = agent_data.get('server')
        agent_name = agent_summary.get('name', 'Unknown')

        # Convert to unified schema
        unified_agent = self.convert_to_unified_schema(agent_data)

        # Fetch documentation
        docs = self.fetch_documentation(unified_agent)
        unified_agent['documentation'] = docs

        # Extract pricing
        extracted_pricing = self.pricing_extractor.extract_pricing(
            source_url=unified_agent.get('source_url', ''),
            readme_text=docs.get('readme'),
            detail_page_text=docs.get('detail_page'),
            description=unified_agent.get('description'),
        )

        if extracted_pricing != "unknown":
            unified_agent['pricing'] = extracted_pricing

        return unified_agent

    def scrape_all_agents(self, limit: Optional[int] = None, max_workers: int = 10) -> List[Dict]:
        """
        Main method: Scrape all agents from registry with parallel processing.

        Deduplicates by URL and skips dead (404) links before any heavy work.
        Uses ThreadPoolExecutor for concurrent doc fetching and pricing.

        Args:
            limit:       Maximum number of agents to fetch.
            max_workers: Number of concurrent threads for processing.

        Returns:
            List of agents in unified schema format.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print("🔍 Fetching agent list from MCP registry...")
        agent_list = self.fetch_agent_list(max_results=limit)

        if not agent_list:
            print("❌ No agents found. The registry might be unavailable or using a different structure.")
            return []

        print(f"📊 Found {len(agent_list)} raw entries to process")

        # ------------------------------------------------------------------
        # Phase 1: Fast dedup + 404 check (serial, lightweight)
        # ------------------------------------------------------------------
        seen_urls: set = set()
        candidates = []   # (agent_data, url) tuples that survived dedup
        skipped_dupes = 0
        skipped_404 = 0

        for agent_data in agent_list:
            server = agent_data.get('server', {})

            # Primary dedup key: source_url (repo or website)
            source_url = (
                server.get('repository', {}).get('url')
                or server.get('websiteUrl')
                or ''
            )
            remote_urls = [r.get('url', '') for r in server.get('remotes', []) if r.get('url')]

            dedup_key = source_url or (remote_urls[0] if remote_urls else '')
            if dedup_key and dedup_key in seen_urls:
                skipped_dupes += 1
                continue
            if dedup_key:
                seen_urls.add(dedup_key)

            # Quick 404 check: try remotes first, fall back to source_url
            if remote_urls:
                any_alive = False
                for rurl in remote_urls:
                    try:
                        head = self.session.head(rurl, timeout=5, allow_redirects=True)
                        if head.status_code != 404:
                            any_alive = True
                    except Exception:
                        any_alive = True  # non-404 errors are fine
                if not any_alive:
                    # All remotes returned 404 — fall back to source_url
                    if source_url:
                        try:
                            head = self.session.head(source_url, timeout=5, allow_redirects=True)
                            if head.status_code == 404:
                                skipped_404 += 1
                                print(f"  ✗ 404 (all remotes + source): {dedup_key}")
                                continue
                        except Exception:
                            pass
                    else:
                        skipped_404 += 1
                        print(f"  ✗ 404 (all remotes, no source): {dedup_key}")
                        continue
            elif source_url:
                try:
                    head = self.session.head(source_url, timeout=5, allow_redirects=True)
                    if head.status_code == 404:
                        skipped_404 += 1
                        print(f"  ✗ 404: {source_url}")
                        continue
                except Exception:
                    pass

            candidates.append(agent_data)

        print(f"  ⏭️  Skipped {skipped_dupes} duplicates, {skipped_404} dead URLs (404)")
        print(f"  ✅ {len(candidates)} unique agents to process\n")

        # ------------------------------------------------------------------
        # Phase 2: Parallel processing (doc fetching + pricing)
        # ------------------------------------------------------------------
        unified_agents: List[Dict] = []
        lock = threading.Lock()
        counter = [0]   # mutable counter for progress

        def _worker(agent_data):
            result = self._process_single_agent(agent_data)
            with lock:
                counter[0] += 1
                idx = counter[0]
            if result:
                name = result.get('name', '?')
                pricing = result.get('pricing', 'unknown')
                has_readme = bool(result.get('documentation', {}).get('readme'))
                doc_src = "README" if has_readme else "detail_page" if result.get('documentation', {}).get('detail_page') else "none"
                print(f"  [{idx}/{len(candidates)}] {name}  (pricing={pricing}, docs={doc_src})")
            return result

        print(f"  Processing {len(candidates)} agents (max_workers={max_workers})...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker, a): a for a in candidates}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    # Mark as available since it passed the 404 check
                    result['is_available'] = True
                    result['availability_status'] = 'reachable'
                    unified_agents.append(result)

        print(f"\n✅ Successfully scraped {len(unified_agents)} agents")
        return unified_agents
    
    def save_to_file(self, agents: List[Dict], filename: str = "mcp_agents.json"):
        """Save scraped agents to JSON file."""
        filepath = f"./{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(agents, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved to {filepath}")
        return filepath




# ---------------------------------------------------------------------------
# Pricing extraction from various sources (GitHub, NPM, PyPI, docs, etc.)
# ---------------------------------------------------------------------------

class PricingExtractor:
    """
    Extract pricing information from a server's source URL and documentation.
    
    Pricing detection strategy (in priority order):
    1. Check {source_url}/pricing or /pricing.html for explicit pricing pages
    2. Analyze README/docs for pricing keywords (free tier + paid = freemium)
    3. Check for LICENSE file mentions in README (parse directly, no API calls)
    4. Fetch LICENSE file from GitHub if available
    5. For package registries (NPM/PyPI): check metadata
    
    Returns one of: "free", "freemium", "paid", "open_source", "unknown"
    """
    
    # Keywords that indicate different pricing models
    PRICING_KEYWORDS = {
        "free": [
            "free", "no cost", "free to use", "free forever", "100% free",
            "completely free", "forever free", "no charge", "without cost",
            "at no cost", "no payment", "zero cost"
        ],
        "paid": [
            "paid", "subscription", "premium", "enterprise pricing", "buy now",
            "purchase", "payment required", "paid plan", "commercial license",
            "license fee", "pricing starts at", "$", "€", "£", "¥",
            "per month", "per year", "/month", "/year", "billing", "/mo", "/yr",
            "pay as you go", "credit card", "checkout"
        ],
        "freemium": [
            "freemium", "free tier", "free plan", "upgrade to", "premium features",
            "paid features", "basic free", "free with limitations", "paid upgrade",
            "free and paid", "free version", "limited free", "upgrade for",
            "pro plan", "starter plan", "enterprise plan", "pricing tiers",
            "pricing plans"
        ],
        "open_source": [
            "open source", "open-source", "oss", "free and open source", "foss"
        ]
    }
    
    # Open source licenses - these indicate the project is free to use
    OPEN_SOURCE_LICENSES = {
        "mit", "apache", "apache-2.0", "apache 2.0", "bsd", "bsd-2-clause", "bsd-3-clause",
        "gpl", "gplv2", "gplv3", "lgpl", "mpl", "mozilla", "agpl",
        "unlicense", "wtfpl", "cc0", "isc", "artistic", "zlib", "boost"
    }
    
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            'User-Agent', 'MCP-Agent-Search-Engine/1.0'
        )
    
    def extract_pricing(
        self,
        source_url: str,
        readme_text: Optional[str] = None,
        detail_page_text: Optional[str] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Extract pricing from source URL and available documentation.
        
        Args:
            source_url: The server's source URL (GitHub, NPM, website, etc.)
            readme_text: README content if already fetched
            detail_page_text: Registry detail page content if available
            description: Short description from API
            
        Returns:
            One of: "free", "freemium", "paid", "open_source", "unknown"
        """
        # Strategy 1: Check for explicit pricing page (most reliable for freemium)
        if source_url:
            pricing_page_result = self._check_pricing_page(source_url)
            if pricing_page_result != "unknown":
                return pricing_page_result
        
        # Strategy 2: Analyze available text for pricing keywords
        text_pricing = self._analyze_text_for_pricing(readme_text, detail_page_text, description)
        if text_pricing != "unknown":
            return text_pricing
        
        # Strategy 3: Check for LICENSE file (from README or fetch directly)
        if source_url and readme_text:
            license_pricing = self._check_license_from_readme(readme_text, source_url)
            if license_pricing != "unknown":
                return license_pricing
        
        # Strategy 4: Fetch LICENSE file from GitHub/GitLab directly
        if source_url:
            parsed = urlparse(source_url)
            domain = parsed.netloc.lower()
            
            if 'github.com' in domain or 'gitlab.com' in domain:
                license_pricing = self._fetch_license_file(source_url)
                if license_pricing != "unknown":
                    return license_pricing
            
            # Check NPM/PyPI metadata as last resort
            elif 'npmjs.com' in domain or 'npmjs.org' in domain:
                npm_pricing = self._extract_from_npm(source_url)
                if npm_pricing != "unknown":
                    return npm_pricing
            
            elif 'pypi.org' in domain or 'pypi.python.org' in domain:
                pypi_pricing = self._extract_from_pypi(source_url)
                if pypi_pricing != "unknown":
                    return pypi_pricing
        
        return "unknown"
    
    def _check_pricing_page(self, source_url: str) -> str:
        """
        Check if there's a /pricing or /pricing.html page.
        This is the most reliable way to detect freemium models.
        
        Example: https://example.com/mcp-server/pricing shows free + paid plans
        """
        parsed = urlparse(source_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        
        # Try common pricing page patterns
        pricing_urls = [
            f"{base_url.rstrip('/')}/pricing",
            f"{base_url.rstrip('/')}/pricing.html",
            f"{base_url.rstrip('/')}/pricing/",
            f"{base_url.rstrip('/')}/plans",
            f"{base_url.rstrip('/')}/plans.html",
        ]
        
        for pricing_url in pricing_urls:
            try:
                response = self.session.get(pricing_url, timeout=5, allow_redirects=True)
                if response.status_code == 200:
                    print(f"    ✓ Found pricing page: {pricing_url}")
                    return self._analyze_pricing_page(response.text)
            except Exception:
                continue
        
        return "unknown"
    
    def _analyze_pricing_page(self, html: str) -> str:
        """
        Analyze a pricing page HTML to determine the pricing model.
        
        If we find both free and paid indicators, it's freemium.
        """
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True).lower()
        except:
            text = html.lower()
        
        # Count indicators
        has_free = any(keyword in text for keyword in ["free", "free plan", "free tier", "$0"])
        has_paid = any(keyword in text for keyword in ["paid", "pro", "premium", "enterprise", "$", "per month", "/month"])
        
        # Freemium: has both free and paid options
        if has_free and has_paid:
            return "freemium"
        
        # Just paid
        if has_paid:
            return "paid"
        
        # Just free (unlikely on a pricing page, but possible)
        if has_free:
            return "free"
        
        return "unknown"
    
    def _analyze_text_for_pricing(
        self,
        readme: Optional[str],
        detail_page: Optional[str],
        description: Optional[str]
    ) -> str:
        """
        Analyze text content for pricing keywords.
        
        Key improvement: Detect freemium by finding BOTH free and paid indicators.
        """
        # Combine all available text (prioritize README)
        text = ""
        if readme:
            text = readme
        elif detail_page:
            text = detail_page
        elif description:
            text = description
        
        if not text:
            return "unknown"
        
        text_lower = text.lower()
        
        # Count keyword matches for each pricing model
        scores = {
            "open_source": 0,
            "free": 0,
            "freemium": 0,
            "paid": 0
        }
        
        for pricing_type, keywords in self.PRICING_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[pricing_type] += 1
        
        # Check for license mentions (strong indicator of open source)
        for license_name in self.OPEN_SOURCE_LICENSES:
            if license_name in text_lower:
                scores["open_source"] += 3  # Weight licenses very high
        
        # Special logic: Freemium detection
        # If we have BOTH free indicators AND paid indicators, it's freemium
        if scores["free"] >= 1 and scores["paid"] >= 2:
            return "freemium"
        
        # Explicit freemium mentions
        if scores["freemium"] >= 1:
            return "freemium"
        
        # Open source (highest priority after freemium)
        if scores["open_source"] >= 1:
            return "open_source"
        
        # Paid only
        if scores["paid"] >= 1 and scores["free"] == 0:
            return "paid"
        
        # Free only
        if scores["free"] >= 1 and scores["paid"] == 0:
            return "free"
        
        return "unknown"
    
    def _check_license_from_readme(self, readme: str, source_url: str) -> str:
        """
        Check if README mentions a license or has a license badge.
        No API calls - just parse the text we already have.
        """
        readme_lower = readme.lower()
        
        # Look for license mentions in README
        for license_name in self.OPEN_SOURCE_LICENSES:
            # Check for explicit mentions
            if f"{license_name} license" in readme_lower:
                return "open_source"
            
            # Check for license badges (common on GitHub)
            if f"license/{license_name}" in readme_lower:
                return "open_source"
            
            # Check for SPDX identifiers
            if f"spdx-license-identifier: {license_name}" in readme_lower:
                return "open_source"
        
        return "unknown"
    
    def _fetch_license_file(self, repo_url: str) -> str:
        """
        Fetch LICENSE file directly from GitHub/GitLab.
        This is a single HTTP request to get the raw LICENSE file.
        """
        parsed = urlparse(repo_url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        
        if len(path_parts) < 2:
            return "unknown"
        
        owner, repo = path_parts[0], path_parts[1]
        
        # Try common LICENSE file names
        license_filenames = [
            "LICENSE", "LICENSE.md", "LICENSE.txt",
            "COPYING", "COPYING.md", "COPYING.txt",
            "LICENSE-MIT", "LICENSE-APACHE"
        ]
        
        for filename in license_filenames:
            if 'github.com' in parsed.netloc:
                # Try both main and master branches
                for branch in ["main", "master"]:
                    license_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}"
                    try:
                        response = self.session.get(license_url, timeout=5)
                        if response.status_code == 200:
                            return self._parse_license_text(response.text)
                    except:
                        continue
            
            elif 'gitlab.com' in parsed.netloc:
                for branch in ["main", "master"]:
                    license_url = f"https://gitlab.com/{owner}/{repo}/-/raw/{branch}/{filename}"
                    try:
                        response = self.session.get(license_url, timeout=5)
                        if response.status_code == 200:
                            return self._parse_license_text(response.text)
                    except:
                        continue
        
        return "unknown"
    
    def _parse_license_text(self, license_text: str) -> str:
        """
        Parse LICENSE file content to determine the license type.
        """
        license_lower = license_text.lower()
        
        for license_name in self.OPEN_SOURCE_LICENSES:
            if license_name in license_lower:
                return "open_source"
        
        # Check for commercial/proprietary keywords
        if "proprietary" in license_lower or "all rights reserved" in license_lower:
            return "paid"
        
        return "unknown"
    
    def _extract_from_npm(self, npm_url: str) -> str:
        """
        Extract pricing from NPM package.
        Checks package.json for license field via NPM registry API.
        """
        try:
            parsed = urlparse(npm_url)
            path_parts = [p for p in parsed.path.strip('/').split('/') if p]
            
            if 'package' in path_parts:
                pkg_idx = path_parts.index('package')
                if pkg_idx + 1 < len(path_parts):
                    package_name = '/'.join(path_parts[pkg_idx + 1:])
                    
                    api_url = f"https://registry.npmjs.org/{package_name}"
                    response = self.session.get(api_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        license_info = data.get('license', '')
                        
                        if isinstance(license_info, str):
                            license_lower = license_info.lower()
                            for oss_license in self.OPEN_SOURCE_LICENSES:
                                if oss_license in license_lower:
                                    return "open_source"
        except Exception as e:
            print(f"    ✗ NPM check failed: {e}")
        
        return "unknown"
    
    def _extract_from_pypi(self, pypi_url: str) -> str:
        """
        Extract pricing from PyPI package.
        Checks PyPI API for license classifiers.
        """
        try:
            parsed = urlparse(pypi_url)
            path_parts = [p for p in parsed.path.strip('/').split('/') if p]
            
            if 'project' in path_parts:
                pkg_idx = path_parts.index('project')
                if pkg_idx + 1 < len(path_parts):
                    package_name = path_parts[pkg_idx + 1]
                    
                    api_url = f"https://pypi.org/pypi/{package_name}/json"
                    response = self.session.get(api_url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        info = data.get('info', {})
                        
                        license_info = (info.get('license') or '').lower()
                        for oss_license in self.OPEN_SOURCE_LICENSES:
                            if oss_license in license_info:
                                return "open_source"
                        
                        classifiers = info.get('classifiers', [])
                        for classifier in classifiers:
                            classifier_lower = classifier.lower()
                            if 'license' in classifier_lower:
                                for oss_license in self.OPEN_SOURCE_LICENSES:
                                    if oss_license in classifier_lower:
                                        return "open_source"
        except Exception as e:
            print(f"    ✗ PyPI check failed: {e}")
        
        return "unknown"


# ---------------------------------------------------------------------------
# AvailabilityChecker — concurrent HTTP probes to verify server reachability
# ---------------------------------------------------------------------------

class AvailabilityChecker:
    """
    Checks whether MCP server URLs are reachable via HTTP.

    For each agent, sends a HEAD request (with GET fallback) to the source_url
    and tags the agent with ``is_available`` and ``availability_status``.

    Uses ``concurrent.futures.ThreadPoolExecutor`` for parallel checks.
    """

    def __init__(
        self,
        timeout: int = 8,
        max_workers: int = 20,
    ):
        self.timeout = timeout
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # URL classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_url(source_url: str) -> str:
        """Classify a source_url as 'http_endpoint', 'github_repo', or 'no_url'."""
        if not source_url or not source_url.startswith('http'):
            return 'no_url'
        parsed = urlparse(source_url)
        if 'github.com' in parsed.netloc:
            return 'github_repo'
        return 'http_endpoint'

    # ------------------------------------------------------------------
    # Single-agent check
    # ------------------------------------------------------------------

    def _probe_url(self, url: str) -> Optional[bool]:
        """
        Probe a single URL. Returns True (reachable), False (unreachable),
        or None (unknown/error).
        """
        try:
            resp = requests.head(url, timeout=self.timeout, allow_redirects=True)
            # Some servers reject HEAD — fall back to a streaming GET
            if resp.status_code in (405, 501):
                resp = requests.get(url, timeout=self.timeout, stream=True)
                resp.close()
            return resp.status_code < 400
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            return False
        except Exception:
            return None

    def _check_single(self, agent: Dict) -> Dict:
        """
        Probe a single agent and return availability info.

        For agents with remotes: tests each remote URL, keeps only working
        ones in ``agent['remotes']``. If none work, falls back to source_url.

        Returns dict with keys ``is_available`` (bool | None) and
        ``availability_status`` ('reachable' | 'unreachable' | 'unknown').
        """
        remotes = agent.get('remotes', [])
        source_url = agent.get('source_url', '')

        # --- Try remotes first ---
        if remotes:
            working_remotes = []
            for remote in remotes:
                rurl = remote.get('url', '')
                if not rurl:
                    continue
                result = self._probe_url(rurl)
                if result is True or result is None:
                    # Keep reachable or ambiguous remotes (only drop confirmed dead)
                    working_remotes.append(remote)

            if working_remotes:
                agent['remotes'] = working_remotes
                return {'is_available': True, 'availability_status': 'reachable'}

            # All remotes failed — clear them and fall back to source_url
            agent['remotes'] = []

        # --- Fall back to source_url ---
        if not source_url or not source_url.startswith('http'):
            return {'is_available': None, 'availability_status': 'unknown'}

        result = self._probe_url(source_url)
        if result is True:
            return {'is_available': True, 'availability_status': 'reachable'}
        elif result is False:
            return {'is_available': False, 'availability_status': 'unreachable'}
        return {'is_available': None, 'availability_status': 'unknown'}

    # ------------------------------------------------------------------
    # Batch (concurrent) check
    # ------------------------------------------------------------------

    def check_all(self, agents: List[Dict]) -> List[Dict]:
        """
        Run availability checks concurrently over *agents*.

        Mutates each agent dict in-place (adds ``is_available`` and
        ``availability_status``) and returns the same list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"  Checking availability of {len(agents)} agents "
              f"(max_workers={self.max_workers})...")

        def _task(agent):
            result = self._check_single(agent)
            agent['is_available'] = result['is_available']
            agent['availability_status'] = result['availability_status']
            return agent['name'], result['availability_status']

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_task, a): a for a in agents}
            for i, future in enumerate(as_completed(futures), 1):
                name, status = future.result()
                print(f"    [{i}/{len(agents)}] {name}: {status}")

        return agents


# ---------------------------------------------------------------------------
# MCP Protocol Prober — fetches tool definitions via initialize + tools/list
# ---------------------------------------------------------------------------

class MCPProber:
    """
    Probes MCP servers via streamable-http transport to retrieve actual tool
    definitions using the MCP protocol (initialize → tools/list).

    Populates ``agent['tools']`` with ground-truth data and sets
    ``agent['probe_status']`` to 'success', 'failed', or 'skipped'.
    """

    def __init__(self, timeout: int = 15, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers
        self.protocol_version = "2024-11-05"
        self.client_info = {"name": "agent-search-engine", "version": "0.1"}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sse_response(text: str) -> Optional[Dict]:
        """Parse an SSE-formatted response or plain JSON into a result dict."""
        # Try SSE format first (event: message\ndata: {...})
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('data: '):
                try:
                    payload = json.loads(line[6:])
                    if 'result' in payload:
                        return payload['result']
                except json.JSONDecodeError:
                    continue
        # Fallback: plain JSON
        try:
            payload = json.loads(text)
            if 'result' in payload:
                return payload['result']
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _build_headers(remote: Dict) -> Dict:
        """Build HTTP headers for an MCP request, including auth if available."""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        for h in remote.get('headers', []):
            name = h.get('name', '')
            value = h.get('value', '')
            if name and value:
                headers[name] = value
        return headers

    def _should_probe(self, agent: Dict) -> bool:
        """Return True if agent has at least one remote we can probe."""
        remotes = agent.get('remotes', [])
        if not remotes:
            return False
        for remote in remotes:
            if not remote.get('url'):
                continue
            # Skip remotes that require auth headers we don't have
            needs_auth = False
            for h in remote.get('headers', []):
                if h.get('isRequired') and not h.get('value'):
                    needs_auth = True
                    break
            if not needs_auth:
                return True
        return False

    # ------------------------------------------------------------------
    # Single probe
    # ------------------------------------------------------------------

    def _probe_single(self, remote_url: str, headers: Dict) -> Optional[List[Dict]]:
        """
        Send MCP initialize + tools/list to a single remote URL.

        Returns list of tool dicts on success, None on failure.
        """
        session = requests.Session()
        session.headers.update(headers)

        # Step 1: initialize
        init_payload = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": self.client_info,
            }
        }
        try:
            resp = session.post(remote_url, json=init_payload, timeout=self.timeout)
            if resp.status_code not in (200, 201):
                return None
            init_result = self._parse_sse_response(resp.text)
            if not init_result:
                return None

            # Carry session ID if provided
            mcp_session_id = resp.headers.get('Mcp-Session-Id')
            if mcp_session_id:
                session.headers['Mcp-Session-Id'] = mcp_session_id
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return None

        # Step 2: tools/list
        tools_payload = {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        }
        try:
            resp = session.post(remote_url, json=tools_payload, timeout=self.timeout)
            if resp.status_code not in (200, 201):
                return None
            tools_result = self._parse_sse_response(resp.text)
            if not tools_result:
                return None
            return tools_result.get('tools', [])
        except (requests.exceptions.RequestException, json.JSONDecodeError):
            return None

    # ------------------------------------------------------------------
    # Agent-level probe (tries each remote)
    # ------------------------------------------------------------------

    def _probe_agent(self, agent: Dict) -> None:
        """Probe an agent's remotes in order. Mutates agent in-place."""
        remotes = agent.get('remotes', [])

        for remote in remotes:
            url = remote.get('url', '')
            if not url:
                continue
            # Skip remotes requiring auth we don't have
            needs_auth = False
            for h in remote.get('headers', []):
                if h.get('isRequired') and not h.get('value'):
                    needs_auth = True
                    break
            if needs_auth:
                continue

            headers = self._build_headers(remote)
            tools = self._probe_single(url, headers)
            if tools is not None:
                agent['tools'] = tools
                agent['probe_status'] = 'success'
                agent['probed_tool_count'] = len(tools)
                return

        agent['probe_status'] = 'failed'
        agent['probed_tool_count'] = 0

    # ------------------------------------------------------------------
    # Batch probe
    # ------------------------------------------------------------------

    def probe_all(self, agents: List[Dict]) -> List[Dict]:
        """
        Probe all agents with accessible remotes concurrently.

        Mutates each agent dict in-place and returns the same list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        probeable = [a for a in agents if self._should_probe(a)]

        # Mark non-probeable agents
        for a in agents:
            if not self._should_probe(a):
                a['probe_status'] = 'skipped'
                a['probed_tool_count'] = 0

        print(f"  Probing {len(probeable)}/{len(agents)} agents with accessible remotes "
              f"(max_workers={self.max_workers})...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._probe_agent, a): a for a in probeable}
            for i, future in enumerate(as_completed(futures), 1):
                agent = futures[future]
                future.result()  # raises if exception
                status = agent.get('probe_status', 'unknown')
                tool_count = agent.get('probed_tool_count', 0)
                print(f"    [{i}/{len(probeable)}] {agent['name']}: {status} ({tool_count} tools)")

        return agents


# ---------------------------------------------------------------------------
# Smithery Config Checker — determines if a Smithery-hosted server needs
# external service credentials beyond the Smithery API key.
# ---------------------------------------------------------------------------

class SmitheryConfigChecker:
    """
    Queries the Smithery registry (registry.smithery.ai) to determine what
    configuration each Smithery-hosted server requires.

    Tags each Smithery agent with ``smithery_config``:
      - 'none'     — no external config, Smithery API key is sufficient
      - 'optional' — has config fields but none are strictly required
      - 'required' — needs external service credentials (e.g. API keys)
      - 'unknown'  — server not found in Smithery registry
    """

    def __init__(self, timeout: int = 10, max_workers: int = 10):
        self.timeout = timeout
        self.max_workers = max_workers

    @staticmethod
    def _extract_smithery_path(agent: Dict) -> Optional[str]:
        """Extract the @user/server path from a Smithery remote URL."""
        for remote in agent.get('remotes', []):
            url = remote.get('url', '')
            if 'server.smithery.ai/@' in url:
                return url.split('server.smithery.ai/@')[1].rstrip('/').rstrip('/mcp')
        return None

    def _check_single(self, agent: Dict) -> str:
        """Check a single agent's config requirements. Returns config category."""
        path = self._extract_smithery_path(agent)
        if not path:
            return 'unknown'

        try:
            url = f'https://registry.smithery.ai/servers/@{path}'
            req = requests.get(url, timeout=self.timeout, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/json',
            })
            if req.status_code != 200:
                return 'unknown'

            data = req.json()
            connections = data.get('connections', [])
            if not connections:
                return 'unknown'

            config_schema = connections[0].get('configSchema', {})
            properties = config_schema.get('properties', {})
            required_fields = config_schema.get('required', [])

            if not properties:
                return 'none'

            # Check if any property is strictly required (in required list, no default, not nullable)
            for prop_name, prop_def in properties.items():
                if prop_name not in required_fields:
                    continue
                if 'default' in prop_def:
                    continue
                if 'null' in str(prop_def.get('anyOf', [])):
                    continue
                # This property is required with no default and not nullable
                return 'required'

            return 'optional'

        except Exception:
            return 'unknown'

    def check_all(self, agents: List[Dict]) -> List[Dict]:
        """
        Check config requirements for all Smithery-hosted agents.

        Mutates each Smithery agent in-place (adds ``smithery_config``)
        and returns the same list.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        smithery_agents = [a for a in agents if self._extract_smithery_path(a)]

        # Mark non-Smithery agents
        for a in agents:
            if not self._extract_smithery_path(a):
                a['smithery_config'] = None

        if not smithery_agents:
            print("  No Smithery-hosted agents found.")
            return agents

        print(f"  Checking config for {len(smithery_agents)} Smithery-hosted agents "
              f"(max_workers={self.max_workers})...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._check_single, a): a for a in smithery_agents}
            for i, future in enumerate(as_completed(futures), 1):
                agent = futures[future]
                config = future.result()
                agent['smithery_config'] = config
                print(f"    [{i}/{len(smithery_agents)}] {agent['name']}: {config}")

        from collections import Counter
        counts = Counter(a.get('smithery_config') for a in smithery_agents)
        print(f"  Summary: {dict(counts)}")

        return agents


# ---------------------------------------------------------------------------
# Helper: fetch text from the MCP registry's own detail page when there is
# no GitHub / README link available.
# ---------------------------------------------------------------------------

class RegistryPageScraper:
    """
    Scrapes the MCP registry's HTML detail page for a server.

    When a server has no GitHub repository link (and therefore no README) we
    can still get more text than the one-liner `description` field by visiting
    the registry page itself.  The page typically contains a longer prose
    description, a list of tools, and sometimes usage notes.

    This is a *fallback only* — README content is always preferred.
    """

    # CSS / tag heuristics that tend to contain useful prose on registry pages.
    _PROSE_SELECTORS = [
        # Explicit semantic containers
        ('tag',   'article'),
        ('tag',   'main'),
        ('class', 'description'),
        ('class', 'readme'),
        ('class', 'content'),
        ('class', 'detail'),
        ('class', 'server-detail'),
        ('class', 'prose'),
        # Generic fall-through
        ('tag',   'section'),
    ]

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            'User-Agent', 'MCP-Agent-Search-Engine/1.0'
        )

    def scrape_detail_page(self, page_url: str, timeout: int = 12) -> str:
        """
        Fetch the registry detail page and extract useful plain text.

        Args:
            page_url: Full URL of the registry detail page for this server.
            timeout:  HTTP request timeout in seconds.

        Returns:
            Extracted plain text, or empty string if scraping fails.
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("    ⚠  beautifulsoup4 not installed — cannot scrape detail page.")
            return ""

        try:
            resp = self.session.get(page_url, timeout=timeout)
            if resp.status_code != 200:
                print(f"    ✗ Detail page returned HTTP {resp.status_code}: {page_url}")
                return ""

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Remove script / style noise
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            # Walk selectors in priority order, return first hit with real content
            for kind, value in self._PROSE_SELECTORS:
                if kind == 'tag':
                    element = soup.find(value)
                else:
                    element = soup.find(class_=lambda c: c and value in c.lower())

                if element:
                    text = element.get_text(separator='\n', strip=True)
                    if len(text) > 100:   # ignore nearly-empty containers
                        print(f"    ✓ Scraped detail page ({len(text)} chars) from {page_url}")
                        return text

            # Last resort: full body text
            body = soup.find('body')
            if body:
                text = body.get_text(separator='\n', strip=True)
                print(f"    ℹ Used full body text ({len(text)} chars) from {page_url}")
                return text

        except Exception as exc:
            print(f"    ✗ Failed to scrape detail page {page_url}: {exc}")

        return ""

    def build_registry_detail_url(
        self, base_url: str, server_name: str, server_id: Optional[str] = None
    ) -> str:
        """
        Construct the registry's HTML detail-page URL for a server.

        The MCP registry typically uses one of these patterns:
            {base}/servers/{id}
            {base}/servers/{name}
            {base}/{name}

        Args:
            base_url:    Registry base URL (e.g. "https://registry.mcp.run").
            server_name: Human-readable server name.
            server_id:   Registry-assigned ID (preferred when available).

        Returns:
            Best-guess URL for the server's detail page.
        """
        slug = (server_id or server_name).lower().replace(' ', '-')
        return f"{base_url}/servers/{slug}"


# ---------------------------------------------------------------------------
# Core LLM analysis — capabilities extraction and quality scoring
# ---------------------------------------------------------------------------

class LLMAnalyser:
    """
    Uses GPT-4 / GPT-5 (OpenAI Chat Completions API) to:
      - Extract capabilities, limitations, and requirements from server docs.
      - Return a documentation quality score (0.0 – 1.0).

    Source priority
    ---------------
    1. README text  (richest signal — always prefer this)
    2. Registry detail page text  (fallback when no GitHub link)
    3. Description field only  (last resort — short, often uninformative)

    The prompt is adjusted per source so the LLM produces calibrated scores
    (a one-liner description should never score 0.8+).

    Usage
    -----
        analyser = LLMAnalyser(api_key="sk-...", model="gpt-4o")
        result = analyser.analyse(agent)

        # result = {
        #     "capabilities": [...],
        #     "limitations":  [...],
        #     "requirements": [...],
        #     "quality_score": 0.72,
        #     "quality_rationale": "...",
        #     "text_source": "readme" | "detail_page" | "description_only"
        # }
    """

    # Maximum characters sent to the LLM (keeps cost / latency manageable).
    # GPT-4o context is ~128 k tokens; 12 000 chars ≈ 3 000 tokens — safe ceiling.
    MAX_CHARS = 12_000

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.2",          # swap to "gpt-5" when available
        registry_scraper: Optional[RegistryPageScraper] = None,
        registry_base_url: str = "https://registry.mcp.run",
        request_timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Args:
            api_key:            OpenAI API key. Falls back to OPENAI_API_KEY env var.
            model:              OpenAI model name (e.g. "gpt-4o", "gpt-5").
            registry_scraper:   RegistryPageScraper instance for fallback scraping.
            registry_base_url:  Base URL used to construct registry detail page URLs.
            request_timeout:    Seconds before the OpenAI request times out.
            max_retries:        How many times to retry on transient failures.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided. "
                "Pass api_key= or set the OPENAI_API_KEY environment variable."
            )
        self.model = model
        self.registry_scraper = registry_scraper or RegistryPageScraper()
        self.registry_base_url = registry_base_url
        self.request_timeout = request_timeout
        self.max_retries = max_retries

        self._endpoint = "https://api.openai.com/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def analyse(self, agent: Dict) -> Dict:
        """
        Run LLM analysis on a single agent and return extracted metadata.

        Args:
            agent: Unified-schema agent dict (must contain at least 'name').

        Returns:
            Dict with keys:
                capabilities    – list[str]
                limitations     – list[str]
                requirements    – list[str]
                quality_score   – float 0.0-1.0
                quality_rationale – str  (why the LLM gave that score)
                text_source     – str   ("readme" | "detail_page" | "description_only")
        """
        text, source_label = self._choose_best_text(agent)

        if not text.strip():
            print(f"    ⚠  No usable text found for '{agent.get('name')}' — skipping LLM call.")
            return self._empty_result("description_only")

        truncated = self._truncate(text)
        print(f"    🤖 Running LLM analysis (source: {source_label}, {len(truncated)} chars)...")

        raw = self._call_openai(
            system_prompt=self._build_system_prompt(source_label),
            user_content=self._build_user_prompt(agent.get('name', ''), truncated, source_label),
        )

        if raw is None:
            return self._empty_result(source_label)

        result = self._parse_response(raw)
        result["text_source"] = source_label
        return result

    def analyse_batch(self, agents: List[Dict], delay: float = 1.0) -> List[Dict]:
        """
        Run analyse() over a list of agents with a small inter-request delay.

        Args:
            agents: List of unified-schema agent dicts.
            delay:  Seconds to sleep between API calls (rate-limit courtesy).

        Returns:
            List of result dicts (same order as input).
        """
        results = []
        for i, agent in enumerate(agents, 1):
            print(f"\n  [{i}/{len(agents)}] Analysing: {agent.get('name', '?')}")
            results.append(self.analyse(agent))
            if i < len(agents):
                time.sleep(delay)
        return results

    # ------------------------------------------------------------------
    # Text sourcing
    # ------------------------------------------------------------------

    def _choose_best_text(self, agent: Dict) -> Tuple[str, str]:
        """
        Return (text, source_label) using the priority order:
          readme → detail_page (from docs dict) → description_only.
        """
        docs = agent.get('documentation') or {}
        
        # 1. README from documentation dict (already fetched by the scraper)
        readme = docs.get('readme', '')
        if readme and len(readme.strip()) > 50:
            return readme, "readme"

        # 2. Registry detail page (already scraped and stored by fetch_documentation)
        detail_page = docs.get('detail_page', '')
        if detail_page and len(detail_page.strip()) > 100:
            return detail_page, "detail_page"

        # 3. Fall back to plain description
        description = agent.get('description', '') or agent.get('short_description', '')
        return description, "description_only"

    def _fetch_detail_page_text(self, agent: Dict) -> str:
        """
        [DEPRECATED] This method is no longer used.
        
        Detail page scraping now happens in MCPRegistryScraper.fetch_documentation()
        and the result is stored in agent['documentation']['detail_page'].
        
        This method is kept for backward compatibility but always returns empty string.
        """
        return ""


    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_system_prompt(source_label: str) -> str:
        source_note = {
            "readme": (
                "You are analysing a full README file — this should give you rich "
                "information.  Score quality generously only when the documentation "
                "is genuinely comprehensive (installation, usage examples, tool "
                "descriptions, limitations)."
            ),
            "detail_page": (
                "You are analysing text scraped from the server's registry detail "
                "page.  This is a secondary source — it may be less structured than "
                "a README.  Reflect this in the quality score: a detail page that "
                "lacks examples or installation instructions should score no higher "
                "than 0.55."
            ),
            "description_only": (
                "You are analysing a short description string only — the server has "
                "no README or detail page available.  Be conservative: even a very "
                "good one-liner should cap at 0.30 because there is simply not enough "
                "information to evaluate the server fully."
            ),
        }.get(source_label, "")

        return f"""You are an expert AI/developer tools analyst.
Your task is to analyse documentation for an MCP (Model Context Protocol) server
and return a structured JSON object.

{source_note}

Return ONLY valid JSON — no markdown fences, no commentary.

JSON structure:
{{
  "capabilities": [
    // List of concise strings describing what the server CAN do.
    // Each string should start with an active verb (e.g. "Search the web",
    // "Create GitHub issues", "Execute SQL queries").
    // 3-10 items.
  ],
  "limitations": [
    // Things the server explicitly CANNOT do, known constraints, rate limits,
    // scope restrictions.  Empty list if none are mentioned.
  ],
  "requirements": [
    // Environment prerequisites: API keys, auth tokens, software versions,
    // accounts, permissions.  Empty list if none are mentioned.
  ],
  "quality_score": 0.0,
  // Float 0.0-1.0.  Scoring rubric:
  //   0.00-0.20  Almost no information (single-sentence description)
  //   0.21-0.40  Basic description only, no examples or structure
  //   0.41-0.60  Some structure (tool list OR examples, not both)
  //   0.61-0.80  Good docs: tool descriptions + usage examples OR installation
  //   0.81-1.00  Excellent: installation + examples + limitations + requirements
  "quality_rationale": "One sentence explaining the score."
}}"""

    @staticmethod
    def _build_user_prompt(name: str, text: str, source_label: str) -> str:
        source_desc = {
            "readme":           "README file",
            "detail_page":      "registry detail page",
            "description_only": "API description field",
        }.get(source_label, "documentation")

        return (
            f"Server name: {name}\n"
            f"Documentation source: {source_desc}\n\n"
            f"--- BEGIN DOCUMENTATION ---\n"
            f"{text}\n"
            f"--- END DOCUMENTATION ---\n\n"
            f"Return the JSON analysis now."
        )

    # ------------------------------------------------------------------
    # OpenAI API call
    # ------------------------------------------------------------------

    def _call_openai(self, system_prompt: str, user_content: str) -> Optional[str]:
        """
        Send a Chat Completions request and return the raw assistant message.

        Retries up to self.max_retries times on 429 / 5xx responses.
        Returns None if all retries are exhausted.
        """
        payload = {
            "model": self.model,
            "temperature": 0.1,     # low temperature → more deterministic JSON
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            "response_format": {"type": "json_object"},  # enforced JSON mode
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self._endpoint,
                    headers=self._headers,
                    json=payload,
                    timeout=self.request_timeout,
                )

                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]

                if resp.status_code == 429:
                    wait = 2 ** attempt   # exponential back-off: 2, 4, 8 s
                    print(f"    ⏳ Rate limited — waiting {wait}s before retry {attempt}/{self.max_retries}")
                    time.sleep(wait)
                    continue

                # Other HTTP error — log and retry
                print(f"    ✗ OpenAI returned HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < self.max_retries:
                    time.sleep(2)

            except requests.exceptions.Timeout:
                print(f"    ✗ Request timed out (attempt {attempt}/{self.max_retries})")
                if attempt < self.max_retries:
                    time.sleep(2)
            except Exception as exc:
                print(f"    ✗ Unexpected error calling OpenAI: {exc}")
                break

        print("    ✗ All retries exhausted — skipping LLM analysis for this agent.")
        return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Dict:
        """
        Parse the LLM's JSON response into a clean result dict.

        Handles minor formatting issues gracefully and falls back to empty
        values rather than raising exceptions.
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Strip accidental markdown fences and try again
            cleaned = re.sub(r'```(?:json)?|```', '', raw).strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"    ✗ Could not parse LLM response as JSON: {raw[:200]}")
                return self._empty_result("unknown")

        # Validate and coerce types
        capabilities = self._ensure_str_list(data.get('capabilities'))
        limitations  = self._ensure_str_list(data.get('limitations'))
        requirements = self._ensure_str_list(data.get('requirements'))

        raw_score = data.get('quality_score', 0.0)
        try:
            quality_score = float(raw_score)
            quality_score = max(0.0, min(1.0, quality_score))   # clamp to [0, 1]
        except (TypeError, ValueError):
            quality_score = 0.0

        return {
            "capabilities":       capabilities,
            "limitations":        limitations,
            "requirements":       requirements,
            "quality_score":      round(quality_score, 3),
            "quality_rationale":  str(data.get('quality_rationale', '')),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """Truncate text to MAX_CHARS, cutting at a newline boundary if possible."""
        if len(text) <= self.MAX_CHARS:
            return text
        cut = text[:self.MAX_CHARS]
        last_nl = cut.rfind('\n')
        return cut[:last_nl] if last_nl > self.MAX_CHARS // 2 else cut

    @staticmethod
    def _ensure_str_list(value) -> List[str]:
        """Coerce a value to a list of non-empty strings."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    @staticmethod
    def _empty_result(source_label: str) -> Dict:
        return {
            "capabilities":       [],
            "limitations":        [],
            "requirements":       [],
            "quality_score":      0.0,
            "quality_rationale":  "No usable documentation found.",
            "text_source":        source_label,
        }

    # ------------------------------------------------------------------
    # Agent-type classification (AI agent vs API wrapper)
    # ------------------------------------------------------------------

    def classify_agent_type(self, agent: Dict) -> Dict:
        """
        Classify whether this MCP server is an AI agent or a plain API wrapper.

        Returns dict with keys:
            is_ai_agent              – bool | None
            agent_classification     – "ai_agent" | "api_wrapper" | "unknown"
            classification_rationale – str
        """
        text, source_label = self._choose_best_text(agent)

        if not text.strip():
            return self._empty_classification()

        truncated = self._truncate(text)
        print(f"    Classifying agent type (source: {source_label}, "
              f"{len(truncated)} chars)...")

        raw = self._call_openai(
            system_prompt=self._build_classification_system_prompt(),
            user_content=self._build_classification_user_prompt(
                agent.get('name', ''), truncated, source_label,
            ),
        )

        if raw is None:
            return self._empty_classification()

        return self._parse_classification_response(raw)

    @staticmethod
    def _build_classification_system_prompt() -> str:
        return (
            "You are an expert in AI systems and MCP (Model Context Protocol) servers.\n"
            "\n"
            "Your task is to classify whether an MCP server is a genuine AI agent or a\n"
            "plain API wrapper.\n"
            "\n"
            "Definitions:\n"
            '- "ai_agent": The server uses AI/ML models, language models, or autonomous\n'
            "  reasoning internally to perform tasks.  It has its own intelligence layer,\n"
            "  not just routing.  Examples: a server that autonomously writes code,\n"
            "  summarises documents, plans tasks, or makes decisions using an embedded\n"
            "  LLM or ML model.\n"
            "\n"
            '- "api_wrapper": The server is a thin adapter that exposes an existing\n'
            "  non-AI API (database, REST service, file system, payment gateway, version\n"
            "  control, etc.) to MCP clients.  It executes deterministic operations —\n"
            "  read, write, query — without AI reasoning.  The *calling* LLM is the\n"
            "  intelligence; this server is just plumbing.\n"
            "\n"
            "Decision heuristics:\n"
            "  AGENT signals: mentions of LLM, AI model, embeddings, reasoning, planning,\n"
            "    agent, autonomous, GPT, Claude, Gemini, inference, generation,\n"
            "    summarisation, classification.\n"
            "  WRAPPER signals: mentions of CRUD operations, API keys for third-party\n"
            "    services, GitHub/Jira/Slack/database connectors, file reading/writing,\n"
            "    search indexing (without ML), sending messages/emails/notifications with\n"
            "    no AI step.\n"
            "  AMBIGUOUS: Some servers do both.  If >50% of described functionality is\n"
            "    AI-driven, lean toward \"ai_agent\"; otherwise \"api_wrapper\".\n"
            "\n"
            "Return ONLY valid JSON — no markdown fences, no commentary.\n"
            "\n"
            "JSON structure:\n"
            "{\n"
            '  "agent_classification": "ai_agent" | "api_wrapper" | "unknown",\n'
            '  "is_ai_agent": true | false | null,\n'
            '  "classification_rationale": "One or two sentences explaining the decision."\n'
            "}"
        )

    @staticmethod
    def _build_classification_user_prompt(
        name: str, text: str, source_label: str,
    ) -> str:
        source_desc = {
            "readme":           "README file",
            "detail_page":      "registry detail page",
            "description_only": "API description field",
        }.get(source_label, "documentation")

        return (
            f"Server name: {name}\n"
            f"Documentation source: {source_desc}\n\n"
            f"--- BEGIN DOCUMENTATION ---\n"
            f"{text}\n"
            f"--- END DOCUMENTATION ---\n\n"
            f"Classify this MCP server and return the JSON now."
        )

    def _parse_classification_response(self, raw: str) -> Dict:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(r'```(?:json)?|```', '', raw).strip()
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                print(f"    ✗ Could not parse classification response: {raw[:200]}")
                return self._empty_classification()

        classification = str(data.get('agent_classification', 'unknown')).lower()
        if classification not in ('ai_agent', 'api_wrapper', 'unknown'):
            classification = 'unknown'

        raw_is_agent = data.get('is_ai_agent')
        if isinstance(raw_is_agent, bool):
            is_ai_agent = raw_is_agent
        elif classification == 'ai_agent':
            is_ai_agent = True
        elif classification == 'api_wrapper':
            is_ai_agent = False
        else:
            is_ai_agent = None

        return {
            'is_ai_agent':              is_ai_agent,
            'agent_classification':     classification,
            'classification_rationale': str(data.get('classification_rationale', '')),
        }

    @staticmethod
    def _empty_classification() -> Dict:
        return {
            'is_ai_agent':              None,
            'agent_classification':     'unknown',
            'classification_rationale': '',
        }


# ---------------------------------------------------------------------------
# DocumentationProcessor — now with real LLM-backed methods
# ---------------------------------------------------------------------------

class DocumentationProcessor:
    """
    Processes documentation for semantic chunking and embedding.
    Implements the pipeline from section 1.7.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        llm_analyser=None,
    ):
        """
        Args:
            chunk_size:    Target token count per documentation chunk (~4 chars/token).
            overlap:       Token overlap between consecutive chunks.
            llm_analyser:  LLMAnalyser instance.  If None, LLM analysis is skipped
                           and empty placeholder values are returned instead.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.llm_analyser = llm_analyser
    
    def chunk_documentation(self, text: str) -> List[Dict]:
        """
        Chunk documentation semantically with overlap.
        """
        # Token approximation: ~4 chars per token
        char_chunk_size = self.chunk_size * 4
        char_overlap = self.overlap * 4
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate the ideal end position
            ideal_end = start + char_chunk_size
            end = ideal_end
            
            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('.\n', start, end),
                    text.rfind('!\n', start, end),
                    text.rfind('?\n', start, end)
                )
                if sentence_end > start:
                    end = sentence_end + 1
            else:
                # We're at the end of the text
                end = len(text)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'token_count_estimate': len(chunk_text) // 4
                })
                chunk_id += 1
            
            min_progress = char_chunk_size - char_overlap
            next_start = start + min_progress
            
            # But if we found a good sentence boundary, use that with overlap
            if end >= start + min_progress:
                next_start = end - char_overlap
            
            # Safety check: ensure we always move forward
            if next_start <= start:
                next_start = start + 1
            
            start = next_start
        
        return chunks
    

    def calculate_quality_score(self, doc: Dict) -> float:
        """
        Kept for backward compatibility.
        Returns 0.0 when no LLM analyser is configured (score is computed
        inside process_agent_documentation instead).
        """
        return 0.0

    def process_agent_documentation(self, agent: Dict, skip_llm: bool = False) -> Dict:
        """
        Full documentation processing pipeline.

        Steps:
          1. Chunk any available documentation text (README or detail_page) for embedding.
          2. Call LLMAnalyser.analyse() to extract capabilities, limitations,
             requirements, and a quality score in a single API call.
          3. Store all results back onto the agent dict in-place.

        If no llm_analyser is set or skip_llm is True, LLM fields are left
        as empty placeholders and the quality score remains 0.0.
        """
        docs = agent.get("documentation", {})
        all_chunks = []

        # --- Step 1: chunk available documentation ---
        # Chunk README if present
        if docs.get("readme"):
            readme_chunks = self.chunk_documentation(docs["readme"])
            for chunk in readme_chunks:
                chunk["source_type"] = "readme"
                chunk["agent_id"] = agent["agent_id"]
            all_chunks.extend(readme_chunks)
        
        # Chunk detail_page if present (and no README)
        elif docs.get("detail_page"):
            detail_chunks = self.chunk_documentation(docs["detail_page"])
            for chunk in detail_chunks:
                chunk["source_type"] = "detail_page"
                chunk["agent_id"] = agent["agent_id"]
            all_chunks.extend(detail_chunks)

        # --- Step 2: LLM analysis ---
        if self.llm_analyser is not None and not skip_llm:
            # Full LLM pipeline: capability extraction + classification
            llm_result = self.llm_analyser.analyse(agent)
            agent["llm_extracted"] = {
                "capabilities": llm_result.get("capabilities", []),
                "limitations":  llm_result.get("limitations",  []),
                "requirements": llm_result.get("requirements", []),
            }
            agent["documentation_quality"]   = llm_result.get("quality_score",     0.0)
            agent["quality_rationale"]       = llm_result.get("quality_rationale",  "")
            agent["llm_text_source"]         = llm_result.get("text_source",        "unknown")

            cls_result = self.llm_analyser.classify_agent_type(agent)
            agent["is_ai_agent"]              = cls_result["is_ai_agent"]
            agent["agent_classification"]     = cls_result["agent_classification"]
            agent["classification_rationale"] = cls_result["classification_rationale"]
        elif self.llm_analyser is not None and skip_llm:
            # Probed agent: skip capability extraction, still run classification
            agent["llm_extracted"] = {
                "capabilities": [],
                "limitations":  [],
                "requirements": [],
            }
            agent["documentation_quality"] = 0.0
            agent["quality_rationale"]     = "Skipped — tools obtained via MCP protocol."
            agent["llm_text_source"]       = "none"

            print(f"    🏷️  Running LLM classification...")
            cls_result = self.llm_analyser.classify_agent_type(agent)
            agent["is_ai_agent"]              = cls_result["is_ai_agent"]
            agent["agent_classification"]     = cls_result["agent_classification"]
            agent["classification_rationale"] = cls_result["classification_rationale"]
        else:
            # No LLM configured at all
            agent["llm_extracted"] = {
                "capabilities": [],
                "limitations":  [],
                "requirements": [],
            }
            agent["documentation_quality"] = 0.0
            agent["quality_rationale"]     = "LLM analysis not configured."
            agent["llm_text_source"]       = "none"
            agent["is_ai_agent"]              = None
            agent["agent_classification"]     = "unknown"
            agent["classification_rationale"] = ""

        # --- Step 3: store chunks ---
        agent["documentation_chunks"] = all_chunks

        print(f"  📝 Created {len(all_chunks)} documentation chunks")
        print(f"  ⭐ Quality score : {agent['documentation_quality']:.2f}")
        print(f"  📖 LLM text source: {agent['llm_text_source']}")
        print(f"  🏷️  Classification : {agent['agent_classification']}")
        if agent["llm_extracted"]["capabilities"]:
            print(f"  🎯 Capabilities  : {len(agent['llm_extracted']['capabilities'])} extracted")

        return agent


def main(probeable: bool = False, smithery: bool = False):
    """Main execution function."""
    print("="*70)
    print("MCP REGISTRY WEB SCRAPER")
    print("="*70)

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------
    LIMIT = 500      # Set to None to fetch ALL agents from the registry.

    # Set to True to enable LLM capability extraction and quality scoring.
    # Requires OPENAI_API_KEY to be set in the environment.
    ENABLE_LLM = True

    # Model to use. 
    LLM_MODEL  = "gpt-4.1-mini"

    # Seconds to sleep between LLM API calls (respect rate limits).
    LLM_DELAY  = 0
    # -----------------------------------------------------------------------

    # Step 1: Scrape agents
    print("\n" + "="*70)
    print("STEP 1: SCRAPING AGENTS FROM MCP REGISTRY")
    print("="*70)

    scraper = MCPRegistryScraper()
    agents = scraper.scrape_all_agents(limit=LIMIT)

    if not agents:
        print("\n❌ No agents were scraped. Exiting.")
        return

    # Step 1.5: Probe MCP servers for tool definitions
    print("\n" + "="*70)
    print("STEP 1.5: PROBING MCP SERVERS FOR TOOL DEFINITIONS")
    print("="*70)

    prober = MCPProber(timeout=15, max_workers=10)
    agents = prober.probe_all(agents)

    probed_ok = sum(1 for a in agents if a.get('probe_status') == 'success')
    print(f"\n  ✅ Successfully probed: {probed_ok}/{len(agents)} agents")

    # Step 1.6: Check Smithery config requirements (only when --smithery flag is used)
    if smithery:
        print("\n" + "="*70)
        print("STEP 1.6: CHECKING SMITHERY CONFIG REQUIREMENTS")
        print("="*70)

        config_checker = SmitheryConfigChecker(timeout=10, max_workers=10)
        agents = config_checker.check_all(agents)

    # Step 2: Set up optional LLM analyser
    llm_analyser = None
    if ENABLE_LLM:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("\n⚠  OPENAI_API_KEY not set — LLM analysis will be skipped.")
        else:
            print(f"\n🤖 LLM analysis enabled (model: {LLM_MODEL})")
            llm_analyser = LLMAnalyser(
                api_key=api_key,
                model=LLM_MODEL,
                registry_base_url=scraper.base_url,
            )

    # Step 3: Process documentation
    print("\n" + "="*70)
    print("STEP 2: PROCESSING DOCUMENTATION & LLM ANALYSIS")
    print("="*70)

    processor = DocumentationProcessor(
        chunk_size=512,
        overlap=50,
        llm_analyser=llm_analyser,
    )

    processed_agents = []
    for i, agent in enumerate(agents, 1):
        print(f"\n[{i}/{len(agents)}] 📦 Processing: {agent['name']}")

        if agent.get('probe_status') == 'success':
            # Tools already populated via MCP protocol — skip LLM capability extraction
            print(f"    ⚡ Skipping LLM (probed: {agent.get('probed_tool_count', 0)} tools)")
            processed_agent = processor.process_agent_documentation(agent, skip_llm=True)
        else:
            processed_agent = processor.process_agent_documentation(agent)

        processed_agents.append(processed_agent)

        # Inter-request delay to avoid hammering the LLM API
        if llm_analyser and agent.get('probe_status') != 'success' and i < len(agents):
            time.sleep(LLM_DELAY)

    # Optional filter: only probeable AI agents
    if probeable or smithery:
        print("\n" + "="*70)
        if smithery:
            print("FILTERING: PROBEABLE + SMITHERY AI AGENTS")
        else:
            print("FILTERING: PROBEABLE AI AGENTS ONLY")
        print("="*70)
        before = len(processed_agents)

        def _is_probeable(a):
            return (
                a.get('agent_classification') == 'ai_agent'
                and a.get('remotes')
                and a.get('pricing') in ('free', 'open_source')
                and not any(
                    h.get('isRequired')
                    for r in a.get('remotes', [])
                    for h in r.get('headers', [])
                )
            )

        def _is_smithery_accessible(a):
            """Smithery-hosted agent that only needs a Smithery API key."""
            return (
                a.get('agent_classification') == 'ai_agent'
                and a.get('remotes')
                and a.get('smithery_config') == 'none'
            )

        if smithery:
            processed_agents = [
                a for a in processed_agents
                if _is_probeable(a) or _is_smithery_accessible(a)
            ]
        else:
            processed_agents = [a for a in processed_agents if _is_probeable(a)]

        print(f"  Filtered {before} -> {len(processed_agents)} agents")

    # Step 3: Save results
    print("\n" + "="*70)
    print("STEP 3: SAVING RESULTS")
    print("="*70)

    output_file = scraper.save_to_file(processed_agents, "mcp_agents.json")

    # Summary
    print("\n" + "="*70)
    print("SCRAPING SUMMARY")
    print("="*70)
    total = len(processed_agents)
    avg_quality = sum(a.get("documentation_quality", 0) for a in processed_agents) / total
    with_caps   = sum(1 for a in processed_agents if a.get("llm_extracted", {}).get("capabilities"))
    reachable   = sum(1 for a in processed_agents if a.get('availability_status') == 'reachable')
    unreachable = sum(1 for a in processed_agents if a.get('availability_status') == 'unreachable')
    ai_agents   = sum(1 for a in processed_agents if a.get('agent_classification') == 'ai_agent')
    api_wrappers = sum(1 for a in processed_agents if a.get('agent_classification') == 'api_wrapper')
    print(f"✅ Total agents processed : {total}")
    print(f"✅ Total doc chunks       : {sum(len(a.get('documentation_chunks', [])) for a in processed_agents)}")
    print(f"✅ Avg quality score      : {avg_quality:.2f}")
    print(f"✅ Agents with LLM caps   : {with_caps}/{total}")
    print(f"✅ Reachable agents       : {reachable}/{total}")
    print(f"✅ Unreachable agents     : {unreachable}/{total}")
    print(f"✅ AI agents classified   : {ai_agents}/{total}")
    print(f"✅ API wrappers classified: {api_wrappers}/{total}")
    probed     = sum(1 for a in processed_agents if a.get('probe_status') == 'success')
    probe_fail = sum(1 for a in processed_agents if a.get('probe_status') == 'failed')
    total_tools = sum(a.get('probed_tool_count', 0) for a in processed_agents)
    print(f"✅ Probed successfully    : {probed}/{total}")
    print(f"✅ Probe failed           : {probe_fail}/{total}")
    print(f"✅ Total tools discovered : {total_tools}")
    print(f"\n📁 Output file: {output_file}")

    # Save filtered subset: reachable AI agents only
    reachable_ai = [a for a in processed_agents
                    if a.get('is_available') and a.get('agent_classification') == 'ai_agent']
    if reachable_ai:
        ai_file = scraper.save_to_file(reachable_ai, "mcp_ai_agents.json")
        print(f"📁 Reachable AI agents: {len(reachable_ai)} → {ai_file}")

    # Show sample
    print("\n" + "="*70)
    print("SAMPLE AGENT DATA")
    print("="*70)
    if processed_agents:
        sample = processed_agents[0]
        print(f"\n📦 {sample['name']}")
        print(f"   ID              : {sample['agent_id']}")
        print(f"   Source URL      : {sample['source_url']}")
        print(f"   Description     : {sample['description']}")
        print(f"   Pricing         : {sample['pricing']}")
        print(f"   Tools           : {len(sample['tools'])}")
        print(f"   Doc chunks      : {len(sample.get('documentation_chunks', []))}")
        print(f"   Quality score   : {sample.get('documentation_quality', 0):.2f}")
        print(f"   LLM text source : {sample.get('llm_text_source', 'n/a')}")
        print(f"   Available       : {sample.get('availability_status', 'unknown')}")
        print(f"   Classification  : {sample.get('agent_classification', 'unknown')}")
        llm = sample.get("llm_extracted", {})
        if llm.get("capabilities"):
            print(f"   Capabilities ({len(llm['capabilities'])}):")
            for cap in llm["capabilities"][:5]:
                print(f"     • {cap}")
        if llm.get("limitations"):
            print(f"   Limitations ({len(llm['limitations'])}):")
            for lim in llm["limitations"][:3]:
                print(f"     ⚠  {lim}")
        if llm.get("requirements"):
            print(f"   Requirements ({len(llm['requirements'])}):")
            for req in llm["requirements"][:3]:
                print(f"     📋 {req}")
        if sample.get("quality_rationale"):
            print(f"   Rationale       : {sample['quality_rationale']}")


def parse_args():
    parser = argparse.ArgumentParser(description="MCP Registry Web Scraper")
    parser.add_argument(
        "--probeable",
        action="store_true",
        help="Only output AI agents that have accessible remote endpoints and are free (open_source pricing, no required auth headers).",
    )
    parser.add_argument(
        "--smithery",
        action="store_true",
        help="Like --probeable, but also includes Smithery-hosted agents that only need a Smithery API key (no external service credentials).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(probeable=args.probeable, smithery=args.smithery)