"""
MCP Authentication Manager

Handles authentication for MCP server probing:
  - API key injection (env var → header) for known services
  - OAuth 2.0 Client Credentials flow (no browser)
  - OAuth 2.0 Authorization Code + PKCE flow (one-time browser)
  - Vendor OAuth (pre-registered apps, browser)
  - Token caching and automatic refresh
"""

import os
import json
import time
import secrets
import hashlib
import base64
import threading
import webbrowser
from fnmatch import fnmatch
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlencode, urlparse, parse_qs
from typing import Dict, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# API Key Registry — maps URL patterns to env vars and header formats
# ---------------------------------------------------------------------------

API_KEY_REGISTRY = [
    # Apify servers (9+)
    {
        "pattern": "*.apify.actor*",
        "env_var": "APIFY_API_TOKEN",
        "header": "Authorization",
        "format": "Bearer {key}",
    },
    # Klavis AI
    {
        "pattern": "*klavis*",
        "env_var": "KLAVIS_API_KEY",
        "header": "Authorization",
        "format": "Bearer {key}",
    },
    # CarsXE
    {
        "pattern": "*carsxe*",
        "env_var": "CARSXE_API_KEY",
        "header": "x-api-key",
        "format": "{key}",
    },
# Packmind
    {
        "pattern": "*packmind*",
        "env_var": "PACKMIND_API_KEY",
        "header": "Authorization",
        "format": "Bearer {key}",
    },
    # CoTrader
    {
        "pattern": "*cotrader*",
        "env_var": "COTRADER_API_KEY",
        "header": "Authorization",
        "format": "Bearer {key}",
    },
    # Guru
    {
        "pattern": "*guru*",
        "env_var": "GURU_API_KEY",
        "header": "Authorization",
        "format": "Bearer {key}",
    },
]

# ---------------------------------------------------------------------------
# OAuth server groups — domains that support specific OAuth flows
# ---------------------------------------------------------------------------

# Group 3: Client Credentials (no browser needed)
CLIENT_CREDENTIALS_DOMAINS = []

# Group 4: Authorization Code + PKCE (one-time browser)
AUTHCODE_PKCE_DOMAINS = [
    "mcp.stripe.com",
    "mcp.sanity.io",
    "mcp.devcycle.com",
    "mcp.emtailabs.com",
    "mcp.microsoft.com",
]

# Group 5: Vendor OAuth (needs client_id/secret from env)
VENDOR_OAUTH_DOMAINS = {
    "mcp.atlassian.com": {
        "client_id_env": "ATLASSIAN_CLIENT_ID",
        "client_secret_env": "ATLASSIAN_CLIENT_SECRET",
    },
    "mcp.monday.com": {
        "client_id_env": "MONDAY_CLIENT_ID",
        "client_secret_env": "MONDAY_CLIENT_SECRET",
    },
    "mcp.webflow.com": {
        "client_id_env": "WEBFLOW_CLIENT_ID",
        "client_secret_env": "WEBFLOW_CLIENT_SECRET",
    },
    "mcp.trayd.co": {
        "client_id_env": "TRAYD_CLIENT_ID",
        "client_secret_env": "TRAYD_CLIENT_SECRET",
    },
}

TOKEN_CACHE_FILE = ".mcp_tokens.json"
TOKEN_REFRESH_BUFFER = 300  # Refresh 5 minutes before expiry
VENDOR_OAUTH_PORT = 8080  # Fixed port for vendor OAuth redirect URIs
VENDOR_OAUTH_REDIRECT = f"http://127.0.0.1:{VENDOR_OAUTH_PORT}/callback"


class MCPAuthManager:
    """
    Manages authentication for MCP server probing.

    Supports:
      - Static API key injection from environment variables
      - OAuth 2.0 Client Credentials (headless)
      - OAuth 2.0 Authorization Code + PKCE (browser-based, one-time)
      - Vendor OAuth (pre-registered app credentials from env)
      - Automatic token refresh
    """

    def __init__(self, cache_path: str = TOKEN_CACHE_FILE):
        self.cache_path = cache_path
        self._cache = self._load_cache()
        self._refresh_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Token cache I/O
    # ------------------------------------------------------------------

    def _load_cache(self) -> Dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r') as f:
                    data = json.load(f)
                if data.get("_version") == 1:
                    return data
            except (json.JSONDecodeError, IOError):
                pass
        return {"_version": 1, "oauth_tokens": {}, "vendor_oauth": {}}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except IOError as e:
            print(f"    Warning: could not save token cache: {e}")

    # ------------------------------------------------------------------
    # API Key matching (Group 1)
    # ------------------------------------------------------------------

    def _match_api_key(self, url: str) -> Optional[Dict]:
        """Match a URL against the API key registry."""
        parsed = urlparse(url)
        host = parsed.hostname or ""
        full = host + parsed.path
        for entry in API_KEY_REGISTRY:
            if fnmatch(full, entry["pattern"]) or fnmatch(host, entry["pattern"]):
                key = os.environ.get(entry["env_var"], "")
                if key:
                    return {
                        entry["header"]: entry["format"].format(key=key),
                    }
        return None

    # ------------------------------------------------------------------
    # OAuth metadata discovery
    # ------------------------------------------------------------------

    def _discover_oauth_metadata(self, domain: str) -> Optional[Dict]:
        """Fetch OAuth metadata from .well-known/oauth-authorization-server."""
        url = f"https://{domain}/.well-known/oauth-authorization-server"
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return None

    # ------------------------------------------------------------------
    # Dynamic client registration
    # ------------------------------------------------------------------

    def _dynamic_register(self, reg_endpoint: str, grant_types: list,
                          redirect_uri: Optional[str] = None) -> Optional[Dict]:
        """Register a dynamic OAuth client."""
        payload = {
            "client_name": "agent-search-engine",
            "grant_types": grant_types,
            "token_endpoint_auth_method": "client_secret_post",
        }
        if redirect_uri:
            payload["redirect_uris"] = [redirect_uri]
        try:
            resp = requests.post(reg_endpoint, json=payload, timeout=15)
            if resp.status_code in (200, 201):
                return resp.json()
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return None

    # ------------------------------------------------------------------
    # PKCE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_pkce() -> Tuple[str, str]:
        """Generate PKCE code_verifier and S256 code_challenge."""
        verifier = secrets.token_urlsafe(64)[:128]
        digest = hashlib.sha256(verifier.encode('ascii')).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
        return verifier, challenge

    # ------------------------------------------------------------------
    # OAuth callback server
    # ------------------------------------------------------------------

    class _OAuthCallbackHandler(BaseHTTPRequestHandler):
        """HTTP handler that captures the OAuth authorization code."""

        auth_code = None
        state_received = None

        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            self.__class__.auth_code = qs.get("code", [None])[0]
            self.__class__.state_received = qs.get("state", [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful!</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )

        def log_message(self, format, *args):
            pass  # Suppress noisy HTTP logs

    def _run_oauth_callback_server(self, timeout: int = 120) -> Tuple[Optional[str], int]:
        """Run ephemeral HTTP server to capture OAuth redirect. Returns (code, port)."""
        # Reset class state
        self._OAuthCallbackHandler.auth_code = None
        self._OAuthCallbackHandler.state_received = None

        server = HTTPServer(("127.0.0.1", 0), self._OAuthCallbackHandler)
        port = server.server_address[1]
        server.timeout = timeout

        server.handle_request()  # Blocks until one request or timeout

        code = self._OAuthCallbackHandler.auth_code
        server.server_close()
        return code, port

    # ------------------------------------------------------------------
    # Client Credentials flow (Group 3)
    # ------------------------------------------------------------------

    def _request_token_client_credentials(self, token_ep: str,
                                           client_id: str,
                                           client_secret: str) -> Optional[Dict]:
        """Exchange client credentials for an access token."""
        try:
            resp = requests.post(token_ep, data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            }, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return None

    def setup_client_credentials(self, domain: str) -> bool:
        """Run Client Credentials OAuth flow for a domain. Returns True on success."""
        print(f"    Discovering OAuth metadata for {domain}...")
        meta = self._discover_oauth_metadata(domain)
        if not meta:
            print(f"    No OAuth metadata found for {domain}")
            return False

        reg_ep = meta.get("registration_endpoint")
        token_ep = meta.get("token_endpoint")
        if not reg_ep or not token_ep:
            print(f"    Missing registration or token endpoint for {domain}")
            return False

        print(f"    Registering dynamic client...")
        client = self._dynamic_register(reg_ep, ["client_credentials"])
        if not client:
            print(f"    Dynamic registration failed for {domain}")
            return False

        client_id = client.get("client_id")
        client_secret = client.get("client_secret")
        if not client_id or not client_secret:
            print(f"    Registration response missing client credentials")
            return False

        print(f"    Requesting access token...")
        token_resp = self._request_token_client_credentials(
            token_ep, client_id, client_secret
        )
        if not token_resp or "access_token" not in token_resp:
            print(f"    Token request failed for {domain}")
            return False

        expires_in = token_resp.get("expires_in", 3600)
        self._cache["oauth_tokens"][domain] = {
            "client_id": client_id,
            "client_secret": client_secret,
            "access_token": token_resp["access_token"],
            "expires_at": int(time.time()) + expires_in,
            "refresh_token": token_resp.get("refresh_token"),
            "token_endpoint": token_ep,
            "grant_type": "client_credentials",
        }
        self._save_cache()
        print(f"    Token acquired for {domain} (expires in {expires_in}s)")
        return True

    # ------------------------------------------------------------------
    # Authorization Code + PKCE flow (Group 4)
    # ------------------------------------------------------------------

    def setup_authcode_pkce(self, domain: str,
                            client_id: Optional[str] = None,
                            client_secret: Optional[str] = None,
                            fixed_port: Optional[int] = None) -> bool:
        """
        Run Authorization Code + PKCE flow for a domain.
        Opens browser for one-time user consent. Returns True on success.
        If client_id/client_secret are provided, skips dynamic registration.
        If fixed_port is set, binds to that port (for vendor OAuth with
        pre-registered redirect URIs).
        """
        meta = self._discover_oauth_metadata(domain)
        if not meta:
            print(f"    No OAuth metadata found for {domain}")
            return False

        auth_ep = meta.get("authorization_endpoint")
        token_ep = meta.get("token_endpoint")
        reg_ep = meta.get("registration_endpoint")
        if not auth_ep or not token_ep:
            print(f"    Missing authorization or token endpoint for {domain}")
            return False

        verifier, challenge = self._generate_pkce()
        state = secrets.token_urlsafe(32)

        # Bind callback server — fixed port for vendor OAuth, ephemeral otherwise
        from http.server import HTTPServer
        self._OAuthCallbackHandler.auth_code = None
        self._OAuthCallbackHandler.state_received = None
        bind_port = fixed_port or 0
        server = HTTPServer(("127.0.0.1", bind_port), self._OAuthCallbackHandler)
        port = server.server_address[1]
        server.timeout = 120
        redirect_uri = f"http://127.0.0.1:{port}/callback"

        # Dynamic registration if no client creds provided
        if not client_id:
            if not reg_ep:
                print(f"    No registration endpoint and no client credentials for {domain}")
                server.server_close()
                return False
            print(f"    Registering dynamic client for {domain}...")
            client = self._dynamic_register(
                reg_ep,
                ["authorization_code"],
                redirect_uri=redirect_uri,
            )
            if not client:
                print(f"    Dynamic registration failed for {domain}")
                server.server_close()
                return False
            client_id = client.get("client_id")
            client_secret = client.get("client_secret", "")

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{auth_ep}?{urlencode(params)}"

        print(f"    Opening browser for authorization...")
        print(f"    URL: {auth_url}")
        webbrowser.open(auth_url)
        print(f"    Waiting for authorization (timeout 120s)...")

        # Wait for callback
        server.handle_request()
        code = self._OAuthCallbackHandler.auth_code
        server.server_close()

        if not code:
            print(f"    Authorization timed out or failed for {domain}")
            return False

        # Verify state
        if self._OAuthCallbackHandler.state_received != state:
            print(f"    State mismatch — possible CSRF attack, aborting")
            return False

        # Exchange code for tokens
        print(f"    Exchanging authorization code for tokens...")
        try:
            token_data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            }
            if client_secret:
                token_data["client_secret"] = client_secret
            resp = requests.post(token_ep, data=token_data, timeout=15)
            if resp.status_code != 200:
                print(f"    Token exchange failed: HTTP {resp.status_code}")
                return False
            token_resp = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"    Token exchange error: {e}")
            return False

        if "access_token" not in token_resp:
            print(f"    Token response missing access_token")
            return False

        expires_in = token_resp.get("expires_in", 3600)
        cache_section = "oauth_tokens"
        self._cache[cache_section][domain] = {
            "client_id": client_id,
            "client_secret": client_secret or "",
            "access_token": token_resp["access_token"],
            "expires_at": int(time.time()) + expires_in,
            "refresh_token": token_resp.get("refresh_token"),
            "token_endpoint": token_ep,
            "grant_type": "authorization_code",
        }
        self._save_cache()
        print(f"    Token acquired for {domain} (expires in {expires_in}s)")
        return True

    # ------------------------------------------------------------------
    # Vendor OAuth (Group 5)
    # ------------------------------------------------------------------

    def setup_vendor_oauth(self, domain: str) -> bool:
        """Run vendor OAuth flow using pre-registered app credentials from env."""
        vendor = VENDOR_OAUTH_DOMAINS.get(domain)
        if not vendor:
            print(f"    {domain} is not a known vendor OAuth domain")
            return False

        client_id = os.environ.get(vendor["client_id_env"], "")
        client_secret = os.environ.get(vendor["client_secret_env"], "")
        if not client_id:
            print(f"    {vendor['client_id_env']} not set — skipping {domain}")
            return False

        return self.setup_authcode_pkce(
            domain, client_id=client_id, client_secret=client_secret,
            fixed_port=VENDOR_OAUTH_PORT,
        )

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    def _get_refresh_lock(self, domain: str) -> threading.Lock:
        with self._global_lock:
            if domain not in self._refresh_locks:
                self._refresh_locks[domain] = threading.Lock()
            return self._refresh_locks[domain]

    def _do_refresh(self, token_ep: str, refresh_token: str,
                    client_id: str, client_secret: str = "") -> Optional[Dict]:
        """POST grant_type=refresh_token."""
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            }
            if client_secret:
                data["client_secret"] = client_secret
            resp = requests.post(token_ep, data=data, timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except (requests.RequestException, json.JSONDecodeError):
            pass
        return None

    def _refresh_if_needed(self, domain: str) -> bool:
        """Check if token is expired and refresh it. Returns True if token is valid."""
        # Check both token sections
        for section in ("oauth_tokens", "vendor_oauth"):
            entry = self._cache.get(section, {}).get(domain)
            if not entry:
                continue

            expires_at = entry.get("expires_at", 0)
            if time.time() < expires_at - TOKEN_REFRESH_BUFFER:
                return True  # Still valid

            lock = self._get_refresh_lock(domain)
            with lock:
                # Re-check after acquiring lock (another thread may have refreshed)
                entry = self._cache.get(section, {}).get(domain)
                if not entry:
                    return False
                if time.time() < entry.get("expires_at", 0) - TOKEN_REFRESH_BUFFER:
                    return True

                token_ep = entry.get("token_endpoint", "")
                refresh_token = entry.get("refresh_token")
                client_id = entry.get("client_id", "")
                client_secret = entry.get("client_secret", "")
                grant_type = entry.get("grant_type", "")

                if refresh_token:
                    token_resp = self._do_refresh(
                        token_ep, refresh_token, client_id, client_secret
                    )
                elif grant_type == "client_credentials" and client_secret:
                    token_resp = self._request_token_client_credentials(
                        token_ep, client_id, client_secret
                    )
                else:
                    return False  # Cannot refresh

                if not token_resp or "access_token" not in token_resp:
                    return False

                entry["access_token"] = token_resp["access_token"]
                entry["expires_at"] = int(time.time()) + token_resp.get("expires_in", 3600)
                if token_resp.get("refresh_token"):
                    entry["refresh_token"] = token_resp["refresh_token"]
                self._save_cache()
                return True

        return False

    # ------------------------------------------------------------------
    # Public interface — called by MCPProber
    # ------------------------------------------------------------------

    def get_auth_headers(self, url: str) -> Dict[str, str]:
        """
        Return auth headers for a URL, or empty dict if no auth available.
        Checks API keys first, then cached OAuth tokens.
        """
        # 1. Static API key match
        api_key_headers = self._match_api_key(url)
        if api_key_headers:
            return api_key_headers

        # 2. OAuth token match (by domain)
        parsed = urlparse(url)
        domain = parsed.hostname or ""

        for section in ("oauth_tokens", "vendor_oauth"):
            entry = self._cache.get(section, {}).get(domain)
            if entry and entry.get("access_token"):
                # Try refresh if needed
                self._refresh_if_needed(domain)
                # Re-read after potential refresh
                entry = self._cache.get(section, {}).get(domain)
                if entry and entry.get("access_token"):
                    return {"Authorization": f"Bearer {entry['access_token']}"}

        return {}

    def has_auth_for(self, url: str) -> bool:
        """Return True if we have (or can provide) auth for this URL."""
        return bool(self.get_auth_headers(url))

    # ------------------------------------------------------------------
    # Interactive setup (--setup-auth)
    # ------------------------------------------------------------------

    def run_interactive_setup(self) -> None:
        """Run interactive auth setup for all supported services."""
        print("=" * 70)
        print("MCP AUTH SETUP")
        print("=" * 70)

        # --- Group 1: API Key status ---
        print("\n--- API Key Status ---")
        for entry in API_KEY_REGISTRY:
            val = os.environ.get(entry["env_var"], "")
            status = "SET" if val else "NOT SET"
            marker = "+" if val else "-"
            print(f"  [{marker}] {entry['env_var']:30s} ({entry['pattern']})")

        # --- Smithery ---
        smithery_key = os.environ.get("SMITHERY_API_KEY", "")
        status = "SET" if smithery_key else "NOT SET"
        marker = "+" if smithery_key else "-"
        print(f"  [{marker}] {'SMITHERY_API_KEY':30s} (server.smithery.ai)")

        # --- Group 3: Client Credentials (automatic) ---
        print("\n--- Client Credentials OAuth (automatic) ---")
        for domain in CLIENT_CREDENTIALS_DOMAINS:
            existing = self._cache.get("oauth_tokens", {}).get(domain)
            if existing and existing.get("access_token"):
                expires = existing.get("expires_at", 0)
                if time.time() < expires - TOKEN_REFRESH_BUFFER:
                    print(f"  [+] {domain} — cached token valid")
                    continue
            ok = self.setup_client_credentials(domain)
            print(f"  [{'+'if ok else '-'}] {domain} — {'success' if ok else 'failed'}")

        # --- Group 4: Authorization Code + PKCE (browser) ---
        print("\n--- Authorization Code + PKCE OAuth (browser required) ---")
        for domain in AUTHCODE_PKCE_DOMAINS:
            existing = self._cache.get("oauth_tokens", {}).get(domain)
            if existing and existing.get("access_token"):
                expires = existing.get("expires_at", 0)
                if time.time() < expires - TOKEN_REFRESH_BUFFER:
                    print(f"  [+] {domain} — cached token valid")
                    continue

            resp = input(f"  Set up {domain}? [y/N] ").strip().lower()
            if resp != 'y':
                print(f"  [-] {domain} — skipped")
                continue
            ok = self.setup_authcode_pkce(domain)
            print(f"  [{'+'if ok else '-'}] {domain} — {'success' if ok else 'failed'}")

        # --- Group 5: Vendor OAuth ---
        print("\n--- Vendor OAuth (requires app registration) ---")
        for domain, vendor in VENDOR_OAUTH_DOMAINS.items():
            existing = self._cache.get("vendor_oauth", {}).get(domain)
            if existing and existing.get("access_token"):
                expires = existing.get("expires_at", 0)
                if time.time() < expires - TOKEN_REFRESH_BUFFER:
                    print(f"  [+] {domain} — cached token valid")
                    continue

            client_id = os.environ.get(vendor["client_id_env"], "")
            if not client_id:
                print(f"  [-] {domain} — {vendor['client_id_env']} not set")
                continue

            resp = input(f"  Set up {domain}? [y/N] ").strip().lower()
            if resp != 'y':
                print(f"  [-] {domain} — skipped")
                continue
            ok = self.setup_vendor_oauth(domain)
            print(f"  [{'+'if ok else '-'}] {domain} — {'success' if ok else 'failed'}")

        print("\n" + "=" * 70)
        cached_count = (
            len(self._cache.get("oauth_tokens", {}))
            + len(self._cache.get("vendor_oauth", {}))
        )
        api_key_count = sum(
            1 for e in API_KEY_REGISTRY if os.environ.get(e["env_var"], "")
        )
        print(f"Auth setup complete: {api_key_count} API keys, {cached_count} OAuth tokens cached")
        print(f"Token cache: {self.cache_path}")
        print("=" * 70)
