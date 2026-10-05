"""Tests for the policy-enforced secure fetcher.

The end-to-end tests talk to a real HTTP server running on 127.0.0.1, but the
"public" host ``public.test`` is resolved to 8.8.8.8 and the injected
connector routes that *validated* address back to the loopback test server.
A connection to any address that the policy should have rejected raises
AssertionError, so a guard that silently opens sockets to internal addresses
fails loudly. Nothing here touches the real network.
"""

from __future__ import annotations

import socket
import threading
import time
import gzip
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from click.testing import CliRunner

from rich_cli import fetch as fetch_mod
from rich_cli.fetch import (
    FetchContentError,
    FetchError,
    FetchHTTPError,
    FetchLimitError,
    FetchPolicy,
    FetchPolicyError,
    FetchRedirectError,
    FetchTimeoutError,
    SecureFetcher,
    sanitize_url,
)

BOMB = gzip.compress(b"0" * (50 * 1024 * 1024), 9)
GZIP_TEXT = gzip.compress(
    b"".join(f"decompressed line number {i}\n".encode() for i in range(4000)),
    6,
)

# Captured before tests monkeypatch the module-level network hooks: lets one
# test exercise the production resolver/connector against a real loopback
# socket with an explicitly relaxed policy.
REAL_RESOLVER = fetch_mod.default_resolver
REAL_CONNECT = fetch_mod.default_connect


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence test server output
        pass

    def _send(
        self,
        status,
        body=b"",
        content_type="text/plain; charset=utf-8",
        extra_headers=None,
    ):
        self.server.hits[self.path] += 1
        self.send_response(status)
        if content_type:
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command == "GET" and body:
            self.wfile.write(body)

    def do_GET(self):
        port = self.server.server_address[1]
        path = self.path

        if path == "/text":
            self._send(200, b"Hello, world!\n")
        elif path == "/json":
            self._send(200, b'{"hello": "world"}\n', "application/json")
        elif path == "/no-ct":
            self._send(200, b"just text, no type\n", content_type=None)
        elif path == "/gzip-text":
            self._send(
                200,
                GZIP_TEXT,
                extra_headers={"Content-Encoding": "gzip"},
            )
        elif path == "/binary":
            self._send(
                200, b"\x00\x01\x02\x03binary payload", "application/octet-stream"
            )
        elif path == "/png":
            self._send(200, b"\x89PNG\r\n\x1a\n-binary", "image/png")
        elif path == "/huge-length":
            self.server.hits[self.path] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "999999999")
            self.end_headers()
        elif path == "/bomb":
            self.server.hits[self.path] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(BOMB)))
            self.end_headers()
            try:
                self.wfile.write(BOMB)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        elif path == "/infinite":
            self.server.hits[self.path] += 1
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                while True:
                    chunk = b"x" * 1024
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
        elif path == "/slow":
            self.server.hits[self.path] += 1
            time.sleep(2)
            self._send(200, b"too late\n")
        elif path == "/secret":
            self._send(200, b"internal secret\n")
        elif path == "/missing":
            self._send(404, b"not found\n")
        elif path == "/redir":
            self._redirect(f"http://127.0.0.1:{port}/secret")
        elif path == "/redir-169":
            self._redirect("http://169.254.169.254/latest/meta-data/")
        elif path == "/redir-10":
            self._redirect("http://10.0.0.5/admin")
        elif path == "/redir-bad-scheme":
            self._redirect("file:///etc/passwd")
        elif path == "/redir-public":
            self._redirect("http://public.test/text")
        elif path == "/redir-chain":
            self._redirect("/redir")
        elif path == "/loop":
            self._redirect("/loop")
        else:
            self._send(404, b"unknown path\n")

    def _redirect(self, location):
        self.server.hits[self.path] += 1
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


class NetworkEnv:
    def __init__(self, server, resolver, connect_calls):
        self.server = server
        self._resolver = resolver
        self.connect_calls = connect_calls

    def url(self, path):
        return f"http://public.test{path}"

    def literal_url(self, host, path):
        return f"http://{host}:{self.server.server_address[1]}{path}"


@pytest.fixture(autouse=True)
def network(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.hits = defaultdict(int)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()
    port = server.server_address[1]

    real_resolver = fetch_mod.default_resolver
    connect_calls: list = []

    def fake_resolver(host, queried_port):
        if host == "public.test":
            return [(socket.AF_INET, "8.8.8.8")]
        return real_resolver(host, queried_port)

    def fake_connect(family, ip, connect_port, timeout):
        connect_calls.append(ip)
        if ip != "8.8.8.8":
            raise AssertionError(
                f"refusing to connect to unvalidated address {ip}:{connect_port}"
            )
        return socket.create_connection(("127.0.0.1", port), timeout=min(timeout, 3.0))

    def fake_ssl_wrap(sock, host, timeout):
        raise AssertionError("TLS should not be used in the HTTP test harness")

    monkeypatch.setattr(fetch_mod, "default_resolver", fake_resolver)
    monkeypatch.setattr(fetch_mod, "default_connect", fake_connect)
    monkeypatch.setattr(fetch_mod, "default_ssl_wrap", fake_ssl_wrap)

    env = NetworkEnv(server, fake_resolver, connect_calls)
    try:
        yield env
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Policy unit tests
# ---------------------------------------------------------------------------


BLOCKED_ADDRESSES = [
    ("127.0.0.1", "loopback"),
    ("127.1.2.3", "loopback"),
    ("10.0.0.1", "private"),
    ("100.64.0.9", "private"),
    ("172.16.5.5", "private"),
    ("172.31.255.255", "private"),
    ("192.168.1.1", "private"),
    ("169.254.169.254", "link-local"),
    ("0.0.0.0", "unspecified"),
    ("224.0.0.1", "multicast"),
    ("255.255.255.255", "reserved"),
    ("192.0.2.7", "reserved"),
    ("198.18.0.1", "reserved"),
    ("::1", "loopback"),
    ("fe80::1234", "link-local"),
    ("fc00::1", "private"),
    ("fd12::1", "private"),
    ("ff02::1", "multicast"),
    ("::", "unspecified"),
    ("::ffff:127.0.0.1", "loopback"),
]

ALLOWED_ADDRESSES = ["8.8.8.8", "1.1.1.1", "93.184.113.10", "2606:4700:4700::1111"]


@pytest.mark.parametrize("address,category", BLOCKED_ADDRESSES)
def test_policy_blocks_unsafe_addresses(address, category):
    reason = FetchPolicy().check_ip(__import__("ipaddress").ip_address(address))
    assert reason is not None
    assert category in reason


@pytest.mark.parametrize("address", ALLOWED_ADDRESSES)
def test_policy_allows_public_addresses(address):
    reason = FetchPolicy().check_ip(__import__("ipaddress").ip_address(address))
    assert reason is None


def test_policy_flags_can_relax_categories():
    policy = FetchPolicy(block_loopback=False, block_private=False)
    import ipaddress

    assert policy.check_ip(ipaddress.ip_address("127.0.0.1")) is None
    assert policy.check_ip(ipaddress.ip_address("10.0.0.1")) is None
    # Other categories are still blocked.
    assert "link-local" in policy.check_ip(ipaddress.ip_address("169.254.1.1"))


@pytest.mark.parametrize(
    "url,word",
    [
        ("file:///etc/passwd", "scheme"),
        ("ftp://public.test/payload", "scheme"),
        ("gopher://public.test/", "scheme"),
        ("http:///nohost", "host"),
        ("http://public.test:70000/", "port"),
    ],
)
def test_policy_validates_url_shape(url, word):
    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch(url)
    assert word in str(info.value)


def test_policy_rejects_url_credentials_and_restricted_ports():
    with pytest.raises(FetchPolicyError, match="userinfo"):
        SecureFetcher().fetch("http://user:hunter2@public.test/text")

    policy = FetchPolicy(allowed_ports=frozenset({80, 443}))
    with pytest.raises(FetchPolicyError, match="port"):
        SecureFetcher(policy).fetch("http://public.test:8080/text")

    # Explicit opt-in keeps the URL working (validation proceeds to DNS).
    policy = FetchPolicy(allow_userinfo=True)
    result = SecureFetcher(policy).fetch("http://user:hunter2@public.test/text")
    assert result.text == "Hello, world!\n"


# ---------------------------------------------------------------------------
# Normal (allowed) traffic
# ---------------------------------------------------------------------------


def test_plain_text_download_succeeds(network):
    result = SecureFetcher().fetch(network.url("/text"))
    assert result.text == "Hello, world!\n"
    assert result.content_type == "text/plain"
    assert result.bytes_body == len(b"Hello, world!\n")
    assert result.trace[-1].outcome == "ok"
    assert result.trace[-1].addresses == ["8.8.8.8"]
    assert network.connect_calls == ["8.8.8.8"]


def test_gzip_response_is_decompressed_and_measured(network):
    result = SecureFetcher().fetch(network.url("/gzip-text"))
    assert result.text.startswith("decompressed line")
    hop = result.trace[-1]
    assert hop.content_encoding == "gzip"
    assert hop.bytes_wire < hop.bytes_body


def test_missing_content_type_is_sniffed_as_text(network):
    result = SecureFetcher().fetch(network.url("/no-ct"))
    assert "just text" in result.text


def test_real_resolver_and_connector_roundtrip(network):
    # Production network hooks (no fakes) against a real loopback socket;
    # loopback blocking is explicitly relaxed for this server test.
    port = network.server.server_address[1]
    policy = FetchPolicy(block_loopback=False)
    fetcher = SecureFetcher(policy, resolver=REAL_RESOLVER, connect=REAL_CONNECT)
    result = fetcher.fetch(f"http://127.0.0.1:{port}/gzip-text")
    assert result.text.startswith("decompressed line number 0")
    assert result.trace[-1].addresses  # really resolved addresses recorded


def test_read_resource_keeps_format_detection(network):
    from rich_cli.__main__ import read_resource

    text, lexer = read_resource(network.url("/json"), None)
    assert '"hello"' in text
    assert lexer == "JSON"


def test_http_errors_are_reported(network):
    with pytest.raises(FetchHTTPError, match="404"):
        SecureFetcher().fetch(network.url("/missing"))


# ---------------------------------------------------------------------------
# SSRF: direct, redirect and DNS rebinding
# ---------------------------------------------------------------------------


def test_direct_loopback_ip_is_blocked(network):
    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch(network.literal_url("127.0.0.1", "/secret"))
    rendered = str(info.value)
    assert "loopback" in rendered
    assert "hop 1" in rendered
    assert network.connect_calls == []
    assert network.server.hits["/secret"] == 0


def test_localhost_name_is_blocked(network):
    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch(network.literal_url("localhost", "/secret"))
    assert "loopback" in str(info.value)
    assert network.connect_calls == []


@pytest.mark.parametrize(
    "host,category",
    [("169.254.169.254", "link-local"), ("10.0.0.9", "private")],
)
def test_direct_forbidden_categories(network, host, category):
    with pytest.raises(FetchPolicyError, match=category):
        SecureFetcher().fetch(network.literal_url(host, "/"))


def test_redirect_to_loopback_is_blocked_with_hop_reasons(network):
    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch(network.url("/redir"))
    rendered = str(info.value)
    assert "hop 1" in rendered
    assert "hop 2" in rendered
    assert "127.0.0.1" in rendered
    assert "loopback" in rendered
    assert "redirect ->" in rendered
    assert network.server.hits["/secret"] == 0
    # Only the validated first-hop address was ever contacted.
    assert network.connect_calls == ["8.8.8.8"]


def test_redirect_to_link_local_and_bad_scheme(network):
    with pytest.raises(FetchPolicyError, match="link-local"):
        SecureFetcher().fetch(network.url("/redir-169"))
    with pytest.raises(FetchPolicyError, match="private"):
        SecureFetcher().fetch(network.url("/redir-10"))
    with pytest.raises(FetchPolicyError, match="scheme"):
        SecureFetcher().fetch(network.url("/redir-bad-scheme"))


def test_redirect_chain_keeps_verifying_every_hop(network):
    # /redir-chain -> /redir -> http://127.0.0.1/secret
    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch(network.url("/redir-chain"))
    rendered = str(info.value)
    assert all(f"hop {n}" in rendered for n in (1, 2, 3))
    assert "loopback" in rendered
    assert network.server.hits["/secret"] == 0


def test_allowed_public_redirect_completes(network):
    result = SecureFetcher().fetch(network.url("/redir-public"))
    assert result.text == "Hello, world!\n"
    assert [hop.outcome for hop in result.trace] == ["redirect", "ok"]


def test_mixed_dns_answers_skip_blocked_and_pin_public_one(network):
    def mixed_resolver(host, port):
        # Internal answer first; a naive connector that re-resolves would
        # happily open the loopback/private socket.
        return [
            (socket.AF_INET, "10.0.0.9"),
            (socket.AF_INET, "8.8.8.8"),
        ]

    result = SecureFetcher(resolver=mixed_resolver).fetch(network.url("/text"))
    assert result.text == "Hello, world!\n"
    assert network.connect_calls == ["8.8.8.8"]
    assert result.trace[0].skipped  # rejection recorded in the trace


def test_dns_rebinding_cannot_change_validated_address(network):
    answers = iter(
        [
            [(socket.AF_INET, "8.8.8.8")],
            [(socket.AF_INET, "127.0.0.1")],
            [(socket.AF_INET, "127.0.0.1")],
        ]
    )
    calls = []

    def rebinding_resolver(host, port):
        calls.append(host)
        return next(answers)

    result = SecureFetcher(resolver=rebinding_resolver).fetch(network.url("/text"))
    assert result.text == "Hello, world!\n"
    # The host was resolved exactly once; the pinned, validated answer is
    # what the connector received. A second answer of 127.0.0.1 is unused.
    assert calls == ["public.test"]
    assert network.connect_calls == ["8.8.8.8"]


def test_dns_rebinding_on_redirect_is_caught(network):
    # First lookup returns a public address, later lookups an internal one.
    answers = iter(
        [
            [(socket.AF_INET, "8.8.8.8")],
            [(socket.AF_INET, "127.0.0.1")],
        ]
    )

    def rebinding_resolver(host, port):
        return next(answers)

    with pytest.raises(FetchPolicyError, match="loopback"):
        SecureFetcher(resolver=rebinding_resolver).fetch(network.url("/redir-public"))
    # Hop one connected to the validated public address; hop two never did.
    assert network.connect_calls == ["8.8.8.8"]


def test_host_resolving_only_to_internal_addresses_is_blocked(network):
    def internal_resolver(host, port):
        return [(socket.AF_INET, "10.2.3.4")]

    with pytest.raises(FetchPolicyError, match="private") as info:
        SecureFetcher(resolver=internal_resolver).fetch("http://mixed-answers.test/x")
    assert "blocked addresses" in str(info.value)
    assert network.connect_calls == []


# ---------------------------------------------------------------------------
# Time, byte and compression budgets
# ---------------------------------------------------------------------------


def test_slow_response_hits_read_timeout(network):
    policy = FetchPolicy(connect_timeout=2.0, read_timeout=0.4, total_timeout=5.0)
    start = time.monotonic()
    with pytest.raises(FetchTimeoutError, match="read timeout"):
        SecureFetcher(policy).fetch(network.url("/slow"))
    assert time.monotonic() - start < 1.5


def test_total_timeout_is_a_hard_budget(network):
    policy = FetchPolicy(connect_timeout=5.0, read_timeout=5.0, total_timeout=0.02)
    start = time.monotonic()
    with pytest.raises(FetchTimeoutError, match="total timeout"):
        SecureFetcher(policy).fetch(network.url("/slow"))
    assert time.monotonic() - start < 1.5


def test_infinite_stream_is_cancelled_at_byte_budget(network):
    policy = FetchPolicy(max_bytes=32 * 1024, read_timeout=2.0, total_timeout=5.0)
    start = time.monotonic()
    with pytest.raises(FetchLimitError, match="maximum of 32768 bytes"):
        SecureFetcher(policy).fetch(network.url("/infinite"))
    assert time.monotonic() - start < 2.0
    assert network.server.hits["/infinite"] == 1


def test_declared_oversize_content_length_is_rejected(network):
    policy = FetchPolicy(max_bytes=1024)
    with pytest.raises(FetchLimitError, match="Content-Length"):
        SecureFetcher(policy).fetch(network.url("/huge-length"))


def test_compression_bomb_hits_byte_budget(network):
    policy = FetchPolicy(
        max_bytes=64 * 1024,
        max_compression_ratio=10_000.0,
        read_timeout=2.0,
        total_timeout=5.0,
    )
    start = time.monotonic()
    with pytest.raises(FetchLimitError, match="maximum of 65536 bytes"):
        SecureFetcher(policy).fetch(network.url("/bomb"))
    assert time.monotonic() - start < 2.0


def test_compression_ratio_guard(network):
    policy = FetchPolicy(
        max_bytes=100 * 1024 * 1024,
        max_compression_ratio=20.0,
        read_timeout=2.0,
        total_timeout=5.0,
    )
    with pytest.raises(FetchLimitError, match="compression bomb"):
        SecureFetcher(policy).fetch(network.url("/bomb"))


# ---------------------------------------------------------------------------
# Content policy
# ---------------------------------------------------------------------------


def test_disallowed_content_type_rejected(network):
    with pytest.raises(FetchContentError, match="image/png"):
        SecureFetcher().fetch(network.url("/png"))


def test_binary_magic_bytes_rejected(network):
    with pytest.raises(FetchContentError, match="binary"):
        SecureFetcher().fetch(network.url("/binary"))


def test_redirect_loop_and_limit(network):
    policy = FetchPolicy(max_redirects=3)
    with pytest.raises(FetchRedirectError):
        SecureFetcher(policy).fetch(network.url("/loop"))


# ---------------------------------------------------------------------------
# Trace hygiene and CLI integration
# ---------------------------------------------------------------------------


def test_trace_is_sanitized_on_success_and_failure(network):
    policy = FetchPolicy(allow_userinfo=True)
    result = SecureFetcher(policy).fetch("http://user:hunter2@public.test/text")
    rendered = repr(result.trace)
    assert "hunter2" not in rendered
    assert "user@" not in rendered
    assert "public.test" in rendered

    with pytest.raises(FetchPolicyError) as info:
        SecureFetcher().fetch("http://user:hunter2@public.test/text")
    assert "hunter2" not in str(info.value)


def test_sanitize_url_strips_userinfo_and_fragment():
    clean = sanitize_url("https://user:p%40ss@example.com/a?b=1#frag")
    assert clean == "https://example.com/a?b=1"


def test_cli_error_shows_hop_chain(network):
    from rich_cli.__main__ import main

    runner = CliRunner()
    result = runner.invoke(main, [network.literal_url("127.0.0.1", "/secret")])
    assert result.exit_code != 0
    assert "unable to fetch" in result.output
    assert "loopback" in result.output
    assert "hop 1" in result.output
    assert network.server.hits["/secret"] == 0
