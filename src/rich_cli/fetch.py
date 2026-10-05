"""Secure HTTP fetching with an explicit :class:`FetchPolicy`.

When ``rich`` renders a URL the fetch has to be safe against a number of
abuses that a plain ``requests.get`` does not protect against:

* Server side request forgery: hosts (including hosts reached through a
  redirect) may resolve to loopback, private, link-local or otherwise
  non-public addresses.
* DNS rebinding: a name may resolve to a public address while it is being
  validated and to an internal address when the connection is opened.
* Resource exhaustion: peers may stall forever, stream an endless body or
  send a small highly compressed "zip bomb".

Every hop of a fetch is therefore validated (scheme, host, resolved IP and
port), the connection is opened to the exact, already validated IP address
(pinning the result of a single DNS lookup) and all reads are bounded by
hard wall-clock and byte budgets.  A redacted, hop-by-hop trace is kept and
attached to every error so failures can be diagnosed without leaking
credentials.
"""

from __future__ import annotations

import codecs
import http.client
import ipaddress
import socket
import ssl
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Base class for fetch failures.

    The (sanitized) hop trace leading up to the failure is attached so that
    callers can show *why* each hop was accepted, redirected or rejected.
    """

    def __init__(self, message: str, *, trace: Optional[List["FetchHop"]] = None):
        super().__init__(message)
        self.message = message
        self.trace: List[FetchHop] = list(trace) if trace else []

    def __str__(self) -> str:
        lines = [self.message]
        lines.extend(_render_hop(hop) for hop in self.trace)
        return "\n".join(lines)


class FetchPolicyError(FetchError):
    """The URL or one of its resolved addresses violates the policy."""


class FetchTimeoutError(FetchError):
    """The connect, read or total wall-clock budget was exhausted."""


class FetchLimitError(FetchError):
    """The body or its decompressed form exceeded a hard byte budget."""


class FetchContentError(FetchError):
    """The declared Content-Type or the body magic bytes are not allowed."""


class FetchRedirectError(FetchError):
    """Redirects were missing, malformed, looping or too numerous."""


class FetchHTTPError(FetchError):
    """The server answered with a non-success HTTP status."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


# Address categories are defined explicitly (rather than relying solely on
# ``ipaddress.is_private`` whose rules changed between Python versions).
_BLOCKED_NETWORK_STRINGS: List[Tuple[str, List[str]]] = [
    (
        "loopback",
        ["127.0.0.0/8", "::1/128"],
    ),
    (
        "unspecified",
        ["0.0.0.0/32", "::/128"],
    ),
    (
        "link-local",
        ["169.254.0.0/16", "fe80::/10"],
    ),
    (
        "private",
        [
            "10.0.0.0/8",
            "100.64.0.0/10",  # carrier-grade NAT
            "172.16.0.0/12",
            "192.168.0.0/16",
            "fc00::/7",  # unique local
        ],
    ),
    (
        "multicast",
        ["224.0.0.0/4", "ff00::/8"],
    ),
    (
        "reserved",
        [
            "0.0.0.0/8",
            "192.0.0.0/24",
            "192.0.2.0/24",  # TEST-NET-1
            "192.88.99.0/24",
            "198.18.0.0/15",  # benchmarking
            "198.51.100.0/24",  # TEST-NET-2
            "203.0.113.0/24",  # TEST-NET-3
            "240.0.0.0/4",  # reserved + limited broadcast
            "2001:db8::/32",  # documentation
        ],
    ),
]

_BLOCKED_NETWORKS: List[Tuple[str, List[ipaddress._BaseNetwork]]] = [
    (category, [ipaddress.ip_network(value) for value in networks])
    for category, networks in _BLOCKED_NETWORK_STRINGS
]

# Content-Types considered text-like even though they live under
# "application/" or elsewhere.
_TEXTUAL_CONTENT_TYPES = frozenset(
    {
        "application/json",
        "application/manifest+json",
        "application/ld+json",
        "application/xml",
        "application/xhtml+xml",
        "application/atom+xml",
        "application/rss+xml",
        "application/rdf+xml",
        "application/soap+xml",
        "application/svg+xml",
        "image/svg+xml",
        "application/mathml+xml",
        "application/javascript",
        "application/x-javascript",
        "application/ecmascript",
        "application/x-ecmascript",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
    }
)

# Generic "binary blob" types carry no information, so they are allowed past
# the declared-type check and decided by the magic-byte sniff.
_GENERIC_CONTENT_TYPES = frozenset(
    {
        "application/octet-stream",
        "binary/octet-stream",
        "application/binary",
        "application/x-download",
        "application/force-download",
    }
)

_SUPPORTED_ENCODINGS = frozenset({"identity", "gzip", "deflate"})


@dataclass(frozen=True)
class FetchPolicy:
    """Security and resource policy applied to every fetch and every hop."""

    allowed_schemes: frozenset = field(
        default_factory=lambda: frozenset({"http", "https"})
    )
    # ``None`` means any port in the valid TCP range is accepted.
    allowed_ports: Optional[frozenset] = None
    allow_userinfo: bool = False

    block_loopback: bool = True
    block_unspecified: bool = True
    block_link_local: bool = True
    block_private: bool = True
    block_multicast: bool = True
    block_reserved: bool = True

    max_redirects: int = 10

    connect_timeout: float = 10.0
    read_timeout: float = 10.0
    total_timeout: float = 30.0

    # Maximum number of *decompressed* body bytes.
    max_bytes: int = 10 * 1024 * 1024
    # Maximum accepted decompressed / compressed byte ratio.
    max_compression_ratio: float = 100.0
    # Bytes inspected for the binary magic-byte sniff.
    sniff_bytes: int = 8192
    block_binary: bool = True

    def check_ip(self, address: ipaddress._BaseAddress) -> Optional[str]:
        """Return ``None`` if *address* is allowed, otherwise a reason."""

        # IPv4-mapped IPv6 addresses (e.g. ``::ffff:127.0.0.1``) are checked
        # against the IPv4 tables.
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped

        flag_for_category = {
            "loopback": self.block_loopback,
            "unspecified": self.block_unspecified,
            "link-local": self.block_link_local,
            "private": self.block_private,
            "multicast": self.block_multicast,
            "reserved": self.block_reserved,
        }
        for category, networks in _BLOCKED_NETWORKS:
            if not flag_for_category[category]:
                continue
            for blocked_network in networks:
                if (
                    blocked_network.version == address.version
                    and address in blocked_network
                ):
                    return f"{address} is a {category} address"
        return None


DEFAULT_FETCH_POLICY = FetchPolicy()


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------


@dataclass
class FetchHop:
    """A sanitized record of one request in a (possibly redirected) fetch."""

    hop: int
    method: str
    url: str
    host: Optional[str] = None
    port: Optional[int] = None
    # IP addresses the policy allowed and that were tried.
    addresses: List[str] = field(default_factory=list)
    # "ip -> reason" entries rejected by the policy (never connected to).
    skipped: List[str] = field(default_factory=list)
    status: Optional[int] = None
    reason: Optional[str] = None
    content_type: Optional[str] = None
    content_encoding: Optional[str] = None
    bytes_wire: int = 0
    bytes_body: int = 0
    elapsed: float = 0.0
    outcome: str = "pending"  # ok | redirect | blocked | error

    def finish(self, outcome: str, reason: Optional[str] = None) -> None:
        self.outcome = outcome
        if reason is not None:
            self.reason = reason


@dataclass
class FetchedResource:
    """The result of a successful policy-checked fetch."""

    url: str
    text: str
    content_type: str
    charset: Optional[str]
    bytes_body: int
    trace: List[FetchHop]


def sanitize_url(url: str) -> str:
    """Return *url* without userinfo (credentials) and fragment."""

    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    host = parsed.hostname or ""
    try:
        port = f":{parsed.port}" if parsed.port else ""
    except ValueError:
        port = ""
    netloc = f"{host}{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))


def _render_hop(hop: FetchHop) -> str:
    bits = [f"  hop {hop.hop}: {hop.method} {hop.url}"]
    if hop.addresses:
        bits.append(f" [{', '.join(hop.addresses)}]")
    if hop.status is not None:
        status_text = f" -> {hop.status}"
        if hop.reason and hop.outcome != "redirect":
            status_text += f" {hop.reason}"
        bits.append(status_text)
    if hop.outcome == "redirect" and hop.reason:
        bits.append(f", redirect -> {hop.reason}")
    if hop.skipped:
        bits.append(f", skipped: {'; '.join(hop.skipped)}")
    if hop.outcome in ("blocked", "error") and hop.reason:
        bits.append(f" -> {hop.outcome}: {hop.reason}")
    if hop.outcome == "ok":
        size = f", {hop.bytes_body} bytes"
        if hop.content_type:
            size = f", {hop.content_type}{size}"
        bits.append(size)
    return "".join(bits)


# ---------------------------------------------------------------------------
# Pluggable network layer (module level so tests can stay offline)
# ---------------------------------------------------------------------------

Resolver = Callable[[str, int], List[Tuple[int, str]]]
Connect = Callable[[int, str, int, float], socket.socket]
SslWrap = Callable[[socket.socket, str, float], socket.socket]


def default_resolver(host: str, port: int) -> List[Tuple[int, str]]:
    """Resolve *host* to ``(address family, ip)`` pairs, no validation."""

    results: List[Tuple[int, str]] = []
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return results
    for family, _kind, _proto, _canon, sockaddr in infos:
        results.append((family, sockaddr[0].split("%", 1)[0]))
    return results


def default_connect(family: int, ip: str, port: int, timeout: float) -> socket.socket:
    """Open a TCP connection to an already resolved *ip*."""

    sock = socket.socket(family, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    address: Tuple = (ip, port) if family == socket.AF_INET else (ip, port, 0, 0)
    try:
        sock.connect(address)
    except OSError:
        sock.close()
        raise
    return sock


def default_ssl_wrap(sock: socket.socket, host: str, timeout: float) -> socket.socket:
    context = ssl.create_default_context()
    sock.settimeout(timeout)
    return context.wrap_socket(sock, server_hostname=host)


# ---------------------------------------------------------------------------
# Streaming decompression
# ---------------------------------------------------------------------------


class _DeflateDecoder:
    """zlib or "raw" deflate depending on the first bytes received."""

    def __init__(self) -> None:
        self._decompressobj: Optional[Any] = None
        self._buffered = b""

    def decompress(self, data: bytes) -> bytes:
        if self._decompressobj is None:
            self._buffered += data
            if len(self._buffered) < 2:
                return b""
            first, second = self._buffered[0], self._buffered[1]
            wbits = (
                zlib.MAX_WBITS if (first * 256 + second) % 31 == 0 else -zlib.MAX_WBITS
            )
            self._decompressobj = zlib.decompressobj(wbits)
            data = self._buffered
            self._buffered = b""
        return self._decompressobj.decompress(data)

    def flush(self) -> bytes:
        if self._decompressobj is None:
            self._decompressobj = zlib.decompressobj(zlib.MAX_WBITS)
            data = self._buffered
            self._buffered = b""
            return self._decompressobj.decompress(data) + self._decompressobj.flush()
        return self._decompressobj.flush()


def _new_decoder(encoding: str):
    encoding = encoding.strip().lower()
    if encoding in ("", "identity"):
        return None
    if encoding == "gzip":
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if encoding == "deflate":
        return _DeflateDecoder()
    raise FetchPolicyError(f"unsupported Content-Encoding {encoding!r}")


# ---------------------------------------------------------------------------
# Body decoding and sniffing
# ---------------------------------------------------------------------------


def _decode_body(raw: bytes, charset: Optional[str]) -> str:
    if charset:
        try:
            return codecs.lookup(charset).decode(raw)[0]
        except LookupError:
            pass
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(raw).best()
        if best is not None:
            return str(best)
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data:
        return True
    if not data:
        return False
    bad = sum(1 for byte in data if byte < 9 or byte in (11, 12) or 14 <= byte < 32)
    return bad / len(data) > 0.30


def _split_content_type(header: Optional[str]) -> Tuple[str, Optional[str]]:
    if not header:
        return "", None
    parts = [part.strip() for part in header.split(";")]
    essence = parts[0].lower()
    charset = None
    for part in parts[1:]:
        if part.lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip().strip('"') or None
    return essence, charset


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------


class SecureFetcher:
    """Fetch URLs while enforcing a :class:`FetchPolicy`."""

    READ_CHUNK = 64 * 1024

    def __init__(
        self,
        policy: Optional[FetchPolicy] = None,
        *,
        resolver: Optional[Resolver] = None,
        connect: Optional[Connect] = None,
        ssl_wrap: Optional[SslWrap] = None,
        user_agent: str = "rich-cli",
    ) -> None:
        self.policy = policy or DEFAULT_FETCH_POLICY
        self._resolver = resolver or default_resolver
        self._connect = connect or default_connect
        self._ssl_wrap = ssl_wrap or default_ssl_wrap
        self._user_agent = user_agent

    # -- URL / address validation -----------------------------------------

    def _validate_url(self, url: str) -> Tuple[str, str, int]:
        try:
            parsed = urlsplit(url)
        except ValueError as error:
            raise FetchPolicyError(f"malformed URL {url!r}: {error}")
        scheme = (parsed.scheme or "").lower()
        if scheme not in self.policy.allowed_schemes:
            allowed = ", ".join(sorted(self.policy.allowed_schemes))
            raise FetchPolicyError(
                f"scheme {scheme!r} is not allowed (allowed: {allowed})"
            )
        if not parsed.hostname:
            raise FetchPolicyError(f"URL has no host: {sanitize_url(url)}")
        if parsed.username is not None and not self.policy.allow_userinfo:
            raise FetchPolicyError(
                "URLs containing userinfo (credentials) are not allowed"
            )
        try:
            host = parsed.hostname.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise FetchPolicyError(f"invalid host name {parsed.hostname!r}: {error}")

        default_port = 443 if scheme == "https" else 80
        try:
            port = parsed.port or default_port
        except ValueError:
            raise FetchPolicyError(
                f"port in {sanitize_url(url)} is outside the valid TCP range"
            )
        if not 1 <= port <= 65535:
            raise FetchPolicyError(f"port {port} is outside the valid TCP range")
        if (
            self.policy.allowed_ports is not None
            and port not in self.policy.allowed_ports
        ):
            raise FetchPolicyError(f"port {port} is not allowed by the policy")
        return scheme, host, port

    def _resolve_allowed(
        self, host: str, port: int, hop: FetchHop
    ) -> List[Tuple[int, str]]:
        """Resolve once, validate every answer and return allowed candidates.

        The returned list is the only address list the connection layer ever
        sees: there is no second resolution at connect time, which closes the
        DNS rebinding window.
        """

        answers = self._resolver(host, port)
        if not answers:
            raise FetchPolicyError(f"could not resolve host {host!r}")

        seen = set()
        allowed: List[Tuple[int, str]] = []
        last_reason = ""
        reason: Optional[str]
        for family, ip_text in answers:
            if ip_text in seen:
                continue
            seen.add(ip_text)
            try:
                ip = ipaddress.ip_address(ip_text)
            except ValueError:
                reason = f"{ip_text} is not a valid IP address"
                hop.skipped.append(f"{ip_text} -> {reason}")
                last_reason = reason
                continue
            reason = self.policy.check_ip(ip)
            if reason is not None:
                hop.skipped.append(f"{ip_text} -> {reason}")
                last_reason = reason
                continue
            allowed.append((family, ip_text))

        if not allowed:
            raise FetchPolicyError(
                f"host {host!r} resolves only to blocked addresses; "
                f"last rejection: {last_reason}"
            )
        hop.addresses = [ip for _family, ip in allowed]
        return allowed

    # -- timeouts ----------------------------------------------------------

    def _remaining(self, deadline: float) -> float:
        return deadline - time.monotonic()

    def _timeout(self, phase: str, deadline: float, budget: float) -> float:
        remaining = self._remaining(deadline)
        if remaining <= 0:
            raise FetchTimeoutError(
                f"total timeout of {self.policy.total_timeout:g}s exceeded"
            )
        return min(budget, remaining)

    # -- public API --------------------------------------------------------

    def fetch(self, url: str) -> FetchedResource:
        policy = self.policy
        deadline = time.monotonic() + policy.total_timeout
        trace: List[FetchHop] = []

        current_url = url
        method = "GET"
        seen_urls = set()

        for hop_number in range(1, policy.max_redirects + 2):
            hop = FetchHop(hop=hop_number, method=method, url=sanitize_url(current_url))
            trace.append(hop)
            started = time.monotonic()
            conn: Optional[http.client.HTTPConnection] = None
            try:
                scheme, host, port = self._validate_url(current_url)
                hop.host, hop.port = host, port

                candidates = self._resolve_allowed(host, port, hop)

                sock = self._open_socket(candidates, port, deadline)

                if scheme == "https":
                    sock = self._ssl_wrap(
                        sock,
                        host,
                        self._timeout("TLS", deadline, policy.connect_timeout),
                    )

                if scheme == "https":
                    conn = http.client.HTTPSConnection(
                        host, port, timeout=policy.read_timeout
                    )
                else:
                    conn = http.client.HTTPConnection(
                        host, port, timeout=policy.read_timeout
                    )
                # The socket is already opened to a validated, pinned IP, so
                # make sure http.client does not resolve/connect again.
                conn.sock = sock

                response = self._send_request(conn, current_url, method, host, deadline)
                hop.status = response.status
                hop.content_type = response.headers.get("Content-Type")
                hop.content_encoding = response.headers.get("Content-Encoding")

                if response.status in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    conn.close()
                    conn = None
                    if not location:
                        raise FetchRedirectError(
                            f"redirect status {response.status} without a Location header"
                        )
                    next_url = urljoin(current_url, location)
                    hop.finish("redirect", sanitize_url(next_url))
                    hop.elapsed = time.monotonic() - started

                    signature = next_url.split("#", 1)[0].rstrip("/").lower()
                    if signature in seen_urls:
                        raise FetchRedirectError(
                            f"redirect loop detected at {sanitize_url(next_url)}"
                        )
                    seen_urls.add(signature)
                    if hop_number > policy.max_redirects:
                        raise FetchRedirectError(
                            f"more than {policy.max_redirects} redirects"
                        )
                    if response.status == 303:
                        method = "GET"
                    current_url = next_url
                    continue

                if not 200 <= response.status < 300:
                    conn.close()
                    conn = None
                    raise FetchHTTPError(
                        f"server returned HTTP {response.status} {response.reason}"
                    )

                raw_body, charset, content_type = self._read_body(
                    response, conn, hop, deadline
                )
                conn.close()
                conn = None

                hop.bytes_body = len(raw_body)
                hop.finish("ok")
                hop.elapsed = time.monotonic() - started
                text = _decode_body(raw_body, charset)
                return FetchedResource(
                    url=current_url,
                    text=text,
                    content_type=content_type,
                    charset=charset,
                    bytes_body=len(raw_body),
                    trace=list(trace),
                )
            except FetchError as error:
                hop.elapsed = time.monotonic() - started
                if hop.outcome == "pending":
                    hop.finish(
                        "blocked" if isinstance(error, FetchPolicyError) else "error",
                        error.message,
                    )
                if not error.trace:
                    error.trace = list(trace)
                raise
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass

        raise FetchRedirectError(
            f"more than {policy.max_redirects} redirects", trace=list(trace)
        )

    # -- internals ---------------------------------------------------------

    def _open_socket(
        self, candidates: Sequence[Tuple[int, str]], port: int, deadline: float
    ) -> socket.socket:
        last_error: Optional[Exception] = None
        for family, ip in candidates:
            timeout = self._timeout("connect", deadline, self.policy.connect_timeout)
            try:
                return self._connect(family, ip, port, timeout)
            except (TimeoutError, socket.timeout) as error:
                if self._remaining(deadline) <= 0:
                    raise FetchTimeoutError(
                        f"total timeout of {self.policy.total_timeout:g}s exceeded"
                    )
                raise FetchTimeoutError(
                    f"connect to {ip}:{port} timed out after "
                    f"{self.policy.connect_timeout:g}s"
                ) from error
            except OSError as error:
                last_error = error
                continue
        raise FetchError(
            f"could not connect to {port} on any validated address: {last_error}"
        )

    def _send_request(
        self,
        conn: http.client.HTTPConnection,
        url: str,
        method: str,
        host: str,
        deadline: float,
    ) -> http.client.HTTPResponse:
        parsed = urlsplit(url)
        target = parsed.path or "/"
        if parsed.query:
            target = f"{target}?{parsed.query}"
        headers = {
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "close",
            "User-Agent": self._user_agent,
        }
        try:
            conn.sock.settimeout(
                self._timeout("read", deadline, self.policy.read_timeout)
            )
            conn.request(method, target, headers=headers)
            return conn.getresponse()
        except (TimeoutError, socket.timeout) as error:
            if self._remaining(deadline) <= 0:
                raise FetchTimeoutError(
                    f"total timeout of {self.policy.total_timeout:g}s exceeded"
                )
            raise FetchTimeoutError(
                f"read timeout of {self.policy.read_timeout:g}s waiting for {host}"
            ) from error
        except (http.client.HTTPException, OSError) as error:
            raise FetchError(f"connection to {host} failed: {error}") from error

    def _read_body(
        self,
        response: http.client.HTTPResponse,
        conn: http.client.HTTPConnection,
        hop: FetchHop,
        deadline: float,
    ) -> Tuple[bytes, Optional[str], str]:
        policy = self.policy
        essence, charset = _split_content_type(response.headers.get("Content-Type"))
        hop.content_type = essence
        if essence and not (
            essence.startswith("text/")
            or essence in _TEXTUAL_CONTENT_TYPES
            or essence in _GENERIC_CONTENT_TYPES
        ):
            raise FetchContentError(f"Content-Type {essence!r} is not allowed")

        encodings = [
            part.strip().lower()
            for part in (response.headers.get("Content-Encoding") or "").split(",")
            if part.strip()
        ] or ["identity"]
        for encoding in encodings:
            if encoding not in _SUPPORTED_ENCODINGS:
                raise FetchContentError(
                    f"Content-Encoding {encoding!r} is not supported"
                )
        compressed = any(encoding not in ("", "identity") for encoding in encodings)
        decoders = [_new_decoder(encoding) for encoding in encodings]
        decoders = [decoder for decoder in decoders if decoder is not None]

        declared_length = response.headers.get("Content-Length")
        if declared_length is not None and not compressed:
            try:
                if int(declared_length) > policy.max_bytes:
                    raise FetchLimitError(
                        f"Content-Length {declared_length} exceeds maximum of "
                        f"{policy.max_bytes} bytes"
                    )
            except ValueError:
                pass

        chunks: List[bytes] = []
        wire_bytes = 0
        body_bytes = 0
        sniffed = False

        try:
            while True:
                try:
                    conn.sock.settimeout(
                        self._timeout("read", deadline, policy.read_timeout)
                    )
                    chunk = response.read(self.READ_CHUNK)
                except (TimeoutError, socket.timeout) as error:
                    if self._remaining(deadline) <= 0:
                        raise FetchTimeoutError(
                            f"total timeout of {policy.total_timeout:g}s exceeded"
                        )
                    raise FetchTimeoutError(
                        f"read timeout of {self.policy.read_timeout:g}s while "
                        "downloading body"
                    ) from error
                except (http.client.HTTPException, OSError) as error:
                    raise FetchError(f"connection interrupted: {error}") from error

                if not chunk:
                    break

                wire_bytes += len(chunk)
                expanded = chunk
                for decoder in decoders:
                    try:
                        expanded = decoder.decompress(expanded)
                    except zlib.error as error:
                        raise FetchContentError(
                            f"could not decompress response body: {error}"
                        ) from error
                body_bytes += len(expanded)

                if body_bytes > policy.max_bytes:
                    raise FetchLimitError(
                        f"response body exceeds maximum of {policy.max_bytes} bytes"
                    )
                if (
                    compressed
                    and wire_bytes >= 128
                    and body_bytes / wire_bytes > policy.max_compression_ratio
                ):
                    raise FetchLimitError(
                        "compression bomb detected: decompressed body is "
                        f"{body_bytes // wire_bytes}x larger than the "
                        f"{policy.max_compression_ratio:g}x limit"
                    )

                chunks.append(expanded)
                hop.bytes_wire, hop.bytes_body = wire_bytes, body_bytes

                if not sniffed and body_bytes >= policy.sniff_bytes:
                    prefix = b"".join(chunks)[: policy.sniff_bytes]
                    if policy.block_binary and _looks_binary(prefix):
                        raise FetchContentError(
                            "response body appears to be binary data"
                        )
                    sniffed = True

            tail = b""
            for decoder in decoders:
                try:
                    tail += decoder.flush()
                except zlib.error as error:
                    raise FetchContentError(
                        f"could not decompress response body: {error}"
                    ) from error
            if tail:
                body_bytes += len(tail)
                if body_bytes > policy.max_bytes:
                    raise FetchLimitError(
                        f"response body exceeds maximum of {policy.max_bytes} bytes"
                    )
                chunks.append(tail)
                hop.bytes_body = body_bytes
        except FetchError:
            # Cancel the stream immediately instead of draining it.
            if conn.sock is not None:
                try:
                    conn.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            raise

        raw_body = b"".join(chunks)
        if not sniffed and policy.block_binary and _looks_binary(raw_body):
            raise FetchContentError("response body appears to be binary data")
        return raw_body, charset, essence
