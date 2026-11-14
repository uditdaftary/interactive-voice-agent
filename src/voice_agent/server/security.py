"""Host allowlist and same-origin checks for the HTTP and WebSocket entry points.

The server has no login, so the network boundary is the only protection: it must answer only to
loopback Host names, and only to pages it served itself. A foreign page can POST a multipart form
to localhost without a CORS preflight, and a WebSocket handshake is never preflighted, so both
need an explicit Origin check. Likewise a form can send text/plain or urlencoded bodies without a
preflight, so the JSON routes insist on a JSON Content-Type.
"""

from collections.abc import Sequence
import logging
import re
from urllib.parse import urlsplit

from fastapi import HTTPException, Request
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Read through the module at call time so tests can widen it (TestClient sends Host: testserver).
DEFAULT_ALLOWED_HOSTS: tuple[str, ...] = ("127.0.0.1", "localhost")

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}
# Same shape FastAPI itself treats as JSON: application/json or application/<vendor>+json.
_JSON_MEDIA_TYPE = re.compile(r"^application/([\w.-]+\+)?json$")


def is_json_content_type(content_type: str | None) -> bool:
    """True for application/json (and +json types), ignoring parameters such as charset and case."""
    return _JSON_MEDIA_TYPE.match((content_type or "").split(";", 1)[0].strip().lower()) is not None


async def require_json_content_type(request: Request) -> None:
    """Route dependency: reject any other Content-Type with 415 before the body is looked at."""
    if not is_json_content_type(request.headers.get("content-type")):
        raise HTTPException(status_code=415, detail="Content-Type must be application/json.")


def origin_allowed(origin: str | None, host: str | None) -> bool:
    """True when the request carries no Origin, or an Origin whose host:port is the request's own Host.

    Host names compare case-insensitively and a missing port means the scheme's default port.
    `Origin: null` (file pages, sandboxed frames) and anything unparseable is rejected.
    """
    if origin is None:
        return True
    try:
        origin_parts = urlsplit(origin)
        host_parts = urlsplit("//" + host) if host else None
        origin_port, host_port = origin_parts.port, (host_parts.port if host_parts else None)
    except ValueError:  # malformed port or bracketed host
        return False
    default_port = _DEFAULT_PORTS.get(origin_parts.scheme)
    if default_port is None or not origin_parts.hostname or host_parts is None or not host_parts.hostname:
        return False
    return (origin_parts.hostname, origin_port or default_port) == (host_parts.hostname, host_port or default_port)


class OriginCheckMiddleware:
    """Reject state-changing HTTP requests whose Origin is not this server's own Host (403)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in _SAFE_METHODS:
            headers = Headers(scope=scope)
            if not origin_allowed(headers.get("origin"), headers.get("host")):
                # Header values are attacker-controlled: %r escapes them and the precision caps their length.
                # Method, path and the two headers only; never the body, which may hold an API key.
                logger.warning(
                    "Rejected cross-origin %s %.100r: Origin %.100r, Host %.100r",
                    scope["method"],
                    scope["path"],
                    headers.get("origin"),
                    headers.get("host"),
                )
                response = JSONResponse({"detail": "cross-origin request rejected"}, status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class LoggingTrustedHostMiddleware(TrustedHostMiddleware):
    """Starlette's Host allowlist, plus a WARNING for every request it turns away.

    The parent has no rejection hook, so the app it wraps marks the scope as admitted. A request that
    comes back without the mark was refused (400, or a redirect) by the parent.
    """

    def __init__(self, app: ASGIApp, allowed_hosts: Sequence[str] | None = None) -> None:
        async def admitted(scope: Scope, receive: Receive, send: Send) -> None:
            scope["host_admitted"] = True
            await app(scope, receive, send)

        super().__init__(admitted, allowed_hosts=allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await super().__call__(scope, receive, send)
        if scope["type"] in ("http", "websocket") and not scope.get("host_admitted"):
            logger.warning(
                "Rejected %s request to %.100r: Host %.100r is not in the allowlist",
                scope["type"],
                scope["path"],
                Headers(scope=scope).get("host"),
            )
