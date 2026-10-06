"""robots.txt gating with a per-host cache.

The pipeline is a polite crawler by default: it identifies itself with a
descriptive User-Agent, backs off with a delay between requests, and checks
``robots.txt`` before fetching. The gate is injectable so tests (and the
bundled demo site, which has no robots file) never touch the network.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib import robotparser
from urllib.parse import urlsplit

RobotFetcher = Callable[[str], str | None]


class RobotsGate:
    """Decides whether a URL may be fetched, honoring per-host robots.txt.

    - robots.txt is fetched once per host and cached for the gate's lifetime.
    - If robots.txt cannot be retrieved (404, network error, non-200), the
      host is treated as fully allowed — the same default most crawlers use.
    - Non-http(s) schemes (e.g. the bundled demo site) are always allowed.
    """

    def __init__(
        self,
        user_agent: str,
        respect_robots: bool = True,
        fetch_robots: RobotFetcher | None = None,
    ) -> None:
        self._user_agent = user_agent
        self._respect = respect_robots
        self._fetch_robots = fetch_robots
        self._cache: dict[str, robotparser.RobotFileParser] = {}
        self._allowed_hosts_with_no_robots: set[str] = set()

    # ------------------------------------------------------------------ #

    def allowed(self, url: str) -> bool:
        """True when fetching ``url`` is permitted."""
        if not self._respect:
            return True
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return True  # bundled demo site / file sources have no robots.txt

        host = parts.netloc.lower()
        if host in self._allowed_hosts_with_no_robots:
            return True
        parser = self._parser_for(parts.scheme, host)
        if parser is None:
            self._allowed_hosts_with_no_robots.add(host)
            return True
        return parser.can_fetch(self._user_agent, url)

    # ------------------------------------------------------------------ #

    def _parser_for(self, scheme: str, host: str) -> robotparser.RobotFileParser | None:
        if host in self._cache:
            return self._cache[host]
        if self._fetch_robots is None:
            return None

        body = self._fetch_robots(f"{scheme}://{host}/robots.txt")
        if body is None:
            return None
        parser = robotparser.RobotFileParser()
        parser.parse(body.splitlines())
        self._cache[host] = parser
        return parser
