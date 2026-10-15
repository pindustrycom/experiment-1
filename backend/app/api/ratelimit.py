"""In-process sliding-window rate limiter (one API process per container; Caddy is the only ingress)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindow:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                return False, int(window - (now - q[0])) + 1
            q.append(now)
            if len(self._hits) > 50000:  # bound memory
                for k in list(self._hits)[:10000]:
                    if not self._hits[k]:
                        del self._hits[k]
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindow()


def client_ip(request) -> str:
    return request.client.host if request.client else "unknown"
