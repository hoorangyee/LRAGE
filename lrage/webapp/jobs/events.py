import queue
import threading
from collections import deque
from typing import Any, Dict, List, Optional

# Sentinel put on subscriber queues when the stream ends.
STREAM_END = None

RING_SIZE = 2000


class EventBuffer:
    """Per-run event log: a ring buffer for Last-Event-ID replay plus live
    fan-out to subscriber queues. Published events get a monotonically
    increasing id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ring: deque = deque(maxlen=RING_SIZE)
        self._subscribers: List[queue.Queue] = []
        self._next_id = 1
        self._closed = False

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            stamped = {"id": self._next_id, **event}
            self._next_id += 1
            self._ring.append(stamped)
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(stamped)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            subscribers = list(self._subscribers)
        for q in subscribers:
            q.put(STREAM_END)

    @property
    def closed(self) -> bool:
        return self._closed

    def subscribe(self, last_event_id: Optional[int] = None) -> queue.Queue:
        """Register a subscriber and replay buffered events newer than
        last_event_id onto its queue (before any live events)."""
        q: queue.Queue = queue.Queue()
        with self._lock:
            for event in self._ring:
                if last_event_id is None or event["id"] > last_event_id:
                    q.put(event)
            if self._closed:
                q.put(STREAM_END)
            else:
                self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
