"""
utils/ring_buffer.py

Thread-safe fixed-length circular (ring) buffer.

Used to hold the sliding window of sensor/prediction values displayed
on the live graphs without growing memory unboundedly.
"""

from collections import deque
from threading import Lock
from typing import Generic, Iterator, List, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-capacity FIFO buffer backed by a ``collections.deque``.

    When full, the oldest element is silently discarded on append.

    Args:
        capacity: Maximum number of elements retained.

    Example::

        buf = RingBuffer[float](capacity=120)
        for v in sensor_readings:
            buf.append(v)
        print(buf.to_list())
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._buf: deque = deque(maxlen=capacity)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, item: T) -> None:
        """Append *item*, dropping the oldest element if at capacity."""
        with self._lock:
            self._buf.append(item)

    def clear(self) -> None:
        """Remove all elements."""
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def to_list(self) -> List[T]:
        """Return a snapshot list (oldest → newest)."""
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def __iter__(self) -> Iterator[T]:
        with self._lock:
            return iter(list(self._buf))

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buf) == self._capacity
