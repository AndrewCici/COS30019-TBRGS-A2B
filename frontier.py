from collections import deque
import heapq
from typing import Any, List, Tuple

class Frontier:

    # Store modes
    def __init__(self, mode: str):
        self.mode = mode
        if mode == 'stack':   #LIFO
            self._container: List[Any] = []
        elif mode == 'queue':   #FIFO
            self._container: deque = deque()
        elif mode == 'priority':   
            self._container: List[Tuple[float, int, Any]] = []   # heap - use later
        else:
            raise ValueError(f"Unknown frontier mode: {mode}")

    # Add node according to mode
    def push(self, item: Any) -> None:
        if self.mode == 'stack':
            self._container.append(item)
        elif self.mode == 'queue':
            self._container.append(item)
        elif self.mode == 'priority':
            heapq.heappush(self._container, item)

    # Remove and return node according to mode
    def pop(self) -> Any:
        if self.mode == 'stack':
            return self._container.pop()
        elif self.mode == 'queue':
            return self._container.popleft()
        elif self.mode == 'priority':
            return heapq.heappop(self._container)

    # Check if frontier is empty
    def is_empty(self) -> bool:
        return len(self._container) == 0