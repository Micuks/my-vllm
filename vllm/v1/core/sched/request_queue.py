# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.sched.goodput_scorer import GoodputScorer


class _GoodputEntry(NamedTuple):
    score: float
    seq: int
    request: Request


class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""

    FCFS = "fcfs"
    PRIORITY = "priority"
    GOODPUT = "goodput"


class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: "RequestQueue") -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue.

        Note: The requests will be prepended in reverse order of their
        appearance in the `requests` queue.
        """
        self.extendleft(requests)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Respects the ordering defined in the Request class, where
    requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[Request] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        return heapq.heappop(self._heap)

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        return self._heap[0]

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap.remove(request)
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        self._heap = [r for r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy)


class GoodputRequestQueue(RequestQueue):
    """Heap-backed queue that orders requests by a dynamic goodput score.

    Unlike PriorityRequestQueue (which sorts on Request.__lt__ using static
    fields), this queue's ordering depends on time and per-request observed
    state. The scheduler must call refresh(now) at the start of each
    scheduling pass to recompute scores and re-heapify before any
    pop/peek/iter operation.

    Each heap entry is (score, sequence_id, request); sequence_id is a
    monotonic tie-breaker to keep ordering deterministic when scores tie.
    """

    def __init__(self, scorer: "GoodputScorer") -> None:
        self._scorer = scorer
        self._heap: list[_GoodputEntry] = []
        self._seq: int = 0
        self._last_refresh: float = 0.0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def refresh(self, now: float) -> None:
        """Recompute scores for every queued request and re-heapify.
        Cost: O(n) in queue size."""
        if not self._heap:
            self._last_refresh = now
            return
        self._heap = [
            _GoodputEntry(self._scorer.score(entry.request, now), entry.seq, entry.request)
            for entry in self._heap
        ]
        heapq.heapify(self._heap)
        self._last_refresh = now

    def add_request(self, request: Request) -> None:
        entry = _GoodputEntry(
            self._scorer.score(request, self._last_refresh),
            self._next_seq(),
            request,
        )
        heapq.heappush(self._heap, entry)

    def pop_request(self) -> Request:
        if not self._heap:
            raise IndexError("pop from empty goodput queue")
        return heapq.heappop(self._heap).request

    def peek_request(self) -> Request:
        if not self._heap:
            raise IndexError("peek from empty goodput queue")
        return self._heap[0].request

    def prepend_request(self, request: Request) -> None:
        """No prepend semantics in a score-ordered queue: just add."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        for i, entry in enumerate(self._heap):
            if entry.request is request:
                self._heap[i] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                return
        raise ValueError(f"request {request.request_id} not in queue")

    def remove_requests(self, requests: Iterable[Request]) -> None:
        # Use id() for identity comparison; supports any iterable (even of
        # objects that override __eq__ in a way that would defeat set hashing).
        target_ids = {id(r) for r in requests}
        self._heap = [
            entry for entry in self._heap if id(entry.request) not in target_ids
        ]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        heap_copy = self._heap[:]
        while heap_copy:
            yield heapq.heappop(heap_copy).request


def create_request_queue(
    policy: SchedulingPolicy,
    *,
    scorer: "GoodputScorer | None" = None,
) -> RequestQueue:
    """Create a request queue for the given scheduling policy.

    `scorer` is required iff policy == GOODPUT; ignored otherwise."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.GOODPUT:
        if scorer is None:
            raise ValueError(
                "GoodputScorer must be provided for policy='goodput'"
            )
        return GoodputRequestQueue(scorer)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
