# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import pytest

from .test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
    make_output,
)

pytestmark = pytest.mark.cpu_test


def test_mlfq_short_request_preempts_long_with_budget_slice():
    scheduler = create_scheduler_with_priority(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        block_size=1,
    )
    scheduler.scheduler_config.mlfq_token_chunk_size = 1
    scheduler.scheduler_config.mlfq_waiting_budget_fraction = 1.0
    scheduler.scheduler_config.mlfq_aging_seconds = 0.0

    long_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=4,
        max_tokens=8,
        block_size=1,
    )[0]
    scheduler.add_request(long_request)

    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(scheduler_output, make_output(scheduler))

    short_request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
    )[0]
    scheduler.add_request(short_request)

    scheduler_output = scheduler.schedule()
    scheduled_new_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
    scheduled_cached_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)

    assert short_request.request_id in scheduled_new_ids
    assert long_request.request_id not in scheduled_cached_ids

    scheduler.update_from_output(scheduler_output, make_output(scheduler))

    long_finished = False
    for _ in range(50):
        scheduler_output = scheduler.schedule()
        if not scheduler.running:
            break
        scheduler.update_from_output(scheduler_output, make_output(scheduler))
        if long_request.request_id not in scheduler.requests:
            long_finished = True
            break
    assert long_finished


def test_mlfq_aging_boosts_waiting_request():
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,
        max_num_batched_tokens=1,
        block_size=1,
    )
    scheduler.scheduler_config.mlfq_token_chunk_size = 4
    scheduler.scheduler_config.mlfq_aging_seconds = 1.0

    requests = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 0],
        arrival_times=[2.0, 1.0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
    )
    req_aged, req_recent = requests[0], requests[1]
    scheduler.add_request(req_aged)
    scheduler.add_request(req_recent)

    now = time.monotonic()
    scheduler.waiting_since[req_aged.request_id] = now - 10.0
    scheduler.waiting_since[req_recent.request_id] = now

    scheduler_output = scheduler.schedule()
    assert scheduler_output.scheduled_new_reqs[0].req_id == req_aged.request_id
