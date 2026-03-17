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
        req_ids=["long_request"],
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
        req_ids=["short_request"],
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


def test_mlfq_las_penalty_penalizes_attained_service():
    scheduler = create_scheduler_with_priority(
        block_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    scheduler.scheduler_config.mlfq_las_token_chunk_size = 256
    scheduler.scheduler_config.mlfq_las_weight = 1.0

    request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=512,
        max_tokens=256,
        block_size=1,
    )[0]
    request.num_computed_tokens = 512

    assert scheduler._las_penalty(request) == int((512 // 256) * 1.0)  # == 2

    # Disabled when chunk_size is 0.
    scheduler.scheduler_config.mlfq_las_token_chunk_size = 0
    assert scheduler._las_penalty(request) == 0


def test_mlfq_sjf_penalty_proportional_to_remaining():
    scheduler = create_scheduler_with_priority(
        block_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    scheduler.scheduler_config.mlfq_sjf_token_chunk_size = 256
    scheduler.scheduler_config.mlfq_sjf_weight = 1.0

    request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=512,
        max_tokens=256,
        block_size=1,
    )[0]
    # total = 512 + 256 = 768, remaining = 768 - 0 = 768
    request.num_computed_tokens = 0
    assert scheduler._sjf_penalty(request) == int((768 // 256) * 1.0)  # == 3

    # After computing 256 tokens, remaining = 768 - 256 = 512
    request.num_computed_tokens = 256
    assert scheduler._sjf_penalty(request) == int((512 // 256) * 1.0)  # == 2


def test_mlfq_locality_boost_with_cap():
    scheduler = create_scheduler_with_priority(
        block_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    scheduler.scheduler_config.mlfq_locality_weight = 1.0
    scheduler.scheduler_config.mlfq_token_chunk_size = 256

    request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
    )[0]
    request.num_cached_tokens = 768

    assert scheduler._locality_boost(request) == 3  # 768 // 256

    # Apply cap.
    scheduler.scheduler_config.mlfq_locality_max_boost = 1
    assert scheduler._locality_boost(request) == 1


def test_mlfq_backpressure_penalty_above_threshold():
    scheduler = create_scheduler_with_priority(
        block_size=1,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    scheduler.scheduler_config.mlfq_enable_experimental = True
    scheduler.scheduler_config.output_backpressure_pending_tokens = 64
    scheduler.scheduler_config.output_backpressure_penalty = 2

    request = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
    )[0]
    request.output_pending_tokens = 128

    penalty = scheduler._backpressure_penalty(request, time.monotonic())
    assert penalty == max(1, 128 // 64) * 2  # == 4

    # Below threshold: no penalty.
    request.output_pending_tokens = 32
    assert scheduler._backpressure_penalty(request, time.monotonic()) == 0


def test_mlfq_low_pressure_bypass():
    scheduler = create_scheduler_with_priority(
        max_num_seqs=1,
        max_num_batched_tokens=1,
        block_size=1,
    )
    scheduler.scheduler_config.mlfq_enable_experimental = True
    scheduler.scheduler_config.mlfq_low_pressure_waiting_threshold = 8

    # Add 3 requests (below threshold of 8).
    requests_low = create_requests_with_priority(
        num_requests=3,
        priorities=[0, 0, 0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
    )
    for r in requests_low:
        scheduler.add_request(r)
    assert scheduler._is_low_pressure() is True

    # Add 6 more (total 9, above threshold of 8).
    requests_high = create_requests_with_priority(
        num_requests=6,
        priorities=[0, 0, 0, 0, 0, 0],
        num_tokens=1,
        max_tokens=1,
        block_size=1,
        starting_idx=3,
    )
    for r in requests_high:
        scheduler.add_request(r)
    assert scheduler._is_low_pressure() is False
