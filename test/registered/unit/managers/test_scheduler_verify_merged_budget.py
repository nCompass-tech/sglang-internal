"""A verify-merged mixed step charges W prefill tokens per running row only when
the running batch can actually be merged."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Stop(Exception):
    pass


def reserved(verify_merged, **running):
    s = Scheduler.__new__(Scheduler)
    s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
    s.enable_priority_preemption = s.is_hybrid_swa = False
    s.waiting_queue = ["req"]
    s.chunked_req = s.min_free_slots_delayer = s.dynamic_chunk_sizer = None
    s.get_num_allocatable_reqs = lambda *a, **k: 8
    s.policy = MagicMock()
    s.processed_tokens_counter = s.page_size = s.tree_cache = None
    s.token_to_kv_pool_allocator = s.new_token_ratio_tracker = MagicMock()
    s.max_prefill_tokens = s.max_prefill_bs = s.max_running_requests = 8192
    s.priority_scheduling_preemption_threshold = s.dllm_config = None
    s.chunked_prefill_size = 8192
    s.tp_worker = MagicMock()
    s.is_mixed_chunk = True
    s.verify_merged_mixed = verify_merged
    s.mixed_tokens_per_running_row = 4 if verify_merged else 1
    batch = dict(
        reqs=[SimpleNamespace(beam_group=None)] * 3,
        batch_is_full=False,
        return_logprob=False,
        has_grammar=False,
        sampling_info=SimpleNamespace(has_custom_logit_processor=False),
    )
    batch.update(running)
    running_batch = SimpleNamespace(is_empty=lambda: not batch["reqs"], **batch)

    def adder(*args, **kwargs):
        raise _Stop(args[7])

    with (
        patch.object(scheduler_module, "PrefillAdder", adder),
        patch.object(scheduler_module, "get_schedule", MagicMock()),
    ):
        try:
            Scheduler._get_new_batch_prefill_raw(s, None, running_batch)
        except _Stop as stop:
            return stop.args[0]
    raise AssertionError("PrefillAdder was not built")


class TestVerifyMergedBudget(CustomTestCase):
    def test_mergeable_running_batch_reserves_w_per_row(self):
        self.assertEqual(reserved(True), 3 * 4)

    def test_refused_running_batch_reserves_one_per_row(self):
        refusals = dict(
            return_logprob=True,
            has_grammar=True,
            sampling_info=SimpleNamespace(has_custom_logit_processor=True),
            reqs=[SimpleNamespace(beam_group=object())] * 3,
        )
        for key, value in refusals.items():
            with self.subTest(key):
                self.assertEqual(reserved(True, **{key: value}), 3)

    def test_flag_off_reserves_one_per_row(self):
        self.assertEqual(reserved(False), 3)


if __name__ == "__main__":
    unittest.main()
