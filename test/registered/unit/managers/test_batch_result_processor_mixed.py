"""Unit tests for splitting a verify-merged mixed batch in the result processor."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components import batch_result_processor
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    _decode_total_seq_lens,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestProcessBatchResultMixed(CustomTestCase):
    def test_decode_half_covers_only_tail_rows(self):
        # Rows [0, 2) are prefill rows, rows [2, 5) verify rows.
        batch = SimpleNamespace(
            num_prefill_rows=2,
            reqs=["p0", "p1", "d0", "d1", "d2"],
            extend_lens=[100, 200, 4, 4, 4],
            prefix_lens=[0, 0, 7, 8, 9],
            seq_lens_cpu=torch.tensor([100, 200, 7, 8, 9]),
        )
        result = SimpleNamespace(
            next_token_ids=torch.arange(5),
            num_correct_drafts=0,
            num_correct_drafts_per_req_cpu=None,
            num_block_accept_tokens=0,
            num_cap_tokens=0,
        )
        seen = {}
        processor = SimpleNamespace(
            process_batch_result_prefill=lambda b, r: None,
            process_batch_result_decode=lambda b, r: seen.setdefault("dec", b),
        )
        with patch.object(
            batch_result_processor,
            "get_observability",
            return_value=SimpleNamespace(enable_metrics=False),
        ):
            SchedulerBatchResultProcessor.process_batch_result_mixed(
                processor, batch, result
            )

        dec = seen["dec"]
        self.assertEqual(dec.reqs, ["d0", "d1", "d2"])
        self.assertEqual(dec.seq_lens_cpu.tolist(), [7, 8, 9])
        self.assertEqual(_decode_total_seq_lens(dec), 24)

    def test_filter_clears_prefill_split(self):
        # The scheduler filters the merged batch before it becomes the running
        # (decode) batch; with no row finished, filter_batch returns early.
        batch = SimpleNamespace(
            num_prefill_rows=1,
            beam_tail=None,
            reqs=[SimpleNamespace(finished=lambda: False)] * 3,
        )
        ScheduleBatch.filter_batch(batch)
        self.assertIsNone(batch.num_prefill_rows)


if __name__ == "__main__":
    unittest.main()
