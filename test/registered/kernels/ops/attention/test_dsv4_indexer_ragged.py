"""DeepSeek-V4 eager ragged C4 indexer vs the paged reference.

Synthetic MIXED chunk: prefill sequences whose extend rows fill 3,840 rows, plus
20 x 6 verify rows, padded to 4,096 rows as the captured segment sees it. Same fp8
K cache pages, page tables, q and weights for:
  paged  = deep_gemm.fp8_paged_mqa_logits over all 4,096 padded rows + one v2 top-k
  ragged = one batched K gather + one deep_gemm.fp8_mqa_logits (compressed logits)
           + one v2 top-k over the prefill rows, plus the paged kernel + top-k over
           the verify rows
Prefill-row logits must be bitwise equal over [0, c4len), and the top-k sets
identical on every real row. The same v2 top-k runs on both sides; it is
deterministic only on rows with at most 2048 candidates in the threshold coarse
bin (see test_dsv4_topk_det.py).
"""

import sys

import pytest
import torch

deep_gemm = pytest.importorskip("deep_gemm")

from sglang.kernels.ops.attention.dsa.index_buf_accessor import _get_k_and_s_triton
from sglang.kernels.ops.attention.dsv4 import topk as topk_mod
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    build_ragged_indexer_plan,
    ragged_indexer_logits,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="DeepSeek-V4 indexer kernels need Blackwell (SM100+)",
)

dev = "cuda"
H, D, TOPK, C4PAGE = 64, 128, 512, 64
ROWS_PRE, N_VER, VER_ROWS, PADDED = 3840, 20, 6, 4096


def topk_launch(logits, lens, pt, out, meta=None):
    if meta is None:
        meta = topk_mod.plan_topk_v2(lens)
    topk_mod.topk_transform_512_v2(logits, lens, pt, out, C4PAGE, meta, None)


def build_case(g, contexts, ver_contexts):
    """Returns dict with cache buf, page_table_all [PADDED, max_pages], c4 lens [PADDED], ext, seq_lens."""
    n_seq = len(contexts)
    ext = [ROWS_PRE // n_seq] * n_seq
    ext[-1] += ROWS_PRE - sum(ext)
    seqs = list(zip(contexts, ext)) + [(c, VER_ROWS) for c in ver_contexts]
    pages_per = [(c // 4 + C4PAGE - 1) // C4PAGE for c, _ in seqs]
    max_pages = max(pages_per)
    n_pages = sum(pages_per) + 1  # page 0 unused (padding rows point at it)
    # K cache: [n_pages, 64*132] uint8 = 64 tokens x (128 fp8 + 4 B fp32 scale)
    k = (
        (torch.randn(n_pages, C4PAGE, D, generator=g) * 0.5)
        .to(FP8_DTYPE)
        .view(torch.uint8)
    )
    sc = (
        (torch.rand(n_pages, C4PAGE, generator=g) * 0.05 + 0.01)
        .to(torch.float32)
        .view(torch.uint8)
        .view(n_pages, C4PAGE, 4)
    )
    buf = (
        torch.cat([k.reshape(n_pages, -1), sc.reshape(n_pages, -1)], dim=1)
        .contiguous()
        .to(dev)
    )
    assert buf.shape[1] == C4PAGE * 132
    rows = sum(n for _, n in seqs)
    page_table_all = torch.zeros(PADDED, max_pages, dtype=torch.int32)
    c4 = torch.ones(
        PADDED, dtype=torch.int32
    )  # padded rows: length 1 (as match_num_queries pads)
    row = 0
    pg = 1
    for (ctx, n), npg in zip(seqs, pages_per):
        pt = torch.arange(pg, pg + npg, dtype=torch.int32)
        pg += npg
        for i in range(n):
            pos = ctx - n + i
            page_table_all[row, :npg] = pt
            c4[row] = min(ctx // 4, (pos + 1) // 4)
            row += 1
    assert row == rows
    q = (torch.randn(PADDED, H, D, generator=g) * 0.5).to(FP8_DTYPE)
    w = (torch.rand(PADDED, H, generator=g) + 0.1).to(torch.float32)
    return dict(
        buf=buf,
        page_table=page_table_all.to(dev),
        c4=c4.to(dev),
        q=q.to(dev),
        w=w.to(dev),
        ext=ext,
        seq_lens=list(contexts),
        real=rows,
        mixed_t=ROWS_PRE,
        max_pages=max_pages,
    )


def paged_path(case, a, b, topk_out):
    pt = case["page_table"][a:b]
    c4l = case["c4"][a:b]
    cache = case["buf"].view(case["buf"].shape[0], C4PAGE, 1, 132)
    meta = deep_gemm.get_paged_mqa_logits_metadata(
        c4l[:, None], C4PAGE, deep_gemm.get_num_sms()
    )
    logits = deep_gemm.fp8_paged_mqa_logits(
        case["q"][a:b].unsqueeze(1),
        cache,
        case["w"][a:b],
        c4l[:, None],
        pt,
        meta,
        case["max_pages"] * C4PAGE,
        False,
    )
    topk_launch(logits, c4l.contiguous(), pt, topk_out)
    return logits


def ragged_plan(case):
    return build_ragged_indexer_plan(
        ext_lens=case["ext"],
        seq_lens=case["seq_lens"],
        c4_seq_lens_all=case["c4"],
        page_table_all=case["page_table"],
        max_c4_seq_len=case["max_pages"] * C4PAGE,
        c4_page_size=C4PAGE,
        topk=TOPK,
        mixed_t=case["mixed_t"],
        real=case["real"],
    )


def gather_fn(buf):
    def gather(seq_len_tensor, page_indices, seq_len_sum, max_seq_len):
        return _get_k_and_s_triton(
            buf=buf,
            page_indices=page_indices,
            seq_lens=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
            page_size=C4PAGE,
            index_head_dim=D,
        )

    return gather


def ragged_path(case, plan, topk_out):
    rg = plan.ragged
    logits = ragged_indexer_logits(
        q_indexer=case["q"], weights=case["w"], gather=gather_fn(case["buf"]), plan=rg
    )
    topk_launch(
        logits,
        rg.lens,
        case["page_table"][: rg.rows],
        topk_out[: rg.rows],
        meta=rg.topk_meta,
    )
    for a, b in plan.paged_ranges:
        paged_path(case, a, b, topk_out[a:b])
    return logits


def run_case(contexts, ver_contexts, g):
    case = build_case(g, contexts, ver_contexts)
    real, mixed_t = case["real"], case["mixed_t"]
    plan = ragged_plan(case)
    rg = plan.ragged
    assert rg is not None and rg.rows == mixed_t, (rg.rows if rg else None, mixed_t)
    assert plan.paged_ranges == [(mixed_t, real)], plan.paged_ranges
    out_p = torch.full((PADDED, TOPK), -1, dtype=torch.int32, device=dev)
    out_r = torch.full((PADDED, TOPK), -1, dtype=torch.int32, device=dev)
    r = {}

    # numerics: logits bitwise on the prefill rows over [0, c4len)
    lp = paged_path(case, 0, PADDED, out_p)
    lr = ragged_path(case, plan, out_r)
    w = min(lp.shape[1], lr.shape[1])
    cols = torch.arange(w, device=dev, dtype=torch.int32)
    valid = cols[None, :] < rg.lens[:, None]
    ne = (lp[:mixed_t, :w] != lr[:mixed_t, :w]) & valid
    r["logits_prefill_rows_mismatch"] = int(ne.any(dim=1).sum())
    r["logits_nan_in_valid"] = int((torch.isnan(lp[:mixed_t, :w]) & valid).sum())
    # top-k sets (same v2 kernel on both sides; exact while the threshold bin
    # holds <= 2048 candidates)
    sp = torch.sort(out_p[:real], dim=1).values
    sr = torch.sort(out_r[:real], dim=1).values
    r["topk_rows_differ"] = int((sp != sr).any(dim=1).sum())
    return r


@pytest.mark.parametrize(
    "contexts",
    [
        [8192, 30720, 61440],
        [8192, 8192, 8192],
        [30720, 30720],
        [61440, 61440],
        [71680],
    ],
    ids=[
        "mixed_8k_30k_60k",
        "uniform_8k_x3",
        "uniform_30k_x2",
        "uniform_60k_x2",
        "uniform_70k_x1",
    ],
)
def test_ragged_matches_paged(contexts):
    g = torch.Generator().manual_seed(7)
    ver = [int(x) for x in torch.randint(8192, 40961, (N_VER,), generator=g).tolist()]
    r = run_case(contexts, ver, g)
    assert r["logits_prefill_rows_mismatch"] == 0, r
    assert r["logits_nan_in_valid"] == 0, r
    assert r["topk_rows_differ"] == 0, r


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
