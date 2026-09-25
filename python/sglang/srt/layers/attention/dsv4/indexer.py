from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4 import (
    fused_q_indexer_rope_hadamard_fp4_quant,
    fused_q_indexer_rope_hadamard_quant,
    plan_topk_v2,
    topk_transform_512,
    topk_transform_512_v2,
)
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.srt.layers.attention.dsv4.metadata import (
    NonPagedIndexerPlan,
    PagedIndexerMetadata,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.state_capturer.indexer_topk import get_global_indexer_capturer
from sglang.srt.utils import add_prefix, is_cuda, is_hip, is_xpu
from sglang.srt.utils.common import is_sm120_supported

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.layers.attention.dsv4.compressor import (
        CompressorBackendMixin,
    )
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


IndexerQuery: TypeAlias = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


_arange_cache = {}


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Vectorized implementation compatible with CUDA graph capture."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]

    assert head_dim == 128
    assert block_size == 64
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_num_pages = page_table.shape[1]
    SCALE_OFFSET = block_size * head_dim
    total_dim = block_size * (head_dim + 4)

    kvcache_flat = kvcache_fp8.view(-1, total_dim)

    pages_clamped = page_table.clamp(min=0)
    kvcache_gathered = kvcache_flat[pages_clamped]

    kv_values_raw = kvcache_gathered[..., :SCALE_OFFSET].contiguous()
    kv_values_fp8 = kv_values_raw.view(dtype=FP8_DTYPE)
    kv_values = kv_values_fp8.to(torch.bfloat16)
    kv_values = kv_values.reshape(batch_size, max_num_pages * block_size, head_dim)

    kv_scales_raw = kvcache_gathered[..., SCALE_OFFSET:].contiguous()
    kv_scales = kv_scales_raw.view(dtype=torch.float32)
    kv_scales = kv_scales.reshape(batch_size, max_num_pages * block_size)

    q_float = q_fp8[:, 0].to(torch.bfloat16)
    scores = torch.bmm(kv_values, q_float.transpose(1, 2))
    scores = F.relu(scores)
    scores = scores * weight.unsqueeze(1)
    scores = scores.sum(dim=2)
    scores = scores * kv_scales

    padded_seq_len = max_num_pages * block_size
    cache = _arange_cache
    arange_key = f"arange_{padded_seq_len}_{scores.device}"
    if arange_key not in cache:
        cache[arange_key] = torch.arange(padded_seq_len, device=scores.device)
    positions = cache[arange_key].unsqueeze(0)
    valid_mask = positions < seq_lens.unsqueeze(1)
    scores = scores.masked_fill(~valid_mask, 0.0)

    if padded_seq_len < max_seq_len:
        scores = F.pad(scores, (0, max_seq_len - padded_seq_len), value=0.0)
    else:
        scores = scores[:, :max_seq_len]

    return scores


def _aiter_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Wrapper adapting aiter's deepgemm_fp8_paged_mqa_logits to SGLang's interface."""
    from aiter.ops.triton.attention.pa_mqa_logits import (
        deepgemm_fp8_paged_mqa_logits,
    )

    batch_size = q_fp8.shape[0]
    next_n = q_fp8.shape[1]
    total_tokens = batch_size * next_n
    _sl = seq_lens.squeeze(-1) if seq_lens.dim() == 2 else seq_lens
    kv_block_size = kvcache_fp8.shape[1]
    logits = torch.empty(
        total_tokens,
        max_seq_len,
        dtype=torch.float32,
        device=q_fp8.device,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kvcache_fp8,
        weight,
        logits,
        _sl.to(torch.int32),
        page_table.to(torch.int32),
        max_seq_len,
        KVBlockSize=kv_block_size,
        Preshuffle=aiter_can_use_preshuffle_paged_mqa(),
    )
    return logits


def fp8_paged_mqa_logits_torch_sm120(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """CUDA-graph-compatible FP8 paged MQA logits for SM120 (vectorized, no .item())."""
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    _QUERY_CHUNK = 1024
    if batch_size > _QUERY_CHUNK:
        return torch.cat(
            [
                fp8_paged_mqa_logits_torch_sm120(
                    q_fp8[start : start + _QUERY_CHUNK],
                    kvcache_fp8,
                    weight[start : start + _QUERY_CHUNK],
                    seq_lens[start : start + _QUERY_CHUNK],
                    page_table[start : start + _QUERY_CHUNK],
                    deep_gemm_metadata,
                    max_seq_len,
                    clean_logits=clean_logits,
                )
                for start in range(0, batch_size, _QUERY_CHUNK)
            ],
            dim=0,
        )

    assert head_dim == 128, "Vectorized torch impl hardcodes DSV4 indexer head_dim=128"
    assert (
        block_size == 64
    ), "Vectorized torch impl hardcodes block_size=64 cache layout"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    if seq_lens.dim() > 1:
        seq_lens = seq_lens.squeeze(-1)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits == False

    max_pages = (max_seq_len + block_size - 1) // block_size
    max_padded_seq = max_pages * block_size

    kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))
    SCALE_OFFSET = block_size * head_dim

    page_ids = page_table[:, :max_pages]
    kvcache_gathered = kvcache_flat[page_ids]

    kv_value_raw = kvcache_gathered[..., :SCALE_OFFSET]
    kv_scale_raw = kvcache_gathered[..., SCALE_OFFSET:]

    kv_value = kv_value_raw.contiguous().view(dtype=FP8_DTYPE).to(torch.bfloat16)
    kv_value = kv_value.view(batch_size, max_padded_seq, head_dim)

    kv_scale = kv_scale_raw.contiguous().view(dtype=torch.float32)
    kv_scale = kv_scale.view(batch_size, max_padded_seq)

    q = q_fp8[:, 0].to(torch.bfloat16)

    score = torch.bmm(kv_value, q.transpose(1, 2))

    score = F.relu(score)
    score = score * weight.unsqueeze(1)
    score = score.sum(dim=2)

    score = score * kv_scale

    out_width = min(max_padded_seq, max_seq_len)
    logits = score.new_full((batch_size, max_seq_len), float("-inf"))
    logits[:, :out_width] = score[:, :out_width]

    positions = torch.arange(max_seq_len, device=device)
    invalid_mask = positions.unsqueeze(0) >= seq_lens.unsqueeze(1)
    logits.masked_fill_(invalid_mask, float("-inf"))

    return logits


def _topk_transform_512_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
    topk_op: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = torch.topk,
    topk_op_kwargs: Optional[Dict[str, object]] = None,
    contiguous_topk_input: bool = False,
) -> None:
    TOPK = out_page_indices.shape[1]
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    cache = _arange_cache
    key_seq = f"arange_{max_seq_len}_{device}"
    key_topk = f"arange_{TOPK}_{device}"
    key_bs = f"arange_{batch_size}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(max_seq_len, device=device)
    if key_topk not in cache:
        cache[key_topk] = torch.arange(TOPK, device=device, dtype=torch.int32)
    if key_bs not in cache:
        cache[key_bs] = torch.arange(batch_size, device=device)

    positions = cache[key_seq].unsqueeze(0).expand(batch_size, -1)
    valid_mask = positions < seq_lens.unsqueeze(1)

    masked_scores = scores.clone()
    masked_scores.masked_fill_(~valid_mask, float("-inf"))

    actual_k = min(TOPK, max_seq_len)
    topk_kwargs = (
        {"dim": 1, "largest": True, "sorted": False}
        if topk_op_kwargs is None
        else topk_op_kwargs
    )
    topk_input = masked_scores.contiguous() if contiguous_topk_input else masked_scores
    _, raw_indices = topk_op(topk_input, actual_k, **topk_kwargs)
    raw_indices = raw_indices.to(torch.int32)

    if actual_k < TOPK:
        raw_indices = F.pad(raw_indices, (0, TOPK - actual_k), value=0)

    batch_indices = cache[key_bs].unsqueeze(1).expand(-1, TOPK)
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = cache[key_topk].unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    needs_sequential = seq_lens <= TOPK
    sequential_indices = cache[key_topk].unsqueeze(0).expand(batch_size, -1)
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

    seq_indices_or_neg1 = sequential_indices.clone()
    seq_indices_or_neg1.masked_fill_(~sequential_valid, -1)

    needs_seq_mask = needs_sequential.unsqueeze(1).expand(-1, TOPK)
    raw_indices = torch.where(needs_seq_mask, seq_indices_or_neg1, raw_indices)
    valid_topk = torch.where(needs_seq_mask, sequential_valid, valid_topk)

    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)
    page_indices.masked_fill_(~valid_topk, -1)

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = raw_indices.clone()
        raw_indices.masked_fill_(~valid_topk, -1)
        out_raw_indices.copy_(raw_indices)


def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Vectorized PyTorch fallback for topk_transform_512.
    All helper tensors (arange, zeros) are cached to avoid device-tensor
    creation during HIP/CUDA graph capture."""

    _topk_transform_512_vectorized(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
        topk_op=torch.topk,
        topk_op_kwargs={"dim": 1, "largest": True, "sorted": False},
    )


def topk_transform_512_flashinfer_unfused(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    import flashinfer

    from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
        _flashinfer_tie_break_value,
    )

    _topk_transform_512_vectorized(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
        topk_op=flashinfer.top_k,
        topk_op_kwargs={
            "sorted": False,
            "deterministic": envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get(),
            "tie_break": _flashinfer_tie_break_value(),
            "dsa_graph_safe": True,
        },
        contiguous_topk_input=True,
    )


# Eager ragged C4 indexer (SGLANG_DSV4_INDEXER_EAGER_RAGGED=1). Under the breakable
# prefill graph, the indexer logits + top-k for the real rows of an EXTEND/MIXED step
# run inside the attention eager break instead of on every padded row inside the
# captured segment. All prefill sequences of the step share one ragged non-paged
# launch: their gathered K is concatenated (one batched gather), each row's DeepGEMM
# range is [off_s, off_s + c4len) and the kernel stores compressed logits (column 0 =
# ks[row]), so one `deep_gemm.fp8_mqa_logits` + one v2 top-k launch cover every
# prefill row. The verify rows [mixed_t, real) keep the paged kernel over their real
# rows only. The indexer compressor (K cache write) stays captured. Unfilled logits
# columns are not masked: the top-k never reads past each row's length.


class RaggedIndexerPlan:
    """One ragged non-paged launch for rows [0, rows) of the step."""

    __slots__ = (
        "rows",  # int: prefill rows covered (a contiguous prefix of the chunk)
        "gather_lens",  # int32 [n_seq] device: c4len gathered per sequence
        "gather_pages",  # int32 [n_seq, max_pages] device: one page-table row per sequence
        "seq_len_sum",  # int: total gathered K rows
        "max_seq_len",  # int: max gathered c4len
        "ks",
        "ke",  # int32 [rows] device: DeepGEMM ranges (empty when c4len <= topk)
        "lens",  # int32 [rows] device: per-row c4len (top-k seq_lens)
        "max_seqlen_k",  # int: logits width (max c4len rounded to the c4 page)
        "topk_meta",  # v2 top-k plan over lens
    )


class PagedIndexerRange:
    __slots__ = ("a", "b", "meta")


class RaggedIndexerStepPlan:
    __slots__ = ("real", "mixed_t", "ragged", "paged", "paged_ranges")


def build_ragged_indexer_plan(
    *,
    ext_lens,  # per-sequence extend rows (list[int]), sequences ordered as their rows
    seq_lens,  # per-sequence total seq_len (list[int])
    c4_seq_lens_all,  # int tensor [rows(, 1)] device: per-row c4 length
    page_table_all,  # int32 [rows, max_pages] device
    max_c4_seq_len: int,
    c4_page_size: int,
    topk: int,
    mixed_t: int,
    real: int,
) -> RaggedIndexerStepPlan:
    """Pure plan builder (testable without a ForwardBatch)."""
    plan = RaggedIndexerStepPlan()
    plan.real = real
    plan.mixed_t = mixed_t
    plan.ragged = None
    plan.paged = []
    c4_all = c4_seq_lens_all.reshape(-1)
    seq_rows, seq_c4, start = [], [], 0
    covered = 0
    for s, n in enumerate(ext_lens):
        n = int(n)
        a, b = start, start + n
        start = b
        if n <= 0:
            continue
        # Bound the gathered-K allocation by the page-table width (what the paged
        # path is bounded by too): a capture-time dummy batch may carry an
        # arbitrary seq_len fill value.
        final_c4 = min(int(seq_lens[s]) // 4, int(max_c4_seq_len))
        if final_c4 <= 0 or a != covered or b > mixed_t:
            break  # keep the ragged set a contiguous prefix; the rest goes paged
        seq_rows.append((a, b))
        seq_c4.append(final_c4)
        covered = b
    if seq_rows:
        dev = c4_all.device
        rg = RaggedIndexerPlan()
        rg.rows = covered
        lens = c4_all[:covered].to(torch.int32)
        # per-row bound: never read beyond what is gathered for the row's sequence
        bound = torch.empty(covered, dtype=torch.int32, device=dev)
        offs = torch.empty(covered, dtype=torch.int32, device=dev)
        off = 0
        for (a, b), c4 in zip(seq_rows, seq_c4):
            bound[a:b] = c4
            offs[a:b] = off
            off += c4
        lens = torch.minimum(lens, bound).clamp_(min=0)
        rg.lens = lens.contiguous()
        rg.gather_lens = torch.tensor(seq_c4, dtype=torch.int32, device=dev)
        rg.gather_pages = (
            page_table_all[[a for a, _ in seq_rows]].to(torch.int32).contiguous()
        )
        rg.seq_len_sum = off
        rg.max_seq_len = max(seq_c4)
        # SGL top-k synthesizes sequential indices for rows with <= topk
        # candidates without reading logits: give DeepGEMM an empty range.
        nontrivial = lens > topk
        rg.ks = offs.contiguous()
        rg.ke = torch.where(nontrivial, offs + lens, offs).contiguous()
        rg.max_seqlen_k = (
            (max(seq_c4) + c4_page_size - 1) // c4_page_size * c4_page_size
        )
        rg.topk_meta = plan_topk_v2(rg.lens)
        plan.ragged = rg
    paged_ranges = []
    if covered < mixed_t:
        paged_ranges.append((covered, mixed_t))
    if real > mixed_t:
        paged_ranges.append((mixed_t, real))
    plan.paged_ranges = paged_ranges
    return plan


def ragged_indexer_logits(*, q_indexer, weights, gather, plan: RaggedIndexerPlan):
    """gather(seq_len_tensor, page_indices, seq_len_sum, max_seq_len) -> (k_u8, scale_u8)."""
    import deep_gemm

    k_u8, scale_u8 = gather(
        plan.gather_lens, plan.gather_pages, plan.seq_len_sum, plan.max_seq_len
    )
    return deep_gemm.fp8_mqa_logits(
        q_indexer[: plan.rows],
        (k_u8.view(FP8_DTYPE), scale_u8.view(torch.float32).squeeze(-1)),
        weights[: plan.rows],
        plan.ks,
        plan.ke,
        clean_logits=False,
        max_seqlen_k=plan.max_seqlen_k,
    )


class _RaggedIndexerMixin:
    """Mixed into C4IndexerBackendMixin below (kept separate for readability)."""

    def eager_ragged_indexer_supported(
        self, c4_indexer: C4Indexer, forward_batch: ForwardBatch
    ) -> bool:
        if not envs.SGLANG_DSV4_INDEXER_EAGER_RAGGED.get():
            return False
        if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
            return False
        if c4_indexer.use_fp4_indexer:
            return False
        if (
            not is_cuda()
            or is_hip()
            or is_xpu()
            or envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
            or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ):
            return False
        if self.hisparse_coordinator is not None:
            return False
        if self.debug_use_external_c4_sparse_indices:
            return False
        # The verify rows' paged top-k reads the v2 plan PagedIndexerMetadata
        # builds only when top-k v2 is enabled.
        if not self.dsa_topk_backend.should_use_topk_v2():
            return False
        if get_parallel().attn_cp_size != 1:
            return False
        if get_global_indexer_capturer() is not None:
            return False
        return True

    def _ragged_step_plan(
        self,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
        c4_indexer: C4Indexer,
    ) -> RaggedIndexerStepPlan:
        plan = forward_batch.ragged_indexer_plan
        if plan is not None:
            return plan
        real = int(forward_batch.num_token_non_padded_cpu)
        mixed_t = forward_batch.mixed_num_prefill_tokens
        n_pre = forward_batch.mixed_num_prefill_rows
        if mixed_t is None:
            mixed_t = real
            n_pre = int(forward_batch.batch_size)
        mixed_t = int(mixed_t)
        n_pre = int(n_pre)
        ext = [int(v) for v in list(forward_batch.extend_seq_lens_cpu)[:n_pre]]
        seq_lens_cpu = [int(v) for v in forward_batch.seq_lens_cpu[:n_pre].tolist()]
        plan = build_ragged_indexer_plan(
            ext_lens=ext,
            seq_lens=seq_lens_cpu,
            c4_seq_lens_all=indexer_metadata.c4_seq_lens,
            page_table_all=indexer_metadata.page_table,
            max_c4_seq_len=int(indexer_metadata.max_c4_seq_len),
            c4_page_size=indexer_metadata.c4_page_size,
            topk=c4_indexer.index_topk,
            mixed_t=mixed_t,
            real=real,
        )
        for a, b in plan.paged_ranges:
            rng = PagedIndexerRange()
            rng.a, rng.b = a, b
            rng.meta = PagedIndexerMetadata(
                page_size=indexer_metadata.page_size,
                page_table=indexer_metadata.page_table[a:b],
                c4_seq_lens=indexer_metadata.c4_seq_lens[a:b],
                force_deep_gemm_metadata=indexer_metadata.force_deep_gemm_metadata,
                use_prefill_cuda_graph=False,
            )
            plan.paged.append(rng)
        forward_batch.ragged_indexer_plan = plan
        return plan

    def forward_c4_indexer_eager(
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
    ) -> None:
        """Indexer logits + top-k on the real rows (eager). The indexer
        compressor (K cache write) already ran inside the captured segment."""
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool
        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata
        assert isinstance(indexer_metadata, PagedIndexerMetadata)
        plan = self._ragged_step_plan(forward_batch, indexer_metadata, c4_indexer)
        real = plan.real
        if real == 0:
            return
        positions = core_metadata.positions[:real]
        x = x[:real]
        q_lora = q_lora[:real]
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        q_indexer, weights = c4_indexer.compute_q(q_lora, positions, weights)
        assert len(q_indexer.shape) == 3 and len(weights.shape) == 3
        weights = weights.squeeze(2)
        sparse_idx = core_metadata.c4_sparse_page_indices
        raw_all = core_metadata.c4_sparse_raw_indices
        c4_page_size = indexer_metadata.c4_page_size

        rg = plan.ragged
        if rg is not None:
            layer_id = c4_indexer.layer_id

            def gather(seq_len_tensor, page_indices, seq_len_sum, max_seq_len):
                return token_to_kv_pool.get_index_k_scale_buffer(
                    layer_id=layer_id,
                    seq_len_tensor=seq_len_tensor,
                    page_indices=page_indices,
                    seq_len_sum=seq_len_sum,
                    max_seq_len=max_seq_len,
                )

            logits = ragged_indexer_logits(
                q_indexer=q_indexer, weights=weights, gather=gather, plan=rg
            )
            n = rg.rows
            topk_transform_512_v2(
                logits,
                rg.lens,
                indexer_metadata.page_table[:n],
                sparse_idx[:n],
                c4_page_size,
                rg.topk_meta,
                raw_all[:n] if raw_all is not None else None,
            )
        if plan.paged:
            from deep_gemm import fp8_paged_mqa_logits

            cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id
            )
            assert cache.dim() == 2
            cache = cache.view(cache.shape[0], 64, 1, 132)
            q = q_indexer.unsqueeze(1)
            for rng in plan.paged:
                a, b = rng.a, rng.b
                meta = rng.meta
                c4sl = meta.c4_seq_lens
                if c4sl.dim() == 1:
                    c4sl = c4sl.unsqueeze(-1)
                logits = fp8_paged_mqa_logits(
                    q[a:b],
                    cache,
                    weights[a:b],
                    c4sl,
                    meta.page_table,
                    meta.deep_gemm_metadata,
                    meta.max_c4_seq_len,
                    False,
                )
                topk_transform_512_v2(
                    logits,
                    meta.c4_seq_lens.reshape(-1).to(torch.int32).contiguous(),
                    meta.page_table,
                    sparse_idx[a:b],
                    c4_page_size,
                    meta.topk_metadata,
                    raw_all[a:b] if raw_all is not None else None,
                )


class C4IndexerBackendMixin(_RaggedIndexerMixin):
    def __init__(self):
        super().__init__()
        self.debug_use_external_c4_sparse_indices: bool = False
        self.dsa_topk_backend: DSATopKBackend = DSATopKBackend.SGL_KERNEL

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        self.forward_indexer_compressor(
            x=x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )

        # The weight projection is small and fast; compute it on its own
        # stream, then have the Q stream wait on it before launching the big
        # fused Q kernel (which folds rope, hadamard, quantization, and
        # weight scaling into one pass).
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            weights_ready = stream_weights.record_event()

        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            stream_q.wait_event(weights_ready)
            q, weights = c4_indexer.compute_q(q_lora, positions, weights)

        current_stream.wait_stream(stream_q)
        return q, weights

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_compressor: bool = False,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackendMixin)

        weights = c4_indexer.compute_weights(x, skip_scale=True)
        q, weights = c4_indexer.compute_q(q_lora, positions, weights)
        if not skip_compressor:
            self.forward_indexer_compressor(
                x=x,
                forward_batch=forward_batch,
                layer_id=c4_indexer.layer_id,
                compressor=c4_indexer.compressor,
            )
        return q, weights

    def _can_use_nonpaged_indexer(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
    ) -> bool:
        if not envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER.get():
            return False
        # This path calls CUDA DeepGEMM and assumes the CUDA FP8+FP32 packed
        # indexer cache layout. Explicitly reject HIP, NPU, and other devices.
        if not is_cuda() or is_hip():
            return False
        # The gather plan is built from eager, child-local ForwardBatch metadata.
        # Rewritten, TBO-split, and graph-backed batches must use the paged path.
        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or forward_batch._original_forward_mode is not None
            or forward_batch.tbo_parent_token_range is not None
            or forward_batch.batch_size != 1
            or indexer_metadata.use_prefill_cuda_graph
        ):
            return False
        if (
            c4_indexer.use_fp4_indexer
            or envs.SGLANG_OPT_USE_TILELANG_INDEXER.get()
            or envs.SGLANG_OPT_USE_AITER_INDEXER.get()
            or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()
        ):
            return False
        if (
            get_parallel().attn_cp_size != 1
            or self.hisparse_coordinator is not None
            or is_in_tc_piecewise_cuda_graph()
            or is_in_breakable_cuda_graph()
        ):
            return False
        return not torch.cuda.is_current_stream_capturing()

    def _get_nonpaged_indexer_plan(
        self,
        *,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        indexer_metadata: PagedIndexerMetadata,
        page_table: torch.Tensor,
        c4_seq_lens: torch.Tensor,
        query_rows: int,
    ) -> Optional[NonPagedIndexerPlan]:
        if query_rows < envs.SGLANG_OPT_DSV4_NONPAGED_INDEXER_MIN_QUERY_TOKENS.get():
            return None
        if not self._can_use_nonpaged_indexer(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
        ):
            return None
        if indexer_metadata.nonpaged_plan is not None:
            return indexer_metadata.nonpaged_plan

        if (
            forward_batch.seq_lens is None
            or forward_batch.seq_lens_cpu is None
            or forward_batch.extend_seq_lens_cpu is None
            or forward_batch.extend_seq_lens is None
            or forward_batch.extend_start_loc is None
            or forward_batch.extend_num_tokens is None
        ):
            return None

        def to_cpu_int_list(values) -> Optional[List[int]]:
            if isinstance(values, torch.Tensor):
                if values.device.type != "cpu":
                    return None
                values = values.tolist()
            return [int(value) for value in values]

        extend_lens_cpu = to_cpu_int_list(forward_batch.extend_seq_lens_cpu)
        seq_lens_cpu = to_cpu_int_list(forward_batch.seq_lens_cpu)
        if (
            extend_lens_cpu is None
            or seq_lens_cpu is None
            or len(extend_lens_cpu) != 1
            or len(seq_lens_cpu) != 1
            or extend_lens_cpu[0] <= 0
        ):
            return None

        actual_queries = extend_lens_cpu[0]
        if (
            actual_queries != query_rows
            or int(forward_batch.extend_num_tokens) != query_rows
            or forward_batch.seq_lens.numel() != 1
            or forward_batch.extend_seq_lens.numel() != 1
            or forward_batch.extend_start_loc.numel() != 1
            or page_table.dim() != 2
            or page_table.shape[0] < query_rows
            or c4_seq_lens.numel() < query_rows
        ):
            return None

        final_c4_len = seq_lens_cpu[0] // 4
        if final_c4_len <= 0:
            return None

        request_page_table = page_table[:1].contiguous()
        ke = c4_seq_lens[:query_rows].reshape(-1).to(torch.int32).contiguous()
        gather_seq_lens = ke[-1:]
        ks = torch.zeros_like(ke)
        # SGL Top-K synthesizes sequential indices for trivial rows without
        # reading logits, so DeepGEMM can receive an empty range for them.
        if self.dsa_topk_backend.is_sgl_kernel():
            ke = torch.where(ke - ks > c4_indexer.index_topk, ke, ks)
        c4_page_size = indexer_metadata.c4_page_size
        max_seqlen_k = (final_c4_len + c4_page_size - 1) // c4_page_size * c4_page_size
        plan = NonPagedIndexerPlan(
            page_table=request_page_table,
            gather_seq_lens=gather_seq_lens,
            ks=ks,
            ke=ke,
            seq_len_sum=final_c4_len,
            max_seq_len=final_c4_len,
            max_seqlen_k=max_seqlen_k,
            query_rows=query_rows,
        )
        indexer_metadata.nonpaged_plan = plan
        return plan

    @staticmethod
    def _forward_nonpaged_indexer(
        *,
        q_indexer: torch.Tensor,
        weights: torch.Tensor,
        c4_indexer: C4Indexer,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        plan: NonPagedIndexerPlan,
    ) -> torch.Tensor:
        import deep_gemm

        k_u8, scale_u8 = token_to_kv_pool.get_index_k_scale_buffer(
            layer_id=c4_indexer.layer_id,
            seq_len_tensor=plan.gather_seq_lens,
            page_indices=plan.page_table,
            seq_len_sum=plan.seq_len_sum,
            max_seq_len=plan.max_seq_len,
        )
        k_fp8 = k_u8.view(FP8_DTYPE)
        k_scale = scale_u8.view(torch.float32).squeeze(-1)
        return deep_gemm.fp8_mqa_logits(
            q_indexer[: plan.query_rows],
            (k_fp8, k_scale),
            weights[: plan.query_rows],
            plan.ks,
            plan.ke,
            clean_logits=False,
            max_seqlen_k=plan.max_seqlen_k,
        )

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
        skip_compressor: bool = False,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = self.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackendMixin)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        positions = core_metadata.positions
        num_queries = min(x.shape[0], q_lora.shape[0], positions.shape[0])
        if x.shape[0] != num_queries:
            x = x[:num_queries]
        if q_lora.shape[0] != num_queries:
            q_lora = q_lora[:num_queries]
        if positions.shape[0] != num_queries:
            positions = positions[:num_queries]

        if enable_multi_stream:
            q_indexer, weights = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_indexer, weights = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=positions,
                forward_batch=forward_batch,
                skip_compressor=skip_compressor,
            )

        use_fp4_indexer = c4_indexer.use_fp4_indexer

        if use_fp4_indexer:
            q_fp4, q_sf = q_indexer
            assert len(q_fp4.shape) == 3
            assert len(q_sf.shape) == 2
            q = (q_fp4.unsqueeze(1), q_sf.unsqueeze(1))
        else:
            assert len(q_indexer.shape) == 3
            q = q_indexer.unsqueeze(1)

        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        if use_fp4_indexer:
            weights = weights.float()
            if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
                raise RuntimeError("DeepSeek V4 FP4 indexer requires DeepGEMM indexer.")
            from deep_gemm import fp8_fp4_paged_mqa_logits as fn
        elif envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_OPT_USE_AITER_INDEXER.get():
            fn = _aiter_fp8_paged_mqa_logits
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            if is_sm120_supported():
                fn = fp8_paged_mqa_logits_torch_sm120
            else:
                fn = fp8_paged_mqa_logits_torch
        elif is_xpu():
            from sgl_kernel import fp8_paged_mqa_logits_triton

            # TODO: switch from triton to SYCL when OOM is resolved

            fn = fp8_paged_mqa_logits_triton
        else:
            from deep_gemm import fp8_paged_mqa_logits as fn

        query_rows = q_indexer[0].shape[0] if use_fp4_indexer else q_indexer.shape[0]

        def match_num_queries(tensor: torch.Tensor, value: int) -> torch.Tensor:
            if tensor.shape[0] == query_rows:
                return tensor
            if tensor.shape[0] > query_rows:
                return tensor[:query_rows]
            pad = (0, 0) * (tensor.dim() - 1) + (0, query_rows - tensor.shape[0])
            return F.pad(tensor, pad, value=value)

        c4_seq_lens = match_num_queries(indexer_metadata.c4_seq_lens, value=1)
        _c4sl = c4_seq_lens
        page_table = match_num_queries(indexer_metadata.page_table, value=0)
        c4_sparse_page_indices = match_num_queries(
            core_metadata.c4_sparse_page_indices, value=-1
        )
        _use_tilelang = (
            envs.SGLANG_OPT_USE_TILELANG_INDEXER.get() and not use_fp4_indexer
        )
        _use_aiter = envs.SGLANG_OPT_USE_AITER_INDEXER.get() and not use_fp4_indexer
        if _c4sl.dim() == 1 and not _use_tilelang and not _use_aiter:
            _c4sl = _c4sl.unsqueeze(-1)
        nonpaged_plan = self._get_nonpaged_indexer_plan(
            c4_indexer=c4_indexer,
            forward_batch=forward_batch,
            indexer_metadata=indexer_metadata,
            page_table=page_table,
            c4_seq_lens=c4_seq_lens,
            query_rows=query_rows,
        )
        if nonpaged_plan is not None:
            assert isinstance(q_indexer, torch.Tensor)
            logits = self._forward_nonpaged_indexer(
                q_indexer=q_indexer,
                weights=weights,
                c4_indexer=c4_indexer,
                token_to_kv_pool=token_to_kv_pool,
                plan=nonpaged_plan,
            )
        else:
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )
            assert c4_indexer_kv_cache.dim() == 2
            head_dim_with_sf = 68 if use_fp4_indexer else 132
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0], 64, 1, head_dim_with_sf
            )
            logits = fn(
                q,
                c4_indexer_kv_cache,
                weights,
                _c4sl,
                page_table,
                indexer_metadata.deep_gemm_metadata,
                indexer_metadata.max_c4_seq_len,
                False,
            )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer is not None

        hisparse_coordinator = self.hisparse_coordinator
        hisparse_decode = (
            hisparse_coordinator is not None and forward_batch.forward_mode.is_decode()
        )

        raw_indices = None
        if capture_enabled:
            raw_indices = torch.empty_like(c4_sparse_page_indices)
        elif hisparse_decode:
            raw_indices = hisparse_coordinator.raw_indices_buffer[
                : c4_sparse_page_indices.size(0)
            ]
        elif core_metadata.c4_sparse_raw_indices is not None:
            raw_indices = core_metadata.c4_sparse_raw_indices

        if self.dsa_topk_backend.is_torch():
            topk_transform_512_pytorch_vectorized(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif self.dsa_topk_backend.is_flashinfer():
            topk_transform_512_flashinfer_unfused(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        elif envs.SGLANG_OPT_USE_TOPK_V2.get() and raw_indices is None:
            topk_transform_512_v2(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                indexer_metadata.topk_metadata,
            )
        else:
            topk_transform_512(
                logits,
                c4_seq_lens,
                page_table,
                c4_sparse_page_indices,
                indexer_metadata.c4_page_size,
                raw_indices,
            )
        if hisparse_coordinator is not None:
            if hisparse_decode:
                compress_layer_id = token_to_kv_pool.layer_mapping[
                    c4_indexer.layer_id
                ].compress_layer_id
                core_metadata.c4_sparse_page_indices = (
                    hisparse_coordinator.swap_in_selected_pages(
                        req_pool_indices=forward_batch.req_pool_indices,
                        compressed_seq_lens=indexer_metadata.c4_seq_lens,
                        top_k_result=raw_indices,
                        layer_id=compress_layer_id,
                    )
                )
            else:
                # flash_mla C4 attention requires int32 page indices.
                core_metadata.c4_sparse_page_indices = (
                    token_to_kv_pool.c4_kv_pool.translate_loc_to_hisparse_device(
                        core_metadata.c4_sparse_page_indices
                    ).to(torch.int32)
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        freqs_cis: torch.Tensor,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        rotary_emb=None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
            rotary_emb=rotary_emb,
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5

        self.use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        self.alt_streams = alt_streams

    def compute_q(
        self,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[IndexerQuery, torch.Tensor]:
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if self.use_fp4_indexer:
            return fused_q_indexer_rope_hadamard_fp4_quant(
                q.contiguous(), weight, self.weight_scale, self.freqs_cis, positions
            )
        return fused_q_indexer_rope_hadamard_quant(
            q, weight, self.weight_scale, self.freqs_cis, positions
        )

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
        skip_compressor: bool = False,
    ) -> None:
        return attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
            skip_compressor=skip_compressor,
        )
