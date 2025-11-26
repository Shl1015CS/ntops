"""
MLA (Multi-Head Latent Attention) kernel using Ninetoothed language
Implements the absorb mode: kv_cache + pe_cache strategy
"""
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


def arrangement(
    q_nope,
    q_pe,
    kv_cache,
    pe_cache,
    wkv_b_nope,
    wkv_b_value,
    mask,
    scale,
    output,
    with_mask,
    block_size_m=None,
    block_size_n=None,
):
    """Arrange tensors for MLA attention computation"""
    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M
    
    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N
    
    # Arrange query components
    def arrange_query(tensor):
        arranged = tensor.tile((1, 1, block_size_m, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged
    
    # Arrange cache
    def arrange_cache(tensor):
        arranged = tensor.tile((1, 1, block_size_n, -1))
        arranged = arranged.expand((-1, -1, q_nope_arranged.shape[-2], -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged
    
    # Arrange weight
    def arrange_weight(tensor):
        arranged = tensor.tile((1, -1, -1))
        arranged.dtype = arranged.dtype.squeeze(0)
        return arranged
    
    # Arrange mask
    def arrange_mask(tensor):
        arranged = tensor.tile((1, 1, block_size_m, block_size_n))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))
        return arranged
    
    q_nope_arranged = arrange_query(q_nope)
    q_pe_arranged = arrange_query(q_pe)
    kv_cache_arranged = arrange_cache(kv_cache)
    pe_cache_arranged = arrange_cache(pe_cache)
    wkv_b_nope_arranged = arrange_weight(wkv_b_nope)
    wkv_b_value_arranged = arrange_weight(wkv_b_value)
    mask_arranged = arrange_mask(mask)
    scale_arranged = scale
    output_arranged = arrange_query(output)
    with_mask_arranged = with_mask
    
    return (
        q_nope_arranged,
        q_pe_arranged,
        kv_cache_arranged,
        pe_cache_arranged,
        wkv_b_nope_arranged,
        wkv_b_value_arranged,
        mask_arranged,
        scale_arranged,
        output_arranged,
        with_mask_arranged,
    )


def application(
    q_nope,
    q_pe,
    kv_cache,
    pe_cache,
    wkv_b_nope,
    wkv_b_value,
    mask,
    scale,
    output,
    with_mask,
):
    """
    MLA attention application (absorb mode)
    
    Computes:
    1. q_nope @ wkv_b_nope -> q_compressed
    2. scores = (q_compressed @ kv_cache.T + q_pe @ pe_cache.T) * scale
    3. attn = softmax(scores + mask)
    4. output = (attn @ kv_cache) @ wkv_b_value
    """
    # Step 1: Compress q_nope using wkv_b_nope
    # q_nope: [batch, seq, heads, nope_dim]
    # wkv_b_nope: [heads, nope_dim, kv_lora_rank]
    # Result: [batch, seq, heads, kv_lora_rank]
    q_compressed = ntl.zeros((q_nope.shape[-2], wkv_b_nope.shape[-1]), dtype=ntl.float32)
    
    for h in range(q_nope.shape[0]):  # heads
        for d in range(wkv_b_nope.shape[-1]):  # kv_lora_rank
            acc = 0.0
            for k in range(q_nope.shape[-1]):  # nope_dim
                acc += q_nope[h][k] * wkv_b_nope[h][k, d]
            q_compressed[:, d] = acc
    
    # Step 2: Compute attention scores
    # scores = q_compressed @ kv_cache.T + q_pe @ pe_cache.T
    acc_scores = ntl.zeros((q_compressed.shape[-2], kv_cache.shape[-2]), dtype=ntl.float32)
    max_scores = ntl.full((q_compressed.shape[-2],), float("-inf"), dtype=ntl.float32)
    
    for i in range(kv_cache.shape[0]):  # cache sequence length
        # q_compressed @ kv_cache[i]
        score_kv = ntl.dot(q_compressed, ntl.trans(kv_cache[i]))
        
        # q_pe @ pe_cache[i]
        score_pe = ntl.dot(q_pe, ntl.trans(pe_cache[i]))
        
        # Combine and scale
        score = (score_kv + score_pe) * scale
        
        # Add mask if needed
        if with_mask:
            score += mask[i]
        
        # Track max for numerical stability
        max_scores = ntl.maximum(max_scores, ntl.max(score, 1))
        acc_scores[:, i] = score
    
    # Step 3: Softmax
    # Stable softmax: exp(x - max) / sum(exp(x - max))
    exp_scores = ntl.zeros_like(acc_scores)
    sum_exp = ntl.zeros((acc_scores.shape[-2],), dtype=ntl.float32)
    
    for i in range(acc_scores.shape[-1]):
        exp_val = ntl.exp(acc_scores[:, i] - max_scores)
        exp_scores[:, i] = exp_val
        sum_exp += exp_val
    
    # Normalize
    attn_weights = ntl.zeros_like(exp_scores)
    for i in range(exp_scores.shape[-1]):
        attn_weights[:, i] = exp_scores[:, i] / sum_exp
    
    # Step 4: Apply attention to get weighted sum
    # attn_weights @ kv_cache -> [batch, seq, kv_lora_rank]
    weighted_kv = ntl.zeros((attn_weights.shape[-2], kv_cache.shape[-1]), dtype=ntl.float32)
    
    for i in range(kv_cache.shape[0]):
        for j in range(kv_cache.shape[-1]):
            weighted_kv[:, j] += attn_weights[:, i] * kv_cache[i][:, j]
    
    # Step 5: Project to output using wkv_b_value
    # weighted_kv @ wkv_b_value -> [batch, seq, heads, v_head_dim]
    for h in range(wkv_b_value.shape[0]):  # heads
        for d in range(wkv_b_value.shape[-1]):  # v_head_dim
            acc = 0.0
            for k in range(weighted_kv.shape[-1]):  # kv_lora_rank
                acc += weighted_kv[:, k] * wkv_b_value[h][k, d]
            output[h][:, d] = acc


def premake(
    n_heads=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    kv_lora_rank=None,
    v_head_dim=None,
    seq_len=None,
    dtype=None,
    with_mask=None,
    block_size_m=None,
    block_size_n=None,
):
    """
    Premake function for MLA attention kernel
    
    Args:
        n_heads: Number of attention heads
        qk_nope_head_dim: Query/key dimension without position encoding
        qk_rope_head_dim: Query/key dimension with rotary position encoding
        kv_lora_rank: KV cache compression rank
        v_head_dim: Value head dimension
        seq_len: Sequence length
        dtype: Data type
        with_mask: Whether to use attention mask
    """
    arrangement_ = functools.partial(
        arrangement,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    
    # Define tensor shapes
    # q_nope: [batch, seq_len, n_heads, qk_nope_head_dim]
    # q_pe: [batch, seq_len, n_heads, qk_rope_head_dim]
    # kv_cache: [batch, cache_len, kv_lora_rank]
    # pe_cache: [batch, cache_len, qk_rope_head_dim]
    # wkv_b_nope: [n_heads, qk_nope_head_dim, kv_lora_rank]
    # wkv_b_value: [n_heads, kv_lora_rank, v_head_dim]
    # mask: [batch, n_heads, seq_len, cache_len]
    # output: [batch, seq_len, n_heads, v_head_dim]
    
    tensors = (
        Tensor(4, dtype=dtype),  # q_nope
        Tensor(4, dtype=dtype),  # q_pe
        Tensor(3, dtype=dtype),  # kv_cache
        Tensor(3, dtype=dtype),  # pe_cache
        Tensor(3, dtype=dtype),  # wkv_b_nope
        Tensor(3, dtype=dtype),  # wkv_b_value
        Tensor(4, dtype=dtype),  # mask
        Tensor(0, dtype=dtype),  # scale
        Tensor(4, dtype=dtype),  # output
        Tensor(0, dtype=dtype, constexpr=True, value=with_mask),  # with_mask flag
    )
    
    return arrangement_, application, tensors
