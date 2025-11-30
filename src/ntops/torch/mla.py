"""
MLA (Multi-Head Latent Attention) operator for Ninetoothed
PyTorch frontend interface
"""
import torch
import torch.nn.functional as F

import ntops
from ntops.torch.utils import _cached_make


def mla_attention(
    q_nope,
    q_pe,
    kv_cache,
    pe_cache,
    wkv_b_nope,
    wkv_b_value,
    mask=None,
    scale=1.0,
):
    """
    MLA attention computation using absorb mode
    
    Args:
        q_nope: Query without position encoding [batch, seq, heads, nope_dim]
        q_pe: Query with position encoding [batch, seq, heads, rope_dim]
        kv_cache: Compressed KV cache [batch, cache_len, kv_lora_rank]
        pe_cache: Position encoding cache [batch, cache_len, rope_dim]
        wkv_b_nope: Weight for nope projection [heads, nope_dim, kv_lora_rank]
        wkv_b_value: Weight for value projection [heads, kv_lora_rank, v_dim]
        mask: Attention mask [batch, heads, seq, cache_len]
        scale: Softmax scaling factor
    
    Returns:
        output: Attention output [batch, seq, heads, v_dim]
    """
    # For now, use PyTorch implementation
    # Will be replaced with Ninetoothed kernel when bfloat16 support is ready
    
    bsz, seqlen, n_heads, nope_dim = q_nope.shape
    _, cache_len, kv_lora_rank = kv_cache.shape
    _, _, v_dim = wkv_b_value.shape
    
    # Compress q_nope using wkv_b_nope
    # [bsz, seqlen, n_heads, nope_dim] @ [n_heads, nope_dim, kv_lora_rank]
    # -> [bsz, seqlen, n_heads, kv_lora_rank]
    q_compressed = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_nope)
    
    # Compute attention scores
    # scores = q_compressed @ kv_cache.T + q_pe @ pe_cache.T
    scores = (
        torch.einsum("bshc,btc->bsht", q_compressed, kv_cache) +
        torch.einsum("bshr,btr->bsht", q_pe, pe_cache)
    ) * scale
    
    # Apply mask if provided
    if mask is not None:
        scores = scores + mask.unsqueeze(1)
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q_nope)
    
    # Apply attention and project to output
    # output = (attn_weights @ kv_cache) @ wkv_b_value
    # weighted_kv: [bsz, seqlen, n_heads, kv_lora_rank]
    # wkv_b_value: [heads, kv_lora_rank, v_dim]
    weighted_kv = torch.einsum("bsht,btc->bshc", attn_weights, kv_cache)
    output = torch.einsum("bshc,hcd->bshd", weighted_kv, wkv_b_value)
    
    return output


def mla_update_cache(kv, k_pe, kv_cache, pe_cache, kv_norm, start_pos, end_pos):
    """
    Update MLA cache (absorb mode)
    
    Args:
        kv: New KV projection [batch, seq, kv_lora_rank]
        k_pe: New position encoding [batch, seq, rope_dim]
        kv_cache: KV cache buffer [batch, max_seq, kv_lora_rank]
        pe_cache: PE cache buffer [batch, max_seq, rope_dim]
        kv_norm: RMS normalization layer
        start_pos: Start position in cache
        end_pos: End position in cache
    """
    bsz = kv.shape[0]
    
    # Normalize and update kv_cache
    if kv_norm is not None:
        kv_normalized = kv_norm(kv)
    else:
        kv_normalized = kv
    
    kv_cache[:bsz, start_pos:end_pos] = kv_normalized
    pe_cache[:bsz, start_pos:end_pos] = k_pe
    
    return kv_cache, pe_cache
