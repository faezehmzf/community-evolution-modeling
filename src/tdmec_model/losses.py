"""TDMEC losses for smoke / TDMEC-G (struct, cluster, reg; sem/temp optional)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import StructuralDecoder


@dataclass
class LossWeights:
    lambda_struct: float = 1.0
    lambda_sem: float = 0.0
    lambda_cluster: float = 1.0
    lambda_reg: float = 0.1
    lambda_temp: float = 0.0
    reg_margin: float = 1.0


@dataclass
class EdgeSplit:
    """Encoder edges vs masked positives for ``L_struct``."""

    enc_edge_index: torch.Tensor
    enc_relation_id: torch.Tensor
    enc_weight_log1p: torch.Tensor
    pos_src: torch.Tensor
    pos_dst: torch.Tensor
    pos_relation_id: torch.Tensor
    pos_weight_log1p: torch.Tensor
    enc_edge_text: Optional[torch.Tensor] = None
    enc_edge_text_mask: Optional[torch.Tensor] = None
    pos_edge_text: Optional[torch.Tensor] = None
    pos_edge_text_mask: Optional[torch.Tensor] = None


def split_edges_for_struct_loss(
    edge_index: torch.Tensor,
    relation_id: torch.Tensor,
    weight_log1p: torch.Tensor,
    *,
    mask_rate: float = 0.15,
    generator: Optional[torch.Generator] = None,
    edge_text: Optional[torch.Tensor] = None,
    edge_text_available_mask: Optional[torch.Tensor] = None,
) -> EdgeSplit:
    """Mask ``mask_rate`` of observed edges; remainder stay in the encoder graph."""
    e = int(edge_index.shape[1])
    if e == 0:
        empty_e = edge_index
        empty_r = relation_id
        empty_w = weight_log1p
        empty_n = edge_index.new_empty((0,), dtype=torch.long)
        empty_t = None if edge_text is None else edge_text[:0]
        empty_m = None if edge_text_available_mask is None else edge_text_available_mask[:0]
        return EdgeSplit(
            empty_e,
            empty_r,
            empty_w,
            empty_n,
            empty_n,
            empty_n,
            empty_w,
            empty_t,
            empty_m,
            empty_t,
            empty_m,
        )

    n_mask = int(round(e * float(mask_rate)))
    n_mask = min(max(n_mask, 1 if e >= 2 else 0), e)
    perm = torch.randperm(e, device=edge_index.device, generator=generator)
    masked = perm[:n_mask]
    keep = perm[n_mask:]

    enc_edge_index = edge_index[:, keep]
    enc_relation_id = relation_id[keep]
    enc_weight = weight_log1p[keep]

    pos_src = edge_index[0, masked]
    pos_dst = edge_index[1, masked]
    pos_rel = relation_id[masked]
    pos_w = weight_log1p[masked]

    enc_et = edge_text[keep] if edge_text is not None else None
    enc_em = edge_text_available_mask[keep] if edge_text_available_mask is not None else None
    pos_et = edge_text[masked] if edge_text is not None else None
    pos_em = edge_text_available_mask[masked] if edge_text_available_mask is not None else None

    return EdgeSplit(
        enc_edge_index,
        enc_relation_id,
        enc_weight,
        pos_src,
        pos_dst,
        pos_rel,
        pos_w,
        enc_et,
        enc_em,
        pos_et,
        pos_em,
    )


def sample_uniform_negatives(
    pos_src: torch.Tensor,
    pos_rel: torch.Tensor,
    edge_index: torch.Tensor,
    relation_id: torch.Tensor,
    *,
    num_nodes: int,
    num_neg: int = 3,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``num_neg`` negatives per positive (same src & relation; not observed)."""
    device = pos_src.device
    n_pos = int(pos_src.numel())
    if n_pos == 0:
        z = pos_src.new_empty((0,), dtype=torch.long)
        return z, z, z

    # Build observed dst sets on CPU for rejection sampling (smoke-scale OK).
    src_all = edge_index[0].detach().cpu().tolist()
    dst_all = edge_index[1].detach().cpu().tolist()
    rel_all = relation_id.detach().cpu().tolist()
    observed: Dict[Tuple[int, int], set] = {}
    for s, d, r in zip(src_all, dst_all, rel_all):
        observed.setdefault((int(s), int(r)), set()).add(int(d))

    g = generator
    neg_src: list[int] = []
    neg_dst: list[int] = []
    neg_rel: list[int] = []
    pos_s = pos_src.detach().cpu().tolist()
    pos_r = pos_rel.detach().cpu().tolist()
    for s, r in zip(pos_s, pos_r):
        banned = observed.get((int(s), int(r)), set())
        banned_self = set(banned)
        banned_self.add(int(s))
        got = 0
        attempts = 0
        while got < num_neg and attempts < num_neg * 50 + 20:
            attempts += 1
            j = int(torch.randint(0, num_nodes, (1,), device=device, generator=g).item())
            if j in banned_self:
                continue
            neg_src.append(int(s))
            neg_dst.append(j)
            neg_rel.append(int(r))
            banned_self.add(j)  # without replacement within this positive
            got += 1
        # If graph nearly complete (unlikely), pad with random distinct leftovers
        while got < num_neg:
            j = (int(s) + got + 1) % num_nodes
            if j == int(s):
                j = (j + 1) % num_nodes
            neg_src.append(int(s))
            neg_dst.append(j)
            neg_rel.append(int(r))
            got += 1

    return (
        torch.tensor(neg_src, dtype=torch.long, device=device),
        torch.tensor(neg_dst, dtype=torch.long, device=device),
        torch.tensor(neg_rel, dtype=torch.long, device=device),
    )


def structural_bce_loss(
    decoder: StructuralDecoder,
    s: torch.Tensor,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    pos_rel: torch.Tensor,
    pos_weight: torch.Tensor,
    neg_src: torch.Tensor,
    neg_dst: torch.Tensor,
    neg_rel: torch.Tensor,
    *,
    pos_edge_text: Optional[torch.Tensor] = None,
    pos_edge_text_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pos_src.numel() == 0:
        return s.new_zeros(())
    pos_logits = decoder(
        s,
        pos_src,
        pos_dst,
        pos_rel,
        pos_weight,
        edge_text=pos_edge_text,
        edge_text_available_mask=pos_edge_text_mask,
    )
    neg_w = s.new_zeros((neg_src.shape[0],))
    # Negatives: no edge text (contract)
    neg_logits = decoder(s, neg_src, neg_dst, neg_rel, neg_w)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat(
        [
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, labels)


def cluster_kl_loss(q: torch.Tensor, p: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """DEC KL(P‖Q). If ``p`` is None, build target from ``q`` (batch update)."""
    if p is None:
        p = dec_target_from_q(q)
    # KL(P||Q) = Σ p log(p/q)
    loss = (p * (torch.log(p.clamp(min=1e-12)) - torch.log(q.clamp(min=1e-12)))).sum(dim=-1).mean()
    return loss, p


def dec_target_from_q(q: torch.Tensor) -> torch.Tensor:
    """``p_ik ∝ q_ik² / f_k`` with ``f_k = Σ_i q_ik``."""
    weight = q * q
    freq = weight.sum(dim=0, keepdim=True).clamp(min=1e-12)
    p = weight / freq
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return p.detach()


def prototype_separation_loss(mu: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """``1/(K(K-1)) Σ_{k≠ℓ} max(0, m - ‖μ_k−μ_ℓ‖²)``."""
    k = mu.shape[0]
    if k < 2:
        return mu.new_zeros(())
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    dist2 = (diff * diff).sum(dim=-1)
    mask = ~torch.eye(k, dtype=torch.bool, device=mu.device)
    hinge = F.relu(float(margin) - dist2[mask])
    return hinge.mean()


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean Jensen–Shannon divergence between rows of ``p`` and ``q``."""
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp(min=eps)) - torch.log(m.clamp(min=eps)))).sum(dim=-1)
    kl_qm = (q * (torch.log(q.clamp(min=eps)) - torch.log(m.clamp(min=eps)))).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean()


def combine_losses(
    *,
    l_struct: torch.Tensor,
    l_sem: torch.Tensor,
    l_cluster: torch.Tensor,
    l_reg: torch.Tensor,
    l_temp: torch.Tensor,
    weights: LossWeights,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = (
        weights.lambda_struct * l_struct
        + weights.lambda_sem * l_sem
        + weights.lambda_cluster * l_cluster
        + weights.lambda_reg * l_reg
        + weights.lambda_temp * l_temp
    )
    components = {
        "L_struct": float(l_struct.detach().item()),
        "L_sem": float(l_sem.detach().item()),
        "L_cluster": float(l_cluster.detach().item()),
        "L_reg": float(l_reg.detach().item()),
        "L_temp": float(l_temp.detach().item()),
        "L_total": float(total.detach().item()),
        "lambda_struct": weights.lambda_struct,
        "lambda_sem": weights.lambda_sem,
        "lambda_cluster": weights.lambda_cluster,
        "lambda_reg": weights.lambda_reg,
        "lambda_temp": weights.lambda_temp,
    }
    return total, components


__all__ = [
    "LossWeights",
    "EdgeSplit",
    "split_edges_for_struct_loss",
    "sample_uniform_negatives",
    "structural_bce_loss",
    "cluster_kl_loss",
    "dec_target_from_q",
    "prototype_separation_loss",
    "js_divergence",
    "combine_losses",
]
