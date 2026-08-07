"""Student-t prototype community head (QHP-03 / QHP-04)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .types import ModelConfig


class StudentTCommunityHead(nn.Module):
    """Soft assignment ``Q`` via Student-t kernel over trainable prototypes ``μ``."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self.config = cfg
        self.alpha = float(cfg.alpha)
        self.prototypes = nn.Parameter(torch.empty(cfg.k_communities, cfg.d_h))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self._initialized_from_kmeans = False

    @property
    def mu(self) -> torch.Tensor:
        return self.prototypes

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            s: ``[N, d_h]`` temporal states
        Returns:
            ``Q`` ``[N, K]`` and aux with hard labels, confidence, entropy
        """
        # pairwise squared distances [N, K]
        # ‖s-μ‖² = ‖s‖² + ‖μ‖² - 2 s·μ
        s_norm = (s * s).sum(dim=-1, keepdim=True)  # [N,1]
        mu_norm = (self.prototypes * self.prototypes).sum(dim=-1).unsqueeze(0)  # [1,K]
        dist2 = (s_norm + mu_norm - 2.0 * s @ self.prototypes.t()).clamp(min=0.0)
        alpha = self.alpha
        logits = -((alpha + 1.0) / 2.0) * torch.log1p(dist2 / alpha)
        q = torch.softmax(logits, dim=-1)
        hard = torch.argmax(q, dim=-1)
        conf = q.max(dim=-1).values
        # entropy in nats
        entropy = -(q * torch.log(q.clamp(min=1e-12))).sum(dim=-1)
        return q, {
            "hard": hard,
            "confidence": conf,
            "entropy": entropy,
            "dist2": dist2,
        }

    @torch.no_grad()
    def init_kmeans_plus_plus(
        self,
        states: torch.Tensor,
        active_mask: torch.Tensor,
        *,
        seed: int = 42,
        n_init: int = 20,
        max_iter: int = 50,
    ) -> Dict[str, float]:
        """Initialize prototypes with KMeans++ on active states (QHP-04).

        For smoke scaffolding this is a compact CPU/GPU implementation sufficient
        for Gate A; snapshot-balanced sampling can be layered by the trainer.
        """
        active = active_mask.to(dtype=torch.bool).reshape(-1)
        x = states[active]
        if x.shape[0] < self.config.k_communities:
            raise ValueError(
                f"need ≥K active states for KMeans++; got {x.shape[0]} < {self.config.k_communities}"
            )
        best_inertia = float("inf")
        best_centers = None
        for run in range(int(n_init)):
            g = torch.Generator(device=x.device)
            g.manual_seed(int(seed) + run)
            centers = _kmeans_plus_plus_init(x, self.config.k_communities, generator=g)
            centers, inertia = _kmeans(x, centers, max_iter=max_iter)
            if inertia < best_inertia:
                best_inertia = float(inertia)
                best_centers = centers
        assert best_centers is not None
        self.prototypes.copy_(best_centers)
        self._initialized_from_kmeans = True
        return {"inertia": best_inertia, "n_active": float(x.shape[0]), "n_init": float(n_init)}


def _kmeans_plus_plus_init(
    x: torch.Tensor,
    k: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    n = x.shape[0]
    centers = []
    idx0 = int(torch.randint(0, n, (1,), generator=generator, device=x.device).item())
    centers.append(x[idx0])
    closest = torch.full((n,), float("inf"), device=x.device, dtype=x.dtype)
    for _ in range(1, k):
        dist = ((x - centers[-1]) ** 2).sum(dim=-1)
        closest = torch.minimum(closest, dist)
        probs = closest / closest.sum().clamp(min=1e-12)
        # multinomial sampling
        idx = int(torch.multinomial(probs, 1, generator=generator).item())
        centers.append(x[idx])
    return torch.stack(centers, dim=0)


def _kmeans(
    x: torch.Tensor,
    centers: torch.Tensor,
    *,
    max_iter: int,
) -> Tuple[torch.Tensor, float]:
    k = centers.shape[0]
    for _ in range(max_iter):
        dist2 = ((x.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
        assign = dist2.argmin(dim=-1)
        new_centers = centers.clone()
        for j in range(k):
            mask = assign == j
            if mask.any():
                new_centers[j] = x[mask].mean(dim=0)
            else:
                # deterministic high-distance reinit: farthest point from current centers
                far = dist2.min(dim=-1).values.argmax()
                new_centers[j] = x[far]
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    dist2 = ((x.unsqueeze(1) - centers.unsqueeze(0)) ** 2).sum(dim=-1)
    inertia = float(dist2.min(dim=-1).values.sum().item())
    return centers, inertia


__all__ = ["StudentTCommunityHead"]
