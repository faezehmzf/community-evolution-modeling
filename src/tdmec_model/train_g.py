"""TDMEC-G smoke trainer (Step 1a Gate A)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from tdmec import constants as C
from tdmec.hashing import sha256_file

from .dataset import TDMECInputDataset
from .losses import (
    LossWeights,
    cluster_kl_loss,
    combine_losses,
    prototype_separation_loss,
    sample_uniform_negatives,
    split_edges_for_struct_loss,
    structural_bce_loss,
)
from .scaling import RobustFeatureScaler
from .tdmec_g import TDMECG
from .types import ModelConfig

_STATUS = (
    "PROVISIONAL_SMOKE_ONLY",
    "ENGINEERING_VALIDATION",
    "NOT_FOR_FINAL_THESIS_CONCLUSIONS",
)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_model_config(cfg: Dict[str, Any]) -> ModelConfig:
    m = cfg.get("model") or {}
    return ModelConfig(
        d_h=int(m.get("d_h", 64)),
        k_communities=int(m.get("k_communities", 10)),
        d_rel=int(m.get("d_rel", 16)),
        alpha=float(m.get("alpha", 1.0)),
        num_layers=int(m.get("num_layers", 1)),
        fanout=tuple(m.get("fanout") or (15,)),
        f_struct=int(m.get("f_struct", C.F_STRUCT)),
        dropout=float(m.get("dropout", 0.0)),
        bptt_window=int(m.get("bptt_window", 3)),
    )


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def train_tdmec_g_smoke(
    config_path: str | Path,
    *,
    max_snapshots: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    train_cfg = cfg.get("train") or {}
    package_root = Path(cfg["package_root"])
    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    seed = int(train_cfg.get("seed", 42))
    torch.manual_seed(seed)
    dev = torch.device(device or train_cfg.get("device") or "cpu")

    model_cfg = build_model_config(cfg)
    weights = LossWeights(
        lambda_struct=float(train_cfg.get("lambda_struct", 1.0)),
        lambda_sem=float(train_cfg.get("lambda_sem", 0.0)),
        lambda_cluster=float(train_cfg.get("lambda_clu", train_cfg.get("lambda_cluster", 1.0))),
        lambda_reg=float(train_cfg.get("lambda_reg", 0.1)),
        lambda_temp=float(train_cfg.get("lambda_temp", 0.0)),
        reg_margin=float(train_cfg.get("reg_margin", 1.0)),
    )
    mask_rate = float(train_cfg.get("mask_rate", 0.15))
    num_neg = int(train_cfg.get("num_negatives", 3))
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 5e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    bptt = int(model_cfg.bptt_window)

    train_ids: List[int] = list(train_cfg.get("train_snapshot_ids") or list(range(24)))
    if max_snapshots is not None:
        train_ids = train_ids[: int(max_snapshots)]

    ds = TDMECInputDataset(package_root)
    scaler = RobustFeatureScaler().fit(
        ds._x_struct,
        ds._struct_mask,
        train_snapshot_ids=train_ids,
        time_index_for=ds._snap_to_t,
    )
    scaler.save(output_root / "scaler.json")

    model = TDMECG(model_cfg).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    history: List[Dict[str, Any]] = []
    t0 = time.time()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    last_mean_epoch = None

    for epoch in range(epochs):
        model.train()
        s_state = model.temporal.initial_state(C.N_NODES, device=dev)
        epoch_loss = 0.0
        n_windows = 0
        # BPTT windows over chronological train ids
        for start in range(0, len(train_ids), bptt):
            window = train_ids[start : start + bptt]
            optim.zero_grad(set_to_none=True)
            # Detach at window boundary (QTR-03)
            s_prev = s_state.detach()
            window_loss = s_prev.new_zeros(())
            window_comps: List[Dict[str, float]] = []

            for sid in window:
                batch = ds.get_snapshot(sid)
                # move batch tensors
                x_raw = batch.x_struct
                mask = batch.struct_active_mask
                x_scaled = scaler.transform(x_raw, mask, as_torch=True)
                assert isinstance(x_scaled, torch.Tensor)
                x_scaled = x_scaled.to(dev)
                edge_index = batch.edge_index.to(dev)
                relation_id = batch.relation_id.to(dev)
                weight = batch.weight_log1p.to(dev)
                struct_mask = mask.to(dev)

                split = split_edges_for_struct_loss(
                    edge_index,
                    relation_id,
                    weight,
                    mask_rate=mask_rate,
                    generator=gen if edge_index.device.type == "cpu" else None,
                )
                # If edges on GPU, re-split without generator constraint
                if edge_index.device.type != "cpu":
                    split = split_edges_for_struct_loss(
                        edge_index, relation_id, weight, mask_rate=mask_rate
                    )

                out = model.encode_snapshot(
                    x_scaled,
                    split.enc_edge_index,
                    split.enc_relation_id,
                    split.enc_weight_log1p,
                    struct_mask,
                    s_prev,
                    use_fanout=True,
                )
                s = out["s"]
                q = out["q"]

                neg_src, neg_dst, neg_rel = sample_uniform_negatives(
                    split.pos_src,
                    split.pos_relation_id,
                    edge_index,
                    relation_id,
                    num_nodes=C.N_NODES,
                    num_neg=num_neg,
                    generator=gen,
                )
                # move pos to same device (already on device from split)
                l_struct = structural_bce_loss(
                    model.decoder,
                    s,
                    split.pos_src,
                    split.pos_dst,
                    split.pos_relation_id,
                    split.pos_weight_log1p,
                    neg_src.to(dev),
                    neg_dst.to(dev),
                    neg_rel.to(dev),
                )
                # cluster only on active nodes
                active = out["model_active"]
                if active.any():
                    l_cluster, _ = cluster_kl_loss(q[active])
                else:
                    l_cluster = s.new_zeros(())
                l_reg = prototype_separation_loss(model.community.mu, margin=weights.reg_margin)
                l_sem = s.new_zeros(())
                l_temp = s.new_zeros(())
                total, comps = combine_losses(
                    l_struct=l_struct,
                    l_sem=l_sem,
                    l_cluster=l_cluster,
                    l_reg=l_reg,
                    l_temp=l_temp,
                    weights=weights,
                )
                window_loss = window_loss + total
                comps["snapshot_id"] = float(sid)
                window_comps.append(comps)
                s_prev = s

            window_loss = window_loss / max(len(window), 1)
            window_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim.step()
            s_state = s_prev.detach()
            epoch_loss += float(window_loss.detach().item())
            n_windows += 1
            history.append(
                {
                    "epoch": epoch,
                    "window_start": window[0],
                    "snapshots": window,
                    "loss": float(window_loss.detach().item()),
                    "components": window_comps,
                }
            )
            print(
                f"epoch={epoch} window={window} loss={float(window_loss.detach().item()):.6f}",
                flush=True,
            )

        last_mean_epoch = epoch_loss / max(n_windows, 1)
        print(f"epoch={epoch} mean_window_loss={last_mean_epoch:.6f}", flush=True)

    ckpt_path = output_root / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "model_config": model_cfg.__dict__,
            "seed": seed,
            "train_snapshot_ids": train_ids,
            "status_labels": list(_STATUS),
            "variant": "TDMEC-G",
        },
        ckpt_path,
    )

    # Quick reload check
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2 = TDMECG(model_cfg)
    model2.load_state_dict(ckpt["model_state_dict"])

    # AP stub (evaluator deferred)
    ap_stub = {
        "schema_version": "tdmec-ap-stub-v1",
        "variant": "TDMEC-G",
        "status": "STUB_NOT_EVALUATED",
        "relation_macro_ap": None,
        "note": "Full AP evaluator deferred to Phase 8; smoke Gate A only.",
        "status_labels": list(_STATUS),
    }
    _atomic_json(output_root / "ap_stub.json", ap_stub)

    report = {
        "schema_version": "tdmec-g-smoke-train-report-v1",
        "variant": "TDMEC-G",
        "package_root": package_root.as_posix(),
        "output_root": output_root.as_posix(),
        "device": str(dev),
        "epochs": epochs,
        "train_snapshot_ids": train_ids,
        "bptt_window": bptt,
        "seed": seed,
        "final_mean_window_loss": last_mean_epoch,
        "n_windows": len(history),
        "elapsed_sec": time.time() - t0,
        "checkpoint": ckpt_path.as_posix(),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "scaler": (output_root / "scaler.json").as_posix(),
        "reload_ok": True,
        "history_tail": history[-3:],
        "status_labels": list(_STATUS),
    }
    _atomic_json(output_root / "train_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train TDMEC-G smoke on TDMEC_INPUT_smoke_e2e")
    p.add_argument(
        "--config",
        default="configs/tdmec_g_smoke.yaml",
    )
    p.add_argument("--max-snapshots", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = train_tdmec_g_smoke(
        args.config,
        max_snapshots=args.max_snapshots,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
