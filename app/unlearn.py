"""UNSC Algorithms 1 & 2: cache subspaces and execute null-space updates."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.evaluate import evaluate_splits
from app.model import FashionCNN, copy_weights, save_checkpoint
from app.utils import state_dict_digest


def energy_rank_threshold(singular_values: torch.Tensor, epsilon: float) -> int:
    """Smallest Frobenius-energy rank ``r`` attaining ``Σ_{i≤r} σ_i² / Σ σ² ≥ ε``."""
    if singular_values.numel() == 0:
        return 0
    eps = float(epsilon)
    if not (0.0 < eps < 1.0):
        return int(singular_values.numel())

    s2 = singular_values.to(torch.float64) ** 2
    denom = torch.sum(s2)
    if float(denom.item()) <= 1e-28:
        return 1

    thresh = eps * denom
    csum = torch.cumsum(s2, dim=0)
    ok = torch.nonzero(csum >= thresh, as_tuple=False)
    if ok.numel() == 0:
        return int(s2.numel())
    idx = int(ok[0, 0].item()) + 1
    return max(1, min(idx, int(s2.numel())))


def collect_layer_columns_single_class(
    model: FashionCNN,
    *,
    loader: DataLoader,
    label_index: int,
    device: torch.device,
    target_cols: int,
) -> dict[str, torch.Tensor]:
    """Stack CPU hook matrices ``ℝ^{feat×N}`` for class ``label_index``."""
    projector_names = model.projector_layer_names_ordered()
    mats: dict[str, torch.Tensor] = {}
    done = {nm: False for nm in projector_names}
    epochs_guard = 0

    model.eval()

    while not all(done.values()) and epochs_guard < 32:
        epochs_guard += 1
        for imgs, ys in loader:
            mask = ys == label_index
            if int(mask.sum()) == 0:
                continue

            xsel = imgs[mask].to(device)
            model.clear_cached_inputs()

            with torch.no_grad():

                model(xsel)

            for nm in projector_names:
                col = model._layer_flat_inputs_cpu[nm]

                mats[nm] = col.float() if nm not in mats else torch.cat([mats[nm], col.float()], dim=1)

                if mats[nm].shape[1] >= target_cols:

                    mats[nm] = mats[nm][:, :target_cols]
                    done[nm] = True

            if all(done.values()):
                break

    if not all(done.values()):
        raise RuntimeError("Algorithm 1 aborted: insufficient per-class batches")

    return mats


def algorithm1_cache_all(theta_o: FashionCNN, loader_tr: DataLoader, device: torch.device, *,
                         samples_per_cls: int, epsilon_trunc: float) -> dict[int, dict[str, torch.Tensor]]:
    """Return ``U^k_l`` dictionaries for every Fashion-MNIST class ``k``.

    Columns per layer truncated per-class using ``epsilon_trunc`` spectral rule.
    """
    cache: dict[int, dict[str, torch.Tensor]] = {}
    theta_o.eval()

    classes = theta_o.num_classes

    for k in range(classes):
        racks = collect_layer_columns_single_class(
            theta_o, loader=loader_tr, label_index=k, device=device, target_cols=max(samples_per_cls, 32))

        buckets: dict[str, torch.Tensor] = {}

        for layer_nm, mtx in racks.items():

            uh, sh, _ = torch.linalg.svd(mtx.double(), full_matrices=False)

            r = energy_rank_threshold(sh.float(), epsilon_trunc)

            sub = uh[:, :r]

            q, _qr = torch.linalg.qr(sub, mode="reduced")

            buckets[layer_nm] = q.float()

        cache[k] = buckets

    return cache


def build_layer_projectors_cpu(
    cache: dict[int, dict[str, torch.Tensor]],
    *,
    forget: int,

    projector_names: list[str],
    epsilon_merge: float,

) -> dict[str, torch.Tensor]:
    """Eq.~(8)-(11): merge retained-class bases excluding ``forget`` then orthogonal complement."""
    out: dict[str, torch.Tensor] = {}

    for nm in projector_names:

        rests = sorted([cid for cid in cache if cid != forget])

        horizontally = torch.cat([cache[cid][nm].double() for cid in rests], dim=1)

        uh, sh, _ = torch.linalg.svd(horizontally, full_matrices=False)

        r_keep = energy_rank_threshold(sh.float(), epsilon_merge)

        core = uh[:, :r_keep]

        qbasis, _ = torch.linalg.qr(core, mode="reduced")

        qf32 = qbasis.float()

        d = qf32.shape[0]

        eye = torch.eye(d, dtype=torch.float32)

        projector = eye - qf32 @ qf32.transpose(0, 1)

        proj_symm = 0.5 * (projector + projector.transpose(0, 1))

        out[nm] = torch.nan_to_num(proj_symm)

    return out


def pseudo_targets(theta_o: FashionCNN, xb: torch.Tensor, forget_cls: int, device: torch.device) -> torch.Tensor:
    """Arg-max among remaining logits (Eq.~13 heuristic)."""

    theta_o.eval()

    with torch.no_grad():

        lg = theta_o(xb.to(device)).clone()

    lg[:, forget_cls] = torch.finfo(lg.dtype).min

    return lg.argmax(dim=1)


def apply_null_projections(student: FashionCNN, proj_cpu: dict[str, torch.Tensor]) -> None:
    """Project ``∇W`` orthogonal to retained span for Conv/Linear modules."""
    for layer_key, layer_mod in student.projection_modules.items():

        g = getattr(layer_mod, "weight").grad

        if g is None:
            continue

        p_mat = proj_cpu[layer_key].to(g.device, g.dtype)

        if isinstance(layer_mod, nn.Conv2d):
            och, ich, kh, kw = layer_mod.weight.shape

            gv = g.view(och, ich * kh * kw)

            gv = gv @ p_mat

            layer_mod.weight.grad = gv.view_as(g)

        elif isinstance(layer_mod, nn.Linear):

            layer_mod.weight.grad = g @ p_mat

        else:
            raise TypeError(type(layer_mod))


def persist_subspaces(path_pt: Path, cache: dict[int, dict[str, torch.Tensor]], meta: dict[str, Any]) -> None:
    torch.save({"bases": cache, "meta": meta}, str(path_pt))

    meta_path = path_pt.with_suffix(path_pt.suffix + ".meta.json")

    meta_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def load_cached_subspaces_if_any(path_pt: Path) -> dict[int, dict[str, torch.Tensor]] | None:
    """Return caches if file exists."""

    if not path_pt.exists():
        return None

    try:
        blob = torch.load(str(path_pt), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(path_pt), map_location="cpu")
    caches = blob.get("bases")
    assert isinstance(caches, dict)
    return caches


def run_unsc(
    *,
    theta_o: FashionCNN,
    theta_u: FashionCNN,
    forget_class: int,
    train_full_loader_alg1: DataLoader,
    train_forgotten_loader: DataLoader,
    testers: dict[str, DataLoader | None],
    device: torch.device,
    outputs_folder: Path,
    epochs_ul: int,
    lr_ul: float,
    momentum_ul: float,
    samples_algo1_cls: int,
    epsilon_trunc_class: float,
    epsilon_union_layer: float,
    seed: int,
    unlearn_ckpt_fp: Path,
) -> dict[str, Any]:
    """Algorithm 2: pseudo-label forgetting with per-layer projector ``P``."""

    outputs_folder.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    copy_weights(theta_o, theta_u)

    digest = state_dict_digest(theta_o)
    cache_disk = outputs_folder / f"unsc_subspaces_{digest}.pt"
    cache_data = load_cached_subspaces_if_any(cache_disk)

    meta_save = {
        "digest": digest,
        "samples_per_cls": samples_algo1_cls,
        "epsilon_trunc_class": epsilon_trunc_class,
        "epsilon_union_layer": epsilon_union_layer,
        "forget_class_requested": forget_class,
    }

    if cache_data is None:
        cache_data = algorithm1_cache_all(
            theta_o,
            train_full_loader_alg1,
            device=device,
            samples_per_cls=samples_algo1_cls,
            epsilon_trunc=epsilon_trunc_class,
        )
        persist_subspaces(cache_disk, cache_data, meta_save)


    projector_map = build_layer_projectors_cpu(
        cache_data,
        forget=forget_class,
        projector_names=theta_o.projector_layer_names_ordered(),
        epsilon_merge=epsilon_union_layer,
    )
    torch.save(
        {"projectors": projector_map, "digest": digest, "forget_class": forget_class},
        outputs_folder / f"projection_{digest}_c{forget_class}.pt",
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        theta_u.parameters(), lr=lr_ul, momentum=momentum_ul, nesterov=True
    )
    theta_o.eval()

    histories: dict[str, list[float]] = {"epoch": [], "loss": [], "pseudo_acc": []}
    theta_u.train(True)

    for epoch in range(1, epochs_ul + 1):
        agg_loss = 0.0
        batches_seen = 0
        corr = 0
        total = 0
        for imgs, _ys in train_forgotten_loader:
            imgs = imgs.to(device, non_blocking=True)
            yt = pseudo_targets(theta_o, imgs, forget_class, device)
            optimizer.zero_grad(set_to_none=True)
            preds = theta_u(imgs)
            loss = criterion(preds, yt)
            loss.backward()
            apply_null_projections(theta_u, projector_map)
            optimizer.step()

            agg_loss += float(loss.item())
            batches_seen += 1
            pred_lab = preds.detach().argmax(dim=1)
            corr += int((pred_lab == yt).sum().item())
            total += int(yt.numel())

        histories["epoch"].append(float(epoch))
        histories["loss"].append(agg_loss / max(1, batches_seen))
        histories["pseudo_acc"].append(float(corr / max(1, total)))

    theta_u.train(False)

    ov, rr, fk = evaluate_splits(
        theta_u,
        test_full_loader=testers.get("test_full"),
        test_retained_loader=testers.get("test_retained"),
        test_forgotten_loader=testers.get("test_forgotten"),
        device=device,
    )

    save_checkpoint(
        str(unlearn_ckpt_fp),
        theta_u,
        extra={"method": "unsc", "forget_class": forget_class, "digest": digest, "seed": int(seed)},
    )

    return {
        "overall_accuracy": ov,
        "retained_accuracy": rr,
        "forgotten_accuracy": fk,
        "training_history_ul": histories,
    }
