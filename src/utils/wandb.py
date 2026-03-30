from __future__ import annotations

import os
from typing import Any


def _wandb_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("wandb", {}))


def wandb_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_wandb_cfg(cfg).get("enabled", False))


def init_wandb_run(cfg: dict[str, Any], *, job_type: str) -> Any | None:
    wandb_cfg = _wandb_cfg(cfg)
    if not bool(wandb_cfg.get("enabled", False)):
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "W&B logging is enabled in the config, but the 'wandb' package is not installed. "
            "Install requirements.txt or run `pip install wandb`."
        ) from exc

    project = os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project") or "rethinking-jepa"
    entity = os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    run_name = wandb_cfg.get("name")
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode") or "online"
    tags = list(wandb_cfg.get("tags", []))
    group = wandb_cfg.get("group")
    save_dir = wandb_cfg.get("dir")
    fallback_to_offline = bool(wandb_cfg.get("fallback_to_offline", False))

    init_kwargs = {
        "project": project,
        "entity": entity,
        "name": run_name,
        "mode": mode,
        "tags": tags,
        "group": group,
        "dir": save_dir,
        "config": cfg,
        "job_type": job_type,
    }

    try:
        run = wandb.init(**init_kwargs)
    except wandb.errors.CommError as exc:
        if mode != "online" or not fallback_to_offline:
            raise
        print(
            "wandb: online init failed; falling back to offline mode. "
            f"Original error: {exc}"
        )
        init_kwargs["mode"] = "offline"
        run = wandb.init(**init_kwargs)

    if run is not None:
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
    return run


def log_wandb_metrics(run: Any | None, metrics: dict[str, Any]) -> None:
    if run is None:
        return
    run.log(metrics)


def finish_wandb_run(run: Any | None, summary: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[key] = value
    run.finish()
