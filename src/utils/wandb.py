from __future__ import annotations

import os
from typing import Any


def _wandb_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("wandb", {}))


def _infer_wandb_run_name(cfg: dict[str, Any], job_type: str) -> str | None:
    app_name = str(cfg.get("app", ""))
    teacher_arch = cfg.get("model", {}).get("architecture")
    student_arch = cfg.get("student_model", {}).get("architecture")

    if "student" in app_name or job_type == "student-train":
        if student_arch and teacher_arch:
            return f"student-{student_arch}-from-{teacher_arch}-teacher"
        if student_arch:
            return f"student-{student_arch}"

    if "train" in app_name or "teacher" in app_name or job_type == "teacher-train":
        if teacher_arch:
            return f"teacher-{teacher_arch}"

    return None


def wandb_enabled(cfg: dict[str, Any]) -> bool:
    return bool(_wandb_cfg(cfg).get("enabled", False))


def _wandb_settings(wandb_module: Any, wandb_cfg: dict[str, Any]) -> Any | None:
    init_timeout = os.environ.get("WANDB_INIT_TIMEOUT") or wandb_cfg.get("init_timeout")
    if init_timeout in (None, ""):
        return None

    try:
        timeout_seconds = float(init_timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "W&B init_timeout must be numeric when provided via config or "
            "WANDB_INIT_TIMEOUT."
        ) from exc

    return wandb_module.Settings(init_timeout=timeout_seconds)


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
    run_name = (
        os.environ.get("WANDB_RUN_NAME")
        or wandb_cfg.get("name")
        or _infer_wandb_run_name(cfg, job_type)
    )
    mode = os.environ.get("WANDB_MODE") or wandb_cfg.get("mode") or "online"
    tags = list(wandb_cfg.get("tags", []))
    group = wandb_cfg.get("group")
    save_dir = wandb_cfg.get("dir") or cfg.get("runtime", {}).get("run_dir")
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
    settings = _wandb_settings(wandb, wandb_cfg)
    if settings is not None:
        init_kwargs["settings"] = settings

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


def log_wandb_artifact(
    run: Any | None,
    *,
    name: str,
    artifact_type: str,
    paths: list[str],
    metadata: dict[str, Any] | None = None,
) -> None:
    if run is None:
        return

    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    for path in paths:
        artifact.add_file(path)
    run.log_artifact(artifact)


def finish_wandb_run(run: Any | None, summary: dict[str, Any] | None = None) -> None:
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            run.summary[key] = value
    run.finish()
