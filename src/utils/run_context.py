from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml


def _runtime_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.setdefault("runtime", {})


def _resolve_run_root(cfg: dict[str, Any]) -> Path:
    checkpoint_path = cfg.get("train", {}).get("checkpoint_path")
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path).expanduser()
        return checkpoint_path.parent if checkpoint_path.suffix else checkpoint_path
    return Path("results/checkpoints")


def prepare_run_directory(
    cfg: dict[str, Any],
    *,
    config_path: str | Path | None = None,
    app_name: str | None = None,
) -> dict[str, Any]:
    runtime = _runtime_cfg(cfg)
    if runtime.get("_prepared_in_process"):
        return runtime

    run_root = _resolve_run_root(cfg)
    run_root.mkdir(parents=True, exist_ok=True)

    run_dir: Path | None = None
    run_id = ""
    while run_dir is None:
        run_id = uuid.uuid4().hex[:8]
        candidate = run_root / run_id
        if not candidate.exists():
            run_dir = candidate

    best_dir = run_dir / "best"
    last_dir = run_dir / "last"
    best_dir.mkdir(parents=True, exist_ok=False)
    last_dir.mkdir(parents=True, exist_ok=False)

    runtime.update(
        {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "best_dir": str(best_dir),
            "last_dir": str(last_dir),
            "best_checkpoint_path": str(best_dir / "checkpoint.pth"),
            "last_checkpoint_path": str(last_dir / "checkpoint.pth"),
            "stdout_path": str(run_dir / "log.out"),
            "stderr_path": str(run_dir / "log.err"),
            "config_copy_path": str(run_dir / "config.yaml"),
        }
    )
    if config_path is not None:
        runtime["source_config_path"] = str(Path(config_path).expanduser().resolve())
    if app_name is not None:
        runtime["app_name"] = app_name

    config_copy_path = Path(runtime["config_copy_path"])
    config_copy_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    runtime["_prepared_in_process"] = True
    return runtime


class _TeeStream:
    def __init__(self, primary: Any, secondary: Any) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


@contextmanager
def redirect_run_logs(cfg: dict[str, Any]) -> Iterator[None]:
    runtime = cfg.get("runtime", {})
    if runtime.get("logs_redirected") or not runtime.get("run_dir"):
        yield
        return

    stdout_path = Path(runtime["stdout_path"])
    stderr_path = Path(runtime["stderr_path"])

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        sys.stdout = _TeeStream(original_stdout, stdout_handle)
        sys.stderr = _TeeStream(original_stderr, stderr_handle)
        runtime["logs_redirected"] = True
        print(f"random_run_id={runtime['run_id']}")
        print(f"run_dir={runtime['run_dir']}")
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            runtime["logs_redirected"] = False
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def checkpoint_paths(cfg: dict[str, Any]) -> tuple[Path, Path]:
    runtime = cfg.get("runtime", {})
    best_path = runtime.get("best_checkpoint_path")
    last_path = runtime.get("last_checkpoint_path")
    if not best_path or not last_path:
        raise ValueError("Run directory has not been prepared yet.")
    return Path(best_path), Path(last_path)
