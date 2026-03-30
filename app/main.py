from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")
    sys.path.insert(0, str(repo_root))
    from app.scaffold import main as app_main
    from src.utils.config import load_config
else:
    load_env_file(Path(__file__).resolve().parents[1] / ".env")
    from .scaffold import main as app_main
    from src.utils.config import load_config


def infer_app_name(config_path: str, cfg: dict) -> str:
    app_name = cfg.get("app")
    if app_name:
        return app_name
    lowered = config_path.lower()
    if "student" in lowered:
        return "rethinking_jepa.student"
    return "rethinking_jepa.train"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--fname", dest="config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    app_name = infer_app_name(args.config, cfg)
    app_main(app_name, cfg)


if __name__ == "__main__":
    main()
