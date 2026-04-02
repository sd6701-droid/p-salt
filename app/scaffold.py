from __future__ import annotations

import importlib


APP_MODULES = {
    "rethinking_jepa.prepare_imagenet_probe_subset": "app.rethinking_jepa.prepare_imagenet_probe_subset",
    "rethinking_jepa.eval_probe_student_imagenet": "app.rethinking_jepa.eval_probe_student_imagenet",
    "rethinking_jepa.probe_student_imagenet": "app.rethinking_jepa.probe_student_imagenet",
    "rethinking_jepa.prepare_kinetics700_subset": "app.rethinking_jepa.prepare_kinetics700_subset",
    "rethinking_jepa.student": "app.rethinking_jepa.student",
    "rethinking_jepa.teacher": "app.rethinking_jepa.train",
    "rethinking_jepa.train": "app.rethinking_jepa.train",
}


def main(app_name: str, cfg: dict) -> None:
    if app_name not in APP_MODULES:
        choices = ", ".join(sorted(APP_MODULES))
        raise ValueError(f"Unknown app '{app_name}'. Available: {choices}")
    module = importlib.import_module(APP_MODULES[app_name])
    module.main(cfg=cfg)
