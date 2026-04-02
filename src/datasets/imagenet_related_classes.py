from __future__ import annotations

KINETICS_TO_IMAGENET_RELATED_LABELS: dict[str, list[str]] = {
    # ImageNet is object-centric, so these are approximate object/context
    # categories related to the user-selected Kinetics actions.
    "abseiling": [
        "alp",
        "cliff",
        "cliff dwelling",
        "mountain tent",
        "crash helmet",
    ],
    "air drumming": [
        "drum",
        "steel drum",
        "gong",
        "maraca",
        "microphone",
    ],
    "applauding": [
        "stage",
        "theater curtain",
        "microphone",
        "home theater",
    ],
    "archery": [
        "bow",
    ],
    "arm wrestling": [
        "barbell",
        "dumbbell",
    ],
}


def related_imagenet_labels_for_actions(action_classes: list[str]) -> list[str]:
    labels: list[str] = []
    for action in action_classes:
        labels.extend(KINETICS_TO_IMAGENET_RELATED_LABELS.get(action, []))
    # Preserve order while deduplicating.
    return list(dict.fromkeys(labels))
