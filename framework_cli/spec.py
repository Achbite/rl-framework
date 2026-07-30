from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
IMAGE_NAMES = {
    "learner": "rl-training/learner",
    "aiserver": "rl-training/aiserver",
    "client": "rl-training/maze-client",
}
MODE_ALIASES = {
    "local-model-validation": "model-evaluation",
}
MODES = {"inference-smoke", "training", "model-evaluation"}


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    run_id: str
    attempt: int
    mode: str
    agents: int
    episodes: int
    max_steps: int
    image_tags: dict[str, str]
    model_path: str | None
    source_path: Path

    @property
    def expected_samples(self) -> int:
        return self.agents * self.episodes * self.max_steps

    def image_ref(self, role: str) -> str:
        return f"{IMAGE_NAMES[role]}:{self.image_tags[role]}"


def _positive_int(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecError(f"{field} must be a positive integer")
    return value


def load_run_spec(path: str) -> RunSpec:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read run spec: {exc}") from exc

    if not isinstance(document, dict):
        raise SpecError("run spec must be a JSON object")

    task_id = document.get("task_id")
    run_id = document.get("run_id")
    mode = document.get("mode")
    if not isinstance(task_id, str) or not ID_PATTERN.fullmatch(task_id):
        raise SpecError("task_id has an invalid format")
    if not isinstance(run_id, str) or not ID_PATTERN.fullmatch(run_id):
        raise SpecError("run_id has an invalid format")
    if mode in MODE_ALIASES:
        mode = MODE_ALIASES[mode]
    if mode not in MODES:
        raise SpecError(
            "mode must be inference-smoke, training, or model-evaluation"
        )

    images = document.get("images")
    if not isinstance(images, dict):
        raise SpecError("images must be an object")
    image_tags: dict[str, str] = {}
    for role in ("learner", "aiserver", "client"):
        tag = images.get(role)
        if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag):
            raise SpecError(f"images.{role} must be a Docker tag")
        image_tags[role] = tag

    attempt = _positive_int(document, "attempt")
    agents = _positive_int(document, "agents")
    episodes = _positive_int(document, "episodes")
    max_steps = _positive_int(document, "max_steps")
    if mode == "training" and agents != 4:
        raise SpecError("training requires exactly 4 agents")

    model_path = document.get("model_path")
    if mode == "model-evaluation":
        if not isinstance(model_path, str) or not model_path:
            raise SpecError("model_path is required for model-evaluation")
        model_path = str(Path(model_path).expanduser().resolve())
    elif model_path is not None and not isinstance(model_path, str):
        raise SpecError("model_path must be a string")

    return RunSpec(
        task_id=task_id,
        run_id=run_id,
        attempt=attempt,
        mode=mode,
        agents=agents,
        episodes=episodes,
        max_steps=max_steps,
        image_tags=image_tags,
        model_path=model_path,
        source_path=source,
    )
