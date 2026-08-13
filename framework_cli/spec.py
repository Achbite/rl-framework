from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")
REPOSITORY_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9._-]*(?::[0-9]+)?"
    r"(?:/[a-z0-9][a-z0-9._-]*)+$"
)
ROLE_NAMES = ("learner", "aiserver", "client")
MODE_ALIASES = {
    "inference-smoke": "local-test",
    "local-model-validation": "model-evaluation",
}
MODES = {"local-test", "training", "model-evaluation"}


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class ResourceSpec:
    cpus: float
    memory_mb: int
    gpus: int


@dataclass(frozen=True)
class ComponentSpec:
    image_repository: str
    image_tag: str
    command: str
    args: tuple[str, ...]
    resources: ResourceSpec

    @property
    def image_ref(self) -> str:
        return f"{self.image_repository}:{self.image_tag}"

    @property
    def command_argv(self) -> list[str]:
        return [self.command, *self.args]

    @property
    def docker_resource_args(self) -> list[str]:
        arguments = [
            "--cpus",
            f"{self.resources.cpus:g}",
            "--memory",
            f"{self.resources.memory_mb}m",
        ]
        if self.resources.gpus > 0:
            arguments.extend(["--gpus", str(self.resources.gpus)])
        return arguments


@dataclass(frozen=True)
class RunSpec:
    schema_version: int
    task_id: str
    run_id: str
    attempt: int
    mode: str
    agents: int
    episodes: int
    max_steps: int
    components: dict[str, ComponentSpec]
    server_pod_same_host: str
    model_path: str | None
    initial_model_dir: str | None
    source_path: Path

    def image_ref(self, role: str) -> str:
        return self.components[role].image_ref


def _positive_int(document: dict[str, Any], field: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SpecError(f"{field} must be a positive integer")
    return value


def _load_resources(role: str, document: Any) -> ResourceSpec:
    if not isinstance(document, dict):
        raise SpecError(f"components.{role}.resources must be an object")
    cpus = document.get("cpus")
    memory_mb = document.get("memory_mb")
    gpus = document.get("gpus")
    if (
        not isinstance(cpus, (int, float))
        or isinstance(cpus, bool)
        or cpus <= 0
    ):
        raise SpecError(f"components.{role}.resources.cpus must be positive")
    if (
        not isinstance(memory_mb, int)
        or isinstance(memory_mb, bool)
        or memory_mb <= 0
    ):
        raise SpecError(
            f"components.{role}.resources.memory_mb must be positive"
        )
    if (
        not isinstance(gpus, int)
        or isinstance(gpus, bool)
        or gpus < 0
    ):
        raise SpecError(
            f"components.{role}.resources.gpus must be non-negative"
        )
    return ResourceSpec(float(cpus), memory_mb, gpus)


def _load_component(role: str, document: Any) -> ComponentSpec:
    if not isinstance(document, dict):
        raise SpecError(f"components.{role} must be an object")
    repository = document.get("image_repository")
    tag = document.get("image_tag")
    command = document.get("command")
    args = document.get("args")
    if (
        not isinstance(repository, str)
        or not REPOSITORY_PATTERN.fullmatch(repository)
    ):
        raise SpecError(
            f"components.{role}.image_repository has an invalid format"
        )
    if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag):
        raise SpecError(f"components.{role}.image_tag must be a Docker tag")
    if not isinstance(command, str) or not command or "\x00" in command:
        raise SpecError(f"components.{role}.command must be a string")
    if (
        not isinstance(args, list)
        or any(not isinstance(item, str) or "\x00" in item for item in args)
    ):
        raise SpecError(f"components.{role}.args must be a string array")
    return ComponentSpec(
        image_repository=repository,
        image_tag=tag,
        command=command,
        args=tuple(args),
        resources=_load_resources(role, document.get("resources")),
    )


def load_run_spec(path: str) -> RunSpec:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read run spec: {exc}") from exc

    if not isinstance(document, dict):
        raise SpecError("run spec must be a JSON object")
    if document.get("schema_version") != 2:
        raise SpecError("run spec schema_version must be 2")

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
            "mode must be local-test, training, or model-evaluation"
        )

    components_document = document.get("components")
    if not isinstance(components_document, dict):
        raise SpecError("components must be an object")
    components = {
        role: _load_component(role, components_document.get(role))
        for role in ROLE_NAMES
    }
    aiserver_args = components["aiserver"].args
    if not aiserver_args or aiserver_args[0] != mode:
        raise SpecError(
            "components.aiserver.args must start with the RunSpec mode"
        )
    reserved_workload_tokens = {
        *MODES,
        *MODE_ALIASES,
        "--workload",
        "--train",
    }
    conflicting_aiserver_args = [
        argument
        for argument in aiserver_args[1:]
        if argument in reserved_workload_tokens
    ]
    if conflicting_aiserver_args:
        raise SpecError(
            "components.aiserver.args cannot override the RunSpec mode"
        )
    if mode == "training":
        learner_args = components["learner"].args
        if not learner_args or learner_args[0] != "training":
            raise SpecError(
                "training requires components.learner.args to start with training"
            )

    placement = document.get("placement")
    if not isinstance(placement, dict):
        raise SpecError("placement must be an object")
    server_pod_same_host = placement.get("aiserver_client_same_host")
    if server_pod_same_host != "required":
        raise SpecError(
            "placement.aiserver_client_same_host must be required"
        )

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

    initial_model_dir = document.get("initial_model_dir")
    if initial_model_dir is not None:
        if mode != "training":
            raise SpecError(
                "initial_model_dir is supported only for training"
            )
        if (
            not isinstance(initial_model_dir, str)
            or not initial_model_dir
        ):
            raise SpecError(
                "initial_model_dir must be a non-empty path"
            )
        initial_model_dir = str(
            Path(initial_model_dir).expanduser().resolve()
        )
    if "initial_checkpoint" in document:
        raise SpecError(
            "initial_checkpoint was removed; use initial_model_dir"
        )
    if "resume_from" in document:
        raise SpecError(
            "resume_from was removed; the same run_id resumes automatically"
        )

    return RunSpec(
        schema_version=2,
        task_id=task_id,
        run_id=run_id,
        attempt=attempt,
        mode=mode,
        agents=agents,
        episodes=episodes,
        max_steps=max_steps,
        components=components,
        server_pod_same_host=server_pod_same_host,
        model_path=model_path,
        initial_model_dir=initial_model_dir,
        source_path=source,
    )
