"""Read-only diagnostics for the RL training workspace."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union


SCHEMA_VERSION = 1

REPOSITORIES = (
    "rl-contracts",
    "rl-sample-pool",
    "rl-model-distributor",
    "rl-learner",
    "rl-aiserver",
    "maze-client",
    "rl-framework",
)

LEARNER_RUNTIME_ARTIFACTS = {
    "rl-sample-pool": {
        "root": ("rl-learner", "sample-pool"),
        "files": {
            "bin/maze_sample_pool": True,
            "config/pool_config.yaml": False,
        },
    },
    "rl-model-distributor": {
        "root": ("rl-learner", "model-distributor"),
        "files": {
            "bin/maze_model_distributor": True,
            "config/model_distributor_config.yaml": False,
        },
    },
}

IMAGE_COMPONENTS = {
    "learner": "learner",
    "aiserver": "aiserver",
    "client": "client",
}

IMAGE_LABELS = {
    "component": "org.rl-training.component",
    "project_image_tag": "org.rl-training.project-image-tag",
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def result(
    scope: str,
    ok: bool,
    data: Optional[dict[str, Any]] = None,
    warnings: Iterable[str] = (),
    error: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "scope": scope,
        "data": data or {},
        "warnings": list(warnings),
        "error": error,
    }


def parameter_error(scope: str, message: str) -> dict[str, Any]:
    return result(scope, False, error=message)


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, **kwargs)


def _regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(
        metadata.st_mode
    )


def _git(
    repository: Path,
    arguments: list[str],
    runner: CommandRunner,
) -> subprocess.CompletedProcess[str]:
    return runner(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def doctor_source(
    root: Optional[Path] = None,
    runner: CommandRunner = _run,
) -> dict[str, Any]:
    root = (root or workspace_root()).absolute()
    repositories: dict[str, Any] = {}
    warnings: list[str] = []
    failures: list[str] = []

    for name in REPOSITORIES:
        path = root / name
        record: dict[str, Any] = {
            "path": str(path),
            "present": path.is_dir(),
            "branch": None,
            "head": None,
            "dirty": None,
            "changed_paths": None,
            "lifecycle": "active",
        }
        repositories[name] = record
        if not path.is_dir():
            failures.append(f"repository is missing: {name}")
            continue
        try:
            head = _git(path, ["rev-parse", "HEAD"], runner)
            branch = _git(path, ["branch", "--show-current"], runner)
            status_result = _git(
                path,
                ["status", "--porcelain=v1", "--untracked-files=all"],
                runner,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"cannot inspect {name}: {exc}")
            continue
        if head.returncode != 0 or branch.returncode != 0 or status_result.returncode != 0:
            detail = (
                head.stderr.strip()
                or branch.stderr.strip()
                or status_result.stderr.strip()
                or "git inspection failed"
            )
            failures.append(f"cannot inspect {name}: {detail}")
            continue
        changed = [
            line for line in status_result.stdout.splitlines() if line.strip()
        ]
        record.update(
            {
                "branch": branch.stdout.strip() or None,
                "head": head.stdout.strip(),
                "dirty": bool(changed),
                "changed_paths": len(changed),
            }
        )
        if changed:
            warnings.append(f"{name} has {len(changed)} changed path(s)")
        if not record["branch"]:
            warnings.append(f"{name} is detached")

    return result(
        "source",
        not failures,
        {
            "workspace_root": str(root),
            "repositories": repositories,
        },
        warnings,
        "; ".join(failures) if failures else None,
    )


def _inspect_learner_runtime_artifact(
    root: Path,
    package: str,
) -> tuple[dict[str, Any], list[str]]:
    specification = LEARNER_RUNTIME_ARTIFACTS[package]
    artifact_root = root.joinpath(*specification["root"])
    failures: list[str] = []
    files: dict[str, Any] = {}
    output: dict[str, Any] = {
        "path": str(artifact_root),
        "present": artifact_root.is_dir(),
        "ok": False,
    }
    for relative, executable in specification["files"].items():
        path = artifact_root / relative
        regular = _regular_file(path)
        executable_ok = regular and (not executable or os.access(path, os.X_OK))
        files[relative] = {
            "path": str(path),
            "regular_file": regular,
            "executable": executable_ok if executable else None,
            "size_bytes": path.stat().st_size if regular else None,
        }
        if not regular:
            failures.append(f"{package} required file is missing: {relative}")
        elif executable and not executable_ok:
            failures.append(
                f"{package} binary is not executable: {relative}"
            )

    output.update(
        {
            "ok": not failures,
            "files": files,
        }
    )
    if failures:
        output["error"] = "; ".join(failures)
    return output, failures


def doctor_artifacts(root: Optional[Path] = None) -> dict[str, Any]:
    root = (root or workspace_root()).absolute()
    artifacts: dict[str, Any] = {}
    failures: list[str] = []
    for package in LEARNER_RUNTIME_ARTIFACTS:
        inspected, package_failures = _inspect_learner_runtime_artifact(
            root, package
        )
        artifacts[package] = inspected
        failures.extend(package_failures)

    return result(
        "artifacts",
        not failures,
        {
            "workspace_root": str(root),
            "artifacts": artifacts,
        },
        error="; ".join(failures) if failures else None,
    )


def doctor_model(model: Union[str, os.PathLike[str]]) -> dict[str, Any]:
    requested = Path(model).expanduser()
    path = Path(os.path.abspath(requested))
    failures: list[str] = []
    if path.is_symlink():
        failures.append("model must not be a symbolic link")
    if not _regular_file(path):
        failures.append("model is missing or is not a regular file")

    data: dict[str, Any] = {
        "path": str(path),
        "filename": path.name,
        "regular_file": _regular_file(path),
        "symbolic_link": path.is_symlink(),
    }
    if not failures:
        size = path.stat().st_size
        data["size_bytes"] = size
        if size <= 0:
            failures.append("model file is empty")

    return result(
        "model",
        not failures,
        data,
        error="; ".join(failures) if failures else None,
    )


def _inspect_image(
    role: str,
    reference: str,
    runner: CommandRunner,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    output: dict[str, Any] = {"reference": reference, "ok": False}
    try:
        completed = runner(
            ["docker", "image", "inspect", reference],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        failures.append(f"cannot inspect {role} image: {exc}")
        output["error"] = failures[-1]
        return output, failures
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "docker image inspect failed"
        failures.append(f"cannot inspect {role} image {reference}: {detail}")
        output["error"] = failures[-1]
        return output, failures
    try:
        documents = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        failures.append(f"invalid docker inspect JSON for {role}: {exc}")
        output["error"] = failures[-1]
        return output, failures
    if not isinstance(documents, list) or len(documents) != 1 or not isinstance(
        documents[0], dict
    ):
        failures.append(f"docker inspect returned an invalid {role} record")
        output["error"] = failures[-1]
        return output, failures

    document = documents[0]
    image_id = document.get("Id")
    image_config = document.get("Config") or {}
    labels = image_config.get("Labels") or {}
    if not isinstance(labels, dict):
        failures.append(f"{role} image labels are invalid")
        labels = {}

    entrypoint = image_config.get("Entrypoint")
    command = image_config.get("Cmd")
    if not entrypoint and not command:
        failures.append(f"{role} image has no executable ENTRYPOINT or CMD")

    runtime_labels = {
        logical: labels.get(key)
        for logical, key in IMAGE_LABELS.items()
    }
    expected_component = IMAGE_COMPONENTS[role]
    if runtime_labels["component"] != expected_component:
        failures.append(f"{role} image component label is missing or incorrect")
    if not isinstance(runtime_labels["project_image_tag"], str) or not (
        runtime_labels["project_image_tag"]
    ):
        failures.append(f"{role} project image tag label is missing")

    output.update(
        {
            "ok": not failures,
            "image_id": image_id,
            "entrypoint": entrypoint,
            "cmd": command,
            "runtime_labels": runtime_labels,
        }
    )
    if failures:
        output["error"] = "; ".join(failures)
    return output, failures


def doctor_images(
    learner_image: str,
    aiserver_image: str,
    client_image: str,
    runner: CommandRunner = _run,
) -> dict[str, Any]:
    requested = {
        "learner": learner_image,
        "aiserver": aiserver_image,
        "client": client_image,
    }
    images: dict[str, Any] = {}
    failures: list[str] = []
    for role, reference in requested.items():
        inspected, image_failures = _inspect_image(role, reference, runner)
        images[role] = inspected
        failures.extend(image_failures)

    return result(
        "images",
        not failures,
        {"images": images},
        error="; ".join(failures) if failures else None,
    )


def doctor_all(
    model: Union[str, os.PathLike[str]],
    learner_image: str,
    aiserver_image: str,
    client_image: str,
    root: Optional[Path] = None,
    source_runner: CommandRunner = _run,
    image_runner: CommandRunner = _run,
) -> dict[str, Any]:
    checks = {
        "source": doctor_source(root, source_runner),
        "artifacts": doctor_artifacts(root),
        "model": doctor_model(model),
        "images": doctor_images(
            learner_image, aiserver_image, client_image, image_runner
        ),
    }
    warnings = [
        f"{scope}: {warning}"
        for scope, check in checks.items()
        for warning in check["warnings"]
    ]
    failures = [
        f"{scope}: {check['error']}"
        for scope, check in checks.items()
        if not check["ok"]
    ]
    return result(
        "all",
        not failures,
        {"checks": checks},
        warnings,
        "; ".join(failures) if failures else None,
    )
