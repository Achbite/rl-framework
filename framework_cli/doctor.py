"""Read-only diagnostics for the RL training workspace."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Union


SCHEMA_VERSION = 1
CONTRACTS_VERSION = "0.15.0"
ARTIFACT_PLATFORM = "linux/arm64"
PLATFORM_DIRECTORY = "linux-arm64"

REPOSITORIES = (
    "rl-contracts",
    "rl-sample-pool",
    "rl-model-distributor",
    "rl-learner",
    "rl-aiserver",
    "maze-client",
    "rl-framework",
    "rl-infra",
)

ARTIFACT_SCHEMA = "rl.artifact-manifest.v1"
ARTIFACTS = {
    "rl-training-contracts": {
        "storage_package": "rl-contracts",
        "storage_suffix": ("training",),
        "platform": None,
        "required_files": {
            "common.proto",
            "training.proto",
            "training_metrics.proto",
            "cpp/common.pb.cc",
            "cpp/training.pb.cc",
            "cpp/training.grpc.pb.cc",
            "cpp/training_metrics.pb.cc",
            "python/common_pb2.py",
            "python/training_pb2.py",
            "python/training_pb2_grpc.py",
            "python/training_metrics_pb2.py",
            "schemas/training.metrics.json",
        },
        "required_executables": set(),
    },
    "rl-sample-pool": {
        "storage_package": "rl-sample-pool",
        "storage_suffix": (PLATFORM_DIRECTORY,),
        "platform": ARTIFACT_PLATFORM,
        "required_files": {
            "bin/maze_sample_pool",
            "config/pool_config.yaml",
        },
        "required_executables": {"bin/maze_sample_pool"},
    },
    "rl-model-distributor": {
        "storage_package": "rl-model-distributor",
        "storage_suffix": (PLATFORM_DIRECTORY,),
        "platform": ARTIFACT_PLATFORM,
        "required_files": {
            "bin/maze_model_distributor",
            "config/model_distributor_config.yaml",
        },
        "required_executables": {"bin/maze_model_distributor"},
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


def _safe_artifact_path(relative: str) -> bool:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        return False
    candidate = PurePosixPath(relative)
    return not candidate.is_absolute() and ".." not in candidate.parts


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
            "lifecycle": "legacy/disabled" if name == "rl-infra" else "active",
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
            "infra": {
                "repository": "rl-infra",
                "lifecycle": "legacy/disabled",
                "entered": False,
            },
        },
        warnings,
        "; ".join(failures) if failures else None,
    )


def _load_manifest(
    path: Path,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not _regular_file(path):
        return None, f"manifest is missing or is not a regular file: {path}"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"manifest is invalid: {path}: {exc}"
    if not isinstance(document, dict):
        return None, f"manifest root is not an object: {path}"
    return document, None


def _verify_artifact(
    root: Path,
    package: str,
    contract_reference: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    specification = ARTIFACTS[package]
    artifact_root = (
        root
        / ".workspace"
        / "artifacts"
        / specification["storage_package"]
        / CONTRACTS_VERSION
    )
    for relative in specification["storage_suffix"]:
        artifact_root /= relative
    manifest_path = artifact_root / "manifest.json"
    document, load_error = _load_manifest(manifest_path)
    failures: list[str] = []
    output: dict[str, Any] = {
        "path": str(artifact_root),
        "manifest": str(manifest_path),
        "present": document is not None,
        "ok": False,
    }
    if load_error or document is None:
        failures.append(load_error or f"cannot load {package} manifest")
        output["error"] = failures[-1]
        return output, failures

    expected_identity = [
        ("schema_version", ARTIFACT_SCHEMA),
        ("artifact_channel", "production"),
        ("package", package),
        ("version", CONTRACTS_VERSION),
    ]
    if specification["platform"] is not None:
        expected_identity.append(("platform", specification["platform"]))
    for key, expected in expected_identity:
        if document.get(key) != expected:
            failures.append(
                f"{package} manifest {key} mismatch: "
                f"expected {expected!r}, found {document.get(key)!r}"
            )

    files = document.get("files")
    verified_files: dict[str, Any] = {}
    if (
        not isinstance(files, list)
        or not files
        or not all(isinstance(relative, str) for relative in files)
    ):
        failures.append(f"{package} files list is empty or invalid")
    else:
        declared = set(files)
        required_files = ARTIFACTS[package]["required_files"]
        missing = required_files - declared
        if missing:
            failures.append(
                f"{package} does not declare required files: {sorted(missing)}"
            )
        for relative in sorted(declared):
            if not _safe_artifact_path(relative):
                failures.append(f"{package} contains unsafe path: {relative!r}")
                continue
            path = artifact_root / PurePosixPath(relative)
            if not _regular_file(path):
                failures.append(f"{package} file is missing: {relative}")
                continue
            verified_files[relative] = {
                "size_bytes": path.stat().st_size,
            }
        required_executables = ARTIFACTS[package]["required_executables"]
        manifest_executables = document.get("executables", [])
        if package != "rl-training-contracts":
            embedded_contract = document.get("contract")
            if not isinstance(embedded_contract, dict) or any(
                embedded_contract.get(key) != expected
                for key, expected in contract_reference.items()
            ):
                failures.append(
                    f"{package} uses an incompatible Contracts package"
                )
            if (
                not isinstance(manifest_executables, list)
                or not all(
                    isinstance(relative, str)
                    for relative in manifest_executables
                )
                or not required_executables.issubset(
                    set(manifest_executables)
                )
            ):
                failures.append(
                    f"{package} executable declaration is incomplete"
                )
        for relative in required_executables:
            path = artifact_root / PurePosixPath(relative)
            if not _regular_file(path) or not os.access(path, os.X_OK):
                failures.append(
                    f"{package} executable is missing or not executable: {relative}"
                )

    output.update(
        {
            "ok": not failures,
            "version": document.get("version"),
            "platform": document.get("platform"),
            "contract": document.get("contract"),
            "files": verified_files,
        }
    )
    if failures:
        output["error"] = "; ".join(failures)
    return output, failures


def doctor_artifacts(root: Optional[Path] = None) -> dict[str, Any]:
    root = (root or workspace_root()).absolute()
    artifacts: dict[str, Any] = {}
    failures: list[str] = []
    contract_reference = {
        "package": "rl-training-contracts",
        "version": CONTRACTS_VERSION,
    }
    for package in ARTIFACTS:
        inspected, package_failures = _verify_artifact(
            root, package, contract_reference
        )
        artifacts[package] = inspected
        failures.extend(package_failures)

    return result(
        "artifacts",
        not failures,
        {
            "workspace_root": str(root),
            "required_version": CONTRACTS_VERSION,
            "required_binary_platform": ARTIFACT_PLATFORM,
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
