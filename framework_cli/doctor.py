"""Read-only diagnostics for the RL training workspace."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Optional, Union


SCHEMA_VERSION = 1
CONTRACTS_VERSION = "0.13.0"
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

ARTIFACTS = (
    ("rl-contracts", 2),
    ("rl-sample-pool", 1),
    ("rl-model-distributor", 1),
)

IMAGE_COMPONENTS = {
    "learner": "learner",
    "aiserver": "aiserver",
    "client": "maze-client",
}

IMAGE_RUNTIME_CONTRACTS = {
    "learner": {
        "entrypoint": ["/opt/rl/learner/scripts/entrypoint.sh"],
        "cmd": ["--config", "configs/learner_config.yaml"],
    },
    "aiserver": {
        "entrypoint": ["/opt/rl/aiserver/scripts/entrypoint.sh"],
        "cmd": ["--config", "configs/server_config.yaml"],
    },
    "client": {
        "entrypoint": ["/opt/rl/maze-client/scripts/entrypoint.sh"],
        "cmd": ["--config", "configs/client_config.yaml"],
    },
}

IMAGE_IDENTITY_LABELS = {
    "component": "org.rl-training.component",
    "component_commit": "org.rl-training.component-commit",
    "stack_source_id": "org.rl-training.stack-source-id",
    "contracts_version": "org.rl-training.contracts-version",
    "contracts_artifact_digest":
        "org.rl-training.contracts-artifact-digest",
    "contracts_manifest_digest":
        "org.rl-training.contracts-manifest-digest",
    "component_config_digest":
        "org.rl-training.component-config-digest",
}

_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{12,40}$")

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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


def _digest_identity(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("algorithm") == "sha256"
        and isinstance(value.get("hex"), str)
        and _LOWER_SHA256.fullmatch(value["hex"]) is not None
    )


def _verify_artifact(
    root: Path,
    package: str,
    expected_schema: int,
) -> tuple[dict[str, Any], list[str]]:
    artifact_root = (
        root
        / ".workspace"
        / "artifacts"
        / package
        / CONTRACTS_VERSION
        / PLATFORM_DIRECTORY
    )
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

    for key, expected in (
        ("schema_version", expected_schema),
        ("package", package),
        ("version", CONTRACTS_VERSION),
        ("platform", ARTIFACT_PLATFORM),
    ):
        if document.get(key) != expected:
            failures.append(
                f"{package} manifest {key} mismatch: "
                f"expected {expected!r}, found {document.get(key)!r}"
            )

    source_commit = document.get("source_commit")
    source_id = document.get("source_id")
    if not isinstance(source_commit, str) or not _GIT_COMMIT.fullmatch(
        source_commit
    ):
        failures.append(f"{package} source_commit is invalid")
    if not isinstance(source_id, str) or not source_id:
        failures.append(f"{package} source_id is missing")

    source_identity: dict[str, Any] = {
        "commit": source_commit,
        "id": source_id,
    }
    if package == "rl-contracts":
        source_identity["digest"] = document.get("source_digest")
        source_identity["tree_state"] = document.get("source_tree_state")
        if not _digest_identity(document.get("source_digest")):
            failures.append("rl-contracts source_digest is invalid")
        if document.get("source_tree_state") != "clean":
            failures.append("rl-contracts artifact is not from clean source")
    else:
        source_identity["sha256"] = document.get("source_sha256")
        if not isinstance(document.get("source_sha256"), str) or not _LOWER_SHA256.fullmatch(
            document["source_sha256"]
        ):
            failures.append(f"{package} source_sha256 is invalid")
        contract = document.get("contract")
        if not isinstance(contract, dict):
            failures.append(f"{package} embedded contract identity is missing")
        else:
            if contract.get("version") != CONTRACTS_VERSION:
                failures.append(f"{package} embeds the wrong Contracts version")
            if contract.get("platform") != ARTIFACT_PLATFORM:
                failures.append(f"{package} embeds the wrong Contracts platform")
            if not _digest_identity(contract.get("artifact_digest")):
                failures.append(f"{package} contract artifact digest is invalid")

    files = document.get("files")
    verified_files: dict[str, Any] = {}
    if not isinstance(files, dict) or not files:
        failures.append(f"{package} files table is empty or invalid")
    else:
        for relative, expected_digest in sorted(files.items()):
            if not _safe_artifact_path(relative):
                failures.append(f"{package} contains unsafe path: {relative!r}")
                continue
            path = artifact_root / PurePosixPath(relative)
            if not isinstance(expected_digest, str) or not _LOWER_SHA256.fullmatch(
                expected_digest
            ):
                failures.append(f"{package} has invalid digest for {relative}")
                continue
            if not _regular_file(path):
                failures.append(f"{package} file is missing: {relative}")
                continue
            actual_digest = _sha256(path)
            verified_files[relative] = {
                "sha256": actual_digest,
                "size_bytes": path.stat().st_size,
                "matches_manifest": actual_digest == expected_digest,
            }
            if actual_digest != expected_digest:
                failures.append(f"{package} file digest mismatch: {relative}")

    if package == "rl-contracts":
        canonical_files = json.dumps(
            files if isinstance(files, dict) else {},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        expected_artifact_digest = hashlib.sha256(canonical_files).hexdigest()
        if document.get("artifact_digest") != {
            "algorithm": "sha256",
            "hex": expected_artifact_digest,
        }:
            failures.append("rl-contracts artifact_digest is invalid")

    output.update(
        {
            "ok": not failures,
            "version": document.get("version"),
            "platform": document.get("platform"),
            "source": source_identity,
            "manifest_sha256": _sha256(manifest_path),
            "artifact_digest": document.get("artifact_digest"),
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
    for package, schema_version in ARTIFACTS:
        inspected, package_failures = _verify_artifact(
            root, package, schema_version
        )
        artifacts[package] = inspected
        failures.extend(package_failures)

    contracts_digest = (
        artifacts.get("rl-contracts", {})
        .get("artifact_digest", {})
        .get("hex")
    )
    if contracts_digest:
        for package in ("rl-sample-pool", "rl-model-distributor"):
            embedded = (
                artifacts.get(package, {})
                .get("contract", {})
                .get("artifact_digest", {})
                .get("hex")
            )
            if embedded != contracts_digest:
                message = f"{package} is not bound to the selected Contracts artifact"
                failures.append(message)
                artifacts[package]["ok"] = False
                artifacts[package]["error"] = message

    return result(
        "artifacts",
        not failures,
        {
            "workspace_root": str(root),
            "required_version": CONTRACTS_VERSION,
            "required_platform": ARTIFACT_PLATFORM,
            "artifacts": artifacts,
        },
        error="; ".join(failures) if failures else None,
    )


def doctor_model(model: Union[str, os.PathLike[str]]) -> dict[str, Any]:
    requested = Path(model).expanduser()
    path = Path(os.path.abspath(requested))
    failures: list[str] = []
    if path.name != "SaveModel.onnx":
        failures.append("model filename must be exactly SaveModel.onnx")
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
        data.update({"size_bytes": size, "sha256": _sha256(path)})
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
    repo_digests = document.get("RepoDigests") or []
    image_config = document.get("Config") or {}
    labels = image_config.get("Labels") or {}
    if not isinstance(image_id, str) or not image_id.startswith("sha256:") or not _LOWER_SHA256.fullmatch(
        image_id.removeprefix("sha256:")
    ):
        failures.append(f"{role} image ID is invalid")
    if not isinstance(repo_digests, list) or not all(
        isinstance(value, str) for value in repo_digests
    ):
        failures.append(f"{role} RepoDigests is invalid")
        repo_digests = []
    if not isinstance(labels, dict):
        failures.append(f"{role} image labels are invalid")
        labels = {}

    runtime_contract = IMAGE_RUNTIME_CONTRACTS[role]
    entrypoint = image_config.get("Entrypoint")
    command = image_config.get("Cmd")
    if entrypoint != runtime_contract["entrypoint"]:
        failures.append(f"{role} image ENTRYPOINT is missing or incorrect")
    if command != runtime_contract["cmd"]:
        failures.append(f"{role} image CMD is missing or incorrect")

    identity_labels = {
        logical: labels.get(key)
        for logical, key in IMAGE_IDENTITY_LABELS.items()
    }
    expected_component = IMAGE_COMPONENTS[role]
    if identity_labels["component"] != expected_component:
        failures.append(f"{role} image component label is missing or incorrect")
    if not isinstance(identity_labels["component_commit"], str) or not _GIT_COMMIT.fullmatch(
        identity_labels["component_commit"]
    ):
        failures.append(f"{role} component commit label is invalid")
    if identity_labels["contracts_version"] != CONTRACTS_VERSION:
        failures.append(f"{role} Contracts version label is invalid")
    if not isinstance(identity_labels["contracts_artifact_digest"], str) or not _LOWER_SHA256.fullmatch(
        identity_labels["contracts_artifact_digest"]
    ):
        failures.append(f"{role} Contracts artifact label is invalid")
    if not isinstance(identity_labels["contracts_manifest_digest"], str) or not _LOWER_SHA256.fullmatch(
        identity_labels["contracts_manifest_digest"]
    ):
        failures.append(f"{role} Contracts manifest label is invalid")
    if not isinstance(identity_labels["component_config_digest"], str) or not _LOWER_SHA256.fullmatch(
        identity_labels["component_config_digest"]
    ):
        failures.append(f"{role} component config label is invalid")
    if not isinstance(identity_labels["stack_source_id"], str) or not _LOWER_SHA256.fullmatch(
        identity_labels["stack_source_id"]
    ):
        failures.append(f"{role} stack source label is invalid")

    assembly_labels = {
        key: value
        for key, value in labels.items()
        if isinstance(key, str) and key.startswith("org.rl-training.")
    }
    if role == "learner":
        for dependency in ("sample-pool", "model-distributor"):
            key = f"org.rl-training.{dependency}-manifest-digest"
            value = assembly_labels.get(key)
            if not isinstance(value, str) or not _LOWER_SHA256.fullmatch(value):
                failures.append(
                    f"learner {dependency} manifest label is invalid"
                )

    output.update(
        {
            "ok": not failures,
            "image_id": image_id,
            "repo_digests": repo_digests,
            "entrypoint": entrypoint,
            "cmd": command,
            "identity_labels": identity_labels,
            "assembly_labels": assembly_labels,
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

    stack_ids = {
        image.get("identity_labels", {}).get("stack_source_id")
        for image in images.values()
        if image.get("identity_labels", {}).get("stack_source_id")
    }
    contract_digests = {
        image.get("identity_labels", {}).get("contracts_artifact_digest")
        for image in images.values()
        if image.get("identity_labels", {}).get("contracts_artifact_digest")
    }
    if len(stack_ids) > 1:
        failures.append("runtime images do not share one stack_source_id")
    if len(contract_digests) > 1:
        failures.append("runtime images do not share one Contracts artifact")

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
