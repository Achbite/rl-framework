#!/usr/bin/env python3
"""Compute the immutable identity shared by the three A3 runtime images."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


VERSION = "0.14.0"
PLATFORM = "linux/arm64"
PLATFORM_DIRECTORY = "linux-arm64"
REPOSITORIES = (
    "rl-contracts",
    "rl-sample-pool",
    "rl-model-distributor",
    "rl-learner",
    "rl-aiserver",
    "maze-client",
)
CONFIGS = {
    "learner": ("rl-learner", "configs/learner_config.yaml"),
    "aiserver": ("rl-aiserver", "configs/server_config.yaml"),
    "maze-client": ("maze-client", "configs/client_config.yaml"),
}
ARTIFACTS = (
    "rl-contracts",
    "rl-sample-pool",
    "rl-model-distributor",
)
LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return not stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(
        metadata.st_mode
    )


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        fail(
            f"cannot inspect {repository.name}: "
            f"{completed.stderr.strip() or 'git command failed'}"
        )
    return completed.stdout.strip()


def repository_heads(root: Path) -> dict[str, str]:
    heads: dict[str, str] = {}
    for name in REPOSITORIES:
        repository = root / name
        if not repository.is_dir():
            fail(f"runtime repository is missing: {repository}")
        changed = git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if changed:
            fail(f"runtime repository must be clean: {name}")
        head = git(repository, "rev-parse", "HEAD")
        if not re.fullmatch(r"[0-9a-f]{40}", head):
            fail(f"runtime repository HEAD is invalid: {name}")
        heads[name] = head
    return heads


def load_manifest(path: Path, package: str) -> dict[str, Any]:
    if not regular_file(path):
        fail(f"artifact manifest is missing or unsafe: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"artifact manifest is invalid: {path}: {exc}")
    if not isinstance(manifest, dict):
        fail(f"artifact manifest root is invalid: {path}")
    if (
        manifest.get("package") != package
        or manifest.get("version") != VERSION
        or manifest.get("platform") != PLATFORM
    ):
        fail(f"artifact identity mismatch: {path}")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(
        r"[0-9a-f]{12,40}", source_commit
    ):
        fail(f"artifact source commit is invalid: {path}")
    return manifest


def artifact_identities(
    root: Path,
    heads: dict[str, str],
) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for package in ARTIFACTS:
        path = (
            root
            / ".workspace"
            / "artifacts"
            / package
            / VERSION
            / PLATFORM_DIRECTORY
            / "manifest.json"
        )
        manifest = load_manifest(path, package)
        if not heads[package].startswith(manifest["source_commit"]):
            fail(f"{package} artifact does not match the current clean HEAD")
        manifests[package] = manifest
        identities[package] = {
            "manifest_sha256": sha256(path),
            "source_commit": manifest["source_commit"],
        }

    contract_digest = (
        manifests["rl-contracts"].get("artifact_digest", {}).get("hex")
    )
    if not isinstance(contract_digest, str) or not LOWER_SHA256.fullmatch(
        contract_digest
    ):
        fail("Contracts artifact digest is invalid")
    if manifests["rl-contracts"].get("source_tree_state") != "clean":
        fail("Contracts artifact is not from clean source")
    identities["rl-contracts"]["artifact_digest"] = contract_digest

    for package in ("rl-sample-pool", "rl-model-distributor"):
        embedded = (
            manifests[package]
            .get("contract", {})
            .get("artifact_digest", {})
            .get("hex")
        )
        if embedded != contract_digest:
            fail(f"{package} is not bound to the selected Contracts artifact")
        identities[package]["contracts_artifact_digest"] = embedded
    return identities


def config_identities(root: Path) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for component, (repository, relative) in CONFIGS.items():
        path = root / repository / relative
        if not regular_file(path):
            fail(f"component config is missing or unsafe: {path}")
        identities[component] = {
            "path": f"{repository}/{relative}",
            "sha256": sha256(path),
        }
    return identities


def compute(root: Path) -> dict[str, Any]:
    root = root.resolve()
    heads = repository_heads(root)
    inputs = {
        "schema_version": 1,
        "contracts_version": VERSION,
        "platform": PLATFORM,
        "repositories": heads,
        "artifacts": artifact_identities(root, heads),
        "configs": config_identities(root),
    }
    canonical = json.dumps(
        inputs, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return {
        **inputs,
        "stack_source_id": hashlib.sha256(canonical).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True, type=Path)
    arguments = parser.parse_args()
    document = compute(arguments.workspace_root)
    print(json.dumps(document, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
