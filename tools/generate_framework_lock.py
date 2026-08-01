#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


REPOSITORIES = (
    "rl-learner",
    "rl-aiserver",
    "rl-sample-pool",
    "rl-model-distributor",
    "rl-contracts",
    "rl-infra",
    "maze-client",
)


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def main() -> int:
    framework_repo = Path(__file__).resolve().parents[1]
    root = framework_repo.parent
    rows = []
    for name in REPOSITORIES:
        repo = root / name
        status = git(repo, "status", "--porcelain=v1")
        if status:
            raise RuntimeError(f"{name} is not clean")
        commit = git(repo, "rev-parse", "HEAD")
        branch = git(repo, "branch", "--show-current")
        rows.append((name, branch, commit))

    lines = ["schema_version: 1", "contract_version: 0.6.0", "repositories:"]
    for name, branch, commit in rows:
        lines.extend(
            [
                f"  {name}:",
                f"    branch: {branch}",
                f"    commit: {commit}",
            ]
        )
    lines.extend(
        [
            "artifacts:",
            '  contracts: "rl-contracts/0.6.0/<platform>"',
            '  sample_pool: "rl-sample-pool/0.6.0/<platform>"',
            '  model_distributor: "rl-model-distributor/0.5.0/<platform>"',
            '  smoke_model: "rl-smoke-model/0.5.0/any"',
        ]
    )
    output = framework_repo / "framework.lock.yaml"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
