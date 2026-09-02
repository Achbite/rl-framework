#!/usr/bin/env bash

set -euo pipefail

framework_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd "${framework_root}/.." && pwd -P)"

echo "[artifacts] building task-neutral training protocol bindings"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-contracts/build_artifact.sh" training

echo "[artifacts] building rl-sample-pool"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-sample-pool/build_artifact.sh"

echo "[artifacts] building rl-model-distributor"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-model-distributor/build_artifact.sh"

echo "[artifacts] synchronizing Learner runtime dependencies"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-learner/scripts/sync_runtime_artifacts.sh"

echo "[artifacts] Learner runtime dependencies synchronized"
