#!/usr/bin/env bash

set -euo pipefail

framework_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd "${framework_root}/.." && pwd -P)"

echo "[training-protocol] building the current task-neutral training protocol files"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-contracts/build_artifact.sh" training

echo "[training-protocol] synchronizing AIServer training inputs"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-aiserver/scripts/sync_training_snapshot.sh"

echo "[training-protocol] synchronizing Learner training inputs"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-learner/proto/gen_proto.sh" training

echo "[training-protocol] task-neutral training protocol files synchronized"
