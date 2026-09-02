#!/usr/bin/env bash

set -euo pipefail

framework_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd "${framework_root}/.." && pwd -P)"

echo "[maze-protocol] building the current Maze task protocol files"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-contracts/build_artifact.sh" task-maze

echo "[maze-protocol] synchronizing AIServer task inputs"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-aiserver/scripts/sync_contract_snapshot.sh"

echo "[maze-protocol] synchronizing Client task inputs"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/maze-client/scripts/sync_contract_snapshot.sh"

echo "[maze-protocol] synchronizing Learner task metric inputs"
RL_TRAINING_WORKSPACE="${workspace_root}" \
  bash "${workspace_root}/rl-learner/proto/gen_proto.sh" task-maze

echo "[maze-protocol] Maze task protocol files synchronized"
