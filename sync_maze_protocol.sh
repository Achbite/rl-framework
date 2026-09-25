#!/usr/bin/env bash
set -euo pipefail
framework_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd "${framework_root}/.." && pwd -P)"
artifact_dir="$(RL_ARTIFACT_ROOT="${workspace_root}/.workspace/artifacts/rl-contracts" \
    bash "${workspace_root}/rl-contracts/build_artifact.sh" task-maze)"
bash "${workspace_root}/rl-aiserver/scripts/sync_contract_snapshot.sh" "${artifact_dir}"
bash "${workspace_root}/maze-client/scripts/sync_contract_snapshot.sh" "${artifact_dir}"
echo "[task-maze] protocol inputs synchronized"
