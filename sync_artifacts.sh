#!/usr/bin/env bash
set -euo pipefail
framework_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_root="$(cd "${framework_root}/.." && pwd -P)"
artifact_root="${workspace_root}/.workspace/artifacts"
build_args=()
if [ "${1:-}" = "--development" ]; then
    artifact_root="${workspace_root}/.workspace/dev-artifacts"
    build_args+=(--development)
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: bash sync_artifacts.sh [--development]" >&2
    exit 2
fi

echo "[artifacts] building task-neutral training protocol bindings"
contract_dir="$(RL_ARTIFACT_ROOT="${artifact_root}/rl-contracts" \
    bash "${workspace_root}/rl-contracts/build_artifact.sh" ${build_args[@]+"${build_args[@]}"} training)"
echo "[artifacts] building rl-sample-pool"
sample_pool_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" RL_ARTIFACT_ROOT="${artifact_root}/rl-sample-pool" \
    bash "${workspace_root}/rl-sample-pool/build_artifact.sh" ${build_args[@]+"${build_args[@]}"})"
echo "[artifacts] building rl-model-distributor"
model_distributor_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" RL_ARTIFACT_ROOT="${artifact_root}/rl-model-distributor" \
    bash "${workspace_root}/rl-model-distributor/build_artifact.sh" ${build_args[@]+"${build_args[@]}"})"
echo "[artifacts] synchronizing Learner runtime dependencies"
bash "${workspace_root}/rl-learner/scripts/sync_runtime_artifacts.sh" \
    --sample-pool-dir "${sample_pool_dir}" --model-distributor-dir "${model_distributor_dir}"
