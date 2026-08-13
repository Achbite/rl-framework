# RL Training Framework

[简体中文](README.md) | English

This repository is the single local entry point for the RL framework. Run the commands below from the directory that contains all eight repositories.

## 1. Prerequisites

Install Git, Python 3, and Docker. On macOS, start Colima first.

```bash
docker version
python3 --version
```

Expected layout:

```text
RL-Training-Framework/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
  rl-framework/
  rl-infra/
```

Repositories must be reviewed, committed, and clean before their first immutable artifact is created.

## 2. Build artifacts and images

Run these commands in order:

```bash
cd /absolute/path/RL-Training-Framework

(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)

(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
(cd rl-learner && RL_LEARNER_IMAGE_TAG=training-001 bash build_image.sh)

(cd rl-aiserver && bash scripts/sync_contract_snapshot.sh)
(cd rl-aiserver && RL_AISERVER_IMAGE_TAG=training-001 bash build_image.sh)

(cd maze-client && bash scripts/sync_contract_snapshot.sh)
(cd maze-client && RL_CLIENT_IMAGE_TAG=training-001 bash build_image.sh)
```

Verify the runtime images:

```bash
docker image inspect \
  rl-training/learner:training-001 \
  rl-training/aiserver:training-001 \
  rl-training/maze-client:training-001
```

## 3. Verify the inference chain

```bash
cd rl-framework
./framework local-test --profile local-test --json
```

The command waits for the inference workload to finish and then removes its containers. Only `ok:true` means that workload completed.

## 4. Start local training

```bash
./framework local-test --profile training --json
```

Training has no total-sample hard limit. The command returns after Learner, AIServer, and Client become ready while training continues. The JSON response includes:

- `run_id`, the Run identity;
- `stop_command`, the exact command that stops this Run;
- `scope: launch_readiness`, which means the chain is ready, not that training quality passed.

Copy and run `stop_command` when training should end. You can also use its RunSpec path:

```bash
./framework status --run-spec /absolute/path/run-spec.json --json
./framework stop --run-spec /absolute/path/run-spec.json --json
```

Standard Framework Training does not publish host port `9005`; use `framework status` for runtime state. See [rl-learner/README.en.md](../rl-learner/README.en.md) for the Learner-only development monitor.

## 5. Resume the same Run

A normal stop preserves the Learner archive and AIServer cache. Restart with the original RunSpec:

```bash
./framework start --run-spec /absolute/path/run-spec.json --json
```

Learner restores the model, optimizer, RNG, update count, and sample count. AIServer validates and reuses its cache. Do not pass the initial model directory again.

## 6. Start a new Run from an existing model

The initial directory must contain exactly four files:

```text
000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

Start the Run:

```bash
./framework local-test \
  --profile training \
  --initial-model-dir /absolute/path/archive/000200 \
  --json
```

The new Run inherits only model weights and provenance. Its lineage, model version, optimizer, RNG, update count, and sample count start from zero.

## 7. Evaluate a local model

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/archive/000200 \
  --json
```

`--model-path` may name the savepoint directory or its `SaveModel.onnx`. Evaluation is a separate local mode and never enters the training chain.

## 8. Use a fixed RunSpec

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

RunSpec must use `schema_version: 2`. Training uses exactly four Agents, and the first Learner and AIServer arguments must both be `training`.

## 9. Troubleshooting

| Error | Fix |
| --- | --- |
| `refusing to create ... from a dirty worktree` | Review and commit that repository before creating the immutable artifact |
| `Contract metric schema artifact is missing` | Rebuild Contracts 0.11.0 and resync downstream snapshots and artifacts |
| `runtime artifacts do not match artifact_versions.env` | Rebuild Sample Pool and Model Distributor, then rerun the Learner sync script |
| Docker image not found | Repeat the three image builds in section 2 |
| `address already in use` | Stop the old workload using replay port `9004`; Training ports stay inside the task network |

## 10. Test Framework

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## License

[MIT License](LICENSE)
