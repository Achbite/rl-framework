# RL Training Framework

[简体中文](README.md) | English

This repository now provides project guidance and read-only diagnostics only.
It no longer generates runtime specs, locks, or executions and never creates,
starts, or stops containers. Developers run A3 validation directly through the
component entrypoints.

## 1. A3 and future Infra boundary

The project has exactly three runtime levels:

1. Static inference/model evaluation: AIServer + Client. AIServer reads one
   explicit `SaveModel.onnx` from its config default or a CLI override. Learner,
   Sample Pool, and Model Distributor are absent.
2. Local training: Learner + AIServer + Client. Complete at least one real PPO
   update and publish/activate `0000001` before building the three formal images.
3. Infra parallel training: a future topology of one Learner Pod and N Server
   Pods, each Server Pod containing AIServer plus Client/environment. Task, Run,
   Pod attempt, restart, resource, and artifact-platform semantics are outside A3.

Reverb is neither part of A3 nor an Infra prerequisite. It can be reviewed later
only after real throughput measurements show that the current QueueBackend is a
bottleneck.

## 2. Repositories and build order

All eight repositories must be siblings:

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

Formal `0.13.0` Contracts, Sample Pool, and Model Distributor artifacts and all
three formal images require reviewed, committed, clean source. Build and sync in
this order:

```bash
cd /absolute/path/RL-Training-Framework

(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
(cd rl-aiserver && bash scripts/sync_contract_snapshot.sh)
(cd maze-client && bash scripts/sync_contract_snapshot.sh)
```

Use `make shell` as the host-side development entry for all three components. It
builds or reuses an independent development image and `.workspace/dev-artifacts`
without requiring the formal artifacts above. Inside a container, invoke
`build.sh` (AIServer/Client) and `run.sh` directly; never call the Docker wrapper.

## 3. Level 1: static evaluation

Prepare a regular, non-symlink file named exactly `SaveModel.onnx`. It may be an
existing product model or a file manually copied from a Learner bootstrap
`0000000`; Learner is not part of the evaluation topology.

AIServer config supplies the default workload and complete model file path, and
the CLI may override them explicitly:

```bash
bash ./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx
```

Then start only Client:

```bash
bash ./run.sh --aiserver maze-aiserver:9002
```

The gate requires one complete Episode, legal actions and RPCs, and a readable
Replay. Solving the maze is a model-quality result, not a system-gate condition.
Evaluation binds only the model file SHA-256 and carries no training lineage,
manifest, or `model_step`.

## 4. Level 2: local training

Start Learner in a new empty isolated workspace. It starts LocalSampleService,
ModelDistributor, and PPO Runtime and atomically publishes random `0000000`.
Then start AIServer in training mode and connect Client:

```bash
# Host: enter the three development containers in separate terminals
(cd rl-learner && make shell)
(cd rl-aiserver && make shell)
(cd maze-client && make shell)

# Inside the Learner container
./run.sh --config configs/learner_config.yaml

# Inside the AIServer container
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200

# Inside the Client container
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

The gate consumes a batch containing one behavior `model_step`, commits a real
PPO transaction, publishes `models/train/0000001/`, exact-ACKs and activates step
1 only at a fragment boundary, and ends with an explicit drain and zero exit
status from all three real processes.

## 5. Read-only doctor

Every doctor scope emits this stable envelope:

```json
{"schema_version":1,"ok":true,"scope":"source","data":{},"warnings":[],"error":null}
```

Exit status is `0` when all requested facts pass, `1` on a factual check failure,
and `2` for invalid parameters.

```bash
./framework doctor source --json
./framework doctor artifacts --json
./framework doctor model --model /path/SaveModel.onnx --json
./framework doctor images \
  --learner-image <ref> \
  --aiserver-image <ref> \
  --client-image <ref> \
  --json
./framework doctor all \
  --model /path/SaveModel.onnx \
  --learner-image <ref> \
  --aiserver-image <ref> \
  --client-image <ref> \
  --json
```

- `source` reads branch, HEAD, and dirty state for all eight repositories and
  reports Infra separately as `legacy/disabled`.
- `artifacts` verifies the formal `0.13.0` Contracts, Sample Pool, and Model
  Distributor manifests, source identities, platform, and every file digest.
  It never creates or synchronizes an artifact.
- `model` checks file boundaries, size, and SHA-256 only. AIServer's startup gate
  owns complete ONNX name, shape, dtype, and finite-inference validation.
- `images` invokes only `docker image inspect` and verifies image ID/digest, the
  stable `ENTRYPOINT`, default-config `CMD`, and embedded source/artifact identity
  labels. It never creates or runs a container.
- `all` aggregates the four scopes above.

Doctor writes only stdout/stderr and creates no lock, receipt, execution, frozen
input, temporary directory, or container.

## 6. Formal image state

Only after local training passes, build three immutable tags sharing one
`stack_source_id`:

```text
rl-training/learner:a3-0.13.0-<first 12 chars of stack_source_id>
rl-training/aiserver:a3-0.13.0-<first 12 chars of stack_source_id>
rl-training/maze-client:a3-0.13.0-<first 12 chars of stack_source_id>
```

The three formal build entrypoints calculate and enforce the same identity. Run them in order:

```bash
(cd rl-learner && bash build_image.sh)
(cd rl-aiserver && bash build_image.sh)
(cd maze-client && bash build_image.sh)
```

A3 builds and inspects these images but does not start them. Record their state
as `A3-IMAGE BUILD_PASS / RUNTIME_NOT_RUN`, never `INFRA_READY`.

## 7. Tests

```bash
bash ./test.sh
```

`test.sh` is this repository's only test entrypoint and runs only the explicit
modules for the current read-only doctor path.

## License

[MIT License](LICENSE)
