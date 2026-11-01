# RL Training Framework

[简体中文](README.md) | English

This repository is the consolidated entry point for RL Training Framework usage, current architecture, and read-only diagnostics.

## Table of contents

- [1. Quick start: local three-container training](#1-quick-start-local-three-container-training)
- [2. Static model evaluation](#2-static-model-evaluation)
- [3. Stopping services and local pages](#3-stopping-services-and-local-pages)
- [4. Read-only diagnostics](#4-read-only-diagnostics)
- [5. Formal artifacts and runtime images](#5-formal-artifacts-and-runtime-images)
- [6. Project description and architecture](#6-project-description-and-architecture)
- [7. Related repositories](#7-related-repositories)
- [License](#license)

## 1. Quick start: local three-container training

### 1.1 Prerequisites

- Docker CLI and the Docker daemon are available.
- Keep the following repositories as siblings in one workspace:

```text
RL-Training-Framework/
  rl-contracts/
  rl-sample-pool/
  rl-model-distributor/
  rl-learner/
  rl-aiserver/
  maze-client/
  rl-framework/
```

Each component's `make shell` is a host command. It builds an image and creates a
development container when that container is absent; otherwise it starts the
container only if needed and enters it directly. It neither synchronizes
dependencies or protocols nor force-replaces a resident container after source,
Dockerfile, or environment changes. Run `./build.sh` or `./run.sh` only after
entering the container.

A fresh workspace should first run one independent artifact script. It builds the
task-neutral Training Proto inputs, Sample Pool, and Model Distributor
artifacts, then copies only the two service artifacts required by Learner into
their repository-owned locations:

```bash
cd RL-Training-Framework/rl-framework
bash ./sync_artifacts.sh
```

This script never replaces Learner, AIServer, or Client Proto and does not
start containers or a control plane. AIServer and Client each already own a
compilable Maze Task Proto. Only when intentionally adopting the current Maze
protocol release from Contracts should a developer run the separate command and
review both consumer diffs:

```bash
bash ./sync_maze_protocol.sh
```

The task-neutral Training Proto follows the same repository-owned rule. Only when
intentionally adopting the current training protocol from Contracts, run:

```bash
bash ./sync_training_protocol.sh
```

That command updates only the AIServer and Learner training-protocol inputs. After a
protocol change, run `sync_artifacts.sh` again so Sample Pool and Model Distributor
are rebuilt from the same current Training Proto.

Each component still owns its `make shell`, `./build.sh`, and `./run.sh`
lifecycle. After Dockerfile.dev, toolchain, port, environment, or mount changes,
run that component's explicit `make dev-refresh`. It refreshes only the container
environment and synchronizes neither protocols nor dependencies.

### 1.2 Start Learner

Open the first host terminal:

```bash
cd RL-Training-Framework/rl-learner
make shell

# Run the following command inside the Learner container
./run.sh --config configs/learner_config.yaml
```

Learner is implemented in Python and does not require a separate build. It starts
Sample Pool, Model Distributor, the Learner runtime, and the local Monitor, then
publishes a randomly initialized `models/train/0000000/SaveModel.onnx`. A default
start clears the training directory for this invocation. Supplying existing weights
requires an explicit initial-model argument.

Learner requests batches as soon as it registers the initial model. Once AIServer loads a model and enough samples reach the Pool, Learner can perform its first update.

### 1.3 Start AIServer

After Learner is running, open the second host terminal:

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# Run the following commands inside the AIServer container
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

AIServer downloads and prepares model step 0 through Model Distributor, sends its
exact ACK, and then waits for Client.

### 1.4 Start Maze Client

After AIServer has entered training mode, open the third host terminal:

```bash
cd RL-Training-Framework/maze-client
make shell

# Run the following commands inside the Client container
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

The startup order is fixed:

```text
Learner → AIServer → Client
```

A working chain shows Client action exchanges, AIServer segment closure and sample
delivery, Learner Train Updates and model publication, followed by AIServer download,
ACK, and activation of the new model.

## 2. Static model evaluation

Static evaluation starts only AIServer and Client. Prepare a non-empty, regular,
non-symlink ONNX model file. Its filename and parent-directory layout are not a
cross-component contract.

AIServer terminal:

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# Run the following commands inside the AIServer container
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/model.onnx
```

Client terminal:

```bash
cd RL-Training-Framework/maze-client
make shell

# Run the following commands inside the Client container
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

AIServer pins the selected model and does not connect to Sample Pool or Model
Distributor or emit training samples. Client exits after completing the evaluation
Episode assigned by AIServer. `run.sh` does not start Replay; start the independent,
resident service from inside the Client development container when needed:

```bash
bash ./run_replay.sh
```

From the host, `make replay` starts the same service inside the container.

## 3. Stopping services and local pages

The local training stop order is fixed:

```text
Client → AIServer → Learner
```

Send `Ctrl-C` to the three foreground `run.sh` processes in that order and wait for
each component to drain and return its real exit status. Do not force-remove a
development container while it still owns a business process.

- Learner prints the complete host-reachable Monitor URL during startup.
- Client Replay is available at `http://127.0.0.1:9004/` by default.
- Monitor shows factual Sample, Train Update, model, and Episode metrics for the local
  run. Replay displays the environment sequence recorded by Client.

Replay is independent of the Client business process. Stop it inside the Client
development container with:

```bash
bash ./run_replay.sh -stop
```

From the host, use `make replay-stop`.

## 4. Read-only diagnostics

`framework` only reads source, artifact, model, and image facts. It does not create or
stop containers and does not generate runtime specs, locks, or executions. A passing
check returns `0`, a factual failure returns `1`, and invalid arguments return `2`.

```bash
cd RL-Training-Framework/rl-framework

./framework doctor source --json
./framework doctor artifacts --json
./framework doctor model --model /path/model.onnx --json
./framework doctor images \
  --learner-image <ref> \
  --aiserver-image <ref> \
  --client-image <ref> \
  --json
./framework doctor all \
  --model /path/model.onnx \
  --learner-image <ref> \
  --aiserver-image <ref> \
  --client-image <ref> \
  --json
```

- `source` reads branch, HEAD, and dirty state for the related source repositories.
- `artifacts` directly checks the Sample Pool and Model Distributor binaries and
  configs staged in Learner, including executable permission on both binaries. It
  compares no manifest, package version, platform, source hash, or content hash.
- `model` verifies that the selected model is a non-empty, regular, non-symlink file;
  it does not constrain the filename or training-directory layout.
- `images` only inspects each image's declared component, entrypoint, and default
  arguments.
- `all` aggregates the checks above.

## 5. Formal artifacts and runtime images

Local `make shell` does not require clean source and performs no artifact
synchronization. Before building or starting Learner in a fresh workspace, build
and synchronize its required dependencies from Framework with one command:

```bash
cd RL-Training-Framework/rl-framework
bash ./sync_artifacts.sh
```

It builds the `training` profile from `rl-contracts` as the Proto input for the
current Sample Pool and Model Distributor builds, then builds both services and
stages only their binaries and configs into Learner. The binaries are always
updated. A default config is copied only when the target config is absent, so an
existing local config is preserved. Learner Proto and the
AIServer/Client task protocol are not rewritten. The script downloads no
repository, starts no runtime service, and invokes no control plane. The equivalent
manual flow is:

```bash
contract_dir="$(bash rl-contracts/build_artifact.sh training)"
pool_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" bash rl-sample-pool/build_artifact.sh)"
distributor_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" bash rl-model-distributor/build_artifact.sh)"
bash rl-learner/scripts/sync_runtime_artifacts.sh \
  --sample-pool-dir "${pool_dir}" --model-distributor-dir "${distributor_dir}"
```

Artifacts supplied by another environment may instead be staged by hand. Put
`maze_sample_pool` in `rl-learner/sample-pool/bin/` and
`maze_model_distributor` in `rl-learner/model-distributor/bin/`, and mark both
executable. Only when the corresponding target config is absent, copy
`pool_config.yaml` and `model_distributor_config.yaml` into the two `config/`
directories. The runtime reads no additional artifact manifest and rejects neither
component by repository version, platform string, or hash. Learner-local Proto is
not automatically replaced from Contracts.

AIServer and Client `proto/` directories are repository-owned source. Normal
`make shell`, builds, dependency synchronization, and `make dev-refresh` use them
directly. To intentionally switch to the current Maze release in Contracts, run:

```bash
bash ./sync_maze_protocol.sh
```

That command builds the `task-maze` profile and updates each consumer's Maze Task
Proto and generated C++ bindings according to its own
inventory. It is not a normal build prerequisite. An uncompilable Proto or RPC
wire mismatch is exposed by the real Client-AIServer build/communication path,
not preemptively locked by central-artifact platform/source hashes or an automatic
pull. After any deliberately required sync, run `make shell` in each component and
compile/start inside the containers as described in section 1.

AIServer and Learner also keep their task-neutral Training Proto as repository-owned
source. To deliberately adopt the current Contracts copy, run
`bash ./sync_training_protocol.sh`, followed by `bash ./sync_artifacts.sh` to rebuild
and stage Sample Pool and Model Distributor. One script adopts source protocol files;
the other assembles runtime binaries. Neither makes a version or hash a runtime
admission condition.

After formal artifacts are synchronized, build the three runtime images separately
from the host while using the same project tag:

```bash
(cd rl-learner && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
(cd rl-aiserver && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
(cd maze-client && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
```

The commands produce `rl-training/learner:maze-tag-001`,
`rl-training/aiserver:maze-tag-001`, and `rl-training/maze-client:maze-tag-001`.
A later tuning build may overwrite the same tag. Building does not start containers;
use the component development-container entrypoints in sections 1 and 2 for local
training or evaluation.

## 6. Project description and architecture

### 6.1 Runtime topology

Local training currently consists of three development containers:

```text
┌──────────────────────┐
│ Maze Client container│
│ State and action I/O │
└──────────┬───────────┘
           │ Maze RPC: state / action / Episode lifecycle
           ▼
┌─────────────────────────────────────────────────────┐
│ AIServer container                                  │
│ Inference · Reward · per-Agent R-PIN · GAE          │
│ Async SampleDistributor · model download/Prepare/ACK│
└──────────┬──────────────────────────────▲───────────┘
           │ SamplePoolIngress            │ model distribution
           ▼                              │
┌─────────────────────────────────────────┴───────────┐
│ Learner container                                   │
│ Sample Pool · PPO Learner · Model Distributor       │
│ Local training Monitor                              │
└─────────────────────────────────────────────────────┘
```

Client and AIServer communicate only through the Maze Task Proto present in both
source trees. AIServer, Sample Pool, Learner, and Model Distributor exchange samples,
model metadata, and lifecycle results through the task-neutral Training Proto. The
two protocol surfaces can evolve independently: the three containers need not
originate from one Contracts release, and no protocol ID, repository version, source
hash, build hash, or platform value binds them into one stack identity. Each
component owns its own configuration and runtime
responsibilities. Configuration files provide complete defaults; supported explicit
command-line arguments may override them.

### 6.2 Component responsibilities

| Component | Current responsibility |
| --- | --- |
| Maze Client | Load the map, maintain environment and Agent state, execute actions returned by AIServer, report terminal state, and generate Replay. |
| AIServer | Manage Client sessions, perform ONNX inference and Reward calculation, pin one model per Agent segment, calculate GAE, and pass discrete transitions to SampleDistributor. |
| SampleDistributor | AIServer's asynchronous producer: bounded local queue, batched envelopes, finite retransmission of the same payload, and shutdown drain. |
| Sample Pool | Learner-container sample backend: store processed transitions, evict the oldest READY transitions by FIFO under capacity pressure, and lease uniformly sampled transitions without replacement to Learner. |
| Learner | Acquire training batches from Sample Pool, run PPO forward/backward and optimizer updates, commit sample outcomes, and publish new models. |
| Model Distributor | Register atomically published Learner model packages, provide exact downloads by lineage/step, and record AIServer load ACK facts. |
| Framework | Provide consolidated usage, repository navigation, and read-only doctor commands without participating in runtime orchestration. |

### 6.3 Training data flow

1. Client creates Agents from the environment facts in AIServer
   `OpenSessionRsp.environment` and continuously returns environment state.
   Action masks are an optional task-contract mode: when disabled, no component
   produces or consumes a mask; when enabled, Client reports only the current
   environment availability fact and AIServer and Learner consume it along the same
   sample path.
2. AIServer uses the model pinned by each Agent for inference, returns actions, and
   accumulates action, reward, value, behavior log-probability, and environment result
   into that Agent's continuous segment.
3. Each Agent segment contains at most `TMax=128` transitions. A terminal result, TMax,
   or controlled shutdown closes it. Terminal closure uses `bootstrap_value=0`; TMax
   and controlled closure compute bootstrap with the pinned model.
4. AIServer calculates GAE over the continuous single-Agent segment, then projects it
   into independent processed transitions. Segment close reason, bootstrap, and
   continuity validation remain internal AIServer producer facts and do not enter
   `ProcessedTransition`; Learner receives only PPO fields and real model provenance.
5. SampleDistributor groups transitions into batched envelopes and sends them
   asynchronously to Sample Pool in the Learner container.
6. Sample Pool draws READY transitions uniformly without replacement. Learner uses
   the train-batch, mini-batch, and epoch settings in `configs/learner_config.yaml`,
   then ACKs and removes the batch after a successful commit.
7. After an update, Learner publishes a new `models/train/<model_step>/` package. Model
   Distributor exposes it; AIServer downloads, validates, prepares, and exactly ACKs
   it, then activates it only at each Agent's segment boundary.
8. Facts emitted by Client, AIServer, Sample Pool, Learner, and the model distribution
   path feed the local Monitor, which presents production, acceptance, training, model
   update, latency, and Episode metrics without rewriting their values.

### 6.4 Model directories

Learner archives public training models by `model_step`, left-padding directory names
to at least seven digits:

```text
models/train/
  0000000/
    SaveModel.onnx
    manifest.pb
  0000001/
    SaveModel.onnx
    manifest.pb
  runtime/checkpoints/
    publication-0000001.checkpoint.pt
```

`0000000` is the bootstrap model created for the current Learner invocation. The first
successful Train Update publishes `0000001`. `manifest.pb` contains the public model
metadata. `runtime/checkpoints` is private to the current invocation; it
is not part of a model package and is not an automatic resume entrypoint. The names
and directories above are Learner's current local publication layout. Model
Distributor validates lineage/step, the regular model file selected by registration,
and its declared size; it preserves contiguous download offsets and records load
ACKs. It computes no content hash and does not treat `SaveModel.onnx` or seven-digit
directories as a cross-component contract.

## 7. Related repositories

| Repository | Git URL | Responsibility |
| --- | --- | --- |
| rl-framework | [github.com/Achbite/rl-framework](https://github.com/Achbite/rl-framework) | Consolidated project documentation and read-only diagnostics. |
| rl-contracts | [github.com/Achbite/rl-contracts](https://github.com/Achbite/rl-contracts) | Cross-language Proto sources and generated artifacts. |
| rl-sample-pool | [github.com/Achbite/rl-sample-pool](https://github.com/Achbite/rl-sample-pool) | Sample Pool service and storage. |
| rl-model-distributor | [github.com/Achbite/rl-model-distributor](https://github.com/Achbite/rl-model-distributor) | Model registration, download, and ACK service. |
| rl-learner | [github.com/Achbite/rl-learner](https://github.com/Achbite/rl-learner) | PPO Learner, model publication, and local Monitor. |
| rl-aiserver | [github.com/Achbite/rl-aiserver](https://github.com/Achbite/rl-aiserver) | Model inference, Agent segments, GAE, and sample delivery. |
| maze-client | [github.com/Achbite/maze-client](https://github.com/Achbite/maze-client) | Maze environment execution and Replay. |

## License

[MIT License](LICENSE)

## Build input and cache ownership

Framework organizes the documented sibling layout. Each component owns its build and defaults
to local `build/` outputs. Framework supplies `RL_ARTIFACT_ROOT` for `.workspace/artifacts/`;
`sync_artifacts.sh --development` selects `.workspace/dev-artifacts/` using the same producer scripts.
It invokes Contracts, passes the explicit `CONTRACT_CPP_DIR` to Pool and Distributor, invokes
those builds and stages their returned outputs into Learner. Component builds never update upstream
or synchronize consumers. `sync_maze_protocol.sh` and `sync_training_protocol.sh` explicitly adopt
protocols; consumers then build themselves. Learner images package staged inputs; AIServer and
Client images compile current source with BuildKit compiler caches. Tests remain explicit via `test.sh`.
