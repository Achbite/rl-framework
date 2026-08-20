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

`make shell` is a host command. It prepares development images and artifacts for
the current source automatically, then opens a source-mounted container. Run
`./build.sh` or `./run.sh` only after entering that container.

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

Learner waits for AIServer to issue an exact ACK for the bootstrap model. Remaining
alive at that point is expected.

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

Static evaluation starts only AIServer and Client. Prepare a regular, non-symlink
model file named exactly `SaveModel.onnx`.

AIServer terminal:

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# Run the following commands inside the AIServer container
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx
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
Distributor or emit training samples. After Client completes the evaluation Episode
assigned by AIServer, its business process exits while `run.sh` keeps the Replay HTTP
service available until it receives a stop signal.

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

## 4. Read-only diagnostics

`framework` only reads source, artifact, model, and image facts. It does not create or
stop containers and does not generate runtime specs, locks, or executions. A passing
check returns `0`, a factual failure returns `1`, and invalid arguments return `2`.

```bash
cd RL-Training-Framework/rl-framework

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

- `source` reads branch, HEAD, and dirty state for the related source repositories.
- `artifacts` verifies formal Contracts, Sample Pool, and Model Distributor manifests
  and file digests.
- `model` checks `SaveModel.onnx` file boundaries, size, and SHA-256.
- `images` only inspects image identity, entrypoints, and default arguments.
- `all` aggregates the checks above.

## 5. Formal artifacts and runtime images

Local `make shell` uses isolated development artifacts and does not require clean
source. To build runtime images, first generate and synchronize formal artifacts from
the workspace root in dependency order:

```bash
(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
(cd rl-aiserver && bash scripts/sync_contract_snapshot.sh)
(cd maze-client && bash scripts/sync_contract_snapshot.sh)
```

After formal artifacts are synchronized and the corresponding source repositories are
clean, build the three runtime images from the host:

```bash
(cd rl-learner && bash build_image.sh)
(cd rl-aiserver && bash build_image.sh)
(cd maze-client && bash build_image.sh)
```

The build scripts verify source and artifact identity and print the resulting immutable
image references.

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

The containers share one Contracts identity, while each component owns only its own
configuration and runtime responsibilities. Configuration files provide complete
defaults; explicit component command-line arguments can override supported fields.

### 6.2 Component responsibilities

| Component | Current responsibility |
| --- | --- |
| Maze Client | Load the map, maintain environment and Agent state, execute actions returned by AIServer, report terminal state, and generate Replay. |
| AIServer | Manage Client sessions, perform ONNX inference and Reward calculation, pin one model per Agent segment, calculate GAE, and pass discrete transitions to SampleDistributor. |
| SampleDistributor | AIServer's asynchronous producer: bounded local queue, batched envelopes, idempotent delivery, finite retries, recovery waiting, and shutdown drain. |
| Sample Pool | Learner-container sample backend: store processed transitions, evict the oldest READY transitions by FIFO under capacity pressure, and lease uniformly sampled transitions without replacement to Learner. |
| Learner | Acquire training batches from Sample Pool, run PPO forward/backward and optimizer updates, commit sample outcomes, and publish new models. |
| Model Distributor | Register atomically published Learner model packages and provide exact model download, identity checks, and ACK facts to AIServer. |
| Framework | Provide consolidated usage, repository navigation, and read-only doctor commands without participating in runtime orchestration. |

### 6.3 Training data flow

1. Client creates Agents from the EnvironmentRuntimeSpec assigned by AIServer and
   continuously returns environment state.
2. AIServer uses the model pinned by each Agent for inference, returns actions, and
   accumulates action, reward, value, behavior log-probability, and environment result
   into that Agent's continuous segment.
3. Each Agent segment contains at most `TMax=128` transitions. A terminal result, TMax,
   or controlled shutdown closes it. Terminal closure uses `bootstrap_value=0`; TMax
   and controlled closure compute bootstrap with the pinned model.
4. AIServer calculates GAE over the continuous single-Agent segment, then splits it
   into independent processed transitions. Only the final transition carries an
   explicit segment close reason and bootstrap.
5. SampleDistributor groups transitions into batched envelopes and sends them
   asynchronously to Sample Pool in the Learner container.
6. Sample Pool draws READY transitions uniformly without replacement. By default,
   Learner leases `4096` transitions per Train Update, runs `4` epochs with mini-batches
   of at most `512` transitions, then ACKs and removes the batch after a successful
   commit.
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
    manifest.json
    metadata.json
  0000001/
    SaveModel.onnx
    manifest.json
    metadata.json
```

`0000000` is the bootstrap model created for the current Learner invocation. The first
successful Train Update publishes `0000001`. Model identity, Train Update count, and
public `model_step` remain aligned.

## 7. Related repositories

| Repository | Git URL | Responsibility |
| --- | --- | --- |
| rl-framework | [github.com/Achbite/rl-framework](https://github.com/Achbite/rl-framework) | Consolidated project documentation and read-only diagnostics. |
| rl-contracts | [github.com/Achbite/rl-contracts](https://github.com/Achbite/rl-contracts) | Cross-language Proto, Schema, and contract artifacts. |
| rl-sample-pool | [github.com/Achbite/rl-sample-pool](https://github.com/Achbite/rl-sample-pool) | Sample Pool service and storage. |
| rl-model-distributor | [github.com/Achbite/rl-model-distributor](https://github.com/Achbite/rl-model-distributor) | Model registration, download, and ACK service. |
| rl-learner | [github.com/Achbite/rl-learner](https://github.com/Achbite/rl-learner) | PPO Learner, model publication, and local Monitor. |
| rl-aiserver | [github.com/Achbite/rl-aiserver](https://github.com/Achbite/rl-aiserver) | Model inference, Agent segments, GAE, and sample delivery. |
| maze-client | [github.com/Achbite/maze-client](https://github.com/Achbite/maze-client) | Maze environment execution and Replay. |

## License

[MIT License](LICENSE)
