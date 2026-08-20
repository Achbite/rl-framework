# RL Training Framework

简体中文 | [English](README.en.md)

本仓库是 RL Training Framework 的综合使用入口，提供当前项目的启动说明、架构说明和只读诊断工具。

## 目录

- [1. 快速开始：本地三容器训练](#1-快速开始本地三容器训练)
- [2. 静态模型评测](#2-静态模型评测)
- [3. 停止服务与本地页面](#3-停止服务与本地页面)
- [4. 只读诊断](#4-只读诊断)
- [5. 正式制品与运行镜像](#5-正式制品与运行镜像)
- [6. 项目说明与架构](#6-项目说明与架构)
- [7. 关联仓库](#7-关联仓库)
- [License](#license)

## 1. 快速开始：本地三容器训练

### 1.1 前置条件

- Docker CLI 和 Docker daemon 可用。
- 下列仓库位于同一个工作区目录中：

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

`make shell` 是宿主机入口，会自动准备当前源码对应的开发镜像和开发制品，并进入源码挂载容器。
进入容器后再执行 `./build.sh` 或 `./run.sh`。

### 1.2 启动 Learner

打开第一个宿主终端：

```bash
cd RL-Training-Framework/rl-learner
make shell

# 以下命令在 Learner 容器内执行
./run.sh --config configs/learner_config.yaml
```

Learner 使用 Python，无需单独编译。启动后会依次运行 Sample Pool、Model Distributor、Learner
训练进程和本地 Monitor，并发布随机初始化的 `models/train/0000000/SaveModel.onnx`。默认启动会清理
本次训练目录；如需使用已有权重，必须通过显式初始模型参数启动。

Learner 会等待 AIServer 对 bootstrap 模型完成精确 ACK，等待期间保持运行属于正常状态。

### 1.3 启动 AIServer

确认 Learner 已启动后，打开第二个宿主终端：

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# 以下命令在 AIServer 容器内执行
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200
```

AIServer 会从 Model Distributor 下载并准备 step 0 模型，完成精确 ACK 后等待 Client 连接。

### 1.4 启动 Maze Client

确认 AIServer 已进入 training 模式后，打开第三个宿主终端：

```bash
cd RL-Training-Framework/maze-client
make shell

# 以下命令在 Client 容器内执行
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

训练启动顺序固定为：

```text
Learner → AIServer → Client
```

链路正常时可以观察到 Client 动作往返、AIServer segment 封口与样本发送、Learner Train Update、
新模型发布，以及 AIServer 下载、ACK 和激活新模型的日志。

## 2. 静态模型评测

静态评测只启动 AIServer 和 Client。准备一个常规、非符号链接且文件名精确为
`SaveModel.onnx` 的模型文件。

AIServer 终端：

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# 以下命令在 AIServer 容器内执行
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx
```

Client 终端：

```bash
cd RL-Training-Framework/maze-client
make shell

# 以下命令在 Client 容器内执行
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

AIServer 会固定加载指定模型，不连接 Sample Pool 或 Model Distributor，也不发送训练样本。Client
完成 AIServer 下发的一次 evaluation Episode 后退出业务进程，`run.sh` 会继续保留 Replay HTTP
服务，直到收到停止信号。

## 3. 停止服务与本地页面

本地训练的停止顺序固定为：

```text
Client → AIServer → Learner
```

依次向三个前台 `run.sh` 发送 `Ctrl-C`，等待各组件完成排空并返回真实退出码。不要直接强制删除仍在运行
业务进程的开发容器。

- Learner 启动日志会打印宿主机可访问的 Monitor 完整地址。
- Client Replay 默认地址为 `http://127.0.0.1:9004/`。
- Monitor 展示本次本地训练的真实 Sample、Train Update、模型和 Episode 指标；Replay 用于查看 Client
  记录的环境过程。

## 4. 只读诊断

`framework` 只读取源码、制品、模型和镜像事实，不创建或启停容器，也不生成运行 spec、lock 或
execution。检查成功返回 `0`，事实检查失败返回 `1`，参数错误返回 `2`。

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

- `source`：读取关联源码仓库的 branch、HEAD 和 dirty 状态。
- `artifacts`：校验 Contracts、Sample Pool 和 Model Distributor 的正式 manifest 与文件 digest。
- `model`：检查 `SaveModel.onnx` 的文件边界、大小和 SHA-256。
- `images`：只执行 image inspect，核对镜像身份、入口和默认参数。
- `all`：聚合以上检查。

## 5. 正式制品与运行镜像

本地 `make shell` 使用独立的开发制品，不要求源码 clean。构建正式运行镜像时，先在工作区根目录
按依赖顺序生成并同步正式制品：

```bash
(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
(cd rl-aiserver && bash scripts/sync_contract_snapshot.sh)
(cd maze-client && bash scripts/sync_contract_snapshot.sh)
```

正式制品同步完成且对应源码仓库 clean 后，在宿主机分别构建三个运行镜像：

```bash
(cd rl-learner && bash build_image.sh)
(cd rl-aiserver && bash build_image.sh)
(cd maze-client && bash build_image.sh)
```

构建脚本会校验源码与制品身份，并打印生成的不可复用镜像引用。

## 6. 项目说明与架构

### 6.1 运行拓扑

当前本地训练由三个开发容器组成：

```text
┌──────────────────────┐
│ Maze Client 容器     │
│ 环境状态与动作执行   │
└──────────┬───────────┘
           │ Maze RPC：状态 / 动作 / Episode 生命周期
           ▼
┌─────────────────────────────────────────────────────┐
│ AIServer 容器                                      │
│ 模型推理 · Reward · per-Agent R-PIN · GAE          │
│ 异步 SampleDistributor · 模型下载 / Prepare / ACK  │
└──────────┬──────────────────────────────▲───────────┘
           │ SamplePoolIngress            │ 模型分发
           ▼                              │
┌─────────────────────────────────────────┴───────────┐
│ Learner 容器                                       │
│ Sample Pool · PPO Learner · Model Distributor      │
│ 本地训练 Monitor                                   │
└─────────────────────────────────────────────────────┘
```

三个容器共享一套 Contracts 身份，但各自只拥有本组件的配置和运行职责。配置文件提供完整默认值；
组件支持的显式命令参数可以覆盖对应配置项。

### 6.2 组件职责

| 组件 | 当前职责 |
| --- | --- |
| Maze Client | 加载地图、维护环境与 Agent 状态、执行 AIServer 下发的动作、报告终局状态并生成 Replay。 |
| AIServer | 管理 Client session，执行 ONNX 推理与 Reward 计算；为每个 Agent 固定一个 segment 使用的模型，计算 GAE，并将离散 transition 交给 SampleDistributor。 |
| SampleDistributor | AIServer 内的异步生产端；负责有界本地队列、批量 envelope、幂等发送、有限重试、恢复等待和停止排空。 |
| Sample Pool | Learner 容器内的样本后端；保存 processed transition，容量压力下按 FIFO 淘汰最老 READY 样本，并向 Learner 随机无放回交付租约批次。 |
| Learner | 从 Sample Pool 取得训练批次，执行 PPO forward/backward 与 optimizer update，提交样本处理结果并发布新模型。 |
| Model Distributor | 注册 Learner 原子发布的模型包，向 AIServer 提供精确模型下载、身份校验和 ACK 事实。 |
| Framework | 提供整个项目的使用说明、仓库导航和只读 doctor，不参与训练运行编排。 |

### 6.3 训练数据链

1. Client 根据 AIServer 下发的 EnvironmentRuntimeSpec 创建 Agent，并持续回传环境状态。
2. AIServer 使用当前 Agent 所固定的模型执行推理，返回动作，并把 action、reward、value、
   behavior log-prob 和环境结果累计到该 Agent 的连续 segment。
3. 每个 Agent 的 segment 最多累计 `TMax=128` 个 transition；terminal、TMax 或受控关闭都会封口。
   terminal 边界使用 `bootstrap_value=0`，TMax 和受控关闭使用 pinned model 计算 bootstrap。
4. AIServer 在同一 Agent 的连续 segment 上计算 GAE，再把 segment 离散为独立 processed
   transition；只有最后一个 transition 携带明确的 segment close reason 和 bootstrap。
5. SampleDistributor 将 transition 组成批量 envelope，异步发送到 Learner 容器内的 Sample Pool。
6. Sample Pool 对 READY transition 进行随机无放回抽取。默认 Learner 每次 Train Update 租用
   `4096` 个 transition，以最大 `512` 个 transition 的 mini-batch 运行 `4` 个 epoch；成功提交后
   ACK 并移除本批样本。
7. Learner 完成一次更新后发布新的 `models/train/<model_step>/` 模型包。Model Distributor 暴露
   已发布模型，AIServer 下载、校验、Prepare 并精确 ACK，只在各 Agent 的 segment 边界激活新模型。
8. Client、AIServer、Sample Pool、Learner 和模型分发链产生的事实进入本地 Monitor，展示真实的
   生产、接受、训练、模型更新、延迟和 Episode 指标。

### 6.4 模型目录

Learner 公开的训练模型按 `model_step` 归档，目录名至少七位左补零：

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

`0000000` 是本次 Learner 启动生成的 bootstrap 模型；第一次成功的 Train Update 发布
`0000001`。模型身份、训练更新计数和公开 `model_step` 保持一致。

## 7. 关联仓库

| 仓库 | Git 地址 | 职责 |
| --- | --- | --- |
| rl-framework | [github.com/Achbite/rl-framework](https://github.com/Achbite/rl-framework) | 项目综合说明与只读诊断。 |
| rl-contracts | [github.com/Achbite/rl-contracts](https://github.com/Achbite/rl-contracts) | 跨语言 Proto、Schema 与契约制品。 |
| rl-sample-pool | [github.com/Achbite/rl-sample-pool](https://github.com/Achbite/rl-sample-pool) | Sample Pool 服务与样本存储。 |
| rl-model-distributor | [github.com/Achbite/rl-model-distributor](https://github.com/Achbite/rl-model-distributor) | 模型注册、下载与 ACK 服务。 |
| rl-learner | [github.com/Achbite/rl-learner](https://github.com/Achbite/rl-learner) | PPO Learner、模型发布与本地 Monitor。 |
| rl-aiserver | [github.com/Achbite/rl-aiserver](https://github.com/Achbite/rl-aiserver) | 模型推理、Agent segment、GAE 与样本发送。 |
| maze-client | [github.com/Achbite/maze-client](https://github.com/Achbite/maze-client) | 迷宫环境执行与 Replay。 |

## License

[MIT License](LICENSE)
