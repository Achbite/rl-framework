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

各组件的 `make shell` 都是宿主机入口：对应开发容器不存在时构建镜像并创建容器；容器存在时只在
必要时启动，然后直接进入源码挂载容器。它不自动同步依赖或协议，也不因源码、Dockerfile 或
环境变化强制替换常驻容器。进入容器后再执行 `./build.sh` 或 `./run.sh`。

全新工作区先运行一次独立制品脚本。它生成 task-neutral Training Proto 编译输入、Sample Pool 与
Model Distributor 正式制品，并只把 Learner 必需的两个服务制品复制到其仓库约定位置：

```bash
cd RL-Training-Framework/rl-framework
bash ./sync_artifacts.sh
```

该脚本不覆盖 Learner、AIServer 或 Client 的 Proto，不启动容器或控制面。AIServer 与
Client 已各自持有可编译的 Maze Task Proto；只有开发者明确决定采用 Contracts 仓当前 Maze 协议
release 时才执行以下独立命令，并审查两个消费仓的 diff：

```bash
bash ./sync_maze_protocol.sh
```

task-neutral Training Proto 也按相同原则由消费仓持有；只有明确采用 Contracts 仓当前训练协议时才执行：

```bash
bash ./sync_training_protocol.sh
```

该命令只更新 AIServer 与 Learner 的训练协议输入。协议变化后再运行 `sync_artifacts.sh`，即可让
Sample Pool 与 Model Distributor 使用同一份当前 Training Proto 重新编译。

同步完成后仍由各组件自己的 `make shell`、`./build.sh` 和 `./run.sh` 负责开发容器与业务进程。
Dockerfile.dev、工具链、端口、环境变量或挂载发生变化时，显式执行组件仓的 `make dev-refresh`；
该命令只刷新容器环境，也不会同步协议或依赖。

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

Learner 注册初始模型后便向 Pool 取批。AIServer 启动、加载模型并送入足够样本后，Learner 即可完成首个更新。

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

静态评测只启动 AIServer 和 Client。准备一个非空、常规且非符号链接的 ONNX 模型文件；文件名和
上层目录不是跨组件合同。

AIServer 终端：

```bash
cd RL-Training-Framework/rl-aiserver
make shell

# 以下命令在 AIServer 容器内执行
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/model.onnx
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
完成 AIServer 下发的一次 evaluation Episode 后退出业务进程。`run.sh` 不启动 Replay；需要查看回放时，
在 Client 开发容器内单独启动常驻服务：

```bash
bash ./run_replay.sh
```

从宿主机使用 `make replay` 启动同一容器内服务。

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

Replay 与 Client 业务进程相互独立；停止 Replay 时在 Client 开发容器中执行：

```bash
bash ./run_replay.sh -stop
```

从宿主机使用 `make replay-stop`。

## 4. 只读诊断

`framework` 只读取源码、制品、模型和镜像事实，不创建或启停容器，也不生成运行 spec、lock 或
execution。检查成功返回 `0`，事实检查失败返回 `1`，参数错误返回 `2`。

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

- `source`：读取关联源码仓库的 branch、HEAD 和 dirty 状态。
- `artifacts`：直接检查 Learner 装配目录中的 Sample Pool、Model Distributor 二进制与配置是否存在，
  并确认两个二进制可执行；不比较 manifest、包版本、平台、源码或内容哈希。
- `model`：检查所选模型是非空、常规且非符号链接的文件，不约束文件名或训练目录布局。
- `images`：只执行 image inspect，核对各镜像自身声明的组件、入口和默认参数。
- `all`：聚合以上检查。

## 5. 正式制品与运行镜像

本地 `make shell` 不要求源码 clean，也不执行制品同步。全新工作区构建或启动 Learner 前，可从
Framework 仓一次完成其必需依赖的顺序构建与同步：

```bash
cd RL-Training-Framework/rl-framework
bash ./sync_artifacts.sh
```

它会构建 `rl-contracts` 的 `training` profile，作为 Sample Pool 与 Model Distributor 本次编译的
Proto 输入，再依次构建这两个服务，最后只把它们的二进制与配置同步到 Learner。同步始终更新二进制；
目标配置不存在时才复制默认配置，已存在的本地配置不会被覆盖。
Learner 的 Proto、AIServer/Client 的任务协议均不会被改写。脚本不下载仓库、不启动运行服务，
也不调用控制面。以下命令是完全等价的手动流程：

```bash
contract_dir="$(bash rl-contracts/build_artifact.sh training)"
pool_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" bash rl-sample-pool/build_artifact.sh)"
distributor_dir="$(CONTRACT_CPP_DIR="${contract_dir}/cpp" bash rl-model-distributor/build_artifact.sh)"
bash rl-learner/scripts/sync_runtime_artifacts.sh \
  --sample-pool-dir "${pool_dir}" --model-distributor-dir "${distributor_dir}"
```

已经由其他环境提供制品时，也可以手工完成同一装配：将 `maze_sample_pool` 放到
`rl-learner/sample-pool/bin/`，将 `maze_model_distributor` 放到
`rl-learner/model-distributor/bin/`，并赋予可执行权限；仅在对应目标配置不存在时，分别复制
`pool_config.yaml` 与 `model_distributor_config.yaml` 到两个 `config/` 目录。运行链不读取额外
manifest，也不按仓库版本、平台字符串或哈希拒绝这些组件。Learner 自己的 Proto 不从 Contracts
自动覆盖。

AIServer 与 Client 的 `proto/` 属于各自源码，正常 `make shell`、构建、依赖同步和
`make dev-refresh` 都直接使用它们。若开发者主动切换到 Contracts 仓当前 Maze release，独立执行：

```bash
bash ./sync_maze_protocol.sh
```

该命令构建 `task-maze` profile，再按两个消费仓各自清单更新 Maze Task Proto 与 SDK；任务 C++
bindings 在各自构建目录生成；它不是正常构建前置步骤。Proto 不可编译或 RPC wire 不一致会由真实
Client↔AIServer 编译/通信失败暴露，不再用中央 artifact 的平台、源码哈希或自动回拉行为提前锁死。
完成所需的显式同步后，即可进入三个组件仓分别执行 `make shell`，再在容器内按第 1 节编译和启动。

AIServer 与 Learner 的 task-neutral Training Proto 同样不自动同步。主动采用 Contracts 仓当前版本时，
执行 `bash ./sync_training_protocol.sh`；随后执行 `bash ./sync_artifacts.sh` 重新编译并装配 Sample Pool
与 Model Distributor。两个脚本分别处理源码协议采用和运行二进制装配，不以版本或哈希做运行准入。

正式制品同步完成后，在宿主机使用同一个项目 tag 分别构建三个运行镜像：

```bash
(cd rl-learner && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
(cd rl-aiserver && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
(cd maze-client && RL_PROJECT_IMAGE_TAG=maze-tag-001 bash build_image.sh)
```

三个命令分别生成 `rl-training/learner:maze-tag-001`、
`rl-training/aiserver:maze-tag-001` 与 `rl-training/maze-client:maze-tag-001`。同名 tag 允许由后续微调
构建直接覆盖；构建本身不会启动容器。需要本地训练或评测时，继续使用第 1、2 节的组件开发容器入口。

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

Client 与 AIServer 只按双方源码中共同采用的 Maze Task Proto 通信；AIServer、Sample Pool、Learner
与 Model Distributor 按 task-neutral Training Proto 传输样本、模型元数据和生命周期结果。两个协议面
可以独立演进，不要求三个容器来自同一个 Contracts release，也不通过协议 ID、仓库版本、源码哈希、
构建哈希或平台值绑定为一个整体。各组件只拥有本组件的配置和运行职责；
配置文件提供完整默认值，组件支持的显式命令参数可以覆盖对应配置项。

### 6.2 组件职责

| 组件              | 当前职责                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Maze Client       | 加载地图、维护环境与 Agent 状态、执行 AIServer 下发的动作、报告终局状态并生成 Replay。                                                              |
| AIServer          | 管理 Client session，执行 ONNX 推理与 Reward 计算；为每个 Agent 固定一个 segment 使用的模型，计算 GAE，并将离散 transition 交给 SampleDistributor。 |
| SampleDistributor | AIServer 内的异步生产端；负责有界本地队列、批量 envelope、同一 payload 的有限重发和停止排空。                                                       |
| Sample Pool       | Learner 容器内的样本后端；保存 processed transition，容量压力下按 FIFO 淘汰最老 READY 样本，并向 Learner 随机无放回交付租约批次。                   |
| Learner           | 从 Sample Pool 取得训练批次，执行 PPO forward/backward 与 optimizer update，提交样本处理结果并发布新模型。                                          |
| Model Distributor | 注册 Learner 原子发布的模型包，按 lineage/step 提供精确模型下载，并记录 AIServer 的加载 ACK 事实。                                                  |
| Framework         | 提供整个项目的使用说明、仓库导航和只读 doctor，不参与训练运行编排。                                                                                 |

### 6.3 训练数据链

1. Client 根据 AIServer `OpenSessionRsp.environment` 下发的环境事实创建 Agent，并持续回传环境状态。
   action mask 是任务合同中的可选模式：关闭时不生成或消费遮罩；开启时 Client 只回传当前环境的
   可用动作事实，AIServer 与 Learner 沿同一条样本链消费它。
2. AIServer 使用当前 Agent 所固定的模型执行推理，返回动作，并把 action、reward、value、
   behavior log-prob 和环境结果累计到该 Agent 的连续 segment。
3. 每个 Agent 的 segment 最多累计 `TMax=128` 个 transition；terminal、TMax 或受控关闭都会封口。
   terminal 边界使用 `bootstrap_value=0`，TMax 和受控关闭使用 pinned model 计算 bootstrap。
4. AIServer 在同一 Agent 的连续 segment 上计算 GAE，再把 segment 投影为独立 processed
   transition。segment close reason、bootstrap 和连续性校验是 AIServer 内部 producer 事实，不进入
   `ProcessedTransition`；Learner 只接收 PPO 所需字段及真实模型 provenance。
5. SampleDistributor 将 transition 组成批量 envelope，异步发送到 Learner 容器内的 Sample Pool。
6. Sample Pool 对 READY transition 进行随机无放回抽取。Learner 按
   `configs/learner_config.yaml` 的 train batch、mini-batch 与 epoch 配置执行 PPO；成功提交后 ACK
   并移除本批样本。
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
    manifest.pb
  0000001/
    SaveModel.onnx
    manifest.pb
  runtime/checkpoints/
    publication-0000001.checkpoint.pt
```

`0000000` 是本次 Learner 启动生成的 bootstrap 模型；第一次成功的 Train Update 发布
`0000001`。`manifest.pb` 是公开模型包的元数据；`runtime/checkpoints` 是本次训练的私有
checkpoint，不属于模型包，也不是后续训练的自动恢复入口。上述文件名和目录是 Learner 当前的
本地发布布局；Model Distributor 检查注册请求中的 lineage/step、常规模型文件和声明大小，下载时
保持连续 offset，并记录加载 ACK；它不计算内容哈希，也不把 `SaveModel.onnx` 或七位目录名解释为
跨组件合同。

## 7. 关联仓库

| 仓库                 | Git 地址                                                                                  | 职责                                      |
| -------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------- |
| rl-framework         | [github.com/Achbite/rl-framework](https://github.com/Achbite/rl-framework)                 | 项目综合说明与只读诊断。                  |
| rl-contracts         | [github.com/Achbite/rl-contracts](https://github.com/Achbite/rl-contracts)                 | 跨语言 Proto 源码与生成制品。             |
| rl-sample-pool       | [github.com/Achbite/rl-sample-pool](https://github.com/Achbite/rl-sample-pool)             | Sample Pool 服务与样本存储。              |
| rl-model-distributor | [github.com/Achbite/rl-model-distributor](https://github.com/Achbite/rl-model-distributor) | 模型注册、下载与 ACK 服务。               |
| rl-learner           | [github.com/Achbite/rl-learner](https://github.com/Achbite/rl-learner)                     | PPO Learner、模型发布与本地 Monitor。     |
| rl-aiserver          | [github.com/Achbite/rl-aiserver](https://github.com/Achbite/rl-aiserver)                   | 模型推理、Agent segment、GAE 与样本发送。 |
| maze-client          | [github.com/Achbite/maze-client](https://github.com/Achbite/maze-client)                   | 迷宫环境执行与 Replay。                   |

## License

[MIT License](LICENSE)

## 构建输入与产物更新

Framework 是组织仓，脚本只适用于 README 约定的同级仓库布局。各仓拥有自己的 build，默认产物在本仓
`build/`；Framework 显式传入 `RL_ARTIFACT_ROOT` 将输出组织到 `.workspace/artifacts/`。
`sync_artifacts.sh --development` 使用 `.workspace/dev-artifacts/`，两种通道共享各生产者的构建实现。

`sync_artifacts.sh` 调用 Contracts 的 Training 构建入口，将返回的路径作为 Pool / Distributor 的
`CONTRACT_CPP_DIR`，调用两仓 build，再调用 Learner 的显式装配入口。组件 build 不自动更新上游或同步消费仓。
`sync_maze_protocol.sh` / `sync_training_protocol.sh` 分别显式采纳任务 / 训练协议；同步之后由消费仓自行 build。
Learner 镜像只打包已装配的输入；AIServer / Client 镜像在各自源码上下文内编译并复用 BuildKit 编译缓存。
要证明工作区当前源码进入镜像，先显式同步需要更新的协议和服务，再执行各仓 build/image 入口。

开发容器 build 使用按工具链隔离的持久目录，构建成功后更新本仓 run 所用路径。各仓测试仍只通过根 `test.sh` 显式运行。
