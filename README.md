# RL Training Framework

简体中文 | [English](README.en.md)

本仓库只承担项目指引和只读诊断，不再生成运行 spec、lock 或 execution，也不创建、启动、
停止容器。A3 的运行验证由开发者直接启动组件完成。

## 1. A3 与未来 Infra 的边界

项目只有三级运行架构：

1. 静态推理/模型评测：AIServer + Client，AIServer 从 config 默认值或 CLI 覆盖读取一个
   显式 `SaveModel.onnx`。该模式不连接 Learner、Sample Pool 或 Model Distributor。
2. 本地训练：Learner + AIServer + Client。至少完成一次真实 PPO 更新、发布并激活
   `0000001` 后，才构建三个正式 image。
3. Infra 并行训练：未来由 1 个 Learner Pod 和 N 个 Server Pod 组成；每个 Server Pod
   包含 AIServer 与 Client/环境。Task/Run/Pod attempt、自动重启、资源与制品平台均不在
   A3 定义或实现。

Reverb 不属于 A3，也不是进入 Infra 的前置条件。只有未来真实吞吐测量证明当前
QueueBackend 成为瓶颈时，才单独评审是否引入。

## 2. 仓库与构建顺序

八仓必须位于同一目录：

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

Contracts、Sample Pool 与 Model Distributor 的正式 `0.13.0` 制品，以及三个正式 image，
都要求相关源码已经 Review、提交且 clean。按以下顺序构建和同步：

```bash
cd /absolute/path/RL-Training-Framework

(cd rl-contracts && bash build_artifact.sh)
(cd rl-sample-pool && bash build_artifact.sh)
(cd rl-model-distributor && bash build_artifact.sh)
(cd rl-learner && bash scripts/sync_runtime_artifacts.sh)
(cd rl-aiserver && bash scripts/sync_contract_snapshot.sh)
(cd maze-client && bash scripts/sync_contract_snapshot.sh)
```

三个组件统一以 `make shell` 作为宿主机开发入口。该入口自动构建/复用独立开发镜像和
`.workspace/dev-artifacts`，不要求上述正式 artifact。进入容器后直接运行组件自己的
`build.sh`（AIServer/Client）和 `run.sh`，不要在容器内再次调用 Docker wrapper。

## 3. Level 1：静态 evaluation

先准备一个常规、非符号链接且文件名精确为 `SaveModel.onnx` 的模型。它可以是已有产品模型，
也可以是开发者从一次 Learner bootstrap 生成的 `0000000` 中手工复制出的文件；evaluation
运行期间不需要 Learner。

AIServer config 提供默认 workload 与完整模型文件路径，CLI 可显式覆盖：

```bash
bash ./run.sh \
  --config configs/server_config.yaml \
  --workload evaluation \
  --evaluation-model /absolute/path/SaveModel.onnx
```

随后只启动 Client：

```bash
bash ./run.sh --aiserver maze-aiserver:9002
```

通过条件是至少完成一个完整 Episode、动作与 RPC 合法并生成可打开的 Replay；是否通关属于
模型质量轨，不属于系统 Gate。evaluation 只以模型文件 SHA-256 绑定，不携带训练 lineage、
manifest 或 `model_step`。

## 4. Level 2：本地 training

在新的空隔离 workspace 启动 Learner；它会启动 LocalSampleService、ModelDistributor 与
PPO Runtime，并原子发布随机 `0000000`。随后以 training 模式启动 AIServer，再启动 Client：

```bash
# 宿主机：分别在三个终端进入开发容器
(cd rl-learner && make shell)
(cd rl-aiserver && make shell)
(cd maze-client && make shell)

# Learner 容器内
./run.sh --config configs/learner_config.yaml

# AIServer 容器内
./build.sh
./run.sh \
  --config configs/server_config.yaml \
  --workload training \
  --sample-distributor maze-learner:9100 \
  --model-distributor maze-learner:9200

# Client 容器内
./build.sh
./run.sh --config configs/client_config.yaml --aiserver maze-aiserver:9002
```

通过条件是消费一个只含单一 behavior `model_step` 的 batch，完成真实 PPO 事务并发布
`models/train/0000001/`，AIServer exact ACK 后只在 fragment 边界激活 step 1，最后显式排空
停止且三个真实进程退出码均为 0。

## 5. 只读 doctor

所有 doctor 输出固定为：

```json
{"schema_version":1,"ok":true,"scope":"source","data":{},"warnings":[],"error":null}
```

请求范围内事实检查通过返回 `0`，事实不成立返回 `1`，参数错误返回 `2`。

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

- `source` 读取八仓 branch、HEAD 与 dirty 状态，并单列 Infra 为 `legacy/disabled`。
- `artifacts` 校验正式 `0.13.0` Contracts、Sample Pool、Model Distributor manifest、源码
  身份、平台及每个文件 digest；不会生成或同步制品。
- `model` 只检查文件边界、大小和 SHA-256；完整 ONNX 名称、shape、dtype 与有限值推理由
  AIServer 启动门负责。
- `images` 只执行 `docker image inspect`，核对 image ID/digest、稳定 `ENTRYPOINT`、默认
  config `CMD` 与内嵌 source/artifact identity labels；不会创建或运行容器。
- `all` 聚合以上四类检查。

Doctor 只写 stdout/stderr，不创建 lock、receipt、execution、临时冻结目录或容器。

## 6. 正式 image 状态

本地 training 通过后，才使用同一 `stack_source_id` 构建三个不可复用 tag：

```text
rl-training/learner:a3-0.13.0-<stack_source_id前12位>
rl-training/aiserver:a3-0.13.0-<stack_source_id前12位>
rl-training/maze-client:a3-0.13.0-<stack_source_id前12位>
```

三个仓库的正式构建入口会共同计算并强制使用该身份；按顺序执行：

```bash
(cd rl-learner && bash build_image.sh)
(cd rl-aiserver && bash build_image.sh)
(cd maze-client && bash build_image.sh)
```

A3 只 build、inspect 并记录 image ID/digest，不启动正式 image。其证据状态必须写为
`A3-IMAGE BUILD_PASS / RUNTIME_NOT_RUN`，不得写成 `INFRA_READY`。

## 7. 测试

```bash
bash ./test.sh
```

`test.sh` 是本仓库唯一测试入口，只执行当前只读 doctor 链的显式测试模块。

## License

[MIT License](LICENSE)
