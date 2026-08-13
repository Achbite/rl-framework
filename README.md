# RL Training Framework

简体中文 | [English](README.en.md)

本仓库是本地 RL 框架的统一启动入口。以下命令均从八仓同级目录执行。

## 1. 准备环境

需要 Git、Python 3 和可用的 Docker。macOS 推荐先启动 Colima。

```bash
docker version
python3 --version
```

目录应为：

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

首次生成正式制品前，相关仓库必须是已经审核并提交的 clean Git 保存点。

## 2. 构建制品和镜像

按顺序执行：

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

确认三个运行镜像存在：

```bash
docker image inspect \
  rl-training/learner:training-001 \
  rl-training/aiserver:training-001 \
  rl-training/maze-client:training-001
```

## 3. 验证推理链路

```bash
cd rl-framework
./framework local-test --profile local-test --json
```

命令等待本地推理任务完成并自动清理容器。返回 `ok:true` 才表示该次任务完成。

## 4. 启动本地训练

```bash
./framework local-test --profile training --json
```

Training 没有总样本硬上限。命令在 Learner、AIServer 和 Client 就绪后返回，训练继续运行。
返回 JSON 中包含：

- `run_id`：本次 Run 身份；
- `stop_command`：本次 Run 的精确停止命令；
- `scope: launch_readiness`：只表示链路已就绪，不表示训练质量验收通过。

复制返回的 `stop_command`，在需要结束训练时执行。也可以使用其中的 RunSpec 路径：

```bash
./framework status --run-spec /absolute/path/run-spec.json --json
./framework stop --run-spec /absolute/path/run-spec.json --json
```

标准 Framework Training 不映射宿主 `9005`；运行状态使用 `framework status` 查看。
Learner 单组件开发监控见 [rl-learner/README.md](../rl-learner/README.md)。

## 5. 恢复同一个 Run

正常停止不会删除 Learner 归档或 AIServer 缓存。使用原 RunSpec 再次启动：

```bash
./framework start --run-spec /absolute/path/run-spec.json --json
```

Learner 恢复模型、优化器、RNG、更新数和样本计数；AIServer 校验并复用缓存。不要再次传入初始模型目录。

## 6. 从已有模型开始新 Run

初始目录必须包含四个文件：

```text
000200/
  SaveModel.onnx
  checkpoint.pt
  manifest.json
  metadata.json
```

启动命令：

```bash
./framework local-test \
  --profile training \
  --initial-model-dir /absolute/path/archive/000200 \
  --json
```

新 Run 只继承权重和来源信息；lineage、模型版本、优化器、RNG、更新数和样本计数从 0 开始。

## 7. 本地评测模型

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/archive/000200 \
  --json
```

`--model-path` 可指向保存点目录或其中的 `SaveModel.onnx`。评测属于独立本地模式，不进入训练链路。

## 8. 使用固定 RunSpec

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

RunSpec 必须使用 `schema_version: 2`。Training 固定为 4 Agent；Learner 和 AIServer 的首个参数必须都是 `training`。

## 9. 常见错误

| 错误 | 处理 |
| --- | --- |
| `refusing to create ... from a dirty worktree` | 先审核并提交对应仓库，再生成正式制品 |
| `Contract metric schema artifact is missing` | 重新构建 Contracts 0.11.0，并重新同步下游快照/制品 |
| `runtime artifacts do not match artifact_versions.env` | 重新构建 Sample Pool、Model Distributor，再执行 Learner 的同步脚本 |
| Docker image not found | 重新执行第 2 节的三个镜像构建命令 |
| `address already in use` | 停止占用回放端口 `9004` 的旧任务；Training 内部端口不映射到宿主 |

## 10. 测试 Framework

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## License

[MIT License](LICENSE)
