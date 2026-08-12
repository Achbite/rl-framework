# RL Training Framework

简体中文 | [English](README.en.md)

用于统一启动和检查本地 Learner、AIServer 与 Maze Client。组件镜像由各仓库独立构建，本仓只负责本地 Docker 编排。

## 快速开始

运行推理通信与回放测试：

```bash
./framework local-test --profile local-test --json
```

运行 4-Agent 本地训练闭环：

```bash
./framework local-test --profile training --json
```

组件默认清理各自的 `local-train` 并从零训练。使用外部完整 checkpoint 初始化时：

```bash
./framework local-test \
  --profile training \
  --initial-checkpoint /absolute/path/checkpoint_v000200.pt \
  --json
```

使用指定保存点运行本地验证。目录中模型文件名固定为
`SaveModel.onnx`，AIServer 从自身配置的 `model.evaluation_dir` 读取：

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/archive/000200 \
  --json
```

使用已有 RunSpec：

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

平台 `start` 在 Learner/AIServer Docker Healthcheck 通过，且 Client 已完成
OpenSession、Init、首个 Training Assignment 校验并发布会话策略后返回，
训练容器继续运行；可选指标读取失败不阻止该 launch/readiness 结果。`status` 持续投影运行态。
`start/stop` 以同一 Run 生命周期锁串行；`stop` 返回时其启动与清理已形成最终停止状态，
不会再被迟到的 ready 覆盖。`local-test --profile training` 使用同一套
无界 Training 语义：核心链路就绪后立即返回 `launch_readiness` 回执，训练继续运行，
必须使用回执中的 `stop_command` 或显式 `framework stop` 执行受控停止。该启动回执不表示
Client workload 完成、样本守恒、PPO 质量或模型制品已经验收通过。

默认镜像 tag 为 `training-001`。RunSpec v2 分别保存三个角色的镜像仓库、tag、启动命令、参数与 CPU/内存/GPU 限制，并要求 AIServer 与 Client 同机部署。Framework 为每个 Run 创建独立 Docker 网络和组件目录，仅在 Docker 标签和外部回执中保存 Task/Run 身份。

Training 的 Framework 标准入口不依赖、也不固定映射可选指标端口 `9005`；
`9002/9100/9200` 仅在任务网络内可见。推理和模型评估只映射回放端口 `9004`。

## 测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
