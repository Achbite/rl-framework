# RL Training Framework

简体中文 | [English](README.en.md)

用于统一启动和检查本地 Learner、AIServer 与 Maze Client。组件镜像由各仓库独立构建，本仓只负责本地 Docker 编排。

## 快速开始

运行推理通信与回放测试：

```bash
./framework local-test --profile inference-smoke --json
```

运行 4-Agent 本地训练闭环：

```bash
./framework local-test --profile training --json
```

使用指定 ONNX 模型运行本地验证：

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/model.onnx \
  --json
```

使用已有 RunSpec：

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

默认镜像 tag 为 `training-001`。运行数据保存在项目根目录的 `.workspace/runtime`，训练指标端口为 `9005`，本地回放端口为 `9004`。

## 测试

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
