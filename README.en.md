# RL Training Framework

English | [简体中文](README.md)

Starts and checks the local Learner, AIServer, and Maze Client as one workflow. Component images are built by their own repositories; this repository only provides local Docker orchestration.

## Quick Start

Run the inference communication and replay smoke test:

```bash
./framework local-test --profile inference-smoke --json
```

Run the local four-agent training loop:

```bash
./framework local-test --profile training --json
```

Components clear their own `local-train` directories and start from zero by default. Initialize from an explicit complete checkpoint with:

```bash
./framework local-test \
  --profile training \
  --initial-checkpoint /absolute/path/checkpoint_v000200.pt \
  --json
```

Evaluate an explicit ONNX model locally:

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/model.onnx \
  --json
```

Use an existing RunSpec:

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

The default image tag is `training-001`. Framework creates a dedicated Docker network plus `learner-local-train` and `aiserver-local-train` directories for every Run. Task and Run identity remain in Docker labels and external receipts only. Training metrics use `9005` without replay; `9004` is reserved for inference and model evaluation.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
