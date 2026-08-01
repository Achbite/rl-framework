# RL Training Framework

English | [简体中文](README.md)

Starts and checks the local Learner, AIServer, and Maze Client as one workflow. Component images are built by their own repositories; this repository only provides local Docker orchestration.

## Quick Start

Run the inference communication and replay smoke test:

```bash
./framework local-test --profile local-test --json
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

Evaluate an explicit savepoint locally. The model filename inside the directory
is fixed as `SaveModel.onnx`; AIServer reads it from its configured
`model.evaluation_dir`:

```bash
./framework local-test \
  --profile model-evaluation \
  --model-path /absolute/path/archive/000200 \
  --json
```

Use an existing RunSpec:

```bash
./framework preflight --run-spec run-spec.json --json
./framework start --run-spec run-spec.json --json
./framework status --run-spec run-spec.json --json
./framework stop --run-spec run-spec.json --json
```

The default image tag is `training-001`. RunSpec v2 stores each role's image repository, tag, command, arguments, and CPU/memory/GPU limits, and requires AIServer and Client to share a host. Framework creates a dedicated Docker network and component directories for every Run. Task and Run identity remain in Docker labels and external receipts only.

Training exposes only metrics port `9005` to the host; `9002/9100/9200` remain internal to the task network. Inference and model evaluation expose only replay port `9004`.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
