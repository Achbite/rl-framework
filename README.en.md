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

The default image tag is `training-001`. Runtime data is stored under `.workspace/runtime` in the project root. Training metrics use port `9005`, and local replay uses port `9004`.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
