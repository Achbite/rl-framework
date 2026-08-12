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

Platform `start` returns after the Learner and AIServer Docker healthchecks pass
and the Client has completed OpenSession, Init, first Training Assignment
validation, and session-policy publication; an unavailable optional metrics snapshot does
not block this launch/readiness result. The containers keep running. `status`
continuously projects runtime health. `start` and `stop` serialize on the same
per-Run lifecycle lock, so once `stop` returns no late readiness commit can
recreate the run. `local-test --profile training` uses the same
unbounded Training semantics: it returns a `launch_readiness` receipt after the
core chain is ready, leaves the workload running, and requires the receipt's
`stop_command` or an explicit `framework stop` for controlled shutdown. This
launch receipt does not prove Client completion, sample conservation, PPO
quality, or model-artifact acceptance.

The default image tag is `training-001`. RunSpec v2 stores each role's image repository, tag, command, arguments, and CPU/memory/GPU limits, and requires AIServer and Client to share a host. Framework creates a dedicated Docker network and component directories for every Run. Task and Run identity remain in Docker labels and external receipts only.

The standard Framework Training path neither depends on nor reserves the
optional metrics port `9005`; `9002/9100/9200` remain internal to the task
network. Inference and model evaluation expose only replay port `9004`.

## Tests

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## License

[MIT License](LICENSE)
