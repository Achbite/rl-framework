import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from framework_cli.spec import SpecError, load_run_spec


class RunSpecTest(unittest.TestCase):
    def write(self, document):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "run.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def baseline(self):
        return {
            "schema_version": 2,
            "task_id": "m0-smoke-20260729-120000-0001",
            "run_id": "m0-smoke-20260729-120000-0001-run-0001",
            "attempt": 1,
            "mode": "training",
            "agents": 4,
            "episodes": 3,
            "max_steps": 256,
            "components": {
                "learner": {
                    "image_repository": "rl-training/learner",
                    "image_tag": "training-001",
                    "command": "/opt/rl/learner/run.sh",
                    "args": ["training"],
                    "resources": {
                        "cpus": 2,
                        "memory_mb": 4096,
                        "gpus": 0,
                    },
                },
                "aiserver": {
                    "image_repository": "rl-training/aiserver",
                    "image_tag": "training-001",
                    "command": "/opt/rl/aiserver/run.sh",
                    "args": ["training"],
                    "resources": {
                        "cpus": 2,
                        "memory_mb": 2048,
                        "gpus": 0,
                    },
                },
                "client": {
                    "image_repository": "rl-training/maze-client",
                    "image_tag": "training-001",
                    "command": "/opt/rl/maze-client/run.sh",
                    "args": [],
                    "resources": {
                        "cpus": 2,
                        "memory_mb": 2048,
                        "gpus": 0,
                    },
                },
            },
            "placement": {
                "aiserver_client_same_host": "required",
            },
        }

    def test_training_expected_samples(self):
        spec = load_run_spec(str(self.write(self.baseline())))
        self.assertEqual(spec.expected_samples, 3072)
        self.assertEqual(
            spec.image_ref("learner"), "rl-training/learner:training-001"
        )

    def test_validation_requires_model(self):
        document = self.baseline()
        document["mode"] = "model-evaluation"
        with self.assertRaises(SpecError):
            load_run_spec(str(self.write(document)))

    def test_removed_sample_flow_mode_is_rejected(self):
        document = self.baseline()
        document["mode"] = "sample-flow"
        with self.assertRaises(SpecError):
            load_run_spec(str(self.write(document)))

    def test_initial_checkpoint_is_explicit_and_training_only(self):
        document = self.baseline()
        document["initial_checkpoint"] = "/tmp/checkpoint.pt"
        spec = load_run_spec(str(self.write(document)))
        self.assertEqual(
            spec.initial_checkpoint,
            str(Path("/tmp/checkpoint.pt").resolve()),
        )

        document["mode"] = "local-test"
        with self.assertRaisesRegex(
            SpecError, "supported only for training"
        ):
            load_run_spec(str(self.write(document)))

    def test_removed_resume_from_is_rejected(self):
        document = self.baseline()
        document["resume_from"] = "/tmp/savepoint"
        with self.assertRaisesRegex(SpecError, "was removed"):
            load_run_spec(str(self.write(document)))

    def test_rejects_full_image_reference(self):
        document = self.baseline()
        document["components"]["learner"]["image_repository"] = (
            "registry/repo:test-001"
        )
        with self.assertRaises(SpecError):
            load_run_spec(str(self.write(document)))

    def test_component_command_and_resources_are_preserved(self):
        spec = load_run_spec(str(self.write(self.baseline())))
        learner = spec.components["learner"]
        self.assertEqual(
            learner.command_argv,
            ["/opt/rl/learner/run.sh", "training"],
        )
        self.assertEqual(
            learner.docker_resource_args,
            ["--cpus", "2", "--memory", "4096m"],
        )
        self.assertEqual(spec.server_pod_same_host, "required")

    def test_local_test_rejects_zero_override(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as workspace:
            environment = os.environ.copy()
            environment["RL_TRAINING_WORKSPACE"] = workspace
            result = subprocess.run(
                [
                    str(repository / "framework"),
                    "local-test",
                    "--profile",
                    "local-test",
                    "--agents",
                    "0",
                    "--json",
                ],
                cwd=repository,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            json.loads(result.stdout)["error"],
            "agents, episodes, and max-steps must be positive",
        )

    def test_legacy_inference_smoke_mode_reads_as_local_test(self):
        document = self.baseline()
        document["mode"] = "inference-smoke"
        spec = load_run_spec(str(self.write(document)))
        self.assertEqual(spec.mode, "local-test")


if __name__ == "__main__":
    unittest.main()
