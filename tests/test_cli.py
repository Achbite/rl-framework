import json
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class FrameworkCliTest(unittest.TestCase):
    def test_training_receipt_is_launch_readiness_and_persists_stop(self):
        namespace = runpy.run_path(str(ROOT / "framework"))
        execute = namespace["execute"]

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            spec = SimpleNamespace(
                task_id="task-training",
                run_id="task-training-run-0001",
                source_path=run_root / "run-spec.json",
            )

            class FakeRuntime:
                def __init__(self, _spec, runtime_scope):
                    self.run_root = run_root
                    self.runtime_scope = runtime_scope

                def start(self, *, wait_for_completion):
                    if wait_for_completion:
                        raise AssertionError(
                            "training must not wait for completion"
                        )
                    return {"actual_status": "ready"}

                @staticmethod
                def _atomic_json(path, document):
                    path.write_text(
                        json.dumps(document, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )

                def stop(self):
                    raise AssertionError("unbounded training remains running")

            args = SimpleNamespace(
                command="local-test",
                profile="training",
                keep_running=False,
            )
            with mock.patch.dict(
                execute.__globals__,
                {
                    "local_test_spec": lambda _args: spec.source_path,
                    "load_run_spec": lambda _path: spec,
                    "DockerRuntime": FakeRuntime,
                },
            ):
                result = execute(args)

            persisted = json.loads(
                (run_root / "acceptance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, result)
            self.assertTrue(persisted["ok"])
            self.assertEqual(persisted["scope"], "launch_readiness")
            self.assertFalse(persisted["workload_completed"])
            self.assertFalse(persisted["cleaned_up"])
            self.assertTrue(persisted["stop_required"])
            self.assertIn("framework stop", persisted["stop_command"])
            self.assertIn("readiness_at", persisted)
            self.assertNotIn("completed_at", persisted)


if __name__ == "__main__":
    unittest.main()
