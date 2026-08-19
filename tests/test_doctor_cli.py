import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class DoctorCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repository = Path(__file__).resolve().parents[1]
        cls.framework = cls.repository / "framework"

    def _run(self, *arguments):
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return subprocess.run(
            [str(self.framework), *arguments],
            cwd=self.repository,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def _envelope(completed):
        document = json.loads(completed.stdout)
        assert set(document) == {
            "schema_version",
            "ok",
            "scope",
            "data",
            "warnings",
            "error",
        }
        return document

    def test_model_scope_returns_stable_json_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "SaveModel.onnx"
            model.write_bytes(b"read-only-model")
            before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            completed = self._run(
                "doctor", "model", "--model", str(model), "--json"
            )
            after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
            self.assertEqual(completed.returncode, 0, completed.stderr)
            document = self._envelope(completed)
            self.assertTrue(document["ok"])
            self.assertEqual(document["scope"], "model")
            self.assertEqual(
                document["data"]["sha256"],
                hashlib.sha256(b"read-only-model").hexdigest(),
            )
            self.assertEqual(before, after)

    def test_parameter_error_is_json_and_returns_two(self):
        completed = self._run("doctor", "model", "--json")
        self.assertEqual(completed.returncode, 2)
        document = self._envelope(completed)
        self.assertFalse(document["ok"])
        self.assertEqual(document["scope"], "model")
        self.assertIn("--model", document["error"])

    def test_public_help_contains_only_doctor_command(self):
        completed = self._run("--help")
        self.assertEqual(completed.returncode, 0)
        self.assertIn("doctor", completed.stdout)


if __name__ == "__main__":
    unittest.main()
