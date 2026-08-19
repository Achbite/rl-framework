import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from framework_cli import doctor


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for repository in doctor.REPOSITORIES:
            (self.root / repository).mkdir(parents=True)
        self.contract_digest = self._write_artifact("rl-contracts", 2)
        self._write_artifact("rl-sample-pool", 1)
        self._write_artifact("rl-model-distributor", 1)
        self.model = self.root / "models" / "SaveModel.onnx"
        self.model.parent.mkdir()
        self.model.write_bytes(b"onnx-model-fixture")
        self.commands = []

    @staticmethod
    def _sha256(data):
        return hashlib.sha256(data).hexdigest()

    def _write_artifact(self, package, schema_version):
        artifact = (
            self.root
            / ".workspace"
            / "artifacts"
            / package
            / doctor.CONTRACTS_VERSION
            / doctor.PLATFORM_DIRECTORY
        )
        relative = {
            "rl-contracts": "maze_task.proto",
            "rl-sample-pool": "bin/maze_sample_pool",
            "rl-model-distributor": "bin/maze_model_distributor",
        }[package]
        payload = f"{package}-payload".encode("utf-8")
        path = artifact / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        files = {relative: self._sha256(payload)}
        source_character = {
            "rl-contracts": "a",
            "rl-sample-pool": "b",
            "rl-model-distributor": "c",
        }[package]
        manifest = {
            "schema_version": schema_version,
            "package": package,
            "version": doctor.CONTRACTS_VERSION,
            "platform": doctor.ARTIFACT_PLATFORM,
            "source_commit": source_character * 40,
            "source_id": f"{package}-source",
            "files": files,
        }
        if package == "rl-contracts":
            artifact_digest = self._sha256(
                json.dumps(
                    files, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            )
            manifest.update(
                {
                    "source_digest": {
                        "algorithm": "sha256",
                        "hex": "d" * 64,
                    },
                    "source_tree_state": "clean",
                    "artifact_digest": {
                        "algorithm": "sha256",
                        "hex": artifact_digest,
                    },
                }
            )
        else:
            artifact_digest = self.contract_digest
            manifest.update(
                {
                    "source_sha256": source_character * 64,
                    "contract": {
                        "version": doctor.CONTRACTS_VERSION,
                        "platform": doctor.ARTIFACT_PLATFORM,
                        "artifact_digest": {
                            "algorithm": "sha256",
                            "hex": artifact_digest,
                        },
                    },
                }
            )
        (artifact / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return artifact_digest

    def _git_runner(self, command, **kwargs):
        self.commands.append(tuple(command))
        if "rev-parse" in command:
            output = "e" * 40 + "\n"
        elif "branch" in command:
            output = "dev-main\n"
        elif "status" in command:
            output = ""
        else:
            return subprocess.CompletedProcess(
                command, 1, "", "unexpected git command"
            )
        return subprocess.CompletedProcess(command, 0, output, "")

    @staticmethod
    def _snapshot(root):
        entries = []
        for path in sorted(root.rglob("*")):
            relative = str(path.relative_to(root))
            if path.is_symlink():
                entries.append((relative, "symlink", os.readlink(path)))
            elif path.is_file():
                entries.append(
                    (
                        relative,
                        "file",
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
            else:
                entries.append((relative, "directory", None))
        return entries

    def test_source_reports_eight_repositories_and_disables_infra(self):
        inspected = doctor.doctor_source(self.root, self._git_runner)
        self.assertTrue(inspected["ok"], inspected["error"])
        self.assertEqual(
            set(inspected["data"]["repositories"]),
            set(doctor.REPOSITORIES),
        )
        self.assertEqual(
            inspected["data"]["repositories"]["rl-infra"]["lifecycle"],
            "legacy/disabled",
        )
        self.assertFalse(inspected["data"]["infra"]["entered"])

    def test_artifacts_verify_source_platform_and_every_file_digest(self):
        inspected = doctor.doctor_artifacts(self.root)
        self.assertTrue(inspected["ok"], inspected["error"])
        contracts = inspected["data"]["artifacts"]["rl-contracts"]
        self.assertEqual(contracts["version"], doctor.CONTRACTS_VERSION)
        self.assertEqual(contracts["platform"], doctor.ARTIFACT_PLATFORM)
        self.assertEqual(contracts["source"]["tree_state"], "clean")
        self.assertTrue(
            contracts["files"]["maze_task.proto"]["matches_manifest"]
        )

    def test_artifacts_fail_closed_after_a_payload_changes(self):
        payload = (
            self.root
            / ".workspace/artifacts/rl-sample-pool/0.13.0/linux-arm64"
            / "bin/maze_sample_pool"
        )
        payload.write_bytes(b"tampered")
        inspected = doctor.doctor_artifacts(self.root)
        self.assertFalse(inspected["ok"])
        self.assertIn("file digest mismatch", inspected["error"])

    def test_model_is_filename_and_digest_only(self):
        inspected = doctor.doctor_model(self.model)
        self.assertTrue(inspected["ok"], inspected["error"])
        self.assertEqual(
            inspected["data"]["sha256"], self._sha256(b"onnx-model-fixture")
        )
        self.assertNotIn("model_step", inspected["data"])
        wrong = self.model.with_name("model.onnx")
        wrong.write_bytes(b"onnx-model-fixture")
        self.assertFalse(doctor.doctor_model(wrong)["ok"])

    def test_model_rejects_a_symbolic_link(self):
        link = self.root / "linked" / "SaveModel.onnx"
        link.parent.mkdir()
        try:
            link.symlink_to(self.model)
        except OSError as exc:
            self.skipTest(f"symbolic links unavailable: {exc}")
        inspected = doctor.doctor_model(link)
        self.assertFalse(inspected["ok"])
        self.assertIn("symbolic link", inspected["error"])

    def test_current_doctor_scopes_leave_the_tree_byte_identical(self):
        before = self._snapshot(self.root)
        inspected = (
            doctor.doctor_source(self.root, self._git_runner),
            doctor.doctor_artifacts(self.root),
            doctor.doctor_model(self.model),
        )
        after = self._snapshot(self.root)
        self.assertTrue(all(result["ok"] for result in inspected), inspected)
        self.assertEqual(before, after)
        self.assertFalse(any("framework.lock" in entry[0] for entry in after))


if __name__ == "__main__":
    unittest.main()
