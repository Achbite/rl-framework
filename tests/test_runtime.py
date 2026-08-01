import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from framework_cli.runtime import DockerRuntime
from framework_cli.spec import ComponentSpec, ResourceSpec, RunSpec


class RuntimePortTransportTest(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace.cleanup)
        self.environment = mock.patch.dict(
            os.environ, {"RL_TRAINING_WORKSPACE": self.workspace.name}
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.spec = RunSpec(
            schema_version=2,
            task_id="m0-test-20260729-120000-0001",
            run_id="m0-test-20260729-120000-0001-run-0001",
            attempt=1,
            mode="training",
            agents=4,
            episodes=3,
            max_steps=256,
            components={
                "learner": ComponentSpec(
                    "rl-training/learner",
                    "training-001",
                    "/opt/rl/learner/run.sh",
                    ("training",),
                    ResourceSpec(2.0, 4096, 0),
                ),
                "aiserver": ComponentSpec(
                    "rl-training/aiserver",
                    "training-001",
                    "/opt/rl/aiserver/run.sh",
                    ("training",),
                    ResourceSpec(2.0, 2048, 0),
                ),
                "client": ComponentSpec(
                    "rl-training/maze-client",
                    "training-001",
                    "/opt/rl/maze-client/run.sh",
                    (),
                    ResourceSpec(2.0, 2048, 0),
                ),
            },
            server_pod_same_host="required",
            model_path=None,
            initial_checkpoint=None,
            source_path=Path(self.workspace.name) / "run.json",
        )

    def test_reads_first_ssh_host(self):
        path = Path(self.workspace.name) / "ssh.config"
        path.write_text(
            "# generated\nHost lima-colima\n  Hostname 127.0.0.1\n",
            encoding="utf-8",
        )
        self.assertEqual(DockerRuntime._ssh_target(path), "lima-colima")

    def test_evaluation_model_resolves_fixed_savepoint_layout(self):
        savepoint = Path(self.workspace.name) / "archive" / "000200"
        savepoint.mkdir(parents=True)
        model = savepoint / "SaveModel.onnx"
        model.write_bytes(b"onnx")

        resolved_dir, resolved_model = (
            DockerRuntime._resolve_evaluation_model(str(savepoint))
        )
        self.assertEqual(resolved_dir, savepoint.resolve())
        self.assertEqual(resolved_model, model.resolve())

        direct_dir, direct_model = (
            DockerRuntime._resolve_evaluation_model(str(model))
        )
        self.assertEqual(direct_dir, savepoint.resolve())
        self.assertEqual(direct_model, model.resolve())

        wrong_name = savepoint / "model_v000200.onnx"
        wrong_name.write_bytes(b"onnx")
        with self.assertRaisesRegex(Exception, "SaveModel.onnx"):
            DockerRuntime._resolve_evaluation_model(str(wrong_name))

    def test_training_artifacts_use_fixed_savepoint_layout(self):
        local_train = Path(self.workspace.name) / "learner-local-train"
        archive = local_train / "archive"
        serving = local_train / "runtime" / "serving"
        checkpoints = local_train / "runtime" / "checkpoints"
        receipts = local_train / "runtime" / "receipts"
        for directory in (archive, serving, checkpoints, receipts):
            directory.mkdir(parents=True)

        archive_checksums = {}
        for version in (0, 2):
            savepoint = archive / f"{version:06d}"
            savepoint.mkdir()
            model = savepoint / "SaveModel.onnx"
            model.write_bytes(f"archive-model-{version}".encode())
            (savepoint / "checkpoint.pt").write_bytes(b"checkpoint")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            archive_checksums[version] = checksum
            (savepoint / "manifest.json").write_text(
                json.dumps(
                    {
                        "contract_version": "0.6.0",
                        "model_version": version,
                        "model_file": "SaveModel.onnx",
                        "checkpoint_file": "checkpoint.pt",
                        "sha256": checksum,
                    }
                ),
                encoding="utf-8",
            )

        receipt_checksums = {
            1: hashlib.sha256(b"receipt-model-1").hexdigest(),
            2: archive_checksums[2],
        }
        for version in (1, 2):
            (receipts / f"update-{version}.json").write_text(
                json.dumps(
                    {
                        "state": "ACKED",
                        "target_model_version": version,
                        "behavior_model_version": version - 1,
                        "train_update_id": f"update-{version}",
                        "archive_committed": version == 2,
                        "manifest": {
                            "sha256": receipt_checksums[version],
                        },
                    }
                ),
                encoding="utf-8",
            )

        for version in (1, 2):
            model = serving / f"model_v{version:06d}.onnx"
            model.write_bytes(f"serving-model-{version}".encode())
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            (serving / f"manifest_v{version:06d}.json").write_text(
                json.dumps(
                    {
                        "contract_version": "0.6.0",
                        "model_version": version,
                        "sha256": checksum,
                    }
                ),
                encoding="utf-8",
            )
            (checkpoints / f"checkpoint_v{version:06d}.pt").write_bytes(
                b"checkpoint"
            )

        runtime = DockerRuntime(self.spec)
        checkpoint_probe = mock.Mock(
            stdout=json.dumps(
                {
                    "version": 2,
                    "optimizer": True,
                    "update": "update-2",
                }
            )
        )
        with mock.patch.object(
            runtime, "_docker", return_value=checkpoint_probe
        ):
            result = runtime._validate_training_artifacts(local_train, 2)

        self.assertEqual(result["archived_versions"], [0, 2])
        self.assertEqual(result["runtime_versions"], [1, 2])

    def test_tunnel_socket_is_short_and_stable(self):
        runtime = DockerRuntime(self.spec)
        first = runtime._tunnel_socket(9005)
        second = runtime._tunnel_socket(9005)
        self.assertEqual(first, second)
        self.assertLess(len(str(first)), 104)

    def test_records_port_transport_by_run(self):
        runtime = DockerRuntime(self.spec)
        runtime._record_port_transport(9005, "colima-ssh-tunnel")
        document = json.loads(
            (runtime.run_root / "port-transports.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(document["run_id"], self.spec.run_id)
        self.assertEqual(
            document["ports"]["9005"], "colima-ssh-tunnel"
        )

    def test_stop_appends_timeline_and_preserves_result(self):
        runtime = DockerRuntime(self.spec)
        runtime._record("preflight_complete")
        runtime._record("training_ready")
        result = {"actual_status": "ready", "counts": {"acked": 1536}}
        runtime._write_state("ready", result=result)

        resumed = DockerRuntime(self.spec)
        with mock.patch.object(resumed, "stop_run"):
            resumed.stop()

        timeline = json.loads(
            (resumed.run_root / "timeline.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            [event["state"] for event in timeline["events"]],
            [
                "preflight_complete",
                "training_ready",
                "stop_requested",
                "stopped",
            ],
        )
        state = json.loads(
            (resumed.run_root / "state.json").read_text(encoding="utf-8")
        )
        self.assertEqual(state["actual_status"], "stopped")
        self.assertEqual(state["result"], result)

    def test_managed_container_query_is_scoped_to_run_id(self):
        runtime = DockerRuntime(self.spec)
        completed = mock.Mock(stdout="")
        with mock.patch.object(runtime, "_docker", return_value=completed) as call:
            runtime._managed_containers()
        arguments = call.call_args.args[0]
        self.assertIn(
            f"label=rl-training.run_id={self.spec.run_id}",
            arguments,
        )

    def test_component_environment_excludes_platform_run_identity(self):
        runtime = DockerRuntime(self.spec)
        environment = runtime._base_env()
        self.assertNotIn("MAZE_RUN_ID", " ".join(environment))
        self.assertNotIn(self.spec.run_id, " ".join(environment))

    def test_container_uses_runspec_image_command_and_resources(self):
        runtime = DockerRuntime(self.spec)
        completed = mock.Mock(stdout="container-id\n")
        with mock.patch.object(
            runtime, "_docker", return_value=completed
        ) as docker:
            container_id = runtime._run_container(
                "learner",
                self.spec.image_ref("learner"),
                ["--env", "EXAMPLE=1"],
            )
        self.assertEqual(container_id, "container-id")
        arguments = docker.call_args.args[0]
        image_index = arguments.index("rl-training/learner:training-001")
        self.assertEqual(
            arguments[image_index + 1 :],
            ["training"],
        )
        entrypoint_index = arguments.index("--entrypoint")
        self.assertEqual(
            arguments[entrypoint_index + 1],
            "/opt/rl/learner/run.sh",
        )
        self.assertIn("--cpus", arguments)
        self.assertIn("4096m", arguments)

    def test_healthcheck_is_required_and_waited(self):
        runtime = DockerRuntime(self.spec)
        healthy = mock.Mock(
            returncode=0,
            stdout=json.dumps({"Status": "healthy"}),
            stderr="",
        )
        with (
            mock.patch.object(runtime, "_require_running"),
            mock.patch.object(runtime, "_docker", return_value=healthy),
        ):
            runtime._wait_healthy("learner", 1)

        missing = mock.Mock(returncode=0, stdout="null\n", stderr="")
        with (
            mock.patch.object(runtime, "_require_running"),
            mock.patch.object(runtime, "_docker", return_value=missing),
        ):
            with self.assertRaisesRegex(Exception, "does not declare"):
                runtime._wait_healthy("learner", 1)

    def test_runs_have_distinct_networks_and_storage_roots(self):
        first = DockerRuntime(self.spec)
        second_spec = replace(
            self.spec,
            run_id="m0-test-20260729-120000-0001-run-0002",
            attempt=2,
        )
        second = DockerRuntime(second_spec)
        self.assertNotEqual(first.network, second.network)
        self.assertNotEqual(first.run_root, second.run_root)
        self.assertEqual(
            first.run_root / "learner-local-train",
            Path(self.workspace.name).resolve()
            / ".workspace/runtime/runs"
            / self.spec.run_id
            / "learner-local-train",
        )
        self.assertEqual(
            first.run_root / "aiserver-local-train",
            Path(self.workspace.name).resolve()
            / ".workspace/runtime/runs"
            / self.spec.run_id
            / "aiserver-local-train",
        )

    def test_network_is_labeled_with_platform_run_identity(self):
        runtime = DockerRuntime(self.spec)
        inspect = mock.Mock(returncode=1, stdout="", stderr="")
        created = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch.object(
            runtime, "_docker", side_effect=[inspect, created]
        ) as docker:
            runtime._ensure_network()
        create_arguments = docker.call_args_list[1].args[0]
        self.assertIn(
            f"rl-training.run_id={self.spec.run_id}",
            create_arguments,
        )
        self.assertEqual(create_arguments[-1], runtime.network)

    def test_training_stop_quiesces_producer_before_consumer(self):
        runtime = DockerRuntime(self.spec)
        roles = {
            "client-id": "client",
            "aiserver-id": "aiserver",
            "learner-id": "learner",
        }
        events = []
        with (
            mock.patch.object(
                runtime,
                "_managed_containers",
                return_value=list(roles),
            ),
            mock.patch.object(
                runtime,
                "_container_role",
                side_effect=lambda container_id: roles[container_id],
            ),
            mock.patch.object(
                runtime,
                "_stop_and_remove",
                side_effect=lambda container_id, timeout: events.append(
                    ("stop", container_id, timeout)
                ),
            ),
            mock.patch.object(
                runtime,
                "_quiesce",
                side_effect=lambda container_id, timeout: events.append(
                    ("quiesce", container_id, timeout)
                ),
            ),
            mock.patch.object(runtime, "_stop_port_tunnels"),
            mock.patch.object(runtime, "_remove_network"),
        ):
            runtime.stop_run()

        self.assertEqual(
            events,
            [
                ("stop", "client-id", 5),
                ("quiesce", "aiserver-id", 20),
                ("quiesce", "learner-id", 130),
                ("stop", "aiserver-id", 5),
                ("stop", "learner-id", 5),
            ],
        )

    def test_training_exposes_only_metrics_and_uses_internal_data_plane(self):
        runtime = DockerRuntime(self.spec, runtime_scope="local-tests")
        started = {}

        def record_container(role, image_ref, arguments):
            started[role] = {
                "image_ref": image_ref,
                "arguments": arguments,
            }
            return f"{role}-id"

        metrics = {
            "actor": {
                "produced": 3072,
                "outbound_pending": 0,
                "final_drop": 0,
                "model_version": 1,
            },
            "distributor": {
                "accepted": 3072,
                "acked": 3072,
                "ready_samples": 0,
                "leased_samples": 0,
            },
            "learner": {
                "model_version": 2,
                "train_updates": 2,
                "trained_samples": 3072,
            },
        }
        with (
            mock.patch.object(
                runtime, "_run_container", side_effect=record_container
            ),
            mock.patch.object(runtime, "_wait_healthy"),
            mock.patch.object(runtime, "_ensure_host_port"),
            mock.patch.object(runtime, "_wait_container_exit", return_value=0),
            mock.patch.object(runtime, "_wait_metrics", return_value=metrics),
            mock.patch.object(
                runtime,
                "_validate_training_artifacts",
                return_value={"current_version": 2},
            ),
            mock.patch.object(
                runtime,
                "_docker",
                return_value=mock.Mock(
                    returncode=0, stdout="{}", stderr=""
                ),
            ),
        ):
            result = runtime._start_training()

        learner_args = " ".join(started["learner"]["arguments"])
        aiserver_args = " ".join(started["aiserver"]["arguments"])
        client_args = " ".join(started["client"]["arguments"])
        self.assertIn("127.0.0.1:9005:9005", learner_args)
        self.assertNotIn("127.0.0.1:9200:9200", learner_args)
        self.assertIn(
            "MAZE_SAMPLE_DISTRIBUTOR_HOST=maze-learner",
            aiserver_args,
        )
        self.assertNotIn("--publish", started["aiserver"]["arguments"])
        self.assertNotIn("MAZE_WORKLOAD", client_args)
        self.assertNotIn("9004", client_args)
        self.assertEqual(
            result["ports"], {"learner_metrics": 9005}
        )
        self.assertEqual(
            result["internal_ports"]["local_sample_service"], 9100
        )

    def test_required_quiesce_failure_is_reported_after_cleanup(self):
        runtime = DockerRuntime(self.spec)
        roles = {
            "aiserver-id": "aiserver",
            "learner-id": "learner",
        }
        stopped = []
        with (
            mock.patch.object(
                runtime,
                "_managed_containers",
                return_value=list(roles),
            ),
            mock.patch.object(
                runtime,
                "_container_role",
                side_effect=lambda container_id: roles[container_id],
            ),
            mock.patch.object(runtime, "_quiesce", return_value=False),
            mock.patch.object(
                runtime,
                "_stop_and_remove",
                side_effect=lambda container_id, timeout: stopped.append(
                    (container_id, timeout)
                ),
            ),
            mock.patch.object(runtime, "_stop_port_tunnels"),
            mock.patch.object(runtime, "_remove_network"),
        ):
            with self.assertRaisesRegex(Exception, "aiserver, learner"):
                runtime.stop_run(require_quiesce=True)

        self.assertEqual(
            stopped,
            [("aiserver-id", 5), ("learner-id", 5)],
        )

    def test_port_conflict_reports_owner(self):
        runtime = DockerRuntime(self.spec)
        completed = mock.Mock(
            stdout="another-run\t127.0.0.1:9002->9002/tcp\n"
        )
        with mock.patch.object(runtime, "_docker", return_value=completed):
            with self.assertRaisesRegex(Exception, "another-run"):
                runtime._assert_host_ports_available((9002,))


if __name__ == "__main__":
    unittest.main()
