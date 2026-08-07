import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from framework_cli.runtime import (
    DockerRuntime,
    learner_monitor_transport_readiness,
    training_completion_status,
    training_record_readiness,
)
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

    def test_embedded_smoke_model_uses_canonical_manifest_identity(self):
        checksum = "a" * 64
        manifest = {
            "manifest_schema_version": 1,
            "package": "rl-smoke-model",
            "version": "0.10.0",
            "ready": True,
            "model_file": "model_v000000.onnx",
            "seed": 0,
            "identity": {
                "artifact_digest": checksum,
                "manifest_digest": "b" * 64,
                "model_lineage_id": "maze-fixed-map-seed-0",
                "model_version": 0,
            },
        }

        model_file, identity = DockerRuntime._embedded_smoke_model_identity(
            manifest, checksum
        )

        self.assertEqual(model_file, "model_v000000.onnx")
        self.assertEqual(identity["package_version"], "0.10.0")
        self.assertEqual(identity["version"], 0)
        self.assertEqual(identity["sha256"], checksum)
        self.assertEqual(
            identity["model_lineage_id"], "maze-fixed-map-seed-0"
        )

        legacy = dict(manifest)
        legacy.pop("identity")
        legacy["sha256"] = checksum
        legacy["model_version"] = 0
        with self.assertRaisesRegex(Exception, "manifest is invalid"):
            DockerRuntime._embedded_smoke_model_identity(legacy, checksum)

        traversal = dict(manifest)
        traversal["model_file"] = "../model_v000000.onnx"
        with self.assertRaisesRegex(Exception, "manifest is invalid"):
            DockerRuntime._embedded_smoke_model_identity(
                traversal, checksum
            )

        with self.assertRaisesRegex(Exception, "checksum or identity"):
            DockerRuntime._embedded_smoke_model_identity(
                manifest, "c" * 64
            )

    def test_learner_monitor_transport_requires_same_current_instance(self):
        status = {
            "schema_version": 1,
            "service": "learner-metrics",
            "stream": "current",
            "service_instance_id": "learner-metrics-a",
            "metrics_source_id": "local-training-a",
            "started_at": 100.0,
            "metrics_dir": "/opt/rl/learner/models/local-train/metrics",
            "mode": "training",
            "latest_sequence": 4,
            "latest_timestamp": 101.0,
            "stale": False,
        }

        readiness = learner_monitor_transport_readiness(
            status,
            dict(status),
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertTrue(readiness["ready"], readiness["reasons"])

        wrong_service = dict(status)
        wrong_service["service"] = "archive-metrics"
        mismatch = learner_monitor_transport_readiness(
            wrong_service,
            dict(status),
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertIn("host_service_mismatch", mismatch["reasons"])

        wrong_instance = dict(status)
        wrong_instance["service_instance_id"] = "learner-metrics-b"
        mismatch = learner_monitor_transport_readiness(
            status,
            wrong_instance,
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertIn(
            "monitor_service_instance_id_mismatch", mismatch["reasons"]
        )

        stale = dict(status)
        stale["stale"] = True
        mismatch = learner_monitor_transport_readiness(
            stale,
            dict(status),
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertIn("host_metrics_stale", mismatch["reasons"])

        wrong_directory = dict(status)
        wrong_directory["metrics_dir"] = "/opt/rl/archive/metrics"
        mismatch = learner_monitor_transport_readiness(
            wrong_directory,
            dict(status),
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertIn("host_metrics_dir_mismatch", mismatch["reasons"])

        old_archive = dict(status)
        old_archive["started_at"] = 90.0
        old_archive["latest_timestamp"] = 91.0
        mismatch = learner_monitor_transport_readiness(
            old_archive,
            dict(old_archive),
            minimum_timestamp=99.0,
            now=102.0,
        )
        self.assertIn("host_service_predates_launch", mismatch["reasons"])
        self.assertIn(
            "container_service_predates_launch", mismatch["reasons"]
        )

        missing = learner_monitor_transport_readiness(
            {}, {}, minimum_timestamp=99.0, now=102.0
        )
        self.assertIn("host_status_missing", missing["reasons"])
        self.assertIn("container_status_missing", missing["reasons"])

    def test_learner_monitor_transport_is_recorded_after_verification(self):
        runtime = DockerRuntime(self.spec)
        with (
            mock.patch.object(runtime, "_tcp_ready", return_value=True),
            mock.patch.object(runtime, "_record_port_transport") as record,
        ):
            transport = runtime._ensure_host_port(
                9005, 1, record_transport=False
            )

        self.assertEqual(transport, "docker-publish")
        record.assert_not_called()

    def test_training_readiness_requires_fresh_consistent_training_identity(self):
        actor_identity = {
            "model_lineage_id": "maze-fixed-map-seed-0",
            "model_version": 1,
            "artifact_digest": "a" * 64,
            "manifest_digest": "b" * 64,
        }
        learner_identity = {
            "model_lineage_id": "maze-fixed-map-seed-0",
            "model_version": 2,
            "artifact_digest": "c" * 64,
            "manifest_digest": "d" * 64,
        }
        record = {
            "mode": "training",
            "timestamp": 100.0,
            "interval_ms": 1000,
            "actor": {
                "ready": True,
                "instance_id": "actor-1",
                "model_identity": actor_identity,
            },
            "distributor": {
                "ready": True,
                "ingress_ready": True,
                "pool_ready": True,
                "instance_id": "pool-1",
            },
            "learner": {
                "model_identity": learner_identity,
                "model_version": 2,
                "max_policy_lag": 2,
                "error": "",
            },
            "model": {
                "ready": True,
                "latest_model_identity": learner_identity,
                "latest_ack_model_identity": actor_identity,
                "latest_ack_status": "MODEL_LOAD_STATUS_LOADED",
            },
            "chain": {"ready": True},
        }

        readiness = training_record_readiness(
            record, minimum_timestamp=99.0, now=101.0
        )
        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["model_lag"], 1)

        record["actor"]["model_identity"] = {
            **actor_identity,
            "artifact_digest": "e" * 64,
        }
        mismatch = training_record_readiness(
            record, minimum_timestamp=99.0, now=101.0
        )
        self.assertIn("actor_model_ack_mismatch", mismatch["reasons"])

        record["actor"]["model_identity"] = actor_identity
        stale = training_record_readiness(
            record, minimum_timestamp=101.0, now=110.0
        )
        self.assertIn("metrics_predate_launch", stale["reasons"])
        self.assertIn("metrics_stale", stale["reasons"])

        container_record = json.loads(json.dumps(record))
        container_record["actor"]["instance_id"] = "actor-other"
        source_mismatch = training_record_readiness(
            record,
            minimum_timestamp=99.0,
            now=101.0,
            container_record=container_record,
        )
        self.assertIn(
            "container_actor_identity_mismatch",
            source_mismatch["reasons"],
        )

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
            identity = {
                "model_lineage_id": "maze-fixed-map-seed-0",
                "model_version": version,
                "artifact_digest": checksum,
                "manifest_digest": hashlib.sha256(
                    f"archive-manifest-{version}".encode()
                ).hexdigest(),
            }
            (savepoint / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "runtime_manifest_identity": identity,
                        "runtime_manifest_digest": identity[
                            "manifest_digest"
                        ],
                        "model_file": "SaveModel.onnx",
                        "checkpoint_file": "checkpoint.pt",
                        "artifact_digest": checksum,
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
                        "behavior_model": {
                            "model_lineage_id": "maze-fixed-map-seed-0",
                            "minimum_model_version": version - 1,
                            "maximum_model_version": version - 1,
                            "models": [
                                {
                                    "model_lineage_id": "maze-fixed-map-seed-0",
                                    "model_version": version - 1,
                                    "artifact_digest": "d" * 64,
                                    "manifest_digest": "e" * 64,
                                }
                            ],
                        },
                        "train_update_id": f"update-{version}",
                        "model": {
                            "model_lineage_id": "maze-fixed-map-seed-0",
                            "model_version": version,
                            "artifact_digest": receipt_checksums[version],
                            "manifest_digest": hashlib.sha256(
                                f"receipt-manifest-{version}".encode()
                            ).hexdigest(),
                        },
                    }
                ),
                encoding="utf-8",
            )

        for version in (0, 1, 2):
            model = serving / f"model_v{version:06d}.onnx"
            model.write_bytes(f"serving-model-{version}".encode())
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            (serving / f"manifest_v{version:06d}.json").write_text(
                json.dumps(
                    {
                        "manifest_schema_version": 1,
                        "contract": {
                            "package_name": "rl-contracts",
                            "package_version": "0.10.0",
                        },
                        "identity": {
                            "model_lineage_id": "maze-fixed-map-seed-0",
                            "model_version": version,
                            "artifact_digest": checksum,
                            "manifest_digest": hashlib.sha256(
                                f"runtime-manifest-{version}".encode()
                            ).hexdigest(),
                        },
                        "model_file": model.name,
                        "ready": True,
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
            result = runtime._validate_training_artifacts(local_train, 2, 2)

        self.assertEqual(result["archived_versions"], [0, 2])
        self.assertEqual(result["runtime_versions"], [0, 1, 2])
        self.assertEqual(result["required_runtime_versions"], [0, 1, 2])
        self.assertEqual(result["v2_behavior_model_version_range"], [1, 1])

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
        joined = " ".join(environment)
        self.assertNotIn("MAZE_", joined)
        self.assertNotIn("RL_RUN_ID", joined)
        self.assertNotIn(self.spec.run_id, joined)
        self.assertIn("RL_AISERVER_ID=aiserver-0", joined)
        self.assertIn("RL_CLIENT_INSTANCE_ID=maze-client-0", joined)

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
        timeline = []

        def record_container(role, image_ref, arguments):
            timeline.append(f"start:{role}")
            started[role] = {
                "image_ref": image_ref,
                "arguments": arguments,
            }
            return f"{role}-id"

        def record_monitor(*_args, **_kwargs):
            timeline.append("monitor:semantic")
            return {"ready": True, "reasons": []}

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
                "trained": 3072,
                "stale": 0,
                "invalid": 0,
                "shutdown_untrained": 0,
            },
            "learner": {
                "model_version": 6,
                "initial_model_version": 0,
                "train_updates": 6,
                "trained_samples": 3072,
                "run_train_updates": 6,
                "run_trained_samples": 3072,
                "train_batch_size": 512,
                "max_train_batch_size": 639,
                "max_policy_lag": 2,
            },
        }
        with (
            mock.patch.object(
                runtime, "_run_container", side_effect=record_container
            ),
            mock.patch.object(runtime, "_wait_healthy"),
            mock.patch.object(
                runtime,
                "_ensure_host_port",
                return_value="docker-publish",
            ),
            mock.patch.object(
                runtime,
                "_wait_learner_monitor_transport",
                side_effect=record_monitor,
            ),
            mock.patch.object(
                runtime,
                "_wait_training_ready",
                return_value=(metrics, {"ready": True, "reasons": []}),
            ),
            mock.patch.object(runtime, "_wait_container_exit", return_value=0),
            mock.patch.object(runtime, "_request_training_finalize"),
            mock.patch.object(runtime, "_wait_metrics", return_value=metrics),
            mock.patch.object(
                runtime,
                "_validate_training_artifacts",
                return_value={"current_version": 6},
            ),
            mock.patch.object(
                runtime,
                "_docker",
                return_value=mock.Mock(
                    returncode=0, stdout="{}", stderr=""
                ),
            ),
        ):
            result = runtime._start_training(wait_for_completion=True)

        learner_args = " ".join(started["learner"]["arguments"])
        aiserver_args = " ".join(started["aiserver"]["arguments"])
        client_args = " ".join(started["client"]["arguments"])
        self.assertIn("127.0.0.1:9005:9005", learner_args)
        self.assertNotIn("127.0.0.1:9200:9200", learner_args)
        self.assertIn(
            "RL_SAMPLE_DISTRIBUTOR_HOST=maze-learner",
            aiserver_args,
        )
        self.assertIn("RL_TRAINING_SAMPLE_BUDGET=3072", aiserver_args)
        self.assertNotIn("--publish", started["aiserver"]["arguments"])
        self.assertNotIn("MAZE_", client_args)
        self.assertIn("RL_AISERVER_HOST=maze-aiserver", client_args)
        self.assertNotIn("9004", client_args)
        self.assertLess(
            timeline.index("start:aiserver"),
            timeline.index("monitor:semantic"),
        )
        self.assertEqual(
            result["ports"], {"learner_metrics": 9005}
        )
        self.assertEqual(
            result["internal_ports"]["local_sample_service"], 9100
        )
        self.assertEqual(
            result["monitor_url"], "http://127.0.0.1:9005/"
        )

    def test_completion_requires_trained_disposition_not_only_ack(self):
        record = {
            "actor": {
                "produced": 3072,
                "outbound_pending": 0,
                "final_drop": 0,
            },
            "distributor": {
                "accepted": 3072,
                "acked": 3072,
                "ready_samples": 0,
                "leased_samples": 0,
                "trained": 1024,
                "stale": 2048,
                "invalid": 0,
                "shutdown_untrained": 0,
            },
            "learner": {
                "model_version": 2,
                "initial_model_version": 0,
                "run_train_updates": 2,
                "run_trained_samples": 1024,
                "train_batch_size": 512,
                "max_train_batch_size": 639,
            },
        }

        completion = training_completion_status(record, 3072)

        self.assertFalse(completion["complete"])
        self.assertTrue(completion["failed"])
        self.assertIn(
            "non_trained_disposition_stale", completion["reasons"]
        )

    def test_completion_uses_run_scoped_resume_counters(self):
        record = {
            "actor": {
                "produced": 3072,
                "outbound_pending": 0,
                "final_drop": 0,
            },
            "distributor": {
                "accepted": 3072,
                "acked": 3072,
                "ready_samples": 0,
                "leased_samples": 0,
                "trained": 3072,
                "stale": 0,
                "invalid": 0,
                "shutdown_untrained": 0,
            },
            "learner": {
                "model_version": 26,
                "initial_model_version": 20,
                "train_updates": 26,
                "trained_samples": 13072,
                "run_train_updates": 6,
                "run_trained_samples": 3072,
                "train_batch_size": 512,
                "max_train_batch_size": 639,
            },
        }

        completion = training_completion_status(record, 3072)

        self.assertTrue(completion["complete"], completion["reasons"])

    def test_completion_accepts_bounded_whole_fragment_update_count(self):
        record = {
            "actor": {
                "produced": 3072,
                "outbound_pending": 0,
                "final_drop": 0,
            },
            "distributor": {
                "accepted": 3072,
                "acked": 3072,
                "ready_samples": 0,
                "leased_samples": 0,
                "trained": 3072,
                "stale": 0,
                "invalid": 0,
                "shutdown_untrained": 0,
            },
            "learner": {
                "model_version": 5,
                "initial_model_version": 0,
                "run_train_updates": 5,
                "run_trained_samples": 3072,
                "train_batch_size": 512,
                "max_train_batch_size": 639,
            },
        }

        completion = training_completion_status(record, 3072)

        self.assertTrue(completion["complete"], completion["reasons"])
        self.assertEqual(completion["counts"]["expected_updates_min"], 5)
        self.assertEqual(completion["counts"]["expected_updates_max"], 6)

    def test_framework_requests_final_drain_only_after_producer_flush(self):
        runtime = DockerRuntime(self.spec)
        record = {
            "actor": {
                "produced": 3072,
                "accepted": 3072,
                "outbound_pending": 0,
                "final_drop": 0,
            },
            "distributor": {"accepted": 3072},
        }
        docker_results = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(returncode=0, stdout="", stderr=""),
        ]
        events = []
        with (
            mock.patch.object(runtime, "_require_running"),
            mock.patch.object(
                runtime,
                "_http_json",
                return_value={"record": record},
            ),
            mock.patch.object(
                runtime, "_docker", side_effect=docker_results
            ) as docker,
            mock.patch.object(
                runtime,
                "_record",
                side_effect=lambda state, detail="": events.append(state),
            ),
        ):
            runtime._request_training_finalize(1)

        self.assertEqual(
            docker.call_args_list[0].args[0][-1],
            "/tmp/rl-training-finalize",
        )
        self.assertEqual(
            docker.call_args_list[1].args[0][-1],
            "/tmp/rl-training-finalized",
        )
        self.assertEqual(
            events,
            ["training_finalize_requested", "training_finalize_complete"],
        )

    def test_platform_training_start_returns_after_chain_readiness(self):
        runtime = DockerRuntime(self.spec)
        record = {
            "actor": {"produced": 1, "final_drop": 0},
            "distributor": {"accepted": 1, "acked": 0},
            "learner": {"model_version": 0, "train_updates": 0},
        }
        with (
            mock.patch.object(runtime, "_run_container"),
            mock.patch.object(runtime, "_wait_healthy"),
            mock.patch.object(
                runtime,
                "_ensure_host_port",
                return_value="docker-publish",
            ),
            mock.patch.object(
                runtime,
                "_wait_learner_monitor_transport",
                return_value={"ready": True, "reasons": []},
            ),
            mock.patch.object(
                runtime,
                "_wait_training_ready",
                return_value=(
                    record,
                    {"ready": True, "reasons": [], "model_lag": 0},
                ),
            ),
            mock.patch.object(
                runtime,
                "_container_state",
                return_value={"exists": True, "running": True},
            ),
            mock.patch.object(runtime, "_wait_container_exit") as wait_exit,
            mock.patch.object(
                runtime, "_validate_training_artifacts"
            ) as validate_artifacts,
        ):
            result = runtime._start_training()

        self.assertEqual(result["actual_status"], "ready")
        self.assertEqual(result["phase"], "running")
        self.assertEqual(result["counts"]["produced"], 1)
        wait_exit.assert_not_called()
        validate_artifacts.assert_not_called()

    def test_status_degrades_when_live_chain_identity_does_not_match(self):
        runtime = DockerRuntime(self.spec)
        checksum = "a" * 64
        identity = {
            "model_lineage_id": "maze-fixed-map-seed-0",
            "model_version": 0,
            "artifact_digest": checksum,
            "manifest_digest": "b" * 64,
        }
        timestamp = 100.0
        runtime._write_state(
            "ready",
            result={"readiness": {"timestamp": timestamp}},
        )
        record = {
            "mode": "training",
            "timestamp": timestamp,
            "interval_ms": 1000,
            "actor": {
                "ready": True,
                "instance_id": "actor-status",
                "model_identity": identity,
            },
            "distributor": {
                "ready": True,
                "instance_id": "pool-status",
                "ingress_ready": True,
                "pool_ready": True,
            },
            "learner": {
                "model_identity": identity,
                "model_version": 0,
                "max_policy_lag": 2,
                "error": "",
            },
            "model": {
                "ready": True,
                "latest_model_identity": identity,
                "latest_ack_model_identity": identity,
                "latest_ack_status": "MODEL_LOAD_STATUS_LOADED",
            },
            "chain": {"ready": True},
        }

        def container_state(role):
            if role == "client":
                return {
                    "exists": True,
                    "running": False,
                    "exit_code": 0,
                }
            return {"exists": True, "running": True, "exit_code": 0}

        with (
            mock.patch.object(
                runtime, "_container_state", side_effect=container_state
            ),
            mock.patch.object(
                runtime, "_http_json", return_value={"record": record}
            ),
            mock.patch.object(
                runtime, "_container_metrics_latest", return_value=record
            ),
            mock.patch(
                "framework_cli.runtime.time.time", return_value=101.0
            ),
        ):
            self.assertEqual(runtime.status()["actual_status"], "ready")
            record["actor"]["model_identity"] = {
                **identity,
                "manifest_digest": "c" * 64,
            }
            degraded = runtime.status()

        self.assertEqual(degraded["actual_status"], "degraded")
        self.assertIn(
            "actor_model_ack_mismatch", degraded["readiness"]["reasons"]
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
