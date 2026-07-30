import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from framework_cli.runtime import DockerRuntime
from framework_cli.spec import RunSpec


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
            task_id="m0-test-20260729-120000-0001",
            run_id="m0-test-20260729-120000-0001-run-0001",
            attempt=1,
            mode="training",
            agents=4,
            episodes=3,
            max_steps=256,
            image_tags={
                "learner": "training-001",
                "aiserver": "training-001",
                "client": "training-001",
            },
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
