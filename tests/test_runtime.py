import json
import os
import tempfile
import unittest
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
