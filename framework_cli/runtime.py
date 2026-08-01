from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .spec import RunSpec

EVALUATION_MODEL_FILE = "SaveModel.onnx"


class RuntimeFailure(RuntimeError):
    pass


def workspace_root() -> Path:
    configured = os.environ.get("RL_TRAINING_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


class DockerRuntime:
    def __init__(self, spec: RunSpec, runtime_scope: str = "runs"):
        self.spec = spec
        self.workspace = workspace_root()
        self.runtime_root = self.workspace / ".workspace/runtime"
        if runtime_scope not in {"runs", "local-tests"}:
            raise ValueError(f"unsupported runtime scope: {runtime_scope}")
        self.runtime_scope = runtime_scope
        self.run_root = self.runtime_root / runtime_scope / spec.run_id
        self.tunnel_root = self.runtime_root / "tunnels"
        run_key = hashlib.sha256(spec.run_id.encode("utf-8")).hexdigest()[:16]
        self.network = f"rl-training-{run_key}"
        self._events = self._load_events()

    @staticmethod
    def _resolve_evaluation_model(configured_path: str) -> tuple[Path, Path]:
        candidate = Path(configured_path).expanduser().resolve()
        savepoint = candidate if candidate.is_dir() else candidate.parent
        model_path = (
            savepoint / EVALUATION_MODEL_FILE
            if candidate.is_dir()
            else candidate
        )
        if model_path.name != EVALUATION_MODEL_FILE:
            raise RuntimeFailure(
                "model-evaluation requires a savepoint directory or "
                f"{EVALUATION_MODEL_FILE}"
            )
        if not model_path.is_file():
            raise RuntimeFailure(
                f"evaluation model does not exist: {model_path}"
            )
        return savepoint, model_path

    def _load_events(self) -> list[dict[str, Any]]:
        timeline_path = self.run_root / "timeline.json"
        if not timeline_path.is_file():
            return []
        try:
            document = json.loads(timeline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if document.get("run_id") != self.spec.run_id:
            return []
        events = document.get("events")
        return events if isinstance(events, list) else []

    def _docker(
        self,
        arguments: list[str],
        *,
        check: bool = True,
        timeout: float | None = 60.0,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                ["docker", *arguments],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeFailure(f"docker command failed: {exc}") from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeFailure(f"docker {' '.join(arguments[:2])}: {detail}")
        return result

    def _name(self, role: str) -> str:
        raw = f"rl-training-{self.spec.run_id}-{role}".lower()
        return re.sub(r"[^a-z0-9_.-]", "-", raw)

    def _record(self, state: str, detail: str = "") -> None:
        self._events.append(
            {
                "sequence": len(self._events) + 1,
                "timestamp": time.time(),
                "state": state,
                "detail": detail,
            }
        )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._atomic_json(
            self.run_root / "timeline.json",
            {"schema_version": 1, "run_id": self.spec.run_id, "events": self._events},
        )

    @staticmethod
    def _atomic_json(path: Path, document: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_state(self, actual_status: str, **extra: Any) -> None:
        document = {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "attempt": self.spec.attempt,
            "mode": self.spec.mode,
            "actual_status": actual_status,
            "updated_at": time.time(),
            **extra,
        }
        self._atomic_json(self.run_root / "state.json", document)

    def preflight(self) -> dict[str, Any]:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        info = self._docker(
            ["info", "--format", "{{json .ServerVersion}}"], timeout=20
        )
        model_path = None
        model_savepoint = None
        if self.spec.mode == "model-evaluation":
            savepoint, candidate = self._resolve_evaluation_model(
                self.spec.model_path or ""
            )
            model_path = str(candidate)
            model_savepoint = str(savepoint)
        initial_checkpoint = None
        if self.spec.initial_checkpoint:
            candidate = Path(self.spec.initial_checkpoint)
            if not candidate.is_file():
                raise RuntimeFailure(
                    f"initial checkpoint does not exist: {candidate}"
                )
            initial_checkpoint = str(candidate)
        images = {}
        for role in ("learner", "aiserver", "client"):
            image_ref = self.spec.image_ref(role)
            inspect = self._docker(["image", "inspect", image_ref], check=False)
            images[role] = {
                "ref": image_ref,
                "available": inspect.returncode == 0,
            }
        required_roles = {
            "local-test": ("aiserver", "client"),
            "training": ("learner", "aiserver", "client"),
            "model-evaluation": ("aiserver", "client"),
        }[self.spec.mode]
        missing = [
            images[role]["ref"]
            for role in required_roles
            if not images[role]["available"]
        ]
        if missing:
            raise RuntimeFailure(
                "required local images are missing: " + ", ".join(missing)
            )
        return {
            "schema_version": 1,
            "ok": True,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "mode": self.spec.mode,
            "docker_ready": True,
            "docker_server_version": json.loads(info.stdout.strip()),
            "images": images,
            "model_path": model_path,
            "model_savepoint": model_savepoint,
            "initial_checkpoint": initial_checkpoint,
        }

    def _ensure_network(self) -> None:
        result = self._docker(["network", "inspect", self.network], check=False)
        if result.returncode != 0:
            self._docker(
                [
                    "network",
                    "create",
                    "--label",
                    "rl-training.managed=true",
                    "--label",
                    f"rl-training.run_id={self.spec.run_id}",
                    self.network,
                ]
            )

    def _remove_network(self) -> None:
        result = self._docker(
            ["network", "inspect", self.network], check=False
        )
        if result.returncode == 0:
            self._docker(["network", "rm", self.network], check=False)

    def _managed_containers(self) -> list[str]:
        result = self._docker(
            [
                "ps",
                "-aq",
                "--filter",
                "label=rl-training.managed=true",
                "--filter",
                f"label=rl-training.run_id={self.spec.run_id}",
            ]
        )
        return [line for line in result.stdout.splitlines() if line]

    def _container_role(self, container_id: str) -> str:
        result = self._docker(
            [
                "inspect",
                "--format",
                "{{index .Config.Labels \"rl-training.role\"}}",
                container_id,
            ],
            check=False,
        )
        return result.stdout.strip()

    def _stop_and_remove(self, container_id: str, timeout: int) -> None:
        state = self._docker(
            ["inspect", "--format", "{{.State.Running}}", container_id],
            check=False,
        )
        if state.returncode == 0 and state.stdout.strip() == "true":
            self._docker(
                ["stop", "--time", str(timeout), container_id],
                check=False,
                timeout=timeout + 10,
            )
        self._docker(["rm", container_id], check=False)

    def _quiesce(self, container_id: str, timeout: int) -> bool:
        state = self._docker(
            ["inspect", "--format", "{{.State.Running}}", container_id],
            check=False,
        )
        if state.returncode != 0 or state.stdout.strip() != "true":
            return True
        signal_result = self._docker(
            ["kill", "--signal", "USR1", container_id],
            check=False,
        )
        if signal_result.returncode != 0:
            return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            marker = self._docker(
                [
                    "exec",
                    container_id,
                    "test",
                    "-f",
                    "/tmp/rl-training-quiesced",
                ],
                check=False,
            )
            if marker.returncode == 0:
                return True
            state = self._docker(
                ["inspect", "--format", "{{.State.Running}}", container_id],
                check=False,
            )
            if state.returncode != 0 or state.stdout.strip() != "true":
                return False
            time.sleep(0.2)
        return False

    def stop_run(self, require_quiesce: bool = False) -> None:
        containers = self._managed_containers()
        roles: dict[str, list[str]] = {}
        for container_id in containers:
            roles.setdefault(self._container_role(container_id), []).append(container_id)
        quiesce_failures = []
        for container_id in roles.get("client", []):
            self._stop_and_remove(container_id, 5)
        for container_id in roles.get("aiserver", []):
            if not self._quiesce(container_id, 20):
                quiesce_failures.append("aiserver")
        for container_id in roles.get("learner", []):
            if not self._quiesce(container_id, 130):
                quiesce_failures.append("learner")
        for container_id in roles.get("aiserver", []):
            self._stop_and_remove(container_id, 5)
        for container_id in roles.get("learner", []):
            self._stop_and_remove(container_id, 5)
        self._stop_port_tunnels()
        self._remove_network()
        if require_quiesce and quiesce_failures:
            failed = ", ".join(sorted(set(quiesce_failures)))
            raise RuntimeFailure(
                f"training shutdown forced after quiesce failed: {failed}"
            )

    def _assert_host_ports_available(self, ports: tuple[int, ...]) -> None:
        result = self._docker(
            [
                "ps",
                "--format",
                "{{.Names}}\t{{.Ports}}",
            ]
        )
        conflicts = []
        for line in result.stdout.splitlines():
            name, _, bindings = line.partition("\t")
            for port in ports:
                pattern = rf"(?:127\.0\.0\.1|0\.0\.0\.0|\[::\]):{port}->"
                if re.search(pattern, bindings):
                    conflicts.append(f"{port} ({name})")
        if conflicts:
            raise RuntimeFailure(
                "fixed host port conflict: " + ", ".join(sorted(set(conflicts)))
            )

    def _run_container(
        self,
        role: str,
        image_ref: str,
        arguments: list[str],
    ) -> str:
        name = self._name(role)
        component = self.spec.components[role]
        command = [
            "run",
            "--pull",
            "never",
            "--detach",
            "--name",
            name,
            "--network",
            self.network,
            "--network-alias",
            f"maze-{role}",
            "--label",
            "rl-training.managed=true",
            "--label",
            f"rl-training.role={role}",
            "--label",
            f"rl-training.task_id={self.spec.task_id}",
            "--label",
            f"rl-training.run_id={self.spec.run_id}",
            *component.docker_resource_args,
            "--entrypoint",
            component.command,
            *arguments,
            image_ref,
            *component.args,
        ]
        result = self._docker(command, timeout=120)
        return result.stdout.strip()

    def _container_state(self, role: str) -> dict[str, Any]:
        name = self._name(role)
        result = self._docker(
            ["inspect", "--format", "{{json .State}}", name], check=False
        )
        if result.returncode != 0:
            return {"exists": False, "running": False}
        state = json.loads(result.stdout)
        return {
            "exists": True,
            "running": bool(state.get("Running")),
            "status": state.get("Status", "unknown"),
            "exit_code": state.get("ExitCode"),
            "started_at": state.get("StartedAt"),
            "finished_at": state.get("FinishedAt"),
        }

    def _require_running(self, role: str) -> None:
        state = self._container_state(role)
        if not state.get("running"):
            logs = self._docker(
                ["logs", "--tail", "80", self._name(role)], check=False
            )
            detail = logs.stderr.strip() or logs.stdout.strip()
            raise RuntimeFailure(f"{role} exited before readiness: {detail}")

    def _wait_healthy(self, role: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last_status = "starting"
        while time.monotonic() < deadline:
            self._require_running(role)
            result = self._docker(
                [
                    "inspect",
                    "--format",
                    "{{json .State.Health}}",
                    self._name(role),
                ],
                check=False,
            )
            if result.returncode != 0:
                time.sleep(0.2)
                continue
            raw = result.stdout.strip()
            if raw in {"", "null", "<no value>"}:
                raise RuntimeFailure(
                    f"{role} image does not declare a Healthcheck"
                )
            health = json.loads(raw)
            last_status = str(health.get("Status", "unknown"))
            if last_status == "healthy":
                return
            if last_status == "unhealthy":
                entries = health.get("Log") or []
                detail = ""
                if entries:
                    detail = str(entries[-1].get("Output", "")).strip()
                raise RuntimeFailure(
                    f"{role} became unhealthy"
                    + (f": {detail}" if detail else "")
                )
            time.sleep(0.5)
        raise RuntimeFailure(
            f"{role} health readiness timeout; status={last_status}"
        )

    @staticmethod
    def _tcp_ready(port: int, timeout: float = 0.5) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except OSError:
            return False

    @classmethod
    def _wait_tcp(cls, port: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cls._tcp_ready(port):
                return
            time.sleep(0.2)
        raise RuntimeFailure(f"port {port} readiness timeout")

    def _colima_ssh_config(self) -> Path | None:
        configured = os.environ.get("RL_COLIMA_SSH_CONFIG")
        candidate = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".colima/_lima/colima/ssh.config"
        )
        return candidate if candidate.is_file() else None

    @staticmethod
    def _ssh_target(config_path: Path) -> str:
        for line in config_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*Host\s+(\S+)\s*$", line)
            if match:
                return match.group(1)
        raise RuntimeFailure(f"SSH config has no Host entry: {config_path}")

    def _tunnel_socket(self, port: int) -> Path:
        run_key = hashlib.sha256(self.spec.run_id.encode("utf-8")).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / f"rlfw-{run_key}-{port}.sock"

    def _tunnel_marker(self, port: int) -> Path:
        run_key = hashlib.sha256(self.spec.run_id.encode("utf-8")).hexdigest()[:12]
        return self.tunnel_root / f"{run_key}-{port}.json"

    def _record_port_transport(self, port: int, transport: str) -> None:
        path = self.run_root / "port-transports.json"
        document: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.spec.run_id,
            "ports": {},
        }
        if path.is_file():
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        document.setdefault("ports", {})[str(port)] = transport
        self._atomic_json(path, document)

    def _ensure_host_port(self, port: int, timeout: float) -> None:
        direct_deadline = time.monotonic() + min(timeout, 3.0)
        while time.monotonic() < direct_deadline:
            if self._tcp_ready(port):
                self._record_port_transport(port, "docker-publish")
                return
            time.sleep(0.2)

        config_path = self._colima_ssh_config()
        ssh = shutil.which("ssh")
        if config_path is None or ssh is None:
            self._wait_tcp(port, max(0.0, timeout - 3.0))
            self._record_port_transport(port, "docker-publish")
            return

        self.tunnel_root.mkdir(parents=True, exist_ok=True)
        socket_path = self._tunnel_socket(port)
        if socket_path.exists():
            socket_path.unlink()
        target = self._ssh_target(config_path)
        command = [
            ssh,
            "-F",
            str(config_path),
            "-o",
            "ControlMaster=yes",
            "-o",
            f"ControlPath={socket_path}",
            "-o",
            "ControlPersist=no",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{port}",
            "-N",
            "-f",
            target,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeFailure(
                f"Colima port tunnel failed for {port}: {exc}"
            ) from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeFailure(
                f"Colima port tunnel failed for {port}: {detail}"
            )
        self._atomic_json(
            self._tunnel_marker(port),
            {
                "schema_version": 1,
                "run_id": self.spec.run_id,
                "port": port,
                "socket_path": str(socket_path),
                "config_path": str(config_path),
                "target": target,
            },
        )
        self._wait_tcp(port, max(1.0, timeout - 3.0))
        self._record_port_transport(port, "colima-ssh-tunnel")

    def _stop_port_tunnels(self) -> None:
        if not self.tunnel_root.is_dir():
            return
        ssh = shutil.which("ssh")
        if ssh is None:
            return
        run_key = hashlib.sha256(self.spec.run_id.encode("utf-8")).hexdigest()[:12]
        for marker_path in self.tunnel_root.glob(f"{run_key}-*.json"):
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                socket_path = Path(marker["socket_path"])
                config_path = Path(marker["config_path"])
                target = marker["target"]
                subprocess.run(
                    [
                        ssh,
                        "-F",
                        str(config_path),
                        "-o",
                        f"ControlPath={socket_path}",
                        "-O",
                        "exit",
                        target,
                    ],
                    check=False,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except (
                KeyError,
                OSError,
                json.JSONDecodeError,
                subprocess.TimeoutExpired,
            ):
                socket_path = None
            marker_path.unlink(missing_ok=True)
            if socket_path is None:
                continue
            socket_path.unlink(missing_ok=True)

    def _wait_file(self, path: Path, role: str, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return
            self._require_running(role)
            time.sleep(0.2)
        raise RuntimeFailure(f"file readiness timeout: {path}")

    @staticmethod
    def _http_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise RuntimeFailure(f"HTTP request failed for {url}: {exc}") from exc

    def _wait_metrics(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        expected = self.spec.expected_samples
        url = "http://127.0.0.1:9005/api/metrics/latest"
        last_record: dict[str, Any] = {}
        while time.monotonic() < deadline:
            self._require_running("learner")
            self._require_running("aiserver")
            try:
                payload = self._http_json(url)
                last_record = payload.get("record", {})
            except RuntimeFailure:
                time.sleep(0.3)
                continue
            actor = last_record.get("actor", {})
            distributor = last_record.get("distributor", {})
            learner = last_record.get("learner", {})
            model = last_record.get("model", {})
            produced = int(actor.get("produced", 0))
            accepted = int(distributor.get("accepted", 0))
            acked = int(distributor.get("acked", 0))
            final_drop = int(actor.get("final_drop", 0))
            train_updates = int(learner.get("train_updates", 0))
            model_version = int(learner.get("model_version", 0))
            loaded_version = int(actor.get("model_version", -1))
            latest_ack_version = int(model.get("latest_ack_version", -1))
            if (
                (produced, accepted, acked, final_drop)
                == (expected, expected, expected, 0)
                and train_updates >= 2
                and model_version >= 2
                and loaded_version >= 1
                and latest_ack_version >= 1
            ):
                return last_record
            if any(value > expected for value in (produced, accepted, acked)):
                raise RuntimeFailure(
                    "sample counters exceeded the RunSpec expected total"
                )
            time.sleep(0.5)
        raise RuntimeFailure(
            f"training loop timeout; expected={expected}, last={last_record}"
        )

    def _validate_training_artifacts(
        self, local_train_root: Path, current_version: int
    ) -> dict[str, Any]:
        runtime = local_train_root / "runtime"
        published = runtime / "serving"
        checkpoints = runtime / "checkpoints"
        archive = local_train_root / "archive"
        updates = runtime / "receipts"
        if current_version < 2:
            raise RuntimeFailure("training did not publish model v2")

        receipts: dict[int, dict[str, Any]] = {}
        for path in updates.glob("*.json"):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if receipt.get("state") == "ACKED":
                receipts[int(receipt["target_model_version"])] = receipt

        identity_checksums: list[str] = []
        archive_manifests: dict[int, dict[str, Any]] = {}
        for version in (0, 2):
            version_dir = archive / f"{version:06d}"
            manifest_path = version_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeFailure(
                    f"training archive is missing: {manifest_path}"
                )
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            model_path = version_dir / str(manifest.get("model_file", ""))
            checkpoint_path = version_dir / str(
                manifest.get("checkpoint_file", "")
            )
            if not model_path.is_file() or not checkpoint_path.is_file():
                raise RuntimeFailure(
                    f"training archive v{version} is incomplete"
                )
            checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if (
                manifest.get("contract_version") != "0.6.0"
                or manifest.get("model_version") != version
                or manifest.get("model_file") != EVALUATION_MODEL_FILE
                or manifest.get("checkpoint_file") != "checkpoint.pt"
                or manifest.get("sha256") != checksum
                or "run_id" in manifest
            ):
                raise RuntimeFailure(
                    f"archive v{version} identity is invalid"
                )
            archive_manifests[version] = manifest
            if version == 0:
                identity_checksums.append(checksum)

        for version in (1, 2):
            receipt = receipts.get(version)
            checksum = (receipt or {}).get("manifest", {}).get("sha256")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise RuntimeFailure(
                    f"model v{version} receipt identity is missing"
                )
            identity_checksums.append(checksum)
        if len(set(identity_checksums)) != 3:
            raise RuntimeFailure("v0, v1, and v2 checksums must differ")

        v2_receipt = receipts.get(2, {})
        v2_update_id = v2_receipt.get("train_update_id")
        if not v2_update_id:
            raise RuntimeFailure("model v2 update receipt is missing")
        if (
            v2_receipt.get("target_model_version") != 2
            or v2_receipt.get("behavior_model_version") != 1
            or not v2_receipt.get("archive_committed")
        ):
            raise RuntimeFailure(
                "model v2 receipt must ACK behavior model v1 samples"
            )

        retained_versions = list(
            range(max(0, current_version - 1), current_version + 1)
        )
        actual_retained = sorted(
            int(path.stem.removeprefix("manifest_v"))
            for path in published.glob("manifest_v*.json")
        )
        if actual_retained != retained_versions:
            raise RuntimeFailure(
                "runtime serving directory does not retain exactly two versions"
            )
        for version in retained_versions:
            manifest_path = published / f"manifest_v{version:06d}.json"
            model_path = published / f"model_v{version:06d}.onnx"
            checkpoint_path = checkpoints / f"checkpoint_v{version:06d}.pt"
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
            if (
                manifest.get("contract_version") != "0.6.0"
                or manifest.get("model_version") != version
                or manifest.get("sha256") != checksum
                or not checkpoint_path.is_file()
                or "run_id" in manifest
            ):
                raise RuntimeFailure(
                    f"runtime model v{version} identity is invalid"
                )
        if (published / "manifest_v000000.json").exists():
            raise RuntimeFailure("runtime v0 was not removed by retention")

        checkpoint_probe = self._docker(
            [
                "exec",
                self._name("learner"),
                "python3",
                "-c",
                (
                    "import json,sys,torch;"
                    "p=torch.load(sys.argv[1],map_location='cpu',"
                    "weights_only=False);"
                    "print(json.dumps({'version':p.get('model_version'),"
                    "'optimizer':bool(p.get('optimizer_state_dict')),"
                    "'update':p.get('metadata',{}).get('train_update_id')}))"
                ),
                (
                    "/opt/rl/learner/models/local-train/runtime/checkpoints/"
                    f"checkpoint_v{current_version:06d}.pt"
                ),
            ]
        )
        checkpoint = json.loads(checkpoint_probe.stdout)
        if (
            checkpoint.get("version") != current_version
            or not checkpoint.get("optimizer")
            or not checkpoint.get("update")
        ):
            raise RuntimeFailure(
                "current runtime checkpoint identity is invalid"
            )
        return {
            "runtime_versions": retained_versions,
            "archived_versions": [0, 2],
            "identity_checksums_v0_v1_v2": identity_checksums,
            "v2_train_update_id": v2_update_id,
            "v2_behavior_model_version": 1,
            "current_version": current_version,
        }

    def _wait_container_exit(self, role: str, timeout: float) -> int:
        name = self._name(role)
        try:
            result = self._docker(["wait", name], timeout=timeout)
        except RuntimeFailure as exc:
            raise RuntimeFailure(f"{role} completion timeout: {exc}") from exc
        try:
            return int(result.stdout.strip())
        except ValueError as exc:
            raise RuntimeFailure(f"invalid docker wait result for {role}") from exc

    def _base_env(self) -> list[str]:
        return [
            "--env",
            "MAZE_AISERVER_ID=aiserver-0",
            "--env",
            "MAZE_CLIENT_ID=client-0",
            "--env",
            "MAZE_ENV_ID=env-0",
            "--env",
            "MAZE_SESSION_ID=0",
        ]

    def start(self) -> dict[str, Any]:
        self.preflight()
        self.stop_run()
        self.run_root.mkdir(parents=True, exist_ok=True)
        self._events = []
        self._record("preflight_complete")
        self._write_state("starting")
        required_ports = {
            "local-test": (9004,),
            "training": (9005,),
            "model-evaluation": (9004,),
        }[self.spec.mode]
        try:
            self._assert_host_ports_available(required_ports)
            self._ensure_network()
            if self.spec.mode == "training":
                result = self._start_training()
            else:
                result = self._start_replay_workload()
            self._write_state("ready", result=result)
            return result
        except Exception as exc:
            self._record("failed", str(exc))
            self._write_state("failed", error=str(exc))
            self.stop_run()
            raise

    def _start_training(self) -> dict[str, Any]:
        learner_logs = self.run_root / "learner/logs"
        learner_local_train = self.run_root / "learner-local-train"
        aiserver_local_train = self.run_root / "aiserver-local-train"
        aiserver_logs = self.run_root / "aiserver/log"
        client_logs = self.run_root / "client/log"
        for directory in (
            learner_logs,
            learner_local_train,
            aiserver_local_train,
            aiserver_logs,
            client_logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        learner_arguments = [
            *self._base_env(),
            "--env",
            "MAZE_LEARNER_ID=learner-0",
            "--env",
            "MAZE_AISERVER_HOST=maze-aiserver",
            "--env",
            "MAZE_LOCAL_TRAIN_ROOT=/opt/rl/learner/models/local-train",
            "--publish",
            "127.0.0.1:9005:9005",
            "--volume",
            f"{learner_logs}:/opt/rl/learner/logs",
            "--volume",
            (
                f"{learner_local_train}:"
                "/opt/rl/learner/models/local-train"
            ),
        ]
        if self.runtime_scope == "local-tests":
            learner_arguments.extend(
                ["--env", "MAZE_ARCHIVE_INTERVAL_UPDATES=2"]
            )
        if self.spec.initial_checkpoint:
            checkpoint = Path(self.spec.initial_checkpoint)
            container_checkpoint = (
                "/opt/rl/learner/initial-checkpoint/"
                f"{checkpoint.name}"
            )
            learner_arguments.extend(
                [
                    "--volume",
                    (
                        f"{checkpoint.parent}:"
                        "/opt/rl/learner/initial-checkpoint:ro"
                    ),
                    "--env",
                    f"MAZE_INITIAL_CHECKPOINT={container_checkpoint}",
                ]
            )
        self._run_container(
            "learner",
            self.spec.image_ref("learner"),
            learner_arguments,
        )
        self._wait_healthy("learner", 180)
        self._ensure_host_port(9005, 180)
        self._record("learner_pod_ready")

        self._run_container(
            "aiserver",
            self.spec.image_ref("aiserver"),
            [
                *self._base_env(),
                "--env",
                "MAZE_MODEL_DISTRIBUTOR_HOST=maze-learner",
                "--env",
                "MAZE_SAMPLE_DISTRIBUTOR_HOST=maze-learner",
                "--env",
                "MAZE_LOCAL_TRAIN_ROOT=/opt/rl/aiserver/models/local-train",
                "--volume",
                f"{aiserver_logs}:/opt/rl/aiserver/log",
                "--volume",
                (
                    f"{aiserver_local_train}:"
                    "/opt/rl/aiserver/models/local-train"
                ),
            ],
        )
        self._wait_healthy("aiserver", 60)
        self._record("model_v0_loaded_ack")
        self._record("aiserver_ready")

        self._run_container(
            "client",
            self.spec.image_ref("client"),
            [
                *self._base_env(),
                "--env",
                "MAZE_AISERVER_HOST=maze-aiserver",
                "--env",
                f"MAZE_AGENT_NUM={self.spec.agents}",
                "--env",
                f"MAZE_MAX_EPISODES={self.spec.episodes}",
                "--env",
                f"MAZE_MAX_STEPS={self.spec.max_steps}",
                "--volume",
                f"{client_logs}:/opt/rl/maze-client/log",
            ],
        )
        self._record("client_started")
        client_exit = self._wait_container_exit("client", 300)
        if client_exit != 0:
            logs = self._docker(
                ["logs", "--tail", "120", self._name("client")], check=False
            )
            raise RuntimeFailure(
                f"client exited with {client_exit}: "
                f"{logs.stderr.strip() or logs.stdout.strip()}"
            )
        self._record("client_completed")

        metrics = self._wait_metrics(180)
        current_version = int(metrics["learner"]["model_version"])
        artifacts = self._validate_training_artifacts(
            learner_local_train, current_version
        )
        client_ports = self._docker(
            [
                "inspect",
                "--format",
                "{{json .HostConfig.PortBindings}}",
                self._name("client"),
            ],
            check=False,
        )
        if "9004" in client_ports.stdout:
            raise RuntimeFailure("training Client published Replay port 9004")
        self._record("training_ready")
        counts = {
            "expected": self.spec.expected_samples,
            "produced": int(metrics["actor"]["produced"]),
            "accepted": int(metrics["distributor"]["accepted"]),
            "acked": int(metrics["distributor"]["acked"]),
            "final_drop": int(metrics["actor"]["final_drop"]),
            "ready": int(metrics["distributor"]["ready_samples"]),
            "leased": int(metrics["distributor"]["leased_samples"]),
        }
        if (
            counts["produced"]
            != counts["accepted"] + int(metrics["actor"]["outbound_pending"])
            + counts["final_drop"]
            or counts["accepted"]
            != counts["ready"] + counts["leased"] + counts["acked"]
        ):
            raise RuntimeFailure("training sample conservation is violated")
        self._atomic_json(
            self.run_root / "sample-conservation.json",
            {"schema_version": 1, "run_id": self.spec.run_id, **counts},
        )
        return {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "mode": self.spec.mode,
            "actual_status": "ready",
            "counts": counts,
            "models": artifacts,
            "train_updates": int(metrics["learner"]["train_updates"]),
            "trained_samples": int(metrics["learner"]["trained_samples"]),
            "monitor_url": (
                "http://127.0.0.1:9010/infra?"
                f"source=local-dev&run_id={urllib.parse.quote(self.spec.run_id)}"
            ),
            "ports": {
                "learner_metrics": 9005,
            },
            "internal_ports": {
                "aiserver": 9002,
                "local_sample_service": 9100,
                "model_distributor": 9200,
            },
        }

    def _start_replay_workload(self) -> dict[str, Any]:
        replay_dir = self.run_root / "replay"
        aiserver_logs = self.run_root / "aiserver/log"
        replay_dir.mkdir(parents=True, exist_ok=True)
        aiserver_logs.mkdir(parents=True, exist_ok=True)

        aiserver_arguments = [
            *self._base_env(),
            "--volume",
            f"{aiserver_logs}:/opt/rl/aiserver/log",
        ]
        model_identity: dict[str, Any]
        if self.spec.mode == "model-evaluation":
            savepoint, model_path = self._resolve_evaluation_model(
                self.spec.model_path or ""
            )
            aiserver_arguments.extend(
                [
                    "--volume",
                    f"{savepoint}:/opt/rl/aiserver/models/local:ro",
                ]
            )
            model_identity = {
                "source": "external",
                "path": str(model_path),
                "savepoint": str(savepoint),
                "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            }
        else:
            model_identity = {"source": "embedded-smoke"}

        self._run_container(
            "aiserver",
            self.spec.image_ref("aiserver"),
            aiserver_arguments,
        )
        self._wait_healthy("aiserver", 60)
        if self.spec.mode == "local-test":
            manifest_result = self._docker(
                [
                    "exec",
                    self._name("aiserver"),
                    "cat",
                    "/opt/rl/aiserver/models/smoke/manifest.json",
                ]
            )
            manifest = json.loads(manifest_result.stdout)
            checksum_result = self._docker(
                [
                    "exec",
                    self._name("aiserver"),
                    "sha256sum",
                    "/opt/rl/aiserver/models/smoke/model_v000000.onnx",
                ]
            )
            actual_checksum = checksum_result.stdout.split()[0]
            if actual_checksum != manifest.get("sha256"):
                raise RuntimeFailure(
                    "embedded smoke model checksum does not match its manifest"
                )
            model_identity = {
                "source": "embedded-smoke",
                "version": manifest.get("model_version"),
                "sha256": actual_checksum,
                "seed": manifest.get("seed"),
            }
        self._record("aiserver_ready")
        self._atomic_json(
            replay_dir / "validation-manifest.json",
            {
                "schema_version": 1,
                "run_id": self.spec.run_id,
                "mode": self.spec.mode,
                "model": model_identity,
                "parameters": {
                    "agents": self.spec.agents,
                    "episodes": self.spec.episodes,
                    "max_steps": self.spec.max_steps,
                },
            },
        )

        self._run_container(
            "client",
            self.spec.image_ref("client"),
            [
                *self._base_env(),
                "--env",
                "MAZE_VALIDATION_RESULT_PATH=/opt/rl/maze-client/replays/client-result.json",
                "--env",
                "MAZE_AISERVER_HOST=maze-aiserver",
                "--env",
                f"MAZE_AGENT_NUM={self.spec.agents}",
                "--env",
                f"MAZE_MAX_EPISODES={self.spec.episodes}",
                "--env",
                f"MAZE_MAX_STEPS={self.spec.max_steps}",
                "--env",
                "MAZE_VIZ_OUTPUT_DIR=/opt/rl/maze-client/replays",
                "--publish",
                "127.0.0.1:9004:9004",
                "--volume",
                f"{replay_dir}:/opt/rl/maze-client/replays",
            ],
        )
        self._ensure_host_port(9004, 30)
        result_path = replay_dir / "client-result.json"
        self._wait_file(result_path, "client", 180)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result.get("exit_code", -1)) != 0:
            raise RuntimeFailure(f"validation client failed: {result}")
        deadline = time.monotonic() + 10
        status: dict[str, Any] = {}
        while time.monotonic() < deadline:
            status = self._http_json("http://127.0.0.1:9004/api/status")
            if int(status.get("file_count", 0)) > 0:
                break
            time.sleep(0.5)
        if int(status.get("file_count", 0)) == 0:
            raise RuntimeFailure("validation completed without a replay file")
        files = self._http_json("http://127.0.0.1:9004/api/files")
        required_frames = min(4, self.spec.max_steps)
        available_frames = max(
            (int(item.get("frame_count", 0)) for item in files),
            default=0,
        )
        if available_frames < required_frames:
            raise RuntimeFailure(
                f"replay has {available_frames} frames; "
                f"expected at least {required_frames}"
            )
        self._record("replay_ready")
        return {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "mode": self.spec.mode,
            "actual_status": "ready",
            "client_result": result,
            "replay": status,
            "model": model_identity,
            "ports": {"replay": 9004},
            "internal_ports": {"aiserver": 9002},
        }

    def stop(self) -> dict[str, Any]:
        self._record("stop_requested")
        try:
            self.stop_run(require_quiesce=self.spec.mode == "training")
        except RuntimeFailure as exc:
            self._record("stop_failed", str(exc))
            self._write_state("stop_failed", error=str(exc))
            raise
        self._record("stopped")
        previous_result = None
        state_path = self.run_root / "state.json"
        if state_path.is_file():
            try:
                previous_result = json.loads(
                    state_path.read_text(encoding="utf-8")
                ).get("result")
            except (OSError, json.JSONDecodeError):
                previous_result = None
        if previous_result is None:
            self._write_state("stopped")
        else:
            self._write_state("stopped", result=previous_result)
        return {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "actual_status": "stopped",
        }

    def status(self) -> dict[str, Any]:
        state_path = self.run_root / "state.json"
        persisted = {}
        if state_path.is_file():
            try:
                persisted = json.loads(state_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                persisted = {"actual_status": "invalid-runtime-state"}
        containers = {
            role: self._container_state(role)
            for role in ("learner", "aiserver", "client")
        }
        metrics = {}
        if self.spec.mode == "training" and containers["learner"].get("running"):
            try:
                metrics = self._http_json(
                    "http://127.0.0.1:9005/api/metrics/latest"
                ).get("record", {})
            except RuntimeFailure:
                metrics = {}
        return {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "mode": self.spec.mode,
            "persisted": persisted,
            "containers": containers,
            "metrics": metrics,
        }
