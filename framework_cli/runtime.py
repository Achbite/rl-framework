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
AISERVER_LOCAL_TEST_MODEL_DIR = "/opt/rl/aiserver/models/local-test"
LEARNER_FINALIZE_REQUEST = "/tmp/rl-training-finalize"
LEARNER_FINALIZE_COMPLETE = "/tmp/rl-training-finalized"


class RuntimeFailure(RuntimeError):
    pass


def learner_monitor_transport_readiness(
    host_status: dict[str, Any],
    container_status: dict[str, Any],
    *,
    minimum_timestamp: float = 0.0,
    now: float | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    current_time = time.time() if now is None else now
    required_identity = (
        "service_instance_id",
        "metrics_source_id",
        "started_at",
        "metrics_dir",
    )

    for source, document in (
        ("host", host_status),
        ("container", container_status),
    ):
        if not isinstance(document, dict) or not document:
            reasons.append(f"{source}_status_missing")
            continue
        if document.get("schema_version") != 1:
            reasons.append(f"{source}_schema_mismatch")
        if document.get("service") != "learner-metrics":
            reasons.append(f"{source}_service_mismatch")
        if document.get("stream") != "current":
            reasons.append(f"{source}_stream_mismatch")
        if document.get("mode") != "training":
            reasons.append(f"{source}_mode_mismatch")
        if document.get("stale") is not False:
            reasons.append(f"{source}_metrics_stale")
        for field in required_identity:
            if document.get(field) in (None, ""):
                reasons.append(f"{source}_{field}_missing")
        if not str(document.get("metrics_dir", "")).endswith(
            "/models/local-train/metrics"
        ):
            reasons.append(f"{source}_metrics_dir_mismatch")
        try:
            started_at = float(document.get("started_at", 0.0))
        except (TypeError, ValueError):
            started_at = 0.0
            reasons.append(f"{source}_started_at_invalid")
        try:
            latest_timestamp = float(
                document.get("latest_timestamp", 0.0)
            )
        except (TypeError, ValueError):
            latest_timestamp = 0.0
            reasons.append(f"{source}_latest_timestamp_invalid")
        try:
            latest_sequence = int(document.get("latest_sequence", 0))
        except (TypeError, ValueError):
            latest_sequence = 0
            reasons.append(f"{source}_latest_sequence_invalid")
        if started_at < minimum_timestamp:
            reasons.append(f"{source}_service_predates_launch")
        if latest_timestamp < started_at:
            reasons.append(f"{source}_metrics_predate_service")
        if latest_timestamp > current_time + 5.0:
            reasons.append(f"{source}_metrics_timestamp_future")
        if latest_sequence < 1:
            reasons.append(f"{source}_latest_sequence_missing")

    if host_status and container_status:
        for field in required_identity:
            if host_status.get(field) != container_status.get(field):
                reasons.append(f"monitor_{field}_mismatch")
        try:
            host_sequence = int(host_status.get("latest_sequence", 0))
            container_sequence = int(
                container_status.get("latest_sequence", 0)
            )
        except (TypeError, ValueError):
            host_sequence = 0
            container_sequence = 0
        if host_sequence > container_sequence:
            reasons.append("monitor_sequence_ahead_of_container")

    return {
        "ready": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "service_instance_id": host_status.get(
            "service_instance_id", ""
        ),
        "metrics_source_id": host_status.get("metrics_source_id", ""),
    }


def training_record_readiness(
    record: dict[str, Any],
    *,
    minimum_timestamp: float = 0.0,
    now: float | None = None,
    container_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    current_time = time.time() if now is None else now
    if not record:
        return {
            "ready": False,
            "reasons": ["metrics_missing"],
            "timestamp": 0.0,
            "age_seconds": None,
            "model_lag": None,
        }

    try:
        timestamp = float(record.get("timestamp", 0.0))
    except (TypeError, ValueError):
        timestamp = 0.0
    try:
        interval_seconds = max(
            0.0, float(record.get("interval_ms", 0.0)) / 1000.0
        )
    except (TypeError, ValueError):
        interval_seconds = 0.0
    stale_after = max(5.0, 3.0 * interval_seconds)
    age_seconds = max(0.0, current_time - timestamp) if timestamp else None
    if timestamp < minimum_timestamp:
        reasons.append("metrics_predate_launch")
    if timestamp > current_time + 5.0:
        reasons.append("metrics_timestamp_future")
    if age_seconds is None or age_seconds > stale_after:
        reasons.append("metrics_stale")
    if record.get("mode") != "training":
        reasons.append("metrics_mode_mismatch")

    actor = record.get("actor") if isinstance(record.get("actor"), dict) else {}
    distributor = (
        record.get("distributor")
        if isinstance(record.get("distributor"), dict)
        else {}
    )
    learner = (
        record.get("learner")
        if isinstance(record.get("learner"), dict)
        else {}
    )
    model = record.get("model") if isinstance(record.get("model"), dict) else {}
    chain = record.get("chain") if isinstance(record.get("chain"), dict) else {}

    if not actor.get("ready"):
        reasons.append("actor_not_ready")
    if not actor.get("instance_id"):
        reasons.append("actor_instance_missing")
    for field in ("ready", "ingress_ready", "pool_ready"):
        if not distributor.get(field):
            reasons.append(f"sample_pool_{field}_false")
    if not distributor.get("instance_id"):
        reasons.append("sample_pool_instance_missing")
    if not model.get("ready"):
        reasons.append("model_distributor_not_ready")
    if learner.get("error"):
        reasons.append("learner_error")
    if chain.get("ready") is not True:
        reasons.append("learner_chain_not_ready")

    def integer(document: dict, field: str, default: int = -1) -> int:
        try:
            return int(document.get(field, default))
        except (TypeError, ValueError):
            return default

    def model_identity(document: dict, field: str) -> dict[str, Any]:
        value = document.get(field)
        return value if isinstance(value, dict) else {}

    def valid_model_identity(identity: dict[str, Any]) -> bool:
        version = identity.get("model_version")
        return (
            isinstance(identity.get("model_lineage_id"), str)
            and bool(identity.get("model_lineage_id"))
            and isinstance(version, int)
            and not isinstance(version, bool)
            and version >= 0
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(identity.get("artifact_digest", "")),
            )
            is not None
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(identity.get("manifest_digest", "")),
            )
            is not None
        )

    learner_identity = model_identity(learner, "model_identity")
    actor_identity = model_identity(actor, "model_identity")
    latest_identity = model_identity(model, "latest_model_identity")
    latest_ack_identity = model_identity(
        model, "latest_ack_model_identity"
    )
    for name, identity in (
        ("learner_model_identity_invalid", learner_identity),
        ("actor_model_identity_invalid", actor_identity),
        ("published_model_identity_invalid", latest_identity),
        ("actor_ack_identity_invalid", latest_ack_identity),
    ):
        if not valid_model_identity(identity):
            reasons.append(name)

    learner_version = integer(learner_identity, "model_version")
    actor_version = integer(actor_identity, "model_version")
    max_policy_lag = integer(learner, "max_policy_lag")
    model_lag = learner_version - actor_version
    if latest_identity != learner_identity:
        reasons.append("published_model_identity_mismatch")
    if (
        actor_version < 0
        or learner_version < 0
        or max_policy_lag < 0
        or model_lag < 0
        or model_lag > max_policy_lag
    ):
        reasons.append("actor_model_lag_invalid")
    if latest_ack_identity != actor_identity:
        reasons.append("actor_model_ack_mismatch")
    if model.get("latest_ack_status") != "MODEL_LOAD_STATUS_LOADED":
        reasons.append("actor_model_ack_not_loaded")

    if container_record is not None:
        if not container_record:
            reasons.append("container_metrics_unreachable")
        else:
            container_actor = (
                container_record.get("actor")
                if isinstance(container_record.get("actor"), dict)
                else {}
            )
            container_distributor = (
                container_record.get("distributor")
                if isinstance(container_record.get("distributor"), dict)
                else {}
            )
            if actor.get("instance_id") != container_actor.get("instance_id"):
                reasons.append("container_actor_identity_mismatch")
            if distributor.get("instance_id") != container_distributor.get(
                "instance_id"
            ):
                reasons.append("container_sample_pool_identity_mismatch")
            if integer(record, "sequence", 0) > integer(
                container_record, "sequence", 0
            ):
                reasons.append("container_metrics_sequence_mismatch")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "timestamp": timestamp,
        "age_seconds": age_seconds,
        "stale_after_seconds": stale_after,
        "model_lag": model_lag,
        "actor_instance_id": actor.get("instance_id", ""),
        "sample_pool_instance_id": distributor.get("instance_id", ""),
    }


def training_completion_status(
    record: dict[str, Any], expected_samples: int
) -> dict[str, Any]:
    actor = record.get("actor") if isinstance(record.get("actor"), dict) else {}
    distributor = (
        record.get("distributor")
        if isinstance(record.get("distributor"), dict)
        else {}
    )
    learner = (
        record.get("learner")
        if isinstance(record.get("learner"), dict)
        else {}
    )

    def integer(document: dict[str, Any], field: str, default: int = 0) -> int:
        try:
            return int(document.get(field, default))
        except (TypeError, ValueError):
            return default

    counts = {
        "expected": int(expected_samples),
        "produced": integer(actor, "produced"),
        "outbound_pending": integer(actor, "outbound_pending"),
        "final_drop": integer(actor, "final_drop"),
        "accepted": integer(distributor, "accepted"),
        "acked": integer(distributor, "acked"),
        "ready": integer(distributor, "ready_samples"),
        "leased": integer(distributor, "leased_samples"),
        "trained": integer(distributor, "trained"),
        "stale": integer(distributor, "stale"),
        "invalid": integer(distributor, "invalid"),
        "shutdown_untrained": integer(distributor, "shutdown_untrained"),
        "learner_run_trained": integer(learner, "run_trained_samples", -1),
        "learner_run_updates": integer(learner, "run_train_updates", -1),
        "train_batch_size": integer(learner, "train_batch_size", -1),
        "max_train_batch_size": integer(
            learner, "max_train_batch_size", -1
        ),
        "initial_model_version": integer(
            learner, "initial_model_version", -1
        ),
        "model_version": integer(learner, "model_version", -1),
    }
    reasons: list[str] = []
    fatal_reasons: list[str] = []

    for field in ("stale", "invalid", "shutdown_untrained"):
        if counts[field] != 0:
            reason = f"non_trained_disposition_{field}"
            reasons.append(reason)
            fatal_reasons.append(reason)
    if counts["final_drop"] != 0:
        reasons.append("actor_final_drop_nonzero")
        fatal_reasons.append("actor_final_drop_nonzero")

    for field in ("produced", "accepted", "acked", "trained"):
        if counts[field] > expected_samples:
            reason = f"{field}_exceeds_expected"
            reasons.append(reason)
            fatal_reasons.append(reason)
    if counts["learner_run_trained"] > expected_samples:
        reasons.append("learner_run_trained_exceeds_expected")
        fatal_reasons.append("learner_run_trained_exceeds_expected")

    if counts["produced"] != (
        counts["accepted"]
        + counts["outbound_pending"]
        + counts["final_drop"]
    ):
        reasons.append("actor_sample_conservation_mismatch")
    if counts["accepted"] != (
        counts["ready"] + counts["leased"] + counts["acked"]
    ):
        reasons.append("sample_pool_resident_conservation_mismatch")
    if counts["acked"] != (
        counts["trained"]
        + counts["stale"]
        + counts["invalid"]
        + counts["shutdown_untrained"]
    ):
        reasons.append("ack_disposition_conservation_mismatch")

    target_batch = counts["train_batch_size"]
    max_batch = counts["max_train_batch_size"]
    if target_batch <= 0 or max_batch < target_batch:
        minimum_updates = maximum_updates = -1
        reasons.append("bounded_batch_contract_invalid")
    else:
        minimum_updates = (expected_samples + max_batch - 1) // max_batch
        maximum_updates = (
            expected_samples + target_batch - 1
        ) // target_batch
    counts["expected_updates_min"] = minimum_updates
    counts["expected_updates_max"] = maximum_updates
    if not (
        minimum_updates >= 0
        and minimum_updates
        <= counts["learner_run_updates"]
        <= maximum_updates
    ):
        reasons.append("learner_run_update_count_mismatch")
    if counts["learner_run_trained"] != expected_samples:
        reasons.append("learner_run_trained_count_mismatch")
    if (
        counts["model_version"] - counts["initial_model_version"]
        != counts["learner_run_updates"]
    ):
        reasons.append("model_version_delta_mismatch")

    for field in ("produced", "accepted", "acked", "trained"):
        if counts[field] != expected_samples:
            reasons.append(f"{field}_not_expected")
    if counts["outbound_pending"] != 0:
        reasons.append("actor_outbound_pending_nonzero")
    if counts["ready"] != 0:
        reasons.append("sample_pool_ready_nonzero")
    if counts["leased"] != 0:
        reasons.append("sample_pool_leased_nonzero")

    reasons = list(dict.fromkeys(reasons))
    return {
        "complete": not reasons,
        "failed": bool(fatal_reasons),
        "reasons": reasons,
        "fatal_reasons": list(dict.fromkeys(fatal_reasons)),
        "counts": counts,
    }


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

    @staticmethod
    def _embedded_smoke_model_identity(
        manifest: dict[str, Any], actual_checksum: str
    ) -> tuple[str, dict[str, Any]]:
        identity = manifest.get("identity")
        model_file = manifest.get("model_file")
        if (
            manifest.get("manifest_schema_version") != 1
            or manifest.get("package") != "rl-smoke-model"
            or manifest.get("ready") is not True
            or not isinstance(identity, dict)
            or not isinstance(model_file, str)
            or not model_file
            or Path(model_file).name != model_file
            or not model_file.endswith(".onnx")
        ):
            raise RuntimeFailure("embedded smoke model manifest is invalid")
        artifact_digest = identity.get("artifact_digest")
        model_version = identity.get("model_version")
        lineage_id = identity.get("model_lineage_id")
        if (
            not isinstance(artifact_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", artifact_digest) is None
            or not isinstance(model_version, int)
            or isinstance(model_version, bool)
            or model_version < 0
            or not isinstance(lineage_id, str)
            or not lineage_id
            or actual_checksum != artifact_digest
        ):
            raise RuntimeFailure(
                "embedded smoke model checksum or identity is invalid"
            )
        return model_file, {
            "source": "embedded-smoke",
            "package_version": manifest.get("version"),
            "version": model_version,
            "sha256": artifact_digest,
            "seed": manifest.get("seed"),
            "model_lineage_id": lineage_id,
        }

    @staticmethod
    def _runtime_model_identity(
        manifest: dict[str, Any],
        *,
        version: int,
        checksum: str,
        model_file: str,
    ) -> dict[str, Any]:
        contract = manifest.get("contract")
        identity = manifest.get("identity")
        if (
            manifest.get("manifest_schema_version") != 1
            or not isinstance(contract, dict)
            or contract.get("package_name") != "rl-contracts"
            or contract.get("package_version") != "0.10.0"
            or not isinstance(identity, dict)
            or identity.get("model_version") != version
            or identity.get("artifact_digest") != checksum
            or manifest.get("model_file") != model_file
            or manifest.get("ready") is not True
            or "run_id" in manifest
        ):
            raise RuntimeFailure(
                f"runtime model v{version} identity is invalid"
            )
        for field in ("artifact_digest", "manifest_digest"):
            if re.fullmatch(
                r"[0-9a-f]{64}", str(identity.get(field, ""))
            ) is None:
                raise RuntimeFailure(
                    f"runtime model v{version} identity is invalid"
                )
        if not identity.get("model_lineage_id"):
            raise RuntimeFailure(
                f"runtime model v{version} identity is invalid"
            )
        return identity

    @staticmethod
    def _archive_model_identity(
        manifest: dict[str, Any], *, version: int, checksum: str
    ) -> dict[str, Any]:
        identity = manifest.get("runtime_manifest_identity")
        if (
            manifest.get("schema_version") != 1
            or not isinstance(identity, dict)
            or identity.get("model_version") != version
            or identity.get("artifact_digest") != checksum
            or manifest.get("runtime_manifest_digest")
            != identity.get("manifest_digest")
            or manifest.get("model_file") != EVALUATION_MODEL_FILE
            or manifest.get("checkpoint_file") != "checkpoint.pt"
            or manifest.get("artifact_digest") != checksum
            or "run_id" in manifest
        ):
            raise RuntimeFailure(
                f"archive v{version} identity is invalid"
            )
        for field in ("artifact_digest", "manifest_digest"):
            if re.fullmatch(
                r"[0-9a-f]{64}", str(identity.get(field, ""))
            ) is None:
                raise RuntimeFailure(
                    f"archive v{version} identity is invalid"
                )
        if not identity.get("model_lineage_id"):
            raise RuntimeFailure(
                f"archive v{version} identity is invalid"
            )
        return identity

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

    def _ensure_host_port(
        self,
        port: int,
        timeout: float,
        *,
        record_transport: bool = True,
    ) -> str:
        direct_deadline = time.monotonic() + min(timeout, 3.0)
        while time.monotonic() < direct_deadline:
            if self._tcp_ready(port):
                transport = "docker-publish"
                if record_transport:
                    self._record_port_transport(port, transport)
                return transport
            time.sleep(0.2)

        config_path = self._colima_ssh_config()
        ssh = shutil.which("ssh")
        if config_path is None or ssh is None:
            self._wait_tcp(port, max(0.0, timeout - 3.0))
            transport = "docker-publish"
            if record_transport:
                self._record_port_transport(port, transport)
            return transport

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
        transport = "colima-ssh-tunnel"
        if record_transport:
            self._record_port_transport(port, transport)
        return transport

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

    def _container_metrics_latest(self) -> dict[str, Any]:
        script = (
            "import sys,urllib.request;"
            "sys.stdout.write(urllib.request.urlopen("
            "sys.argv[1],timeout=2).read().decode())"
        )
        result = self._docker(
            [
                "exec",
                self._name("learner"),
                "python3",
                "-c",
                script,
                "http://127.0.0.1:9005/api/metrics/latest",
            ],
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeFailure(
                f"Learner container metrics are unavailable: {detail}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeFailure(
                "Learner container metrics returned invalid JSON"
            ) from exc
        record = payload.get("record") if isinstance(payload, dict) else None
        return record if isinstance(record, dict) else {}

    def _container_metrics_status(self) -> dict[str, Any]:
        script = (
            "import sys,urllib.request;"
            "sys.stdout.write(urllib.request.urlopen("
            "sys.argv[1],timeout=2).read().decode())"
        )
        result = self._docker(
            [
                "exec",
                self._name("learner"),
                "python3",
                "-c",
                script,
                "http://127.0.0.1:9005/api/status",
            ],
            check=False,
            timeout=5,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeFailure(
                f"Learner container metrics status is unavailable: {detail}"
            )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeFailure(
                "Learner container metrics status returned invalid JSON"
            ) from exc
        return payload if isinstance(payload, dict) else {}

    def _wait_learner_monitor_transport(
        self, minimum_timestamp: float, timeout: float
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last_readiness = learner_monitor_transport_readiness(
            {}, {}, minimum_timestamp=minimum_timestamp
        )
        endpoint_seen = False
        while time.monotonic() < deadline:
            try:
                host_status = self._http_json(
                    "http://127.0.0.1:9005/api/status"
                )
                endpoint_seen = True
            except RuntimeFailure:
                host_status = {}
            try:
                container_status = self._container_metrics_status()
            except RuntimeFailure:
                container_status = {}
            last_readiness = learner_monitor_transport_readiness(
                host_status,
                container_status,
                minimum_timestamp=minimum_timestamp,
            )
            if last_readiness["ready"]:
                return last_readiness
            time.sleep(0.2)
        prefix = (
            "PORT_IDENTITY_CONFLICT"
            if endpoint_seen
            else "MONITOR_TARGET_UNAVAILABLE"
        )
        raise RuntimeFailure(
            f"{prefix}: learner monitor transport verification failed; "
            f"reasons={last_readiness['reasons']}"
        )

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
            completion = training_completion_status(last_record, expected)
            if completion["complete"]:
                return last_record
            if completion["failed"]:
                raise RuntimeFailure(
                    "training completion gate failed: "
                    + ",".join(completion["fatal_reasons"])
                )
            time.sleep(0.5)
        raise RuntimeFailure(
            f"training loop timeout; expected={expected}, last={last_record}"
        )

    def _request_training_finalize(self, timeout: float) -> None:
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
                time.sleep(0.2)
                continue
            actor = last_record.get("actor", {})
            distributor = last_record.get("distributor", {})
            produced = int(actor.get("produced", -1))
            actor_accepted = int(actor.get("accepted", -1))
            pool_accepted = int(distributor.get("accepted", -1))
            outbound = int(actor.get("outbound_pending", -1))
            final_drop = int(actor.get("final_drop", -1))
            if any(
                value > expected
                for value in (produced, actor_accepted, pool_accepted)
            ) or final_drop > 0:
                raise RuntimeFailure(
                    "producer completion exceeded its finite Run contract"
                )
            if (
                produced == expected
                and actor_accepted == expected
                and pool_accepted == expected
                and outbound == 0
                and final_drop == 0
            ):
                break
            time.sleep(0.2)
        else:
            raise RuntimeFailure(
                "producer did not flush the finite Run before final drain: "
                f"{last_record}"
            )

        request = self._docker(
            [
                "exec",
                self._name("learner"),
                "touch",
                LEARNER_FINALIZE_REQUEST,
            ],
            check=False,
        )
        if request.returncode != 0:
            detail = request.stderr.strip() or request.stdout.strip()
            raise RuntimeFailure(
                f"Learner finalize request failed: {detail}"
            )
        self._record("training_finalize_requested")
        while time.monotonic() < deadline:
            self._require_running("learner")
            marker = self._docker(
                [
                    "exec",
                    self._name("learner"),
                    "test",
                    "-f",
                    LEARNER_FINALIZE_COMPLETE,
                ],
                check=False,
            )
            if marker.returncode == 0:
                self._record("training_finalize_complete")
                return
            time.sleep(0.2)
        raise RuntimeFailure("Learner final sample drain timed out")

    def _wait_training_ready(
        self, minimum_timestamp: float, timeout: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        deadline = time.monotonic() + timeout
        url = "http://127.0.0.1:9005/api/metrics/latest"
        last_record: dict[str, Any] = {}
        last_readiness = training_record_readiness(
            last_record, minimum_timestamp=minimum_timestamp
        )
        while time.monotonic() < deadline:
            self._require_running("learner")
            self._require_running("aiserver")
            client = self._container_state("client")
            if (
                client.get("exists")
                and not client.get("running")
                and int(client.get("exit_code", 0)) != 0
            ):
                raise RuntimeFailure(
                    f"client exited before training readiness: {client}"
                )
            try:
                payload = self._http_json(url)
                last_record = payload.get("record", {})
            except RuntimeFailure:
                time.sleep(0.3)
                continue
            try:
                container_record = self._container_metrics_latest()
            except RuntimeFailure:
                container_record = {}
            last_readiness = training_record_readiness(
                last_record,
                minimum_timestamp=minimum_timestamp,
                container_record=container_record,
            )
            if last_readiness["ready"]:
                return last_record, last_readiness
            time.sleep(0.3)
        raise RuntimeFailure(
            "training readiness timeout; "
            f"reasons={last_readiness['reasons']}, last={last_record}"
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
        archive_identities: dict[int, dict[str, Any]] = {}
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
            archive_identities[version] = self._archive_model_identity(
                manifest, version=version, checksum=checksum
            )
            archive_manifests[version] = manifest
            if version == 0:
                identity_checksums.append(checksum)

        for version in (1, 2):
            receipt = receipts.get(version)
            identity = (receipt or {}).get("model")
            if (
                not isinstance(identity, dict)
                or identity.get("model_version") != version
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(identity.get("artifact_digest", "")),
                )
                is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(identity.get("manifest_digest", "")),
                )
                is None
                or not identity.get("model_lineage_id")
            ):
                raise RuntimeFailure(
                    f"model v{version} receipt identity is missing"
                )
            identity_checksums.append(identity["artifact_digest"])
        if len(set(identity_checksums)) != 3:
            raise RuntimeFailure("v0, v1, and v2 checksums must differ")
        if (
            archive_identities[2]["artifact_digest"]
            != receipts[2]["model"]["artifact_digest"]
        ):
            raise RuntimeFailure("archive v2 and receipt identity differ")

        def behavior_version_range(
            receipt: dict[str, Any], *, target_version: int
        ) -> tuple[int, int]:
            behavior = receipt.get("behavior_model")
            models = (
                behavior.get("models")
                if isinstance(behavior, dict)
                else None
            )
            minimum = (
                behavior.get("minimum_model_version")
                if isinstance(behavior, dict)
                else None
            )
            maximum = (
                behavior.get("maximum_model_version")
                if isinstance(behavior, dict)
                else None
            )
            target_identity = receipt.get("model")
            if (
                receipt.get("target_model_version") != target_version
                or not isinstance(target_identity, dict)
                or not isinstance(behavior, dict)
                or not isinstance(models, list)
                or not models
                or not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or not isinstance(maximum, int)
                or isinstance(maximum, bool)
            ):
                raise RuntimeFailure(
                    f"model v{target_version} behavior identity is invalid"
                )

            versions: list[int] = []
            for identity in models:
                version = (
                    identity.get("model_version")
                    if isinstance(identity, dict)
                    else None
                )
                if (
                    not isinstance(identity, dict)
                    or not isinstance(version, int)
                    or isinstance(version, bool)
                    or version < 0
                    or identity.get("model_lineage_id")
                    != behavior.get("model_lineage_id")
                    or identity.get("model_lineage_id")
                    != target_identity.get("model_lineage_id")
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(identity.get("artifact_digest", "")),
                    )
                    is None
                    or re.fullmatch(
                        r"[0-9a-f]{64}",
                        str(identity.get("manifest_digest", "")),
                    )
                    is None
                ):
                    raise RuntimeFailure(
                        f"model v{target_version} behavior identity is invalid"
                    )
                versions.append(version)

            reference_version = target_version - 1
            if (
                len(versions) != len(set(versions))
                or minimum != min(versions)
                or maximum != max(versions)
                or minimum < 0
                or maximum > reference_version
                or reference_version - minimum > 2
            ):
                raise RuntimeFailure(
                    f"model v{target_version} receipt has an invalid "
                    "behavior policy range"
                )
            return minimum, maximum

        behavior_ranges = {
            version: behavior_version_range(
                receipts[version], target_version=version
            )
            for version in (1, 2)
        }
        v2_receipt = receipts[2]
        v2_update_id = v2_receipt.get("train_update_id")
        if not v2_update_id:
            raise RuntimeFailure("model v2 update receipt is missing")

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
            self._runtime_model_identity(
                manifest,
                version=version,
                checksum=checksum,
                model_file=model_path.name,
            )
            if not checkpoint_path.is_file():
                raise RuntimeFailure(
                    f"runtime model v{version} checkpoint is missing"
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
            "v2_behavior_model_version_range": list(behavior_ranges[2]),
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
            "RL_AISERVER_ID=aiserver-0",
            "--env",
            "RL_CLIENT_INSTANCE_ID=maze-client-0",
            "--env",
            "RL_ENVIRONMENT_INSTANCE_ID=maze-env-0",
        ]

    def start(self, *, wait_for_completion: bool = False) -> dict[str, Any]:
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
                result = self._start_training(
                    wait_for_completion=wait_for_completion
                )
            else:
                result = self._start_replay_workload()
            self._write_state("ready", result=result)
            return result
        except Exception as exc:
            self._record("failed", str(exc))
            self._write_state("failed", error=str(exc))
            self.stop_run()
            raise

    def _start_training(
        self, *, wait_for_completion: bool = False
    ) -> dict[str, Any]:
        launch_started_at = time.time()
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
            "RL_LEARNER_INSTANCE=learner-0",
            "--env",
            "RL_AISERVER_HOST=maze-aiserver",
            "--env",
            "RL_LOCAL_TRAIN_ROOT=/opt/rl/learner/models/local-train",
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
                ["--env", "RL_ARCHIVE_INTERVAL_UPDATES=2"]
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
                    f"RL_INITIAL_CHECKPOINT={container_checkpoint}",
                ]
            )
        self._run_container(
            "learner",
            self.spec.image_ref("learner"),
            learner_arguments,
        )
        self._wait_healthy("learner", 180)
        monitor_transport = self._ensure_host_port(
            9005, 180, record_transport=False
        )

        self._run_container(
            "aiserver",
            self.spec.image_ref("aiserver"),
            [
                *self._base_env(),
                "--env",
                "RL_MODEL_DISTRIBUTOR_HOST=maze-learner",
                "--env",
                "RL_SAMPLE_DISTRIBUTOR_HOST=maze-learner",
                "--env",
                "RL_LOCAL_TRAIN_ROOT=/opt/rl/aiserver/models/local-train",
                "--env",
                f"RL_TRAINING_SAMPLE_BUDGET={self.spec.expected_samples}",
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
        self._wait_learner_monitor_transport(launch_started_at, 15)
        self._record_port_transport(9005, monitor_transport)
        self._record("learner_pod_ready")
        self._record("aiserver_ready")

        self._run_container(
            "client",
            self.spec.image_ref("client"),
            [
                *self._base_env(),
                "--env",
                "RL_AISERVER_HOST=maze-aiserver",
                "--volume",
                f"{client_logs}:/opt/rl/maze-client/log",
            ],
        )
        self._record("client_started")
        readiness_record, readiness = self._wait_training_ready(
            launch_started_at, 90
        )
        self._record("training_chain_ready")
        if not wait_for_completion:
            actor = readiness_record.get("actor", {})
            distributor = readiness_record.get("distributor", {})
            learner = readiness_record.get("learner", {})
            client_state = self._container_state("client")
            self._record("training_ready")
            monitor_source = (
                "local-dev" if self.runtime_scope == "local-tests" else "platform"
            )
            return {
                "schema_version": 1,
                "task_id": self.spec.task_id,
                "run_id": self.spec.run_id,
                "mode": self.spec.mode,
                "actual_status": "ready",
                "phase": (
                    "running"
                    if client_state.get("running")
                    else "workload-completed"
                ),
                "readiness": readiness,
                "counts": {
                    "expected": self.spec.expected_samples,
                    "produced": int(actor.get("produced", 0)),
                    "accepted": int(distributor.get("accepted", 0)),
                    "acked": int(distributor.get("acked", 0)),
                    "final_drop": int(actor.get("final_drop", 0)),
                },
                "train_updates": int(learner.get("train_updates", 0)),
                "model_version": int(learner.get("model_version", 0)),
                "monitor_url": (
                    "http://127.0.0.1:9005/"
                    if self.runtime_scope == "local-tests"
                    else (
                        "http://127.0.0.1:9010/infra?"
                        f"source={monitor_source}&run_id="
                        f"{urllib.parse.quote(self.spec.run_id)}"
                    )
                ),
                "ports": {"learner_metrics": 9005},
                "internal_ports": {
                    "aiserver": 9002,
                    "local_sample_service": 9100,
                    "model_distributor": 9200,
                },
            }
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

        self._request_training_finalize(90)

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
        completion = training_completion_status(
            metrics, self.spec.expected_samples
        )
        if not completion["complete"]:
            raise RuntimeFailure(
                "training completion changed before artifact validation: "
                + ",".join(completion["reasons"])
            )
        counts = completion["counts"]
        self._atomic_json(
            self.run_root / "sample-conservation.json",
            {"schema_version": 2, "run_id": self.spec.run_id, **counts},
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
                "http://127.0.0.1:9005/"
                if self.runtime_scope == "local-tests"
                else (
                    "http://127.0.0.1:9010/infra?source=platform&run_id="
                    f"{urllib.parse.quote(self.spec.run_id)}"
                )
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
                    f"{AISERVER_LOCAL_TEST_MODEL_DIR}/manifest.json",
                ]
            )
            manifest = json.loads(manifest_result.stdout)
            model_file = manifest.get("model_file")
            if (
                not isinstance(model_file, str)
                or Path(model_file).name != model_file
            ):
                raise RuntimeFailure("embedded smoke model manifest is invalid")
            checksum_result = self._docker(
                [
                    "exec",
                    self._name("aiserver"),
                    "sha256sum",
                    f"{AISERVER_LOCAL_TEST_MODEL_DIR}/{model_file}",
                ]
            )
            actual_checksum = checksum_result.stdout.split()[0]
            _, model_identity = self._embedded_smoke_model_identity(
                manifest, actual_checksum
            )
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
                "RL_VALIDATION_RESULT_PATH=/opt/rl/maze-client/replays/client-result.json",
                "--env",
                "RL_AISERVER_HOST=maze-aiserver",
                "--env",
                f"RL_VALIDATION_ID={self.spec.run_id}",
                "--env",
                "RL_VIZ_OUTPUT_DIR=/opt/rl/maze-client/replays",
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
        readiness: dict[str, Any] = {}
        if self.spec.mode == "training" and containers["learner"].get("running"):
            try:
                metrics = self._http_json(
                    "http://127.0.0.1:9005/api/metrics/latest"
                ).get("record", {})
            except RuntimeFailure:
                metrics = {}
            try:
                container_metrics = self._container_metrics_latest()
            except RuntimeFailure:
                container_metrics = {}
            persisted_readiness = (
                persisted.get("result", {}).get("readiness", {})
                if isinstance(persisted.get("result"), dict)
                else {}
            )
            minimum_timestamp = max(
                0.0,
                float(persisted_readiness.get("timestamp", 0.0)) - 0.001,
            )
            readiness = training_record_readiness(
                metrics,
                minimum_timestamp=minimum_timestamp,
                container_record=container_metrics,
            )
        actual_status = persisted.get("actual_status", "unknown")
        if self.spec.mode == "training" and actual_status in {
            "starting",
            "ready",
            "degraded",
        }:
            required_running = all(
                containers[role].get("running")
                for role in ("learner", "aiserver")
            )
            client_failed = bool(
                containers["client"].get("exists")
                and not containers["client"].get("running")
                and int(containers["client"].get("exit_code", 0)) != 0
            )
            if required_running and readiness.get("ready") and not client_failed:
                actual_status = "ready"
            elif actual_status != "starting":
                actual_status = "degraded"
        return {
            "schema_version": 1,
            "task_id": self.spec.task_id,
            "run_id": self.spec.run_id,
            "mode": self.spec.mode,
            "actual_status": actual_status,
            "persisted": persisted,
            "containers": containers,
            "metrics": metrics,
            "readiness": readiness,
        }
