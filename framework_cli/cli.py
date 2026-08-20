"""Command-line entrypoint for the read-only Framework doctor."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

from framework_cli.doctor import (
    doctor_all,
    doctor_artifacts,
    doctor_images,
    doctor_model,
    doctor_source,
    parameter_error,
    result,
)


class ParameterError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ParameterError(message)


def _json_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        action="store_true",
        required=True,
        help="emit the stable diagnostic JSON envelope",
    )


def _image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--learner-image", required=True)
    parser.add_argument("--aiserver-image", required=True)
    parser.add_argument("--client-image", required=True)


def build_parser() -> Parser:
    parser = Parser(prog="framework")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="run read-only workspace diagnostics"
    )
    scopes = doctor.add_subparsers(dest="scope", required=True)

    source = scopes.add_parser("source")
    _json_flag(source)

    artifacts = scopes.add_parser("artifacts")
    _json_flag(artifacts)

    model = scopes.add_parser("model")
    model.add_argument("--model", required=True)
    _json_flag(model)

    images = scopes.add_parser("images")
    _image_arguments(images)
    _json_flag(images)

    all_checks = scopes.add_parser("all")
    all_checks.add_argument("--model", required=True)
    _image_arguments(all_checks)
    _json_flag(all_checks)
    return parser


def _scope_from_argv(arguments: Sequence[str]) -> str:
    if "doctor" in arguments:
        index = arguments.index("doctor")
        if index + 1 < len(arguments) and arguments[index + 1] in {
            "source",
            "artifacts",
            "model",
            "images",
            "all",
        }:
            return arguments[index + 1]
    return "cli"


def dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command != "doctor":
        return parameter_error("cli", "only the doctor command is supported")
    if arguments.scope == "source":
        return doctor_source()
    if arguments.scope == "artifacts":
        return doctor_artifacts()
    if arguments.scope == "model":
        return doctor_model(arguments.model)
    if arguments.scope == "images":
        return doctor_images(
            arguments.learner_image,
            arguments.aiserver_image,
            arguments.client_image,
        )
    if arguments.scope == "all":
        return doctor_all(
            arguments.model,
            arguments.learner_image,
            arguments.aiserver_image,
            arguments.client_image,
        )
    return parameter_error("cli", "unknown doctor scope")


def emit(document: dict[str, Any]) -> None:
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = build_parser().parse_args(arguments)
        document = dispatch(parsed)
    except ParameterError as exc:
        document = parameter_error(_scope_from_argv(arguments), str(exc))
        emit(document)
        return 2
    except Exception as exc:  # Fail closed while preserving the JSON contract.
        document = result(
            _scope_from_argv(arguments),
            False,
            error=f"diagnostic failed: {type(exc).__name__}: {exc}",
        )
        emit(document)
        return 1

    emit(document)
    return 0 if document["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
