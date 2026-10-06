"""Command-line interface for artifact, figure, inference, and training tasks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

from .artifacts import ArtifactCheck, verify_artifacts
from .figures import FIGURE_SPECS, get_figure_spec, render_figure
from .paths import ProjectPaths
from .verification import FigureCheck, verify_figure, write_report


def _add_figure_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="process all nine paper figures",
    )
    selection.add_argument(
        "--figure",
        type=int,
        action="append",
        metavar="N",
        help="process one figure; repeat to select more than one",
    )


def _selected_numbers(arguments: argparse.Namespace) -> list[int]:
    numbers = list(FIGURE_SPECS) if arguments.all else list(arguments.figure)
    unique = sorted(set(numbers))
    for number in unique:
        get_figure_spec(number)
    return unique


def _project_paths(arguments: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths.discover(arguments.root)


def _reject_reference_destination(path: Path, project: ProjectPaths) -> None:
    resolved = path.resolve()
    references = project.references.resolve()
    if resolved == references or references in resolved.parents:
        raise ValueError(
            "Generated or comparison images cannot use reference_figures "
            "as their destination"
        )


def _write_artifact_report(path: Path, checks: list[ArtifactCheck]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "PASS" if all(check.passed for check in checks) else "FAIL",
        "files": [
            {
                **asdict(check),
                "passed": check.passed,
            }
            for check in checks
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _checks_by_path(checks: list[ArtifactCheck]) -> dict[str, ArtifactCheck]:
    return {check.path.replace("\\", "/"): check for check in checks}


def _required_checks(
    numbers: Sequence[int],
    checks: list[ArtifactCheck],
) -> list[ArtifactCheck]:
    indexed = _checks_by_path(checks)
    required = {
        path
        for number in numbers
        for path in get_figure_spec(number).required_files
    }
    missing_manifest_entries = sorted(required.difference(indexed))
    if missing_manifest_entries:
        joined = ", ".join(missing_manifest_entries)
        raise RuntimeError(f"Figure inputs are absent from the manifest: {joined}")
    return [indexed[path] for path in sorted(required)]


def _print_artifact_checks(checks: Sequence[ArtifactCheck]) -> None:
    passed = sum(check.passed for check in checks)
    print(f"Frozen files: {passed}/{len(checks)} passed")
    for check in checks:
        if not check.passed:
            actual = check.actual_sha256 or "missing"
            print(f"  FAIL {check.path}: {actual}")


def _command_artifacts(arguments: argparse.Namespace) -> int:
    paths = _project_paths(arguments)
    checks = verify_artifacts(paths.root)
    report_path = Path(arguments.report) if arguments.report else (
        paths.output_reports / "artifact_verification.json"
    )
    _write_artifact_report(report_path, checks)
    _print_artifact_checks(checks)
    print(f"Report: {report_path.resolve()}")
    return 0 if all(check.passed for check in checks) else 1


def _command_figures(arguments: argparse.Namespace) -> int:
    paths = _project_paths(arguments)
    numbers = _selected_numbers(arguments)
    checks = verify_artifacts(paths.root)
    required = _required_checks(numbers, checks)
    _print_artifact_checks(required)
    if not all(check.passed for check in required):
        print("Figure generation stopped because a required frozen file changed.")
        return 1

    destination = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else paths.output_figures
    )
    _reject_reference_destination(destination, paths)
    for number in numbers:
        spec = get_figure_spec(number)
        output = render_figure(
            number,
            paths=paths,
            output_path=destination / spec.filename,
        )
        print(f"Figure {number}: {output.resolve()}")
    return 0


def _print_figure_check(number: int, check: FigureCheck) -> None:
    print(f"Figure {number}: {check.status} [{check.reason_code}]")
    print(f"  {check.inferred_reason}")
    if check.metrics.normalized_rmse is not None:
        print(
            "  pixels: "
            f"RMSE={check.metrics.normalized_rmse:.8f}, "
            f"MAE={check.metrics.normalized_mae:.8f}, "
            f"SSIM={check.metrics.ssim:.8f}"
        )


def _command_verify(arguments: argparse.Namespace) -> int:
    paths = _project_paths(arguments)
    numbers = _selected_numbers(arguments)
    artifact_checks = verify_artifacts(paths.root)
    required = _required_checks(numbers, artifact_checks)
    _print_artifact_checks(required)
    if not all(check.passed for check in required):
        print("Verification stopped because a required frozen file changed.")
        return 1

    generated_root = (
        Path(arguments.generated_dir).resolve()
        if arguments.generated_dir
        else paths.output_figures
    )
    _reject_reference_destination(generated_root, paths)
    checks: list[FigureCheck] = []
    for number in numbers:
        spec = get_figure_spec(number)
        generated = generated_root / spec.filename
        if not arguments.existing:
            render_figure(number, paths=paths, output_path=generated)
        if not generated.is_file():
            print(f"Figure {number}: missing generated image: {generated}")
            return 1
        reference = paths.references / spec.filename
        check = verify_figure(
            reference,
            generated,
            semantic_ok=True,
            static_raster_source=spec.static_raster_source,
        )
        checks.append(check)
        _print_figure_check(number, check)

    report_path = Path(arguments.report) if arguments.report else (
        paths.output_reports / "figure_verification.json"
    )
    write_report(report_path, checks)
    print(f"Report: {report_path.resolve()}")
    return 0 if all(check.status.startswith("PASS_") for check in checks) else 1


def _command_verify_inference(arguments: argparse.Namespace) -> int:
    from .inference_verification import verify_saved_inference

    paths = _project_paths(arguments)
    report_path = Path(arguments.report) if arguments.report else (
        paths.output_reports / "inference_verification.json"
    )
    report = verify_saved_inference(
        paths.checkpoints / "solution_selector.pth",
        paths.evaluation / "measured_standards" / "measurement_outputs.npz",
        output_json=report_path,
    )
    print(f"Inference: {report.status}")
    print(
        "  lithium-6 maximum absolute error: "
        f"{report.isotope.max_absolute_error:.10f} pp "
        f"(limit {report.isotope.tolerance:g} pp)"
    )
    print(
        "  shift maximum absolute error: "
        f"{report.shift.max_absolute_error:.10f} px "
        f"(limit {report.shift.tolerance:g} px)"
    )
    print(f"Report: {report_path.resolve()}")
    return 0 if report.passed else 1


def _command_training_spec(arguments: argparse.Namespace) -> int:
    from .training import TrainingConfig

    config = TrainingConfig(objective=arguments.objective)
    payload = {
        "schema_version": 1,
        "configuration": asdict(config),
        "adam_steps_per_online_update": config.adam_steps_per_online_update,
        "configured_adam_steps": config.configured_adam_steps,
        "loss_weights": asdict(config.loss_weights),
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if arguments.output:
        destination = Path(arguments.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"Training specification: {destination}")
    else:
        print(rendered, end="")
    return 0


def _command_verify_training(arguments: argparse.Namespace) -> int:
    from .training_verification import verify_training_implementation

    paths = _project_paths(arguments)
    report = verify_training_implementation()
    report_path = Path(arguments.report).resolve() if arguments.report else (
        paths.output_reports / "training_implementation_verification.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Training implementation: {report.status}")
    for check in report.checks:
        marker = "PASS" if check.passed else "FAIL"
        print(f"  {marker} {check.name}: {check.detail}")
    print(f"Report: {report_path}")
    return 0 if report.passed else 1


def _command_train(arguments: argparse.Namespace) -> int:
    from .training import TrainingConfig, run_training

    paths = _project_paths(arguments)
    config = TrainingConfig(
        objective=arguments.objective,
        random_seed=arguments.seed,
    )
    overrides = {
        "online_updates": arguments.online_updates,
        "samples_per_online_update": arguments.samples_per_update,
        "validation_samples": arguments.validation_samples,
        "batch_size": arguments.batch_size,
        "number_of_workers": arguments.workers,
    }
    config = replace(
        config,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    if arguments.smoke_test:
        config = config.smoke_test()
    destination = (
        Path(arguments.output_dir).resolve()
        if arguments.output_dir
        else paths.output_training / arguments.objective
    )
    print(
        f"Objective={config.objective}; online updates={config.online_updates}; "
        f"Adam steps={config.configured_adam_steps}; device={arguments.device or 'auto'}"
    )
    if config.random_seed is None:
        print(
            "Random seed is intentionally unset, matching the historical run. "
            "Pass --seed for a repeatable new run."
        )
    result = run_training(
        config=config,
        output_directory=destination,
        device=arguments.device,
    )
    print(f"Best checkpoint: {result.best_checkpoint}")
    print(f"Best validation loss: {result.best_validation_loss:.8g}")
    print(f"Completed Adam steps: {result.completed_adam_steps}")
    return 0


def _reject_frozen_destination(path: Path, project: ProjectPaths) -> None:
    resolved = path.resolve()
    for protected in (project.artifacts, project.references, project.figure_sources):
        if resolved == protected or protected in resolved.parents:
            raise ValueError("Compression outputs cannot overwrite frozen scientific inputs")


def _teacher_hash(arguments: argparse.Namespace, checkpoint: Path, paths: ProjectPaths) -> str | None:
    from .inference_verification import RELEASE_CHECKPOINT_SHA256

    if arguments.expected_sha256 is not None:
        return arguments.expected_sha256
    if checkpoint.resolve() == (paths.checkpoints / "solution_selector.pth").resolve():
        return RELEASE_CHECKPOINT_SHA256
    return None


def _command_export_checkpoint(arguments: argparse.Namespace) -> int:
    from .checkpoints import export_checkpoint

    paths = _project_paths(arguments)
    checkpoint = Path(arguments.checkpoint) if arguments.checkpoint else (
        paths.checkpoints / "solution_selector.pth"
    )
    destination = Path(arguments.output)
    _reject_frozen_destination(destination, paths)
    model_config = None
    if arguments.model_config:
        model_config = json.loads(Path(arguments.model_config).read_text(encoding="utf-8"))
        if not isinstance(model_config, dict):
            raise ValueError("model-config must contain a JSON object")
    report = export_checkpoint(
        checkpoint, destination,
        expected_sha256=_teacher_hash(arguments, checkpoint, paths),
        model_config=model_config,
    )
    print(json.dumps(asdict(report), indent=2))
    return 0


def _command_distill(arguments: argparse.Namespace) -> int:
    from .distillation import DistillationConfig, run_distillation

    paths = _project_paths(arguments)
    config = (
        DistillationConfig.from_dict(json.loads(Path(arguments.config).read_text(encoding="utf-8")))
        if arguments.config else DistillationConfig()
    )
    config.validate()
    checkpoint = Path(arguments.teacher) if arguments.teacher else (
        paths.checkpoints / "solution_selector.pth"
    )
    destination = Path(arguments.output_dir) if arguments.output_dir else (
        paths.output_training / "distillation"
    )
    _reject_frozen_destination(destination, paths)
    print(json.dumps({
        "configuration": asdict(config),
        "teacher_checkpoint": str(checkpoint.resolve()),
        "expected_teacher_sha256": _teacher_hash(arguments, checkpoint, paths),
        "output_directory": str(destination.resolve()),
        "device": arguments.device or "auto",
        "configured_adam_steps": config.student.configured_adam_steps,
    }, indent=2))
    if arguments.dry_run:
        return 0
    result = run_distillation(
        config=config, teacher_checkpoint=checkpoint,
        output_directory=destination, device=arguments.device,
        expected_teacher_sha256=_teacher_hash(arguments, checkpoint, paths),
    )
    print(f"Best training checkpoint: {result.best_checkpoint}")
    print(f"Inference weights: {result.output_directory / 'best_model_inference.pth'}")
    print(f"Best supervised validation loss: {result.best_validation_loss:.8g}")
    print(f"Completed Adam steps: {result.completed_adam_steps}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lithium-libs",
        description=(
            "Reproduce and verify the lithium-LIBS paper figures, inference, "
            "generator, loss, and online-training protocol."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="repository root; otherwise discover it from the package or current directory",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    artifacts = commands.add_parser(
        "artifacts",
        help="verify frozen checkpoints, evaluation records, and references",
    )
    artifacts.add_argument("--report", help="write the JSON report to this path")
    artifacts.set_defaults(handler=_command_artifacts)

    figures = commands.add_parser("figures", help="generate paper figures")
    _add_figure_selection(figures)
    figures.add_argument("--output-dir", help="override outputs/figures")
    figures.set_defaults(handler=_command_figures)

    verify = commands.add_parser(
        "verify",
        help="render figures and compare them with approved references",
    )
    _add_figure_selection(verify)
    verify.add_argument("--generated-dir", help="override outputs/figures")
    verify.add_argument(
        "--existing",
        action="store_true",
        help="compare existing outputs without rendering them again",
    )
    verify.add_argument("--report", help="write the JSON report to this path")
    verify.set_defaults(handler=_command_verify)

    inference = commands.add_parser(
        "verify-inference",
        help="re-run representative checkpoint inference on the CPU",
    )
    inference.add_argument("--report", help="write the JSON report to this path")
    inference.set_defaults(handler=_command_verify_inference)

    training_spec = commands.add_parser(
        "training-spec",
        help="print the exact model, optimizer, online-update, and loss settings",
    )
    training_spec.add_argument(
        "--objective",
        choices=("full", "component-supervision-ablation"),
        default="full",
    )
    training_spec.add_argument("--output", help="write the JSON specification")
    training_spec.set_defaults(handler=_command_training_spec)

    training_check = commands.add_parser(
        "verify-training",
        help="check simulator and loss calculations against frozen-source goldens",
    )
    training_check.add_argument("--report", help="write the JSON report to this path")
    training_check.set_defaults(handler=_command_verify_training)

    train = commands.add_parser(
        "train",
        help="run the final paper online-training algorithm",
    )
    train.add_argument(
        "--objective",
        choices=("full", "component-supervision-ablation"),
        default="full",
    )
    train.add_argument("--output-dir", help="training output directory")
    train.add_argument("--device", help="PyTorch device, for example cuda or cpu")
    train.add_argument("--seed", type=int, help="seed a repeatable new run")
    train.add_argument("--online-updates", type=int, help="override 20,000 updates")
    train.add_argument(
        "--samples-per-update",
        type=int,
        help="override 16,384 freshly generated samples per update",
    )
    train.add_argument(
        "--validation-samples",
        type=int,
        help="override the fixed 4,096-spectrum validation set",
    )
    train.add_argument("--batch-size", type=int, help="override batch size 1,024")
    train.add_argument("--workers", type=int, help="override generator worker count")
    train.add_argument(
        "--smoke-test",
        action="store_true",
        help="run one generated training and validation sample",
    )
    train.set_defaults(handler=_command_train)

    export = commands.add_parser(
        "export-checkpoint", help="remove optimizer state from a trusted checkpoint",
    )
    export.add_argument("--checkpoint", help="trusted source; defaults to the released teacher")
    export.add_argument("--output", required=True, help="new inference checkpoint file")
    export.add_argument("--expected-sha256", help="verify the source hash before loading")
    export.add_argument("--model-config", help="architecture JSON for a custom bare state_dict")
    export.set_defaults(handler=_command_export_checkpoint)

    distill = commands.add_parser(
        "distill", help="train a small student with physics supervision and a frozen teacher",
    )
    distill.add_argument("--config", help="JSON experiment configuration; omitted fields use defaults")
    distill.add_argument("--teacher", help="trusted checkpoint; defaults to the released teacher")
    distill.add_argument("--expected-sha256", help="verify the teacher hash before loading")
    distill.add_argument("--output-dir", help="new/empty experiment output directory")
    distill.add_argument("--device", help="PyTorch device, for example cuda or cpu")
    distill.add_argument(
        "--dry-run", action="store_true",
        help="validate and print settings without loading models, generating data or training",
    )
    distill.set_defaults(handler=_command_distill)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
