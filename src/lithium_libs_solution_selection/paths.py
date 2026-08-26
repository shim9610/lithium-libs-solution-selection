"""Stable repository paths used by the command-line entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve package inputs without machine-specific absolute paths."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> "ProjectPaths":
        candidates = (
            (start,)
            if start is not None
            else (Path(__file__), Path.cwd())
        )
        visited: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            for directory in (resolved, *resolved.parents):
                if directory in visited:
                    continue
                visited.add(directory)
                if (directory / "pyproject.toml").is_file() and (
                    directory / "artifacts"
                ).is_dir():
                    return cls(directory)
        raise RuntimeError("Could not locate the repository root")

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def evaluation(self) -> Path:
        return self.artifacts / "evaluation"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def references(self) -> Path:
        return self.root / "reference_figures"

    @property
    def figure_sources(self) -> Path:
        return self.root / "figure_sources"

    @property
    def output_figures(self) -> Path:
        return self.root / "outputs" / "figures"

    @property
    def output_reports(self) -> Path:
        return self.root / "outputs" / "reports"

    @property
    def output_training(self) -> Path:
        return self.root / "outputs" / "training"
