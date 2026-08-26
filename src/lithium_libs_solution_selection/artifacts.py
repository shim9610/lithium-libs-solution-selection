"""Integrity checks for frozen checkpoints, evaluation records, and figures."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactCheck:
    path: str
    expected_size: int
    actual_size: int | None
    expected_sha256: str
    actual_sha256: str | None

    @property
    def passed(self) -> bool:
        return (
            self.actual_size == self.expected_size
            and self.actual_sha256 == self.expected_sha256
        )


def load_manifest(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported artifact manifest: {path}")
    return manifest


def verify_artifacts(root: Path, manifest_path: Path | None = None) -> list[ArtifactCheck]:
    path = manifest_path or root / "artifacts" / "manifest.json"
    manifest = load_manifest(path)
    checks: list[ArtifactCheck] = []
    for entry in manifest["artifacts"]:
        relative = str(entry["path"])
        artifact = root / relative
        if artifact.is_file():
            actual_size = artifact.stat().st_size
            actual_sha256 = sha256_file(artifact)
        else:
            actual_size = None
            actual_sha256 = None
        checks.append(
            ArtifactCheck(
                path=relative,
                expected_size=int(entry["bytes"]),
                actual_size=actual_size,
                expected_sha256=str(entry["sha256"]),
                actual_sha256=actual_sha256,
            )
        )
    return checks


def require_artifacts(checks: Iterable[ArtifactCheck]) -> None:
    failures = [check for check in checks if not check.passed]
    if failures:
        details = "\n".join(
            f"- {item.path}: expected {item.expected_sha256}, got {item.actual_sha256}"
            for item in failures
        )
        raise RuntimeError(f"Artifact integrity check failed:\n{details}")
