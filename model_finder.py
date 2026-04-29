from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelFiles:
    model_dir: Path
    model_name: str
    gguf: Path
    mmproj: Path | None


class ModelDiscoveryError(RuntimeError):
    pass


def discover_model_files(model_dir: Path) -> ModelFiles:
    model_dir = model_dir.expanduser().resolve()

    if not model_dir.exists():
        raise ModelDiscoveryError(f"Model folder does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise ModelDiscoveryError(f"Model path is not a directory: {model_dir}")

    model_name = model_dir.name

    gguf_matches = sorted(model_dir.glob(f"{model_name}*.gguf"))
    if not gguf_matches:
        raise ModelDiscoveryError(
            "No GGUF model found matching:\n"
            f"  {model_dir / (model_name + '*.gguf')}"
        )
    if len(gguf_matches) > 1:
        files = "\n".join(f"  {p}" for p in gguf_matches)
        raise ModelDiscoveryError(
            "Multiple GGUF model files found:\n"
            f"{files}\n"
            "Stop: please keep only one matching model file, or refine the matching rule."
        )

    mmproj_matches = sorted(model_dir.glob("mmproj*.gguf"))
    if len(mmproj_matches) > 1:
        files = "\n".join(f"  {p}" for p in mmproj_matches)
        raise ModelDiscoveryError(
            "Multiple mmproj files found:\n"
            f"{files}\n"
            "Stop: please keep only one mmproj file, or refine the matching rule."
        )

    return ModelFiles(
        model_dir=model_dir,
        model_name=model_name,
        gguf=gguf_matches[0],
        mmproj=mmproj_matches[0] if mmproj_matches else None,
    )
