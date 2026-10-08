"""Metadata and shared policy for the installable Harness plugin."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessPluginDescriptor:
    """Validated files that make up the local Triton-RISCV Harness bundle."""

    root: Path
    name: str
    version: str
    entry_path: Path
    patch_path: Path
    policy_path: Path

    @classmethod
    def load(cls, root: Path) -> "HarnessPluginDescriptor":
        resolved = root.expanduser().resolve()
        manifest_path = resolved / "package.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Harness plugin manifest does not exist: {manifest_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Harness plugin manifest is invalid: {manifest_path}") from error

        bundle = manifest.get("dsh", {}).get("bundle", {})
        patch = bundle.get("patch")
        if not isinstance(patch, str) or not patch:
            raise ValueError("Harness plugin manifest does not declare dsh.bundle.patch")

        name = manifest.get("name")
        version = manifest.get("version")
        if not isinstance(name, str) or not name:
            raise ValueError("Harness plugin manifest does not declare a name")
        if not isinstance(version, str) or not version:
            raise ValueError("Harness plugin manifest does not declare a version")

        descriptor = cls(
            root=resolved,
            name=name,
            version=version,
            entry_path=resolved / "index.js",
            patch_path=(resolved / patch).resolve(),
            policy_path=resolved / "policy.md",
        )
        for required in (
            descriptor.entry_path,
            descriptor.patch_path,
            descriptor.policy_path,
        ):
            if not required.is_file():
                raise ValueError(f"Harness plugin file does not exist: {required}")
            required.relative_to(resolved)
        return descriptor

    def policy(self) -> str:
        text = self.policy_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Harness plugin policy cannot be empty")
        return text

    def public_metadata(self) -> dict[str, str | bool]:
        return {
            "name": self.name,
            "version": self.version,
            "loaded": True,
        }


def default_plugin_root() -> Path:
    return Path(__file__).resolve().parent / "bundle"
