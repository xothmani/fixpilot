"""Settings read from environment variables, with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# src/fixpilot/config.py -> project root is two levels above "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_REPOS = PROJECT_ROOT / "benchmark" / "repos"
DOCKERFILE = PROJECT_ROOT / "docker" / "Dockerfile.sandbox"


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "fixpilot-sandbox:latest"
    memory: str = "512m"
    cpus: float = 1.0
    pids_limit: int = 128
    timeout_s: int = 60
    max_output_chars: int = 6000

    @classmethod
    def from_env(cls) -> SandboxConfig:
        d = cls()
        env = os.environ
        return cls(
            image=env.get("FIXPILOT_SANDBOX_IMAGE", d.image),
            memory=env.get("FIXPILOT_SANDBOX_MEMORY", d.memory),
            cpus=float(env.get("FIXPILOT_SANDBOX_CPUS", d.cpus)),
            pids_limit=int(env.get("FIXPILOT_SANDBOX_PIDS", d.pids_limit)),
            timeout_s=int(env.get("FIXPILOT_SANDBOX_TIMEOUT", d.timeout_s)),
            max_output_chars=int(env.get("FIXPILOT_SANDBOX_MAX_OUTPUT", d.max_output_chars)),
        )
