#!/usr/bin/env python3
"""
Shared helpers for host-side reset scripts.

These scripts are often launched from Windows while the actual Django and
service dependencies live inside Docker containers. When host Python cannot
reach PostgreSQL/Qdrant or lacks required drivers, we retry inside the
running `web` container.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent

_DOCKER_FALLBACK_MARKERS = (
    "Error loading psycopg2 or psycopg module",
    "No module named 'psycopg'",
    "No module named 'psycopg2'",
    "could not translate host name",
    "Name or service not known",
    "No such host is known",
    "Temporary failure in name resolution",
    "No module named 'qdrant_client'",
)


def should_retry_in_docker(error: BaseException) -> bool:
    """Return True when a reset failure should be retried in Docker."""

    error_text = f"{error.__class__.__name__}: {error}"
    return any(marker in error_text for marker in _DOCKER_FALLBACK_MARKERS)


def run_script_in_docker(script_name: str, extra_args: Iterable[str] | None = None) -> bool:
    """Run a reset script inside the Docker `web` container."""

    docker_cli = shutil.which("docker")
    if not docker_cli:
        print("Docker CLI was not found on PATH.")
        return False

    args = list(extra_args or [])
    command = [
        docker_cli,
        "compose",
        "exec",
        "-T",
        "web",
        "python",
        f"/usr/src/app/{script_name}",
        "--yes",
        "--in-container",
        *args,
    ]

    print("Retrying inside the Docker web container...")
    result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)

    if result.returncode == 0:
        return True

    print(f"Docker fallback failed with exit code {result.returncode}.")
    print(
        "Run manually from agent_developer: "
        f'{" ".join(command)}'
    )
    return False
