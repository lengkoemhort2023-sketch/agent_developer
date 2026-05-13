from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> Path | None:
    """Load environment variables from the first matching env file.

    Resolution order:
    1. `APP_ENV_FILE` if explicitly provided
    2. `.env.dev`
    3. `.env`
    """

    explicit_env = os.environ.get("APP_ENV_FILE")
    candidates = []

    if explicit_env:
        candidates.append(Path(explicit_env).expanduser())

    candidates.extend(
        [
            PROJECT_ROOT / ".env.dev",
            PROJECT_ROOT / ".env",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate

    return None
