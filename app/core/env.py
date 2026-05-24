from __future__ import annotations

import os
from pathlib import Path
import importlib

# Import dotenv if available; provide a no-op fallback to satisfy runtime
# and static analysis when the package is not installed.
if importlib.util.find_spec("dotenv") is not None:
    from dotenv import load_dotenv  # type: ignore
else:
    def load_dotenv(path, override=False):
        """Fallback if python-dotenv is not installed."""
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_environment() -> Path | None:
    """Load environment variables from the first matching env file.

    Resolution order:
    1. `APP_ENV_FILE` if explicitly provided
    2. `.env.local`
    3. `.env`
    """

    explicit_env = os.environ.get("APP_ENV_FILE")
    candidates = []

    if explicit_env:
        candidates.append(Path(explicit_env).expanduser())

    candidates.extend(
        [
            #PROJECT_ROOT / ".env.dev",
            PROJECT_ROOT / ".env.local",
            PROJECT_ROOT / ".env",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate

    return None
