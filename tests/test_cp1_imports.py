"""The CP1 import gate must hold without application configuration."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_cp1_modules_import_with_an_empty_environment(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import juena_core.log, juena_core.schema.server, "
                "juena_core.llms_providers"
            ),
        ],
        cwd=tmp_path,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
