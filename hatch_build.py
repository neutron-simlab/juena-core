"""Build the Files panel into wheels, including installs directly from Git."""

from pathlib import Path
import shutil
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        npm = shutil.which("npm")
        if npm is None:
            raise RuntimeError(
                "Building juena-core from source requires Node.js 24 and npm. "
                "Install them before running uv sync or building a wheel."
            )
        frontend = Path(self.root) / "frontend" / "filesystem"
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=frontend, check=True)
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)
        build_data["artifacts"].append("src/juena_core/ui/filesystem_dist/**")
