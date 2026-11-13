"""Build the Files panel into wheels, including installs directly from Git."""

from pathlib import Path
import shutil
import subprocess

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendBuildHook(BuildHookInterface):
    """Hatch build hook that builds the Files panel with npm before packaging."""

    def initialize(self, version, build_data):
        """Install the panel's npm packages, build it, and add the output to the wheel.

        Args:
            version: Which variant of the target hatch is building, such as
                ``standard`` or ``editable`` for a wheel. Not used here.
            build_data: Hatch's build data; the built panel is added to its
                ``artifacts``.

        Raises:
            RuntimeError: npm is not installed.
            subprocess.CalledProcessError: an npm step failed.
        """
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
