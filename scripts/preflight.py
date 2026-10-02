"""Cross-platform prerequisite and project check."""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REQUIRED_PACKAGES = {
    "anyio": "4.14.0",
    "fastapi": "0.128.2",
    "httpx": "0.28.1",
    "joblib": "1.5.3",
    "numpy": "2.4.2",
    "pandas": "2.2.3",
    "pytest": "9.0.2",
    "scikit-learn": "1.8.0",
    "uvicorn": "0.48.0",
}


def check_command(label: str, command: list[str], *, require_output: bool = False) -> bool:
    executable = command[0]
    if shutil.which(executable) is None:
        print(f"[FAIL] {label}: '{executable}' was not found on PATH")
        return False
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except Exception as exc:
        print(f"[FAIL] {label}: {exc}")
        return False
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    output = (stdout or stderr).splitlines()
    detail = output[0] if output else "no response"
    succeeded = result.returncode == 0 and (bool(stdout) or not require_output)
    status = "PASS" if succeeded else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return succeeded


def check_project() -> bool:
    checks: list[bool] = []
    installed_versions: dict[str, str] = {}
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            print(f"[FAIL] Python package: {package} is not installed")
            checks.append(False)
            continue
        installed_versions[package] = installed
        matches = installed == expected
        print(
            f"[{'PASS' if matches else 'FAIL'}] Python package: "
            f"{package} {installed} (expected {expected})"
        )
        checks.append(matches)

    model_path = ROOT / "models" / "churn_model.joblib"
    metadata_path = ROOT / "models" / "metadata.json"
    artifacts_exist = model_path.exists() and metadata_path.exists()
    print(
        f"[{'PASS' if artifacts_exist else 'FAIL'}] Model artifacts: "
        f"{'present' if artifacts_exist else 'missing from models/'}"
    )
    checks.append(artifacts_exist)

    if artifacts_exist:
        try:
            model_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            print("[PASS] Model metadata: valid JSON")
            checks.append(True)
        except Exception as exc:
            print(f"[FAIL] Model metadata: {exc}")
            model_metadata = {}
            checks.append(False)
        expected_framework = model_metadata.get("framework_version")
        installed_framework = installed_versions.get("scikit-learn")
        if installed_framework is None:
            print("[FAIL] Model compatibility: scikit-learn is not installed")
            checks.append(False)
        else:
            compatible = expected_framework == installed_framework
            print(
                f"[{'PASS' if compatible else 'FAIL'}] Model compatibility: "
                f"scikit-learn {installed_framework} "
                f"(artifact expects {expected_framework or 'an unspecified version'})"
            )
            checks.append(compatible)

        try:
            from app.model import load_model

            load_model()
            print("[PASS] Model load: artifact loaded successfully")
            checks.append(True)
        except Exception as exc:
            print(f"[FAIL] Model load: {exc}")
            checks.append(False)

    return all(checks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="also verify pinned packages and load the model artifact",
    )
    args = parser.parse_args()

    print("Customer retention application preflight\n")
    python_ok = sys.version_info >= (3, 11)
    print(
        f"[{'PASS' if python_ok else 'FAIL'}] Python: {sys.version.split()[0]} "
        "(requires 3.11+; 3.12 recommended)"
    )

    git_ok = check_command("Git", ["git", "--version"], require_output=True)
    docker_cli_ok = check_command(
        "Docker CLI", ["docker", "--version"], require_output=True
    )
    docker_engine_ok = False
    if docker_cli_ok:
        # Some Docker clients return exit code 0 with an empty stdout when the
        # daemon is unavailable, so a non-empty server version is required.
        docker_engine_ok = check_command(
            "Docker engine",
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            require_output=True,
        )

    project_ok = check_project() if args.full else True
    all_ok = python_ok and git_ok and docker_cli_ok and docker_engine_ok and project_ok
    print(
        "\n"
        + (
            "Preflight passed."
            if all_ok
            else "Preflight needs attention before continuing."
        )
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
