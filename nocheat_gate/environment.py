"""Observed execution diagnostics, not a classifier of defect responsibility.

An import error can be an absent dependency OR a source regression. Keep the
operational symptom separate from the gate verdict and from causal attribution.
No target module is imported to collect interpreter/distribution metadata.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import site
import sys
import sysconfig
from pathlib import Path


def diagnose_execution(*, stage: str, returncode: int, stdout: str = "",
                       stderr: str = "", timed_out: bool = False,
                       runner_mode: str | None = None) -> dict:
    text = stdout + "\n" + stderr
    phase = stage.removesuffix('_base').removesuffix('_head')
    category = "PROCESS_ERROR"
    evidence = None
    if timed_out:
        category = "TIMEOUT"
    elif returncode == 0:
        category = "EXECUTION_PASSED"
    elif phase in {"venv", "dependencies", "engine", "metadata", "dependency_check", "identity"}:
        category = "ENVIRONMENT_PREPARATION_FAILED"
    elif stage.startswith("build_"):
        category = "BUILD_FAILED"
    else:
        missing = re.search(r"(?:^|\n)(?:E\s+)?ModuleNotFoundError: No module named ['\"]([^'\"\n]+)['\"]", text)
        if missing:
            category, evidence = "IMPORT_UNAVAILABLE", missing.group(1)
        elif re.search(r"(?:^|\n)(?:E\s+)?ImportError:", text):
            category = "IMPORT_UNAVAILABLE"
        elif runner_mode == "pytest" and returncode == 5:
            category = "NO_TESTS_COLLECTED"
        elif runner_mode == "pytest" and returncode == 2:
            category = "COLLECTION_OR_INTERRUPTION"
        elif runner_mode == "pytest" and returncode == 1:
            category = "TEST_FAILURE"
    result = {"stage": stage, "category": category, "returncode": returncode,
              "timed_out": timed_out, "attribution": "UNRESOLVED",
              "scope": "observed process outcome; not proof of environment or source fault"}
    if evidence is not None:
        result["missing_module"] = evidence
    return result


def interpreter_snapshot() -> dict:
    """Describe this process only; no claim of CI equivalence or OS isolation."""
    distributions = sorted(
        ({"name": d.metadata.get("Name", ""), "version": d.version}
         for d in importlib.metadata.distributions()), key=lambda d: (d["name"].lower(), d["version"]))
    prefix = Path(sys.prefix).resolve()
    cfg = prefix / "pyvenv.cfg"
    config = {}
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep:
                config[key.strip().lower()] = value.strip().lower()
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "executable": sys.executable, "prefix": str(prefix), "base_prefix": sys.base_prefix,
            "in_venv": sys.prefix != sys.base_prefix,
            "isolated_flag": bool(sys.flags.isolated), "user_site_enabled": site.ENABLE_USER_SITE,
            "system_site_packages": config.get("include-system-site-packages", "UNMEASURED"),
            "platform": sys.platform, "machine": platform.machine(),
            "soabi": sysconfig.get_config_var("SOABI"), "distributions": distributions,
            "scope": "local interpreter metadata; not OS sandbox or CI snapshot equivalence"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-prefix", required=True, type=Path)
    args = parser.parse_args()
    snapshot = interpreter_snapshot()
    ok = (snapshot["in_venv"] and snapshot["isolated_flag"] and
          snapshot["user_site_enabled"] is False and snapshot["system_site_packages"] == "false" and
          Path(snapshot["prefix"]) == args.expected_prefix.resolve())
    snapshot["status"] = "LOCAL_VENV_VERIFIED" if ok else "ENVIRONMENT_IDENTITY_MISMATCH"
    print(json.dumps(snapshot, sort_keys=True))
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
