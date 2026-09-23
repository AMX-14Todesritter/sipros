"""Output layout shared by command-line tools; importing never creates files."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def output_root(project_root=PROJECT_ROOT):
    """Resolve relative SIPROS_OUTPUT_ROOT against the project, never the cwd."""
    configured = Path(os.environ.get("SIPROS_OUTPUT_ROOT") or "output")
    return (Path(project_root) / configured).resolve()


def run_directory(purpose, component, label="run", *, project_root=PROJECT_ROOT):
    """Return a new run path. The caller must create it with exist_ok=False."""
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", label) or "run"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return output_root(project_root) / purpose / component / f"{label}_{timestamp}"


def resolve_output(explicit, purpose, component, label="run", *, project_root=PROJECT_ROOT):
    """Explicit output paths retain their usual cwd-relative CLI meaning."""
    if explicit is not None:
        return Path(explicit).resolve()
    return run_directory(purpose, component, label, project_root=project_root)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose")
    parser.add_argument("component")
    parser.add_argument("label", nargs="?", default="run")
    args = parser.parse_args()
    print(run_directory(args.purpose, args.component, args.label))
