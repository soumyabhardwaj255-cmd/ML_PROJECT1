"""Atomic, identifiable artifacts. Existing run directories are never reused."""
import hashlib
import json
import os
import platform
import uuid
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def environment():
    packages = {}
    for name in ("numpy", "pandas", "scipy", "mne", "antropy", "scikit-learn",
                 "xgboost", "torch", "joblib", "matplotlib", "tabulate"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
