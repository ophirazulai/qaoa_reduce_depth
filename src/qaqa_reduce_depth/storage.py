"""Small atomic file helpers shared by coordinator and workers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as stream:
            writer(stream)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda stream: json.dump(data, stream, indent=2, allow_nan=False))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
