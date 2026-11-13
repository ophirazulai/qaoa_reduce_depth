"""Small atomic file helpers shared by coordinator and workers."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def _umask_mode():
    mask = os.umask(0)
    os.umask(mask)
    return 0o666 & ~mask


# mkstemp always creates 0600 files. Results are ordinary shared outputs, so give
# them the permissions a plain open() would. Read once, while single-threaded.
FILE_MODE = _umask_mode()


def atomic_write(path, writer, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        # Text is always UTF-8: the report holds non-ASCII characters and must
        # not depend on the locale of the coordinator or a compute node.
        with os.fdopen(fd, "wb" if binary else "w",
                       encoding=None if binary else "utf-8") as stream:
            writer(stream)
        try:
            os.chmod(temporary, FILE_MODE)
        except OSError:
            pass  # Some filesystems do not support modes; the content still counts.
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path, data):
    atomic_write(path, lambda stream: json.dump(data, stream, indent=2, allow_nan=False))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
