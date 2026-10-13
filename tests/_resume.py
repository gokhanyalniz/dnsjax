"""Shared fixtures of the resume tests, core and lab."""

from __future__ import annotations

import json
import tarfile


def write_meta_tar(path, meta: dict) -> None:
    """A metadata-only tar: all the parameter layer reads of a snapshot."""
    import io

    from dnsjax.snapshot_meta import META_MEMBER

    payload = json.dumps(meta).encode()
    with tarfile.open(path, "w") as tf:
        info = tarfile.TarInfo(META_MEMBER)
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
