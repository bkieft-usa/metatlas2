"""Small filesystem helpers shared by the untargeted modules."""

from __future__ import annotations

import grp
import os
from pathlib import Path

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.untargeted.status import tab_print

logger = lcf.get_logger("untargeted.utils")


def set_group(path: str | os.PathLike, group: str | None) -> None:
    """Recursively set the Unix group of *path* (file or directory).

    Lightweight replacement for the legacy ``recursive_chown``. Failures are
    logged at INFO (as before) so they don't pollute the warnings/errors email.
    ``group=None`` disables the call.
    """
    if not group:
        return
    path = Path(path)
    try:
        gid = grp.getgrnam(group).gr_gid
        if path.is_dir():
            for root, dirs, files in os.walk(path):
                for name in dirs + files:
                    _chgrp(Path(root) / name, gid)
        _chgrp(path, gid)
    except (KeyError, OSError) as err:
        logger.info(tab_print(f"Note: Could not change group ownership of {path}: {err}", 2))


def _chgrp(path: Path, gid: int) -> None:
    try:
        if path.stat().st_gid != gid:
            os.chown(path, -1, gid)
    except PermissionError:
        # Files owned by other users can't be changed; that's expected and harmless.
        pass
