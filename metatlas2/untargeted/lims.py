"""Backwards-compatible alias: the LIMS client now lives in :mod:`metatlas2.shared_tools.lims`."""

from metatlas2.shared_tools.lims import (  # noqa: F401
    WRITE_METHODS,
    InMemoryLims,
    LimsBackend,
    LimsClient,
    clean_update_df,
)
