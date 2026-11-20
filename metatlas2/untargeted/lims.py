"""LabKey LIMS access for the untargeted pipeline.

:class:`LimsClient` talks to the real LabKey server. The ``APIWrapper`` is
created lazily (on first use) so importing this module never reads
credentials or touches the network. :class:`InMemoryLims` implements the same
interface on pandas DataFrames for tests and offline experiments.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

import metatlas2.logging_config as lcf

logger = lcf.get_logger("untargeted.lims")

WRITE_METHODS = ("update", "insert", "delete")


class LimsBackend(Protocol):
    """Minimal interface used by the pipeline steps."""

    dry_run: bool

    def get_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame: ...

    def update_table(self, df: pd.DataFrame, table: str, method: str = "update") -> None: ...


def clean_update_df(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy cleanup applied before LIMS updates: ``'NaN'`` strings and NaNs become 0."""
    df = df.replace("NaN", 0)
    return df.fillna(0)


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-serialisable records (numpy scalars -> Python)."""
    records = df.to_dict("records")
    out = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if hasattr(v, "item"):
                v = v.item()
            clean[k] = v
        out.append(clean)
    return out


class LimsClient:
    """LabKey LIMS client using ``labkey.api_wrapper.APIWrapper``.

    Args:
        server: LabKey domain, e.g. ``"metatlas.jgi.doe.gov"``.
        container: LabKey container path, e.g. ``"LIMS/"``.
        api_key_file: File containing the API key on its first line.
        schema: LabKey schema holding the lists (``"lists"``).
        timeout: Request timeout passed to LabKey (legacy value 10000).
        max_rows: Maximum rows fetched by :meth:`get_table`.
        batch_size: Rows per write request (~1000 avoids 504 errors).
        dry_run: If True, writes are logged but not sent.
    """

    def __init__(
        self,
        server: str,
        container: str,
        api_key_file: str,
        schema: str = "lists",
        timeout: int = 10000,
        max_rows: int = 1_000_000,
        batch_size: int = 1000,
        dry_run: bool = False,
    ) -> None:
        self.server = server
        self.container = container
        self.api_key_file = api_key_file
        self.schema = schema
        self.timeout = int(timeout)
        self.max_rows = int(max_rows)
        self.batch_size = int(batch_size)
        self.dry_run = dry_run
        self._api = None

    @classmethod
    def from_config(cls, lims_config: dict[str, Any], dry_run: bool = False) -> "LimsClient":
        """Build a client from the ``LIMS`` config section."""
        return cls(
            server=lims_config["server"],
            container=lims_config["container"],
            api_key_file=lims_config["api_key_file"],
            schema=lims_config.get("schema", "lists"),
            timeout=lims_config.get("timeout", 10000),
            max_rows=lims_config.get("max_rows", 1_000_000),
            batch_size=lims_config.get("batch_size", 1000),
            dry_run=dry_run,
        )

    @property
    def api(self):
        """The lazily-created LabKey ``APIWrapper``."""
        if self._api is None:
            from labkey.api_wrapper import APIWrapper

            api_key = Path(self.api_key_file).read_text().strip()
            self._api = APIWrapper(self.server, self.container, use_ssl=True, api_key=api_key)
        return self._api

    def get_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch a LabKey list as a DataFrame (columns starting with ``_`` are dropped).

        Raises:
            RuntimeError: If LabKey returns no result.
        """
        cols = "*" if columns is None else ",".join(columns)
        sql = f"SELECT {cols} FROM {table};"
        result = self.api.query.execute_sql(self.schema, sql, max_rows=self.max_rows, timeout=self.timeout)
        if result is None:
            raise RuntimeError(f"execute_sql: failed to load results from {self.schema}.{table}")
        df = pd.DataFrame(result["rows"])
        return df[[c for c in df.columns if not str(c).startswith("_")]]

    def update_table(
        self, df: pd.DataFrame, table: str, method: str = "update", pause_time: float | None = None
    ) -> None:
        """Update, insert, or delete rows in a LabKey list in batches.

        For ``update``/``delete`` each row must carry its ``Key``.
        """
        if method not in WRITE_METHODS:
            raise ValueError(f"Unknown LIMS write method {method!r}; expected one of {WRITE_METHODS}")
        if df.empty:
            return
        if self.dry_run:
            logger.info(f"[dry-run] Would {method} {len(df)} row(s) in LIMS list '{table}':")
            for rec in _json_safe_records(df):
                logger.info(f"[dry-run]   {rec}")
            return
        n_batches = math.ceil(df.shape[0] / self.batch_size)
        writer = {
            "update": self.api.query.update_rows,
            "insert": self.api.query.insert_rows,
            "delete": self.api.query.delete_rows,
        }[method]
        for i in range(n_batches):
            sub_df = df.iloc[i * self.batch_size : (i + 1) * self.batch_size]
            writer(self.schema, table, _json_safe_records(sub_df), timeout=self.timeout)
            if pause_time is not None:
                time.sleep(pause_time)


class InMemoryLims:
    """In-memory stand-in for :class:`LimsClient` (used by tests).

    Tables are DataFrames keyed by name. ``insert`` assigns an integer ``Key``
    and every write stamps a ``Modified`` timestamp, like LabKey does.
    """

    def __init__(self, tables: dict[str, pd.DataFrame] | None = None, dry_run: bool = False) -> None:
        self.tables: dict[str, pd.DataFrame] = {k: v.copy() for k, v in (tables or {}).items()}
        self.dry_run = dry_run
        self.writes: list[tuple[str, str, pd.DataFrame]] = []

    def get_table(self, table: str, columns: list[str] | None = None) -> pd.DataFrame:
        df = self.tables.get(table, pd.DataFrame()).copy()
        if columns is not None:
            df = df[columns]
        return df.reset_index(drop=True)

    def update_table(self, df: pd.DataFrame, table: str, method: str = "update", pause_time: float | None = None) -> None:
        if method not in WRITE_METHODS:
            raise ValueError(f"Unknown LIMS write method {method!r}")
        self.writes.append((table, method, df.copy()))
        if self.dry_run or df.empty:
            return
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current = self.tables.get(table, pd.DataFrame())
        if method == "insert":
            start = int(current["Key"].max()) + 1 if "Key" in current.columns and not current.empty else 1
            new = df.copy().reset_index(drop=True)
            new["Key"] = range(start, start + len(new))
            new["Modified"] = now
            self.tables[table] = pd.concat([current, new], ignore_index=True)
        elif method == "update":
            current = current.copy()
            for rec in df.to_dict("records"):
                mask = current["Key"] == rec["Key"]
                for col, val in rec.items():
                    if col == "Key":
                        continue
                    if col not in current.columns:
                        current[col] = None
                    current[col] = current[col].astype(object)
                    current.loc[mask, col] = val
                current.loc[mask, "Modified"] = now
            self.tables[table] = current
        else:
            self.tables[table] = current[~current["Key"].isin(df["Key"])].reset_index(drop=True)
