from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import requests

import metatlas2.logging_config as lcf

logger = lcf.get_logger('modelseed_retrieval')

_MODELSEED_API_URL = (
    "https://api.github.com/repos/ModelSEED/ModelSEEDDatabase/"
    "contents/Biochemistry?ref=dev"
)
_MODELSEED_RAW_BASE = (
    "https://raw.githubusercontent.com/ModelSEED/ModelSEEDDatabase/"
    "dev/Biochemistry/"
)
_SHARD_PATTERN = re.compile(r"^compounds_\d+\.tsv$")


def _get_modelseed_compounds(cache_path: Path) -> pd.DataFrame:
    """Load the ModelSEED compounds table, fetching and caching it if needed.

    Args:
        cache_path: Permanent local path to store/load the merged TSV.

    Returns:
        DataFrame with all ModelSEED compound columns.  Returns an empty
        DataFrame if the fetch fails.
    """
    if cache_path.exists():
        logger.info(f"Loading ModelSEED compounds from local cache: {cache_path}")
    else:
        logger.info(f"Querying GitHub Contents API for ModelSEED compound shards: {_MODELSEED_API_URL}")
        try:
            api_resp = requests.get(
                _MODELSEED_API_URL,
                timeout=30,
                headers={"Accept": "application/vnd.github+json"},
            )
            api_resp.raise_for_status()
            shard_names = sorted(
                entry["name"]
                for entry in api_resp.json()
                if _SHARD_PATTERN.match(entry.get("name", ""))
            )
            if not shard_names:
                raise ValueError(
                    f"No compound shard files matching 'compounds_NN.tsv' found via "
                    f"GitHub Contents API at {_MODELSEED_API_URL}"
                )
            logger.info(f"Found {len(shard_names)} ModelSEED compound shards: {shard_names[0]} … {shard_names[-1]}")

            frames: list[pd.DataFrame] = []
            for name in shard_names:
                url = _MODELSEED_RAW_BASE + name
                logger.info(f"Fetching ModelSEED shard: {url}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                frames.append(pd.read_csv(io.StringIO(resp.text), sep="\t", low_memory=False))

            merged = pd.concat(frames, ignore_index=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_csv(cache_path, sep="\t", index=False)
            logger.info(f"Saved merged ModelSEED compounds cache ({len(merged)} rows) to {cache_path}")
        except Exception as exc:
            logger.error(f"Failed to fetch ModelSEED compounds: {exc}")
            return pd.DataFrame()

    return pd.read_csv(cache_path, sep="\t", low_memory=False)


def _build_inchikey_to_cpd(cache_path: Path) -> dict[str, str]:
    """Return a mapping of InChIKey to semicolon-joined ModelSEED CPD IDs.

    Rows with no InChIKey are dropped immediately to reduce memory usage.
    When multiple CPD IDs share the same InChIKey, all are retained and
    joined with ``';'`` (e.g. ``'cpd00001;cpd99999'``).

    Args:
        cache_path: Passed through to :func:`_get_modelseed_compounds`.

    Returns:
        Dict mapping InChIKey strings to semicolon-joined CPD ID strings.
        Empty dict if the ModelSEED table cannot be loaded.
    """
    df = _get_modelseed_compounds(cache_path)

    if df.empty or "inchikey" not in df.columns or "id" not in df.columns:
        logger.error(
            "ModelSEED compounds TSV does not contain expected columns "
            "'id' / 'inchikey'. Available: %s", list(df.columns)
        )
        return {}

    df = df[["id", "inchikey"]].dropna(subset=["inchikey"])
    df = df[df["inchikey"].str.strip() != ""]

    mapping = (
        df.groupby("inchikey")["id"]
        .agg(lambda ids: ";".join(ids))
        .to_dict()
    )

    n_multi = sum(1 for v in mapping.values() if ";" in v)
    logger.info(
        "Built ModelSEED InChIKey→CPD mapping: %d unique InChIKeys "
        "(%d with multiple CPD IDs).",
        len(mapping), n_multi,
    )
    return mapping


def retrieve_modelseed_info(
    compounds: pd.DataFrame,
    modelseed_cache_path: str,
) -> pd.DataFrame:
    """Add a ``modelseed_id`` column to a compounds DataFrame by InChIKey lookup.

    Loads (or fetches and caches) the ModelSEED compound table, builds an
    InChIKey → CPD ID mapping, and maps each row's ``inchi_key`` to the
    corresponding ModelSEED identifier(s).  When a compound maps to multiple
    CPD IDs they are joined with ``';'``.

    Args:
        compounds:             DataFrame with an ``inchi_key`` column.
        modelseed_cache_path:  Path to the local ModelSEED TSV cache file.
                               Created on first call if it does not exist.

    Returns:
        The input DataFrame with a new ``modelseed_id`` column (empty string
        for compounds not found in ModelSEED).  The original DataFrame is not
        modified.
    """
    if not modelseed_cache_path:
        logger.warning("modelseed_cache_path not provided — skipping ModelSEED lookup.")
        compounds = compounds.copy()
        compounds["modelseed_id"] = ""
        return compounds

    cache_path = Path(modelseed_cache_path)
    inchikey_to_cpd = _build_inchikey_to_cpd(cache_path)

    if not inchikey_to_cpd:
        logger.warning("ModelSEED mapping is empty — modelseed_id will be blank for all compounds.")
        compounds = compounds.copy()
        compounds["modelseed_id"] = ""
        return compounds

    compounds = compounds.copy()
    compounds["modelseed_id"] = (
        compounds["inchi_key"]
        .map(inchikey_to_cpd)
        .fillna("")
    )

    found = (compounds["modelseed_id"] != "").sum()
    logger.info(
        "ModelSEED lookup complete: %d / %d compounds matched a CPD ID.",
        found, len(compounds),
    )
    return compounds
