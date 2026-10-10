from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import metatlas2.logging_config as lcf

logger = lcf.get_logger('chebi_retrieval')

_CHEBI_REST_URL = "https://www.ebi.ac.uk/chebi/backend/api/public/compound/{num}/"

# Shared session with automatic retries on rate limiting / server errors
_SESSION = requests.Session()
_SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
    ),
)


def fetch_metacyc_id_from_chebi(chebi_id: str, timeout: int = 15) -> str:
    """Query the ChEBI public API and return MetaCyc accession(s) for a ChEBI ID.

    Args:
        chebi_id: ChEBI identifier, e.g. ``"CHEBI:17154"`` or ``"17154"``.
        timeout:  HTTP request timeout in seconds.

    Returns:
        MetaCyc ID string (semicolon-joined if several, e.g. ``"NIACINAMIDE"``),
        or ``""`` if none is found or the lookup fails for any reason.
    """
    if not chebi_id or not chebi_id.strip():
        return ""

    num = chebi_id.strip().upper().replace("CHEBI:", "")
    try:
        resp = _SESSION.get(
            _CHEBI_REST_URL.format(num=num),
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("ChEBI lookup failed for %s: %s", chebi_id, exc)
        return ""

    # database_accessions is a dict keyed by type (e.g. "MANUAL_X_REF", "CITATION"),
    # each value is a list of dicts with "source_name" and "accession_number".
    # MetaCyc entries appear under "MANUAL_X_REF" with source_name == "MetaCyc".
    db_acc = data.get("database_accessions", {})
    all_entries = []
    if isinstance(db_acc, dict):
        for entries in db_acc.values():
            if isinstance(entries, list):
                all_entries.extend(entries)
    elif isinstance(db_acc, list):
        all_entries = db_acc

    ids = [
        str(acc["accession_number"]).strip()
        for acc in all_entries
        if isinstance(acc, dict)
        and str(acc.get("source_name", "")).lower() == "metacyc"
        and acc.get("accession_number")
    ]
    result = ";".join(dict.fromkeys(ids))  # dedupe, keep order
    logger.debug("ChEBI %s -> MetaCyc: %r", chebi_id, result or "(none)")
    return result


def load_or_create_chebi_cache(cache_path: str, use_cache: bool = True) -> dict[str, str]:
    """Load the ChEBI->MetaCyc cache from disk, or return an empty dict."""
    if not use_cache:
        logger.info("ChEBI cache disabled - starting with empty cache.")
        return {}

    path = Path(cache_path)
    if not path.exists():
        logger.info("ChEBI cache not found at %s - creating new cache.", cache_path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info("Loaded ChEBI cache with %d entries from %s.", len(cache), cache_path)
        return cache
    except Exception as exc:
        logger.error("Error loading ChEBI cache: %s - starting fresh.", exc)
        return {}


def save_chebi_cache(cache: dict[str, str], cache_path: str) -> None:
    """Write the cache to a JSON file, creating parent directories as needed."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info("Saved ChEBI cache with %d entries to %s.", len(cache), cache_path)
    except Exception as exc:
        logger.error("Error saving ChEBI cache: %s", exc)


def retrieve_metacyc_info(
    compounds: pd.DataFrame,
    chebi_cache_path: str,
    use_chebi_cache: bool = True,
    update_chebi_cache: bool = False,
) -> pd.DataFrame:
    """Add a ``metacyc_id`` column to ``compounds`` using its ``chebi_id`` column.

    Compounds with no ChEBI ID, a failed lookup, or no MetaCyc link get ``""``.
    All results, including blanks, are cached.

    Args:
        compounds:          DataFrame with a ``chebi_id`` column.
        chebi_cache_path:   Path to the JSON cache file.
        use_chebi_cache:    Load/use the cache when True.
        update_chebi_cache: Force re-fetch of all ChEBI IDs.

    Returns:
        Copy of ``compounds`` with a new ``metacyc_id`` column.
    """
    result = compounds.copy()

    if not chebi_cache_path:
        logger.warning("chebi_cache_path not provided - skipping MetaCyc lookup.")
        result["metacyc_id"] = ""
        return result

    if "chebi_id" not in result.columns:
        logger.warning("compounds DataFrame has no 'chebi_id' column - skipping.")
        result["metacyc_id"] = ""
        return result

    cache = load_or_create_chebi_cache(chebi_cache_path, use_cache=use_chebi_cache)

    all_chebi_ids = (
        result["chebi_id"].dropna().astype(str).str.strip().unique().tolist()
    )
    all_chebi_ids = [c for c in all_chebi_ids if c and c.lower() != "nan"]

    if update_chebi_cache or not use_chebi_cache:
        to_fetch = all_chebi_ids
    else:
        to_fetch = [c for c in all_chebi_ids if c not in cache]

    logger.info(
        "ChEBI IDs total: %d; to look up: %d; cached: %d.",
        len(all_chebi_ids), len(to_fetch), len(all_chebi_ids) - len(to_fetch),
    )

    for i, chebi_id in enumerate(to_fetch, start=1):
        cache[chebi_id] = fetch_metacyc_id_from_chebi(chebi_id)
        if use_chebi_cache and i % 100 == 0:
            save_chebi_cache(cache, chebi_cache_path)  # periodic checkpoint
        time.sleep(0.1)

    if to_fetch and use_chebi_cache:
        save_chebi_cache(cache, chebi_cache_path)

    result["metacyc_id"] = (
        result["chebi_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(cache)
        .fillna("")
    )

    found = (result["metacyc_id"] != "").sum()
    logger.info(
        "MetaCyc lookup complete: %d / %d compounds matched a MetaCyc accession.",
        found, len(result),
    )
    return result