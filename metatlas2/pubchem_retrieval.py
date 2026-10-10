from __future__ import annotations

import json
import pandas as pd
import requests
import time
import re
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

import pubchempy as pcp

import metatlas2.logging_config as lcf
from metatlas2.utils import get_provenance
logger = lcf.get_logger('pubchem_retrieval')

# PubChem PUG view base URL for full compound records
_PUG_VIEW_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"

# Mapping from PubChem TOCHeading values (in Section 2.3) to our field names
_XREF_HEADING_MAP: dict[str, str] = {
    "CAS":                        "cas_number",
    "ChEBI ID":                   "chebi_id",
    "ChEMBL ID":                  "chembl_id",
    "HMDB ID":                    "hmdb_id",
    "KEGG ID":                    "kegg_id",
    "DrugBank ID":                "drugbank_id",
    "MetaboLights ID":            "metabolights_id",
    "Metabolomics Workbench ID":  "metabolomics_workbench_id",
    "Wikipedia":                  "wikipedia_url",
}


def fetch_pubchem_pug_view(cid: str, heading: str, timeout: int = 15) -> dict:
    """Call the PubChem PUG view API for a single compound and heading filter.

    Uses the ``?heading=`` query parameter to request only the relevant
    subsection of the full compound record, keeping the response payload small.

    Args:
        cid:     PubChem Compound ID (as a string or int).
        heading: Section heading to filter on, e.g. ``"Other Identifiers"``
                 or ``"Description"``.
        timeout: HTTP request timeout in seconds (default 15).

    Returns:
        Parsed JSON response as a ``dict``, or an empty ``dict`` if the
        request fails or the response cannot be parsed.
    """
    url = _PUG_VIEW_BASE.format(cid=cid)
    try:
        resp = requests.get(url, params={"heading": heading}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.debug(f"PUG view request failed for CID {cid} heading={heading!r}: {exc}")
        return {}


def _extract_pug_view_strings(data: dict, target_heading: str) -> list[str]:
    """Walk a PUG view JSON record and collect all string values under a heading.

    Recursively searches ``Section`` lists for a ``TOCHeading`` matching
    *target_heading* (case-sensitive), then collects every
    ``Information[*].Value.StringWithMarkup[*].String`` value found within
    that section.

    Args:
        data:           Parsed PUG view JSON (the full ``Record`` dict or any
                        sub-section dict).
        target_heading: The ``TOCHeading`` value to search for.

    Returns:
        List of string values found, in document order.  Empty list if the
        heading is not present or contains no string values.
    """
    results: list[str] = []

    def _walk(node: dict | list) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return

        if node.get("TOCHeading") == target_heading:
            # Collect all string values from Information entries
            for info in node.get("Information", []):
                value = info.get("Value", {})
                for swm in value.get("StringWithMarkup", []):
                    s = swm.get("String", "")
                    if s:
                        results.append(s)
            return  # Don't recurse further into matched section

        # Recurse into nested sections
        for section in node.get("Section", []):
            _walk(section)

    record = data.get("Record", data)
    for section in record.get("Section", []):
        _walk(section)

    return results


def fetch_pubchem_description(cid: str) -> str:
    """Retrieve the compound description text from PubChem PUG view.

    Requests the ``Description`` heading from the PUG view API and returns
    the first non-empty description string found.  Prefers descriptions from
    authoritative sources (MeSH, FDA, NCI) over Wikipedia by checking the
    ``Reference`` field when available, but falls back to the first available
    string if no preference can be determined.

    Args:
        cid: PubChem Compound ID.

    Returns:
        Description string, or ``""`` if none is available.
    """
    data = fetch_pubchem_pug_view(cid, "Description")
    if not data:
        return ""

    # Collect all Information entries from the Description section with source info
    record = data.get("Record", data)
    descriptions: list[tuple[str, str]] = []  # (source_name, text)

    def _collect_descriptions(node: dict | list) -> None:
        if isinstance(node, list):
            for item in node:
                _collect_descriptions(item)
            return
        if not isinstance(node, dict):
            return

        if node.get("TOCHeading") == "Description":
            for info in node.get("Information", []):
                # Try to get source name from Reference
                ref_number = info.get("ReferenceNumber")
                source_name = ""
                if ref_number is not None:
                    # Source name is in the top-level References list
                    for ref in record.get("Reference", []):
                        if ref.get("ReferenceNumber") == ref_number:
                            source_name = ref.get("SourceName", "")
                            break

                value = info.get("Value", {})
                for swm in value.get("StringWithMarkup", []):
                    text = swm.get("String", "").strip()
                    if text:
                        descriptions.append((source_name, text))
            return

        for section in node.get("Section", []):
            _collect_descriptions(section)

    for section in record.get("Section", []):
        _collect_descriptions(section)

    if not descriptions:
        return ""

    # Prefer authoritative sources over Wikipedia
    preferred_sources = ("MeSH", "FDA", "NCI", "WHO", "KEGG", "ChEBI", "DrugBank")
    for source, text in descriptions:
        if any(pref.lower() in source.lower() for pref in preferred_sources):
            return text

    # Fall back to first available description
    return descriptions[0][1]


def fetch_pubchem_xrefs(cid: str) -> dict[str, str]:
    """Retrieve cross-reference identifiers from PubChem Section 2.3.

    Calls the PUG view API with ``heading=Other Identifiers`` and extracts
    identifiers for third-party databases.  The ``Wikipedia`` entry is
    converted to a full English Wikipedia URL.

    Args:
        cid: PubChem Compound ID.

    Returns:
        Dict with keys: ``cas_number``, ``chebi_id``, ``chembl_id``,
        ``hmdb_id``, ``kegg_id``, ``drugbank_id``, ``metabolights_id``,
        ``metabolomics_workbench_id``, ``wikipedia_url``.
        All values default to ``""`` if not found.
    """
    result: dict[str, str] = {field: "" for field in _XREF_HEADING_MAP.values()}

    data = fetch_pubchem_pug_view(cid, "Other Identifiers")
    if not data:
        return result

    for toc_heading, field_name in _XREF_HEADING_MAP.items():
        values = _extract_pug_view_strings(data, toc_heading)
        if not values:
            continue
        raw = values[0].strip()
        if not raw:
            continue

        if field_name == "wikipedia_url":
            # Convert article title to full URL
            article = raw.replace(" ", "_")
            result[field_name] = f"https://en.wikipedia.org/wiki/{article}"
        else:
            result[field_name] = raw

    return result


def fetch_pubchem_entry(inchi_key: str, timestamp: str) -> dict[str, Any]:
    """Retrieve comprehensive compound data from PubChem for a single InChIKey.

    Looks up the PubChem CID via the InChIKey namespace, fetches the full
    :class:`pubchempy.Compound` record for core properties and computed
    descriptors, then makes two additional PUG view API calls to retrieve
    the compound description and cross-reference identifiers (Section 2.3
    "Other Identifiers").

    Args:
        inchi_key: Standard InChIKey string used to query PubChem.
        timestamp: ISO-format timestamp string to record as
                   ``pubchem_retrieval_date``.

    Returns:
        A ``dict`` with all compound fields (see :data:`_COMPOUND_FIELDS`),
        or ``None`` if the compound is not found or an unrecoverable error
        occurs.
    """
    try:
        # ── Step 1: resolve InChIKey → CID ──────────────────────────────────
        cid_result = pcp.get_compounds(inchi_key, namespace='inchikey',
                                        as_dataframe=True, listkey_count=5)
        if cid_result.empty:
            return None

        cid_result = cid_result.reset_index()
        cid = cid_result['cid'].to_string(index=False)

        # Handle multiple CIDs — take the last (highest, most canonical)
        if "\n" in cid:
            cid = (cid.rstrip().split('\n'))[-1]
        cid = cid.strip()

        # ── Step 2: fetch full compound record via pubchempy ─────────────────
        compound = pcp.Compound.from_cid(cid)

        filtered_synonyms = _filter_synonym_list(compound.synonyms) if compound.synonyms else "Undefined"

        # ── Step 3: extract computed descriptors (no extra API call) ─────────
        def _safe_float(val) -> float | None:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        def _safe_int(val) -> int | None:
            try:
                return int(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        # ── Step 4: PUG view — description ───────────────────────────────────
        description = fetch_pubchem_description(cid)

        # ── Step 5: PUG view — cross-reference identifiers ───────────────────
        xrefs = fetch_pubchem_xrefs(cid)

        # CAS from PUG view is authoritative; fall back to synonym regex
        cas_number = xrefs.pop("cas_number", "")
        if not cas_number and compound.synonyms:
            for synonym in compound.synonyms:
                if '-' in synonym and len(synonym.split('-')) == 3:
                    parts = synonym.split('-')
                    if all(p.isdigit() for p in parts):
                        cas_number = synonym
                        break

        compound_data: dict[str, Any] = {
            # ── Core identifiers ─────────────────────────────────────────────
            "pubchem_cid":                    str(compound.cid) if compound.cid else "",
            "iupac_name":                     str(compound.iupac_name) if compound.iupac_name else "",
            "synonyms":                       filtered_synonyms,
            "inchi":                          str(compound.inchi) if compound.inchi else "",
            "smiles":                         str(compound.connectivity_smiles) if compound.connectivity_smiles else "",
            "formula":                        str(compound.molecular_formula) if compound.molecular_formula else "",
            "mono_isotopic_molecular_weight": _safe_float(compound.monoisotopic_mass),
            "cas_number":                     cas_number,
            # ── Computed descriptors (from pubchempy, no extra API call) ──────
            "isomeric_smiles":                str(compound.smiles) if compound.smiles else "",
            "molecular_weight":               _safe_float(compound.molecular_weight),
            "xlogp":                          _safe_float(compound.xlogp),
            "tpsa":                           _safe_float(compound.tpsa),
            "complexity":                     _safe_float(compound.complexity),
            "charge":                         _safe_int(compound.charge),
            "heavy_atom_count":               _safe_int(compound.heavy_atom_count),
            "hbd_count":                      _safe_int(compound.h_bond_donor_count),
            "hba_count":                      _safe_int(compound.h_bond_acceptor_count),
            "rotatable_bond_count":           _safe_int(compound.rotatable_bond_count),
            # ── Cross-reference identifiers (PUG view Section 2.3) ────────────
            "chebi_id":                       xrefs.get("chebi_id", ""),
            "chembl_id":                      xrefs.get("chembl_id", ""),
            "hmdb_id":                        xrefs.get("hmdb_id", ""),
            "kegg_id":                        xrefs.get("kegg_id", ""),
            "drugbank_id":                    xrefs.get("drugbank_id", ""),
            "metabolights_id":                xrefs.get("metabolights_id", ""),
            "metabolomics_workbench_id":      xrefs.get("metabolomics_workbench_id", ""),
            "wikipedia_url":                  xrefs.get("wikipedia_url", ""),
            # ── Description (PUG view) ────────────────────────────────────────
            "description":                    description,
            # ── Retrieval metadata ────────────────────────────────────────────
            "pubchem_retrieval_date":         timestamp,
            "pubchem_compound_url":           f"https://pubchem.ncbi.nlm.nih.gov/compound/{compound.cid}" if compound.cid else "",
        }

        return compound_data

    except Exception as e:
        logger.debug(f"fetch_pubchem_entry failed for {inchi_key}: {e}")
        return None


def _filter_synonym_list(synonyms: list[str]) -> str:
    """Select the best human-readable synonym from a PubChem synonym list.

    Removes entries that start with known database-code prefixes (e.g. ``CHEMBL``,
    ``ZINC``, ``DTXSID``), strips entries that are purely numeric codes, and
    returns the shortest remaining synonym as a proxy for the most concise name.

    Args:
        synonyms: List of synonym strings as returned by PubChem, or a single
            string.  ``None`` or ``["Undefined"]`` are handled gracefully.

    Returns:
        The shortest non-problematic synonym string, or ``"Undefined"`` if no
        suitable synonym remains after filtering.
    """
    if not synonyms or synonyms == ["Undefined"]:
        return "Undefined"
    
    if isinstance(synonyms, str):
        synonyms = [synonyms]
    
    problematic_prefixes = (
        "nan", "Untitled", "Oprea", "Opera", "AKO", "CHEMBL", "SR-", "SCHEM", 
        "EU-", "MLS", "NSC", "ChemDiv", "ST0", "TimTec", "HMS", "BIM", "CB", 
        "CCG-", "Cambridge", "SMR", "AB0", "BRD-", "NCG", "BDBM", "CBKinase", 
        "BAS ", "ZINC", "GNF", "SQX", "CDS", "STK", "NCI", "TNP", "Boc-Tyr-OH", 
        "PD", "UNM", "BSP", "CCRIS", "MFCD", "IDI", "ST5", "AC1", "WAY-", "KUC", 
        "DTXSID", "MixCom", "CK-", "ASN ", "MMV", "SKI-", "VU", "SMSF", "Bio2", 
        "REGID", "SDCC", "BCBc", "SMP", "TCMDC", "cid_", "BCP", "AST ", "SY0", 
        "AM-", "IFLab", "Cream"
    )

    filtered_synonyms = [x for x in synonyms if not x.startswith(problematic_prefixes)]
    filtered_synonyms = [x for x in filtered_synonyms if not "cream" in x.lower()]
    filtered_synonyms = [x for x in filtered_synonyms
                        if not re.sub(r'[A-Z\-]', '', x).isdigit()]
    
    if not filtered_synonyms:
        return "Undefined"
    
    return min(filtered_synonyms, key=len)


def load_or_create_pubchem_cache(pubchem_cache_path: str, use_cache: bool = True) -> dict[str, dict[str, Any]]:
    """
    Load existing PubChem cache or create new one.
    
    Args:
        pubchem_cache_path: Path to cache JSON file
        use_cache: If True, load existing cache; if False, return empty cache
        
    Returns:
        Dictionary mapping inchi_keys to PubChem data
    """
    if not use_cache:
        logger.info("Cache disabled - starting with empty cache")
        return {}
    
    try:
        cache_path = Path(pubchem_cache_path)
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                pubchem_cache = json.load(f)
            logger.info(f"Loaded PubChem cache with {len(pubchem_cache)} entries from {pubchem_cache_path}")
            return pubchem_cache
        else:
            logger.info(f"Cache file not found at {pubchem_cache_path} - creating new cache")
            return {}
    except Exception as e:
        logger.error(f"Error loading PubChem cache: {e}")
        logger.info("Starting with empty cache")
        return {}


def save_pubchem_cache(cache: dict[str, dict], cache_filename: str) -> None:
    """Persist the in-memory PubChem cache to a JSON file on disk.

    Creates any missing parent directories before writing.  Logs a warning if
    the write fails but does not raise.

    Args:
        cache: Mapping of InChIKey strings to PubChem data dicts, as maintained
            by :func:`retrieve_pubchem_info`.
        cache_filename: Destination file path (string or path-like).

    Returns:
        None.
    """
    cache_file = Path(cache_filename)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved global PubChem cache with {len(cache)} entries to {cache_file}")
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def retrieve_pubchem_info(compounds: pd.DataFrame, pubchem_cache_path: str) -> pd.DataFrame:
    """Retrieve PubChem information for compounds, using and updating the on-disk cache.

    Compounds already present in the cache are returned immediately without
    making any API calls.  New compounds are fetched from PubChem (up to 3
    API calls each) and added to the cache before returning.

    Args:
        compounds:          DataFrame with an ``inchi_key`` column.
        pubchem_cache_path: Path to the JSON cache file.  Created on first
                            call if it does not exist.

    Returns:
        The input DataFrame enriched with all PubChem-derived columns.
    """
    pubchem_cache = load_or_create_pubchem_cache(pubchem_cache_path, use_cache=True)
    prov = get_provenance()
    unique_inchi_keys = compounds['inchi_key'].dropna().unique()

    # Only fetch compounds not already in the cache
    compounds_to_fetch = [key for key in unique_inchi_keys if key not in pubchem_cache]
    compounds_in_cache = [key for key in unique_inchi_keys if key in pubchem_cache]
    logger.info(f"Compounds already in cache: {len(compounds_in_cache)}")
    logger.info(f"Compounds needing PubChem lookup: {len(compounds_to_fetch)}")

    if compounds_to_fetch:
        logger.info(f"Fetching PubChem data for {len(compounds_to_fetch)} compounds...")
        logger.info("This may take several minutes depending on the number of compounds.")
        logger.info("Each compound requires up to 3 API calls (core data + description + xrefs).")
        
        new_entries = 0
        updated_entries = 0
        
        for inchi_key in tqdm(compounds_to_fetch, desc="Fetching PubChem data"):
            was_in_cache = inchi_key in pubchem_cache
            
            pubchem_data = fetch_pubchem_entry(inchi_key, prov['timestamp'])

            if pubchem_data:
                pubchem_cache[inchi_key] = pubchem_data
                
                if was_in_cache:
                    updated_entries += 1
                else:
                    new_entries += 1

            # Sleep 1.0s per compound: up to 3 API calls × ~0.33s each stays
            # well within PubChem's 5 requests/second rate limit.
            time.sleep(1.0)
        
        save_pubchem_cache(pubchem_cache, pubchem_cache_path)
        logger.info(f"Cache update completed: {new_entries} new entries added, {updated_entries} entries updated")
        
    else:
        logger.info("All compounds already in cache!")

    pubchem_df = pd.DataFrame.from_dict(pubchem_cache, orient='index')
    pubchem_df.index.name = 'inchi_key'
    pubchem_df = pubchem_df.reset_index()
    
    compounds = compounds.merge(pubchem_df, on='inchi_key', how='left', suffixes=('', '_pubchem'))
    
    for col in pubchem_df.columns:
        if col != 'inchi_key' and col in compounds.columns:
            # Fill missing values with PubChem data
            pubchem_col = f"{col}_pubchem"
            if pubchem_col in compounds.columns:
                compounds[col] = compounds[col].fillna(compounds[pubchem_col])
                compounds = compounds.drop(columns=[pubchem_col])

    successful_retrievals = [k for k, v in pubchem_cache.items() 
                            if v.get('pubchem_cid') and v.get('pubchem_cid') != '']
    failed_retrievals = [k for k, v in pubchem_cache.items() 
                        if not v.get('pubchem_cid') or v.get('pubchem_cid') == '']
                        
    if failed_retrievals:
        logger.warning(f"Some compounds not found in PubChem: {failed_retrievals[:5]}...")
        logger.warning("These will be created with minimal information from the input table.")

    logger.info(f"PubChem data retrieval complete!")
    logger.info(f"    Total compounds in global cache: {len(pubchem_cache)}")
    logger.info(f"    Successful PubChem retrievals in cache: {len(successful_retrievals)}")
    logger.info(f"    Failed retrievals in cache: {len(failed_retrievals)}")
    
    return compounds
