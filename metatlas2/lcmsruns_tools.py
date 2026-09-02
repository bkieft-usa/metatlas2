from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf

logger = lcf.get_logger('lcmsruns_tools')

SAMPLE_NAME_TO_FILE_TYPE: dict[str, str] = {
    'qc':       'qc',
    'istd':     'istd',
    'exctrl':   'exctrl',
    'txctrl':   'exctrl',
    'injbl':    'injbl',
    'blank':    'injbl',
    'refstd':   'refstd',
    'standard': 'refstd',
}


def _classify_file_type(fields: dict) -> str:
    """Return the file_type category for a parsed filename field dict.

    Classification is driven exclusively by the ``sample_name`` field from
    ``fpf.parse_file_name()``, with ``run_metadata`` used as a fallback.
    Both fields are exact-matched (case-insensitive) against
    ``SAMPLE_NAME_TO_FILE_TYPE``; if neither matches, ``'experimental'`` is
    returned.
    """
    sample_name = fields.get("sample_name", "").lower().strip()
    run_metadata = fields.get("run_metadata", "").lower().strip()

    # Primary: exact match on sample_name
    if sample_name in SAMPLE_NAME_TO_FILE_TYPE:
        return SAMPLE_NAME_TO_FILE_TYPE[sample_name]

    # Fallback: exact match on run_metadata
    if run_metadata in SAMPLE_NAME_TO_FILE_TYPE:
        return SAMPLE_NAME_TO_FILE_TYPE[run_metadata]

    return 'experimental'


def get_project_lcmsruns_from_disk(project_raw_files_path: str) -> list[dict]:
    project_path = Path(project_raw_files_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    if failed := list(project_path.glob("*.failed")):
        logger.error("\n".join([f"  - {f.name}" for f in failed]))
        raise ValueError(f"Please address {len(failed)} .failed files in {project_path}.")

    lcmsruns = []
    for ext in ['raw', 'mzML', 'h5']:
        files = list(project_path.glob(f"*.{ext}"))
        logger.info(f"Found {len(files)} .{ext} files")
        
        for file_path in files:
            try:
                fields = fpf.parse_file_name(file_path.name)

                lcmsruns.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "file_format": ext,
                    "file_type": _classify_file_type(fields),
                    "chromatography": fields.get("chromatography", "unknown"),
                    "ms_level": fields.get("ms_level", "unknown").lower(),
                    "polarity": fields.get("polarity", "unknown").lower(),
                    "created_by": None if fields.get("created_by") is None else str(fields.get("created_by")).lower(),
                    "created_date": None if fields.get("created_date") is None else str(fields.get("created_date")).lower(),
                })
            except Exception as e:
                raise ValueError(f"Error parsing filename '{file_path.name}': {e}")

        if files:
            file_types = [r["file_type"] for r in lcmsruns if r["file_format"] == ext]
            chroms = [r["chromatography"] for r in lcmsruns if r["file_format"] == ext]
            ms_levels = [r["ms_level"] for r in lcmsruns if r["file_format"] == ext]
            polarities = [r["polarity"] for r in lcmsruns if r["file_format"] == ext]
            logger.info(f"  file_type counts: {dict(Counter(file_types))}")
            logger.info(f"  chromatography counts: {dict(Counter(chroms))}")
            logger.info(f"  ms_level counts: {dict(Counter(ms_levels))}")
            logger.info(f"  polarity counts: {dict(Counter(polarities))}")

    if not lcmsruns:
        raise ValueError(f"No .raw, .mzML, or .h5 files found in {project_path}")
    
    logger.info(f"Returning {len(lcmsruns)} LCMS runs.")
    return lcmsruns

def filter_lcmsruns_list(
    lcmsruns: list[dict],
    include_file_type: list[str] = None,
    exclude_file_type: list[str] = None,
    file_format: str = "h5",
    chromatography: str = None,
    polarity: str = None,
    ms_level: str = None
) -> list[dict]:
    """Filter a list of LCMS run dicts/objects by the given criteria.
    """

    if chromatography:
        chromatography_set = {fpf.normalize_chromatography(chromatography)}
    else:
        chromatography_set = None

    pol_set = set()
    if polarity:
        if isinstance(polarity, (list, set, tuple)):
            pol_set = {fpf.normalize_polarity(str(p)) for p in polarity}
        else:
            pol_set = {fpf.normalize_polarity(str(polarity))}

    # Split each exclude/include list into file_type tokens and polarity tokens.
    def _split_tokens(token_list: list[str] | None):
        """Return (file_type_set, polarity_set) from a mixed token list."""
        _POLARITY_TOKENS = frozenset({"pos", "neg", "fps", "positive", "negative"})
        if not token_list:
            return None, set()
        ft_set, pol_exc = set(), set()
        for tok in token_list:
            tok_lc = tok.lower()
            if tok_lc in _POLARITY_TOKENS:
                pol_exc.add(fpf.normalize_polarity(tok_lc))
            else:
                ft_set.add(tok_lc)
        return (ft_set if ft_set else None), pol_exc

    inc_ft_set, inc_pol_set = _split_tokens(include_file_type)
    exc_ft_set, exc_pol_set = _split_tokens(exclude_file_type)


    if pol_set and "fps" not in pol_set and "fps" not in exc_pol_set:
        if "pos" in pol_set or "neg" in pol_set:
            pol_set.add("fps")
    if inc_pol_set and "fps" not in inc_pol_set and "fps" not in exc_pol_set:
        if "pos" in inc_pol_set or "neg" in inc_pol_set:
            inc_pol_set.add("fps")

    def match(run):
        # Support both dict and object (namedtuple/dataclass) access.
        def get_val(k):
            v = run[k] if isinstance(run, dict) else getattr(run, k, "")
            return v.lower() if isinstance(v, str) else str(v).lower()

        run_ft  = get_val('file_type')
        run_pol = get_val('polarity')

        if inc_ft_set and run_ft not in inc_ft_set:
            return False
        if inc_pol_set and run_pol not in inc_pol_set:
            return False
        if exc_ft_set and run_ft in exc_ft_set:
            return False
        if exc_pol_set and run_pol in exc_pol_set:
            return False
        if file_format and get_val('file_format') != file_format:
            return False
        if chromatography_set and get_val('chromatography') not in chromatography_set:
            return False
        if pol_set and run_pol not in pol_set:
            return False
        if ms_level is not None and get_val('ms_level') != str(ms_level).lower():
            return False
        return True

    logger.info(f"Filtering {len(lcmsruns)} LCMS runs with criteria: ")
    logger.info(f"  include_file_type={inc_ft_set}, include_polarity={inc_pol_set or None}")
    logger.info(f"  exclude_file_type={exc_ft_set}, exclude_polarity={exc_pol_set or None}")
    logger.info(f"  file_format={file_format}")
    logger.info(f"  chromatography={chromatography_set or None}")
    logger.info(f"  polarity={pol_set or None}")
    logger.info(f"  ms_level={ms_level}")
    filtered = [run for run in lcmsruns if match(run)]

    logger.info(f"Filtered to {len(filtered)} out of {len(lcmsruns)} total files.")
    if not filtered:
        raise ValueError("No LCMS runs matched the filter criteria.")

    return filtered