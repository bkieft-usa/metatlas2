from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter
import metatlas2.shared_tools.file_and_project_format as fpf
import metatlas2.shared_tools.logging_config as lcf

logger = lcf.get_logger('lcmsruns_tools')

_FILE_TYPE_RULES: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {
    "jgi": [
        ("qc-c18qu", "run_metadata",  ("qc", "c18qu")),
        ("qc",       "run_metadata",  ("qc",)),
        ("istd",     "run_metadata",  ("istd",)),
        ("injbl",    "run_metadata",  ("injbl",)),
        ("exctrl",   "sample_name",  ("exctrl",)),
        ("refstd",   "sample_number", ("refstd",)),
    ],
    "egsb": [
        ("qc",       "sample_name",   ("qc",)),
        ("istd",     "sample_name",   ("istd",)),
        ("injbl",    "sample_name",   ("injbl",)),
        ("exctrl",   "sample_name",   ("exctrl",)),
        ("txctrl",   "sample_name",   ("txctrl",)),
        ("refstd",   "sample_number", ("refstd",)),
    ],
}


def _has_segments(segments: list[str], pattern: tuple[str, ...]) -> bool:
    """Return True if *pattern* appears as consecutive whole segments."""
    n = len(pattern)
    return any(tuple(segments[i:i + n]) == pattern for i in range(len(segments) - n + 1))


def _classify_file_type(fields: dict, owner: str) -> str:
    """Return the file_type for a parsed filename field dict.

    *owner* selects the file-format rule set ("jgi" or "egsb").
    """
    owner_lc = (owner or "").lower().strip()
    if owner_lc == "eb":
        owner_lc = "egsb"
    if owner_lc not in _FILE_TYPE_RULES:
        raise ValueError(f"Unknown owner '{owner}'. Expected one of: {list(_FILE_TYPE_RULES)}")

    segs: dict[str, list[str]] = {}
    for file_type, field, pattern in _FILE_TYPE_RULES[owner_lc]:
        if field not in segs:
            segs[field] = (fields.get(field) or "").lower().strip().split("-")
        if _has_segments(segs[field], pattern):
            return file_type

    return "experimental"


def get_project_lcmsruns_from_disk(project_raw_files_path: str, owner: str = "jgi") -> list[dict]:
    project_path = Path(project_raw_files_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    if failed := list(project_path.glob("*.failed")):
        logger.error("\n".join([f"  - {f.name}" for f in failed]))
        raise ValueError(f"Please address {len(failed)} .failed files in {project_path}.")

    counts: dict[str, int] = {}
    lcmsruns = []
    for ext in ['raw', 'mzML', 'h5']:
        files = list(project_path.glob(f"*.{ext}"))
        counts[ext] = len(files)
        logger.info(f"Found {len(files)} .{ext} files")
        
        for file_path in files:
            try:
                fields = fpf.parse_file_name(file_path.name)

                lcmsruns.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "file_format": ext,
                    "file_type": _classify_file_type(fields, owner),
                    "chromatography": fields.get("chromatography", "unknown"),
                    "ms_level": fields.get("ms_level", "unknown").lower(),
                    "polarity": fields.get("polarity", "unknown").lower(),
                    "created_by": None if fields.get("created_by") is None else str(fields.get("created_by")).lower(),
                    "created_date": None if fields.get("created_date") is None else str(fields.get("created_date")).lower(),
                })
            except Exception as e:
                raise ValueError(f"Error parsing filename '{file_path.name}': {e}")

    if not lcmsruns:
        raise ValueError(f"No .raw, .mzML, or .h5 files found in {project_path}")

    # Verify all formats have the same number of files, only if there is at least 1 file of each format. If any format has 0 files, skip the check.
    if all(count > 0 for count in counts.values()):
        unique_counts = set(counts.values())
        if len(unique_counts) > 1:
            counts_str = ", ".join(f".{ext}: {n}" for ext, n in counts.items())
            raise ValueError(
                f"File count mismatch across formats ({counts_str}). "
                "Each .raw file should have a corresponding .mzML and .h5 file."
            )

    # Log combined category breakdown for .h5 files only.
    h5_runs = [r for r in lcmsruns if r["file_format"] == "h5"]
    if h5_runs:
        rows = sorted(
            Counter(
                (r['chromatography'], r['file_type'], r['polarity'], r['ms_level'])
                for r in h5_runs
            ).items()
        )
        headers = ("chromatography", "file_type", "polarity", "ms_level", "count")
        col_widths = [
            max(len(headers[i]), max(len(str(row[0][i])) for row in rows))
            for i in range(4)
        ] + [max(len(headers[4]), max(len(str(row[1])) for row in rows))]
        fmt = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)
        lines = [fmt.format(*headers)]
        lines += [fmt.format(*combo, count) for combo, count in rows]
        logger.info(".h5 category breakdown:\n" + "\n".join(lines))
    
    logger.info(f"Returning {len(lcmsruns)} LCMS runs.")
    return lcmsruns

def filter_lcmsruns_list(
    lcmsruns: list["LCMSRun"],
    include_file_type: list[str] = None,
    exclude_file_type: list[str] = None,
    file_format: str = "h5",
    chromatography: str = None,
    polarity: str = None,
    ms_level: str = None
) -> list["LCMSRun"]:
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