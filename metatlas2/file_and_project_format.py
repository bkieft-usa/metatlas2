import re
from typing import Any

FILE_PATTERN = re.compile(
    r"^(?P<date>[^_]+)_"
    r"(?P<owner>[^_]+)_"
    r"(?P<pi>[^_]+)_"
    r"(?P<project_id>[\d-]+)_"
    r"(?P<project_shortname>[^_]+)_"
    r"(?P<experiment>[^_]+)_"
    r"(?P<instrument>[^_]+)_"
    r"(?P<chromatography>[^_]+)_"
    r"(?P<run_id>[^_]+)_"
    r"(?P<polarity>POS|NEG|FPS)_"
    r"(?P<ms_level>MS1|MS2|MSMS)_"
    r"(?P<sample_number>[^_]+)_"
    r"(?P<sample_name>[^_]+)_"
    r"(?P<replicate>[^_]*)_"
    r"(?P<run_metadata>[^_]*)_"
    r"(?P<run_number>Run\d+|\d+)\."
    r"(?P<ext>raw|mzML|h5)$"
)

PROJECT_PATTERN = re.compile(
    r"^(?P<date>\d{8})_"
    r"(?P<owner>JGI|EB|EGSB)_"
    r"(?P<pi>[^_]+)_"
    r"(?P<project_id>[\d-]+)_"
    r"(?P<project_shortname>[^_]+)_"
    r"(?P<experiment>[^_]+)_"
    r"(?P<instrument>[^_]+)_"
    r"(?P<chromatography>[^_]+)_"
    r"(?P<run_id>[^_]+)"
    r"(?:_(?P<suffix>[^_]+))?$"
)

# Chromatography aliases: maps lowercase input -> canonical lowercase form.
_CHROM_NORMALIZATION_MAP: dict[str, str] = {
    "hilicz": "hilic",
    "c18-ep": "c18",
    "c18ep":  "c18",
}

# Polarity aliases: maps lowercase input -> canonical lowercase form.
_POLARITY_MAP: dict[str, str] = {
    "pos":      "pos",
    "positive": "pos",
    "neg":      "neg",
    "negative": "neg",
    "fps":      "fps",
}


def normalize_chromatography(chrom: str) -> str:
    """Return the canonical *lowercase* chromatography label for matching/storage keys.

    Use this when you need a case-insensitive key (e.g. dict lookups, DB WHERE
    clauses).  For display/output that should preserve the original YAML case,
    use :func:`canonical_chromatography` instead.

    All unrecognised values are simply lowercased.
    """
    if not chrom:
        return chrom
    return _CHROM_NORMALIZATION_MAP.get(chrom.lower(), chrom.lower())


def canonical_chromatography(chrom: str) -> str:
    """Return the canonical chromatography label, preserving the case of *chrom*.

    Alias substitutions (e.g. ``"HILICZ"`` → ``"HILIC"``, ``"C18EP"`` → ``"C18"``)
    are applied while keeping the same case style as the input.  For example::

        canonical_chromatography("HILICZ")  # → "HILIC"
        canonical_chromatography("hilicz")  # → "hilic"
        canonical_chromatography("Hilicz")  # → "Hilic"
        canonical_chromatography("HILIC")   # → "HILIC"

    Use this for display strings, folder names, log messages, and file names
    where the original YAML capitalisation should be preserved.
    """
    if not chrom:
        return chrom
    lc = chrom.lower()
    canonical_lc = _CHROM_NORMALIZATION_MAP.get(lc, lc)
    # Re-apply the same case transformation that the input used.
    if chrom.isupper():
        return canonical_lc.upper()
    if chrom.islower():
        return canonical_lc
    if chrom.istitle():
        return canonical_lc.title()
    # Mixed case: apply the alias substitution but keep the original casing
    # for characters that were not changed by the alias.
    if canonical_lc == lc:
        return chrom  # no alias substitution needed
    # Alias changed the string — return the canonical form in the same case as input.
    if chrom.isupper():
        return canonical_lc.upper()
    return canonical_lc  # fall back to lowercase canonical


def normalize_polarity(polarity: str) -> str:
    """Return the canonical *lowercase* polarity label: ``"pos"``, ``"neg"``, or ``"fps"``.

    Use this when you need a case-insensitive key (e.g. dict lookups, DB WHERE
    clauses).  For display/output that should preserve the original YAML case,
    use :func:`canonical_polarity` instead.
    """
    if not polarity:
        raise ValueError(f"Polarity value is empty or None: {polarity!r}")
    return _POLARITY_MAP.get(polarity.lower(), polarity.lower())


def canonical_polarity(polarity: str) -> str:
    """Return the canonical polarity label, preserving the case of *polarity*.

    Alias substitutions (e.g. ``"POSITIVE"`` → ``"POS"``) are applied while
    keeping the same case style as the input.  For example::

        canonical_polarity("POS")       # → "POS"
        canonical_polarity("pos")       # → "pos"
        canonical_polarity("POSITIVE")  # → "POS"
        canonical_polarity("Positive")  # → "Pos"

    Use this for display strings, folder names, log messages, and file names
    where the original YAML capitalisation should be preserved.
    """
    if not polarity:
        raise ValueError(f"Polarity value is empty or None: {polarity!r}")
    lc = polarity.lower()
    canonical_lc = _POLARITY_MAP.get(lc, lc)
    if polarity.isupper():
        return canonical_lc.upper()
    if polarity.islower():
        return canonical_lc
    if polarity.istitle():
        return canonical_lc.title()
    return canonical_lc

def parse_file_name(filename: str):
    """Parse a filename and return a dict of named capture groups.

    The ``chromatography`` field is alias-substituted (e.g. ``"HILICZ"`` →
    ``"HILIC"``) while preserving the original capitalisation from the filename.
    All other fields are returned verbatim from the regex match.
    """
    match = FILE_PATTERN.match(filename)
    if not match:
        raise Exception(f"Filename '{filename}' does not match the expected format.")
    fields = match.groupdict()
    fields["chromatography"] = canonical_chromatography(fields.get("chromatography", ""))
    return fields

def parse_project_name(project_name: str):
    match = PROJECT_PATTERN.match(project_name)
    if not match:
        raise Exception(f"Project name '{project_name}' does not match the expected format.")
    num_fields = project_name.count('_') + 1
    if num_fields > 9:
        suffix = match.group('suffix')
        if suffix:
            print(f"Warning: Project name '{project_name}' has a suffix '{suffix}'.")
    return project_name

def get_project_chromatography(project_name: str) -> str:
    """Return the chromatography field from a project name, preserving its case.

    Alias substitutions (e.g. ``"HILICZ"`` → ``"HILIC"``) are applied while
    keeping the same capitalisation style as the project name field.
    """
    match = PROJECT_PATTERN.match(project_name)
    if not match:
        raise ValueError(f"Project name '{project_name}' does not match the expected format.")
    return canonical_chromatography(match.group("chromatography"))

def get_file_parts(name: str, part: str):
    """Return the named capture group *part* from a file stem or project name.

    For the ``chromatography`` part, alias substitutions are applied while
    preserving the original capitalisation from the filename.

    Raises:
        ValueError: If *name* does not match the expected filename format or
            *part* is not a recognised capture-group name.
    """
    try:
        if ".h5" not in name:
            name += ".h5"
        match = FILE_PATTERN.match(name)
        if not match:
            raise ValueError(f"File name '{name}' does not match the expected format.")
        try:
            value = match.group(part)
        except IndexError:
            raise ValueError(f"File name '{name}' does not match the expected format.")
        if part == "chromatography":
            return canonical_chromatography(value)
        return value
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"File name '{name}' does not match the expected format.")

# File types that use the type label itself (not sample_name) for group/short_groupname
_CONTROL_FILE_TYPES: frozenset[str] = frozenset({"istd", "istdsetc", "qc", "injbl", "exctrl"})

# Human-readable labels for each file_type
_FILE_TYPE_LABELS: dict[str, str] = {
    "istd":         "ISTD",
    "istdsetc":     "ISTDsEtc",
    "qc":           "QC",
    "injbl":        "InjBl",
    "exctrl":       "ExCtrl",
    "refstd":       "RefStd",
    "experimental": "Experimental",
}


def _file_type_label(file_type: str, sample_name: str) -> str:
    """Return the display label for the group/short_groupname suffix.

    Control types (ISTD, QC, InjBl, ExCtrl) use the type label.
    RefStd and Experimental use the sample_name field.
    """
    ft = (file_type or "experimental").lower()
    if ft in _CONTROL_FILE_TYPES:
        return _FILE_TYPE_LABELS.get(ft, ft.upper())
    return sample_name


def get_filename_header_labels(
    filename: str,
    analyst: str = "",
    rt_alignment_number: int | None = None,
    analysis_number: int | None = None,
    file_type: str | None = None,
) -> dict[str, str]:
    """Return the 6 header-row labels for one data-sheet column.

    Parameters
    ----------
    filename:
        Full filename (with or without path; with or without extension).
    analyst:
        Username string (e.g. ``"bkieft"``).
    rt_alignment_number, analysis_number:
        RTA / TGA numbers used in the ``group`` label.
    file_type:
        Pre-classified file type (``"istd"``, ``"qc"``, ``"injbl"``,
        ``"exctrl"``, ``"refstd"``, ``"experimental"``).  When ``None``
        the function attempts to classify from the filename fields.

    Returns
    -------
    dict with keys: ``group``, ``file``, ``short_groupname``,
    ``sample_treatment``, ``short_filename``, ``short_samplename``.
    Falls back to the raw filename for all labels when parsing fails.
    """
    import os

    basename = os.path.basename(filename)
    # Ensure the basename has an extension for the pattern to match
    if "." not in basename:
        basename_for_parse = basename + ".h5"
    else:
        basename_for_parse = basename

    try:
        fields = parse_file_name(basename_for_parse)
    except Exception:
        # Graceful fallback: return the raw filename for every label
        return {
            "group":            basename,
            "file":             basename,
            "short_groupname":  basename,
            "sample_treatment": basename,
            "short_filename":   basename,
            "short_samplename": basename,
        }

    date          = fields.get("date", "")
    owner         = fields.get("owner", "")
    pi            = fields.get("pi", "")
    project_id    = fields.get("project_id", "")
    proj_short    = fields.get("project_shortname", "")
    experiment    = fields.get("experiment", "")
    instrument    = fields.get("instrument", "")
    # Use the raw (pre-canonical) chromatography value from the filename for
    # header labels so that the group/short_filename strings match the original
    # filename exactly (e.g. "HILICZ" rather than the canonical "HILIC").
    # parse_file_name() already succeeded above, so FILE_PATTERN.match() is safe.
    _raw_match = FILE_PATTERN.match(basename_for_parse)
    chromatography_raw = _raw_match.group("chromatography") if _raw_match else fields.get("chromatography", "")
    chromatography = fields.get("chromatography", "")
    run_id        = fields.get("run_id", "")
    polarity      = fields.get("polarity", "")
    ms_level      = fields.get("ms_level", "")
    sample_number = fields.get("sample_number", "")
    sample_name   = fields.get("sample_name", "")
    replicate     = fields.get("replicate", "")
    run_metadata  = fields.get("run_metadata", "")
    run_number    = fields.get("run_number", "")
    ext           = fields.get("ext", "h5")

    # Classify file type if not provided
    if file_type is None:
        try:
            from metatlas2.lcmsruns_tools import _classify_file_type
            file_type = _classify_file_type(fields)
        except Exception:
            file_type = "experimental"

    ft_label = _file_type_label(file_type, sample_name)

    # ── 1. group ──────────────────────────────────────────────────────────
    # Format: date_owner_pi_project_id_proj_short_experiment_instrument_
    #         chromatography(raw)_run_id_polarity_ms_level_rta_tga_ft_label
    # RTA/TGA are bare numbers (no prefix); analyst is omitted.
    rta_str = str(rt_alignment_number) if rt_alignment_number is not None else ""
    tga_str = str(analysis_number) if analysis_number is not None else ""
    group = "_".join(filter(None, [
        date, owner, pi, project_id, proj_short, experiment,
        instrument, chromatography_raw, run_id, polarity, ms_level,
        rta_str, tga_str, ft_label,
    ]))

    # ── 2. file ───────────────────────────────────────────────────────────
    file_label = basename if "." in basename else f"{basename}.{ext}"

    # ── 3. short_groupname ────────────────────────────────────────────────
    short_groupname = f"{polarity}_{ft_label}" if polarity else ft_label

    # ── 4. sample_treatment ───────────────────────────────────────────────
    sample_treatment = sample_name

    # ── 5. short_filename ─────────────────────────────────────────────────
    # Keep: date, pi, proj_short, experiment, chromatography(raw), polarity, run_metadata
    # Remove: owner, project_id, instrument, run_id, ms_level,
    #         sample_number, sample_name, replicate, run_number, ext
    short_filename = "_".join(filter(None, [
        date, pi, proj_short, experiment, chromatography_raw, polarity, run_metadata,
    ]))

    # ── 6. short_samplename ───────────────────────────────────────────────
    # Keep: polarity, sample_name, replicate, run_metadata
    # Remove: date, owner, pi, project_id, proj_short, experiment,
    #         instrument, chromatography, run_id, ms_level,
    #         sample_number, run_number, ext
    short_samplename = "_".join(filter(None, [
        polarity, sample_name, replicate, run_metadata,
    ]))

    return {
        "group":            group,
        "file":             file_label,
        "short_groupname":  short_groupname,
        "sample_treatment": sample_treatment,
        "short_filename":   short_filename,
        "short_samplename": short_samplename,
    }