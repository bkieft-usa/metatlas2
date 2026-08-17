import re

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

def normalize_chromatography(chrom: str) -> str:
    """Return the canonical lowercase chromatography label.

    All other values are simply lowercased.
    """

    _CHROM_NORMALIZATION_MAP = {
        "hilicz": "hilic",
        "c18-ep": "c18",
        "c18ep":  "c18",
    }

    if not chrom:
        return chrom
    return _CHROM_NORMALIZATION_MAP.get(chrom.lower(), chrom.lower())

def parse_file_name(filename: str):
    match = FILE_PATTERN.match(filename)
    if not match:
        raise Exception(f"Filename '{filename}' does not match the expected format.")
    fields = match.groupdict()
    fields["chromatography"] = normalize_chromatography(fields.get("chromatography", ""))
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
    """Return the normalized chromatography field from a project name."""
    match = PROJECT_PATTERN.match(project_name)
    if not match:
        raise ValueError(f"Project name '{project_name}' does not match the expected format.")
    return normalize_chromatography(match.group("chromatography"))

def get_file_parts(name: str, part: str):
    """Return the named capture group *part* from a file stem or project name.

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
            return normalize_chromatography(value)
        return value
    except ValueError:
        raise
    except Exception:
        raise ValueError(f"File name '{name}' does not match the expected format.")