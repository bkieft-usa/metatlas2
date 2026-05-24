from pathlib import Path
from typing import List, Any, Dict
from collections import Counter
import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf

logger = lcf.get_logger('lcmsruns_tools')

# Mapping for analysis type categorization
ANALYSIS_TYPE_MAP = {
    'qc': ['qc'],
    'istd': ['istd'],
    'exctrl': ['exctrl', 'txctrl'],
    'injbl': ['injbl', 'blank'],
    'refstd': ['refstd', 'standard'],
}

def get_project_lcmsruns_from_disk(project_raw_files_path: str) -> List[Dict]:
    project_path = Path(project_raw_files_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    if failed := list(project_path.glob("*.failed")):
        logger.error("\n".join([f"  - {f.name}" for f in failed]))
        raise ValueError(f"Please address {len(failed)} .failed files in {project_path}.")

    lcmsruns = []
    # Process all target extensions in one loop
    for ext in ['raw', 'mzML', 'h5']:
        files = list(project_path.glob(f"*.{ext}"))
        logger.info(f"Found {len(files)} .{ext} files")
        
        for file_path in files:
            try:
                fields = fpf.parse_file_name(file_path.name)
                s_name = fields.get("sample_name", "").lower()
                r_meta = fields.get("run_metadata", "").lower()
                combined = f"{s_name} {r_meta}"

                # Determine analysis type using the mapping
                analysis_type = 'experimental'
                for category, keywords in ANALYSIS_TYPE_MAP.items():
                    if any(k in combined for k in keywords):
                        analysis_type = category
                        break

                lcmsruns.append({
                    "file_path": str(file_path),
                    "filename": file_path.name,
                    "file_format": ext.lower(),
                    "file_type": analysis_type.lower(),
                    "chromatography": fields.get("chromatography", "Unknown").lower(),
                    "ms_level": fields.get("ms_level", "Unknown").lower(),
                    "polarity": fields.get("polarity", "Unknown").lower(),
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
    lcmsruns: List[Any], 
    include_file_type: List[str] = None,
    exclude_file_type: List[str] = None,
    file_format: str = "h5",
    chromatography: str = None,
    polarity: str = None,
    ms_level: str = None
) -> List[Any]:

    # Normalize and lowercase all filter inputs
    if chromatography:
        chromatography = chromatography.lower()
        # Accept both 'hilic' and 'hilicz' as equivalent if either is requested
        if chromatography in ["hilic", "hilicz"]:
            chromatography_set = {"hilic", "hilicz"}
        else:
            chromatography_set = {chromatography}
    else:
        chromatography_set = None

    file_format = file_format.lower() if file_format else None

    pol_set = set()
    if polarity:
        if isinstance(polarity, (list, set, tuple)):
            pol_set = {str(p).lower() for p in polarity}
        else:
            pol = str(polarity).lower()
            if pol in ["pos", "positive", "fps"]:
                pol_set = {"pos", "fps"}
            elif pol in ["neg", "negative"]:
                pol_set = {"neg"}
            else:
                pol_set = {pol}

    inc_set = {ft.lower() for ft in include_file_type} if include_file_type else None
    exc_set = {ft.lower() for ft in exclude_file_type} if exclude_file_type else None

    def match(run):
        # Support both dict and object (namedtuple/dataclass) access
        def get_val(k):
            v = run[k] if isinstance(run, dict) else getattr(run, k, "")
            return v.lower() if isinstance(v, str) else str(v).lower()

        if inc_set and get_val('file_type') not in inc_set:
            return False
        if exc_set and get_val('file_type') in exc_set:
            return False
        if file_format and get_val('file_format') != file_format:
            return False
        if chromatography_set and get_val('chromatography') not in chromatography_set:
            return False
        if pol_set and get_val('polarity') not in pol_set:
            return False
        if ms_level is not None and get_val('ms_level') != str(ms_level).lower():
            return False
        return True

    logger.info(f"Filtering {len(lcmsruns)} LCMS runs with criteria: "
                f"include_file_type={inc_set}, "
                f"exclude_file_type={exc_set}, "
                f"file_format={file_format}, "
                f"chromatography={chromatography}, "
                f"polarity={pol_set}, "
                f"ms_level={ms_level}")
    filtered = [run for run in lcmsruns if match(run)]

    logger.info(f"Filtered to {len(filtered)} out of {len(lcmsruns)} total files.")
    if not filtered:
        raise ValueError("No LCMS runs matched the filter criteria.")

    return filtered