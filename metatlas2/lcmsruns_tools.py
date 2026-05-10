from pathlib import Path
from typing import List

import metatlas2.logging_config as lcf
logger = lcf.get_logger('lcmsruns_tools')

def filter_lcmsruns_list(
    lcmsruns: List["LCMSRun"], 
    include_file_type: List[str] = None,
    exclude_file_type: List[str] = None,
    file_format: str = "parquet",
    chromatography: str = None,
    polarity: str = None,
    ms_level: int = None
) -> List["LCMSRun"]:
    """
    Filter a list of LCMSRun objects by file type, file format, chromatography, and polarity.
    """
    if chromatography:
        if chromatography == "HILICZ":
            chromatography = "HILIC"

    if polarity:
        if polarity.lower() in ["pos", "positive"]:
            polarity = "positive"
        elif polarity.lower() in ["neg", "negative"]:
            polarity = "negative"

    if ms_level is not None:
        ms_level = int(ms_level)

    logger.info(f"Filtering {len(lcmsruns)} LCMS runs with parameters: include_file_type={include_file_type}, exclude_file_type={exclude_file_type}, file_format={file_format}, chromatography={chromatography}, polarity={polarity}, ms_level={ms_level}...")
    filtered_runs = lcmsruns.copy()

    # Fix: ensure file_type is always a list
    if include_file_type:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'file_type', '').lower() in [ft.lower() for ft in include_file_type]]
    if exclude_file_type:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'file_type', '').lower() not in [ft.lower() for ft in exclude_file_type]]
    if file_format:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'file_format', '').lower() == file_format.lower()]
    if chromatography:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'chromatography', '').lower() == chromatography.lower()]
    if polarity:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'polarity', '').lower() == polarity.lower()]
    if ms_level is not None:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'ms_level', None) == ms_level]

    logger.info(f"Filtered to {len(filtered_runs)} out of {len(lcmsruns)} total files.")
    return filtered_runs

def get_project_lcmsruns_from_disk(project_raw_files_path: str) -> list:
    """
    Scan project directory for LCMS files and return a flat list of LCMS run metadata dicts.
    Each dict is suitable for direct insertion into the database or for LCMSRun(**row).
    """
    files_by_group = _get_project_files(project_raw_files_path)
    lcmsruns = []
    for file_format, chrom_dict in files_by_group.items():
        for chrom, ms_level_dict in chrom_dict.items():
            for ms_level, pol_dict in ms_level_dict.items():
                for pol, analysis_dict in pol_dict.items():
                    for file_type, file_list in analysis_dict.items():
                        for file_path in file_list:
                            lcmsruns.append({
                                "file_path": file_path,
                                "filename": Path(file_path).name,
                                "file_format": file_format,
                                "file_type": file_type,
                                "chromatography": chrom,
                                "ms_level": ms_level,
                                "polarity": pol,
                                "created_by": None,
                                "created_date": None,
                            })
    return lcmsruns

def _get_project_files(project_raw_files_path: str) -> dict:
    """
    Scan project directory for LCMS files and organize by chromatography/ms_level/polarity/analysis type.
    
    Args:
        project_raw_files_path: Path to directory containing .parquet files
        
    Returns:
        Nested dictionary: {chromatography: {ms_level: {polarity: {analysis_type: [file_paths]}}}}
    """
    project_path = Path(project_raw_files_path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    # Find all files
    failed_conversion_files = list(project_path.glob("*.failed"))
    if failed_conversion_files:
        for f in failed_conversion_files:
            logger.error(f"  - {f.name}")
        raise ValueError(f"Please address the .failed files before proceeding. Found {len(failed_conversion_files)} .failed files in {project_path}.")
    
    # Check on directories and files
    parquet_dir = project_path / "parquet"
    if not parquet_dir.exists():
        raise FileNotFoundError(f"parquet/ directory does not exist in {project_path}")
    parquet_files = list(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"Missing .parquet files for {project_path}")
    logger.info(f"Found {len(parquet_files)} .parquet files in {project_path}")
    
    files_by_group = {'parquet': {}}
    _organize_files(parquet_files, files_by_group['parquet'])
    
    return files_by_group

def _organize_files(files: List[Path], files_dict: dict) -> None:
    """
    Helper function to organize files by chromatography/polarity/ms_level/analysis_type.
    
    Args:
        files: List of file paths to organize
        files_dict: Dictionary to populate with organized files
    """
    for file_path in files:
        filename = file_path.name
        
        # Infer chromatography from filename
        if any(x in filename.upper() for x in ['HILIC', 'HILICZ']):
            chromatography = 'HILIC'
        elif any(x in filename.upper() for x in ['C18', 'RP']):
            chromatography = 'C18'
        else:
            chromatography = 'Unknown'
        
        # Infer MS level and polarity from filename
        filename_lower = filename.lower()
        if filename_lower.endswith('_ms1_pos.parquet'):
            ms_level, polarity = 1, 'positive'
        elif filename_lower.endswith('_ms2_pos.parquet'):
            ms_level, polarity = 2, 'positive'
        elif filename_lower.endswith('_ms1_neg.parquet'):
            ms_level, polarity = 1, 'negative'
        elif filename_lower.endswith('_ms2_neg.parquet'):
            ms_level, polarity = 2, 'negative'
        else:
            ms_level, polarity = 'unknown', 'unknown'

        # Infer analysis type from filename
        if any(x in filename.upper() for x in ['-QC']):
            analysis_type = 'qc'
        elif any(x in filename.upper() for x in ['-ISTD']):
            analysis_type = 'istd'
        elif any(x in filename.upper() for x in ['EXCTRL-', 'TXCTRL-']):
            analysis_type = 'exctrl'
        elif any(x in filename.upper() for x in ['-INJBL', 'BLANK']):
            analysis_type = 'injbl'
        elif any(x in filename.upper() for x in ['-REFSTD', '-STANDARD']):
            analysis_type = 'refstd'
        else:
            analysis_type = 'experimental'
        
        # Initialize nested structure
        if chromatography not in files_dict:
            files_dict[chromatography] = {}
        if ms_level not in files_dict[chromatography]:
            files_dict[chromatography][ms_level] = {}
        if polarity not in files_dict[chromatography][ms_level]:
            files_dict[chromatography][ms_level][polarity] = {}
        if analysis_type not in files_dict[chromatography][ms_level][polarity]:
            files_dict[chromatography][ms_level][polarity][analysis_type] = []
        
        # Add file to appropriate category
        files_dict[chromatography][ms_level][polarity][analysis_type].append(str(file_path))