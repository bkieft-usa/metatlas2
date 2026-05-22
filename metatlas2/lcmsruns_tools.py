from pathlib import Path
from typing import List

import metatlas2.logging_config as lcf
logger = lcf.get_logger('lcmsruns_tools')

def filter_lcmsruns_list(
    lcmsruns: List["LCMSRun"], 
    include_file_type: List[str] = None,
    exclude_file_type: List[str] = None,
    file_format: str = "h5",
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
            polarity = ["positive", "fps"]
        elif polarity.lower() in ["neg", "negative"]:
            polarity = ["negative", "fps"]

    if ms_level is not None:
        ms_level = int(ms_level)

    logger.info(f"Filtering {len(lcmsruns)} LCMS runs with parameters: include_file_type={include_file_type}, exclude_file_type={exclude_file_type}, file_format={file_format}, chromatography={chromatography}, polarity={polarity}, ms_level={ms_level}...")
    logger.info(f"Input files have the following unique types:")
    logger.info(f"  - file_type: {set(getattr(run, 'file_type', None) for run in lcmsruns)}")
    logger.info(f"  - file_format: {set(getattr(run, 'file_format', None) for run in lcmsruns)}")
    logger.info(f"  - chromatography: {set(getattr(run, 'chromatography', None) for run in lcmsruns)}")
    logger.info(f"  - polarity: {set(getattr(run, 'polarity', None) for run in lcmsruns)}")
    logger.info(f"  - ms_level: {set(getattr(run, 'ms_level', None) for run in lcmsruns)}")
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
        filtered_runs = [run for run in filtered_runs if getattr(run, 'polarity', '').lower() in [p.lower() for p in polarity]]
    if ms_level is not None:
        filtered_runs = [run for run in filtered_runs if getattr(run, 'ms_level', None) == ms_level]

    logger.info(f"Filtered to {len(filtered_runs)} out of {len(lcmsruns)} total files.")
    if len(filtered_runs) == 0:
        logger.warning("No LCMS runs matched the filter criteria. Please check your parameters.")
        raise ValueError("No LCMS runs matched the filter criteria. Please check your parameters.")

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
        project_raw_files_path: Path to directory containing raw files
        
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
    
    # Find files directly in the project directory
    raw_files = list(project_path.glob("*.raw"))
    mzML_files = list(project_path.glob("*.mzML"))
    h5_files = list(project_path.glob("*.h5"))
    if not raw_files or not mzML_files or not h5_files:
        raise ValueError(f"Missing .raw, .mzML, or .h5 files for {project_path}")

    logger.info(f"Found {len(raw_files)} .raw files in {project_path}")
    logger.info(f"Found {len(mzML_files)} .mzML files in {project_path}")
    logger.info(f"Found {len(h5_files)} .h5 files in {project_path}")

    # Initialize nested dictionary
    files_by_group = {'raw': {}, 'mzML': {}, 'h5': {}}

    # Organize each file type
    _organize_files(raw_files, 'raw', files_by_group['raw'])
    _organize_files(mzML_files, 'mzML', files_by_group['mzML'])
    _organize_files(h5_files, 'h5', files_by_group['h5'])

    return files_by_group

def _organize_files(files: List[Path], file_type: str, files_dict: dict) -> None:
    """
    Helper function to organize files by chromatography/polarity/ms_level/analysis_type.
    
    Args:
        files: List of file paths to organize
        file_type: Type of file ('raw', 'mzML', 'h5')
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
        
        if any(x in filename.upper() for x in ['_POS_', '_POSITIVE_']):
            polarity = 'positive'
        elif any(x in filename.upper() for x in ['_NEG_', '_NEGATIVE_']):
            polarity = 'negative'
        elif any(x in filename.upper() for x in ['_FPS_']):
            polarity = 'fps'
        else:
            polarity = 'Unknown'
        if any(x in filename.upper() for x in ['_MS1_']):
            ms_level = 1
        elif any(x in filename.upper() for x in ['_MS2_', '_MSMS_']):
            ms_level = 2
        else:
            ms_level = 'Unknown'

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