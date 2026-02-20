import sys
from pathlib import Path
from typing import List

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('lcmsruns_tools')


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
    raw_dir = project_path / "raw"
    parquet_dir = project_path / "parquet"
    mzML_dir = project_path / "mzML"
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw/ directory does not exist in {project_path}")
    if not parquet_dir.exists():
        raise FileNotFoundError(f"parquet/ directory does not exist in {project_path}")
    if not mzML_dir.exists():
        raise FileNotFoundError(f"mzML/ directory does not exist in {project_path}")
    raw_files = list(raw_dir.glob("*.raw"))
    parquet_files = list(parquet_dir.glob("*.parquet"))
    mzML_files = list(mzML_dir.glob("*.mzML"))
    if not parquet_files or not raw_files or not mzML_files:
        raise ValueError(f"Missing .parquet, .raw, or .mzML files for {project_path}")
    
    logger.info(f"Found {len(parquet_files)} .parquet files in {project_path}")
    logger.info(f"Found {len(raw_files)} .raw files in {project_path}")
    logger.info(f"Found {len(mzML_files)} .mzML files in {project_path}")
    
    # Initialize nested dictionary
    files_by_group = {'raw': {}, 'parquet': {}, 'mzML': {}}
    
    # Organize each file type
    _organize_files(parquet_files, 'parquet', files_by_group['parquet'])
    _organize_files(raw_files, 'raw', files_by_group['raw'])
    _organize_files(mzML_files, 'mzML', files_by_group['mzML'])
    
    return files_by_group

def _organize_files(files: List[Path], file_type: str, files_dict: dict) -> None:
    """
    Helper function to organize files by chromatography/polarity/ms_level/analysis_type.
    
    Args:
        files: List of file paths to organize
        file_type: Type of file ('parquet', 'raw', 'mzML')
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
        if file_type == 'parquet':
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
        else:  # raw or mzML
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