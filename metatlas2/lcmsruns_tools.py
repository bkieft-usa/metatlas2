import glob
import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple

def get_project_files(project_path: str, lcmsrun_dir: str) -> dict:
    """
    Retrieve h5 files from a specified raw data path, grouped by chromatography, polarity, and analysis type.

    Returns:
        dict: {chrom: {pol: {'experimental': [...], 'qc': [...], 'injbl': [...], 'exctrl': [...], 'istd': [...]}}}
    """
    all_files = glob.glob(os.path.join(project_path, lcmsrun_dir, "*.h5"))
    files_by_group = {}

    abnormal_filenames = 0
    for file in all_files:
        base = os.path.basename(file)
        print(base)
        parts = base.split('_')

        if len(parts) != 16:
            abnormal_filenames += 1
            continue  # skip abnormal filenames

        chrom = parts[7]
        pol = parts[9]
        analysis = 'experimental'
        analysis_part = parts[14].lower()

        if "-istd" in analysis_part:
            analysis = 'istd'
        elif "-injbl" in analysis_part:
            analysis = 'injbl'
        elif "-exctrl" in analysis_part:
            analysis = 'exctrl'
        elif "-qc" in analysis_part:
            analysis = 'qc'

        if chrom not in files_by_group:
            files_by_group[chrom] = {}
        if pol not in files_by_group[chrom]:
            files_by_group[chrom][pol] = {'experimental': [], 'qc': [], 'injbl': [], 'exctrl': [], 'istd': []}

        files_by_group[chrom][pol][analysis].append(file)

    if abnormal_filenames > 0:
        print(f"Warning: {abnormal_filenames} files have abnormal filenames and were skipped.")

    return files_by_group

def read_hdf_file(filename,desired_key=None):
    """
    Inputs:
    filename: hdf filename from which to extract desired key
    desired_key: optional key, typically "ms1_pos", "ms2_neg", etc.

    Outputs:
    df_container: a dataframe holding the information for the desired key (e.g., m/z, rt, intensity)
    """
    if desired_key is not None:
        return pd.read_hdf(filename,desired_key)