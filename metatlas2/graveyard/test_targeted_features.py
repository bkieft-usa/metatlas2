"""
Test script for examining isomer detection and RT bounds suggestion features independently.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import targeted_analysis as tga
import database_interact as dbi
import data_classes as dcl
import logging_config as lcf

logger = lcf.get_logger('test_targeted_features')

def test_isomer_detection_workflow(project_db_path: str, atlas_uid: str, config: Dict[str, Any]):
    """Test isomer detection independently."""
    logger.info("=== TESTING ISOMER DETECTION ===")
    
    # Load atlas
    atlas_df = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path,
        main_db_path=config["paths"]["main_database"],
        atlas_uid=atlas_uid
    )
    
    logger.info(f"Loaded atlas with {len(atlas_df)} compounds")
    
    # Test all compounds
    logger.info("\n--- Testing all compounds ---")
    all_results = tga.test_isomer_detection(atlas_df)
    
    # Test specific compound (if you want to examine one in detail)
    # Pick a compound that likely has isomers
    compounds_with_isomers = [(k, v) for k, v in all_results.items() if v]
    if compounds_with_isomers:
        test_inchi_key = compounds_with_isomers[0][0]  # Pick first one
        logger.info(f"\n--- Testing specific compound: {test_inchi_key} ---")
        specific_result = tga.test_isomer_detection(atlas_df, test_inchi_key)
    
    return all_results

def test_rt_bounds_workflow(project_db_path: str, atlas_uid: str, config: Dict[str, Any]):
    """Test RT bounds suggestion independently."""
    logger.info("\n=== TESTING RT BOUNDS SUGGESTION ===")
    
    # Need to run the data extraction first to get EIC data
    logger.info("Running minimal data extraction for RT bounds testing...")
    
    # Load atlas and initialize project
    atlas_df = dbi.get_atlas_compounds_with_metadata(
        project_db_path=project_db_path,
        main_db_path=config["paths"]["main_database"],
        atlas_uid=atlas_uid
    )
    
    project_analysis = dcl.ProjectAnalysis(
        project_db_path=project_db_path,
        atlas_uid=atlas_uid
    )
    project_analysis.load_from_atlas(atlas_df)
    
    # Get experimental files and extract data (minimal subset for testing)
    project_files = dbi.get_experimental_files_from_db(project_db_path)
    
    # Limit to first few files for testing
    test_files = project_files[:3]  # Just use first 3 files for speed
    logger.info(f"Using {len(test_files)} files for RT bounds testing")
    
    # Extract data
    import ms1_ms2_analysis as msa
    
    input_data_list = msa.prepare_feature_tools_inputs(
        atlas_df=atlas_df,
        h5_files=test_files,
        ppm_tolerance=config["analysis_settings"]["default_ppm_error"],
        extra_time=config["analysis_settings"]["extra_time"]
    )
    
    experimental_data = msa.extract_eic_and_ms2_data(input_data_list, atlas_df, config)
    project_analysis.add_experimental_data_simple(experimental_data)
    
    # Test all compounds with EIC data
    logger.info("\n--- Testing all compounds with EIC data ---")
    all_results = tga.test_rt_bounds_suggestion(project_analysis)
    
    # Test specific compound
    compounds_with_eic = [k for k, v in all_results.items() if v['suggested']]
    if compounds_with_eic:
        test_inchi_key = compounds_with_eic[0]  # Pick first one
        logger.info(f"\n--- Testing specific compound: {test_inchi_key} ---")
        specific_result = tga.test_rt_bounds_suggestion(project_analysis, test_inchi_key)
    
    return all_results

def compare_original_vs_suggested_rt(rt_results: Dict[str, Dict]):
    """Compare original vs suggested RT bounds and show statistics."""
    logger.info("\n=== RT BOUNDS COMPARISON ANALYSIS ===")
    
    if not rt_results:
        logger.info("No RT results to analyze")
        return
    
    rt_shifts = []
    width_changes = []
    confidences = []
    
    for inchi_key, result in rt_results.items():
        if result['suggested']:
            original = result['original']
            suggested = result['suggested']
            
            rt_shift = suggested['rt_peak'] - original['rt_peak']
            width_change = (suggested['rt_max'] - suggested['rt_min']) - (original['rt_max'] - original['rt_min'])
            confidence = suggested['confidence']
            
            rt_shifts.append(rt_shift)
            width_changes.append(width_change)
            confidences.append(confidence)
    
    if rt_shifts:
        import numpy as np
        
        logger.info(f"RT Shift Statistics (n={len(rt_shifts)}):")
        logger.info(f"  Mean: {np.mean(rt_shifts):+.3f} ± {np.std(rt_shifts):.3f} min")
        logger.info(f"  Median: {np.median(rt_shifts):+.3f} min")
        logger.info(f"  Range: {np.min(rt_shifts):+.3f} to {np.max(rt_shifts):+.3f} min")
        
        logger.info(f"Width Change Statistics:")
        logger.info(f"  Mean: {np.mean(width_changes):+.3f} ± {np.std(width_changes):.3f} min")
        logger.info(f"  Median: {np.median(width_changes):+.3f} min")
        
        logger.info(f"Confidence Statistics:")
        logger.info(f"  Mean: {np.mean(confidences):.3f} ± {np.std(confidences):.3f}")
        logger.info(f"  Median: {np.median(confidences):.3f}")
        
        # Show compounds with largest shifts
        shifts_with_names = [(abs(shift), inchi_key, rt_results[inchi_key]['compound_name'], shift) 
                           for shift, inchi_key in zip(rt_shifts, rt_results.keys()) 
                           if rt_results[inchi_key]['suggested']]
        shifts_with_names.sort(reverse=True)
        
        logger.info(f"Largest RT shifts:")
        for i, (abs_shift, inchi_key, name, shift) in enumerate(shifts_with_names[:5]):
            logger.info(f"  {i+1}. {name}: {shift:+.3f} min")

def run_full_feature_test(project_db_path: str, atlas_uid: str, config: Dict[str, Any]):
    """Run complete test of both features."""
    logger.info("Starting comprehensive feature testing...")
    
    # Test isomer detection
    isomer_results = test_isomer_detection_workflow(project_db_path, atlas_uid, config)
    
    # Test RT bounds suggestion
    rt_results = test_rt_bounds_workflow(project_db_path, atlas_uid, config)
    
    # Analysis and comparison
    compare_original_vs_suggested_rt(rt_results)
    
    logger.info("\n=== SUMMARY ===")
    logger.info(f"Isomer detection: {sum(1 for v in isomer_results.values() if v)} compounds with isomers")
    logger.info(f"RT bounds suggestion: {sum(1 for v in rt_results.values() if v['suggested'])} compounds with suggestions")
    
    return {
        'isomers': isomer_results,
        'rt_bounds': rt_results
    }

# Example usage:
if __name__ == "__main__":
    # Set up your paths and config
    project_db_path = "/path/to/your/project.duckdb"
    atlas_uid = "your-atlas-uid"
    config = {
        "paths": {
            "main_database": "/path/to/main.duckdb",
            "msms_refs": "/path/to/msms_refs.tsv"
        },
        "analysis_settings": {
            "default_ppm_error": 20,
            "extra_time": 0.1
        }
    }
    
    # Run tests
    results = run_full_feature_test(project_db_path, atlas_uid, config)
