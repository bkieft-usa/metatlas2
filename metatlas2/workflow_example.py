"""
Example script showing how to use the new workflow-focused class organization.
This replaces the complex nested approach with clear workflow stages.
"""

import sys
from pathlib import Path
sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')

import workflow_objects as wfo
import logging_config as lcf

logger = lcf.get_logger('workflow_example')

def run_targeted_metabolomics_project(config: dict, project_db_path: str, project_directory: str) -> None:
    """
    Example of running the complete targeted metabolomics workflow using the new class organization.
    
    Args:
        config: Configuration dictionary
        project_db_path: Path to project DuckDB database
        project_directory: Path to project directory containing H5 files
    """
    
    main_db_path = config["paths"]["main_database"]
    
    # Initialize workflow orchestrator
    workflow = wfo.TargetedMetabolomicsWorkflow(
        config=config,
        project_db_path=project_db_path,
        project_directory=project_directory,
        main_db_path=main_db_path
    )
    
    logger.info("Starting targeted metabolomics workflow...")
    
    # Run workflow up to manual curation (returns GUI)
    gui = workflow.run_complete_workflow(stop_at_stage=wfo.WorkflowStage.MANUAL_CURATION)
    
    # Display status
    status = workflow.get_workflow_status()
    logger.info(f"Workflow status: {status}")
    
    # The GUI is returned for manual curation
    if gui:
        logger.info("Manual curation GUI is ready. Complete your annotations and then run:")
        logger.info("final_report = workflow.continue_to_final_report()")
        return gui
    
    return workflow

def run_workflow_by_atlas_type(config: dict, project_db_path: str, project_directory: str, 
                              atlas_type: str = 'qc') -> None:
    """
    Example of running workflow for a specific atlas type only.
    
    Args:
        config: Configuration dictionary
        project_db_path: Path to project DuckDB database
        project_directory: Path to project directory
        atlas_type: Type of atlas to process ('qc', 'istd', 'ema')
    """
    
    main_db_path = config["paths"]["main_database"]
    
    # Initialize and run through putative identification
    workflow = wfo.TargetedMetabolomicsWorkflow(
        config=config,
        project_db_path=project_db_path,
        project_directory=project_directory,
        main_db_path=main_db_path
    )
    
    # Run up to putative identification
    workflow.run_complete_workflow(stop_at_stage=wfo.WorkflowStage.PUTATIVE_IDENTIFICATION)
    
    # Create GUI for specific atlas type
    curation_manager = wfo.ManualCurationManager(workflow.putative_identification)
    gui = curation_manager.create_curation_gui(config, atlas_type=atlas_type)
    
    logger.info(f"Created curation GUI for {atlas_type} atlas type")
    return gui

def inspect_putative_identifications(project_db_path: str, main_db_path: str) -> None:
    """
    Example of inspecting putative identifications without running the full workflow.
    """
    
    # Load existing putative identifications from database
    import database_interact as dbi
    
    # Get all targeted analysis results
    with dbi.get_db_connection(project_db_path) as conn:
        results = conn.execute("""
            SELECT DISTINCT atlas_type, chromatography_polarity, 
                   COUNT(*) as compound_count,
                   SUM(CASE WHEN curation_status = 'finalized' THEN 1 ELSE 0 END) as finalized_count
            FROM targeted_analysis 
            GROUP BY atlas_type, chromatography_polarity
            ORDER BY atlas_type, chromatography_polarity
        """).fetchall()
    
    logger.info("Putative identification summary:")
    for row in results:
        atlas_type, chrom_pol, total, finalized = row
        logger.info(f"  {atlas_type} - {chrom_pol}: {total} compounds ({finalized} finalized)")

def generate_atlas_type_reports(config: dict, project_db_path: str, output_dir: str) -> None:
    """
    Example of generating separate reports for each atlas type.
    """
    
    main_db_path = config["paths"]["main_database"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load existing workflow state
    # Note: In practice, you'd reconstruct this from database or saved state
    workflow = wfo.TargetedMetabolomicsWorkflow(
        config=config,
        project_db_path=project_db_path,
        project_directory=str(output_path.parent),
        main_db_path=main_db_path
    )
    
    # Generate reports for each atlas type
    atlas_types = ['qc', 'istd', 'ema']
    
    for atlas_type in atlas_types:
        logger.info(f"Generating report for {atlas_type} atlas")
        
        # Create filtered report
        # This would need to be implemented in the FinalReportManager
        report_path = output_path / f"{atlas_type}_targeted_analysis_report"
        
        # Placeholder - would implement filtered report generation
        logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    # Example usage
    
    # Sample configuration
    config = {
        "paths": {
            "main_database": "/path/to/main_database.duckdb",
            "pubchem_cache": "/path/to/pubchem_cache.pkl"
        },
        "analysis_settings": {
            "default_ppm_error": 5.0,
            "extra_time": 1.0,
            "use_data_cache": True
        },
        "database_options": {
            "overwrite_existing_project_db": False
        }
    }
    
    project_db_path = "/path/to/project.duckdb"
    project_directory = "/path/to/project"
    
    # Run complete workflow
    workflow_result = run_targeted_metabolomics_project(
        config, 
        project_db_path, 
        project_directory
    )
    
    # Or run for specific atlas type
    qc_gui = run_workflow_by_atlas_type(
        config, 
        project_db_path, 
        project_directory, 
        atlas_type='qc'
    )
    
    # Inspect existing results
    inspect_putative_identifications(project_db_path, config["paths"]["main_database"])