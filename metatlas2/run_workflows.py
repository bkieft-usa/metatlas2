from typing import Dict, Any
import sys

import threading, time, os
from IPython.display import display, HTML

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import ms2_hit_detection as mhd
import extract_data_from_parquet as pdx
import manual_curation_summarizer as mcs
import extract_data_from_parquet as edp
import lcmsruns_tools as lrt
import analysis_gui_dash as agd
import analysis_summary as asm
from workflow_objects import NewCompound, NewAtlas, Atlas, Project, RTAlign, AutoIdentification, AnalysisGUI, AnalysisSummary

logger = lcf.get_logger('workflow_objects')

def add_compounds_to_db(
    config: Dict[str, Any],
    overwrite_db: bool = False,
) -> None:
    """
    Creates main database (if needed) and loads compounds from config file paths.
    """
    new_compound_obj = NewCompound(
        config=config, 
        overwrite_db=overwrite_db
    )
    
    new_compound_obj.run()

def add_atlases_to_db(
    config: Dict[str, Any]
) -> None:
    """
    Creates atlases from config file paths and saves them to the database.
    """
    new_atlas_obj = NewAtlas(
        config=config
    )
    
    new_atlas_obj.run()

def run_project_setup(
    project_name: str,
    config: Dict,
    overwrite_existing: bool = False
) -> None:
    """
    Creates project database and loads LCMS run files.
    """

    project_obj = Project()

    project_obj.setup(
        project_name=project_name,
        config=config,
        overwrite_existing=overwrite_existing
    )

def run_rt_alignment(
    config: dict,
    project_name: str,
    rt_alignment_number: int,
    chromatography: str
) -> None:
    """Run fresh RT alignment using an RT alignment atlas"""
    
    rt_align_obj = RTAlign()

    rt_align_obj.setup(
        config=config,
        project_name=project_name,
        chromatography=chromatography,
        rt_alignment_number=rt_alignment_number
    )

    logger.info(f"Checking for existing RT model in database with UID {rt_align_obj.qc_atlas_uid} ({rt_align_obj.chromatography}) and RT Alignment number {rt_align_obj.rt_alignment_number}")
    rt_align_obj.check_existing_rt_alignment()

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=rt_align_obj.paths['project_db_path'],
    )

    logger.info(f"Filtering {len(project_lcmsruns)} LCMS runs...")
    rt_align_obj.aligner_lcmsruns = lrt.filter_lcmsruns_list(
        lcmsruns=project_lcmsruns,
        file_type=['qc'],
        chromatography=rt_align_obj.chromatography
    )

    if rt_align_obj.rt_alignment_model is None:
        
        template_atlas_obj = Atlas.from_database(
            database_path=rt_align_obj.paths['main_db_path'],
            atlas_uid=rt_align_obj.qc_atlas_uid
        )

        logger.info("Passing Atlas and LCMSRuns to data extractor...")
        experimental_data_obj = edp.extract_eic_and_ms2_from_parquet(
            atlas=template_atlas_obj,
            stage="rt_alignment",
            lcmsruns=rt_align_obj.aligner_lcmsruns,
            workflow_params=rt_align_obj.rt_alignment_params,
            only_ms_level=1
        )

        logger.info("Passing ExperimentalData and Atlas to summarizer...")
        rat.create_file_matching_summary(
            experimental_data=experimental_data_obj,
            atlas=template_atlas_obj
        )

        logger.info("Passing ExperimentalData, Atlas, and RTAlign to RT alignment model builder...")
        rat.build_rt_alignment_model(
            experimental_data=experimental_data_obj,
            atlas=template_atlas_obj, 
            rt_align=rt_align_obj
        )

        logger.info("Passing RTAlign object to model database table saver...")
        dbi.save_rt_alignment_model_to_db(
            rt_align_obj=rt_align_obj
        )
        
        logger.info("Passing RTAlign object to model visualizer...")
        rat.visualize_rt_alignment_model(
            rt_align_obj=rt_align_obj
        )
    
    logger.info("Passing RTAlign object to alignment applicator...")
    rat.apply_rt_alignment_to_target_atlases(
        rt_align_obj=rt_align_obj
    )

    logger.info("Passing aligned Atlases to database saver...")
    for aligned_atlas_uid, aligned_atlas_obj in rt_align_obj.rt_aligned_atlases.items():
        dbi.save_atlas_to_database(
            atlas_obj=aligned_atlas_obj, 
            db_path=rt_align_obj.paths['project_db_path'],
            main_db_path=rt_align_obj.paths['main_db_path']
        )

    logger.info("Passing RTAlign object to RT alignment summary generator...")
    rat.display_rt_alignment_summary(
        rt_align_obj=rt_align_obj
    )

    logger.info(f"RT alignment procedure complete for RT alignment number {rt_align_obj.rt_alignment_number} and chromatography {rt_align_obj.chromatography}!")


def run_auto_identification(
    config: dict,
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_atlas: str = None
) -> Dict[str, int]:
    """
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """
    
    auto_id_obj = AutoIdentification()

    auto_id_obj.setup(
        config=config, 
        project_name=project_name,
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number, 
        analysis_atlas_uid=analysis_atlas
    )

    logger.info(f"Checking for existing Auto Identification results within RT Alignment number {auto_id_obj.rt_alignment_number} and analysis number {auto_id_obj.analysis_number}...")
    dbi.check_existing_auto_identification(auto_id_obj)

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=auto_id_obj.paths['project_db_path'],
    )

    auto_id_obj.pre_autoid_atlas_obj = Atlas.from_database(
        database_path=auto_id_obj.paths['project_db_path'],
        atlas_uid=auto_id_obj.analysis_atlas_uid,
        main_db_path=auto_id_obj.paths['main_db_path']
    )

    logger.info(f"Loading workflow parameters for targeted analysis from config...")
    auto_id_obj.workflow_params = auto_id_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][auto_id_obj.pre_autoid_atlas_obj.chromatography][auto_id_obj.pre_autoid_atlas_obj.polarity][auto_id_obj.pre_autoid_atlas_obj.analysis_type]['PARAMS']

    logger.info("Finding LCMSRuns matching criteria for auto-identification...")
    auto_id_obj.autoid_lcmsruns = lrt.filter_lcmsruns_list(
        lcmsruns=project_lcmsruns,
        file_type=['experimental', 'istd', 'exctrl', 'refstd'],
        chromatography=auto_id_obj.pre_autoid_atlas_obj.chromatography,
        polarity=auto_id_obj.pre_autoid_atlas_obj.polarity
    )

    logger.info("Passing Atlas and LCMSRuns to data extractor...")
    auto_id_obj.experimental_data = pdx.extract_eic_and_ms2_from_parquet(
        atlas=auto_id_obj.pre_autoid_atlas_obj,
        stage="auto_identification",
        lcmsruns=auto_id_obj.autoid_lcmsruns,
        workflow_params=auto_id_obj.workflow_params
    )

    logger.info("Passing ExperimentalData to MS2 hit finder...")
    mhd.find_ms2_hits(
        auto_id_obj=auto_id_obj,
        msms_refs_path=auto_id_obj.paths['msms_refs_path']
    )

    logger.info("Passing ExperimentalData and Atlas to ManualCuration creator...")
    mcs.create_manual_curation_obj(
        auto_id_obj=auto_id_obj
    )

    logger.info("Passing finalized AutoIdentification object to database saver...")
    dbi.save_auto_identification_results_to_db(
        auto_id_obj=auto_id_obj
    )

    logger.info("Passing AutoIdentification object to summary generator...")
    dbi.display_auto_id_summary(
        auto_id_obj=auto_id_obj
    )

    logger.info("Passing AutoIdentification object to new Atlas generator...")
    dbi.create_new_atlas_after_auto_id(
        auto_id_obj=auto_id_obj,
        remove_unidentified_compounds=True
    )

def run_analysis_gui(
    config: dict,
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    remove_unidentified_compounds: bool = True,
    dash_app_port: int = 8050
) -> "CurationApp":
    """
    Runs the analysis GUI for interactive exploration of results.
    Requires RT alignment and auto identification to have been completed.
    """

    analysis_gui_obj = AnalysisGUI()

    analysis_gui_obj.setup(
        config=config, 
        project_name=project_name, 
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number
    )

    logger.info("Launching Analysis GUI...")
    dash_app = agd.build_dash_app(
        analysis_gui_obj, 
        remove_unidentified_compounds=remove_unidentified_compounds,
        port=dash_app_port,
    )

    def _run():
        dash_app.run(
            host="0.0.0.0",
            port=dash_app_port,
            debug=False,
            use_reloader=False,
            dev_tools_hot_reload=False,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(3)

    service_prefix = os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')
    url = f"{service_prefix}proxy/{dash_app_port}/"
    
    return display(HTML(f'<a href="{url}" target="_blank">▶ Open Dash App ↗</a>'))

def run_analysis_summary(
    config: dict,
    project_name: str,
    rt_alignment_number: int,
    analysis_number: int,
    analysis_atlas: str = None,
    overwrite: bool = False,
) -> None:
    """
    Run all summary outputs for a completed analysis.

    Creates and configures an :class:`AnalysisSummary` object, loads all
    analysis data tables from the project database exactly once, then
    produces all summary files in order:
    """

    summary_obj = AnalysisSummary()
    
    summary_obj.setup(
        config=config,
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number
    )

    summary_obj.pre_curation_atlas_obj = Atlas.from_database(
        database_path=summary_obj.paths['project_db_path'],
        atlas_uid=analysis_atlas,
        main_db_path=summary_obj.paths['main_db_path']
    )
    
    dbi.create_new_atlas_after_manual_curation(
        summary_obj=summary_obj,
        remove_flagged_compounds=True
    )

    asm.run_all_summaries(
        summary_obj=summary_obj,
        overwrite=overwrite,
    )