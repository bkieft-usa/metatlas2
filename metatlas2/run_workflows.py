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
import analysis_gui as agu
import analysis_summary as asm
import notebook_generator as nbg

from workflow_objects import Compound, Atlas, Project, RTAlign, AutoIdentification, AnalysisGUI, AnalysisSummary

logger = lcf.get_logger('run_workflows')

def run_project_setup(
    project_name: str,
    config_path: str,
    overwrite_existing: bool = False
) -> None:
    """
    Creates project database and loads LCMS run files.
    """

    project_obj = Project()

    project_obj.setup(
        project_name=project_name,
        config_path=config_path,
        overwrite_existing=overwrite_existing
    )

def run_rt_alignment(
    config_path: str,
    project_name: str,
    rt_alignment_number: int,
) -> None:
    """Run fresh RT alignment using an RT alignment atlas"""
    
    rt_align_obj = RTAlign()

    rt_align_obj.setup(
        config_path=config_path,
        project_name=project_name,
        rt_alignment_number=rt_alignment_number
    )

    if self.rt_alignment_params.get('do_alignment', True) is False:
        logger.warning(f"RT alignment is disabled in config. Exiting.")
        return
    
    logger.info(f"Checking for existing RT model in database with UID {rt_align_obj.pre_align_atlas_uid} ({rt_align_obj.chromatography}) and RT Alignment number {rt_align_obj.rt_alignment_number}")
    rt_align_obj.check_existing_rt_alignment()

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=rt_align_obj.paths['project_db_path'],
    )

    logger.info(f"Filtering {len(project_lcmsruns)} LCMS runs to run alignment against...")
    rt_align_obj.aligner_lcmsruns = lrt.filter_lcmsruns_list(
        lcmsruns=project_lcmsruns,
        include_file_type=rt_align_obj.rt_alignment_params.get('include_lcmsruns', ["QC"]),
        exclude_file_type=rt_align_obj.rt_alignment_params.get('exclude_lcmsruns', ["NEG"]),
        chromatography=rt_align_obj.chromatography
    )

    if rt_align_obj.rt_alignment_model is None:
        
        rt_align_obj.pre_align_atlas_obj = Atlas.from_database(
            database_path=rt_align_obj.paths['main_db_path'],
            atlas_uid=rt_align_obj.pre_align_atlas_uid
        )

        logger.info("Passing Atlas and LCMSRuns to data extractor...")
        experimental_data_obj = edp.extract_eic_and_ms2_from_parquet(
            obj=rt_align_obj,
            stage="rt_alignment"
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
    for _, aligned_atlas_obj in rt_align_obj.rt_aligned_atlases.items():
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
    config_path: str,
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_subset: list = None
) -> Dict[str, int]:
    """
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """

    auto_id_obj = AutoIdentification()

    auto_id_obj.setup(
        config_path=config_path,
        project_name=project_name,
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number,
        analysis_subset=analysis_subset
    )

    logger.info(f"Checking for existing Auto Identification results within RT Alignment number {auto_id_obj.rt_alignment_number} and analysis number {auto_id_obj.analysis_number}...")
    dbi.check_existing_auto_identification(auto_id_obj)

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=auto_id_obj.paths['project_db_path'],
    )

    logger.info(f"Retrieving all atlases from database for auto identification...")
    pre_autoid_atlas_uids = dbi.get_all_atlases_for_autoid(
        auto_id_obj=auto_id_obj,
        use_config_atlases=False
    )

    for pre_autoid_atlas_uid in pre_autoid_atlas_uids:

        auto_id_obj.pre_autoid_atlas_obj = Atlas.from_database(
            database_path=auto_id_obj.paths['project_db_path'],
            atlas_uid=pre_autoid_atlas_uid,
            main_db_path=auto_id_obj.paths['main_db_path']
        )

        logger.info(f"Loading workflow parameters for targeted analysis from config...")
        auto_id_obj.workflow_params = auto_id_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][auto_id_obj.pre_autoid_atlas_obj.chromatography][auto_id_obj.pre_autoid_atlas_obj.polarity][auto_id_obj.pre_autoid_atlas_obj.analysis_type]['PARAMS']

        logger.info("Finding LCMSRuns matching criteria for auto identification...")
        auto_id_obj.autoid_lcmsruns = lrt.filter_lcmsruns_list(
            lcmsruns=project_lcmsruns,
            include_file_type=auto_id_obj.workflow_params.get('include_lcmsruns', []),
            exclude_file_type=auto_id_obj.workflow_params['exclude_lcmsruns'].get('data_extraction', []),
            chromatography=auto_id_obj.pre_autoid_atlas_obj.chromatography,
            polarity=auto_id_obj.pre_autoid_atlas_obj.polarity
        )

        logger.info("Passing Atlas and LCMSRuns to data extractor...")
        auto_id_obj.experimental_data = pdx.extract_eic_and_ms2_from_parquet(
            obj=auto_id_obj,
            stage="auto_identification",
        )

        logger.info("Passing ExperimentalData to MS2 hit finder...")
        mhd.find_ms2_hits(
            auto_id_obj=auto_id_obj
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
        auto_id_obj.post_autoid_atlas_obj = dbi.create_new_atlas_after_auto_id(
            auto_id_obj=auto_id_obj
        )

        logger.info("Passing Atlas object to curation notebook generator...")
        nbg.generate_gui_notebooks(
            auto_id_obj=auto_id_obj
        )

def run_analysis_gui(
    config_path: str,
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_atlas: str = None,
    override_parameters: Dict[str, Any] = None,
    dash_app_port: int = 8050
) -> "CurationApp":
    """
    Runs the analysis GUI for interactive exploration of results.
    Requires RT alignment and auto identification to have been completed.
    """

    analysis_gui_obj = AnalysisGUI()

    analysis_gui_obj.setup(
        config_path=config_path,
        project_name=project_name, 
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number,
    )

    analysis_gui_obj.pre_curation_atlas_obj = Atlas.from_database(
        database_path=analysis_gui_obj.paths['project_db_path'],
        atlas_uid=analysis_atlas,
        main_db_path=analysis_gui_obj.paths['main_db_path']
    )

    logger.info("Loading workflow parameters for analysis GUI from config...")
    analysis_gui_obj.workflow_params = analysis_gui_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][analysis_gui_obj.pre_curation_atlas_obj.chromatography][analysis_gui_obj.pre_curation_atlas_obj.polarity][analysis_gui_obj.pre_curation_atlas_obj.analysis_type]['PARAMS']

    logger.info("Launching Analysis GUI...")
    dash_app = agu.build_dash_app(
        analysis_gui_obj=analysis_gui_obj,
        override_parameters=override_parameters,
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
    config_path: str,
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
        config_path=config_path,
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
    )

    summary_obj.pre_curation_atlas_obj = Atlas.from_database(
        database_path=summary_obj.paths['project_db_path'],
        atlas_uid=analysis_atlas,
        main_db_path=summary_obj.paths['main_db_path']
    )

    logger.info(f"Loading workflow parameters for analysis summary from config...")
    summary_obj.workflow_params = summary_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][summary_obj.pre_curation_atlas_obj.chromatography][summary_obj.pre_curation_atlas_obj.polarity][summary_obj.pre_curation_atlas_obj.analysis_type]['PARAMS']
    
    logger.info("Passing AnalysisSummary object to new Atlas generator...")
    summary_obj.post_curation_atlas_obj = dbi.create_new_atlas_after_manual_curation(
        summary_obj=summary_obj
    )

    logger.info("Creating and saving summary files and figures to output directory...")
    asm.run_all_summaries(
        summary_obj=summary_obj,
        overwrite=overwrite,
    )