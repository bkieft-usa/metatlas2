from typing import Dict, Any
import logging
import threading, time, os
from IPython.display import display, HTML

from werkzeug.serving import make_server

import metatlas2.database_interact as dbi
import metatlas2.rt_align_tools as rat
import metatlas2.ms2_hit_detection as mhd
import metatlas2.manual_curation_summarizer as mcs
import metatlas2.extract_data_from_parquet as edp
import metatlas2.lcmsruns_tools as lrt
import metatlas2.analysis_gui as agu
import metatlas2.analysis_summary as asm
import metatlas2.notebook_generator as nbg
import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
logger = lcf.get_logger('workflows')

def run_project_setup(
    project_name: str,
    config: Dict[str, Any],
    paths: Dict[str, str],
    overwrite_existing: bool = False,
) -> None:
    """
    Creates project database and loads LCMS run files.
    """

    from metatlas2.workflow_objects import Project

    project_obj = Project()

    project_obj.setup(
        project_name=project_name,
        config=config,
        paths=paths,
        overwrite_existing=overwrite_existing,
    )

def run_rt_alignment(
    project_name: str,
    rt_alignment_number: int,
    config: Dict[str, Any],
    paths: Dict[str, str],
) -> None:
    """Run fresh RT alignment using an RT alignment atlas"""

    from metatlas2.workflow_objects import RTAlign, Atlas

    if not os.path.exists(paths["project_db_path"]):
        raise FileNotFoundError(
            f"Project database not found: {paths['project_db_path']}. "
            "Please run project setup first."
        )

    rt_align_obj = RTAlign()

    rt_align_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        config=config,
        paths=paths,
    )

    if rt_align_obj.run_alignment is False:
        logger.info(f"RT alignment is disabled or aligned atlases already exist for RT alignment number {rt_align_obj.rt_alignment_number}. Skipping RT alignment procedure and exiting.")
        return

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=rt_align_obj.paths['project_db_path'],
    )

    logger.info(f"Filtering {len(project_lcmsruns)} LCMS runs to run alignment against...")
    rt_align_obj.aligner_lcmsruns = lrt.filter_lcmsruns_list(
        lcmsruns=project_lcmsruns,
        include_file_type=rt_align_obj.rt_alignment_params.get('include_lcmsruns', ["QC"]),
        exclude_file_type=rt_align_obj.rt_alignment_params.get('exclude_lcmsruns', ["NEG"]),
        chromatography=rt_align_obj.chromatography,
        ms_level=1
    )
    
    logger.info("Retrieving template Atlas from database...")
    rt_align_obj.align_atlas_obj = Atlas.from_database(
        database_path=rt_align_obj.paths['main_db_path'],
        atlas_uid=rt_align_obj.align_atlas_uid
    )

    logger.info("Passing Atlas and LCMSRuns to data extractor...")
    experimental_data_obj = edp.extract_eic_and_ms2_from_parquet(
        obj=rt_align_obj,
        stage="rt_alignment"
    )

    logger.info("Passing ExperimentalData and Atlas to summarizer...")
    rat.create_file_matching_summary(
        experimental_data=experimental_data_obj,
        atlas=rt_align_obj.align_atlas_obj
    )

    logger.info("Passing ExperimentalData, Atlas, and RTAlign to RT alignment model builder...")
    rat.build_rt_alignment_model(
        experimental_data=experimental_data_obj,
        atlas=rt_align_obj.align_atlas_obj, 
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
    for uid, aligned_atlas_obj in rt_align_obj.rt_aligned_atlases.items():
        dbi.save_atlas_to_database(
            atlas_obj=aligned_atlas_obj,
            db_path=rt_align_obj.paths['project_db_path'],
            main_db_path=rt_align_obj.paths['main_db_path']
        )
        ldt.save_atlas_data_to_csv(
            atlas_obj=aligned_atlas_obj,
            output_path=rt_align_obj.paths['aligned_atlases_store_file']
        )

    logger.info("Passing RTAlign object to RT alignment summary generator...")
    rat.display_rt_alignment_summary(
        rt_align_obj=rt_align_obj
    )

    logger.info(f"RT alignment procedure complete for RT alignment number {rt_align_obj.rt_alignment_number} and chromatography {rt_align_obj.chromatography}!")

def run_auto_identification(
    project_name: str,
    config: Dict[str, Any],
    paths: Dict[str, str],
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_subset: list = None,
) -> Dict[str, int]:
    """
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """

    from metatlas2.workflow_objects import Atlas, AutoIdentification

    if not os.path.exists(paths["project_db_path"]):
        raise FileNotFoundError(
            f"Project database not found: {paths['project_db_path']}. "
            "Please run project setup first."
        )
    if not os.path.exists(paths["msms_refs_path"]):
        raise FileNotFoundError(
            f"MS/MS reference file not found: {paths['msms_refs_path']}. "
            "Please ensure the path is correct in the config file."
        )

    auto_id_obj = AutoIdentification()

    auto_id_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number,
        config=config,
        paths=paths,
        analysis_subset=analysis_subset,
    )

    logger.info(f"Checking for existing Auto Identification results within RT Alignment number {auto_id_obj.rt_alignment_number} and analysis number {auto_id_obj.analysis_number}...")
    dbi.check_existing_auto_identification(auto_id_obj)

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=auto_id_obj.paths['project_db_path'],
    )

    logger.info(f"Retrieving atlases from CSV file {auto_id_obj.paths['aligned_atlases_store_file']} for auto identification...")
    pre_autoid_atlases = ldt.load_atlas_data_from_csv(
        file_path=auto_id_obj.paths['aligned_atlases_store_file']
    )

    for _, atlas_to_autoid in pre_autoid_atlases.iterrows():
        if auto_id_obj.analysis_subset:
            analysis_filters = [tuple(subset.split('-', 1)) for subset in auto_id_obj.analysis_subset]
            if (atlas_to_autoid['polarity'], atlas_to_autoid['analysis_type']) not in analysis_filters:
                logger.info(f"Skipping auto ID for atlas {atlas_to_autoid['atlas_uid']} with polarity {atlas_to_autoid['polarity']} and analysis type {atlas_to_autoid['analysis_type']} since it's not in the specified analysis subset: {auto_id_obj.analysis_subset}")
                continue

        auto_id_obj.pre_autoid_atlas_obj = Atlas.from_database(
            database_path=auto_id_obj.paths['project_db_path'],
            atlas_uid=atlas_to_autoid['atlas_uid'],
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
        auto_id_obj.experimental_data = edp.extract_eic_and_ms2_from_parquet(
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

        logger.info("Saving post-autoid Atlas data to CSV...")
        ldt.save_atlas_data_to_csv(
            atlas_obj=auto_id_obj.post_autoid_atlas_obj,
            output_path=auto_id_obj.paths['auto_ided_atlases_store_file']
        )

        logger.info("Passing Atlas object to curation notebook generator...")
        nbg.generate_gui_notebooks(
            auto_id_obj=auto_id_obj
        )

    logger.info(f"Auto identification procedure complete for RT alignment number {auto_id_obj.rt_alignment_number} and analysis number {auto_id_obj.analysis_number}!")

def run_analysis_gui(
    project_name: str,
    config: Dict[str, Any],
    paths: Dict[str, str],
    rt_alignment_number: int = None,
    analysis_number: int = None,
    pre_curation_atlas: str = None,
    override_parameters: Dict[str, Any] = None,
    dash_app_port: int = 8050,
) -> "CurationApp":
    """
    Runs the analysis GUI for interactive exploration of results.
    Requires RT alignment and auto identification to have been completed.
    """

    from metatlas2.workflow_objects import Atlas, AnalysisGUI

    if not os.path.exists(paths["project_db_path"]):
        raise FileNotFoundError(
            f"Project database not found: {paths['project_db_path']}. "
            "Please run project setup first."
        )

    analysis_gui_obj = AnalysisGUI()

    analysis_gui_obj.setup(
        project_name=project_name, 
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number,
        config=config,
        paths=paths,
    )

    analysis_gui_obj.pre_curation_atlas_obj = Atlas.from_database(
        database_path=analysis_gui_obj.paths['project_db_path'],
        atlas_uid=pre_curation_atlas,
        main_db_path=analysis_gui_obj.paths['main_db_path']
    )

    logger.info("Loading workflow parameters for analysis GUI from config...")
    analysis_gui_obj.workflow_params = analysis_gui_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][analysis_gui_obj.pre_curation_atlas_obj.chromatography][analysis_gui_obj.pre_curation_atlas_obj.polarity][analysis_gui_obj.pre_curation_atlas_obj.analysis_type]['PARAMS']

    logger.info("Loading and filtering GUI inputs...")
    dbi.load_and_filter_gui_inputs(
        analysis_gui_obj=analysis_gui_obj,
        override_parameters=override_parameters
    )

    logger.info("Launching Analysis GUI...")
    shutdown_holder = [None]

    dash_app = agu.build_dash_app(
        analysis_gui_obj=analysis_gui_obj,
        port=dash_app_port,
        shutdown_holder=shutdown_holder,
    )

    server = make_server("0.0.0.0", dash_app_port, dash_app.server)
    shutdown_holder[0] = server.shutdown

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(3)

    service_prefix = os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')
    url = f"{service_prefix}proxy/{dash_app_port}/"
    
    return display(HTML(f'<a href="{url}" target="_blank">▶ Open Dash App ↗</a>'))

def run_analysis_summary(
    project_name: str,
    rt_alignment_number: int,
    analysis_number: int,
    config: Dict[str, Any],
    paths: Dict[str, str],
    pre_curation_atlas: str = None,
    overwrite: bool = False,
) -> None:
    """
    Run all summary outputs for a completed analysis.

    Creates and configures an :class:`AnalysisSummary` object, loads all
    analysis data tables from the project database exactly once, then
    produces all summary files in order:
    """

    from metatlas2.workflow_objects import Atlas, AnalysisSummary

    summary_obj = AnalysisSummary()
    
    summary_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
        config=config,
        paths=paths,
    )

    summary_obj.pre_curation_atlas_obj = Atlas.from_database(
        database_path=summary_obj.paths['project_db_path'],
        atlas_uid=pre_curation_atlas,
        main_db_path=summary_obj.paths['main_db_path']
    )

    logger.info(f"Loading workflow parameters for analysis summary from config...")
    summary_obj.workflow_params = summary_obj.config['WORKFLOWS']['TARGETED_ANALYSES'][summary_obj.pre_curation_atlas_obj.chromatography][summary_obj.pre_curation_atlas_obj.polarity][summary_obj.pre_curation_atlas_obj.analysis_type]['PARAMS']
    
    logger.info("Passing AnalysisSummary object to new Atlas generator...")
    summary_obj.post_curation_atlas_obj = dbi.create_new_atlas_after_manual_curation(
        summary_obj=summary_obj
    )

    logger.info("Saving post-manual-curation Atlas data to CSV...")
    ldt.save_atlas_data_to_csv(
        atlas_obj=summary_obj.post_curation_atlas_obj,
        output_path=summary_obj.paths['curated_atlases_store_file']
    )

    logger.info("Creating and saving summary files and figures to output directory...")
    asm.run_all_summaries(
        summary_obj=summary_obj,
        overwrite=overwrite,
    )

def run_atlas_finder(
    project_db_path: str,
    atlas_uid: str = None,
    atlas_name: str = None,
    analysis_type: str = None,
    chromatography: str = None,
    polarity: str = None,
    atlas_type: str = None,
    created_by: str = None,
    created_date: str = None,
    source: str = None
):
    """
    Run the atlas finder utility to query the database for atlases matching user-specified criteria.
    """
    
    df = dbi.find_atlases_in_database(
        database_path=project_db_path,
        atlas_uid=atlas_uid,
        atlas_name=atlas_name,
        analysis_type=analysis_type,
        chromatography=chromatography,
        polarity=polarity,
        atlas_type=atlas_type,
        created_by=created_by,
        created_date=created_date,
        source=source
    )

    return df