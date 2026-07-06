from typing import Dict, Any
import logging
import socket
import threading, time, os
from IPython.display import display, HTML
from pathlib import Path

from werkzeug.serving import make_server

import metatlas2.database_interact as dbi
import metatlas2.rt_align_tools as rat
import metatlas2.ms2_hit_detection as mhd
import metatlas2.create_curation_container as ccc
import metatlas2.extract_data_from_h5 as edh
import metatlas2.lcmsruns_tools as lrt
import metatlas2.analysis_gui as agu
import metatlas2.analysis_summary as asm
import metatlas2.notebook_generator as nbg
import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
import metatlas2.gdrive_upload as gdu
logger = lcf.get_logger('workflows')


def run_project_setup(
    project_name: str,
    config: Dict[str, Any],
    paths: Dict[str, str],
    overwrite_existing: bool = False,
    config_path: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None,
) -> None:

    from metatlas2.workflow_objects import Project

    project_obj = Project()

    project_obj.setup(
        project_name=project_name,
        config=config,
        paths=paths,
        overwrite_existing=overwrite_existing,
        config_path=config_path,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
    )

def run_rt_alignment(
    project_name: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> None:

    from metatlas2.workflow_objects import RTAlign, Atlas

    rt_align_obj = RTAlign()

    rt_align_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
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
        ms_level="ms1"
    )
    
    logger.info("Retrieving QC Atlas from database...")
    rt_align_obj.align_atlas_obj = Atlas.from_database(
        database_path=rt_align_obj.paths['main_db_path'],
        atlas_uid=rt_align_obj.align_atlas_uid
    )
    
    logger.info("Passing Atlas and LCMSRuns to data extractor...")
    edh.extract_data_from_raw(
        obj=rt_align_obj,
    )

    logger.info("Passing ExperimentalData, Atlas, and RTAlign to RT alignment model builder...")
    rat.build_rt_alignment_model(
        rt_align_obj=rt_align_obj
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
    for ta in rt_align_obj.config.targeted_analyses:

        rt_align_obj.unaligned_atlas_obj = Atlas.from_database(
            rt_align_obj.paths['main_db_path'] , 
            ta.atlas_uid # the atlas from the config
        )

        logger.info(f"Cloning config Atlas {rt_align_obj.unaligned_atlas_obj.atlas_uid} for RT alignment stage...")
        rt_align_obj.aligned_atlas_obj = dbi.clone_atlas(
            obj=rt_align_obj,
            stage='RT_ALIGNED',
            ta=ta
        )

        logger.info("Applying RT alignment model to new Atlas compound_mzrts...")
        rt_shifted_data = rat.calculate_rt_shifts(
            rt_align_obj=rt_align_obj,
        )

        logger.info("Updating RT-aligned Atlas compound_mzrts in database...")
        dbi.update_compound_mzrt_for_atlas(
            obj=rt_align_obj,
            mz_rt_update_df=rt_shifted_data,
            stage='RT_ALIGNED',
        )

    if rt_align_obj.rt_alignment_params.get('upload_to_gdrive', False):
        logger.info("Uploading RT alignment results to Google Drive...")
        gdu.copy_outputs_to_google_drive(rt_align_obj, "RT_ALIGNMENT", overwrite=False)

def run_auto_identification(
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_subset: list = None,
    image_tag: str = "latest",
) -> Dict[str, int]:

    from metatlas2.workflow_objects import Atlas, AutoIdentification

    auto_id_obj = AutoIdentification()

    auto_id_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
        analysis_subset=analysis_subset,
        image_tag=image_tag,
    )

    logger.info(f"Retrieving all LCMS runs for project...")
    project_lcmsruns = dbi.get_lcmsruns_from_db(
        project_db_path=auto_id_obj.paths['project_db_path'],
    )

    for ta in auto_id_obj.config.targeted_analyses:
        
        auto_id_obj.ta = ta
        auto_id_obj.paths['analysis_results_output_dir'] = Path(auto_id_obj.paths["analysis_output_dir"]) / f"{auto_id_obj.ta.chromatography}-{auto_id_obj.ta.polarity}-{auto_id_obj.ta.analysis_type}-{auto_id_obj.ta.analysis_name}"
        os.makedirs(auto_id_obj.paths['analysis_results_output_dir'], exist_ok=True)

        autoid_atlas_info = dbi.get_atlas_uid_from_stage(
            obj=auto_id_obj,
            stage='RT_ALIGNED'
        )
        if not autoid_atlas_info: continue

        auto_id_obj.aligned_atlas_obj = Atlas.from_database(
            database_path=auto_id_obj.paths['project_db_path'],
            atlas_uid=autoid_atlas_info['atlas_uid'],
            main_db_path=auto_id_obj.paths['main_db_path']
        )

        logger.info(f"Cloning aligned Atlas {auto_id_obj.aligned_atlas_obj.atlas_uid} for auto identification stage...")
        auto_id_obj.auto_ided_atlas_obj = dbi.clone_atlas(
            obj=auto_id_obj,
            stage='AUTO_IDED',
            ta=ta
        )

        logger.info("Finding LCMSRuns matching criteria for auto identification...")
        auto_id_obj.autoid_lcmsruns = lrt.filter_lcmsruns_list(
            lcmsruns=project_lcmsruns,
            include_file_type=auto_id_obj.ta.params.get('include_lcmsruns', []),
            exclude_file_type=auto_id_obj.ta.params['exclude_lcmsruns'].get('data_extraction', []),
            chromatography=ta.chromatography,
            polarity=ta.polarity
        )

        logger.info("Passing Atlas and LCMSRuns to data extractor...")
        edh.extract_data_from_raw(
            obj=auto_id_obj,
        )

        logger.info("Passing ExperimentalData to MS2 hit finder...")
        mhd.find_ms2_hits(
            auto_id_obj=auto_id_obj
        )

        logger.info("Passing ExperimentalData and Atlas to ManualCuration creator...")
        ccc.create_manual_curation_obj(
            auto_id_obj=auto_id_obj
        )

        logger.info("Passing filtered AutoIdentification object to database saver...")
        dbi.save_auto_identification_results_to_db(
            auto_id_obj=auto_id_obj
        )

        logger.info("Creating post-auto-ID Atlas from filtered data...")
        dbi.update_compound_mzrt_for_atlas(
            obj=auto_id_obj,
            mz_rt_update_df=auto_id_obj.experimental_data.curation_df,
            stage='AUTO_IDED',
        )

        logger.info("Passing Atlas object to curation notebook generator...")
        nbg.generate_gui_notebooks(
            auto_id_obj=auto_id_obj
        )

    return

def run_analysis_gui(
    run_parameters: Dict[str, Any],
    override_parameters: Dict[str, Any] = None,
    dash_app_port: int = 8050,
) -> "CurationApp":

    from metatlas2.workflow_objects import Atlas, AnalysisGUI

    analysis_gui_obj = AnalysisGUI()

    analysis_gui_obj.setup(
        run_parameters=run_parameters,
        override_parameters=override_parameters,
    )

    logger.info("Looking up Atlas UID for manual curation stage from database based on config parameters...")
    curation_atlas_info = dbi.get_atlas_uid_from_stage(
        obj=analysis_gui_obj,
        stage='AUTO_IDED', # updated data during GUI only lives in curation_df for now
    )
    if curation_atlas_info['atlas_uid'] != run_parameters['input_atlas_uid']:
        logger.warning(f"Atlas UID for manual curation stage from database {curation_atlas_info['atlas_uid']} does not match input Atlas UID {run_parameters['input_atlas_uid']}.")
        logger.warning("Please verify that you intended to input a different Atlas UID in the notebook before proceeding.")
    
    logger.info(f"Retrieving Atlas UID {curation_atlas_info['atlas_uid']} for manual curation stage.")
    analysis_gui_obj.auto_ided_atlas_obj = Atlas.from_database(
        database_path=analysis_gui_obj.paths['project_db_path'],
        atlas_uid=curation_atlas_info['atlas_uid'],
        main_db_path=analysis_gui_obj.paths['main_db_path']
    )

    logger.info("Loading and filtering GUI inputs...")
    dbi.load_and_filter_for_gui(
        analysis_gui_obj=analysis_gui_obj
    )

    logger.info("Launching Analysis GUI...")
    shutdown_holder = [None]

    def _find_free_port(start: int, max_tries: int = 20) -> int:
        """Return the first free TCP port in [start, start+max_tries)."""
        for port in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(
            f"No free port found in the range {start}-{start + max_tries - 1}. "
            "Close another notebook or specify a different starting port."
        )
    dash_app_port = _find_free_port(dash_app_port)

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
    time.sleep(1)

    if os.getenv('METATLAS2_STANDALONE') == 'true':
        url = f"http://localhost:{dash_app_port}/"
    else:
        service_prefix = os.getenv('JUPYTERHUB_SERVICE_PREFIX', '/')
        url = f"{service_prefix}proxy/{dash_app_port}/"

    return display(HTML(f'<a href="{url}" target="_blank">▶ Open Dash App ↗</a>'))

def run_analysis_summary(
    run_parameters: Dict[str, Any],
    override_parameters: Dict[str, Any] = None,
    overwrite: bool = False,
) -> None:

    from metatlas2.workflow_objects import Atlas, AnalysisSummary

    summary_obj = AnalysisSummary()
    
    summary_obj.setup(
        run_parameters=run_parameters,
        override_parameters=override_parameters,
    )

    logger.info("Looking up Atlas UID for analysis summary stage from database based on config parameters...")
    summary_atlas_info = dbi.get_atlas_uid_from_stage(
        obj=summary_obj,
        stage='AUTO_IDED', # start with auto ided atlas again (like GUI), but this time clone and update based on mc GUI (curation_df)
    )
    if summary_atlas_info['atlas_uid'] != run_parameters['input_atlas_uid']:
        logger.warning(f"Atlas UID for summary stage from database {summary_atlas_info['atlas_uid']} does not match input Atlas UID {run_parameters['input_atlas_uid']}.")
        logger.warning("Please verify that you intended to input a different Atlas UID in the notebook before proceeding.")
    
    logger.info(f"Retrieving Atlas UID {summary_atlas_info['atlas_uid']} for analysis summary stage.")
    summary_obj.auto_ided_atlas_obj = Atlas.from_database(
        database_path=summary_obj.paths['project_db_path'],
        atlas_uid=summary_atlas_info['atlas_uid'],
        main_db_path=summary_obj.paths['main_db_path']
    )

    logger.info(f"Cloning aligned Atlas {summary_obj.auto_ided_atlas_obj.atlas_uid} for analysis summary stage...")
    summary_obj.manually_curated_atlas_obj = dbi.clone_atlas(
        obj=summary_obj,
        stage='MANUALLY_CURATED',
        ta=summary_obj.ta
    )

    logger.info("Loading analysis data scoped to atlas after manual curation...")
    dbi.load_and_filter_for_summary(
        summary_obj=summary_obj,
        update_raw_in_feature=False
    )

    if summary_obj.ta.params.get("gui_require_all_evaluated", True):
        logger.info("Checking that all compounds have been evaluated in the GUI before allowing summary generation...")
        dbi.check_require_evaluated(summary_obj)

    logger.info("Updating compound mz_rt values for the atlas based on the manual curation...")
    dbi.update_compound_mzrt_for_atlas(
        obj=summary_obj,
        mz_rt_update_df=summary_obj.experimental_data.curation_df,
        stage='MANUALLY_CURATED',
    )

    if summary_obj.override_parameters:
        logger.info("Persisting GUI override parameters to workflow_runs table...")
        dbi.update_config_overrides(
            obj=summary_obj,
            stage='MANUALLY_CURATED',
        )

    logger.info("Creating and saving summary files and figures to output directory...")
    asm.run_all_summaries(
        summary_obj=summary_obj,
        overwrite=overwrite,
    )

    logger.info("Changing group ownership of project output folder to the 'metatlas' group...")
    ldt.change_ownership_to_metatlas_group(
        project_dir_path=summary_obj.paths['project_directory']
    )

    logger.info(f"Analysis summary procedure complete for RTA{summary_obj.rt_alignment_number} and TGA{summary_obj.analysis_number}!")