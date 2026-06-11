from typing import Dict, Any
import logging
import socket
import threading, time, os
from IPython.display import display, HTML

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
logger = lcf.get_logger('workflows')


def _get_workflow_params(config: "Metatlas2Config", chromatography: str, polarity: str, analysis_type: str, analysis_name: str) -> Dict[str, Any]:
    """
    Look up PARAMS for a specific named analysis entry from the config.

    Delegates to :meth:`Metatlas2Config.get_targeted_analysis` and returns
    the ``params`` dict of the matching :class:`TargetedAnalysis`.
    """
    ta = config.get_targeted_analysis(chromatography, polarity, analysis_type, analysis_name)
    logger.info(f"Loaded workflow parameters for analysis with the following specs:")
    logger.info(f"  Chromatography: {ta.chromatography}")
    logger.info(f"  Polarity: {ta.polarity}")
    logger.info(f"  Analysis type: {ta.analysis_type}")
    logger.info(f"  Analysis name: {ta.analysis_name}")
    return ta.params


def _atlas_matches_subset(atlas_row, analysis_subset: list) -> bool:
    """
    Check if an atlas row matches any entry in the analysis_subset list.
    Subset entries must be in 'POL-TYPE-NAME' format (e.g. 'POS-EMA-default').
    """
    pol = atlas_row['polarity']
    atype = atlas_row['analysis_type']
    aname = atlas_row['analysis_name']
    for subset in analysis_subset:
        parts = subset.split('-', 2)
        if len(parts) == 3:
            s_pol, s_type, s_name = parts
            if pol == s_pol and atype == s_type and aname == s_name:
                return True
        else:
            raise ValueError(f"Invalid analysis_subset entry '{subset}'. Must be in 'POL-TYPE-NAME' format (e.g. 'POS-EMA-DEFAULT').")
    return False


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


def run_project_setup(
    project_name: str,
    config: Dict[str, Any],
    paths: Dict[str, str],
    overwrite_existing: bool = False,
    config_path: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None,
) -> None:
    """
    Creates project database and loads LCMS run files.
    When ``config_path``, ``rt_alignment_number``, and ``analysis_number`` are
    all provided, saves a single config snapshot to the project database keyed
    by RTA+TGA.  All subsequent workflow stages read the config from this row.
    """

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
    """Run fresh RT alignment using an RT alignment atlas.
    Config and paths are loaded automatically from the project database.
    """

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
    
    logger.info("Retrieving template Atlas from database...")
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
    dbi.create_new_atlases_after_rt_alignment(
        rt_align_obj=rt_align_obj
    )

    logger.info(f"RT alignment procedure complete for RT alignment number {rt_align_obj.rt_alignment_number} and chromatography {rt_align_obj.chromatography}!")

def run_auto_identification(
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_subset: list = None,
    image_tag: str = "latest",
) -> Dict[str, int]:
    """
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    Config and paths are loaded automatically from the project database.

    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """

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

    logger.info(f"Retrieving atlases from database for auto identification...")
    pre_autoid_atlases = dbi.get_atlases_for_stage(
        project_db_path=auto_id_obj.paths['project_db_path'],
        rt_alignment_number=auto_id_obj.rt_alignment_number,
        stage='RT_ALIGNED',
    )

    logger.info(f"Beginning Auto Identification for {len(pre_autoid_atlases)} RT-aligned atlases from RT alignment number {auto_id_obj.rt_alignment_number}.")
    for _, atlas_to_autoid in pre_autoid_atlases.iterrows():
        if auto_id_obj.analysis_subset and not _atlas_matches_subset(atlas_to_autoid, auto_id_obj.analysis_subset):
            logger.info(
                f"Skipping auto ID for atlas {atlas_to_autoid['atlas_uid']} "
                f"({atlas_to_autoid['polarity']}-{atlas_to_autoid['analysis_type']}-{atlas_to_autoid['analysis_name']}) "
                f"since it's not in the specified analysis subset: {auto_id_obj.analysis_subset}"
            )
            continue

        auto_id_obj.pre_autoid_atlas_obj = Atlas.from_database(
            database_path=auto_id_obj.paths['project_db_path'],
            atlas_uid=atlas_to_autoid['atlas_uid'],
            main_db_path=auto_id_obj.paths['main_db_path']
        )

        logger.info(f"Loading workflow parameters for targeted analysis from config...")
        auto_id_obj.workflow_params = _get_workflow_params(
            auto_id_obj.config,
            auto_id_obj.pre_autoid_atlas_obj.chromatography,
            auto_id_obj.pre_autoid_atlas_obj.polarity,
            auto_id_obj.pre_autoid_atlas_obj.analysis_type,
            auto_id_obj.pre_autoid_atlas_obj.analysis_name,
        )

        logger.info("Finding LCMSRuns matching criteria for auto identification...")
        auto_id_obj.autoid_lcmsruns = lrt.filter_lcmsruns_list(
            lcmsruns=project_lcmsruns,
            include_file_type=auto_id_obj.workflow_params.get('include_lcmsruns', []),
            exclude_file_type=auto_id_obj.workflow_params['exclude_lcmsruns'].get('data_extraction', []),
            chromatography=auto_id_obj.pre_autoid_atlas_obj.chromatography,
            polarity=auto_id_obj.pre_autoid_atlas_obj.polarity
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
        dbi.create_new_atlas_after_auto_id(
            auto_id_obj=auto_id_obj
        )

        logger.info("Passing Atlas object to curation notebook generator...")
        nbg.generate_gui_notebooks(
            auto_id_obj=auto_id_obj
        )

    logger.info(f"Auto identification procedure complete for RT alignment number {auto_id_obj.rt_alignment_number} and analysis number {auto_id_obj.analysis_number}!")

    return

def run_analysis_gui(
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    post_autoid_atlas: str = None,
    override_parameters: Dict[str, Any] = None,
    dash_app_port: int = 8050,
) -> "CurationApp":
    """
    Runs the analysis GUI for interactive exploration of results.
    Requires RT alignment and auto identification to have been completed.

    Config is loaded automatically from the project database
    """

    from metatlas2.workflow_objects import Atlas, AnalysisGUI

    analysis_gui_obj = AnalysisGUI()

    analysis_gui_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
        override_parameters=override_parameters,
    )

    analysis_gui_obj.get_note_options()

    analysis_gui_obj.post_autoid_atlas_obj = Atlas.from_database(
        database_path=analysis_gui_obj.paths['project_db_path'],
        atlas_uid=post_autoid_atlas,
        main_db_path=analysis_gui_obj.paths['main_db_path']
    )

    if analysis_gui_obj.override_parameters:
        logger.info("Persisting GUI override parameters to workflow_runs table...")
        dbi.update_config_overrides(
            project_db_path=analysis_gui_obj.paths['project_db_path'],
            rt_alignment_number=analysis_gui_obj.rt_alignment_number,
            analysis_number=analysis_gui_obj.analysis_number,
            atlas_uid=analysis_gui_obj.post_autoid_atlas_obj.atlas_uid,
            override_params=analysis_gui_obj.override_parameters,
            stage='AUTO_IDED',
        )

    logger.info("Loading workflow parameters for analysis GUI from config...")
    analysis_gui_obj.workflow_params = _get_workflow_params(
        analysis_gui_obj.config,
        analysis_gui_obj.post_autoid_atlas_obj.chromatography,
        analysis_gui_obj.post_autoid_atlas_obj.polarity,
        analysis_gui_obj.post_autoid_atlas_obj.analysis_type,
        analysis_gui_obj.post_autoid_atlas_obj.analysis_name,
    )

    logger.info("Loading and filtering GUI inputs...")
    dbi.load_and_filter_for_gui(
        analysis_gui_obj=analysis_gui_obj,
    )

    logger.info("Launching Analysis GUI...")
    shutdown_holder = [None]

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
    project_name: str,
    rt_alignment_number: int,
    analysis_number: int,
    post_autoid_atlas: str = None,
    override_parameters: Dict[str, Any] = None,
    overwrite: bool = False,
) -> None:
    """
    Run all summary outputs for a completed analysis.

    Creates and configures an :class:`AnalysisSummary` object, loads all
    analysis data tables from the project database exactly once, then
    produces all summary files in order.

    Config is loaded automatically from the project database
    """

    from metatlas2.workflow_objects import Atlas, AnalysisSummary

    summary_obj = AnalysisSummary()
    
    summary_obj.setup(
        project_name=project_name,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
    )

    summary_obj.post_autoid_atlas_obj = Atlas.from_database(
        database_path=summary_obj.paths['project_db_path'],
        atlas_uid=post_autoid_atlas,
        main_db_path=summary_obj.paths['main_db_path']
    )

    logger.info(f"Loading workflow parameters for analysis summary from config...")
    summary_obj.workflow_params = _get_workflow_params(
        summary_obj.config,
        summary_obj.post_autoid_atlas_obj.chromatography,
        summary_obj.post_autoid_atlas_obj.polarity,
        summary_obj.post_autoid_atlas_obj.analysis_type,
        summary_obj.post_autoid_atlas_obj.analysis_name,
    )
    
    logger.info(f"Setting override parameters for analysis summary...")
    summary_obj.override_parameters = override_parameters if override_parameters is not None else {}

    logger.info("Passing AnalysisSummary object to new Atlas generator...")
    dbi.create_new_atlas_after_manual_curation(
        summary_obj=summary_obj
    )

    logger.info("Saving post-manual-curation Atlas to database...")
    dbi.save_curated_atlaes_to_db(
        summary_obj=summary_obj
    )

    logger.info("Loading analysis data scoped to post-manual-curation atlas...")
    dbi.load_and_filter_for_summary(
        summary_obj=summary_obj,
        update_raw_in_feature=False
    )

    logger.info("Saving post-manual-curation Atlas data to TSV...")
    ldt.save_atlas_data_to_tsv(
        atlas_obj=summary_obj.post_curation_atlas_obj,
        output_path=summary_obj.paths['analysis_output_dir']
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