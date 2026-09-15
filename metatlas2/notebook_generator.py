import os
import nbformat

import metatlas2.logging_config as lcf
import metatlas2.file_and_project_format as fpf
logger = lcf.get_logger('notebook_generator')

def generate_gui_notebooks(
    auto_id_obj: "AutoIdentification",
) -> str:
    """Build a complete notebook for one analysis."""

    if auto_id_obj.ta.params.get("create_curation_notebooks", False) is not True:
        logger.info("create_curation_notebooks is not True, skipping notebook generation.")
        return

    logger.info(f"Generating analysis GUI notebook for atlas {auto_id_obj.auto_ided_atlas_obj.atlas_uid}")

    image_tag = getattr(auto_id_obj, "image_tag", "latest")
    if image_tag == "latest":
        kernel_name = "metatlas2"
        kernel_display_name = "metatlas2 (latest)"
    else:
        kernel_name = f"metatlas2-{image_tag}"
        kernel_display_name = f"metatlas2 ({image_tag})"

    run_params = {
        "project_name": auto_id_obj.project_name,
        "rt_alignment_number": auto_id_obj.rt_alignment_number,
        "analysis_number": auto_id_obj.analysis_number,
        "chromatography": fpf.normalize_chromatography(auto_id_obj.auto_ided_atlas_obj.chromatography),
        "polarity": auto_id_obj.auto_ided_atlas_obj.polarity,
        "analysis_type": auto_id_obj.auto_ided_atlas_obj.analysis_type,
        "analysis_name": auto_id_obj.auto_ided_atlas_obj.analysis_name,
        "input_atlas_uid": auto_id_obj.auto_ided_atlas_obj.atlas_uid,
    }

    logger.info("Building notebook cells...")
    nb = nbformat.v4.new_notebook()

    logger.info(f"Notebook kernel set to '{kernel_name}'")

    nb.metadata["kernelspec"] = {
        "display_name": kernel_display_name,
        "language": "python",
        "name": kernel_name,
    }
    nb.metadata["language_info"] = {"name": "python"}

    nb.cells = [
        _make_header_cell(run_params),
        _make_imports_cell(),
        _make_run_params_cell(run_params),
        _make_override_params_cell(auto_id_obj),
        _make_gui_cell(),
        _make_summary_cell(),
    ]

    logger.info("Generating notebook file...")
    fname = (
        f"{run_params['project_name']}"
        f"_RTA{run_params['rt_alignment_number']}"
        f"_TGA{run_params['analysis_number']}"
        f"_{run_params['chromatography'].upper()}"
        f"_{run_params['polarity'].upper()}"
        f"_{run_params['analysis_type'].upper()}"
        f"_{run_params['analysis_name'].upper()}"
        f".ipynb"
    )
    out_path = os.path.join(auto_id_obj.paths['analysis_output_dir'], fname)
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info(f"Notebook written to {out_path}")

    return out_path

def _make_run_params_cell(run_params: dict) -> nbformat.NotebookNode:
    """Cell 3: RUN_PARAMS — fixed identifiers for this analysis run."""
    src = "# Run-specific identifiers.\n"
    src += "RUN_PARAMS = {\n"
    for key in run_params:
        if isinstance(run_params[key], str):
            src += f"    '{key}': '{run_params[key]}',\n"
        else:
            src += f"    '{key}': {run_params[key]},\n"
    src += "}"
    return nbformat.v4.new_code_cell(src)


def _make_override_params_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    """Cell 4: OVERRIDE_PARAMS — optional per-run overrides for manual curation.

    Analysis params come from ta.params; GUI params come from config.gui_config.
    All values default to None (meaning: use the config value as-is).
    """
    ta_params = auto_id_obj.ta.params
    _gui_cfg = auto_id_obj.config.gui_config if auto_id_obj.config else {}

    analysis_param_keys = [
        "ms1_min_peak_intensity",
        "ms1_min_num_points",
        "ms2_min_score",
        "ms2_min_matching_frags",
        "remove_unided_compounds",
        "remove_flagged_compounds",
        "apply_istd_curation_to_ema",
        "apply_cross_polarity_curation",
        "upload_to_gdrive",
    ]

    gui_param_keys = [
        "gui_width",
        "gui_height",
        "gui_require_all_evaluated",
        "gui_top_n_hits",
        "gui_lcmsruns_colors",
        "note_options_overrides",
    ]

    src = "# Set a value to override the config default; leave as None to use the config value.\n"
    src += "OVERRIDE_PARAMS = {\n"
    for key in analysis_param_keys:
        current_value = ta_params.get(key, None)
        src += f"    '{key}': None,  # current value: {repr(current_value)}\n"
    for key in gui_param_keys:
        current_value = _gui_cfg.get(key, None)
        src += f"    '{key}': None,  # current value: {repr(current_value)}\n"
    src += "}"
    return nbformat.v4.new_code_cell(src)

def _make_header_cell(run_params: dict) -> nbformat.NotebookNode:
    text = (
        f"# **`{run_params['project_name']}`**  \n"
        #f"**Input Atlas UID:** {run_params['input_atlas_uid']}  \n"
        #f"**RT alignment number:** {run_params['rt_alignment_number']}  \n"
        #f"**Analysis number:** {run_params['analysis_number']}  \n"
        #f"**Chromatography:** {run_params['chromatography']}  \n"
        #f"**Polarity:** {run_params['polarity']}  \n"
        #f"**Analysis type:** {run_params['analysis_type']}  \n"
        #f"**Analysis name:** {run_params['analysis_name']}  \n"
    )
    return nbformat.v4.new_markdown_cell(text)

def _make_imports_cell() -> nbformat.NotebookNode:
    src = (
        "import logging\n"
        "import pandas as pd\n"
        "import metatlas2.workflows as wfs\n"
        "import metatlas2.logging_config as lcf\n"
        "lcf.setup_logging(log_level=logging.INFO)\n"
        "logger = lcf.get_logger('analysis_gui')"
    )
    return nbformat.v4.new_code_cell(src)

def _make_gui_cell() -> nbformat.NotebookNode:
    src = (
        "# Manual Curation\n"
        "wfs.run_analysis_gui(\n"
        f"    run_parameters=RUN_PARAMS,\n"
        f"    override_parameters=OVERRIDE_PARAMS,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)

def _make_summary_cell() -> nbformat.NotebookNode:
    src = (
        "# Analysis Summary\n"
        "wfs.run_analysis_summary(\n"
        f"    run_parameters=RUN_PARAMS,\n"
        f"    override_parameters=OVERRIDE_PARAMS,\n"
        f"    overwrite=False,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)
