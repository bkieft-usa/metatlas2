import os
import nbformat

import metatlas2.logging_config as lcf
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
        "chromatography": auto_id_obj.auto_ided_atlas_obj.chromatography,
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
        _make_parameters_cell(auto_id_obj, run_params),
        _make_gui_cell(),
        _make_summary_cell(),
    ]

    logger.info("Generating notebook file...")
    fname = (
        f"{run_params['project_name']}"
        f"_RTA{run_params['rt_alignment_number']}"
        f"_TGA{run_params['analysis_number']}"
        f"_{run_params['chromatography']}"
        f"_{run_params['polarity']}"
        f"_{run_params['analysis_type']}"
        f"_{run_params['analysis_name']}"
        f".ipynb"
    )
    out_path = os.path.join(auto_id_obj.paths['analysis_output_dir'], fname)
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info(f"Notebook written to {out_path}")

    return out_path


def _make_parameters_cell(auto_id_obj: "AutoIdentification", run_params: dict) -> nbformat.NotebookNode:
    params = auto_id_obj.ta.params
    param_keys = [
        "ms1_min_peak_intensity",
        "ms1_min_num_points",
        "ms2_min_score",
        "ms2_min_matching_frags",
        "gui_lcmsruns_colors",
        "gui_require_all_evaluated",
        "gui_top_n_hits",
        "note_options_overrides",
        "remove_unided_compounds",
        "remove_flagged_compounds",
        "apply_istd_curation_to_ema",
        "apply_cross_polarity_curation",
        "upload_to_gdrive"
    ]
    src = "# Optionally override default parameters for manual curation\n"
    src += "OVERRIDE_PARAMS = {\n"
    for key in param_keys:
        current_value = params.get(key, None)
        if key == "note_options_overrides" and current_value == {}:
            current_value = f"{auto_id_obj.owner} defaults"
        src += f"    '{key}': None, # current value: {repr(current_value)}\n"
    src += "}\n\n"
    src += "# Run-specific metrics\n"
    src += "RUN_PARAMS = {\n"
    for key in run_params:
        if isinstance(run_params[key], str):
            src += f"    '{key}': '{run_params[key]}',\n"
        else:
            src += f"    '{key}': {run_params[key]},\n"
    src += "}"
    return nbformat.v4.new_code_cell(src)


def _make_header_cell(run_params: dict) -> nbformat.NotebookNode:
    text = (
        f"# **`{run_params['project_name']}`**  \n"
        f"**Input Atlas UID:** {run_params['input_atlas_uid']}  \n"
        f"**RT alignment number:** {run_params['rt_alignment_number']}  \n"
        f"**Analysis number:** {run_params['analysis_number']}\n"
        f"**Chromatography:** {run_params['chromatography']}  \n"
        f"**Polarity:** {run_params['polarity']}  \n"
        f"**Analysis type:** {run_params['analysis_type']}  \n"
        f"**Analysis name:** {run_params['analysis_name']}  \n"
    )
    return nbformat.v4.new_markdown_cell(text)


def _make_imports_cell() -> nbformat.NotebookNode:
    src = (
        "import logging\n"
        "import pandas as pd\n"
        "pd.options.display.max_colwidth = 600\n\n"
        "import metatlas2.workflows as wfs\n\n"
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
