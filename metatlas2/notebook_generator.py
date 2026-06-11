import os
import nbformat

import metatlas2.logging_config as lcf
logger = lcf.get_logger('notebook_generator')

def generate_gui_notebooks(
    auto_id_obj: "AutoIdentification",
) -> str:
    """Build a complete notebook for one analysis."""

    if auto_id_obj.workflow_params.get("create_curation_notebooks", False) is not True:
        logger.info("create_curation_notebooks is not True, skipping notebook generation.")
        return

    logger.info(f"Generating analysis GUI notebook for atlas {auto_id_obj.post_autoid_atlas_obj.atlas_uid}")

    image_tag = getattr(auto_id_obj, "image_tag", "latest")
    if image_tag == "latest":
        kernel_name = "metatlas2"
        kernel_display_name = "metatlas2 (latest)"
    else:
        kernel_name = f"metatlas2-{image_tag}"
        kernel_display_name = f"metatlas2 ({image_tag})"

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
        _make_header_cell(auto_id_obj),
        _make_imports_cell(),
        _make_variables_cell(auto_id_obj),
        _make_parameters_cell(auto_id_obj),
        _make_gui_cell(),
        _make_summary_cell(),
    ]

    logger.info("Generating notebook file...")
    fname = (
        f"{auto_id_obj.project_name}"
        f"_{auto_id_obj.post_autoid_atlas_obj.analysis_type}"
        f"-{getattr(auto_id_obj.post_autoid_atlas_obj, 'analysis_name', 'default') or 'default'}"
        f"_{auto_id_obj.post_autoid_atlas_obj.polarity}"
        f"_RTA{auto_id_obj.rt_alignment_number}"
        f"_TGA{auto_id_obj.analysis_number}"
        f".ipynb"
    )
    out_path = os.path.join(auto_id_obj.paths['analysis_output_dir'], fname)
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info(f"Notebook written to {out_path}")

    return out_path


def _make_parameters_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    params = auto_id_obj.workflow_params
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
    src = "# Parameters to override for GUI analysis\n"
    src += "OVERRIDE_PARAMS = {\n"
    for key in param_keys:
        current_value = params.get(key, None)
        src += f"    '{key}': None, # current value: {repr(current_value)}\n"
    src += "}"
    return nbformat.v4.new_code_cell(src)


def _make_header_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    text = (
        f"# **`{auto_id_obj.project_name}`**  \n"
        f"**Chromatography:** {auto_id_obj.post_autoid_atlas_obj.chromatography}  \n"
        f"**Polarity:** {auto_id_obj.post_autoid_atlas_obj.polarity}  \n"
        f"**Analysis type:** {auto_id_obj.post_autoid_atlas_obj.analysis_type}  \n"
        f"**Analysis name:** {getattr(auto_id_obj.post_autoid_atlas_obj, 'analysis_name', 'default') or 'default'}  \n"
        f"**RT alignment number:** {auto_id_obj.rt_alignment_number}  \n"
        f"**Analysis number:** {auto_id_obj.analysis_number}"
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


def _make_variables_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    src = (
        f"PROJECT_NAME     = {auto_id_obj.project_name!r}\n"
        f"RT_ALIGN_NUM     = {auto_id_obj.rt_alignment_number!r}\n"
        f"ANALYSIS_NUM     = {auto_id_obj.analysis_number!r}\n"
        f"ANALYSIS_ATLAS   = {auto_id_obj.post_autoid_atlas_obj.atlas_uid!r}\n"
        f"CHROMATOGRAPHY   = {auto_id_obj.post_autoid_atlas_obj.chromatography!r}"
    )
    return nbformat.v4.new_code_cell(src)


def _make_gui_cell() -> nbformat.NotebookNode:
    src = (
        "# Manual Curation\n"
        "wfs.run_analysis_gui(\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    post_autoid_atlas=ANALYSIS_ATLAS,\n"
        "    override_parameters=OVERRIDE_PARAMS,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)


def _make_summary_cell() -> nbformat.NotebookNode:
    src = (
        "# Analysis Summary\n"
        "wfs.run_analysis_summary(\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    post_autoid_atlas=ANALYSIS_ATLAS,\n"
        "    override_parameters=OVERRIDE_PARAMS,\n"
        "    overwrite=False,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)