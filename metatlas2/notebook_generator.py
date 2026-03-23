import os
import sys
import nbformat

# Type hint workaround: AutoIdentification is imported at runtime
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from workflow_objects import AutoIdentification

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('load_tools')

def generate_gui_notebooks(
    auto_id_obj: "AutoIdentification",
) -> nbformat.NotebookNode:
    """Build a complete notebook for one analysis."""

    if auto_id_obj.workflow_params.get("create_curation_notebooks", False) is not True:
        logger.info("create_curation_notebooks is not True, skipping notebook generation.")
        return

    logger.info(f"Generating analysis GUI notebook for atlas {auto_id_obj.post_autoid_atlas_obj.atlas_uid}")

    logger.info("Building notebook cells...")
    nb = nbformat.v4.new_notebook()

    kernel_name = "metatlas2"
    nb.metadata["kernelspec"] = {
        "display_name": kernel_name,
        "language": "python",
        "name": kernel_name,
    }
    logger.info(f"Notebook kernel set to '{kernel_name}'")

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
        f"{auto_id_obj.post_autoid_atlas_obj.chromatography}"
        f"_{auto_id_obj.post_autoid_atlas_obj.analysis_type}"
        f"_{auto_id_obj.post_autoid_atlas_obj.polarity}"
        f"_RTA{auto_id_obj.rt_alignment_number}"
        f"_TGA{auto_id_obj.analysis_number}"
        f".ipynb"
    )
    out_path = os.path.join(auto_id_obj.paths['analysis_output_dir'], fname)
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info(f"Notebook written to {out_path}")

def _make_parameters_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    params = auto_id_obj.workflow_params
    param_keys = [
        "ms1_min_peak_intensity",
        "ms1_min_num_points",
        "ms2_min_score",
        "ms2_min_matching_frags",
        "gui_lcmsruns_colors",
    ]
    src = "# Parameters to override for GUI analysis\n"
    src += "override_parameters = {\n"
    for key in param_keys:
        current_value = params.get(key, None)
        src += f"    '{key}': None, # current value: {repr(current_value)}\n"
    src += "}\n"
    return nbformat.v4.new_code_cell(src)

def _make_header_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    text = (
        f"# Analysis GUI & Summary\n"
        f"**Project:** `{auto_id_obj.project_name}`  \n"
        f"**Chromatography:** {auto_id_obj.post_autoid_atlas_obj.chromatography}  \n"
        f"**Polarity:** {auto_id_obj.post_autoid_atlas_obj.polarity}  \n"
        f"**Analysis type:** {auto_id_obj.post_autoid_atlas_obj.analysis_type}  \n"
        f"**Analysis number:** {auto_id_obj.analysis_number}  \n\n"
        f"Run the **GUI cell**, curate, then run the **Summary cell**."
    )
    return nbformat.v4.new_markdown_cell(text)


def _make_imports_cell() -> nbformat.NotebookNode:
    src = (
        "import sys\n"
        "import logging\n"
        "import pandas as pd\n"
        "pd.options.display.max_colwidth = 600\n\n"
        "sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')\n"
        "import logging_config as lcf\n"
        "import run_workflows as rwf\n\n"
        "lcf.setup_logging(log_level=logging.INFO)\n"
        "logger = lcf.get_logger('analysis_gui')\n"
    )
    return nbformat.v4.new_code_cell(src)


def _make_variables_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    src = (
        f"ANALYSIS_CONFIG  = {auto_id_obj.config_path!r}\n"
        f"PROJECT_NAME     = {auto_id_obj.project_name!r}\n"
        f"RT_ALIGN_NUM     = {auto_id_obj.rt_alignment_number!r}\n"
        f"ANALYSIS_NUM     = {auto_id_obj.analysis_number!r}\n"
        f"ANALYSIS_ATLAS   = {auto_id_obj.post_autoid_atlas_obj.atlas_uid!r}\n"
        f"CHROMATOGRAPHY   = {auto_id_obj.post_autoid_atlas_obj.chromatography!r}"
    )
    return nbformat.v4.new_code_cell(src)


def _make_gui_cell() -> nbformat.NotebookNode:
    src = (
        "# ── Manual Curation ──────────────────────────────────────────\n"
        "rwf.run_analysis_gui(\n"
        "    config_path=ANALYSIS_CONFIG,\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    pre_curation_atlas=ANALYSIS_ATLAS,\n"
        "    override_parameters=override_parameters,\n"
        ")\n"
    )
    return nbformat.v4.new_code_cell(src)


def _make_summary_cell() -> nbformat.NotebookNode:
    src = (
        "# ── Analysis Summary ─────────────────────────────────────────\n"
        "rwf.run_analysis_summary(\n"
        "    config_path=ANALYSIS_CONFIG,\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    pre_curation_atlas=ANALYSIS_ATLAS,\n"
        "    overwrite=False,\n"
        ")\n"
    )
    return nbformat.v4.new_code_cell(src)