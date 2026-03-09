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

    pre_curation_atlas = auto_id_obj.post_autoid_atlas_obj
    logger.info(f"Generating analysis GUI notebook for atlas {pre_curation_atlas.atlas_uid}")

    logger.info("Building notebook cells...")
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        _make_header_cell(auto_id_obj),
        _make_imports_cell(),
        _make_variables_cell(auto_id_obj),
        _make_parameters_cell(auto_id_obj),
        _make_gui_cell(),
        _make_summary_divider_cell(),
        _make_summary_cell(),
    ]

    logger.info("Generating notebook file...")
    fname = (
        f"_{pre_curation_atlas.chromatography}"
        f"_{pre_curation_atlas.analysis_type}"
        f"_{pre_curation_atlas.polarity}"
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
        "ppm_error",
        "extra_time",
        "ms2_min_score",
        "ms2_min_matches",
        "ms2_frag_mz_tolerance",
        "gui_require_all_evaluated",
        "gui_lcmsruns_colors",
    ]
    src = "# Parameters to override for GUI analysis\n"
    src += "override_parameters = {\n"
    for key in param_keys:
        value = params.get(key, None)
        src += f"    '{key}': {repr(value)},\n"
    src += "}\n"
    src += "# Edit override_parameters as needed before running the GUI cell."
    return nbformat.v4.new_code_cell(src)

def _make_header_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    pre_curation_atlas = auto_id_obj.post_autoid_atlas_obj
    text = (
        f"# Analysis GUI & Summary\n"
        f"**Project:** `{auto_id_obj.project_name}`  \n"
        f"**Chromatography:** {pre_curation_atlas.chromatography}  \n"
        f"**Polarity:** {pre_curation_atlas.polarity}  \n"
        f"**Analysis type:** {pre_curation_atlas.analysis_type}  \n"
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
        "import load_tools as ldt\n"
        "import run_workflows as rwf\n\n"
        "lcf.setup_logging(log_level=logging.INFO)\n"
        "logger = lcf.get_logger('analysis_gui')\n"
    )
    return nbformat.v4.new_code_cell(src)


def _make_variables_cell(auto_id_obj: "AutoIdentification") -> nbformat.NotebookNode:
    pre_curation_atlas = auto_id_obj.post_autoid_atlas_obj
    src = (
        f"ANALYSIS_CONFIG  = ldt.load_metatlas2_config({auto_id_obj.config_path!r})\n"
        f"PROJECT_NAME     = {auto_id_obj.project_name!r}\n"
        f"RT_ALIGN_NUM     = {auto_id_obj.rt_alignment_number!r}\n"
        f"ANALYSIS_NUM     = {auto_id_obj.analysis_number!r}\n"
        f"ANALYSIS_ATLAS   = {pre_curation_atlas.atlas_uid!r}\n"
        f"CHROMATOGRAPHY   = {pre_curation_atlas.chromatography!r}\n"
    )
    return nbformat.v4.new_code_cell(src)


def _make_gui_cell() -> nbformat.NotebookNode:
    src = (
        "# ── Manual Curation ──────────────────────────────────────────\n"
        "# Run this cell, curate in the GUI, then run the summary cell below.\n"
        "rwf.run_analysis_gui(\n"
        "    config_path=ANALYSIS_CONFIG,\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    analysis_atlas=ANALYSIS_ATLAS,\n"
        "    override_parameters=override_parameters,\n"
        ")\n"
    )
    return nbformat.v4.new_code_cell(src)


def _make_summary_divider_cell() -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(
        "---\n## Analysis Summary\nRun after curation is complete."
    )


def _make_summary_cell() -> nbformat.NotebookNode:
    src = (
        "# ── Analysis Summary ─────────────────────────────────────────\n"
        "rwf.run_analysis_summary(\n"
        "    config_path=ANALYSIS_CONFIG,\n"
        "    project_name=PROJECT_NAME,\n"
        "    rt_alignment_number=RT_ALIGN_NUM,\n"
        "    analysis_number=ANALYSIS_NUM,\n"
        "    analysis_atlas=ANALYSIS_ATLAS,\n"
        "    overwrite=False,\n"
        ")\n"
    )
    return nbformat.v4.new_code_cell(src)