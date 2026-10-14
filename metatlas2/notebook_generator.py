import os
import nbformat
from pathlib import Path

import metatlas2.logging_config as lcf
import metatlas2.file_and_project_format as fpf
logger = lcf.get_logger('notebook_generator')

#### Manual curation GUI notebook generator

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

    from metatlas2.workflow_objects import AtlasCategory
    _atlas_cat = auto_id_obj.auto_ided_atlas_obj.atlas_category
    _cat_val = _atlas_cat.value if isinstance(_atlas_cat, AtlasCategory) else str(_atlas_cat)
    run_params = {
        "project_name": auto_id_obj.project_name,
        "rt_alignment_number": auto_id_obj.rt_alignment_number,
        "analysis_number": auto_id_obj.analysis_number,
        "chromatography": fpf.canonical_chromatography(auto_id_obj.auto_ided_atlas_obj.chromatography),
        "polarity": auto_id_obj.auto_ided_atlas_obj.polarity,
        "atlas_category": _cat_val,
        "atlas_name": auto_id_obj.auto_ided_atlas_obj.atlas_name,
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
        "CurationNotebook__"
        f"{auto_id_obj.ta.label}"
        f"__{run_params['project_name']}"
        f".ipynb"
    )
    out_path = os.path.join(auto_id_obj.paths['analysis_output_dir'], fname)
    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info(f"Notebook written to {out_path}")

    os.makedirs(
        str(Path(auto_id_obj.paths["analysis_output_dir"]) / auto_id_obj.ta.label),
        exist_ok=True,
    )

    logger.info(f"Analysis output directory created at {auto_id_obj.paths['analysis_output_dir']}/{auto_id_obj.ta.label}")

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
        "apply_istd_curation_to_ref",
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
        #f"**Atlas category:** {run_params['atlas_category']}  \n"
        #f"**Atlas name:** {run_params['atlas_name']}  \n"
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
        "# Manual Curation - Dash application (opens in browser window)\n"
        "database_flush_errors = wfs.run_analysis_gui(\n"
        f"    run_parameters=RUN_PARAMS,\n"
        f"    override_parameters=OVERRIDE_PARAMS,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)

def _make_summary_cell() -> nbformat.NotebookNode:
    src = (
        "# Analysis Summary - generates analysis summaries based on curated data.\n"
        "wfs.run_analysis_summary(\n"
        f"    run_parameters=RUN_PARAMS,\n"
        f"    override_parameters=OVERRIDE_PARAMS,\n"
        f"    overwrite=False,\n"
        f"    database_flush_errors=database_flush_errors,\n"
        ")"
    )
    return nbformat.v4.new_code_cell(src)


#### Parquet query notebook generator

def generate_query_notebook(
    auto_id_obj: "AutoIdentification",
) -> str | None:
    """Build and write a single QueryNotebook for one RTA/TGA run.
    """

    logger.info(
        "Generating query notebook for project %s (%s / RTA%s / TGA%s)",
        auto_id_obj.project_name,
        auto_id_obj.chromatography,
        auto_id_obj.rt_alignment_number,
        auto_id_obj.analysis_number,
    )

    image_tag = getattr(auto_id_obj, "image_tag", "latest")
    if image_tag == "latest":
        kernel_name = "metatlas2"
        kernel_display_name = "metatlas2 (latest)"
    else:
        kernel_name = f"metatlas2-{image_tag}"
        kernel_display_name = f"metatlas2 ({image_tag})"

    run_params = {
        "project_name": auto_id_obj.project_name,
        "chromatography": auto_id_obj.chromatography,
        "rt_alignment_number": auto_id_obj.rt_alignment_number,
        "analysis_number": auto_id_obj.analysis_number,
        "parquet_output_dir": auto_id_obj.paths.get("parquet_output_dir", ""),
    }

    logger.info("Building query notebook cells...")
    nb = nbformat.v4.new_notebook()

    nb.metadata["kernelspec"] = {
        "display_name": kernel_display_name,
        "language": "python",
        "name": kernel_name,
    }
    nb.metadata["language_info"] = {"name": "python"}

    nb.cells = [
        _qnb_make_header_cell(run_params),
        _qnb_make_imports_cell(),
        _qnb_make_run_params_cell(run_params),
        _make_schema_reference_cell(),
        _make_query_builder_guide_cell(),
        _make_significant_compound_screen_cell(run_params),
        _make_high_confidence_core_hits_cell(run_params),
        _make_custom_query_template_cell(),
    ]

    fname = f"QueryNotebook__{run_params['project_name']}.ipynb"
    out_path = os.path.join(auto_id_obj.paths["analysis_output_dir"], fname)

    with open(out_path, "w") as f:
        nbformat.write(nb, f)

    logger.info("Query notebook written to %s", out_path)
    return out_path


def _qnb_make_header_cell(run_params: dict) -> nbformat.NotebookNode:
    chrom = run_params.get("chromatography", "")
    rta = run_params.get("rt_alignment_number", "")
    tga = run_params.get("analysis_number", "")
    text = (
        f"# **`{run_params['project_name']}`**  \n"
        f"**Chromatography:** {chrom} &nbsp;|&nbsp; **RT Alignment No.:** {rta} &nbsp;|&nbsp; **Targeted Analysis No.:** {tga}  \n\n"
        "See the **Schema Reference** and **Query Builder Guide** cells for "
        "instructions on writing your own queries."
    )
    return nbformat.v4.new_markdown_cell(text)


def _qnb_make_imports_cell() -> nbformat.NotebookNode:
    src = (
        "import logging\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        "import metatlas2.workflows as wfs\n"
        "import metatlas2.logging_config as lcf\n\n"
        "pd.set_option('display.max_columns', None)\n"
        "lcf.setup_logging(log_level=logging.INFO, log_to_stdout=True)\n"
        "logger = lcf.get_logger('query_notebook')"
    )
    return nbformat.v4.new_code_cell(src)


def _qnb_make_run_params_cell(run_params: dict) -> nbformat.NotebookNode:
    src = (
        "# Project-specific identifiers (defined at notebook generation time, do not change)\n"
        "RUN_PARAMS = {\n"
    )
    for key, val in run_params.items():
        if isinstance(val, str):
            src += f"    '{key}': '{val}',\n"
        else:
            src += f"    '{key}': {val!r},\n"
    src += (
        "}\n\n"
        "PARQUET_ROOT = RUN_PARAMS['parquet_output_dir']\n"
        "PROJECT_NAME = RUN_PARAMS['project_name']\n"
        "CHROMATOGRAPHY = RUN_PARAMS['chromatography']\n"
        "RTA = RUN_PARAMS['rt_alignment_number']\n"
        "TGA = RUN_PARAMS['analysis_number']\n"
    )
    return nbformat.v4.new_code_cell(src)

def _make_schema_reference_cell() -> nbformat.NotebookNode:
    text = """\
---
## Schema Reference

Use these column names as keys in your query `params` dict.
Add a suffix (see **Query Builder Guide** below) to turn a column name into a filter.

### `grain: "compound_file"` — one row per compound x raw file

| Column | Type | Description |
|--------|------|-------------|
| `compound_name` | string | Compound display name |
| `inchi_key` | string | InChI key (standard 27-char) |
| `adduct` | string | Detected adduct, e.g. `[M+H]+` |
| `formula` | string | Molecular formula |
| `smiles` | string | SMILES string |
| `filename` | string | Raw LC-MS filename |
| `file_group` | string | Sample group label |
| `msi_level` | string | MSI identification level (`1`-`4`) |
| `atlas_category` | string | Atlas category, e.g. `REF`, `EMA` |
| `atlas_name` | string | Atlas name |
| `chromatography` | string | Chromatography method, e.g. `HILICZ` |
| `polarity` | string | `POS` or `NEG` |
| `project_name` | string | Full project name |
| `control_filter` | string | Extraction-control filter result |
| `ms1_notes` | string | MS1 curation notes |
| `ms2_notes` | string | MS2 curation notes |
| `atlas_mz` | float | Theoretical m/z from atlas |
| `atlas_rt_peak` | float | Theoretical RT peak (min) |
| `best_ms1_intensity` | float | Best MS1 peak intensity |
| `best_ms1_ppm_error` | float | MS1 mass accuracy (ppm) |
| `best_ms1_rt` | float | RT of best MS1 hit (min) |
| `best_ms2_score` | float | Best MS2 spectral match score (0-1) |
| `best_ms2_mz_ppm_error` | float | MS2 precursor mass accuracy (ppm) |
| `peak_height` | float | Per-file peak height |
| `peak_area` | float | Per-file peak area |
| `total_score` | float | Composite quality score (mz + rt + ms2) |
| `mz_quality` | float | m/z quality sub-score |
| `rt_quality` | float | RT quality sub-score |
| `msms_quality` | float | MS2 quality sub-score |
| `mz_ppmerror` | float | MS1 ppm error (alias) |

### `grain: "compound_lfc"` — one row per compound x condition pair

All `compound_file` columns above **except** `filename`, `file_group`,
`peak_height`, `peak_area`, `total_score`, `mz_quality`, `rt_quality`,
`msms_quality`, plus:

| Column | Type | Description |
|--------|------|-------------|
| `log2_fold_change` | float | log₂(group1 / group2) |
| `condition_1` | string | Numerator group label |
| `condition_2` | string | Denominator group label |
"""
    return nbformat.v4.new_markdown_cell(text)


def _make_query_builder_guide_cell() -> nbformat.NotebookNode:
    """Markdown cell: filter suffix system and custom query instructions."""
    text = """\
---
## Query Builder Guide

Every query is a plain Python `dict` passed to `wfs.run_parquet_query()`.
The `grain` key selects the table; all other keys are **filter parameters**.

### Filter suffix system

Append a suffix to a column name to create a filter:

| Suffix | Meaning | Example |
|--------|---------|---------|
| *(none)* | Exact equality | `'polarity': 'POS'` |
| `_in` | Value is in list | `'adduct_in': ['[M+H]+', '[M+Na]+']` |
| `_min` | Column ≥ value | `'best_ms1_intensity_min': 10_000` |
| `_max` | Column ≤ value | `'best_ms1_ppm_error_max': 5.0` |
| `_abs_gt` | \|column\| > value | `'log2_fold_change_abs_gt': 1.5` |
| `_abs_lt` | \|column\| < value | `'log2_fold_change_abs_lt': 0.1` |
| `_abs_min` | \|column\| ≥ value | `'log2_fold_change_abs_min': 1.0` |
| `_abs_max` | \|column\| ≤ value | `'log2_fold_change_abs_max': 3.0` |

### Special keys (compound_lfc only)

| Key | Value | Meaning |
|-----|-------|---------|
| `condition_pair` | `[group_a, group_b]` | Symmetric match — finds rows where `{condition_1, condition_2} == {group_a, group_b}` regardless of stored order |
| `lfc_directional_min` | `{'treatment': ..., 'control': ..., 'threshold': ...}` | Directional threshold — correctly sign-flips when the pair is stored in reverse order |

### Minimal query template

```python
result = wfs.run_parquet_query(
    project_root=PARQUET_ROOT,
    rt_alignment_number=RTA,
    analysis_number=TGA,
    params={
        'grain': 'compound_lfc',       # or 'compound_file'
        'chromatography': CHROMATOGRAPHY,
        'project_name': PROJECT_NAME,
        # add filter keys here …
    },
)
print(result[0])   # human-readable SQL equivalent
result[1].head(20)  # pandas DataFrame of matching rows
```
"""
    return nbformat.v4.new_markdown_cell(text)


def _make_significant_compound_screen_cell(run_params: dict) -> nbformat.NotebookNode:
    """Query 1: significant_compound_screen — large |log2FC| across all analyses."""
    src = (
        "# ── Query 1: Significant Compound Screen ────────────────────────────────────\n"
        "#   Which compounds show a large absolute log2 fold-change between any two\n"
        "#   sample groups in this project?  This is the primary 'what changed?'\n"
        "#   screen — a wide net cast across all condition comparisons\n"
        "#   and polarities stored in the compound_lfc grain for this RTA/TGA run.\n"
        "#   Adjust log2_fold_change_abs_gt to tighten or loosen the threshold\n"
        "#   (1.5 ≈ 2.8x change).  Uncomment 'polarity' to restrict to one mode.\n"
        "significant_compound_screen = wfs.run_parquet_query(\n"
        "    project_root=PARQUET_ROOT,\n"
        "    rt_alignment_number=RTA,\n"
        "    analysis_number=TGA,\n"
        "    params={\n"
        "        'grain': 'compound_lfc',\n"
        "        'project_name': PROJECT_NAME,\n"
        "        'chromatography': CHROMATOGRAPHY,\n"
        "        # 'polarity': 'POS',  # uncomment to restrict to one polarity\n"
        "        'log2_fold_change_abs_gt': 1.5,\n"
        "    },\n"
        ")\n"
        "print(significant_compound_screen[0])\n"
        "significant_compound_screen[1].head(20)"
    )
    return nbformat.v4.new_code_cell(src)


def _make_high_confidence_core_hits_cell(run_params: dict) -> nbformat.NotebookNode:
    """Query 2: high_confidence_core_hits — publication-grade compound list."""
    src = (
        "# ── Query 2: High-Confidence Core Hits ──────────────────────────────────────\n"
        "#   What is the curated, publication-grade compound list for this project —\n"
        "#   i.e. compounds identified with a strong composite quality score AND\n"
        "#   sufficient intensity to be quantitatively meaningful?  This is the\n"
        "#   typical filter applied before pathway enrichment or supplementary-table\n"
        "#   export, since it excludes low-abundance or borderline-quality hits.\n"
        "#   Uncomment 'polarity' to restrict to one analysis.\n"
        "high_confidence_core_hits = wfs.run_parquet_query(\n"
        "    project_root=PARQUET_ROOT,\n"
        "    rt_alignment_number=RTA,\n"
        "    analysis_number=TGA,\n"
        "    params={\n"
        "        'grain': 'compound_file',\n"
        "        'project_name': PROJECT_NAME,\n"
        "        'chromatography': CHROMATOGRAPHY,\n"
        "        # 'polarity': 'POS',           # uncomment to restrict to one polarity\n"
        "        'atlas_category': 'REF',       # this removes any internal standards\n"
        "        'total_score_min': 1.5,\n"
        "        'best_ms1_intensity_min': 10_000,\n"
        "    },\n"
        ")\n"
        "print(high_confidence_core_hits[0])\n"
        "high_confidence_core_hits[1].head(20)"
    )
    return nbformat.v4.new_code_cell(src)

def _make_custom_query_template_cell() -> nbformat.NotebookNode:
    """Blank template cell for the analyst to write their own query."""
    src = (
        "# ── Custom Query Template ────────────────────────────────────────────────────\n"
        "# Copy this cell, rename the result variable, and fill in your own filters.\n"
        "# See the Schema Reference and Query Builder Guide cells above for all\n"
        "# available columns and filter suffixes.\n"
        "#\n"
        "# Example: find compounds with the adduct [M+H]+ with MS2 score > 0.6 and |log2FC| > 2\n"
        "# across any condition pair:\n"
        "#\n"
        "#   my_query = wfs.run_parquet_query(\n"
        "#       project_root=PARQUET_ROOT,\n"
        "#       rt_alignment_number=RTA,\n"
        "#       analysis_number=TGA,\n"
        "#       params={\n"
        "#           'grain': 'compound_lfc',\n"
        "#           'project_name': PROJECT_NAME,\n"
        "#           'chromatography': CHROMATOGRAPHY,\n"
        "#           'adduct': '[M+H]+',\n"
        "#           'best_ms2_score_min': 0.6,\n"
        "#           'log2_fold_change_abs_gt': 2.0,\n"
        "#       },\n"
        "#   )\n"
        "#   print(my_query.sql)\n"
        "#   my_query.df.head(20)\n"
    )
    return nbformat.v4.new_code_cell(src)
