import ast
from types import SimpleNamespace
from unittest.mock import Mock

import nbformat

from metatlas2.shared_tools.notebook_generator import _make_header_cell, generate_standards_annotation_notebook


def test_header_displays_input_atlas_tags_and_uid():
    cell = _make_header_cell({
        "project_name": "project",
        "atlas_tags": ["EMA", "ISTD"],
        "input_atlas_uid": "atlas-uid",
    })

    assert "Input atlas tags: `EMA, ISTD`" in cell.source
    assert "Input atlas UID: `atlas-uid`" in cell.source


def test_header_omits_empty_tag_provenance_for_uid_selector():
    cell = _make_header_cell({
        "project_name": "project",
        "atlas_tags": [],
        "input_atlas_uid": "custom-atlas-uid",
    })

    assert "Input atlas tags:" not in cell.source
    assert "Input atlas UID: `custom-atlas-uid`" in cell.source


def test_standards_notebook_passes_reviewed_run_ids_and_options(tmp_path):
    config_path = str(tmp_path / "analyst's config.yaml")
    path = generate_standards_annotation_notebook(
        SimpleNamespace(project_db_path=tmp_path / "project.duckdb"),
        {"project": {"standards_run_name": "run"}},
        config_path,
        tmp_path,
    )
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, "<notebook>", "exec")

    step_headers = [
        cell.source.splitlines()[0] for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## Step ")
    ]
    assert step_headers == [
        "## Step 0: Load Data",
        "## Step 1: Peak Selection GUI",
        "## Step 2: Annotation Summary (run after completing peak selection in GUI)",
        "## Step 3: Reverse RT Alignment (run after peak selections are reviewed)",
        "## Step 4: Upload and Deposit Atlas and MSMS Refs (run after peak selections and alignment are approved)",
    ]

    imports = ast.parse(notebook.cells[1].source)
    imported_functions = {
        (node.module, alias.name)
        for node in imports.body if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {
        ("metatlas2.standards_annotation.standards_annotation_db", "load_extraction_from_db"),
        ("metatlas2.standards_annotation.standards_annotation_db", "start_new_annotation_run"),
        ("metatlas2.standards_annotation.standards_annotation_db", "discard_draft_annotation_run"),
        ("metatlas2.standards_annotation.standards_annotation_review", "show_standards_project_history"),
        ("metatlas2.standards_annotation.standards_annotation_gui", "build_standards_annotation_gui"),
        ("metatlas2.standards_annotation.standards_annotation_summary", "run_standards_annotation_summary"),
        ("metatlas2.shared_tools.load_tools", "load_standards_config"),
        ("metatlas2.shared_tools.logging_config", "setup_logging"),
        ("metatlas2.shared_tools.logging_config", "get_logger"),
        ("metatlas2.standards_annotation.standards_annotation_rt_align", "run_reverse_rt_align"),
        ("metatlas2.standards_annotation.deposit_ref_standards_atlas_and_msms_refs", "run_atlas_and_msms_deposit"),
    } == imported_functions

    def command_index(name):
        return next(
            i for i, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and f"{name}(" in cell.source
        )

    gui_index = command_index("build_standards_annotation_gui")
    summary_index = command_index("run_standards_annotation_summary")
    alignment_index = command_index("run_reverse_rt_align")
    deposit_index = command_index("run_atlas_and_msms_deposit")
    assert gui_index < summary_index < alignment_index - 1 < deposit_index - 1
    gui_params = notebook.cells[gui_index - 1]
    alignment_params = notebook.cells[alignment_index - 1]
    deposit_params = notebook.cells[deposit_index - 1]
    assert gui_params.cell_type == alignment_params.cell_type == deposit_params.cell_type == "code"
    for index in (gui_index, alignment_index, deposit_index):
        assert "show_standards_project_history(" in notebook.cells[index - 2].source

    align, deposit, history = Mock(), Mock(), Mock()
    load_config = Mock(return_value={"configured": True})
    load_data = Mock(return_value="loaded data")
    gui, summary = Mock(), Mock()
    namespace = {
        "load_standards_config": load_config,
        "load_extraction_from_db": load_data,
        "build_standards_annotation_gui": gui,
        "run_standards_annotation_summary": summary,
        "run_reverse_rt_align": align,
        "run_atlas_and_msms_deposit": deposit,
        "show_standards_project_history": history,
    }
    exec(notebook.cells[2].source, namespace)
    assert "ANNOTATION_TIMESTAMP" not in namespace
    assert "ALIGNMENT_TIMESTAMP" not in namespace
    exec(notebook.cells[command_index("load_extraction_from_db")].source, namespace)
    exec(notebook.cells[gui_index - 2].source, namespace)
    history.assert_called_once_with(str(tmp_path / "project.duckdb"))
    exec(gui_params.source, namespace)
    assert namespace["ANNOTATION_RUN_ID"] is None
    exec(notebook.cells[gui_index].source, namespace)
    summary_params = next(cell for cell in notebook.cells if "UPLOAD_SUMMARY_TO_GDRIVE = " in cell.source)
    exec(summary_params.source, namespace)
    assert namespace["UPLOAD_SUMMARY_TO_GDRIVE"] is True
    namespace["UPLOAD_SUMMARY_TO_GDRIVE"] = False
    exec(notebook.cells[summary_index].source, namespace)
    load_config.assert_called_once_with(config_path)
    load_data.assert_called_once_with(str(tmp_path / "project.duckdb"), {"configured": True})
    gui.assert_called_once_with(data="loaded data", config={"configured": True}, annotation_run_id=None)
    summary.assert_called_once_with(data="loaded data", config={"configured": True}, upload_to_gdrive=False)

    exec(alignment_params.source, namespace)
    assert namespace["ANNOTATION_RUN_ID"] is None
    assert namespace["SKIP_RT_ALIGNMENT"] is False
    assert namespace["UPLOAD_ALIGNMENT_TO_GDRIVE"] is True
    assert "ALIGNMENT_RUN_ID" not in namespace
    assert "DRY_RUN" not in namespace
    namespace["ANNOTATION_RUN_ID"] = "from step 3"
    exec(deposit_params.source, namespace)
    assert namespace["ANNOTATION_RUN_ID"] is None
    assert namespace["DRY_RUN"] is True
    assert namespace["ALIGNMENT_RUN_ID"] is None
    assert namespace["FORCE_ADD"] is False
    assert namespace["SKIP_ATLAS_UPLOAD"] is False
    assert namespace["SKIP_MSMS_REFS_DEPOSIT"] is False
    assert namespace["UPLOAD_ATLAS_AND_MSMS_DEPOSIT_PREVIEW_TO_GDRIVE"] is True

    namespace.update(
        ANNOTATION_RUN_ID=2,
        ALIGNMENT_RUN_ID=3,
        SKIP_RT_ALIGNMENT=True,
        DRY_RUN=False,
        FORCE_ADD=True,
        SKIP_ATLAS_UPLOAD=True,
        SKIP_MSMS_REFS_DEPOSIT=True,
        UPLOAD_ALIGNMENT_TO_GDRIVE=False,
        UPLOAD_ATLAS_AND_MSMS_DEPOSIT_PREVIEW_TO_GDRIVE=False,
    )
    exec(notebook.cells[alignment_index].source, namespace)
    exec(notebook.cells[deposit_index].source, namespace)
    align.assert_called_once_with(
        config_path=config_path, annotation_run_id=2, skip_rt_alignment=True,
        upload_to_gdrive=False,
    )
    deposit.assert_called_once_with(
        config_path=config_path,
        annotation_run_id=2,
        alignment_run_id=3,
        dry_run=False,
        force_add=True,
        skip_atlas_upload=True,
        skip_msms_refs_deposit=True,
        upload_to_gdrive=False,
    )
