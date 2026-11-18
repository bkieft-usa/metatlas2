from metatlas2.notebook_generator import _make_header_cell


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
