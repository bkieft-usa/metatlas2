from pathlib import Path
from unittest.mock import Mock, call

import pytest

from metatlas2 import gdrive_upload as drive


@pytest.fixture
def annotation(tmp_path):
    directory = tmp_path / "outputs" / "annotation_run_20260101T120000_000000Z"
    directory.mkdir(parents=True)
    (directory / "summary.csv").write_text("summary")
    alignment = directory / "rt_align_20260101T120001_000000Z"
    alignment.mkdir()
    (alignment / "atlas.csv").write_text("atlas")
    (alignment / "refs.tab").write_text("refs")
    return directory, alignment


def test_standards_upload_preserves_hierarchy(annotation, monkeypatch):
    directory, alignment = annotation
    config = {"project": {"standards_run_name": "run"}, "gdrive": {"root_folder_id": "root"}}
    monkeypatch.setattr(drive, "_get_drive_name_for_id", lambda root: "jgi" if root == "root" else None)
    monkeypatch.setattr(drive, "_has_drive_access", lambda _: (True, None))
    copy = Mock()
    monkeypatch.setattr(drive, "_rclone_copy", copy)
    folder_url = Mock(return_value="https://drive.google.com/drive/folders/123")
    monkeypatch.setattr(drive, "_drive_path_to_url", folder_url)

    dest = Path("Reference_Standards_Annotation/run") / directory.name
    for sources in ([directory], [alignment], [alignment / "atlas.csv", alignment / "refs.tab"]):
        assert drive.upload_standards_outputs(config, directory, sources) == folder_url.return_value
    assert copy.call_args_list == [
        call(directory, "jgi", dest, overwrite=True, apply_excludes=False, required=True),
        call(alignment, "jgi", dest / alignment.name, overwrite=True, apply_excludes=False, required=True),
        call(alignment / "atlas.csv", "jgi", dest / alignment.name, overwrite=True, apply_excludes=False, required=True),
        call(alignment / "refs.tab", "jgi", dest / alignment.name, overwrite=True, apply_excludes=False, required=True),
    ]
    folder_url.assert_called_with("jgi", dest)


def test_standards_upload_fails_explicitly(annotation, monkeypatch, tmp_path):
    directory, alignment = annotation
    config = {"project": {"standards_run_name": "run"}, "gdrive": {"root_folder_id": "root"}}
    monkeypatch.setattr(drive, "_get_drive_name_for_id", lambda _: "jgi")
    monkeypatch.setattr(drive, "_has_drive_access", lambda _: (False, "denied"))
    with pytest.raises(RuntimeError, match="denied"):
        drive.upload_standards_outputs(config, directory, [alignment])
    with pytest.raises(ValueError, match="outside"):
        drive.upload_standards_outputs(config, directory, [tmp_path])
    config["gdrive"] = {}
    with pytest.raises(ValueError, match="root_folder_id"):
        drive.upload_standards_outputs(config, directory, [directory])


def test_standards_upload_requires_resolvable_link(annotation, monkeypatch):
    directory, _ = annotation
    config = {"project": {"standards_run_name": "run"}, "gdrive": {"root_folder_id": "root"}}
    monkeypatch.setattr(drive, "_get_drive_name_for_id", lambda _: "jgi")
    monkeypatch.setattr(drive, "_has_drive_access", lambda _: (True, None))
    monkeypatch.setattr(drive, "_rclone_copy", Mock())
    monkeypatch.setattr(drive, "_drive_path_to_url", lambda *_: None)
    with pytest.raises(RuntimeError, match="could not find Drive folder"):
        drive.upload_standards_outputs(config, directory, [directory])
