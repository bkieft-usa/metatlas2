"""Tests for metatlas2.untargeted.packaging (Step 7; rclone is mocked)."""

from __future__ import annotations

import zipfile

from metatlas2 import gdrive_upload as gdu
from metatlas2.untargeted import packaging as pk
from untargeted_fixtures import PROJECT_EB, PROJECT_JGI


def test_abridged_project_name():
    assert pk.abridged_project_name(PROJECT_JGI, []) == "20260101_JGI_AB_123456_C18"
    assert pk.abridged_project_name(PROJECT_JGI + "_test", ["_test"]) == "20260101_JGI_AB_123456_C18"
    assert pk.abridged_project_name("a_b_c", []) is None
    assert pk.abridged_project_name(PROJECT_JGI.replace("C18", "PGC"), []) is None


def test_zip_is_flat_excludes_markers_and_renames(tmp_path):
    pos = tmp_path / f"{PROJECT_EB}_positive"
    neg = tmp_path / f"{PROJECT_EB}_negative"
    for d in (pos, neg):
        d.mkdir()
        (d / f"{d.name}_peak-height.csv").write_text("x")
        (d / f"{d.name}_mzmine-status.txt").write_text("DONE 1")
        (d / f"{d.name}_mzmine-queued.txt").write_text("1")
    doc = tmp_path / "Guide.docx"
    doc.write_text("doc")
    archive = tmp_path / f"{PROJECT_EB}.zip"
    abridged = pk.abridged_project_name(PROJECT_EB, [])
    pk.zip_untargeted_results([pos, neg], archive, PROJECT_EB, [doc], abridged)
    names = sorted(zipfile.ZipFile(archive).namelist())
    assert names == sorted([f"{abridged}_positive_peak-height.csv", f"{abridged}_negative_peak-height.csv",
                            "Guide.docx"])
    assert "/" not in "".join(names)


def test_upload_skips_existing_without_overwrite(monkeypatch, tmp_path):
    z = tmp_path / "P.zip"
    z.write_text("z")
    calls = []
    monkeypatch.setattr(gdu, "rclone_list", lambda remote, **kw: [["1", "P.zip"]])
    monkeypatch.setattr(gdu, "_rclone_copy", lambda *a, **kw: calls.append(kw))
    assert pk.upload_to_google_drive(z, "remote:/out", overwrite=False) is False
    assert calls == []


def test_upload_new_file_and_overwrite(monkeypatch, tmp_path):
    z = tmp_path / "P.zip"
    z.write_text("z")
    listing = []
    calls = []

    def fake_copy(source, drive, dest, overwrite=False, **kw):
        calls.append((drive, str(dest), kw))
        listing.append(["1", "P.zip"])

    monkeypatch.setattr(gdu, "rclone_list", lambda remote, **kw: list(listing))
    monkeypatch.setattr(gdu, "_rclone_copy", fake_copy)
    assert pk.upload_to_google_drive(z, "remote:/out", overwrite=False) is True
    assert calls[0][:2] == ("remote", "/out") and calls[0][2]["ignore_existing"] is True
    assert pk.upload_to_google_drive(z, "remote:/out", overwrite=True) is True
    assert calls[1][2]["update"] is True


def test_add_gnps2_documentation(monkeypatch, tmp_path):
    def fake_run(args):
        (tmp_path / "Guide.docx").write_text("doc")
        return ""

    monkeypatch.setattr(gdu, "rclone_run", fake_run)
    assert pk.add_gnps2_documentation(str(tmp_path), "Guide.docx", "remote:/out")
    assert not pk.add_gnps2_documentation(str(tmp_path), None, "remote:/out")
