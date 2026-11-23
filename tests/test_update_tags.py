from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from metatlas2.shared_tools import update_tags


KEY = "GTZCVFVGUGFEME-IWQZZHSRSA-N"
OTHER_KEY = "GFFGJBXGBJISGV-UHFFFAOYSA-N"


class Sheet:
    def __init__(self, values):
        self.values = values
        self.changes = []

    def get_all_values(self):
        return self.values

    def batch_update(self, changes, **kwargs):
        self.changes.append((changes, kwargs))


def _csv(tmp_path, contents):
    path = tmp_path / "tags.csv"
    path.write_text(contents, encoding="utf-8")
    return path


@pytest.mark.parametrize("contents", [
    "tags,inchi_key\nx," + KEY + "\n",
    "inchi_key,tags,other\n" + KEY + ",x,y\n",
    "inchi_key,tags\nbad-key,x\n",
    "inchi_key,tags\n" + KEY.lower() + ",x\n",
    "inchi_key,tags\n" + KEY + ",\n",
    "inchi_key,tags\n" + KEY + ",x;;y\n",
    "inchi_key,tags\n" + KEY + ",\"x\nsecond line\"\n",
    "inchi_key,tags\n" + KEY + ",x\n" + KEY + ",y\n",
    "inchi_key,tags\n" + KEY + ",valid\nbad-key,x\n",
    "inchi_key,tags\n",
    "inchi_key,tags\n" + KEY + ",x,extra\n",
])
def test_invalid_csv_is_rejected_before_opening_sheet(tmp_path, monkeypatch, contents):
    path = _csv(tmp_path, contents)
    monkeypatch.setattr(
        update_tags.atlas_sheet, "open_centralized_atlas_worksheet",
        lambda: pytest.fail("invalid CSV must not open the sheet"),
    )
    with pytest.raises(ValueError):
        update_tags.main(["add", "--input-csv", str(path)])


@pytest.mark.parametrize(
    ("operation", "requested", "expected"),
    [
        ("add", ["EMA", "New Set", "New Set"], ["EMA;EMAX;New Set", "Other;EMA;New Set"]),
        ("remove", ["EMA"], ["EMAX", "Other"]),
        ("retag", ["New Set", "EMA"], ["New Set;EMA", "New Set;EMA"]),
    ],
)
def test_updates_every_matching_row_only(operation, requested, expected):
    sheet = Sheet([
        ["tags", "extra", "inchi_key"],
        ["EMA;EMAX", "unchanged", KEY],
        [],
        ["Other", "untouched", OTHER_KEY],
        ["Other;EMA", "unchanged", KEY],
    ])
    assert update_tags.update_tags(sheet, operation, {KEY: requested}) == 2
    assert sheet.changes == [([
        {"range": "A2", "values": [[expected[0]]]},
        {"range": "A5", "values": [[expected[1]]]},
    ], {"value_input_option": "RAW"})]


def test_missing_key_and_existing_tag_warn_without_writing(monkeypatch):
    sheet = Sheet([["inchi_key", "tags"], [KEY, "EMA"]])
    warnings = []
    monkeypatch.setattr(update_tags.logger, "warning", lambda message, *args: warnings.append(message % args))
    assert update_tags.update_tags(sheet, "add", {KEY: ["EMA"], OTHER_KEY: ["Other"]}) == 0
    assert not sheet.changes
    assert any("already exists" in message for message in warnings)
    assert any(OTHER_KEY in message and "not found" in message for message in warnings)


def test_remove_all_tags_writes_empty_cell():
    sheet = Sheet([["inchi_key", "tags"], [KEY, "EMA"]])
    update_tags.update_tags(sheet, "remove", {KEY: ["EMA"]})
    assert sheet.changes[0][0] == [{"range": "B2", "values": [[""]]}]


def test_sheet_change_aborts_before_writing():
    class ChangedSheet(Sheet):
        def get_all_values(self):
            values = super().get_all_values()
            self.values = [["inchi_key", "tags"], [OTHER_KEY, "different"]]
            return values

    sheet = ChangedSheet([["inchi_key", "tags"], [KEY, "EMA"]])
    with pytest.raises(RuntimeError, match="changed during preparation"):
        update_tags.update_tags(sheet, "add", {KEY: ["New"]})
    assert not sheet.changes


@pytest.mark.parametrize("header", [[], ["inchi_key", "tags", "tags"], ["inchi_key"]])
def test_invalid_sheet_header_does_not_write(header):
    sheet = Sheet([header, [KEY, "EMA"]])
    with pytest.raises(ValueError, match="exactly one"):
        update_tags.update_tags(sheet, "add", {KEY: ["New"]})
    assert not sheet.changes


def test_retrieve_prints_unique_sorted_tokens_only(monkeypatch, capsys):
    sheet = Sheet([
        ["inchi_key", "tags"],
        [KEY, " EMA; PlantHormone ;EMA"],
        [OTHER_KEY, "New Set;EMA"],
        [KEY],
    ])
    monkeypatch.setattr(update_tags.atlas_sheet, "open_centralized_atlas_worksheet", lambda: (None, sheet))
    update_tags.main(["retrieve"])
    assert capsys.readouterr().out == "EMA\nNew Set\nPlantHormone\n"
    assert not sheet.changes


def test_retrieve_rejects_multiline_tag():
    sheet = Sheet([["inchi_key", "tags"], [KEY, "two\nlines"]])
    with pytest.raises(ValueError, match="control character"):
        update_tags.retrieve_tags(sheet)


def test_main_reads_csv_and_updates_sheet(tmp_path, monkeypatch):
    path = _csv(tmp_path, f"inchi_key,tags\n{KEY}, New Set ; Second \n")
    sheet = Sheet([["inchi_key", "tags"], [KEY, "EMA"]])
    monkeypatch.setattr(update_tags.atlas_sheet, "open_centralized_atlas_worksheet", lambda: (None, sheet))
    update_tags.main(["add", "--input_csv", str(path)])
    assert sheet.changes[0][0] == [{"range": "B2", "values": [["EMA;New Set;Second"]]}]


@pytest.mark.parametrize("runtime", ["docker", "shifter"])
@pytest.mark.parametrize("operation", ["add", "retrieve"])
def test_entrypoint_dispatches_tag_command(tmp_path, runtime, operation):
    script = Path(__file__).resolve().parents[1] / "scripts" / "metatlas2.sh"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for command in ("dirname", "basename", "id"):
        (bin_dir / command).symlink_to(f"/usr/bin/{command}")
    executable = bin_dir / runtime
    executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$ARGS_FILE"\n', encoding="utf-8")
    executable.chmod(0o755)
    data = tmp_path / "data"
    data.mkdir()
    csv_path = _csv(tmp_path, f"inchi_key,tags\n{KEY},new\n")
    capture = tmp_path / "args.txt"
    env = {
        "PATH": str(bin_dir),
        "HOME": str(tmp_path),
        "METATLAS_DATA_DIR": str(data),
        "ARGS_FILE": str(capture),
    }
    arguments = [operation]
    if operation == "add":
        arguments += ["--input-csv", str(csv_path)]
    result = subprocess.run(
        ["/bin/bash", str(script), "--dev", "update-tags", *arguments],
        env=env, text=True, capture_output=True, check=True,
    )
    args = capture.read_text(encoding="utf-8").splitlines()
    assert result.stdout == ""
    expected = ["/app/.venv/bin/python", "-m", "metatlas2.shared_tools.update_tags", *arguments]
    assert args[-len(expected):] == expected
    if runtime == "docker" and operation == "add":
        assert f"{csv_path}:{csv_path}:ro" in args
