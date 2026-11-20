"""Tests for metatlas2.untargeted.gnps2 (no network: HTTP/SFTP are mocked)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metatlas2.untargeted import gnps2 as g
from metatlas2.untargeted.config import DEFAULT_CONFIG
from untargeted_fixtures import PROJECT_EB, make_raw_project, write_mgf

STATUS_HTML = """<html><body><table>
<tr><td>Status</td><td>DONE</td></tr>
<tr><td>Version</td><td>2026.10.01</td></tr>
</table></body></html>"""


def test_read_password_formats(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text('MYVARIABLE="s3cret"\n')
    assert g.read_gnps2_password(p) == "s3cret"
    p.write_text("PW='abc'\n")
    assert g.read_gnps2_password(p) == "abc"
    assert g.read_gnps2_password(tmp_path / "missing") is None


def test_extract_task_id_and_status_page():
    tid = "a" * 32
    assert g.extract_task_id(f'<a href="/status?task={tid}">x</a>') == tid
    assert g.extract_task_id("<html>nothing</html>") is None
    assert g.parse_status_page(STATUS_HTML) == ("DONE", "2026.10.01")
    assert g.parse_status_page("<html></html>") == ("N/A", "N/A")


def test_fbmn_and_eb_parameters():
    cfg = DEFAULT_CONFIG["gnps2"]
    p = g.fbmn_parameters(cfg["fbmn_params"], "desc", "q.csv", "s.mgf", "m.tab", "raw")
    assert p["description"] == "desc" and p["workflowname"] == "feature_based_molecular_networking_workflow"
    assert (p["inputfeatures"], p["inputspectra"], p["metadata_filename"], p["input_raw_spectra"]) == (
        "q.csv", "s.mgf", "m.tab", "raw")
    assert p["library_min_cosine"] == "0.7" and p["normalization"] == "None"
    eb = g.everything_bagel_parameters(cfg["everything_bagel_params"], "d", "a;b", "m.tab")
    assert eb["workflowname"] == "everything_bagel_workflow" and eb["input_spectra"] == "a;b"
    assert eb["metadata_file"] == "m.tab" and eb["mode"] == "fbmn"


def test_build_eb_file_list(tmp_path):
    names = make_raw_project(tmp_path, "egsb", PROJECT_EB, groups=(("Sample", 2),))
    local = tmp_path / "egsb" / PROJECT_EB
    out = g.build_eb_file_list(local, "USERUPLOAD/u/raw_data/egsb/P", "positive")
    pos = sorted(n for n in names if "_POS_" in n)
    assert out == ";".join(f"USERUPLOAD/u/raw_data/egsb/P/{n}" for n in pos)
    with pytest.raises(ValueError):
        g.build_eb_file_list(local, "x", "fps")
    (local / "bad_name.mzML").write_text("")
    with pytest.raises(ValueError, match="Cannot determine polarity"):
        g.build_eb_file_list(local, "x", "positive")


def test_mgf_helpers(tmp_path):
    mgf = tmp_path / "P_positive.mgf"
    write_mgf(mgf, 3, calibrant=202.0777)
    assert g.count_mgf_lines(mgf) == 3
    removed = g.remove_contaminant_from_mgf(mgf, {"positive": 202.0777, "negative": 202.0788}, 5)
    assert removed == 3
    assert "202.07770" not in mgf.read_text()
    assert g.count_mgf_lines(mgf) == 3
    assert g.remove_contaminant_from_mgf(tmp_path / "nopol.mgf", {"positive": 1, "negative": 1}) == -1
    assert g.count_mgf_lines(tmp_path / "missing.mgf") == 0


class _Resp:
    def __init__(self, json_data=None, text="", status=200, content=b""):
        self._json, self.text, self.status_code, self.content = json_data, text, status, content

    def json(self):
        if self._json is None:
            raise ValueError("not json")
        return self._json

    def iter_content(self, chunk_size=1):
        yield self.content


def test_submit_json_and_html(monkeypatch, tmp_path):
    cred = tmp_path / "c.txt"
    cred.write_text('MYVARIABLE="pw"\n')
    posts = []

    class Session:
        def post(self, url, data=None, timeout=None):
            posts.append((url, data))
            if url.endswith("/login"):
                return _Resp({})
            return responses.pop(0)

    import requests
    monkeypatch.setattr(requests, "Session", Session)
    client = g.Gnps2Client("u", str(cred))
    responses = [_Resp({"task": "t1"})]
    assert client.submit({"workflowname": "w"}) == {"task": "t1"}
    assert posts[0][1] == {"username": "u", "password": "pw"}
    tid = "b" * 32
    responses = [_Resp(None, text=f"<a href='/status?task={tid}'>")]
    assert client.submit({"workflowname": "w"}) == {"task": tid}
    assert g.Gnps2Client("u", str(cred), dry_run=True).submit({"workflowname": "w"}) == {}


def test_check_status_and_download(monkeypatch, tmp_path):
    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kw: _Resp(content=STATUS_HTML.encode())
                        if "status" in url else _Resp(content=b"data", status=200))
    client = g.Gnps2Client("u", "x")
    assert client.check_status("t") == ("DONE", "2026.10.01")
    target = tmp_path / "f.tsv"
    assert client.download("https://gnps2.org/resultfile?task=t", target) and target.read_bytes() == b"data"
    assert client.user_upload_prefix == "USERUPLOAD/u"
    assert client.graphml_url("t").endswith("task=t&viewname=graphml&resultdisplay_type=task")


class _FakeSftp:
    def __init__(self, existing_dirs=(), remote_files=None):
        self.dirs = set(existing_dirs)
        self.remote = dict(remote_files or {})
        self.put_calls = []

    def mkdir(self, d):
        if d in self.dirs:
            raise OSError("mkdir file exists")
        self.dirs.add(d)

    def listdir(self, d):
        if d not in self.dirs:
            raise FileNotFoundError(d)
        return []

    def stat(self, p):
        if p not in self.remote:
            raise FileNotFoundError(p)
        return self.remote[p]

    def put(self, local, remote):
        self.put_calls.append(remote)

    def close(self):
        pass


def test_mirror_raw_data_modes(monkeypatch, tmp_path):
    names = make_raw_project(tmp_path, "egsb", PROJECT_EB, groups=(("Sample", 1),))
    local = tmp_path / "egsb" / PROJECT_EB
    remote_dir = f"/raw_data/egsb/{PROJECT_EB}"
    client = g.Gnps2Client("u", "x")

    sftp = _FakeSftp()
    monkeypatch.setattr(client, "connect_sftp", lambda: (SimpleNamespace(close=lambda: None), sftp))
    assert client.mirror_raw_data(PROJECT_EB, str(local), mode="pipeline")
    assert len(sftp.put_calls) == len(names)

    # existing directory -> soft sync skips unchanged files
    first = names[0]
    st = (local / first).stat()
    sftp = _FakeSftp(existing_dirs=[remote_dir],
                     remote_files={f"{remote_dir}/{first}": SimpleNamespace(st_size=st.st_size, st_mtime=st.st_mtime + 1)})
    monkeypatch.setattr(client, "connect_sftp", lambda: (SimpleNamespace(close=lambda: None), sftp))
    assert client.mirror_raw_data(PROJECT_EB, str(local), mode="pipeline")
    assert f"{remote_dir}/{first}" not in sftp.put_calls and len(sftp.put_calls) == len(names) - 1

    # hard mode uploads everything
    sftp = _FakeSftp(existing_dirs=[remote_dir])
    monkeypatch.setattr(client, "connect_sftp", lambda: (SimpleNamespace(close=lambda: None), sftp))
    assert client.mirror_raw_data(PROJECT_EB, str(local), mode="hard")
    assert len(sftp.put_calls) == len(names)


def test_mirror_mzmine_results_uploads_only_mgf_csv_tab(monkeypatch, tmp_path):
    d = tmp_path / "P_positive"
    d.mkdir()
    for n in ("P_positive.mgf", "P_positive_quant.csv", "P_positive_metadata.tab", "P_positive_batch-params.xml"):
        (d / n).write_text("x")
    sftp = _FakeSftp()
    client = g.Gnps2Client("u", "x")
    monkeypatch.setattr(client, "connect_sftp", lambda: (SimpleNamespace(close=lambda: None), sftp))
    assert client.mirror_mzmine_results("P", "positive", str(tmp_path))
    assert sorted(p.rsplit("/", 1)[1] for p in sftp.put_calls) == [
        "P_positive.mgf", "P_positive_metadata.tab", "P_positive_quant.csv"]
