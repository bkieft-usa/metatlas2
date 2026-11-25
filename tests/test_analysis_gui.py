"""Tests for the targeted Analysis GUI callbacks (no browser, no database).

The central guarantee: an MS1 figure updated by a ``dash.Patch`` after an RT
change is identical to a figure rendered from scratch for the same state.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import dash
import pytest
from plotly.io.json import to_json_plotly

sys.path.insert(0, str(Path(__file__).parent))
from analysis_gui_fixtures import make_gui_obj

import metatlas2.targeted.analysis_gui as agu


@pytest.fixture(scope="module")
def gui():
    obj = make_gui_obj(n_files=12, n_points=400, n_ms2=40)
    app = agu.build_dash_app(obj)
    callbacks = {}
    for entry in app.callback_map.values():
        if "callback" in entry:
            callbacks[entry["callback"].__wrapped__.__name__] = entry["callback"].__wrapped__
    initial_state = app.layout.children[0].data
    return obj, app, callbacks, initial_state


def _jsonify(obj):
    return json.loads(to_json_plotly(obj))


def _apply_patch(fig: dict, patch) -> dict:
    fig = copy.deepcopy(fig)
    for op in _jsonify(patch)["operations"]:
        assert op["operation"] == "Assign", op
        target = fig
        *path, last = op["location"]
        for p in path:
            target = target[p]
        target[last] = op["params"]["value"]
    return fig


def _state(base, **changes):
    s = dict(base)
    s.update(changes)
    s["edit_seq"] = int(base.get("edit_seq", 0)) + 1
    return s


def _iso_colors(fig):
    return [t.get("fillcolor") for t in fig["data"] if t.get("fill") == "toself"]


def test_ms1_patch_equals_full_render(gui):
    _, _, cb, s0 = gui
    ms1 = cb["update_ms1_figure"]

    fig, _, key = ms1(s0, "linear", None)
    fig = _jsonify(fig)
    # leucine 4.7-5.3 overlaps isoleucine 5.2-5.9 -> red; norleucine far away -> blue
    assert _iso_colors(fig) == ["rgba(255,96,96,0.28)", "rgba(150,205,255,0.28)"]

    moves = [
        (4.75, 5.30),  # small nudge
        (4.60, 5.10),  # no longer overlaps isoleucine -> its box turns blue
        (4.20, 5.00),  # expanded window shifts -> MS2 marker set and y-range change
        (4.70, 5.60),  # overlap again -> red
    ]
    prev_state = s0
    markers_seen = set()
    for rt_min, rt_max in moves:
        st = _state(prev_state, rt_min=rt_min, rt_max=rt_max)
        out, _, key = ms1(st, "linear", key)
        assert isinstance(out, dash.Patch), f"expected a Patch for move to {(rt_min, rt_max)}"
        fig = _apply_patch(fig, out)

        full, _, _ = ms1(st, "linear", None)
        assert fig == _jsonify(full), f"patched figure diverged from full render at {(rt_min, rt_max)}"
        markers_seen.add(tuple(fig["data"][-1]["x"]))
        prev_state = st

    assert _iso_colors(_jsonify(ms1(_state(s0, rt_min=4.6, rt_max=5.1), "linear", None)[0]))[0] == "rgba(150,205,255,0.28)"
    assert len(markers_seen) > 1, "test should exercise a change in MS2 markers"


def test_ms1_patch_equals_full_render_log_scale(gui):
    _, _, cb, s0 = gui
    ms1 = cb["update_ms1_figure"]
    fig, _, key = ms1(s0, "log", None)
    st = _state(s0, rt_min=4.5, rt_max=5.2)
    out, _, _ = ms1(st, "log", key)
    assert isinstance(out, dash.Patch)
    assert _apply_patch(_jsonify(fig), out) == _jsonify(ms1(st, "log", None)[0])


def test_ms1_large_move_triggers_full_render(gui):
    _, _, cb, s0 = gui
    ms1 = cb["update_ms1_figure"]
    _, _, key = ms1(s0, "linear", None)
    out, _, _ = ms1(_state(s0, rt_min=1.0, rt_max=2.0), "linear", key)
    assert isinstance(out, dict) and "data" in out


def test_note_change_does_not_touch_ms1(gui):
    obj, _, cb, s0 = gui
    ms1 = cb["update_ms1_figure"]
    _, _, key = ms1(s0, "linear", None)
    st = _state(s0, ms1_note=obj.notes["ms1_notes"][1], other_note=[obj.notes["other_notes"][0]])
    with pytest.raises(dash.exceptions.PreventUpdate):
        ms1(st, "linear", key)


def test_highlight_click_roundtrip(gui):
    obj, _, cb, s0 = gui
    fig = _jsonify(cb["update_ms1_figure"](s0, "linear", None)[0])
    hover = next(t for t in fig["data"] if t.get("hovertemplate", "").startswith("File:"))
    idx = next(i for i, c in enumerate(hover["customdata"]) if c is not None and c >= 0)
    click = {"points": [{"customdata": hover["customdata"][idx]}]}

    new_state, cleared = cb["toggle_ms1_highlight"](click, s0)
    assert cleared is None
    assert len(new_state["highlighted_files"]) == 1
    fn = new_state["highlighted_files"][0]
    assert fn in set(obj.experimental_data.ms1_df["filename"])
    assert hover["text"][idx] in fn

    # Highlight -> full render with a thick trace
    out, _, _ = cb["update_ms1_figure"](new_state, "linear", None)
    assert any(t.get("line", {}).get("width") == 5.0 for t in _jsonify(out)["data"])

    # Clicking again un-highlights
    again, _ = cb["toggle_ms1_highlight"](click, new_state)
    assert again["highlighted_files"] == []


def _expected_ms2_count(obj, uid, lo, hi, top_n=20):
    df = obj.experimental_data.ms2_df
    sub = df[(df["mz_rt_uid"] == uid) & (df["scan_rt"] >= lo) & (df["scan_rt"] <= hi)]
    return min(top_n, len(sub))


def test_ms2_scan_counts_follow_rt_window(gui):
    obj, _, cb, s0 = gui
    uid = obj.experimental_data.curation_df.iloc[s0["compound_idx"]]["mz_rt_uid"]
    for lo, hi in [(4.7, 5.3), (4.9, 5.0), (3.0, 7.0), (4.7, 5.3)]:
        st = _state(s0, rt_min=lo, rt_max=hi)
        status = cb["update_status"](st)
        n = _expected_ms2_count(obj, uid, lo, hi)
        assert status[3] == (f"MS2 scan 1 / {n}" if n else "No MS2 scans")


def test_ms2_redraws_when_window_gains_scans(gui):
    _, _, cb, s0 = gui
    ms2 = cb["update_ms2_figure"]
    narrow = _state(s0, rt_min=4.95, rt_max=5.0)
    wide = _state(s0, rt_min=3.5, rt_max=6.5)
    _, key = ms2(narrow, "linear", None)
    with pytest.raises(dash.exceptions.PreventUpdate):
        ms2(_state(narrow), "linear", key)  # nothing changed for MS2
    fig, key2 = ms2(wide, "linear", key)
    assert key2 != key and isinstance(fig, dict) and fig["data"]


def test_ms2_figure_log_and_linear(gui):
    _, _, cb, s0 = gui
    for scale in ("linear", "log"):
        fig, _ = cb["update_ms2_figure"](s0, scale, None)
        js = _jsonify(fig)
        assert js["data"], scale
        assert "CoS" in js["layout"]["annotations"][-1]["text"]


def test_sync_controls_does_not_echo(gui):
    obj, _, cb, s0 = gui
    out = cb["sync_controls"](s0, s0["analyst_notes"], s0["ms1_note"], s0["ms2_note"], list(s0["other_note"]))
    assert out[0] is dash.no_update and out[2] is dash.no_update
    assert out[3] is dash.no_update and out[4] is dash.no_update

    other = obj.notes["ms1_notes"][1]
    out = cb["sync_controls"](_state(s0, ms1_note=other), s0["analyst_notes"], s0["ms1_note"], s0["ms2_note"], [])
    assert out[2] == other


def test_save_indicator(gui):
    obj, _, cb, s0 = gui
    assert cb["update_status"](s0)[5] == ""  # nothing saved yet -> blank
    edited = _state(s0, ms2_note=obj.notes["ms2_notes"][2])
    assert cb["update_status"](edited)[5] == ""  # pending edits are not announced
    saved = _state(s0, last_saved={"index": 1, "name": "leucine", "timestamp": "14:02:11"})
    assert "Saved [1] leucine at 14:02:11" in json.dumps(_jsonify(cb["update_status"](saved)[5]))
