"""Synthetic inputs for exercising the targeted Analysis GUI without a database.

``make_gui_obj`` returns an object shaped like ``AnalysisGUI`` after
``dbi.load_and_filter_for_gui`` has run: ``experimental_data.curation_df``,
``ms1_df`` and ``ms2_df`` plus the config/notes attributes the app factory
reads.  Nothing here touches HDF5 files or DuckDB.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from metatlas2.targeted import note_options as gno


def _notes(owner: str = "jgi") -> dict:
    ms2_opts, ms1_opts, other_opts = gno.get_notes_opts(owner=owner)
    ms1_options, ms1_hotkeys = gno.get_note_options_and_hotkeys({}, ms1_opts)
    ms2_options, ms2_hotkeys = gno.get_note_options_and_hotkeys({}, ms2_opts)
    other_options, other_hotkeys = gno.get_note_options_and_hotkeys({}, other_opts)
    return {
        "ms1_notes": ms1_options,
        "ms1_hotkeys": ms1_hotkeys,
        "ms2_notes": ms2_options,
        "ms2_hotkeys": ms2_hotkeys,
        "other_notes": other_options,
        "other_hotkeys": other_hotkeys,
        "ms1_key_to_label": {v: k for k, v in ms1_hotkeys.items()},
        "ms2_key_to_label": {v: k for k, v in ms2_hotkeys.items()},
        "other_key_to_label": {v: k for k, v in other_hotkeys.items()},
    }


def _filename(i: int, kind: str) -> str:
    # Production names have >= 12 underscore-separated fields; the GUI's short
    # name is fields[11:].
    head = "20240101_JGI_AK_XX_000000_TEST_final_QE-HF_HILICZ_USHXG01583_POS_MSMS"
    return f"/data/raw/{head}_{i}_{kind}_Rg80to1200-CE102040-sample-S1_Run{i}.h5"


def make_gui_obj(n_files: int = 40, n_points: int = 600, n_ms2: int = 60, seed: int = 0):
    """Three compounds; compounds 0 and 1 are isomers of each other, and
    compound 2 is an isomer of 0 whose window sits far away (non-overlapping)."""
    rng = np.random.default_rng(seed)
    compounds = [
        # label, uid, atlas_rt_peak, rt_min, rt_max, inchi
        ("leucine", "uid-leu", 5.0, 4.7, 5.3, "ROHFNLRQFUQHCH-YFKPBYRVSA-N"),
        ("isoleucine", "uid-ile", 5.6, 5.2, 5.9, "AGPKZVBTJJNPAG-WHFBIAKZSA-N"),
        ("norleucine", "uid-nle", 8.5, 8.2, 8.8, "LRQKBLKVPFOOQJ-YFKPBYRVSA-N"),
    ]
    rows = []
    for idx, (label, uid, peak, rmin, rmax, inchi) in enumerate(compounds):
        others = [c for c in compounds if c[1] != uid]
        isomers = [
            {"mz_rt_uid": o[1], "inchi_key": o[5], "label": o[0], "adduct": "[M+H]+",
             "rt": o[2], "mz": 132.1019}
            for o in others
        ]
        max_rt = np.linspace(0, 12, 300)
        max_int = 1e6 * np.exp(-((max_rt - peak) ** 2) / 0.02) + 1e4
        rows.append({
            "label": label, "adduct": "[M+H]+", "mz_rt_uid": uid, "inchi_key": inchi,
            "rt_min": rmin, "rt_max": rmax, "rt_peak": peak + 0.01,
            "initial_rt_min": rmin, "initial_rt_max": rmax,
            "atlas_rt_min": rmin, "atlas_rt_max": rmax, "atlas_rt_peak": peak,
            "atlas_mz": 132.1019, "mz": 132.1021, "rt_error": 0.01, "mz_error": 1.5,
            "suggested_rt_min": rmin + 0.05, "suggested_rt_max": rmax - 0.05,
            "ms1_notes": "keep", "ms2_notes": "", "other_notes": "",
            "analyst_notes": "", "identification_notes": f"notes for {label}",
            "isomers": json.dumps(isomers), "analysis_number": 0,
            "max_eic_rt": max_rt.tolist(), "max_eic_intensity": max_int.tolist(),
        })
    curation_df = pd.DataFrame(rows)

    ms1_rows, ms2_rows = [], []
    for label, uid, peak, rmin, rmax, _ in compounds:
        for f in range(n_files):
            kind = "QC" if f % 10 == 0 else ("InjBl" if f % 10 == 1 else "S1")
            rts = np.sort(rng.uniform(max(0, peak - 6), peak + 6, n_points))
            ints = 1e5 * (1 + f / n_files) * np.exp(-((rts - peak) ** 2) / 0.01) + rng.uniform(0, 1e3, n_points)
            ms1_rows.append({
                "mz_rt_uid": uid, "filename": _filename(f, kind),
                "spec_rts": rts.tolist(), "spec_ints": ints.tolist(),
                "spec_mzs": (132.1019 + rng.normal(0, 1e-4, n_points)).tolist(),
                "in_feature": ((rts >= rmin) & (rts <= rmax)).tolist(),
            })
        for s in range(n_ms2):
            scan_rt = float(rng.uniform(peak - 1.5, peak + 1.5))
            q_mz = sorted(rng.uniform(50, 130, 8).tolist())
            hits = [{
                "score": float(rng.uniform(0, 1)),
                "query_aligned": [q_mz, rng.uniform(0, 1e4, 8).tolist()],
                "ref_aligned": [q_mz, rng.uniform(0, 1e2, 8).tolist()],
                "fragment_colors": ["green" if k % 2 else "red" for k in range(8)],
                "ref_frags": 8, "matched_fragments": [1, 2, 3],
                "mz_theoretical": 132.1019, "ppm_error": 1.2, "ref_name": label,
            } for _ in range(2)]
            ms2_rows.append({
                "mz_rt_uid": uid, "scan_rt": scan_rt, "hits": hits,
                "collision_energy": 23.333, "filename": _filename(s % n_files, "S1"),
                "precursor_MZ": 132.102, "frag_mzs": q_mz, "frag_ints": [1.0] * 8,
            })

    obj = SimpleNamespace(
        experimental_data=SimpleNamespace(
            curation_df=curation_df,
            ms1_df=pd.DataFrame(ms1_rows),
            ms2_df=pd.DataFrame(ms2_rows),
        ),
        config=SimpleNamespace(gui_config={"gui_lcmsruns_colors": {"QC": "#1f77b4", "InjBl": "#d62728"}}),
        override_parameters={},
        project_name="20240101_JGI_AK_XX_TESTPROJ_final",
        chromatography="hilicz",
        polarity="positive",
        ta=SimpleNamespace(atlas_category="metabolites", atlas_name="test"),
        rt_alignment_number=0,
        analysis_number=0,
        notes=_notes(),
        paths={"project_db_path": "/nonexistent.duckdb"},
    )
    return obj
