"""Builders for synthetic untargeted-pipeline test data (no network, no Slurm)."""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from metatlas2.untargeted.config import load_untargeted_config

PROJECT_JGI = "20260101_JGI_AB_123456_Test_final_EXP120B_C18_USDAY00001"
PROJECT_EB = "20260102_EB_CD_107000-001_Exu_20251201_IQX_HILICZ_USHXG00001"
PLACEHOLDER = "/Users/bpb/Downloads/mzmine_outputs"

# (sample group, count) per polarity
DEFAULT_GROUPS = (("ExCtrl", 2), ("TxCtrl", 1), ("Sample", 3), ("QC", 1), ("InjBl", 1), ("ISTD", 1))


def mzml_name(project: str, pol: str, group: str, idx: int) -> str:
    """16-field LC-MS run file name."""
    return f"{project}_{pol}_MS2_{idx}_{group}_{idx % 3 + 1}_Rg80to1200-CE102040norm_Run{idx}.mzML"


def make_raw_project(
    raw_dir: Path,
    subdir: str,
    project: str,
    polarities: tuple[str, ...] = ("POS", "NEG"),
    groups=DEFAULT_GROUPS,
    extra_names: tuple[str, ...] = (),
) -> list[str]:
    """Create empty mzML files under ``raw_dir/subdir/project``; return their basenames."""
    pdir = raw_dir / subdir / project
    pdir.mkdir(parents=True, exist_ok=True)
    names = []
    idx = 1
    for pol in polarities:
        for group, count in groups:
            for _ in range(count):
                names.append(mzml_name(project, pol, group, idx))
                idx += 1
    names.extend(extra_names)
    for n in names:
        (pdir / n).write_text("")
    return names


def lcmsruns_df(project: str, names: list[str], raw_dir: Path, subdir: str, timeepoch: float) -> pd.DataFrame:
    """Rows shaped like LIMS ``lcmsrun_plus``."""
    rows = []
    for n in names:
        fields = n.split("_")
        rows.append({
            "mzml_file": str(raw_dir / subdir / project / n),
            "mzml_file_container": "x",
            "sample_group": fields[12] if len(fields) > 12 else None,
            "parent_dir": project,
            "chromatography": fields[7],
            "instrument": fields[6],
            "timeepoch": timeepoch,
            "polarity": fields[9] if len(fields) > 9 else None,
        })
    return pd.DataFrame(rows)


def write_templates(batch_dir: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    for name in ("IQX-mzmine-3.7.2-batchparams.xml", "mzmine-3.7.2-batchparams.xml"):
        (batch_dir / name).write_text(
            f"<batch template='{name}'><out>{PLACEHOLDER}_peak-height.csv</out>"
            f"<mgf>{PLACEHOLDER}.mgf</mgf></batch>\n")


def make_config(tmp_path: Path, **section_overrides) -> dict:
    """Config with every path under *tmp_path* (directories created)."""
    raw = tmp_path / "raw_data"
    out = tmp_path / "untargeted_tasks"
    dl = tmp_path / "untargeted_outputs"
    batch = tmp_path / "batch_files"
    for d in (raw, out, dl):
        d.mkdir(parents=True, exist_ok=True)
    write_templates(batch)
    overrides = {
        "paths": {"raw_data_dir": str(raw), "output_dir": str(out), "download_dir": str(dl),
                  "log_file": str(tmp_path / "pipeline.log"), "group": None},
        "mzmine": {"batch_file_dir": str(batch), "executable": "/opt/MZmine/bin/MZmine"},
        "gnps2": {"credentials_file": str(tmp_path / "gnps2.txt")},
        "lims": {"api_key_file": str(tmp_path / "labkey.txt")},
    }
    for section, values in section_overrides.items():
        overrides.setdefault(section, {}).update(values)
    return load_untargeted_config(None, copy.deepcopy(overrides))


def peak_height_table(project: str, pol: str, n_features: int = 6, rts=None) -> pd.DataFrame:
    """MZmine-style peak-height table with ExCtrl, TxCtrl and sample columns."""
    def col(group, idx):
        return f"{mzml_name(project, pol, group, idx)} Peak height"

    rts = rts or [0.2, 1.0, 2.0, 3.0, 5.0, 9.0][:n_features]
    data = {
        "row ID": list(range(1, n_features + 1)),
        "row m/z": [100.0 + i for i in range(n_features)],
        "row retention time": rts,
        col("ExCtrl", 1): [1000, 1000, 0, 5000, 100, 0][:n_features],
        col("TxCtrl", 2): [0, 0, 0, 0, 1e6, 0][:n_features],
        col("Sample", 3): [5000, 2000, 0, 6000, 2e5, 300][:n_features],
        col("Sample", 4): [4000, 0, 0, 1e6, 0, 0][:n_features],
    }
    df = pd.DataFrame(data)
    df["Unnamed: 7"] = float("nan")
    return df


def write_mgf(path: Path, n_spectra: int = 3, calibrant: float | None = None) -> None:
    lines = []
    for i in range(1, n_spectra + 1):
        lines += ["BEGIN IONS", f"FEATURE_ID={i}", "PEPMASS=150.1", "MSLEVEL=2", "100.0 50", "120.5 30"]
        if calibrant is not None:
            lines.append(f"{calibrant:.5f} 999")
        lines.append("END IONS")
    path.write_text("\n".join(lines) + "\n")


def simulate_mzmine_outputs(task, n_features: int = 6, mgf_spectra: int = 3) -> None:
    """Write the files a successful MZmine run leaves behind."""
    df = peak_height_table(task.project, task.polarity[:3].upper(), n_features)
    df.to_csv(task.path("peak_height"), index=False)
    df.to_csv(task.path("quant"), index=False)
    write_mgf(task.path("mgf"), mgf_spectra)
    task.path("mzmine_out").write_text("MZMINE IS DONE\n")
