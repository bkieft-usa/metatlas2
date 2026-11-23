"""Preview and deposit reviewed standards to the atlas sheet and main database.

Everything is generated from, and recorded in, the standards-annotation
project database:

1. ``dry_run=True`` builds atlas and MS/MS reference rows for a chosen RT
   alignment run, writes editable previews to
   ``annotation_run_<N>/rt_alignment_<M>/deposit_<K>/`` and records them as
   deposit K.
2. The analyst may edit, add or delete rows in the previews.
3. ``dry_run=False`` reads the previews back, validates them, records the
   differences, writes the edits back into the alignment run's deposit
   content (``sa_atlas_entries`` / ``sa_msms_entries``), snapshots the exact
   rows, then imports compounds, appends the atlas to Google Sheets and
   imports MS/MS references, recording each row's outcome.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors

import metatlas2.shared_tools.logging_config as lcf
from metatlas2.shared_tools.atlas_sheet import (
    ATLAS_COLUMNS,
    ATLAS_ROW_IDENTITY,
    ATLAS_SPREADSHEET,
    ATLAS_WORKSHEET,
    GOOGLE_SERVICE_ACCOUNT_FILE,
    open_centralized_atlas_worksheet,
)
from metatlas2.shared_tools.chem_info import inchi_or_smiles_to_molecule
from metatlas2.shared_tools.file_and_project_format import msms_reference_file_metadata
from metatlas2.shared_tools.logging_config import get_logger
from metatlas2.standards_annotation import standards_annotation_db as sadb
from metatlas2.standards_annotation.standards_annotation_db import (
    ATLAS_COLUMN_TYPES,
    ATLAS_LINK_COLUMNS,
    MSMS_COLUMN_TYPES,
    MSMS_COLUMNS,
    MSMS_LINK_COLUMNS,
)
from metatlas2.standards_annotation.standards_annotation_review import deposit_dir
from metatlas2.standards_annotation.standards_annotation_tools import STANDARD_CE_LABELS

logger = get_logger("deposit_ref_standards_atlas_and_msms_refs")

__all__ = [
    "ATLAS_COLUMNS", "MSMS_COLUMNS", "run_atlas_and_msms_deposit", "create_deposit_preview",
    "execute_deposit", "build_deposit_entries", "deposit_atlas_to_google_sheet",
]


def _file_names(run_name: str) -> dict[str, str]:
    return {
        "atlas": f"Atlas_Deposit_{run_name}.csv",
        "msms": f"MsmsRefs_Deposit_{run_name}.tab",
        "atlas_final": f"Atlas_Deposit_{run_name}.final.csv",
        "msms_final": f"MsmsRefs_Deposit_{run_name}.final.tab",
        "changes": "deposit_changes.csv",
    }


# ---------------------------------------------------------------------------
# Notebook entry point
# ---------------------------------------------------------------------------

def run_atlas_and_msms_deposit(
    config_path: str,
    annotation_run_id: int | None,
    alignment_run_id: int | None,
    dry_run: bool = True,
    force_add: bool = False,
    skip_atlas_upload: bool = False,
    skip_msms_refs_deposit: bool = False,
    upload_to_gdrive: bool = True,
    deposit_id: int | None = None,
) -> None:
    """Preview (``dry_run=True``) or deposit (``dry_run=False``) an alignment run.

    Args:
        config_path:            Standards-annotation YAML config.
        annotation_run_id:      Required annotation run that the alignment
                                run aligned. Must match the alignment run.
        alignment_run_id:       Required alignment run to deposit (see the
                                project history printed in the notebook).
        dry_run:                Write new editable previews only.
        force_add:              Append/import rows even if their identities
                                already exist in the sheet / ``msms_refs``.
        skip_atlas_upload:      Do not append to the Google Sheet.
        skip_msms_refs_deposit: Do not import MS/MS references.
        upload_to_gdrive:       Upload the deposit directory to Google Drive.
        deposit_id:             For ``dry_run=False``, the previewed deposit to
                                use (default: the alignment run's latest).
    """
    from metatlas2.shared_tools.load_tools import load_standards_config

    if annotation_run_id is None or alignment_run_id is None:
        raise ValueError(
            "Set both ANNOTATION_RUN_ID and ALIGNMENT_RUN_ID to the approved annotation run and its "
            "RT alignment run (see the project history above)."
        )
    config = load_standards_config(str(Path(config_path).resolve()))
    output_root = sadb.get_project_output_root(config)
    db_path = sadb.get_project_db_path(config)
    if not output_root.is_dir() or not Path(db_path).is_file():
        raise FileNotFoundError(f"Standards annotation project database not found: {db_path}")
    aligned_run = int(sadb.get_alignment_run(db_path, alignment_run_id)["annotation_run_id"])
    if aligned_run != int(annotation_run_id):
        raise ValueError(
            f"Alignment run {alignment_run_id} aligned annotation run {aligned_run}, not annotation run "
            f"{annotation_run_id}. Check ANNOTATION_RUN_ID and ALIGNMENT_RUN_ID against the project history."
        )

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with lcf.temporary_logging(
        log_level=logging.INFO,
        log_file=str(log_dir / "atlas_and_msms_deposit.log"),
        log_to_stdout=True,
        reconfigure_existing=False,
    ):
        if dry_run:
            if deposit_id is not None:
                raise ValueError("deposit_id selects an existing preview; omit it for a dry run.")
            deposit, directory = create_deposit_preview(
                db_path, alignment_run_id, config, upload_to_gdrive=upload_to_gdrive,
            )
            message = (f"Deposit {deposit} previewed for annotation run {annotation_run_id}, "
                       f"alignment run {alignment_run_id}. "
                       "Review or edit the files, then rerun with DRY_RUN = False.")
        else:
            deposit, directory = execute_deposit(
                db_path, alignment_run_id, config, deposit_id=deposit_id, force_add=force_add,
                skip_atlas_upload=skip_atlas_upload, skip_msms_refs_deposit=skip_msms_refs_deposit,
                upload_to_gdrive=upload_to_gdrive,
            )
            status = sadb.get_deposit(db_path, deposit)["status"]
            message = (f"Deposit {deposit} of annotation run {annotation_run_id}, "
                       f"alignment run {alignment_run_id}: {status}.")
    from IPython.display import HTML, display
    display(HTML(
        f"<div style='font-size: 20px; font-weight: bold;'>{message}</div><div>Files: {directory}</div>"
    ))


# ---------------------------------------------------------------------------
# Building rows from the project database
# ---------------------------------------------------------------------------

def _number(value: str | int | float, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column} in row {row_number}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {column} in row {row_number}: {value!r}")
    return result


def _spectrum(value: Any, ce: str, row_number: int) -> list[list[float]] | None:
    """Parse and validate ``[[m/z...], [intensity...]]``; ``None`` if empty."""
    if value is None or (isinstance(value, str) and not value.strip()):
        logger.warning("No %s spectrum for row %s", ce, row_number)
        return None
    spectrum = value
    if isinstance(value, str):
        try:
            spectrum = json.loads(value)
        except json.JSONDecodeError:
            try:
                spectrum = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid {ce} spectrum in row {row_number}") from exc
    if (not isinstance(spectrum, (list, tuple)) or len(spectrum) != 2
            or not all(isinstance(part, (list, tuple)) for part in spectrum)
            or len(spectrum[0]) != len(spectrum[1])):
        raise ValueError(f"Invalid {ce} spectrum in row {row_number}")
    if not spectrum[0]:
        logger.warning("Empty %s spectrum for row %s", ce, row_number)
        return None
    return [[_number(v, f"{ce} spectrum", row_number) for v in part] for part in spectrum]


def _structure_info(smiles: str, inchi: str, label: str) -> tuple[str, float | None]:
    if not (smiles or inchi):
        logger.warning("No structure for %s; formula and exact mass are blank", label)
        return "", None
    mol = inchi_or_smiles_to_molecule(smiles or inchi)
    return rdMolDescriptors.CalcMolFormula(mol), Descriptors.ExactMolWt(mol)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def build_deposit_entries(
    source: pd.DataFrame, chem_source: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build atlas and MS/MS reference rows from :func:`load_deposit_source` rows.

    Returns:
        ``(atlas, msms)``: atlas has ``sa_peak_uid`` + ATLAS_COLUMNS; msms has
        ``sa_peak_uid``, ``sa_ce_label``, ``sa_measured_collision_energy`` +
        MSMS_COLUMNS.
    """
    date = datetime.now().strftime("%Y%m%d")
    atlas_rows, msms_rows = [], []
    for row_number, row in enumerate(source.to_dict("records"), start=1):
        label = _text(row["peak_label"])
        smiles, inchi = _text(row.get("smiles")), _text(row.get("inchi"))
        formula, exact_mass = _structure_info(smiles, inchi, label)
        polarity = {"POS": "positive", "NEG": "negative"}.get(_text(row["polarity"]).upper())
        if polarity is None:
            raise ValueError(f"Invalid polarity for {label}: {row['polarity']!r}")
        notes = "; ".join(
            note.strip() for note in (_text(row.get("analyst_notes")), _text(row.get("annotation_notes")))
            if note.strip()
        )
        atlas_rows.append({
            "sa_peak_uid": row["peak_uid"],
            "label": label, "compound_name": _text(row["compound_name"]),
            "inchi_key": _text(row["inchi_key"]), "smiles": smiles, "inchi": inchi, "formula": formula,
            "adduct": _text(row["adduct"]),
            "mz": _number(row["mz_theoretical"], "theoretical m/z", row_number),
            "rt_peak": _number(row["rt_peak_aligned"], "aligned RT peak", row_number),
            "rt_min": _number(row["rt_min_aligned"], "aligned RT min", row_number),
            "rt_max": _number(row["rt_max_aligned"], "aligned RT max", row_number),
            "mz_tolerance": 5.0, "chromatography": _text(row["chromatography"]), "polarity": polarity,
            "identification_notes": notes, "chem_source": chem_source,
            "lcmsrun_source": Path(_text(row["filename"])).name,
            "date_created": date, "date_modified": date, "tags": _text(row["tags"]),
        })
        scans_by_ce: dict[str, dict] = {}
        for scan in row.get("ms2_scans") or []:
            filename = _text(scan.get("filename"))
            instrument, ramp = msms_reference_file_metadata(filename)
            ce = "CE" + ramp.removeprefix("ramp-").removesuffix("norm")
            if ce in scans_by_ce:
                raise ValueError(f"More than one {ce} MS2 scan for {label}")
            scans_by_ce[ce] = {**scan, "instrument": instrument, "ramp": ramp}
        for ce in STANDARD_CE_LABELS:
            scan = scans_by_ce.get(ce)
            if scan is None:
                logger.warning("No %s spectrum for %s", ce, label)
                continue
            spectrum = _spectrum([list(scan.get("frag_mzs") or []), list(scan.get("frag_ints") or [])],
                                 ce, row_number)
            if spectrum is None:
                continue
            precursor = scan.get("precursor_mz")
            energy = scan.get("collision_energy")
            precursor = None if precursor is None or pd.isna(precursor) else float(precursor)
            energy = None if energy is None or pd.isna(energy) else float(energy)
            if precursor is None or energy is None:
                logger.warning("Missing precursor m/z or collision energy for %s %s", ce, label)
            msms_rows.append({
                "sa_peak_uid": row["peak_uid"], "sa_ce_label": ce, "sa_measured_collision_energy": energy,
                "database": "metatlas", "id": uuid.uuid4().hex, "name": label,
                "spectrum": json.dumps(spectrum), "decimal": 4, "precursor_mz": precursor,
                "polarity": polarity, "adduct": _text(row["adduct"]), "fragmentation_method": "HCD",
                "collision_energy": scan["ramp"], "instrument": scan["instrument"],
                "instrument_type": "Orbitrap", "formula": formula, "exact_mass": exact_mass,
                "inchi_key": _text(row["inchi_key"]), "inchi": inchi, "smiles": smiles,
            })
    atlas = pd.DataFrame(atlas_rows, columns=["sa_peak_uid", *ATLAS_COLUMNS])
    msms = pd.DataFrame(
        msms_rows, columns=["sa_peak_uid", "sa_ce_label", "sa_measured_collision_energy", *MSMS_COLUMNS],
    )
    return atlas, msms


# ---------------------------------------------------------------------------
# Preview files
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_layout(kind: str) -> list[str]:
    return ([*ATLAS_COLUMNS, *ATLAS_LINK_COLUMNS] if kind == "atlas"
            else [*MSMS_COLUMNS, *MSMS_LINK_COLUMNS])


def _write_table(path: Path, frame: pd.DataFrame, columns: list[str], separator: str) -> None:
    out = frame.reindex(columns=columns).copy()
    if "sa_entry_id" in out:
        out["sa_entry_id"] = out["sa_entry_id"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    if "decimal" in out:
        out["decimal"] = out["decimal"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    with path.open("x", encoding="utf-8", newline="") as handle:
        out.to_csv(handle, sep=separator, index=False)


def create_deposit_preview(
    db_path: str, alignment_run_id: int, config: dict[str, Any], *, upload_to_gdrive: bool = True,
) -> tuple[int, Path]:
    """Write editable deposit previews for an alignment run and record them.

    On the first preview of an alignment run, its deposit content is
    generated from the database (aligned peaks, structures, the annotation
    run's notes and selected MS2 scans). Later previews reuse that content,
    including edits written back by earlier deposits.

    Returns:
        The new deposit ID and its directory.
    """
    alignment = sadb.get_alignment_run(db_path, alignment_run_id)
    latest = sadb.latest_deposit_for_alignment(db_path, alignment_run_id)
    if latest is not None and latest["status"] == "in_progress":
        raise ValueError(
            f"Deposit {latest['deposit_id']} of alignment run {alignment_run_id} is partially complete. "
            "Retry it with DRY_RUN = False before creating a new preview."
        )
    deposits = sadb.list_deposits(db_path)
    done = deposits.loc[deposits["status"] == "deposited", "deposit_id"].tolist()
    if done:
        logger.warning(
            "Deposit(s) %s of this project were already deposited; depositing again may duplicate "
            "sheet rows and MS/MS references.", done,
        )
    chem_source = str(config["project"].get("chem_source", "") or "")
    if not sadb.has_deposit_entries(db_path, alignment_run_id):
        source = sadb.load_deposit_source(db_path, alignment_run_id)
        atlas, msms = build_deposit_entries(source, chem_source=chem_source)
        sadb.save_generated_entries(db_path, alignment_run_id, atlas, msms)
        logger.info(f"Generated deposit content for alignment run {alignment_run_id} from the project database")
    else:
        logger.info(f"Using stored deposit content of alignment run {alignment_run_id}, including earlier edits")
    atlas, msms = sadb.load_deposit_entries(db_path, alignment_run_id)
    if atlas.empty:
        raise ValueError(f"Alignment run {alignment_run_id} has no atlas rows to deposit.")

    deposit_id = sadb.next_deposit_id(db_path)
    directory = deposit_dir(Path(alignment["output_dir"]), deposit_id)
    directory.mkdir(parents=False, exist_ok=False)
    names = _file_names(_run_name(config))
    try:
        _write_table(directory / names["atlas"], atlas, _file_layout("atlas"), ",")
        _write_table(directory / names["msms"], msms, _file_layout("msms"), "\t")
        sadb.record_deposit_preview(
            db_path, deposit_id=deposit_id, alignment_run_id=alignment_run_id,
            output_dir=directory, chem_source=chem_source,
            atlas=atlas, msms=msms,
            atlas_sha256=_sha256(directory / names["atlas"]),
            msms_sha256=_sha256(directory / names["msms"]),
        )
    except Exception:
        for name in ("atlas", "msms"):
            (directory / names[name]).unlink(missing_ok=True)
        directory.rmdir()
        raise
    logger.info(f"Prepared {len(atlas)} atlas rows and {len(msms)} MS/MS spectra as deposit {deposit_id}")
    logger.info(f"Atlas preview: {directory / names['atlas']}")
    logger.info(f"MS/MS preview: {directory / names['msms']}")
    if upload_to_gdrive:
        _upload(db_path, config, deposit_id, directory)
    return deposit_id, directory


def _run_name(config: dict[str, Any]) -> str:
    return config["project"]["standards_run_name"]


def _upload(db_path: str, config: dict[str, Any], deposit_id: int, directory: Path) -> None:
    from metatlas2.shared_tools.gdrive_upload import upload_standards_outputs
    url = upload_standards_outputs(config, directory.parent.parent, [directory])
    sadb.set_deposit_gdrive_url(db_path, deposit_id, url)


# ---------------------------------------------------------------------------
# Reading previews back
# ---------------------------------------------------------------------------

def _read_table(path: Path, kind: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Deposit preview not found: {path}. Run a dry run first.")
    frame = pd.read_csv(path, sep="," if kind == "atlas" else "\t", dtype=str, keep_default_na=False)
    expected = _file_layout(kind)
    missing = [c for c in expected if c not in frame.columns]
    unknown = [c for c in frame.columns if c not in expected]
    if missing or unknown:
        raise ValueError(f"{path.name}: missing columns {missing}; unexpected columns {unknown}")
    frame = frame[expected]
    blank = frame.apply(lambda row: not any(v.strip() for v in row), axis=1)
    return frame.loc[~blank].reset_index(drop=True)


def _optional_number(value: str, column: str, row_number: int) -> float | None:
    return _number(value, column, row_number) if value.strip() else None


def _entry_id(value: str, path: Path, row_number: int) -> float:
    if not value.strip():
        return math.nan
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{path.name} row {row_number}: invalid sa_entry_id {value!r}") from exc


def _parse_atlas(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for row_number, raw in enumerate(frame.to_dict("records"), start=2):
        row = {c: raw[c].strip() for c in raw}
        for column in ("label", "inchi_key", "adduct", "chromatography"):
            if not row[column]:
                raise ValueError(f"{path.name} row {row_number}: {column} is required")
        if row["polarity"] not in {"positive", "negative"}:
            raise ValueError(f"{path.name} row {row_number}: polarity must be positive or negative")
        for column, sql_type in ATLAS_COLUMN_TYPES.items():
            if sql_type == "DOUBLE":
                row[column] = _number(row[column], column, row_number)
        rmin, peak, rmax = row["rt_min"], row["rt_peak"], row["rt_max"]
        if not rmin <= peak <= rmax:
            raise ValueError(f"{path.name} row {row_number}: require rt_min <= rt_peak <= rt_max")
        row["sa_entry_id"] = _entry_id(row["sa_entry_id"], path, row_number)
        row["sa_peak_uid"] = row["sa_peak_uid"] or None
        rows.append(row)
    return pd.DataFrame(rows, columns=_file_layout("atlas"))


def _parse_msms(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for row_number, raw in enumerate(frame.to_dict("records"), start=2):
        row = {c: raw[c].strip() for c in raw}
        if not row["inchi_key"]:
            raise ValueError(f"{path.name} row {row_number}: inchi_key is required")
        if row["polarity"] not in {"positive", "negative"}:
            raise ValueError(f"{path.name} row {row_number}: polarity must be positive or negative")
        if not row["collision_energy"]:
            raise ValueError(f"{path.name} row {row_number}: collision_energy is required")
        spectrum = _spectrum(row["spectrum"], row["sa_ce_label"] or "MS/MS", row_number)
        if spectrum is None:
            raise ValueError(f"{path.name} row {row_number}: spectrum is empty; delete the row instead")
        row["spectrum"] = json.dumps(spectrum)
        if not row["id"]:
            row["id"] = uuid.uuid4().hex
            logger.info("%s row %s: assigned new id %s", path.name, row_number, row["id"])
        row["decimal"] = int(_number(row["decimal"] or "4", "decimal", row_number))
        row["precursor_mz"] = _optional_number(row["precursor_mz"], "precursor_mz", row_number)
        row["exact_mass"] = _optional_number(row["exact_mass"], "exact_mass", row_number)
        row["sa_measured_collision_energy"] = _optional_number(
            row["sa_measured_collision_energy"], "sa_measured_collision_energy", row_number,
        )
        row["sa_entry_id"] = _entry_id(row["sa_entry_id"], path, row_number)
        for column in ("sa_peak_uid", "sa_ce_label"):
            row[column] = row[column] or None
        rows.append(row)
    parsed = pd.DataFrame(rows, columns=_file_layout("msms"))
    duplicated = parsed["id"].duplicated()
    if duplicated.any():
        raise ValueError(f"{path.name}: duplicate id values {sorted(parsed.loc[duplicated, 'id'])}")
    return parsed


def _comparable(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value), 10)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _diff(
    kind: str, final: pd.DataFrame, generated: pd.DataFrame, path: Path,
) -> tuple[pd.DataFrame, list[dict], list[int]]:
    """Classify final rows and list column-level changes against the preview."""
    columns = [c for c in _file_layout(kind) if c != "sa_entry_id"]
    by_id = {int(r["sa_entry_id"]): r for r in generated.to_dict("records")}
    final = final.copy()
    status, changes, seen = [], [], set()
    for row_number, row in enumerate(final.to_dict("records"), start=2):
        entry = row["sa_entry_id"]
        if pd.isna(entry):
            status.append("added")
            continue
        entry = int(entry)
        if entry in seen:
            raise ValueError(f"{path.name} row {row_number}: sa_entry_id {entry} appears more than once")
        seen.add(entry)
        original = by_id.get(entry)
        if original is None:
            raise ValueError(
                f"{path.name} row {row_number}: sa_entry_id {entry} is not in this preview. "
                "Leave sa_entry_id blank for new rows."
            )
        edits = [
            {"file": kind, "sa_entry_id": entry, "change_type": "edited", "column_name": column,
             "old_value": _text(original.get(column)), "new_value": _text(row.get(column))}
            for column in columns if _comparable(original.get(column)) != _comparable(row.get(column))
        ]
        changes.extend(edits)
        status.append("edited" if edits else "unchanged")
    removed = sorted(set(by_id) - seen)
    changes.extend({"file": kind, "sa_entry_id": entry, "change_type": "removed",
                    "column_name": None, "old_value": None, "new_value": None} for entry in removed)
    final["change_status"] = status
    return final, changes, removed


# ---------------------------------------------------------------------------
# Deposit
# ---------------------------------------------------------------------------

def _resolve_deposit(db_path: str, alignment_run_id: int, deposit_id: int | None) -> dict[str, Any]:
    if deposit_id is not None:
        deposit = sadb.get_deposit(db_path, deposit_id)
        if int(deposit["alignment_run_id"]) != int(alignment_run_id):
            raise ValueError(f"Deposit {deposit_id} belongs to alignment run {deposit['alignment_run_id']}.")
    else:
        deposit = sadb.latest_deposit_for_alignment(db_path, alignment_run_id)
        if deposit is None:
            raise ValueError(
                f"No deposit preview exists for alignment run {alignment_run_id}; run with DRY_RUN = True first."
            )
    if deposit["status"] == "superseded":
        raise ValueError(f"Deposit {deposit['deposit_id']} was superseded by a newer preview.")
    return deposit


def execute_deposit(
    db_path: str,
    alignment_run_id: int,
    config: dict[str, Any],
    *,
    deposit_id: int | None = None,
    force_add: bool = False,
    skip_atlas_upload: bool = False,
    skip_msms_refs_deposit: bool = False,
    upload_to_gdrive: bool = True,
) -> tuple[int, Path]:
    """Deposit the (possibly edited) previews of a deposit.

    A deposit that already started (``in_progress``, or ``deposited`` with
    skipped stages) is resumed: completed stages are not repeated, and the
    preview files must be unchanged since the deposit started.

    Returns:
        The deposit ID and its directory.
    """
    deposit = _resolve_deposit(db_path, alignment_run_id, deposit_id)
    deposit_id = int(deposit["deposit_id"])
    directory = Path(deposit["output_dir"])
    names = _file_names(_run_name(config))
    atlas_path, msms_path = directory / names["atlas"], directory / names["msms"]
    atlas_sha, msms_sha = _sha256(atlas_path), _sha256(msms_path)

    if deposit["status"] == "previewed":
        _start_deposit(db_path, deposit, directory, names, atlas_sha, msms_sha, force_add)
    else:
        if (atlas_sha, msms_sha) != (deposit["atlas_final_sha256"], deposit["msms_final_sha256"]):
            raise ValueError(
                f"Deposit {deposit_id} already started and its preview files changed since. "
                "Restore them, or create a new preview with DRY_RUN = True."
            )
        reopen = [stage for stage, skip in (("atlas", skip_atlas_upload), ("msms", skip_msms_refs_deposit))
                  if deposit[f"{stage}_stage"] == "skipped" and not skip]
        if deposit["status"] == "deposited" and not reopen:
            raise ValueError(
                f"Deposit {deposit_id} is already deposited. Create a new preview with DRY_RUN = True "
                "to deposit again."
            )
        sadb.reopen_skipped_stages(db_path, deposit_id, reopen)
        logger.info(f"Resuming deposit {deposit_id}")

    sadb.load_deposit_changes(db_path, deposit_id).to_csv(directory / names["changes"], index=False)
    _run_stages(db_path, deposit_id, directory, names, force_add=force_add,
                skip_atlas_upload=skip_atlas_upload, skip_msms_refs_deposit=skip_msms_refs_deposit)
    if upload_to_gdrive:
        _upload(db_path, config, deposit_id, directory)
    return deposit_id, directory


def _start_deposit(
    db_path: str, deposit: dict, directory: Path, names: dict[str, str],
    atlas_sha: str, msms_sha: str, force_add: bool,
) -> None:
    deposit_id = int(deposit["deposit_id"])
    atlas_path, msms_path = directory / names["atlas"], directory / names["msms"]
    atlas = _parse_atlas(_read_table(atlas_path, "atlas"), atlas_path)
    msms = _parse_msms(_read_table(msms_path, "msms"), msms_path)
    if atlas.empty:
        raise ValueError(f"{atlas_path.name} contains no rows.")
    generated_atlas, generated_msms = sadb.load_deposit_rows(db_path, deposit_id, "generated")
    atlas, atlas_changes, atlas_removed = _diff("atlas", atlas, generated_atlas, atlas_path)
    msms, msms_changes, msms_removed = _diff("msms", msms, generated_msms, msms_path)
    changes = pd.DataFrame(
        atlas_changes + msms_changes,
        columns=["file", "sa_entry_id", "change_type", "column_name", "old_value", "new_value"],
    )
    counts = {kind: frame["change_status"].value_counts().to_dict() for kind, frame in
              (("atlas", atlas), ("msms", msms))}
    logger.info(
        f"Deposit {deposit_id} preview changes — atlas: {counts['atlas']}, removed {len(atlas_removed)}; "
        f"MS/MS: {counts['msms']}, removed {len(msms_removed)}"
    )
    # Exact upload/import files (no sa_* columns), written before the
    # database records the deposit as started so a resume always has them.
    for key, frame, columns, separator in (
        ("atlas_final", atlas, ATLAS_COLUMNS, ","),
        ("msms_final", msms, MSMS_COLUMNS, "\t"),
    ):
        path = directory / names[key]
        path.unlink(missing_ok=True)
        _write_table(path, frame, columns, separator)
    sadb.commit_deposit_final(
        db_path, deposit_id, atlas=atlas, msms=msms, changes=changes,
        removed={"atlas": atlas_removed, "msms": msms_removed},
        atlas_sha256=atlas_sha, msms_sha256=msms_sha, force_add=force_add,
    )


def _final_frames(db_path: str, deposit_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    atlas, msms = sadb.load_deposit_rows(db_path, deposit_id, "final")
    atlas_out = atlas[ATLAS_COLUMNS].copy()
    for column, sql_type in ATLAS_COLUMN_TYPES.items():
        atlas_out[column] = (atlas_out[column].astype(float) if sql_type == "DOUBLE"
                             else atlas_out[column].fillna("").astype(str))
    return atlas_out, msms


def _run_stages(
    db_path: str, deposit_id: int, directory: Path, names: dict[str, str], *,
    force_add: bool, skip_atlas_upload: bool, skip_msms_refs_deposit: bool,
) -> None:
    from metatlas2.shared_tools.database_interact import get_compound_uids_by_inchi_keys
    from metatlas2.shared_tools.paths import set_up_paths
    from metatlas2.shared_tools.workflow_objects import NewMsmsRefsConfig, add_compounds_from_files

    deposit = sadb.get_deposit(db_path, deposit_id)
    atlas, msms = _final_frames(db_path, deposit_id)
    run_atlas = deposit["atlas_stage"] == "pending" and not skip_atlas_upload
    run_msms = deposit["msms_stage"] == "pending" and not skip_msms_refs_deposit
    atlas_final, msms_final = directory / names["atlas_final"], directory / names["msms_final"]

    paths = None
    if (run_atlas or run_msms) and deposit["compounds_stage"] != "done":
        paths = set_up_paths(config=None)
        keys = atlas["inchi_key"].fillna("")
        logger.info("Importing compounds from %s", atlas_final)
        add_compounds_from_files([str(atlas_final)], paths)
        missing = set(keys) - set(get_compound_uids_by_inchi_keys(paths["main_db_path"], keys.unique().tolist()))
        if missing:
            raise RuntimeError(f"Compound import did not add InChIKeys: {sorted(missing)}")
        sadb.mark_deposit_stage(db_path, deposit_id, "compounds", "done")

    if deposit["atlas_stage"] == "pending":
        if skip_atlas_upload:
            logger.info("Skipping atlas upload")
            sadb.mark_deposit_stage(db_path, deposit_id, "atlas", "skipped",
                                    outcomes=["skipped_by_flag"] * len(atlas))
        else:
            logger.info("Appending atlas rows to Google Sheets")
            outcomes = deposit_atlas_to_google_sheet(atlas, force_add=force_add)
            sadb.mark_deposit_stage(db_path, deposit_id, "atlas", "done", outcomes=outcomes)

    if deposit["msms_stage"] == "pending":
        if skip_msms_refs_deposit:
            logger.info("Skipping MS/MS reference deposit")
            sadb.mark_deposit_stage(db_path, deposit_id, "msms", "skipped",
                                    outcomes=["skipped_by_flag"] * len(msms))
        else:
            paths = paths or set_up_paths(config=None)
            outcomes, ref_uids = _import_msms(msms, msms_final, paths["main_db_path"], force_add,
                                              NewMsmsRefsConfig)
            sadb.mark_deposit_stage(db_path, deposit_id, "msms", "done", outcomes=outcomes, ref_uids=ref_uids)
    if not (run_atlas or run_msms):
        if deposit["compounds_stage"] == "pending":
            sadb.mark_deposit_stage(db_path, deposit_id, "compounds", "skipped")
        logger.info("Atlas upload and MS/MS reference deposit both skipped")


def _import_msms(
    msms: pd.DataFrame, path: Path, main_db_path: str, force_add: bool, config_class,
) -> tuple[list[str], list[str | None]]:
    """Import MS/MS references and return each row's outcome and ref_uid."""
    if msms.empty:
        return [], []
    ids = msms["id"].tolist()
    pairs = list(zip(msms["inchi_key"].fillna(""), msms["adduct"].fillna("")))
    existing_pairs: set[tuple[str, str]] = set()
    with duckdb.connect(str(main_db_path), read_only=True) as conn:
        existing_pairs = {
            (key or "", adduct or "") for key, adduct in conn.execute(
                "SELECT DISTINCT inchi_key, COALESCE(TRIM(adduct), '') FROM msms_refs"
            ).fetchall()
        }
    energies = {
        ref_id: (None if pd.isna(energy) else float(energy))
        for ref_id, energy in zip(ids, msms["sa_measured_collision_energy"])
    }
    logger.info("Importing MS/MS references from %s into %s", path, main_db_path)
    config_class(entries=[{"path": str(path), "collision_energy_by_ref_id": energies}]).execute(
        skip_existing_inchi_adduct=not force_add,
    )
    placeholders = ", ".join("?" for _ in ids)
    with duckdb.connect(str(main_db_path), read_only=True) as conn:
        inserted = dict(conn.execute(
            f"SELECT ref_id, ref_uid FROM msms_refs WHERE ref_id IN ({placeholders})", ids,
        ).fetchall())
    outcomes, ref_uids = [], []
    for ref_id, pair in zip(ids, pairs):
        ref_uids.append(inserted.get(ref_id))
        if ref_id in inserted:
            outcomes.append("inserted")
        elif not force_add and pair in existing_pairs:
            outcomes.append("skipped_existing")
        else:
            outcomes.append("filtered")
    logger.info(
        "MS/MS references: %s inserted, %s skipped (existing InChIKey+adduct), %s filtered by the importer",
        outcomes.count("inserted"), outcomes.count("skipped_existing"), outcomes.count("filtered"),
    )
    return outcomes, ref_uids


def deposit_atlas_to_google_sheet(atlas: pd.DataFrame, force_add: bool = False) -> list[str]:
    """Append atlas rows to complete_atlas and sort its data by rt_peak.

    An append and the subsequent sort are separate operations. If the sort
    fails, the rows have already been appended and must not be appended again.

    Returns:
        One outcome per input row: ``appended`` or ``skipped_duplicate``.
    """
    if atlas.empty:
        raise ValueError("Cannot deposit an empty atlas.")
    if atlas.columns.tolist() != ATLAS_COLUMNS:
        raise ValueError("Atlas columns do not match the expected atlas schema.")

    spreadsheet, sheet = open_centralized_atlas_worksheet(
        spreadsheet_id=ATLAS_SPREADSHEET,
        worksheet_name=ATLAS_WORKSHEET,
        service_account_file=GOOGLE_SERVICE_ACCOUNT_FILE,
    )
    if sheet.row_values(1) != ATLAS_COLUMNS:
        raise ValueError("Google Sheet header does not match the atlas columns; nothing was appended.")

    outcomes = ["appended"] * len(atlas)
    if not force_add:
        existing_rows = sheet.get_all_values()[1:]
        column_indices = [ATLAS_COLUMNS.index(column) for column in ATLAS_ROW_IDENTITY]
        existing_keys = {
            tuple((row + [""] * len(ATLAS_COLUMNS))[index].strip() for index in column_indices)
            for row in existing_rows
            if any(value.strip() for value in row)
        }
        input_keys = atlas[ATLAS_ROW_IDENTITY].fillna("").astype(str).apply(
            lambda row: tuple(value.strip() for value in row), axis=1
        )
        duplicates = input_keys.isin(existing_keys) | input_keys.duplicated(keep="first")
        for identity in sorted(set(input_keys[duplicates])):
            logger.warning("Skipping duplicate centralized-atlas row identity: %s", identity)
        outcomes = ["skipped_duplicate" if dup else "appended" for dup in duplicates.tolist()]
        atlas = atlas.loc[~duplicates]
        if atlas.empty:
            logger.info("No new atlas rows to append.")
            return outcomes

    sheet.append_rows(atlas.values.tolist(), value_input_option="RAW")
    logger.info("Appended %s atlas rows; if sorting fails, do not append them again.", len(atlas))
    spreadsheet.batch_update({
        "requests": [{
            "sortRange": {
                "range": {"sheetId": sheet.id, "startRowIndex": 1},
                "sortSpecs": [{"dimensionIndex": 8, "sortOrder": "ASCENDING"}],
            },
        }],
    })
    logger.info(f"Successfully appended and sorted {len(atlas)} atlas rows in {ATLAS_SPREADSHEET}.")
    return outcomes
