"""Prepare reviewed standards and deposit atlas rows to Google Sheets."""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
from rdkit.Chem import Descriptors, rdMolDescriptors

from metatlas2.chem_info import inchi_or_smiles_to_molecule
from metatlas2.logging_config import get_logger
from metatlas2.standards_annotation_tools import STANDARD_CE_LABELS

logger = get_logger("deposit_ref_standards_atlas_and_msms_refs")

ATLAS_SPREADSHEET = "1915jP523jCWwtcFM5rv0X3AmbZzPhN1ijcoX5Dw0Svw"
GOOGLE_SERVICE_ACCOUNT_FILE = (
    "/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
)
ATLAS_COLUMNS = [
    "label", "compound_name", "inchi_key", "smiles", "inchi", "formula", "adduct",
    "mz", "rt_peak", "rt_min", "rt_max", "mz_tolerance", "chromatography", "polarity",
    "identification_notes", "lcmsrun_source", "date_created", "date_modified", "tags",
]
MSMS_COLUMNS = [
    "database", "id", "name", "spectrum", "decimal", "precursor_mz", "polarity",
    "adduct", "fragmentation_method", "collision_energy", "instrument", "instrument_type",
    "formula", "exact_mass", "inchi_key", "inchi", "smiles",
]
SUMMARY_COLUMNS = [
    "Compound Label", "Compound Name", "InChIKey", "Adduct", "Polarity",
    "Chromatography", "Source File", "Theoretical m/z", "Aligned RT peak",
    "Aligned RT min", "Aligned RT max", "Analyst Notes", "Annotation Notes",
    *[f"{ce} {field}" for ce in STANDARD_CE_LABELS
      for field in ("spectrum", "MS2 Precursor M/Z", "MS2 CE")],
]


def _number(value: str | int | float, column: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column} in aligned summary row {row_number}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {column} in aligned summary row {row_number}: {value!r}")
    return result


def _spectrum(value: str, ce: str, row_number: int) -> list[list[float]] | None:
    if not value:
        logger.warning("No %s spectrum for aligned summary row %s", ce, row_number)
        return None
    try:
        spectrum = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {ce} spectrum in aligned summary row {row_number}") from exc
    if (not isinstance(spectrum, list) or len(spectrum) != 2
            or not all(isinstance(part, list) for part in spectrum)
            or len(spectrum[0]) != len(spectrum[1])):
        raise ValueError(f"Invalid {ce} spectrum in aligned summary row {row_number}")
    if not spectrum[0]:
        logger.warning("Empty %s spectrum for aligned summary row %s", ce, row_number)
        return None
    return [[_number(value, f"{ce} spectrum", row_number) for value in part] for part in spectrum]


def prepare_deposit_data(
    summary_path: Path, db_path: str | Path, run_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build sheet and importer-format data from the explicitly reviewed summary.

    Only structures are loaded from the project database; peak data and spectra
    come from the aligned summary. The analyst is responsible for choosing the
    reviewed run and keeping its project data consistent.
    """
    summary = pd.read_csv(summary_path, dtype=str, keep_default_na=False)
    missing = set(SUMMARY_COLUMNS) - set(summary.columns)
    if missing:
        raise ValueError(f"Aligned summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError(f"Aligned summary contains no selected peaks: {summary_path}")
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Project database not found: {db_path}")
    with duckdb.connect(str(db_path), read_only=True) as conn:
        compounds = conn.execute(
            "SELECT compound_name, adduct, polarity, inchi_key, smiles, inchi FROM sa_compounds"
        ).df()
    compounds = compounds.fillna("")
    keys = ["compound_name", "adduct", "polarity", "inchi_key"]
    if compounds.duplicated(keys).any():
        raise ValueError("Project database contains ambiguous compound identities.")
    structures = compounds.set_index(keys)[["smiles", "inchi"]]
    date = datetime.now().strftime("%Y%m%d")
    atlas_rows = []
    msms_rows = []
    for row_number, row in enumerate(summary.to_dict("records"), start=2):
        key = (row["Compound Name"], row["Adduct"], row["Polarity"], row["InChIKey"])
        if key not in structures.index:
            raise ValueError(f"Compound identity in aligned summary row {row_number} not found in project database: {key}")
        structure = structures.loc[key]
        smiles, inchi = structure["smiles"], structure["inchi"]
        formula = ""
        exact_mass: float | str = ""
        if smiles or inchi:
            mol = inchi_or_smiles_to_molecule(smiles or inchi)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            exact_mass = Descriptors.ExactMolWt(mol)
        else:
            logger.warning("No structure for aligned summary row %s (%s); formula and exact mass are blank", row_number, key)
        polarity = {"POS": "positive", "NEG": "negative"}.get(row["Polarity"].upper())
        if polarity is None:
            raise ValueError(f"Invalid polarity in aligned summary row {row_number}: {row['Polarity']!r}")
        notes = "; ".join(note.strip() for note in (row["Analyst Notes"], row["Annotation Notes"]) if note.strip())
        tags = row.get("tags", "EMA")
        atlas_rows.append({
            "label": row["Compound Label"], "compound_name": row["Compound Name"],
            "inchi_key": row["InChIKey"], "smiles": smiles, "inchi": inchi, "formula": formula,
            "adduct": row["Adduct"], "mz": _number(row["Theoretical m/z"], "Theoretical m/z", row_number),
            "rt_peak": _number(row["Aligned RT peak"], "Aligned RT peak", row_number),
            "rt_min": _number(row["Aligned RT min"], "Aligned RT min", row_number),
            "rt_max": _number(row["Aligned RT max"], "Aligned RT max", row_number),
            "mz_tolerance": 5.0, "chromatography": row["Chromatography"], "polarity": polarity,
            "identification_notes": notes, "lcmsrun_source": row["Source File"],
            "date_created": str(date), "date_modified": str(date), "tags": tags,
        })
        for ce in STANDARD_CE_LABELS:
            spectrum = _spectrum(row[f"{ce} spectrum"], ce, row_number)
            if spectrum is None:
                continue
            precursor = row[f"{ce} MS2 Precursor M/Z"]
            collision_energy = row[f"{ce} MS2 CE"]
            if not precursor or not collision_energy:
                logger.warning("Missing precursor m/z or collision energy for %s in aligned summary row %s", ce, row_number)
            msms_rows.append({
                "database": "standards_annotation", "id": f"{run_name}_{len(msms_rows) + 1:04d}",
                "name": row["Compound Name"], "spectrum": json.dumps(spectrum), "decimal": 4,
                "precursor_mz": _number(precursor, f"{ce} MS2 Precursor M/Z", row_number) if precursor else "",
                "polarity": polarity, "adduct": row["Adduct"], "fragmentation_method": "HCD",
                "collision_energy": _number(collision_energy, f"{ce} MS2 CE", row_number) if collision_energy else "",
                "instrument": "Unknown", "instrument_type": "Unknown", "formula": formula,
                "exact_mass": exact_mass, "inchi_key": row["InChIKey"], "inchi": inchi, "smiles": smiles,
            })
    return pd.DataFrame(atlas_rows, columns=ATLAS_COLUMNS), pd.DataFrame(msms_rows, columns=MSMS_COLUMNS)


def deposit_atlas_to_google_sheet(atlas: pd.DataFrame, force_add: bool = False) -> None:
    """Append atlas rows to complete_atlas and sort its data by rt_peak.

    An append and the subsequent sort are separate operations. If the sort
    fails, the rows have already been appended and must not be appended again.
    """
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    if atlas.empty:
        raise ValueError("Cannot deposit an empty atlas.")
    if atlas.columns.tolist() != ATLAS_COLUMNS:
        raise ValueError("Atlas columns do not match the expected atlas schema.")

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        GOOGLE_SERVICE_ACCOUNT_FILE,
        ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
    )
    spreadsheet = gspread.authorize(credentials).open_by_key(ATLAS_SPREADSHEET)
    sheet = spreadsheet.worksheet("complete_atlas")
    if sheet.row_values(1) != ATLAS_COLUMNS:
        raise ValueError("Google Sheet header does not match the atlas columns; nothing was appended.")

    if not force_add:
        existing_keys = set(sheet.col_values(ATLAS_COLUMNS.index("inchi_key") + 1)[1:])
        existing_keys.discard("")
        duplicates = atlas["inchi_key"].isin(existing_keys)
        for inchi_key in sorted(atlas.loc[duplicates, "inchi_key"].unique()):
            logger.warning("Skipping atlas rows for existing inchi_key: %s", inchi_key)
        atlas = atlas.loc[~duplicates]
        if atlas.empty:
            logger.info("No new atlas rows to append.")
            return

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


def deposit_ref_standards_atlas_and_msms_refs(
    summary_path: Path, db_path: str | Path, run_name: str, dry_run: bool = False,
    force_add: bool = False,
) -> tuple[Path, Path]:
    """Write previews and, unless dry_run is set, deposit to the sheet and main DB."""
    atlas, msms = prepare_deposit_data(summary_path, db_path, run_name)
    atlas_path = summary_path.parent / f"Atlas_Deposit_Preview_{run_name}.csv"
    msms_path = summary_path.parent / f"MsmsRefs_Deposit_Preview_{run_name}.tab"
    created: list[Path] = []
    try:
        for path, frame, separator in ((atlas_path, atlas, ","), (msms_path, msms, "\t")):
            with path.open("x", encoding="utf-8", newline="") as handle:
                created.append(path)
                frame.to_csv(handle, sep=separator, index=False)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    logger.info("Prepared %s atlas rows and %s MS/MS spectra", len(atlas), len(msms))
    logger.info("Atlas preview: %s; MS/MS preview: %s", atlas_path, msms_path)
    if dry_run:
        logger.info("Dry run: nothing was deposited")
    else:
        from metatlas2.run_targeted_analysis import set_up_paths
        from metatlas2.database_interact import get_compound_uids_by_inchi_keys
        from metatlas2.workflow_objects import NewMsmsRefsConfig, add_compounds_from_files

        paths = set_up_paths(config=None)
        keys = atlas["inchi_key"].fillna("")
        if (keys == "").any():
            raise ValueError("Cannot deposit atlas rows with missing InChIKeys.")
        logger.info("Importing compounds from %s", atlas_path)
        add_compounds_from_files([str(atlas_path)], paths)
        missing_keys = set(keys) - set(get_compound_uids_by_inchi_keys(paths["main_db_path"], keys.unique().tolist()))
        if missing_keys:
            raise RuntimeError(f"Compound import did not add InChIKeys: {sorted(missing_keys)}")
        logger.info("Appending atlas rows to Google Sheets")
        deposit_atlas_to_google_sheet(atlas, force_add=force_add)
        logger.info("Importing MS/MS references from %s", msms_path)
        NewMsmsRefsConfig(entries=[{"path": str(msms_path)}]).execute()
    return atlas_path, msms_path
