"""Read and resolve tagged input atlases from the shared Google Sheet."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import metatlas2.file_and_project_format as fpf
import metatlas2.logging_config as lcf
import metatlas2.load_tools as ldt

logger = lcf.get_logger("atlas_sheet")

ATLAS_SPREADSHEET = "1915jP523jCWwtcFM5rv0X3AmbZzPhN1ijcoX5Dw0Svw"
ATLAS_WORKSHEET = "complete_atlas"
GOOGLE_SERVICE_ACCOUNT_FILE = (
    "/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json"
)
ATLAS_COLUMNS = [
    "label", "compound_name", "inchi_key", "smiles", "inchi", "formula", "adduct",
    "mz", "rt_peak", "rt_min", "rt_max", "mz_tolerance", "chromatography", "polarity",
    "identification_notes", "chem_source", "lcmsrun_source", "date_created", "date_modified", "tags",
]
ATLAS_ROW_IDENTITY = ["label", "inchi_key", "adduct", "chromatography", "polarity"]
_NUMERIC_COLUMNS = ["mz", "rt_peak", "rt_min", "rt_max", "mz_tolerance"]


def read_centralized_atlas_sheet(
    spreadsheet_id: str = ATLAS_SPREADSHEET,
    worksheet_name: str = ATLAS_WORKSHEET,
    service_account_file: str = GOOGLE_SERVICE_ACCOUNT_FILE,
) -> pd.DataFrame:
    """Read the complete atlas worksheet as strings and validate its schema."""
    _, worksheet = open_centralized_atlas_worksheet(
        spreadsheet_id=spreadsheet_id,
        worksheet_name=worksheet_name,
        service_account_file=service_account_file,
    )
    values = worksheet.get_all_values()
    if not values:
        raise ValueError(f"Google Sheet worksheet {worksheet_name!r} is empty.")
    header = [str(name).strip() for name in values[0]]
    missing = [column for column in ATLAS_COLUMNS if column not in header]
    if missing:
        raise ValueError(
            f"Google Sheet worksheet {worksheet_name!r} is missing required columns: {missing}"
        )
    # Select by header name so extra/reordered sheet columns cannot shift fields.
    positions = [header.index(column) for column in ATLAS_COLUMNS]

    rows = []
    for row in values[1:]:
        if not any(str(value).strip() for value in row):
            continue
        padded = list(row) + [""] * len(header)
        rows.append([padded[position] for position in positions])
    return pd.DataFrame(rows, columns=ATLAS_COLUMNS, dtype=str)


def open_centralized_atlas_worksheet(
    spreadsheet_id: str = ATLAS_SPREADSHEET,
    worksheet_name: str = ATLAS_WORKSHEET,
    service_account_file: str = GOOGLE_SERVICE_ACCOUNT_FILE,
) -> tuple[Any, Any]:
    """Authorize and open the shared centralized-atlas worksheet."""
    credential_path = Path(service_account_file)
    if not credential_path.is_file():
        raise FileNotFoundError(f"Google Sheets service account file not found: {credential_path}")

    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    credentials = ServiceAccountCredentials.from_json_keyfile_name(
        str(credential_path),
        ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"],
    )
    spreadsheet = gspread.authorize(credentials).open_by_key(spreadsheet_id)
    return spreadsheet, spreadsheet.worksheet(worksheet_name)


def _normalize_tags(tags: str | list[str]) -> list[str]:
    return ldt.normalize_atlas_tags(tags)


def select_centralized_atlas_rows(
    atlas_rows: pd.DataFrame,
    tags: str | list[str],
    chromatography: str,
    polarity: str,
    atlas_category: str,
) -> pd.DataFrame:
    """Select rows matching any exact tag token and the requested method."""
    selected_tags = _normalize_tags(tags)
    missing_columns = set(ATLAS_COLUMNS) - set(atlas_rows.columns)
    if missing_columns:
        raise ValueError(f"Centralized atlas data is missing columns: {sorted(missing_columns)}")

    requested_chrom = fpf.normalize_chromatography(chromatography)
    requested_polarity = fpf.normalize_polarity(polarity)

    def row_has_tag(value: Any) -> bool:
        row_tags = {tag.strip() for tag in str(value or "").split(";") if tag.strip()}
        return any(tag in row_tags for tag in selected_tags)

    matches = atlas_rows.loc[
        atlas_rows["tags"].map(row_has_tag)
        & atlas_rows["chromatography"].map(
            lambda value: fpf.normalize_chromatography(str(value).strip()) == requested_chrom
        )
        & atlas_rows["polarity"].map(
            lambda value: fpf.normalize_polarity(str(value).strip()) == requested_polarity
        )
    ].copy()
    if matches.empty:
        raise ValueError(
            f"No centralized-atlas rows found for tags {selected_tags}, "
            f"chromatography {chromatography!r}, and polarity {polarity!r}."
        )

    matches["chromatography"] = chromatography
    matches["polarity"] = polarity
    duplicated = matches.duplicated(ATLAS_ROW_IDENTITY, keep=False)
    if duplicated.any():
        identities = (
            matches.loc[duplicated, ATLAS_ROW_IDENTITY]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate centralized-atlas row identities selected: {identities}")

    for column in _NUMERIC_COLUMNS:
        values = matches[column].astype(str).str.strip()
        if column in {"mz", "rt_peak"} and values.eq("").any():
            raise ValueError(f"Selected centralized-atlas rows are missing required values in {column!r}.")
        if column == "mz_tolerance":
            values = values.mask(values.eq(""), "5.0")
        if column in {"rt_min", "rt_max"}:
            fallback = matches["rt_peak"].astype(float)
            default_offset = -0.5 if column == "rt_min" else 0.5
            values = values.mask(values.eq(""), (fallback + default_offset).astype(str))
        converted = pd.to_numeric(values, errors="coerce")
        if converted.isna().any() or not converted.map(math.isfinite).all():
            raise ValueError(f"Selected centralized-atlas rows contain invalid numeric values in {column!r}.")
        matches[column] = converted.astype(float)

    return ldt.sort_atlas_input_dataframe(matches, atlas_category=atlas_category)


def _source_key(
    rows: pd.DataFrame,
    tags: list[str],
    chromatography: str,
    polarity: str,
) -> str:
    ordered = rows.sort_values(ATLAS_ROW_IDENTITY, kind="stable")
    digest = hashlib.sha256(
        ordered[ATLAS_COLUMNS].to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    return (
        f"google-sheet:{ATLAS_SPREADSHEET}/{ATLAS_WORKSHEET};"
        f"chrom={fpf.normalize_chromatography(chromatography)};"
        f"polarity={fpf.normalize_polarity(polarity)};"
        f"tags={json.dumps(sorted(tags), separators=(',', ':'))};sha256={digest}"
    )


def _create_or_reuse_atlas(
    rows: pd.DataFrame,
    *,
    tags: list[str],
    atlas_name: str,
    atlas_category: Any,
    chromatography: str,
    polarity: str,
    main_db_path: str,
) -> str:
    import metatlas2.database_interact as dbi

    source = _source_key(rows, tags, chromatography, polarity)
    existing_uid = dbi.get_atlas_uid_by_source(
        main_db_path,
        source=source,
        atlas_name=atlas_name,
        atlas_category=atlas_category.value,
        chromatography=chromatography,
        polarity=polarity,
    )
    if existing_uid:
        logger.info("Reusing existing centralized-sheet atlas %s", existing_uid)
        return existing_uid

    inchi_keys = rows["inchi_key"].dropna().astype(str).tolist()
    compound_uids = dbi.get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)
    missing = sorted(set(inchi_keys) - set(compound_uids))
    if missing:
        raise ValueError(
            f"Centralized-atlas rows reference InChIKeys missing from the compounds table: {missing}"
        )

    atlas = dbi.create_new_atlas_from_dataframe(
        atlas_df=rows,
        atlas_description=f"Centralized atlas sheet selection for tags {', '.join(tags)}",
        atlas_category=atlas_category,
        atlas_name=atlas_name,
        chromatography=chromatography,
        polarity=polarity,
        atlas_file_path=source,
        main_db_path=main_db_path,
    )
    dbi.save_atlas_to_database(atlas, main_db_path)
    logger.info("Created centralized-sheet atlas %s from tags %s", atlas.atlas_uid, tags)
    return atlas.atlas_uid


def _validate_database_atlas_uid(
    atlas_uid: str,
    *,
    chromatography: str,
    polarity: str,
    atlas_category: str,
    main_db_path: str,
) -> None:
    """Ensure a configured UID identifies a compatible main-database atlas."""
    from metatlas2.workflow_objects import Atlas, AtlasCategory

    try:
        atlas = Atlas.from_database(main_db_path, atlas_uid)
    except ValueError as exc:
        raise ValueError(
            f"Configured atlas UID {atlas_uid!r} is not a valid atlas in the main database."
        ) from exc
    if atlas.atlas_uid != atlas_uid:
        raise ValueError(
            f"Configured atlas UID {atlas_uid!r} was not found in the main database."
        )

    actual_category = (
        atlas.atlas_category.value
        if isinstance(atlas.atlas_category, AtlasCategory)
        else str(atlas.atlas_category)
    )
    expected = (
        fpf.normalize_chromatography(chromatography),
        fpf.normalize_polarity(polarity),
        atlas_category.upper(),
    )
    actual = (
        fpf.normalize_chromatography(atlas.chromatography),
        fpf.normalize_polarity(atlas.polarity),
        actual_category.upper(),
    )
    if actual != expected:
        raise ValueError(
            f"Atlas UID {atlas_uid!r} has chromatography/polarity/category "
            f"{actual!r}; config expects {expected!r}."
        )
    logger.info("Using existing main-database atlas UID %s", atlas_uid)


def resolve_config_atlas_inputs(config: Any, main_db_path: str) -> None:
    """Resolve tag selectors and validate configured main-database atlas UIDs."""
    selectors = []
    uid_selectors = []
    for chromatography, block in config.rt_alignment_config.items():
        atlas_config = block["ATLAS"]
        if atlas_config.get("uid"):
            uid_selectors.append((
                atlas_config["uid"],
                chromatography,
                atlas_config["polarity"],
                "QC",
            ))
        else:
            selectors.append((
                atlas_config,
                atlas_config["tags"],
                chromatography,
                atlas_config["polarity"],
                "QC",
                f"QC-{chromatography}",
            ))
    for targeted_analysis in config.targeted_analyses:
        if targeted_analysis.atlas_uid:
            uid_selectors.append((
                targeted_analysis.atlas_uid,
                targeted_analysis.chromatography,
                targeted_analysis.polarity,
                targeted_analysis.atlas_category.value,
            ))
        else:
            selectors.append((
                targeted_analysis,
                targeted_analysis.tags,
                targeted_analysis.chromatography,
                targeted_analysis.polarity,
                targeted_analysis.atlas_category.value,
                targeted_analysis.atlas_name,
            ))

    for atlas_uid, chromatography, polarity, category in uid_selectors:
        _validate_database_atlas_uid(
            atlas_uid,
            chromatography=chromatography,
            polarity=polarity,
            atlas_category=category,
            main_db_path=main_db_path,
        )

    if not selectors:
        return
    atlas_rows = read_centralized_atlas_sheet()
    for target, tags, chromatography, polarity, category, atlas_name in selectors:
        selected = select_centralized_atlas_rows(
            atlas_rows,
            tags=tags,
            chromatography=chromatography,
            polarity=polarity,
            atlas_category=category,
        )
        from metatlas2.workflow_objects import AtlasCategory

        uid = _create_or_reuse_atlas(
            selected,
            tags=_normalize_tags(tags),
            atlas_name=atlas_name,
            atlas_category=AtlasCategory(category),
            chromatography=chromatography,
            polarity=polarity,
            main_db_path=main_db_path,
        )
        if isinstance(target, dict):
            target["uid"] = uid
        else:
            target.atlas_uid = uid
