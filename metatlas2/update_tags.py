"""Update tags in the centralized atlas worksheet."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from metatlas2 import atlas_sheet
import metatlas2.logging_config as lcf

logger = lcf.get_logger("update_tags")
_INCHI_KEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def _read_input(path: str | Path) -> dict[str, list[str]]:
    entries: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file, strict=True)
        if next(reader, None) != ["inchi_key", "tags"]:
            raise ValueError("CSV must have exactly the columns inchi_key,tags, in that order.")
        for row in reader:
            if len(row) != 2:
                raise ValueError(f"CSV line {reader.line_num} must have exactly two columns.")
            key = row[0].strip()
            if not _INCHI_KEY.fullmatch(key):
                raise ValueError(f"Invalid InChIKey on CSV line {reader.line_num}: {key!r}")
            if key in entries:
                raise ValueError(f"Duplicate InChIKey on CSV line {reader.line_num}: {key}")
            tags = [tag.strip() for tag in row[1].split(";")]
            if any(not tag or any(ord(char) < 32 or ord(char) == 127 for char in tag) for tag in tags):
                raise ValueError(f"CSV line {reader.line_num} contains an empty tag or control character.")
            entries[key] = tags
    if not entries:
        raise ValueError("CSV must contain at least one data row.")
    return entries


def _sheet_columns(values: list[list[str]]) -> tuple[int, int]:
    if not values:
        raise ValueError("Centralized atlas worksheet is empty.")
    header = [column.strip() for column in values[0]]
    for column in ("inchi_key", "tags"):
        if header.count(column) != 1:
            raise ValueError(f"Centralized atlas worksheet must have exactly one {column!r} column.")
    return header.index("inchi_key"), header.index("tags")


def _tokens(value: str) -> list[str]:
    return [tag.strip() for tag in value.split(";") if tag.strip()]


def retrieve_tags(sheet: Any) -> list[str]:
    """Return all distinct tag tokens, sorted for deterministic output."""
    values = sheet.get_all_values()
    _, tag_column = _sheet_columns(values)
    tags = {
        tag
        for row in values[1:]
        for tag in _tokens(row[tag_column] if tag_column < len(row) else "")
    }
    if any(any(ord(char) < 32 or ord(char) == 127 for char in tag) for tag in tags):
        raise ValueError("Centralized atlas contains a tag with a control character.")
    return sorted(tags)


def update_tags(sheet: Any, operation: str, entries: dict[str, list[str]]) -> int:
    """Apply a validated set of changes to all matching rows in one batch."""
    if operation not in {"add", "remove", "retag"}:
        raise ValueError(f"Unknown tag operation: {operation!r}")
    values = sheet.get_all_values()
    key_column, tag_column = _sheet_columns(values)
    found: set[str] = set()
    changes = []
    for row_number, row in enumerate(values[1:], start=2):
        key = row[key_column].strip() if key_column < len(row) else ""
        if key not in entries:
            continue
        found.add(key)
        old = row[tag_column] if tag_column < len(row) else ""
        existing = _tokens(old)
        requested = entries[key]
        if operation == "add":
            result = existing.copy()
            for tag in requested:
                if tag in result:
                    logger.warning("Tag %r already exists for %s on sheet row %s; skipping.", tag, key, row_number)
                else:
                    result.append(tag)
        elif operation == "remove":
            for tag in dict.fromkeys(requested):
                if tag not in existing:
                    logger.warning("Tag %r not found for %s on sheet row %s.", tag, key, row_number)
            result = [tag for tag in existing if tag not in requested]
        else:
            result = list(dict.fromkeys(requested))

        # Do not rewrite cells when token membership and order have not changed.
        if result != existing:
            from gspread.utils import rowcol_to_a1

            changes.append({
                "range": rowcol_to_a1(row_number, tag_column + 1),
                "values": [[";".join(result)]],
            })

    for key in sorted(entries.keys() - found):
        logger.warning("InChIKey %s was not found in the centralized atlas.", key)
    if changes:
        if sheet.get_all_values() != values:
            raise RuntimeError("Centralized atlas changed during preparation; no tags were updated. Retry.")
        sheet.batch_update(changes, value_input_option="RAW")
    logger.info("Updated tags in %s sheet row(s).", len(changes))
    return len(changes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("add", "remove", "retag"):
        subparser = subparsers.add_parser(operation)
        subparser.add_argument("--input-csv", "--input_csv", required=True, type=Path)
    subparsers.add_parser("retrieve")
    args = parser.parse_args(argv)

    entries = _read_input(args.input_csv) if args.operation != "retrieve" else None
    _, sheet = atlas_sheet.open_centralized_atlas_worksheet()
    if args.operation == "retrieve":
        for tag in retrieve_tags(sheet):
            print(tag)
    else:
        update_tags(sheet, args.operation, entries)


if __name__ == "__main__":
    main()
