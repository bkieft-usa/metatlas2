#!/global/cfs/cdirs/metatlas/tools/metatlas2/.venv/bin/python

import ast
import csv
import json
import sys
from tqdm.auto import tqdm
from pathlib import Path


def parse_spectrum(spectrum_str: str) -> tuple[list[float], list[float]]:
    """
    Parse the spectrum string into separate mz and intensity lists.
    Expected format: [[mz1, mz2, ...], [int1, int2, ...]]
    """
    parsed = ast.literal_eval(spectrum_str.strip())
    mz_values = [float(v) for v in parsed[0]]
    intensities = [float(v) for v in parsed[1]]
    return mz_values, intensities


def round_mz(values: list[float], decimal: float) -> list[float]:
    """
    Round mz values to the number of significant decimal places,
    stripping trailing zeros naturally via float representation.
    """
    decimals = int(decimal)
    rounded = []
    for v in values:
        r = round(v, decimals)
        # Convert to float to strip unnecessary trailing zeros (e.g. 96.9620 -> 96.962)
        rounded.append(float(f"{r:.{decimals}f}".rstrip("0").rstrip(".") or "0"))
    return rounded


def parse_nullable(value: str):
    """Return None if the value is empty/whitespace, otherwise return stripped string."""
    stripped = value.strip()
    return stripped if stripped else None


def parse_float_nullable(value: str):
    """Return None if empty, otherwise parse as float."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def tsv_to_jsonl(input_path: Path, output_path: Path) -> None:
    """Read the TSV file and write JSONL output."""
    with (
        input_path.open("r", encoding="utf-8") as infile,
        output_path.open("w", encoding="utf-8") as outfile,
    ):
        reader = csv.DictReader(infile, delimiter="\t")

        for row in tqdm(reader, desc="Converting rows to JSONL", unit="row"):
            # The first unnamed column is the row index
            ix = int(row.get("", row.get("\ufeff", 0)))  # handle BOM if present

            decimal = parse_float_nullable(row.get("decimal", "")) or 4.0
            mz_values, intensities = parse_spectrum(row["spectrum"])
            mz_rounded = round_mz(mz_values, decimal)

            record = {
                "ix": ix,
                "database": parse_nullable(row["database"]),
                "id": parse_nullable(row["id"]),
                "name": parse_nullable(row["name"]),
                "decimal": decimal,
                "inchi_key": parse_nullable(row["inchi_key"]),
                "precursor_mz": parse_float_nullable(row["precursor_mz"]),
                "polarity": parse_nullable(row["polarity"]),
                "adduct": parse_nullable(row["adduct"]),
                "fragmentation_method": parse_nullable(row["fragmentation_method"]),
                "collision_energy": parse_float_nullable(row["collision_energy"]),
                "instrument": parse_nullable(row["instrument"]),
                "instrument_type": parse_nullable(row["instrument_type"]),
                "formula": parse_nullable(row["formula"]),
                "mono_isotopic_molecular_weight": parse_float_nullable(row["exact_mass"]),
                "inchi": parse_nullable(row["inchi"]),
                "smiles": parse_nullable(row["smiles"]),
                "mz": mz_rounded,
                "intensities": intensities,
            }

            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Output written to: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert.py <input.tsv> [output.json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: input file '{input_path}' not found.")
        sys.exit(1)

    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".json")

    tsv_to_jsonl(input_path, output_path)


if __name__ == "__main__":
    main()