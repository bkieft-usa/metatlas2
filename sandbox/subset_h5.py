"""Shrink sandbox h5 files to the m/z + RT windows of the sandbox atlases.

Used by sandbox/prepare_sandbox_data_package.sh. Filters each h5 file in place,
keeping only MS1/MS2 rows that fall inside at least one atlas window, so the
Zenodo sandbox package stays small.

Usage:
    python sandbox/subset_h5.py H5_DIR ATLAS_TSV [ATLAS_TSV ...]
        [--ppm-ms1 5.0] [--ppm-ms2 20.0] [--extra-time 0.5]
"""
import argparse
import csv
import sys
import tables
import numpy as np
from pathlib import Path


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_windows(tsv_path):
    windows = []
    if not tsv_path.exists():
        return windows
    with tsv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mz = parse_float(row.get("mz"))
            rt_min = parse_float(row.get("rt_min"))
            rt_max = parse_float(row.get("rt_max"))
            if mz is None or rt_min is None or rt_max is None:
                continue
            windows.append((mz, rt_min, rt_max))
    return windows


def ppm_bounds(mz, ppm):
    delta = mz * ppm / 1e6
    return mz - delta, mz + delta


def filter_h5_table(h5file, node_path, mz_col, rt_col, windows, ppm, extra_time):
    """Read a PyTables node and return a filtered structured array.

    Mirrors the logic in extract_data_from_h5._load_h5_table + _join_ms1/ms2_to_atlas:
    - For MS1 nodes, the production code reads the *_mz sorted variant and uses
      searchsorted for a fast mz-range pre-filter, then does a full mz+rt interval
      join per atlas window.
    - We replicate this by keeping any row whose (mz, rt) falls within at least one
      atlas window (mz ± ppm, rt ± extra_time).  The result is a superset of what
      the pipeline would actually use, ensuring no data is lost.
    """
    try:
        node = h5file.get_node(node_path)
    except tables.NoSuchNodeError:
        return None

    try:
        data = node.read()
    except Exception as exc:
        print(f"    Warning: could not read {node_path}: {exc}")
        return None

    if len(data) == 0:
        return data

    mz_vals = data[mz_col].astype(np.float64)
    rt_vals = data[rt_col].astype(np.float64)

    keep = np.zeros(len(data), dtype=bool)
    for mz, rt_min, rt_max in windows:
        mz_lo, mz_hi = ppm_bounds(mz, ppm)
        rt_lo = rt_min - extra_time
        rt_hi = rt_max + extra_time
        mask = (mz_vals >= mz_lo) & (mz_vals <= mz_hi) & (rt_vals >= rt_lo) & (rt_vals <= rt_hi)
        keep |= mask

    return data[keep]


def rewrite_h5_file(h5_path, windows_ms1, windows_ms2, ppm_ms1, ppm_ms2, extra_time):
    """Filter an h5 file in-place, keeping only rows within atlas windows.

    The production pipeline (extract_data_from_h5._load_h5_table) reads MS1 data
    from the *_mz sorted variant (e.g. ms1_pos_mz) using searchsorted for a fast
    mz pre-filter, then does a full interval join.  We must filter BOTH the base
    node (ms1_pos) AND its sorted variant (ms1_pos_mz) identically so the file
    remains consistent and the pipeline can read either form.

    MS2 nodes have no _mz variant — the pipeline reads ms2_pos directly.
    """
    import shutil

    # Base MS1/MS2 node paths
    ms1_base_keys = {"/ms1_pos", "/ms1_neg"}
    ms2_base_keys = {"/ms2_pos", "/ms2_neg"}
    # Sorted-by-mz MS1 variants read by the production pipeline
    ms1_mz_keys = {"/ms1_pos_mz", "/ms1_neg_mz"}

    tmp_path = h5_path.with_suffix(".tmp.h5")
    changed = False

    try:
        with tables.open_file(str(h5_path), mode="r") as src, \
             tables.open_file(str(tmp_path), mode="w") as dst:

            for node in src.walk_nodes("/", classname="Table"):
                node_path = node._v_pathname
                is_ms1_base = node_path in ms1_base_keys
                is_ms1_mz   = node_path in ms1_mz_keys
                is_ms2      = node_path in ms2_base_keys

                if (is_ms1_base or is_ms1_mz) and windows_ms1:
                    # Both the base and the _mz sorted variant are filtered the same way.
                    # The _mz variant is sorted by mz, so searchsorted works on it; our
                    # mask approach is equivalent and preserves the sort order because we
                    # keep a contiguous boolean mask (not a shuffle).
                    filtered = filter_h5_table(src, node_path, "mz", "rt", windows_ms1, ppm_ms1, extra_time)
                elif is_ms2 and windows_ms2:
                    filtered = filter_h5_table(src, node_path, "precursor_MZ", "rt", windows_ms2, ppm_ms2, extra_time)
                else:
                    filtered = None

                parent_path = node._v_parent._v_pathname
                node_name = node._v_name

                if filtered is not None:
                    before = len(node.read())
                    after = len(filtered)
                    if after != before:
                        changed = True
                    # Create parent groups if needed
                    if parent_path != "/":
                        dst.create_group("/", parent_path.lstrip("/"), createparents=True)
                    dst.create_table(parent_path, node_name, obj=filtered,
                                     filters=tables.Filters(complevel=1, complib="blosc"))
                else:
                    # Copy node as-is (metadata tables, etc.)
                    node.copy(dst.get_node(parent_path) if parent_path != "/" else dst.root,
                              newname=node_name)

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"    Warning: failed to filter {h5_path.name}: {exc}")
        return False, 0, 0

    if changed:
        shutil.move(str(tmp_path), str(h5_path))
    else:
        tmp_path.unlink(missing_ok=True)

    return changed, 0, 0



def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5_dir", type=Path)
    parser.add_argument("atlases", type=Path, nargs="+")
    parser.add_argument("--ppm-ms1", type=float, default=5.0)
    parser.add_argument("--ppm-ms2", type=float, default=20.0)
    parser.add_argument("--extra-time", type=float, default=0.5)
    args = parser.parse_args()

    windows = []
    for atlas in args.atlases:
        windows.extend(parse_windows(atlas))
    if not windows:
        sys.exit("No compound windows found in the atlas files.")

    files = sorted(args.h5_dir.glob("*.h5"))
    if not files:
        sys.exit(f"No h5 files found in {args.h5_dir}.")

    print(f"Filtering {len(files)} h5 files to atlas windows "
          f"(ppm_ms1={args.ppm_ms1}, ppm_ms2={args.ppm_ms2}, extra_time={args.extra_time} min)...")
    updated = 0
    for hfile in files:
        changed, _, _ = rewrite_h5_file(hfile, windows, windows, args.ppm_ms1, args.ppm_ms2, args.extra_time)
        updated += int(changed)
        print(f"  {hfile.name}: {'filtered' if changed else 'unchanged'}")
    print(f"H5 subset summary: {len(files)} files processed, {updated} rewritten")


if __name__ == "__main__":
    main()
