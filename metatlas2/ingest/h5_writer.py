"""mzML -> metatlas HDF5 (format version 5). Runs inside the metatlas2 container.

Port of the legacy ``metatlas.io.mzml_loader.mzml_to_hdf``. The file layout
and values are identical: the same four tables (``ms1_neg``, ``ms1_pos``,
``ms2_neg``, ``ms2_pos``) with the same columns, the same ``*_mz`` copies
sorted by m/z, blosc level-1 compression, and ``format_version = 5``. The
same spectra are skipped, and the Easy-IC lock mass is removed the same way.

What changed (no effect on the data):

* Peaks are handled as numpy arrays and appended in large batches, instead
  of building one Python tuple per peak and appending once per spectrum.
* The file is written to ``<name>.h5.partial`` and renamed when complete, so
  a killed conversion never leaves a truncated ``.h5`` that looks finished.
* A summary (spectra read/skipped, rows per table, timings) is returned for
  logging.

Command line (used by the host-side converter)::

    python -m metatlas2.ingest.h5_writer FILE.mzML [--h5 FILE.h5] [--no-easyic-filter]

prints one JSON summary line on stdout and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 5
LEGACY_METATLAS_VERSION = "0.7"
WRITER = "metatlas2.ingest.h5_writer"

#: Easy-IC (fluoranthene) lock-mass m/z by polarity (1 = positive, 0 = negative).
EASYIC_MZ = {1: 202.07770, 0: 202.07880}
EASYIC_TOLERANCE = 0.001

TABLE_NAMES = ("ms1_neg", "ms2_neg", "ms1_pos", "ms2_pos")
MS1_DTYPE = np.dtype([("mz", "<f4"), ("i", "<f4"), ("rt", "<f4"), ("polarity", "<i2")])
MS2_DTYPE = np.dtype(MS1_DTYPE.descr + [
    ("precursor_MZ", "<f4"), ("precursor_intensity", "<f4"), ("collision_energy", "<f4")])

# Rows buffered per table before appending to HDF5 (bounds memory for huge files).
_FLUSH_ROWS = 2_000_000


def _descriptions():
    """PyTables descriptions identical to the legacy ``MS1Data``/``MS2Data`` classes."""
    import tables

    class MS1Data(tables.IsDescription):
        mz = tables.Float32Col(pos=0)
        i = tables.Float32Col(pos=1)
        rt = tables.Float32Col(pos=2)
        polarity = tables.Int16Col(pos=3)

    class MS2Data(MS1Data):
        precursor_MZ = tables.Float32Col(pos=4)
        precursor_intensity = tables.Float32Col(pos=5)
        collision_energy = tables.Float32Col(pos=6)

    return MS1Data, MS2Data


class _SkipSpectrum(Exception):
    """Spectrum is skipped (legacy behaviour for KeyError/TypeError and read errors)."""


def _read_spectrum_fields(spectrum) -> tuple[str, np.ndarray, float, int, tuple | None]:
    """Extract what the legacy ``read_spectrum`` used, with the same failure modes.

    Returns ``(table_key, peaks, rt, polarity, precursor)`` where *precursor* is
    ``(mz, intensity, collision_energy)`` for MS2 and None for MS1.

    Raises:
        _SkipSpectrum: wherever the legacy code would skip the spectrum.
    """
    try:
        if spectrum["negative scan"] is True:
            polarity = 0
        elif spectrum["positive scan"] is True:
            polarity = 1
        else:
            raise _SkipSpectrum("unknown polarity")
        ms_level = spectrum.ms_level
        rt = spectrum.scan_time_in_minutes()
        if ms_level == 1:
            precursor = None
        else:
            prec = spectrum.selected_precursors
            if len(prec) != 1:
                raise _SkipSpectrum("not exactly one precursor")
            prec = prec[0]
            collision_energy = spectrum["collision energy"]
            precursor = (prec["mz"], prec["i"], collision_energy)
        peaks = spectrum.peaks("centroided")
    except _SkipSpectrum:
        raise
    except (KeyError, TypeError) as err:
        raise _SkipSpectrum(type(err).__name__) from err
    except Exception as err:  # legacy: "Read spectrum error", then skip
        raise _SkipSpectrum(f"read error: {err}") from err
    key = f"ms{1 if ms_level == 1 else 2}_{'pos' if polarity else 'neg'}"
    return key, np.asarray(peaks), rt, polarity, precursor


def _spectrum_rows(key: str, peaks: np.ndarray, rt: float, polarity: int,
                   precursor: tuple | None, filter_easyic: bool) -> np.ndarray:
    """Structured rows for one spectrum (lock mass removed if requested)."""
    peaks = peaks.reshape(-1, 2) if peaks.size else np.empty((0, 2))
    mz = peaks[:, 0].astype(np.float64, copy=False)
    intensity = peaks[:, 1]
    if filter_easyic and len(mz):
        keep = np.abs(mz - EASYIC_MZ[polarity]) > EASYIC_TOLERANCE
        mz, intensity = mz[keep], intensity[keep]
    dtype = MS1_DTYPE if key.startswith("ms1") else MS2_DTYPE
    rows = np.empty(len(mz), dtype=dtype)
    rows["mz"] = mz
    rows["i"] = intensity
    rows["rt"] = rt
    rows["polarity"] = polarity
    if precursor is not None:
        # Not inside the skip handler: a None here fails the conversion, as in legacy.
        rows["precursor_MZ"] = precursor[0]
        rows["precursor_intensity"] = precursor[1]
        rows["collision_energy"] = precursor[2]
    return rows


def mzml_to_h5(
    mzml_path: str | os.PathLike,
    h5_path: str | os.PathLike | None = None,
    filter_easyic: bool = True,
) -> dict[str, Any]:
    """Convert an mzML file to a metatlas HDF5 file.

    Args:
        mzml_path: Input mzML.
        h5_path: Output path (default: same name with ``.h5``).
        filter_easyic: Remove the Easy-IC lock-mass peak (legacy default True).

    Returns:
        Summary dict: ``h5_file``, ``spectra_read``, ``spectra_skipped``,
        ``skip_reasons``, ``rows`` per table, ``seconds``.

    Raises:
        TypeError: If the mzML cannot be opened (legacy message).
        Exception: Any error while writing; the partial file is removed.
    """
    import tables
    from pymzml.run import Reader

    t0 = time.time()
    mzml_path = str(mzml_path)
    h5_path = Path(h5_path) if h5_path else Path(mzml_path.replace(".mzML", ".h5"))
    partial = h5_path.with_name(h5_path.name + ".partial")
    try:
        reader = Reader(mzml_path)
    except Exception as err:
        raise TypeError(f'Not a valid mzML file: "{mzml_path}" ({err})') from err

    ms1_desc, ms2_desc = _descriptions()
    buffers: dict[str, list[np.ndarray]] = {k: [] for k in TABLE_NAMES}
    buffered = dict.fromkeys(TABLE_NAMES, 0)
    rows_written = dict.fromkeys(TABLE_NAMES, 0)
    spectra_read = skipped = 0
    skip_reasons: dict[str, int] = {}

    h5 = tables.open_file(str(partial), "w", filters=tables.Filters(complib="blosc", complevel=1))
    try:
        out = {
            "ms1_neg": h5.create_table("/", "ms1_neg", description=ms1_desc),
            "ms1_pos": h5.create_table("/", "ms1_pos", description=ms1_desc),
            "ms2_neg": h5.create_table("/", "ms2_neg", description=ms2_desc),
            "ms2_pos": h5.create_table("/", "ms2_pos", description=ms2_desc),
        }

        def flush(key: str) -> None:
            if buffers[key]:
                block = np.concatenate(buffers[key])
                out[key].append(block)
                rows_written[key] += len(block)
                buffers[key].clear()
                buffered[key] = 0

        for spectrum in reader:
            spectra_read += 1
            try:
                key, peaks, rt, polarity, precursor = _read_spectrum_fields(spectrum)
            except _SkipSpectrum as err:
                skipped += 1
                reason = str(err).split(":")[0]
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            if peaks.size == 0:  # legacy: "if not data: continue"
                skipped += 1
                skip_reasons["no peaks"] = skip_reasons.get("no peaks", 0) + 1
                continue
            rows = _spectrum_rows(key, peaks, rt, polarity, precursor, filter_easyic)
            if len(rows):
                buffers[key].append(rows)
                buffered[key] += len(rows)
                if buffered[key] >= _FLUSH_ROWS:
                    flush(key)
        for key in TABLE_NAMES:
            flush(key)
        for name in TABLE_NAMES:
            table = h5.get_node("/" + name)
            table.cols.mz.create_csindex()
            table.copy(sortby="mz", newname=name + "_mz")
            table.cols.mz.remove_index()
        h5.set_node_attr("/", "format_version", FORMAT_VERSION)
        h5.set_node_attr("/", "metatlas_version", LEGACY_METATLAS_VERSION)
        h5.set_node_attr("/", "writer", WRITER)
        h5.close()
    except BaseException:
        if h5.isopen:
            h5.close()
        partial.unlink(missing_ok=True)
        raise
    os.replace(partial, h5_path)
    return {
        "h5_file": str(h5_path),
        "spectra_read": spectra_read,
        "spectra_skipped": skipped,
        "skip_reasons": skip_reasons,
        "rows": rows_written,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert an mzML file to metatlas HDF5.")
    parser.add_argument("mzml_file")
    parser.add_argument("--h5", default=None, help="Output .h5 path (default: next to the mzML).")
    parser.add_argument("--no-easyic-filter", action="store_true", help="Keep the Easy-IC lock-mass peaks.")
    args = parser.parse_args(argv)
    try:
        summary = mzml_to_h5(args.mzml_file, args.h5, filter_easyic=not args.no_easyic_filter)
    except Exception as err:
        print(json.dumps({"ok": False, "error": f"{type(err).__name__}: {err}"}), flush=True)
        return 1
    print(json.dumps({"ok": True, **summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
