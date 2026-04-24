from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
import os
import statistics
import textwrap
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import metatlas2.database_interact as dbi
import metatlas2.logging_config as lcf
logger = lcf.get_logger('analysis_summary')

def _get_file_color(file_path: str, color_map: Optional[Dict[str, str]] = None) -> str:
    """Determine color for a file based on color mapping.
    
    Matches file name against keys in color_map (case-insensitive substring match),
    similar to the GUI logic. Falls back to gray if no match is found.
    
    Parameters
    ----------
    file_path : str
        Full path to the file
    color_map : dict, optional
        Mapping from LCMS run identifier to color string (e.g. {'ISTD': 'blue', 'QC': 'green'})
        If None, returns gray for all files.
        
    Returns
    -------
    str
        Color string (matched color or 'gray' as fallback)
    """
    if color_map is None:
        return "gray"
    
    # Extract short filename (similar to GUI)
    raw_name = os.path.basename(str(file_path))
    name_parts = raw_name.split(".")[0].split("_")
    short_name = "_".join(name_parts[11:]) if len(name_parts) > 11 else raw_name
    
    # Match against color_map keys (case-insensitive substring match)
    for key, color in color_map.items():
        if key.lower() in short_name.lower():
            return color
    
    # No match found, use gray
    return "gray"

def _parse_spectrum(raw_spectrum_json: str) -> Tuple[List, List]:
    """Parse a JSON-encoded spectrum string into (x_array, y_array) lists.

    The stored format is a JSON array of two lists: [[x0, x1, ...], [y0, y1, ...]].
    For ms1_data.raw_spectrum the x-axis is retention time; for ms2_data and
    ms2_hits spectra the x-axis is m/z.
    """
    if raw_spectrum_json is None:
        return [], []
    if isinstance(raw_spectrum_json, float) and np.isnan(raw_spectrum_json):
        return [], []
    try:
        x_arr, y_arr = json.loads(raw_spectrum_json)
        # Replace any NaN intensities with 0
        y_arr = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in y_arr]
        return list(x_arr), list(y_arr)
    except Exception:
        return [], []

def _get_compound_info(
    main_db_path: str,
    inchi_key: str,
) -> Optional[Tuple[str, str, str, str]]:
    """Return the compound info for *inchi_key* from main_db: formula, smiles, inchi, pubchem_cid."""
    try:
        with dbi.get_db_connection(main_db_path) as conn:
            row = conn.execute(
                "SELECT formula, smiles, inchi, pubchem_cid FROM compounds WHERE inchi_key = ? LIMIT 1",
                [inchi_key],
            ).fetchone()
        if row and any(row):
            return row[0], row[1], row[2], row[3]
    except Exception as exc:
        logger.warning("Could not retrieve info from %s for %s: %s", main_db_path, inchi_key, exc)
    return None, None, None, None

def _get_monoisotopic_mass(
    main_db_path: str,
    inchi_key: str,
) -> Optional[float]:
    """Return the monoisotopic mass for *inchi_key*."""
    try:
        with dbi.get_db_connection(main_db_path) as conn:
            row = conn.execute(
                "SELECT mono_isotopic_molecular_weight FROM compounds WHERE inchi_key = ? LIMIT 1",
                [inchi_key],
            ).fetchone()
        if row and row[0] is not None:
            return float(row[0])
    except Exception as exc:
        logger.warning("Could not retrieve mass from %s for %s: %s", main_db_path, inchi_key, exc)
    return None

def _plot_mirror(
    ax,
    qry_mz: List, qry_int: List,
    ref_mz: List, ref_int: List,
    frag_colors: Optional[List] = None,
    score: float = np.nan,
    rt: float = np.nan,
    title: str = "",
) -> None:
    """Draw a mirror plot on *ax*: query spectrum above, reference below."""
    ax.axhline(y=0, color="black", linewidth=1.0)

    if frag_colors is None or len(frag_colors) != len(qry_mz):
        frag_colors = ["tomato"] * len(qry_mz)

    if qry_mz and qry_int:
        for mz, intensity, color in zip(qry_mz, qry_int, frag_colors):
            ax.bar(mz, intensity, color=color, width=1.1, alpha=0.85)

    scale = 1.0
    if ref_mz and ref_int:
        scale = (max(qry_int) / max(ref_int)) if (qry_int and max(ref_int) > 0) else 1.0
        for mz, intensity, color in zip(ref_mz, ref_int, frag_colors):
            ax.bar(mz, -intensity * scale, color=color, width=1.1, alpha=0.6)

    if qry_mz and qry_int:
        mz_np = np.array(qry_mz, dtype=float)
        int_np = np.array(qry_int, dtype=float)
        valid = np.isfinite(mz_np) & np.isfinite(int_np) & (int_np > 0)

        if np.any(valid):
            valid_idx = np.where(valid)[0]
            top_n = min(5, len(valid_idx))
            order = np.argsort(int_np[valid_idx])[::-1][:top_n]
            top_idx = valid_idx[order]

            # Sort labels left-to-right so stagger logic is applied in x-order
            top_idx_sorted = sorted(top_idx, key=lambda i: mz_np[i])
            MIN_MZ_GAP = 5.0  # m/z units below which labels are considered overlapping horizontally
            y_max = float(int_np[valid_idx].max())
            TEXT_HEIGHT_FRACTION = 0.09  # fraction of y-axis range per stagger level
            
            prev_x = None
            stagger_level = 0

            for idx in top_idx_sorted:
                x_txt = float(mz_np[idx])
                y_base = float(int_np[idx]) * 1.02

                # Check for horizontal overlap with previous label
                if prev_x is not None and abs(x_txt - prev_x) < MIN_MZ_GAP:
                    stagger_level += 1
                else:
                    stagger_level = 0

                # Add stagger offset in data coordinates
                y_txt = y_base + stagger_level * (y_max * TEXT_HEIGHT_FRACTION)

                ax.text(
                    x_txt, y_txt, f"{mz_np[idx]:.4f}",
                    fontsize=10, ha="center", va="bottom", rotation=0, color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.2),
                    clip_on=False,
                )
                prev_x = x_txt

            ax.margins(y=0.20)

    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel(f"Intensity (Ref x{scale:.2f})", fontsize=14, weight="bold")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(labelsize=14)

    score_str = f"{score:.3f}" if (isinstance(score, (int, float)) and not np.isnan(score)) else "N/A"
    rt_str = f"{rt:.2f}" if (isinstance(rt, (int, float)) and not np.isnan(rt)) else "N/A"

    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, f"Score: {score_str}", fontsize=14, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: {rt_str} min", fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_raw_ms2(
    ax,
    mz_arr: List, int_arr: List,
    rt: float = np.nan,
    title: str = "",
) -> None:
    """Draw a raw MS2 spectrum (query only, no reference match) on *ax*."""
    ax.axhline(y=0, color="black", linewidth=1.0)
    if mz_arr and int_arr:
        for mz, intensity in zip(mz_arr, int_arr):
            ax.bar(mz, intensity, color="tomato", width=1.1, alpha=0.85)
    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity", fontsize=14)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(labelsize=14)

    rt_str = f"{rt:.2f}" if (isinstance(rt, (int, float)) and not np.isnan(rt)) else "N/A"
    ax.text(0.5, 1.13, title,               fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, "Score: N/A",        fontsize=12, weight="bold",   ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: {rt_str} min", fontsize=10,  weight="normal", ha="center", va="top",    transform=ax.transAxes)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_empty_ms2(ax, title: str = "") -> None:
    """Draw a blank MS2 panel with a 'No MS2 Data' annotation."""
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1e6)
    ax.axhline(y=0, color="black", linewidth=1.0)
    ax.set_xlabel("m/z", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity", fontsize=14, weight="bold")
    ax.tick_params(labelsize=14)
    ax.text(0.5, 0.5, "No MS2 Data", transform=ax.transAxes,
            ha="center", va="center", fontsize=14, weight="bold", color="gray")
    ax.text(0.5, 1.10, title,        fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.04, "Score: N/A", fontsize=12, weight="bold",   ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 0.99, "RT: N/A",    fontsize=10,  weight="normal", ha="center", va="top",    transform=ax.transAxes)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

def _plot_structure(
    ax,
    smiles: Optional[str],
    inchi: Optional[str],
    inchi_key: str,
    size: int = 500,
) -> None:
    """Draw molecular structure using RDKit's MolToImage (requires libXrender, available in container)."""
    ax.axis("off")

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from PIL import Image, ImageDraw, ImageFont

        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None and inchi:
            mol = Chem.MolFromInchi(inchi)

        if mol is not None:
            # Generate structure image directly (requires libXrender)
            img = Draw.MolToImage(mol, size=(size, size), kekulize=True)

            # Add InChIKey annotation below the structure
            if inchi_key:
                font_size = 25
                line_height = int(font_size * 1.2)
                avg_char_width = font_size / 2
                lines, current_line = [], ""
                
                # Word-wrap the InChIKey
                for char in inchi_key:
                    test_line = current_line + char
                    if len(test_line) * avg_char_width > size - 20:
                        lines.append(current_line)
                        current_line = char
                    else:
                        current_line = test_line
                if current_line:
                    lines.append(current_line)

                # Create new image with space for text
                text_height = len(lines) * line_height + 20
                new_img = Image.new("RGB", (size, size + text_height), "white")
                new_img.paste(img, (0, 0))
                
                # Draw text
                draw_obj = ImageDraw.Draw(new_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except Exception:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)
                    except Exception:
                        font = ImageFont.load_default()
                
                y_position = size + 10
                for line in lines:
                    draw_obj.text((10, y_position), line, fill="black", font=font)
                    y_position += line_height
                
                img = new_img

            ax.imshow(img, aspect="equal")
            return

    except Exception as exc:
        logger.warning("Could not draw structure for %s: %s", inchi_key, exc)

    ax.text(0.5, 0.5, f"InChIKey:\n{inchi_key}", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, weight="bold", color="gray")

def _plot_eic(
    ax,
    ms1_compound_df: pd.DataFrame,
    mc_row: pd.Series,
    log_scale: bool = False,
    color_map: Optional[Dict[str, str]] = None,
) -> None:
    """Plot EIC traces for all files of one compound.

    *ms1_compound_df* is already filtered to the target inchi_key + adduct.
    Each row holds the full EIC for one file in ``raw_spectrum`` (JSON [rt_list, i_list]).
    
    Parameters
    ----------
    ax : matplotlib axis
        Axis to plot on
    ms1_compound_df : pd.DataFrame
        MS1 data for one compound (filtered to target inchi_key + adduct)
    mc_row : pd.Series
        Manual curation row for the compound
    log_scale : bool
        If True, plot log10(intensity)
    color_map : dict, optional
        Mapping from LCMS run identifier to color string (e.g. {'ISTD': 'blue'})
        If None, all traces will be gray
    """
    rt_min  = mc_row.get("rt_min",       np.nan)
    rt_max  = mc_row.get("rt_max",       np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    # RT boundary and atlas peak lines
    if not np.isnan(rt_min):
        ax.axvline(rt_min,  color="red",   linestyle="--", linewidth=1.5, alpha=0.7, label="_rt_min")
    if not np.isnan(rt_max):
        ax.axvline(rt_max,  color="red",   linestyle="--", linewidth=1.5, alpha=0.7, label="_rt_max")
    if not np.isnan(rt_peak):
        ax.axvline(rt_peak, color="black", linestyle=":",  linewidth=1.5, alpha=0.7, label="_atlas_peak")

    for file_idx, row in ms1_compound_df.iterrows():
        rt_arr, i_arr = _parse_spectrum(row["raw_spectrum"])
        if not rt_arr:
            continue
        color = _get_file_color(row["file_path"], color_map)
        # Shorten file label: take the sample-name portion after the 11th underscore segment
        raw_name  = os.path.basename(str(row["file_path"]))
        name_parts = raw_name.split(".")[0].split("_")
        fname_short = "_".join(name_parts[11:]) if len(name_parts) > 11 else raw_name
        if log_scale:
            i_arr = [np.log10(max(v, 1)) for v in i_arr]
        ax.plot(rt_arr, i_arr, color=color, linewidth=1.0, alpha=0.7, label=fname_short)

    ax.set_xlabel("Retention Time (min)", fontsize=14, weight="bold")
    ax.set_ylabel("Intensity (log₁₀)" if log_scale else "Intensity", fontsize=14, weight="bold")
    ax.tick_params(labelsize=14)
    if not log_scale:
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if ms1_compound_df.empty:
        ax.text(0.5, 0.5, "No MS1 data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")


def _plot_compound_info_table(ax, mc_row: pd.Series) -> None:
    """Render a two-column key/value table of compound metadata on *ax*."""
    ax.axis("off")

    def _fmt(val, fmt=None):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return fmt.format(val) if fmt else str(val)

    rt_err = mc_row.get("best_ms1_rt_error")
    ppm    = mc_row.get("best_ms1_ppm_error")

    rows = [
        ("Compound",        _fmt(mc_row.get("compound_name"))),
        ("Formula",         _fmt(mc_row.get("formula"))),
        ("Adduct",          _fmt(mc_row.get("adduct"))),
        ("Polarity",        _fmt(mc_row.get("polarity"))),
        ("Chromatography",  _fmt(mc_row.get("chromatography"))),
        ("Atlas m/z",       _fmt(mc_row.get("atlas_mz"),      "{:.4f}")),
        ("Measured m/z",    _fmt(mc_row.get("best_ms1_mz"),   "{:.4f}")),
        ("m/z ppm Δ",  _fmt(ppm, "{:.2f}")),
        ("Atlas RT range",  f"{_fmt(mc_row.get('atlas_rt_min'), '{:.3f}')} - {_fmt(mc_row.get('atlas_rt_max'), '{:.3f}')} min"),
        ("Measured RT range", f"{_fmt(mc_row.get('rt_min'), '{:.3f}')} - {_fmt(mc_row.get('rt_max'), '{:.3f}')} min"),
        ("Atlas RT peak",   _fmt(mc_row.get("atlas_rt_peak"), "{:.3f} min")),
        ("Measured RT",     f"{_fmt(mc_row.get('best_ms1_rt'), '{:.3f} min')} "),
        ("RT Δ",   _fmt(rt_err, '{:.3f}')),
    ]

    y_start = 1.03
    y_end  = 0.09
    for label, value in rows:
        ax.text(0.02, y_start, f"{label}:", fontsize=13, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.34, y_start, value, fontsize=13, va="center", transform=ax.transAxes)
        y_start -= y_end

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1)


def _plot_hit_info_table(
    ax,
    ms2_hits_compound_df: pd.DataFrame,
    mc_row: pd.Series,
) -> None:
    """Render the best MS2 hit as a vertical Theoretical/Measured/Difference table on *ax*."""
    ax.axis("off")

    if ms2_hits_compound_df.empty:
        ax.text(0.5, 0.5, "No MS2 hits found.", transform=ax.transAxes,
                ha="center", va="center", fontsize=25, color="gray")
        return

    # Single best hit overall
    best_hit = ms2_hits_compound_df.sort_values("score", ascending=False).iloc[0]

    # Abbreviated file name used as the table title
    raw_name   = os.path.basename(str(best_hit["file_path"]))
    name_without_ext = raw_name.split(".")[0]
    name_without_ext = name_without_ext.replace("_ms2_pos", "").replace("_ms2_neg", "")

    # Scalar values
    atlas_mz    = float(mc_row.get("atlas_mz",           np.nan))
    measured_mz = float(mc_row.get("best_ms1_mz",        np.nan))
    ppm_error   = float(mc_row.get("best_ms1_ppm_error", np.nan))
    atlas_rt    = float(mc_row.get("atlas_rt_peak",      np.nan))
    measured_rt = float(mc_row.get("best_ms1_rt",        np.nan))
    rt_error    = float(mc_row.get("best_ms1_rt_error",  np.nan))
    score       = float(best_hit.get("score",            np.nan))
    num_matches = int(best_hit.get("num_matches", 0))
    ref_frags   = int(best_hit.get("ref_frags",   0))

    # Fragment match list
    try:
        mf       = json.loads(best_hit.get("matched_fragments", "[]"))
        frag_str = ", ".join(f"{m:.3f}" for m in mf) if mf else "N/A"
    except Exception:
        frag_str = "N/A"

    def _v(val, fmt):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return fmt.format(val)

    score_str = f"{score:.4f}\n{num_matches}/{ref_frags}" if not np.isnan(score) else "N/A"

    # Wrap the fragment list so it stays within the table width
    FRAG_WRAP_WIDTH = 80
    if frag_str != "N/A":
        frag_lines   = textwrap.wrap(frag_str, width=FRAG_WRAP_WIDTH)
        frag_display = "\n".join(frag_lines) if frag_lines else "N/A"
    else:
        frag_display = "N/A"
        frag_lines   = ["N/A"]
    n_frag_lines = min(len(frag_lines), 3)

    # Column x-positions: row-label | Theoretical | Measured | Error/Score
    col_x = [0, 0.3, 0.6, 0.8]

    # Layout constants
    std_row_h  = 0.135
    frag_row_h = std_row_h * max(1, n_frag_lines)

    # Row y-centres (header first, then rows stacked downward)
    header_center = 0.83
    ma_center     = header_center - std_row_h
    rt_center     = header_center - 2 * std_row_h
    frag_top      = rt_center - std_row_h / 2
    frag_center   = frag_top - frag_row_h / 2

    # File name as title (above the table) - aligned with table left edge
    ax.text(col_x[0], 0.97, name_without_ext, fontsize=14, weight="bold",
            ha="left", va="center", transform=ax.transAxes)

    GAP = 0.005
    for y_center, h, color in [
        (header_center, std_row_h,  "#d0d0d0"),
        (ma_center,     std_row_h,  "white"),
        (rt_center,     std_row_h,  "#f0f0f0"),
        (frag_center,   frag_row_h, "white"),
    ]:
        ax.add_patch(Rectangle(
            (0, y_center - h / 2 + GAP), 1.0, h - 2 * GAP,
            transform=ax.transAxes, color=color, zorder=0,
        ))

    for x, label in zip(col_x, ["BEST MATCH", "Theoretical", "Measured", "Error/Score"]):
        ax.text(x, header_center, label, fontsize=15, weight="bold", ha="left",
                va="center", transform=ax.transAxes)

    standard_rows = [
        (ma_center, ("Mass Accuracy",
                     _v(atlas_mz,    "{:.4f} m/z"),
                     _v(measured_mz, "{:.4f} m/z"),
                     _v(ppm_error,   "{:.2f} ppm"))),
        (rt_center, ("RT Accuracy",
                     _v(atlas_rt,    "{:.3f} min"),
                     _v(measured_rt, "{:.3f} min"),
                     _v(rt_error,    "{:.3f} min"))),
    ]
    for y_center, row_vals in standard_rows:
        for col_idx, val in enumerate(row_vals):
            ax.text(col_x[col_idx], y_center, val, fontsize=15,
                    weight="bold" if col_idx == 0 else "normal",
                    ha="left", va="center", transform=ax.transAxes)

    frag_text_top = frag_top - GAP * 2.5
    ax.text(col_x[0], frag_text_top, "Fragment Matches", fontsize=15,
            weight="bold", ha="left", va="top", transform=ax.transAxes)
    ax.text(col_x[1], frag_text_top, frag_display, fontsize=15,
            ha="left", va="top", transform=ax.transAxes)
    ax.text(col_x[3], frag_text_top, score_str, fontsize=15,
            ha="left", va="top", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def _plot_ms2(
    ax,
    panel_idx: int,
    top3: pd.DataFrame,
    ms2_raw_df: pd.DataFrame,
) -> None:
    """Populate one MS2 panel (mirror, raw, or empty) on *ax*.

    Parameters
    ----------
    panel_idx:
        0, 1, or 2 — which of the three top-hit panels to render.
    top3:
        Up to three best MS2 hits for the compound, sorted descending by score.
    ms2_raw_df:
        Raw MS2 scans for the compound (used as fallback when there are no hits).
    """
    _MIRROR_TITLES = ["Best MS2 Match", "2nd Best MS2 Match", "3rd Best MS2 Match"]
    title = _MIRROR_TITLES[panel_idx]

    if panel_idx < len(top3):
        hit = top3.iloc[panel_idx]
        qry_mz, qry_int = _parse_spectrum(hit.get("qry_spectrum"))
        ref_mz, ref_int = _parse_spectrum(hit.get("ref_spectrum"))

        raw_colors = hit.get("aligned_fragment_colors")
        frag_colors = None
        try:
            if raw_colors and pd.notnull(raw_colors):
                parsed = json.loads(raw_colors)
                if len(parsed) == len(qry_mz):
                    frag_colors = parsed
        except Exception:
            pass

        if qry_mz and ref_mz:
            _plot_mirror(
                ax, qry_mz, qry_int, ref_mz, ref_int,
                frag_colors=frag_colors,
                score=float(hit.get("score", np.nan)),
                rt=float(hit.get("rt", np.nan)),
                title=title,
            )
        elif qry_mz:
            _plot_raw_ms2(
                ax, qry_mz, qry_int,
                rt=float(hit.get("rt", np.nan)),
                title=title,
            )
        else:
            _plot_empty_ms2(ax, title=title)

    elif panel_idx == 0 and not ms2_raw_df.empty:
        raw_row = ms2_raw_df.iloc[0]
        mz_arr, int_arr = _parse_spectrum(raw_row.get("raw_spectrum"))
        _plot_raw_ms2(
            ax, mz_arr, int_arr,
            rt=float(raw_row.get("rt", np.nan)),
            title=title,
        )
    else:
        _plot_empty_ms2(ax, title=title)

def make_identification_figure(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
) -> None:
    """Generate per-compound identification figures for an analysis.

    Parameters
    ----------
    summary_obj:
        A configured ``AnalysisSummary`` object with ``paths``,
        ``rt_alignment_number``, and ``analysis_number`` already set
        (call ``summary_obj.setup(...)`` first).  Data tables are loaded
        automatically on first call via ``summary_obj.load_data()`` if
        they have not already been cached on the object.
    output_loc:
        Override output directory.  Defaults to
        ``<project_directory>/identification_figures/rt<N>_analysis<M>/``.
    overwrite:
        When *False*, skip compounds whose PDF already exists on disk.
    """
    main_db_path     = summary_obj.paths.get("main_db_path")

    # ── Get color mapping for EIC plots (from override or config) ───────────
    color_map = None
    if hasattr(summary_obj, 'override_parameters') and summary_obj.override_parameters.get('gui_lcmsruns_colors'):
        color_map = summary_obj.override_parameters['gui_lcmsruns_colors']
    elif hasattr(summary_obj, 'workflow_params') and summary_obj.workflow_params.get('gui_lcmsruns_colors'):
        color_map = summary_obj.workflow_params['gui_lcmsruns_colors']

    # ── Resolve output directory ─────────────────────────────────────────────
    if output_loc is None:
        raise ValueError("output_loc must be provided as a Path or string")
    else:
        output_loc = Path(output_loc, "identification_figures")
    output_loc.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting identification figures to %s", output_loc)

    # ── Ensure analysis data is available on the summary object ─────────────
    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms1_all_df         = summary_obj.ms1_all_df
    ms2_raw_all_df     = summary_obj.ms2_raw_all_df
    ms2_hits_all_df    = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    total_files = ms1_all_df["file_path"].nunique() if (ms1_all_df is not None and not ms1_all_df.empty) else 0
    logger.info(
        "Plotting %d compounds across %d files.",
        len(manual_curation_df), total_files,
    )

    for cmp_idx, mc_row in tqdm(manual_curation_df.reset_index(drop=True).iterrows(), total=len(manual_curation_df), desc="Generating ID figures"):
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"
        inchi_key     = mc_row.get("inchi_key", "")
        adduct        = mc_row.get("adduct", "")
        formula, smiles, inchi, _ = _get_compound_info(main_db_path, inchi_key) or (None, None, None, None)
        mc_row['formula'] = formula
        mc_row['smiles']  = smiles
        mc_row['inchi']   = inchi

        safe_name = f"{cmp_idx + 1:04d}_{compound_name}_{adduct}".replace("/", "-").replace(" ", "_")
        fig_path  = output_loc / f"{safe_name}.pdf"

        if not overwrite and fig_path.exists():
            logger.debug("Skipping %s (already exists).", fig_path)
            continue

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            return df[
                (df["inchi_key"] == inchi_key) & (df["adduct"] == adduct)
            ].reset_index(drop=True)

        ms1_df      = _filter(ms1_all_df)
        ms2_raw_df  = _filter(ms2_raw_all_df)
        ms2_hits_df = _filter(ms2_hits_all_df)

        if not ms2_hits_df.empty:
            top3 = (ms2_hits_df
                    .sort_values("score", ascending=False)
                    .drop_duplicates("file_path")
                    .head(3)
                    .reset_index(drop=True))
        else:
            top3 = pd.DataFrame()

        fig = plt.figure(figsize=(25, 15))
        gs  = fig.add_gridspec(
            3, 4,
            hspace=0.38, wspace=0.30,
            height_ratios=[1.45, 1.2, 1.35],
        )

        # Row 1, Cols 0-2: up to three MS2 hit panels (mirror, raw, or empty)
        for i in range(3):
            _plot_ms2(fig.add_subplot(gs[0, i]), i, top3, ms2_raw_df)

        # Row 1, Col 3: molecular structure
        ax_struct = fig.add_subplot(gs[0, 3])
        _plot_structure(ax_struct, mc_row.get('smiles', None), mc_row.get('inchi', None), inchi_key, size=500)

        # Row 2, Col 0: linear EIC
        ax_eic_lin = fig.add_subplot(gs[1, 0])
        _plot_eic(ax_eic_lin, ms1_df, mc_row, log_scale=False, color_map=color_map)
        ax_eic_lin.set_title("EIC (linear scale)", fontsize=18)

        # Row 2, Col 1: log-scale EIC
        ax_eic_log = fig.add_subplot(gs[1, 1])
        _plot_eic(ax_eic_log, ms1_df, mc_row, log_scale=True, color_map=color_map)
        ax_eic_log.set_title("EIC (log₁₀ scale)", fontsize=18)

        # Row 2, Cols 2-3: compound metadata table
        ax_info = fig.add_subplot(gs[1, 2:4])
        _plot_compound_info_table(ax_info, mc_row)

        # Row 3, all cols: MS2 hit summary
        ax_hits = fig.add_subplot(gs[2, 0:4])
        _plot_hit_info_table(ax_hits, ms2_hits_df, mc_row)

        # Figure-level title and section dividers
        plt.suptitle(f"[{cmp_idx + 1:04d}] |  {adduct}  |  {inchi_key}\n{compound_name}\n", fontsize=20, weight="bold", y=0.97)
        for y_line in [0.61, 0.345]:
            fig.add_artist(plt.Line2D(
                [0.08, 0.92], [y_line, y_line],
                transform=fig.transFigure,
                color="black", linewidth=1, clip_on=False,
            ))

        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Exported identification figure for %s → %s", compound_name, fig_path)

    logger.info("Identification figure export complete → %s", output_loc)

def _mz_quality(ppm_error: float, mz_delta: float) -> float:
    """Return 0/0.5/1 mz quality score from ppm and absolute mass error."""
    if np.isnan(ppm_error) or np.isnan(mz_delta):
        return np.nan
    if ppm_error <= 5 or mz_delta <= 0.0015:
        return 1.0
    if ppm_error <= 10:
        return 0.5
    return 0.0

def _rt_quality(rt_error: float, chromatography: str) -> float:
    """Return 0/0.5/1 RT quality score, with thresholds that depend on method."""
    if np.isnan(rt_error):
        return np.nan
    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()
    if is_c18:
        if rt_error <= 0.25:
            return 1.0
        if rt_error <= 0.5:
            return 0.5
        return 0.0
    else:
        if rt_error <= 0.5:
            return 1.0
        if rt_error <= 2.0:
            return 0.5
        return 0.0

def _total_score_and_msi(
    msms_q: float, mz_q: float, rt_q: float
) -> Tuple[float, str]:
    """Compute total ID score (0-3) and MSI level string.
    """
    scores = [v for v in [msms_q, mz_q, rt_q] if isinstance(v, float) and not np.isnan(v)]
    total = float(np.nansum([msms_q, mz_q, rt_q]))

    if isinstance(msms_q, float) and msms_q == -1:
        msi = "REMOVE, INVALIDATED BY BAD MSMS MATCH"
    elif len(scores) > 0 and statistics.median(scores) < 1:
        msi = "putative"
    elif total == 3:
        msi = "Exceeds Level 1"
    else:
        msi = "Level 1"
    return total, msi

def _find_overlapping_compounds(
    manual_curation_df: pd.DataFrame,
    compound_idx: int,
    main_db_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Find compounds with overlapping RT windows and similar m/z or monoisotopic mass.
    
    Returns a tuple of (compound_names, compound_indices) where each is a '//' separated string.
    Compounds are considered overlapping if:
    - Their RT windows overlap (rt_min to rt_max ranges intersect), AND
    - Either their m/z values are within 0.005 OR their monoisotopic masses are within 0.005
    """
    mc = manual_curation_df.reset_index(drop=True)
    row = mc.iloc[compound_idx]
    
    mz_target = row["atlas_mz"]
    rt_min_target = row["rt_min"]
    rt_max_target = row["rt_max"]
    inchi_key_target = row["inchi_key"]
    
    # Get monoisotopic mass for target compound
    mass_target = _get_monoisotopic_mass(main_db_path, inchi_key_target)
    
    overlapping: List[Tuple[int, str, str]] = []  # (index, name, inchi_key)
    
    for idx, other in mc.iterrows():
        if idx == compound_idx:
            continue
            
        # Check RT overlap: ranges [a1, a2] and [b1, b2] overlap if max(a1, b1) <= min(a2, b2)
        rt_min_other = other["rt_min"]
        rt_max_other = other["rt_max"]
        rt_overlap = (max(rt_min_target, rt_min_other) <= min(rt_max_target, rt_max_other))
        
        if not rt_overlap:
            continue
            
        # Check m/z similarity (exclude if both are 0 or invalid)
        mz_other = other["atlas_mz"]
        mz_similar = False
        if mz_target != 0 and mz_other != 0 and not np.isnan(mz_target) and not np.isnan(mz_other):
            mz_similar = abs(mz_target - mz_other) <= 0.005
        
        # Check monoisotopic mass similarity (exclude if both are 0 or invalid)
        mass_similar = False
        if mass_target is not None and mass_target != 0 and main_db_path:
            inchi_key_other = other["inchi_key"]
            mass_other = _get_monoisotopic_mass(main_db_path, inchi_key_other)
            if mass_other is not None and mass_other != 0:
                mass_similar = abs(mass_target - mass_other) <= 0.005
        
        if mz_similar or mass_similar:
            overlapping.append((int(idx) + 1, other["compound_name"], other["inchi_key"]))
    
    # Include the target compound itself
    overlapping.append((compound_idx + 1, row["compound_name"], row["inchi_key"]))
    
    # Sort by index
    overlapping.sort(key=lambda x: x[0])
    
    # Format as names only (no InChI keys) and indices
    names = [name for _, name, _ in overlapping]
    indices = [str(idx) for idx, _, _ in overlapping]
    
    return "//".join(names), "//".join(indices)

def make_final_id_sheet(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "Final_Identifications.xlsx",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Build the per-compound summary Excel workbook for one analysis.

    Reads ``manual_curation`` and ``ms2_hits`` from the project database and
    produces a single Excel file: **Final_Identifications** - one row per compound with quality scores,
      MS1/MS2 summary values, mz/RT metrics, and formatted section headers.

    Parameters
    ----------
    summary_obj:
        A configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
    output_loc:
        Override the output directory.  Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/``.
    output_filename:
        Name of the Excel file to write.
    overwrite:
        When *False*, raises if the output file already exists.
    """
    chromatography = summary_obj.chromatography

    if output_loc is None:
        raise ValueError("output_loc must be provided as a Path or string")
    else:
        output_loc = Path(output_loc)
    output_loc.mkdir(parents=True, exist_ok=True)

    analysis_info = f"{summary_obj.post_curation_atlas_obj.chromatography}-{summary_obj.post_curation_atlas_obj.polarity}-{summary_obj.post_curation_atlas_obj.analysis_type}"
    run_info = f"RTA{summary_obj.rt_alignment_number}-TGA{summary_obj.analysis_number}"
    output_filename = f"{summary_obj.project_name}_{analysis_info}-{run_info}_{output_filename}"
    if not output_filename.endswith(".xlsx"):
        output_filename += ".xlsx"
    excel_path = output_loc / output_filename

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms2_hits_all_df    = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to export.")
        return pd.DataFrame()

    manual_curation_df = manual_curation_df.reset_index(drop=True)
    logger.info("Processing %d compounds...", len(manual_curation_df))

    rows: List[dict] = []

    for compound_idx, mc_row in tqdm(manual_curation_df.iterrows(), total=len(manual_curation_df), desc="Adding compounds to final ID sheet"):
        compound_name = mc_row.get("compound_name") or f"compound_{compound_idx}"
        inchi_key     = mc_row.get("inchi_key",   "")
        adduct        = mc_row.get("adduct",       "")
        polarity      = mc_row.get("polarity",     "")
        
        # Get compound metadata from main database
        formula, smiles, inchi, pubchem_cid = _get_compound_info(summary_obj.paths.get("main_db_path"), inchi_key) or (None, None, None, None)
        exact_mass = _get_monoisotopic_mass(summary_obj.paths.get("main_db_path"), inchi_key)
        
        # Find overlapping compounds
        overlapping_names, overlapping_indices = _find_overlapping_compounds(
            manual_curation_df, compound_idx, summary_obj.paths.get("main_db_path")
        )
        
        mz_theoretical = float(mc_row.get("atlas_mz",            np.nan))
        mz_measured    = float(mc_row.get("best_ms1_mz",         np.nan))
        ppm_error      = float(mc_row.get("best_ms1_ppm_error",  np.nan))
        rt_error       = float(mc_row.get("best_ms1_rt_error",   np.nan))
        rt_measured    = float(mc_row.get("best_ms1_rt",         np.nan))
        max_intensity  = float(mc_row.get("best_ms1_intensity",  np.nan))
        max_int_file   = mc_row.get("best_ms1_file", "")

        mz_q = _mz_quality(ppm_error, abs(mz_theoretical - mz_measured) if not (np.isnan(mz_theoretical) or np.isnan(mz_measured)) else np.nan)
        rt_q = _rt_quality(rt_error, chromatography)

        ms2_notes = mc_row.get("ms2_notes") or "no selection"
        try:
            msms_q = float(str(ms2_notes).split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = _total_score_and_msi(msms_q, mz_q, rt_q)

        msms_file          = ""
        msms_rt            = np.nan
        msms_score         = np.nan
        msms_num_ions      = ""
        msms_matching_ions = ""

        if not ms2_hits_all_df.empty:
            comp_hits = ms2_hits_all_df[
                (ms2_hits_all_df["inchi_key"] == inchi_key) &
                (ms2_hits_all_df["adduct"]    == adduct)
            ]
            if not comp_hits.empty:
                best_hit = comp_hits.sort_values("score", ascending=False).iloc[0]
                msms_file  = str(best_hit.get("file_path", ""))
                msms_rt    = float(best_hit.get("rt",    np.nan))
                msms_score = float(best_hit.get("score", np.nan))
                
                num_matches = int(best_hit.get("num_matches", 0))
                ref_frags   = int(best_hit.get("ref_frags", 0))
                msms_num_ions = f"{num_matches}/{ref_frags}" if ref_frags > 0 else str(num_matches)

                try:
                    mf = json.loads(best_hit.get("matched_fragments", "[]"))
                    msms_matching_ions = ",".join(f"{m:.3f}" for m in mf)
                except Exception:
                    msms_matching_ions = ""

                if num_matches == 1 and not np.isnan(msms_score):
                    try:
                        mf = json.loads(best_hit.get("matched_fragments", "[]"))
                        single_ion = mf[0] if mf else np.nan
                    except Exception:
                        single_ion = np.nan
                    ppm_tol    = float(mc_row.get("mz_tolerance", 5.0))
                    precursor_match = (
                        abs(single_ion - mz_theoretical) / mz_theoretical * 1e6 <= ppm_tol
                        if (not np.isnan(single_ion) and mz_theoretical > 0) else False
                    )
                    note_tag = " (single matching fragment is the precursor)" if precursor_match else ""
                    if "1.0, single ion match" in ms2_notes or "0.5, single ion match" in ms2_notes:
                        ms2_notes = ms2_notes + note_tag
                    else:
                        ms2_notes = (
                            "Unannotated single ion match, needs review. "
                            "Setting MSMS quality to 0.5. "
                            f"Original annotation: {ms2_notes}"
                        )
                        msms_q     = 0.5
                        total_score, msi_level = _total_score_and_msi(msms_q, mz_q, rt_q)

        rows.append({
            "identified_metabolite":   overlapping_names,
            "identified_metabolite_idx": overlapping_indices,
            "label":                   compound_name,
            "compound_number":         compound_idx + 1,
            "mz_adduct":               adduct,
            "mz_adduct_mz":            round(mz_theoretical, 4) if not np.isnan(mz_theoretical) else np.nan,
            "pubchem_cid":             pubchem_cid if pubchem_cid else "",
            "smiles":                  smiles if smiles else "",
            "inchi":                   inchi if inchi else "",
            "formula":                 formula if formula else "",
            "exact_mass":              round(exact_mass, 6) if exact_mass is not None else np.nan,
            "inchi_key":               inchi_key,
            "polarity":                polarity,
            "chromatography":          chromatography,
            "msms_quality":            msms_q,
            "mz_quality":              mz_q,
            "rt_quality":              rt_q,
            "total_score":             total_score,
            "msi_level":               msi_level,
            "identification_notes":    mc_row.get("identification_notes", ""),
            "analyst_notes":           mc_row.get("analyst_notes", ""),
            "other_notes":             mc_row.get("other_notes",   ""),
            "ms1_notes":               mc_row.get("ms1_notes",            ""),
            "ms2_notes":               ms2_notes,
            "max_intensity":           max_intensity,
            "max_intensity_file":      max_int_file,
            "ms1_rt_peak":             rt_measured,
            "msms_file":               msms_file,
            "msms_rt":                 round(msms_rt, 2)    if not np.isnan(msms_rt)    else np.nan,
            "msms_numberofions":       msms_num_ions,
            "msms_matchingions":       msms_matching_ions,
            "msms_score":              round(msms_score, 4) if not np.isnan(msms_score) else np.nan,
            "mz_theoretical":          round(mz_theoretical, 4) if not np.isnan(mz_theoretical) else np.nan,
            "mz_measured":             round(mz_measured,    4) if not np.isnan(mz_measured)    else np.nan,
            "mz_ppmerror":             round(ppm_error,      4) if not np.isnan(ppm_error)      else np.nan,
            "rt_min":                  round(float(mc_row.get("rt_min", np.nan)), 2),
            "rt_max":                  round(float(mc_row.get("rt_max", np.nan)), 2),
            "rt_theoretical":          round(float(mc_row.get("atlas_rt_peak", np.nan)), 2),
            "rt_measured":             round(rt_measured, 2) if not np.isnan(rt_measured) else np.nan,
            "rt_error":                round(rt_error,    2) if not np.isnan(rt_error)    else np.nan,
        })

    final_df = pd.DataFrame(rows)
    logger.info("Assembled stats table with %d rows.", len(final_df))

    if not overwrite and excel_path.exists():
        raise FileExistsError(
            f"Output file already exists and overwrite=False: {excel_path}"
        )

    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()

    HEADER2 = [
        "Identified Metabolite(s) Name", "Identified Metabolite(s) #",
        "Target Metabolite Name", "Target Metabolite #", "Target Metabolite Adduct",
        "Target Metabolite+Adduct m/z",
        "Target Metabolite PubChem CID", "Target Metabolite SMILES", "Target Metabolite InChI",
        "Target Metabolite Formula", "Target Metabolite Mass", "Target Metabolite InChIKey",
        "Run Polarity", "Run Chromatography",
        "MSMS Score (0 to 1)", "m/z score (0 to 1)", "RT score (0 to 1)",
        "Total ID Score (0 to 3)", "Mass Spec Initiative ID Level",
        "Identification notes", "Analyst notes", "Other notes", "MS1 notes", "MS2 notes",
        "Maximum MS1 intensity across all files", "Filename w/ maximum MS1",
        "Retention time of max intensity MS1 peak",
        "File with highest MSMS match score", "RT of highest matched MSMS scan",
        "Number of ion matches in MSMS spectra to reference",
        "List of ion matches in MSMS spectra to reference",
        "MSMS score (highest across all samples)",
        "Theoretical m/z", "Measured m/z", "Mass error (delta ppm)",
        "Min retention time (min)", "Max retention time (min)",
        "Theoretical retention time (peak)",
        "Detected RT (peak)", "RT error (absolute delta)",
    ]

    rt_q_desc = (
        "1 (delta RT ≤0.25), 0.5 (delta RT >0.25 & ≤0.5), 0 (delta RT >0.5 min)"
        if is_c18 else
        "1 (delta RT ≤0.5), 0.5 (delta RT >0.5 & ≤2), 0 (delta RT >2 min)"
    )
    HEADER3 = [
        "Name of final identification. Some compounds (i.e., isomers) are not chromatographically or spectrally resolvable and are separated by \"//\". Some compounds are detected w/ >1 adduct (increases identification confidence but only use 1 for analysis). \"Unresolvable\" determined by having similar m/z (abs difference <= 0.005) or monoisotopic molecular weight (abs difference <= 0.005) AND overlapping RT (min or max within the RT-min-max-range of similar compound)",
        "Index of the metabolite(s) in nextdoor column, unique for study",
        "Name and InChIKey of standard reference compound that was searched for in the spectral data",
        "Index of the metabolite in nextdoor column, unique for study",
        "",
        "m/z of the compound with the specified adduct",
        "", "", "",
        "Molecular formula from PubChem",
        "Monoisotopic molecular mass (neutral except for permanently charged molecules)",
        "Neutralized version of InChIKey",
        "Metabolite detected in this instrument polarity",
        "Metabolite detected in this column chromatography",
        "1 (MSMS matches ref. std.), 0.5 (possible match), 0 (no MSMS or no ref.), -1 (bad match)",
        "1 (delta ppm ≤5 or delta Da ≤0.0015), 0.5 (delta ppm 5-10), 0 (>10 ppm)",
        rt_q_desc,
        "Sum of m/z, RT and MSMS scores",
        "Level 1 = Two orthogonal properties match authentic standard; else = putative",
        "", "", "", "", "",
        "Highest MS1 peak height across all files in the analysis",
        "File containing the highest MS1 peak height",
        "Retention time of the highest MS1 peak",
        "File with the top-scoring MS2 database hit",
        "Retention time of the top-scoring MS2 scan",
        "Number of matched ions / number of reference ions",
        "m/z values of matching fragment ions",
        "Highest MS2 hit score across all files (0-1)",
        "Theoretical m/z for compound / adduct pair",
        "Measured m/z (best MS1 scan)",
        "PPM difference between theoretical and measured m/z",
        "Atlas RT window minimum",
        "Atlas RT window maximum",
        "Theoretical (atlas) retention time at peak",
        "Measured retention time at peak (best MS1 scan)",
        "Absolute difference between theoretical and measured RT",
    ]

    _SECTION_SPANS = [
        ("A1:L1",  "COMPOUND ANNOTATION"),
        ("M1:N1",  "RUN DETAILS"),
        ("O1:S1",  "COMPOUND IDENTIFICATION SCORES"),
        ("T1:X1",  "ANNOTATION NOTES"),
        ("Y1:AA1",  "MS1 INTENSITY INFORMATION"),
        ("AB1:AF1", "MSMS INFORMATION"),
        ("AG1:AI1","ION / M/Z INFORMATION"),
        ("AJ1:AN1","CHROMATOGRAPHIC PEAK INFORMATION"),
    ]

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        final_df.to_excel(
            writer, sheet_name="Final_Identifications", index=False, startrow=3
        )
        workbook  = writer.book
        worksheet = writer.sheets["Final_Identifications"]

        # Cell formats
        f_blue       = workbook.add_format({"bg_color": "#DCFFFF"})
        f_yellow     = workbook.add_format({"bg_color": "#FFFFDC"})
        f_rose       = workbook.add_format({"bg_color": "#FFDCFF"})
        f_gray       = workbook.add_format({"bg_color": "#D3D3D3"})
        f_header     = workbook.add_format({
            "bold": True, "align": "center", "valign": "vcenter",
            "text_wrap": True, "border": 1,
        })
        f_scientific = workbook.add_format({"num_format": "0.00E+00"})

        worksheet.set_row(1, 60)
        worksheet.set_row(2, 60)
        worksheet.set_column("A:A", 25)
        worksheet.set_column("C:C", 25)
        worksheet.set_column("L:L", 25)
        worksheet.set_column("Y:Y", None, f_scientific)

        # Row 1: section header merges
        for span, title in _SECTION_SPANS:
            start_col, end_col = span.split(":")[0], span.split(":")[1]
            if start_col[:-1] == end_col[:-1] and start_col[-1] == end_col[-1]:
                # single cell
                worksheet.write(span.split(":")[0], title, f_header)
            else:
                worksheet.merge_range(span, title, f_header)

        # Rows 2 and 3: column headers and descriptions
        for i, h in enumerate(HEADER2):
            worksheet.write(1, i, h, f_header)
        for i, h in enumerate(HEADER3):
            worksheet.write(2, i, h, f_header)

        # Conditional background colours per section
        nrows = len(final_df) + 4
        worksheet.conditional_format(f"M1:N{nrows}", {"type": "no_errors", "format": f_gray})
        worksheet.conditional_format(f"O1:S{nrows}", {"type": "no_errors", "format": f_blue})
        worksheet.conditional_format(f"T1:X{nrows}", {"type": "no_errors", "format": f_rose})
        worksheet.conditional_format(f"Y1:AA{nrows}", {"type": "no_errors", "format": f_yellow})
        worksheet.conditional_format(f"AB1:AF{nrows}", {"type": "no_errors", "format": f_rose})
        worksheet.conditional_format(f"AG1:AI{nrows}", {"type": "no_errors", "format": f_yellow})
        worksheet.conditional_format(f"AJ1:AN{nrows}", {"type": "no_errors", "format": f_yellow})

    logger.info("Exported stats table to %s", excel_path)
    return final_df

_GRID_COLS = 5
_GRID_ROWS = 5
_PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS

def _short_fname(file_path: str) -> str:
    """Return an abbreviated filename label (stem after the 11th ``_``)."""
    if not file_path:
        return "no data"
    stem  = os.path.basename(file_path).split(".")[0]
    parts = stem.split("_")
    return "_".join(parts[11:]) if len(parts) > 11 else stem

def _render_eic_thumbnail(
    ax,
    rt_arr: List,
    i_arr:  List,
    rt_min:  float,
    rt_peak: float,
    rt_max:  float,
    fname_short: str,
    y_max: Optional[float],
) -> None:
    """Draw one EIC thumbnail onto *ax*.

    Parameters
    ----------
    y_max:
        When given, fixes the y-axis upper limit (shared-scale mode).
        When *None*, the axis auto-scales to the data (independent mode).
    """

    if rt_arr and i_arr:
        ax.plot(rt_arr, i_arr, color="steelblue", linewidth=0.8)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray")

    # RT vlines: rt_min = red dashed, rt_peak = black dotted, rt_max = black dashed
    _vline = lambda val, color, ls: (
        ax.axvline(val, color=color, linewidth=0.8, linestyle=ls)
        if val is not None and not (isinstance(val, float) and np.isnan(val))
        else None
    )
    _vline(rt_min,  "red",   "--")
    _vline(rt_peak, "black", ":" )
    _vline(rt_max,  "black", "--")

    if y_max is not None and y_max > 0:
        ax.set_ylim(bottom=0, top=y_max * 1.05)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax.tick_params(axis="both", labelsize=5, length=2, pad=1)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    ax.set_title(fname_short, fontsize=10, pad=2, loc="center")

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

def _write_compound_eic_pdf(
    pdf_path: Path,
    compound_name: str,
    adduct: str,
    rt_alignment_num: int,
    analysis_num: int,
    mc_row: pd.Series,
    file_items: List[dict],
    shared_y: bool,
    compound_idx: int = 0,
) -> None:
    """Write a single-compound EIC thumbnail PDF.

    *file_items* is a list of dicts with keys ``file_path``, ``rt_arr``, ``i_arr``.
    """

    rt_min  = mc_row.get("rt_min",        np.nan)
    rt_max  = mc_row.get("rt_max",        np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    if shared_y:
        all_intensities = [v for item in file_items for v in item["i_arr"] if v is not None]
        y_max_global = float(max(all_intensities)) if all_intensities else None
    else:
        y_max_global = None

    total_files = len(file_items)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)
    title_base  = f"[{compound_idx:04d}] {compound_name} | {adduct}   (RT alignment {rt_alignment_num}, analysis {analysis_num})"

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(total_pages):
            start      = page_idx * _PLOTS_PER_PAGE
            end        = min(start + _PLOTS_PER_PAGE, total_files)
            page_items = file_items[start:end]
            n_on_page  = len(page_items)

            fig, axes = plt.subplots(
                _GRID_ROWS, _GRID_COLS,
                figsize=(20, 16),
                constrained_layout=True,
            )
            axes_flat = axes.flatten()

            for slot_idx, item in enumerate(page_items):
                ax       = axes_flat[slot_idx]
                rt_arr   = item["rt_arr"]
                i_arr    = item["i_arr"]
                fname_s  = _short_fname(item["file_path"])

                if shared_y:
                    y_max = y_max_global
                else:
                    y_max = float(max(i_arr)) if i_arr else None

                _render_eic_thumbnail(
                    ax, rt_arr, i_arr,
                    rt_min, rt_peak, rt_max,
                    fname_s, y_max,
                )

            for slot_idx in range(n_on_page, _PLOTS_PER_PAGE):
                axes_flat[slot_idx].set_visible(False)

            page_label = f"({page_idx + 1}/{total_pages})" if total_pages > 1 else ""
            fig.suptitle(f"{title_base}  {page_label}", fontsize=12, y=1.005)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

def make_eic_thumbnails(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
) -> None:
    """Generate per-compound EIC thumbnail PDFs in two output folders.

    Produces **one PDF per compound** in each of two sub-directories:

    * ``eic_thumbnails_shared_y/``   — all files for a compound share the same
      y-axis upper limit (global maximum across that compound's files), making
      it easy to compare relative intensities across files.
    * ``eic_thumbnails_independent_y/`` — each file is auto-scaled to its own
      maximum intensity, showing peak shape detail even for low-intensity files.

    Each page within a PDF holds a **5 x 5 grid** of thumbnails (25 per page),
    so a compound with 50 files produces a 2-page PDF.

    Each thumbnail shows:

    * The EIC trace (blue line)
    * ``rt_min``  → red dashed vertical line
    * ``rt_peak`` → black dotted vertical line
    * ``rt_max``  → black dashed vertical line
    * Tick values on both axes, no axis labels
    * Y-axis in scientific notation with the x10ⁿ scale factor above the axis
    * Short filename as the subplot title

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
    output_loc:
        Override base output directory.  Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/``.
    overwrite:
        When *False*, skips compound PDFs that already exist in both folders.
    """
    project_db_path  = summary_obj.paths["project_db_path"]
    rt_alignment_num = summary_obj.rt_alignment_number
    analysis_num     = summary_obj.analysis_number

    # ── Resolve base output directory ────────────────────────────────────────
    if output_loc is None:
        raise ValueError("output_loc must be provided as a Path or string")
    else:
        base_dir = Path(output_loc)

    dir_shared = base_dir / "eic_thumbnails_shared_y"
    dir_indep  = base_dir / "eic_thumbnails_independent_y"
    dir_shared.mkdir(parents=True, exist_ok=True)
    dir_indep.mkdir(parents=True, exist_ok=True)

    # ── Ensure analysis data is available on the summary object ─────────────
    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms1_all_df         = summary_obj.ms1_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    manual_curation_df = manual_curation_df.reset_index(drop=True)
    n_compounds = len(manual_curation_df)
    logger.info("Generating EIC thumbnail PDFs for %d compounds...", n_compounds)
    for cmp_idx, mc_row in tqdm(manual_curation_df.iterrows(), total=len(manual_curation_df), desc="Generating EIC thumbnails"):
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"
        inchi_key     = mc_row.get("inchi_key", "")
        adduct        = mc_row.get("adduct",    "")

        safe_stem = f"{cmp_idx + 1:04d}_{compound_name}_{adduct}".replace("/", "-").replace(" ", "_")

        path_shared = dir_shared / f"{safe_stem}.pdf"
        path_indep  = dir_indep  / f"{safe_stem}.pdf"

        if not overwrite and path_shared.exists() and path_indep.exists():
            logger.debug("Skipping %s (both PDFs exist).", safe_stem)
            continue

        if ms1_all_df.empty:
            ms1_cmp = pd.DataFrame()
        else:
            ms1_cmp = ms1_all_df[
                (ms1_all_df["inchi_key"] == inchi_key) &
                (ms1_all_df["adduct"]    == adduct)
            ].reset_index(drop=True)

        if ms1_cmp.empty:
            file_items = [{"file_path": "", "rt_arr": [], "i_arr": []}]
        else:
            file_items = []
            for _, file_row in ms1_cmp.iterrows():
                rt_arr, i_arr = _parse_spectrum(file_row.get("raw_spectrum"))
                file_items.append({
                    "file_path": str(file_row.get("file_path", "")),
                    "rt_arr":    rt_arr,
                    "i_arr":     i_arr,
                })

        # shared-y PDF
        if overwrite or not path_shared.exists():
            _write_compound_eic_pdf(
                path_shared, compound_name, adduct,
                rt_alignment_num, analysis_num,
                mc_row, file_items, shared_y=True,
                compound_idx=cmp_idx + 1,
            )

        # independent-y PDF
        if overwrite or not path_indep.exists():
            _write_compound_eic_pdf(
                path_indep, compound_name, adduct,
                rt_alignment_num, analysis_num,
                mc_row, file_items, shared_y=False,
                compound_idx=cmp_idx + 1,
            )

        logger.debug("Wrote EIC PDFs for %s", compound_name)

def _file_group(file_path: str) -> str:
    """Return the file group label: the segment at underscore-separated index 11 of the stem.

    Example: ``..._groupA_...mzML`` → ``"groupA"`` (assuming groupA is at position 11).
    Falls back to the full stem when the filename has fewer than 12 ``_`` segments.
    """
    if not file_path:
        return "unknown"
    stem  = os.path.basename(file_path).split(".")[0]
    parts = stem.split("_")
    return parts[12] if len(parts) > 12 else stem

def extract_per_file_metrics(ms1_all_df: pd.DataFrame) -> pd.DataFrame:
    """Derive per-file summary metrics from the stored ms1_data rows.

    Iterates once over all ms1_data rows and computes three scalar metrics per
    (compound x file) directly from the JSON spectra stored in the database —
    no re-extraction from parquet files is needed.

    Metrics computed
    ----------------
    peak_height
        Maximum intensity value in the EIC (``max(i_array)``).
    rt_peak
        Retention time at the intensity maximum (``rt_array[argmax(i_array)]``).
    mz_centroid
        Intensity-weighted mean m/z across all scans in the EIC window:
        ``sum(mz_i x i_i) / sum(i_i)``, using the per-scan m/z list stored in
        the ``mz`` column alongside the intensity list from ``raw_spectrum``.

    Parameters
    ----------
    ms1_all_df:
        Full ``ms1_data`` table for the analysis (all compounds, all files).

    Returns
    -------
    pd.DataFrame with columns:
        ``inchi_key``, ``adduct``, ``file_path``, ``file_group``,
        ``peak_height``, ``rt_peak``, ``mz_centroid``
    """
    if ms1_all_df.empty:
        return pd.DataFrame(columns=[
            "inchi_key", "adduct", "file_path", "file_group",
            "peak_height", "rt_peak", "mz_centroid",
        ])

    records = []
    for _, row in ms1_all_df.iterrows():
        rt_arr, i_arr = _parse_spectrum(row.get("raw_spectrum"))
        fp    = str(row.get("file_path", ""))
        group = _file_group(fp)
        base  = {
            "inchi_key":  row["inchi_key"],
            "adduct":     row["adduct"],
            "file_path":  fp,
            "file_group": group,
        }

        if not i_arr or max(i_arr) <= 0:
            records.append({**base, "peak_height": np.nan, "rt_peak": np.nan, "mz_centroid": np.nan})
            continue

        i_np  = np.array(i_arr,  dtype=float)
        rt_np = np.array(rt_arr, dtype=float)
        idx_max     = int(np.argmax(i_np))
        peak_height = float(i_np[idx_max])
        rt_peak_val = float(rt_np[idx_max])

        mz_centroid = np.nan
        mz_json = row.get("mz")
        if mz_json is not None and not (isinstance(mz_json, float) and np.isnan(mz_json)):
            try:
                mz_np   = np.array(json.loads(mz_json), dtype=float)
                total_i = i_np.sum()
                if len(mz_np) == len(i_np) and total_i > 0:
                    mz_centroid = float((mz_np * i_np).sum() / total_i)
            except Exception:
                pass

        records.append({**base,
                        "peak_height": peak_height,
                        "rt_peak":     rt_peak_val,
                        "mz_centroid": mz_centroid})

    return pd.DataFrame(records)

def _plot_compound_boxplot(
    ax,
    compound_metrics: pd.DataFrame,
    metric: str,
    log_scale: bool,
    atlas_ref: Optional[float],
    compound_name: str,
    adduct: str,
    ylabel: str,
    compound_idx: int = 0,
) -> None:
    """Draw a grouped boxplot for one compound onto *ax*.

    Parameters
    ----------
    compound_metrics:
        Rows from ``per_file_df`` already filtered to one (inchi_key, adduct).
    metric:
        Column name to plot: ``"peak_height"``, ``"rt_peak"``, or ``"mz_centroid"``.
    log_scale:
        When *True* the y-values are log₁₀-transformed before plotting.
    atlas_ref:
        Atlas reference value to draw as a red dashed horizontal line
        (pass *None* to omit the line).
    compound_name, adduct:
        Used as the subplot title.
    ylabel:
        Y-axis label string.
    """
    rng    = np.random.default_rng(seed=42)
    groups = sorted(compound_metrics["file_group"].dropna().unique()) if not compound_metrics.empty else []

    data_per_group: List[List[float]] = []
    valid_groups:   List[str]         = []
    for g in groups:
        vals = compound_metrics.loc[compound_metrics["file_group"] == g, metric].dropna().tolist()
        if log_scale:
            vals = [np.log10(v) for v in vals if isinstance(v, (int, float)) and v > 0]
        if vals:
            data_per_group.append(vals)
            valid_groups.append(g)

    if not data_per_group:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        ax.set_title(f"[{compound_idx:04d}] {compound_name}\n{adduct}", fontsize=7, pad=2)
        return

    positions = list(range(len(valid_groups)))

    ax.boxplot(
        data_per_group,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="#AED6F1", alpha=0.7),
        medianprops=dict(color="navy", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        showfliers=False,
    )

    for pos, vals in zip(positions, data_per_group):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            np.array([pos] * len(vals), dtype=float) + jitter,
            vals,
            s=8, color="steelblue", alpha=0.65, zorder=3,
        )

    if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)):
        ref_val = np.log10(atlas_ref) if (log_scale and atlas_ref > 0) else atlas_ref
        ax.axhline(ref_val, color="red", linestyle="--", linewidth=0.8, alpha=0.8,
                   label=f"Atlas {atlas_ref:.5g}")
        ax.legend(fontsize=5, loc="upper right", framealpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(valid_groups, rotation=45, ha="right", fontsize=5)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.set_title(f"[{compound_idx:04d}] {compound_name}\n{adduct}", fontsize=7, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

def make_boxplots(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
) -> None:
    """Generate per-compound boxplot PDFs organised into six metric-type folders.

    Six sub-directories are created under ``<output_loc>/boxplots/``, one for
    each metric x scale combination:

    * ``peak_height_linear/``
    * ``peak_height_log/``
    * ``rt_peak_linear/``
    * ``rt_peak_log/``
    * ``mz_centroid_linear/``
    * ``mz_centroid_log/``

    Inside each folder one single-plot PDF is written per compound, named
    ``<compound_name>_<adduct>.pdf``.  Each PDF contains a single boxplot
    showing file groups on the x-axis with box-and-whisker plots and
    individual data points overlaid.  A red dashed atlas reference line is
    drawn for ``mz_centroid`` (atlas m/z) and ``rt_peak`` (atlas RT peak).

    All three metrics are derived directly from the ``ms1_data`` table rows
    stored in the database — no re-extraction from raw parquet is needed.

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
        Data tables and per-file metrics are loaded automatically via
        ``summary_obj.load_data()`` on first call if not already cached.
    output_loc:
        Override output directory.  Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/boxplots/``.
    overwrite:
        When *False*, skips PDF files that already exist on disk.
    """

    rt_alignment_num = summary_obj.rt_alignment_number
    analysis_num     = summary_obj.analysis_number

    if output_loc is None:
        raise ValueError("output_loc must be provided for boxplot output.")
    else:
        output_loc = Path(output_loc, "boxplots")
    output_loc.mkdir(parents=True, exist_ok=True)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    per_file_df        = summary_obj.per_file_metrics_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    manual_curation_df = manual_curation_df.reset_index(drop=True)

    atlas_lookup: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, mc in manual_curation_df.iterrows():
        key = (mc.get("inchi_key", ""), mc.get("adduct", ""))
        atlas_lookup[key] = {
            "atlas_mz":      float(mc.get("atlas_mz",      np.nan)),
            "atlas_rt_peak": float(mc.get("atlas_rt_peak", np.nan)),
        }

    _METRIC_CONFIGS = [
        ("peak_height",  False, "Peak Height (intensity)",   None),
        ("peak_height",  True,  "Peak Height (log₁₀)",       None),
        ("rt_peak",      False, "RT Peak (min)",              "atlas_rt_peak"),
        ("rt_peak",      True,  "RT Peak log₁₀(min)",        "atlas_rt_peak"),
        ("mz_centroid",  False, "m/z Centroid",              "atlas_mz"),
        ("mz_centroid",  True,  "m/z Centroid (log₁₀)",     "atlas_mz"),
    ]

    n_compounds = len(manual_curation_df)
    total_tasks = len(_METRIC_CONFIGS) * n_compounds

    with tqdm(total=total_tasks, desc=f"Generating boxplots ({len(_METRIC_CONFIGS)} types x {n_compounds} compounds)") as pbar:
        for metric, log_scale, ylabel, atlas_attr in _METRIC_CONFIGS:
            scale_tag   = "log" if log_scale else "linear"
            folder_name = f"{metric}_{scale_tag}"
            metric_dir  = output_loc / folder_name
            metric_dir.mkdir(parents=True, exist_ok=True)

            for cmp_idx, mc_row in manual_curation_df.iterrows():
                compound_name = mc_row.get("compound_name") or "unknown"
                adduct        = mc_row.get("adduct", "")
                inchi_key     = mc_row.get("inchi_key", "")

                safe_stem = f"{cmp_idx + 1:04d}_{compound_name}_{adduct}".replace("/", "-").replace(" ", "_")
                pdf_path  = metric_dir / f"{safe_stem}.pdf"

                pbar.update(1)

                if not overwrite and pdf_path.exists():
                    logger.debug("Skipping %s (already exists).", pdf_path)
                    continue

                atlas_ref = atlas_lookup.get((inchi_key, adduct), {}).get(atlas_attr) if atlas_attr else None

                cmp_metrics = (
                    per_file_df[
                        (per_file_df["inchi_key"] == inchi_key) &
                        (per_file_df["adduct"]    == adduct)
                    ]
                    if not per_file_df.empty
                    else pd.DataFrame()
                )

                fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

                _plot_compound_boxplot(
                    ax, cmp_metrics, metric, log_scale,
                    atlas_ref, compound_name, adduct, ylabel,
                    compound_idx=cmp_idx + 1,
                )

                fig.suptitle(
                    f"{ylabel}  |  RT alignment {rt_alignment_num}, analysis {analysis_num}",
                    fontsize=9,
                )

                with PdfPages(pdf_path) as pdf:
                    pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    logger.info("Boxplot PDFs complete → %s", output_loc)

def make_manual_curation_csv(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "manually_curated_compound_data.csv",
    overwrite: bool = True,
) -> pd.DataFrame:
    """Write the ``manual_curation`` table to a CSV file (one row per compound).

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
        ``manual_curation_df`` is loaded automatically if not already cached.
    output_loc:
        Override output directory.  Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/``.
    output_filename:
        CSV filename (the ``.csv`` extension is appended automatically if absent).
    overwrite:
        When *False*, raises :exc:`FileExistsError` if the output file already
        exists.

    Returns
    -------
    pd.DataFrame
        The exported DataFrame (empty on error).
    """

    if output_loc is None:
        raise ValueError("output_loc must be provided for manual curation CSV output.")
    else:
        output_loc = Path(output_loc)
    output_loc.mkdir(parents=True, exist_ok=True)

    if not output_filename.endswith(".csv"):
        output_filename += ".csv"
    csv_path = output_loc / output_filename

    if not overwrite and csv_path.exists():
        raise FileExistsError(
            f"Output file already exists and overwrite=False: {csv_path}"
        )

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - CSV not written.")
        return pd.DataFrame()

    manual_curation_df.to_csv(csv_path, index=False)
    logger.info("Exported manual curation CSV to %s", csv_path)
    return manual_curation_df


def make_best_ms2_hit_fragment_ions_csv(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "best_ms2_hit_fragment_ions.csv",
    overwrite: bool = True,
    min_fragment_intensity: Optional[float] = 1e4,
) -> pd.DataFrame:
    """Write a CSV with the top-scoring MS2 hit fragment spectrum for each compound.

    One row per compound (matched to each row in ``manual_curation_df``).
    Compounds that have no MS2 hits are omitted.

    Columns
    -------
    compound_index  : 1-based integer index matching the persistent compound ID
    compound_name   : compound name from the manual curation table
    adduct          : ion adduct
    file_name       : basename of the file containing the best hit
    rt_peak         : retention time of the best-scoring MS2 scan (min)
    mz_peak         : measured precursor m/z of the best-scoring MS2 scan
    spectrum        : query spectrum as JSON ``[[mz0, mz1, ...], [int0, int1, ...]]``

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
        Data tables are loaded automatically if not already cached.
    output_loc:
        Override output directory.  Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/``.
    output_filename:
        CSV filename (the ``.csv`` extension is appended automatically if absent).
    overwrite:
        When *False*, raises :exc:`FileExistsError` if the output file already
        exists.
    min_fragment_intensity:
        When given, only fragment ions whose intensity is strictly greater than
        this value are retained in the output ``spectrum`` column.  Fragments
        at or below the threshold are dropped before serialisation.  Pass
        ``None`` (default) to include all fragments.

    Returns
    -------
    pd.DataFrame
        The exported DataFrame (empty when no hits exist).
    """
    if output_loc is None:
        raise ValueError("output_loc must be provided for best MS2 hit CSV output.")
    output_loc = Path(output_loc)
    output_loc.mkdir(parents=True, exist_ok=True)

    if not output_filename.endswith(".csv"):
        output_filename += ".csv"
    csv_path = output_loc / output_filename

    if not overwrite and csv_path.exists():
        raise FileExistsError(
            f"Output file already exists and overwrite=False: {csv_path}"
        )

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms2_hits_all_df    = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - best MS2 hit CSV not written.")
        return pd.DataFrame()

    rows: List[dict] = []
    for cmp_idx, mc_row in manual_curation_df.reset_index(drop=True).iterrows():
        inchi_key     = mc_row.get("inchi_key", "")
        adduct        = mc_row.get("adduct", "")
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"

        if ms2_hits_all_df is None or ms2_hits_all_df.empty:
            continue

        comp_hits = ms2_hits_all_df[
            (ms2_hits_all_df["inchi_key"] == inchi_key) &
            (ms2_hits_all_df["adduct"]    == adduct)
        ]
        if comp_hits.empty:
            continue

        best_hit = comp_hits.sort_values("score", ascending=False).iloc[0]

        raw_spectrum = best_hit.get("qry_spectrum", json.dumps([[], []]))
        if min_fragment_intensity is not None:
            mz_arr, int_arr = _parse_spectrum(raw_spectrum)
            if mz_arr and int_arr:
                filtered = [(mz, i) for mz, i in zip(mz_arr, int_arr) if i > min_fragment_intensity]
                if filtered:
                    fmz, fint = zip(*filtered)
                    raw_spectrum = json.dumps([list(fmz), list(fint)])
                else:
                    raw_spectrum = json.dumps([[], []])

        rows.append({
            "compound_index": cmp_idx + 1,
            "compound_name":  compound_name,
            "adduct":         adduct,
            "file_name":      os.path.basename(str(best_hit.get("file_path", ""))),
            "rt_peak":        best_hit.get("rt",          np.nan),
            "mz_peak":        best_hit.get("mz_measured", np.nan),
            "spectrum":       raw_spectrum,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.warning("No MS2 hits found for any compound - best MS2 hit CSV not written.")
        return result_df

    result_df.to_csv(csv_path, index=False)
    logger.info("Exported best MS2 hit fragment ions CSV (%d compounds) → %s", len(result_df), csv_path)
    return result_df

def run_all_summaries(
    summary_obj: "AnalysisSummary",
    overwrite: bool = False,
    skip_outputs: Optional[List[str]] = None
) -> None:
    """Run all summary outputs for one analysis.

    Calls ``summary_obj.load_data()`` (a no-op if data are already cached)
    then runs all five output functions in order:

    1. Per-compound identification figures  (PDF, one per compound)
    2. EIC thumbnail PDFs (shared-y and independent-y variants)
    3. Summary Excel table (``Draft_Final_Identifications.xlsx``)
    4. Six boxplot PDFs (3 metrics x linear / log₁₀)
    5. Manual curation CSV (``manual_curation.csv``)

    Because data tables are stored on ``summary_obj`` after the first call
    to ``load_data()``, all five sub-functions share the same in-memory
    DataFrames with no redundant database queries.

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
    overwrite:
        Passed through to all sub-functions.
    """
    # shared data load
    summary_obj.load_data()
    output_loc = summary_obj.paths.get("analysis_output_dir", None)
    if summary_obj.manual_curation_df is None or summary_obj.manual_curation_df.empty:
        logger.error("No manual curation entries found - aborting run_all_summaries.")
        return

    if "final_id_sheet" not in (skip_outputs or []):
        logger.info("Making Final Identification sheet...")
        make_final_id_sheet(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "id_figures" not in (skip_outputs or []):
        logger.info("Making Identification figures...")
        make_identification_figure(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "eic_thumbnails" not in (skip_outputs or []):
        logger.info("Making EIC thumbnails...")
        make_eic_thumbnails(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "boxplots" not in (skip_outputs or []):
        logger.info("Making Boxplots...")
        make_boxplots(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "manual_curation_csv" not in (skip_outputs or []):
        logger.info("Making Manual curation CSV...")
        make_manual_curation_csv(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "best_ms2_hits_csv" not in (skip_outputs or []):
        logger.info("Making best MS2 hit fragment ions CSV...")
        make_best_ms2_hit_fragment_ions_csv(summary_obj, output_loc=output_loc, overwrite=overwrite)

    logger.info("Exporting post-curation atlas data CSV...")
    summary_obj.post_curation_atlas_obj.to_dataframe().to_csv(
        f"{summary_obj.paths['analysis_output_dir']}/{summary_obj.post_curation_atlas_obj.atlas_uid}.csv", index=False
    )

    logger.info("Saving input yaml config to analysis output directory...")
    with open(f"{summary_obj.paths['analysis_output_dir']}/RTA{summary_obj.rt_alignment_number}_TGA{summary_obj.analysis_number}_analysis_config.yaml", "w") as f:
        with open(summary_obj.config_path, "r") as original:
            f.write(original.read())