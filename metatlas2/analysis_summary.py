import itertools
import shutil
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import json
import os
import statistics
import textwrap
import warnings
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import metatlas2.database_interact as dbi
import metatlas2.rclone as rcl
import metatlas2.logging_config as lcf
logger = lcf.get_logger('analysis_summary')

def _strip_non_chars(text: str) -> str:
    """Remove Unicode non-characters (e.g. U+FFFE/FFFF) that DejaVu Sans cannot render."""
    return "".join(c for c in text if ord(c) not in range(0xFDD0, 0xFDF0) and ord(c) & 0xFFFF not in (0xFFFE, 0xFFFF))

def _get_file_color(file_path: str, color_map: Optional[Dict[str, str]] = None) -> str:
    """Determine color for a file based on color mapping.
    """
    if color_map is None:
        return "gray"
    
    for key, color in color_map.items():
        if key.lower() in file_path.lower():
            return color
    
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

def _get_compound_info_batch(
    main_db_path: str,
    inchi_keys: List[str],
) -> dict[str, tuple]:
    """Fetch all compound metadata in ONE query.

    Returns
    -------
    dict of inchi_key -> (formula, smiles, inchi, pubchem_cid, mono_isotopic_molecular_weight)
    Missing keys are simply absent from the dict.
    """
    if not inchi_keys:
        return {}
    placeholders = ",".join("?" * len(inchi_keys))
    try:
        with dbi.get_db_connection(main_db_path, read_only=True) as conn:
            rows = conn.execute(
                f"""SELECT inchi_key, formula, smiles, inchi,
                           pubchem_cid, mono_isotopic_molecular_weight
                    FROM compounds
                    WHERE inchi_key IN ({placeholders})""",
                inchi_keys,
            ).fetchall()
        return {
            row[0]: (row[1], row[2], row[3], row[4], float(row[5]) if row[5] is not None else None)
            for row in rows
        }
    except Exception as exc:
        logger.warning("Batch compound info query failed: %s", exc)
        return {}

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

    # Add vertical text labels on the left side
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_text_pos = xlim[0] - (xlim[1] - xlim[0]) * 0.02  # Slightly left of the left edge
    
    # "Experimental" label for top half (y>0)
    if ylim[1] > 0:
        y_exp_pos = ylim[1] * 0.5  # Middle of positive y region
        ax.text(
            x_text_pos, y_exp_pos, "Experimental",
            fontsize=12, weight="bold", ha="right", va="center",
            rotation=90, color="black",
        )
    
    # "Reference" label for bottom half (y<0)
    if ylim[0] < 0:
        y_ref_pos = ylim[0] * 0.5  # Middle of negative y region
        ax.text(
            x_text_pos, y_ref_pos, "Reference",
            fontsize=12, weight="bold", ha="right", va="center",
            rotation=90, color="black",
        )

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
    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, "Score: N/A", fontsize=12, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, f"RT: {rt_str} min", fontsize=10, weight="normal", ha="center", va="top", transform=ax.transAxes)

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
    ax.text(0.5, 1.13, title, fontsize=12, weight="normal", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.07, "Score: N/A", fontsize=12, weight="bold", ha="center", va="bottom", transform=ax.transAxes)
    ax.text(0.5, 1.02, "RT: N/A", fontsize=10, weight="normal", ha="center", va="top", transform=ax.transAxes)
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
    rt_min = mc_row.get("rt_min", np.nan)
    rt_max = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    # RT boundary and atlas peak lines
    if not np.isnan(rt_min):
        ax.axvline(rt_min, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="_rt_min")
    if not np.isnan(rt_max):
        ax.axvline(rt_max, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="_rt_max")
    if not np.isnan(rt_peak):
        ax.axvline(rt_peak, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="_atlas_peak")

    for file_idx, row in ms1_compound_df.iterrows():
        rt_arr, i_arr = _parse_spectrum(row["raw_spectrum"])
        if not rt_arr:
            continue
        color = _get_file_color(row["file_path"], color_map)
        if log_scale:
            i_arr = [np.log10(max(v, 1)) for v in i_arr]
        ax.plot(rt_arr, i_arr, color=color, linewidth=1.0, alpha=0.7, label="_nolegend_")

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
    ppm = mc_row.get("best_ms1_ppm_error")

    rows = [
        ("Compound", _fmt(mc_row.get("compound_name"))),
        ("Formula", _fmt(mc_row.get("formula"))),
        ("Adduct", _fmt(mc_row.get("adduct"))),
        ("Polarity", _fmt(mc_row.get("polarity"))),
        ("Chromatography", _fmt(mc_row.get("chromatography"))),
        ("Atlas m/z", _fmt(mc_row.get("atlas_mz"), "{:.4f}")),
        ("Measured m/z", _fmt(mc_row.get("best_ms1_mz"), "{:.4f}")),
        ("m/z ppm Δ", _fmt(ppm, "{:.2f}")),
        ("Atlas RT range", f"{_fmt(mc_row.get('atlas_rt_min'), '{:.3f}')} - {_fmt(mc_row.get('atlas_rt_max'), '{:.3f}')} min"),
        ("Measured RT range", f"{_fmt(mc_row.get('rt_min'), '{:.3f}')} - {_fmt(mc_row.get('rt_max'), '{:.3f}')} min"),
        ("Atlas RT peak", _fmt(mc_row.get("atlas_rt_peak"), "{:.3f} min")),
        ("Measured RT", f"{_fmt(mc_row.get('best_ms1_rt'), '{:.3f} min')} "),
        ("RT Δ", _fmt(rt_err, '{:.3f}')),
    ]

    y_start = 1.03
    y_end = 0.09
    for label, value in rows:
        ax.text(0.02, y_start, f"{label}:", fontsize=12, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.34, y_start, value, fontsize=14, va="center", transform=ax.transAxes)
        y_start -= y_end

    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1)


def _plot_hit_info_table(
    ax,
    top3: pd.DataFrame,
    mc_row: pd.Series,
    ms2_hits_df: pd.DataFrame,
) -> None:
    """Render the best MS2 hit as a vertical Theoretical/Measured/Difference table on *ax*."""
    ax.axis("off")

    if top3.empty:
        ax.text(0.5, 0.5, "No MS2 hits found.", transform=ax.transAxes,
                ha="center", va="center", fontsize=25, color="gray")
        return

    # Best hit is top3.iloc[0] — identical to what the mirror plot displays
    best_hit = top3.iloc[0]

    # Abbreviated file name used as the table title
    raw_name = os.path.basename(str(best_hit["file_path"]))
    name_without_ext = raw_name.split(".")[0]
    name_without_ext = name_without_ext.replace("_ms2_pos", "").replace("_ms2_neg", "")

    # Scalar values
    atlas_mz = float(mc_row.get("atlas_mz", np.nan))
    measured_mz = float(mc_row.get("best_ms1_mz", np.nan))
    ppm_error = float(mc_row.get("best_ms1_ppm_error", np.nan))
    atlas_rt = float(mc_row.get("atlas_rt_peak", np.nan))
    measured_rt = float(mc_row.get("best_ms1_rt", np.nan))
    rt_error = float(mc_row.get("best_ms1_rt_error", np.nan))
    score = float(best_hit.get("score", np.nan))
    num_matches = int(best_hit.get("num_matches", 0))
    ref_frags = int(best_hit.get("ref_frags", 0))

    # Fragment match list
    try:
        mf = json.loads(best_hit.get("matched_fragments", "[]"))
        frag_str = ", ".join(f"{m:.3f}" for m in mf) if mf else "N/A"
    except Exception:
        frag_str = "N/A"

    def _v(val, fmt):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return fmt.format(val)

    score_str = f"{score:.4f}\n({num_matches}/{ref_frags})" if not np.isnan(score) else "N/A"

    # Wrap the fragment list so it stays within the table width
    FRAG_WRAP_WIDTH = 80
    if frag_str != "N/A":
        frag_lines = textwrap.wrap(frag_str, width=FRAG_WRAP_WIDTH)
        frag_display = "\n".join(frag_lines) if frag_lines else "N/A"
    else:
        frag_display = "N/A"
        frag_lines = ["N/A"]
    n_frag_lines = min(len(frag_lines), 3)

    # Column x-positions: row-label | Theoretical | Measured | Error/Score
    col_x = [0, 0.3, 0.6, 0.8]

    # Layout constants
    std_row_h = 0.135
    frag_row_h = std_row_h * max(1, n_frag_lines)

    # Row y-centres (header first, then rows stacked downward)
    header_center = 0.83
    ma_center = header_center - std_row_h
    rt_center = header_center - 2 * std_row_h
    frag_top = rt_center - std_row_h / 2
    frag_center = frag_top - frag_row_h / 2

    # File name as title (above the table) - aligned with table left edge
    ax.text(col_x[0], 0.97, name_without_ext, fontsize=14, weight="bold",
            ha="left", va="center", transform=ax.transAxes)

    GAP = 0.005
    for y_center, h, color in [
        (header_center, std_row_h, "#d0d0d0"),
        (ma_center, std_row_h, "white"),
        (rt_center, std_row_h, "#f0f0f0"),
        (frag_center, frag_row_h, "white"),
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
                     _v(atlas_mz, "{:.4f} m/z"),
                     _v(measured_mz, "{:.4f} m/z"),
                     _v(ppm_error, "{:.2f} ppm"))),
        (rt_center, ("RT Accuracy",
                     _v(atlas_rt, "{:.3f} min"),
                     _v(measured_rt, "{:.3f} min"),
                     _v(rt_error, "{:.3f} min"))),
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

    # Add 2nd best, 3rd best, and total files with database matches
    additional_info_y = frag_center - frag_row_h / 2 - 0.04
    line_spacing = 0.025
    
    # 2nd best hit file
    if len(top3) >= 2:
        second_best_file = os.path.basename(str(top3.iloc[1]["file_path"]))
        ax.text(0, additional_info_y, f"2nd best: {second_best_file}", 
                fontsize=13, ha="left", va="top", transform=ax.transAxes)
    
    # 3rd best hit file
    if len(top3) >= 3:
        third_best_file = os.path.basename(str(top3.iloc[2]["file_path"]))
        ax.text(0, additional_info_y - line_spacing, f"3rd best: {third_best_file}", 
                fontsize=13, ha="left", va="top", transform=ax.transAxes)
    
    # Total files with database matches
    total_files = ms2_hits_df["file_path"].nunique() if not ms2_hits_df.empty else 0
    y_offset = line_spacing * 2 if len(top3) >= 3 else (line_spacing if len(top3) >= 2 else 0)
    ax.text(0, additional_info_y - y_offset, f"Total files with database matches: {total_files}", 
            fontsize=13, ha="left", va="top", transform=ax.transAxes)

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
        parsed = json.loads(raw_colors)
        frag_colors = parsed

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

def _identification_figure_worker(kwargs: dict) -> str:
    """Worker: generate and save one identification figure PDF."""
    import matplotlib
    matplotlib.use("Agg")

    mc_row = pd.Series(kwargs["mc_row_dict"])
    fig_path = Path(kwargs["fig_path"])
    cmp_idx = kwargs["compound_idx"]
    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    inchi_key = kwargs["inchi_key"]
    color_map = kwargs["color_map"]
    ms1_df = kwargs["ms1_df"]
    ms2_raw_df = kwargs["ms2_raw_df"]
    ms2_hits_df = kwargs["ms2_hits_df"]

    top3 = (
        ms2_hits_df
        .sort_values("score", ascending=False)
        .drop_duplicates("file_path")
        .head(3)
        .reset_index(drop=True)
    ) if not ms2_hits_df.empty else pd.DataFrame()

    fig = plt.figure(figsize=(25, 15))
    gs = fig.add_gridspec(
        3, 4, hspace=0.38, wspace=0.30,
        height_ratios=[1.45, 1.2, 1.35],
    )

    for i in range(3):
        _plot_ms2(fig.add_subplot(gs[0, i]), i, top3, ms2_raw_df)

    _plot_structure(
        fig.add_subplot(gs[0, 3]),
        mc_row.get("smiles"), mc_row.get("inchi"), inchi_key, size=500,
    )

    ax_eic_lin = fig.add_subplot(gs[1, 0])
    _plot_eic(ax_eic_lin, ms1_df, mc_row, log_scale=False, color_map=color_map)
    ax_eic_lin.set_title("EIC (linear scale)", fontsize=18)

    ax_eic_log = fig.add_subplot(gs[1, 1])
    _plot_eic(ax_eic_log, ms1_df, mc_row, log_scale=True, color_map=color_map)
    ax_eic_log.set_title("EIC (log₁₀ scale)", fontsize=18)

    _plot_compound_info_table(fig.add_subplot(gs[1, 2:4]), mc_row)
    _plot_hit_info_table(fig.add_subplot(gs[2, 0:4]), top3, mc_row, ms2_hits_df)

    fig.suptitle(
        f"[{cmp_idx:04d}] |  {_strip_non_chars(adduct)}  |  {inchi_key}\n{_strip_non_chars(compound_name)}\n",
        fontsize=20, weight="bold", y=0.97,
    )
    for y_line in [0.61, 0.345]:
        fig.add_artist(plt.Line2D(
            [0.08, 0.92], [y_line, y_line],
            transform=fig.transFigure,
            color="black", linewidth=1, clip_on=False,
        ))

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return compound_name


def make_identification_figure(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    main_db_path = summary_obj.paths.get("main_db_path")

    output_dir = Path(output_loc) / "identification_figures"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info("Overwriting disabled: existing directory %s will be used (existing PDFs will be preserved).", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting identification figures to %s", output_dir)

    color_map = None
    if hasattr(summary_obj, "override_parameters") and summary_obj.override_parameters.get("gui_lcmsruns_colors"):
        color_map = summary_obj.override_parameters["gui_lcmsruns_colors"]
    elif hasattr(summary_obj, "workflow_params") and summary_obj.workflow_params.get("gui_lcmsruns_colors"):
        color_map = summary_obj.workflow_params["gui_lcmsruns_colors"]

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df.reset_index(drop=True)
    ms1_all_df = summary_obj.ms1_all_df
    ms2_raw_all_df = summary_obj.ms2_raw_all_df
    ms2_hits_all_df = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    total_files = ms1_all_df["file_path"].nunique() if (ms1_all_df is not None and not ms1_all_df.empty) else 0
    logger.info("Plotting %d compounds across %d files.", len(manual_curation_df), total_files)

    unique_inchi_keys = manual_curation_df["inchi_key"].dropna().unique().tolist()
    batch_info = _get_compound_info_batch(main_db_path, unique_inchi_keys)

    def _pregroup(df: Optional[pd.DataFrame]) -> dict:
        if df is None or df.empty:
            return {}
        return {
            key: grp.reset_index(drop=True)
            for key, grp in df.groupby(["inchi_key", "adduct"], sort=False)
        }

    ms1_groups = _pregroup(ms1_all_df)
    ms2_raw_groups = _pregroup(ms2_raw_all_df)
    ms2_hits_groups = _pregroup(ms2_hits_all_df)
    empty_df = pd.DataFrame()

    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")

        safe_name = f"{cmp_idx:04d}_{compound_name}_{adduct}".replace("/", "-").replace(" ", "_")
        fig_path = output_dir / f"{safe_name}.pdf"

        info = batch_info.get(inchi_key, (None, None, None, None, None))
        formula, smiles, inchi = info[0], info[1], info[2]

        mc_row_dict = mc_row.to_dict()
        mc_row_dict.update({"formula": formula, "smiles": smiles, "inchi": inchi})

        key = (inchi_key, adduct)
        tasks.append({
            "mc_row_dict":  mc_row_dict,
            "compound_name": compound_name,
            "adduct":       adduct,
            "inchi_key":    inchi_key,
            "compound_idx": cmp_idx,
            "fig_path":     str(fig_path),
            "color_map":    color_map,
            "ms1_df":       ms1_groups.get(key, empty_df),
            "ms2_raw_df":   ms2_raw_groups.get(key, empty_df),
            "ms2_hits_df":  ms2_hits_groups.get(key, empty_df),
        })

    if not tasks:
        logger.info("Nothing to generate (all figures already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info("Generating %d figures using %d workers...", len(tasks), n_workers)

    pbar = tqdm(total=len(tasks), desc="Generating ID figures", unit="compound")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_identification_figure_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Exported identification figure for %s", name)
            except Exception as exc:
                logger.error("Failed to generate figure for %s: %s", name, exc)
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)

    pbar.close()

    logger.info("Identification figure export complete → %s", output_dir)

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

def _compute_all_overlapping_compounds(
    manual_curation_df: pd.DataFrame,
    mass_map: dict[str, Optional[float]],
) -> dict[int, Tuple[str, str]]:
    """Compute every compound's overlapping set in one vectorized pass.

    Parameters
    ----------
    mass_map:
        Pre-fetched monoisotopic masses keyed by inchi_key (from batch query).

    Returns
    -------
    dict of {compound_idx: (names_str, indices_str)}
    """
    mc = manual_curation_df.reset_index(drop=True)
    n = len(mc)

    rt_min = mc["rt_min"].to_numpy(dtype=float)
    rt_max = mc["rt_max"].to_numpy(dtype=float)
    mz = mc["atlas_mz"].to_numpy(dtype=float)
    masses = np.array(
        [mass_map.get(ik) if mass_map.get(ik) is not None else np.nan
         for ik in mc["inchi_key"]],
        dtype=float,
    )
    names = mc["compound_name"].tolist()
    inchi_keys = mc["inchi_key"].tolist()

    rt_overlap = (
        np.maximum(rt_min[:, None], rt_min[None, :]) <=
        np.minimum(rt_max[:, None], rt_max[None, :])
    )

    mz_i, mz_j = mz[:, None], mz[None, :]
    mz_valid = (mz_i != 0) & (mz_j != 0) & ~np.isnan(mz_i) & ~np.isnan(mz_j)
    mz_similar = mz_valid & (np.abs(mz_i - mz_j) <= 0.005)

    m_i, m_j = masses[:, None], masses[None, :]
    mass_valid = (m_i != 0) & (m_j != 0) & ~np.isnan(m_i) & ~np.isnan(m_j)
    mass_similar = mass_valid & (np.abs(m_i - m_j) <= 0.005)

    both_mass_zero = (m_i == 0) & (m_j == 0)
    overlap = rt_overlap & (mz_similar | mass_similar) & ~both_mass_zero
    np.fill_diagonal(overlap, True)

    result: dict[int, Tuple[str, str]] = {}
    for i in range(n):
        js = np.where(overlap[i])[0]
        if len(js) <= 1:
            result[i] = ("", "")
        else:
            result[i] = (
                "//".join(names[j] for j in js),
                "//".join(inchi_keys[j] for j in js),
            )
    return result

def make_final_id_sheet(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "Final_Identifications.xlsx",
    overwrite: bool = True,
) -> None:

    output_loc = Path(output_loc)
    chromatography = summary_obj.chromatography
    analysis_info = f"{summary_obj.post_curation_atlas_obj.chromatography}-{summary_obj.post_curation_atlas_obj.polarity}-{summary_obj.post_curation_atlas_obj.analysis_type}"
    run_info = f"RTA{summary_obj.rt_alignment_number}-TGA{summary_obj.analysis_number}"
    output_filename = f"{summary_obj.project_name}_{analysis_info}-{run_info}_{output_filename}"
    if not output_filename.endswith(".xlsx"):
        output_filename += ".xlsx"
    excel_path = output_loc / output_filename
    if not overwrite and excel_path.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", excel_path)
        return

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms2_hits_all_df = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to export.")
        return

    manual_curation_df = manual_curation_df.reset_index(drop=True)
    db_path = summary_obj.paths.get("main_db_path")

    logger.info("Fetching compound metadata in single batch query...")
    unique_inchi_keys = manual_curation_df["inchi_key"].dropna().unique().tolist()
    batch_info = _get_compound_info_batch(db_path, unique_inchi_keys)

    compound_info_map: dict[str, tuple] = {
        ik: v[:4] for ik, v in batch_info.items()  # formula, smiles, inchi, pubchem_cid
    }
    mass_map: dict[str, Optional[float]] = {
        ik: v[4] for ik, v in batch_info.items()  # mono_isotopic_molecular_weight
    }

    if not ms2_hits_all_df.empty:
        ms2_best: dict[tuple, pd.Series] = {
            key: grp.sort_values("score", ascending=False).iloc[0]
            for key, grp in ms2_hits_all_df.groupby(["inchi_key", "adduct"], sort=False)
        }
    else:
        ms2_best = {}

    is_c18 = "c18" in chromatography.lower() and "lipid" not in chromatography.lower()

    logger.info("Processing %d compounds...", len(manual_curation_df))
    rows: List[dict] = []

    # Precompute overlapping compounds using the helper function
    overlapping_map = _compute_all_overlapping_compounds(manual_curation_df, mass_map)

    for compound_idx, mc_row in tqdm(
        manual_curation_df.iterrows(),
        total=len(manual_curation_df),
        desc="Adding compounds to final ID sheet",
    ):
        compound_name = mc_row.get("compound_name") or f"compound_{compound_idx}"
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")
        polarity = mc_row.get("polarity", "")

        formula, smiles, inchi, pubchem_cid = compound_info_map.get(inchi_key, (None, None, None, None))
        exact_mass = mass_map.get(inchi_key)

        # --- Overlapping compound columns using _compute_all_overlapping_compounds ---
        overlapping_compound, overlapping_inchi_keys = overlapping_map.get(compound_idx, ("", ""))
        identified_metabolite = compound_name if not overlapping_compound else overlapping_compound

        # --- MS1 metrics ---
        mz_theoretical = float(mc_row.get("atlas_mz", np.nan))
        mz_measured = float(mc_row.get("best_ms1_mz", np.nan))
        ppm_error = float(mc_row.get("best_ms1_ppm_error", np.nan))
        rt_error = float(mc_row.get("best_ms1_rt_error", np.nan))
        rt_measured = float(mc_row.get("best_ms1_rt", np.nan))
        max_intensity = float(mc_row.get("best_ms1_intensity", np.nan))
        max_int_file = mc_row.get("best_ms1_file", "")

        mz_error_da = (
            abs(mz_theoretical - mz_measured)
            if not (np.isnan(mz_theoretical) or np.isnan(mz_measured))
            else np.nan
        )

        mz_q = _mz_quality(ppm_error, mz_error_da)
        rt_q = _rt_quality(rt_error, chromatography)

        ms2_notes = mc_row.get("ms2_notes", "") or ""
        try:
            msms_q = float(str(ms2_notes).split(",")[0])
        except (ValueError, AttributeError):
            msms_q = np.nan

        total_score, msi_level = _total_score_and_msi(msms_q, mz_q, rt_q)

        # --- MSMS metrics ---
        msms_file = ""
        msms_rt = np.nan
        msms_score = np.nan
        msms_num_ions = ""
        msms_matching_ions = ""

        best_hit = ms2_best.get((inchi_key, adduct))
        if best_hit is not None:
            msms_file = str(best_hit.get("file_path", ""))
            msms_rt = float(best_hit.get("rt", np.nan))
            msms_score = float(best_hit.get("score", np.nan))

            num_matches = int(best_hit.get("num_matches", 0))
            ref_frags = int(best_hit.get("ref_frags", 0))
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
                ppm_tol = float(mc_row.get("mz_tolerance", 5.0))
                precursor_match = (
                    abs(single_ion - mz_theoretical) / mz_theoretical * 1e6 <= ppm_tol
                    if (not np.isnan(single_ion) and mz_theoretical > 0) else False
                )
                note_tag = " (single matching fragment is the precursor)" if precursor_match else "(single matching fragment is NOT the precursor)"
                if "1.0, single ion match" in ms2_notes or "0.5, single ion match" in ms2_notes:
                    ms2_notes = ms2_notes + note_tag

        rows.append({
            # --- COMPOUND ANNOTATION ---
            "index": compound_idx,
            "identified_metabolite": identified_metabolite,
            "label": compound_name,
            "overlapping_compound": overlapping_compound,
            "overlapping_inchi_keys": overlapping_inchi_keys,
            "formula": formula if formula else "",
            "polarity": polarity,
            "exact_mass": round(exact_mass, 7) if exact_mass is not None else np.nan,
            "inchi_key": inchi_key,
            # --- COMPOUND IDENTIFICATION SCORES ---
            "msms_quality": msms_q,
            "mz_quality": mz_q,
            "rt_quality": rt_q,
            "total_score": total_score,
            "msi_level": msi_level,
            "isomer_details": "",
            "identification_notes": mc_row.get("identification_notes", ""),
            "analyst_notes": mc_row.get("analyst_notes", ""),
            "other_notes": mc_row.get("other_notes", "") or "",
            "ms1_notes": mc_row.get("ms1_notes", "") or "",
            "ms2_notes": ms2_notes,
            # --- MS1 INTENSITY INFORMATION ---
            "max_intensity": max_intensity,
            "max_intensity_file": Path(max_int_file).name if max_int_file else "",
            "ms1_rt_peak": rt_measured,
            # --- MSMS INFORMATION ---
            "msms_file": Path(msms_file).name if msms_file else "",
            "msms_rt": round(msms_rt, 2) if not np.isnan(msms_rt) else np.nan,
            "msms_numberofions": msms_num_ions,
            "msms_matchingions": msms_matching_ions,
            # --- MSMS EVALUATION ---
            "msms_score": round(msms_score, 4) if not np.isnan(msms_score) else np.nan,
            # --- ION INFORMATION ---
            "mz_adduct": adduct,
            "mz_theoretical": round(mz_theoretical, 4) if not np.isnan(mz_theoretical) else np.nan,
            "mz_measured": round(mz_measured, 4) if not np.isnan(mz_measured) else np.nan,
            # --- M/Z EVALUATION ---
            "mz_error": round(mz_error_da, 4) if not np.isnan(mz_error_da) else np.nan,
            "mz_ppmerror": round(ppm_error, 4) if not np.isnan(ppm_error) else np.nan,
            # --- CHROMATOGRAPHIC PEAK INFORMATION ---
            "rt_min": round(float(mc_row.get("rt_min", np.nan)), 2),
            "rt_max": round(float(mc_row.get("rt_max", np.nan)), 2),
            "rt_theoretical": round(float(mc_row.get("atlas_rt_peak", np.nan)), 2),
            "rt_measured": round(rt_measured, 2) if not np.isnan(rt_measured) else np.nan,
            # --- RT EVALUATION ---
            "rt_error": round(rt_error, 2) if not np.isnan(rt_error) else np.nan,
        })

    final_df = pd.DataFrame(rows)
    logger.info("Assembled final ID table with %d rows.", len(final_df))

    # ------------------------------------------------------------------ #
    #  Excel formatting                                                    #
    # ------------------------------------------------------------------ #

    # Row 0 (section headers), row 1 (column names), row 2 (descriptions),
    # row 3 (internal field names), data starts at row 4 (startrow=4)
    COL_NAMES = [
        "Compound #",
        "Identified Metabolite",
        "Name of metabolite searched for",
        "Labels of Overlapping Compounds",
        "Inchi Keys of Overlapping Compounds",
        "Molecular Formula",
        "Polarity",
        "Exact Mass",
        "Inchi Key",
        "MSMS Score (0 to 1)",
        "m/z score (0 to 1)",
        "RT score (0 to 1)",
        "Total ID Score (0 to 3)",
        "Mass Spec Inititative Identification Level",
        "Isomer details",
        "Identification notes",
        "Analyst notes",
        "Other notes",
        "MS1 notes",
        "MS2 notes",
        "Maximum MS1 intensity across all files",
        "Filename w/ maximum MS1",
        "Retention time of max intensity MS1 peak",
        "File with highest MSMS match score",
        "RT of highest matched MSMS scan",
        "Number of ion matches in msms spectra to EMA reference spectra",
        "List of ion matches in msms spectra to EMA reference spectra",
        "MSMS score (highest across all samples)",  # MSMS EVALUATION - lone column
        "Adduct",
        "Theoretical m/z",
        "Measured m/z",
        "mass error (delta Da)",
        "mass error (delta ppm)",
        "Minimum retention time (min)",
        "Maximum retention time (max)",
        "Theoretical retention time (peak)",
        "Detected RT (peak)",
        "RT error (absolute delta)",
    ]

    rt_q_desc = (
"1 (delta RT </= 0.25), 0.5 (delta RT > 0.25 & </= 0.5), 0 (delta RT > 0.5 min)"
        if is_c18 else
        "1 (delta RT </= 0.5), 0.5 (delta RT > 0.5 & </= 2), 0 (delta RT > 2 min)"
    )

    COL_DESCRIPTIONS = [
        # COMPOUND ANNOTATION
        "Unique for study",
        "Some isomers are not chromatographically or spectrally resolvable. Some compounds detected w/ >1 adduct (increases identification confidence but only use 1 for analysis).",
        "Name of standard reference compound in library match.",
        "compound with similar mz (abs difference <= 0.005) or monoisotopic molecular weight (abs difference <= 0.005) and RT (min or max within the RT-min-max-range of similar compound)",
        "List of inchi keys that correspond to the compounds listed in the previous column",
        "",
        "",
        "monoisotopic mass (neutral except for permanently charged molecules)",
        "neutralized version",
        # COMPOUND IDENTIFICATION SCORES
        "1 (MSMS matches ref. std.), 0.5 (possible match), 0 (no MSMS collected or no appropriate ref available), -1 (bad match)",
        "1 (delta ppm </= 5 or delta Da </= 0.0015), 0.5 (delta ppm 5-10 and delta Da > 0.0015), 0 (delta ppm > 10) mz_quality",
        rt_q_desc,
        "sum of m/z, RT and MSMS score",
        "Level 1 = Two independent and orthogonal properties match authentic standard; else = putative [Metabolomics. 2007 Sep; 3(3): 211-221. doi: 10.1007/s11306-007-0082-2]",
        "Isomers have same formula (and m/z) and similar RT - MSMS spectra may be used to differentiate (exceptions) or RT elution order",
        "",  # identification_notes
        "",  # analyst_notes
        "",  # other_notes
        "",  # ms1_notes
        "",  # ms2_notes
        # MS1 INTENSITY INFORMATION
        "",
        "",
        "",
        # MSMS INFORMATION
        "",
        "",
        "mean # of fragment ions matching between compound in sample and reference compound / standard; may include parent and isotope ions and very low intensity background ions (these do not contribute to score)",
        "",
        # MSMS EVALUATION
        "MSMS score (highest across all samples), scale of 0 to 1 based on an algorithm. 0 = no match, 1 = perfect match. If no score, then no MSMS was acquired for that compound (@ m/z & RT window).",
        # ION INFORMATION
        "More than one may be detectable; the one evaluated is listed",
        "theoretical m/z for a given compound / adduct pair",
        "average m/z within 20ppm of theoretical detected across the top three most intense ions @ RT peak",
        # M/Z EVALUATION
        "absolute difference between theoretical and detected m/z",
        "ppm difference between theoretical and detected m/z",
        # CHROMATOGRAPHIC PEAK INFORMATION
        "Retention range including start and end of detection of an m/z value (Note: Peak Height is calculated as the highest intensity of an m/z within the min/max RT range. Peak Area is calculated as the integrated area under the curve for an m/z within the mix/max RT range.)",
        "",
        "theoretical retention time for a compound based upon reference standard at highest intensity point of peak",
        "average retention time for a detected compound at highest intensity point of peak across all samples",
        # RT EVALUATION
        "absolute difference between theoretical and detected RT peak",
    ]

    # Internal field names row (row 3, index 3)
    COL_FIELDS = [
        "index", "identified_metabolite", "label", "overlapping_compound", "overlapping_inchi_keys",
        "formula", "polarity", "exact_mass", "inchi_key",
        "msms_quality", "mz_quality", "rt_quality", "total_score", "msi_level",
        "isomer_details", "identification_notes", "analyst_notes", "other_notes",
        "ms1_notes", "ms2_notes",
        "max_intensity", "max_intensity_file", "ms1_rt_peak",
        "msms_file", "msms_rt", "msms_numberofions", "msms_matchingions",
        "msms_score",
        "mz_adduct", "mz_theoretical", "mz_measured",
        "mz_error", "mz_ppmerror",
        "rt_min", "rt_max", "rt_theoretical", "rt_measured", "rt_error",
    ]

    # Section header definitions: (start_col_idx, end_col_idx, label)
    # Columns are 0-indexed here; in Excel they start at col 0 = A
    # COMPOUND ANNOTATION:        cols 0-8   (A-I)
    # COMPOUND IDENTIFICATION SCORES: cols 9-13 (J-N)  -- note: isomer_details through ms2_notes are under same header
    # MS1 INTENSITY INFORMATION:  cols 20-22 (U-W)
    # MSMS INFORMATION:           cols 23-26 (X-AA)
    # MSMS EVALUATION:            col  27    (AB)
    # ION INFORMATION:            cols 28-30 (AC-AE)
    # M/Z EVALUATION:             cols 31-32 (AF-AG)
    # CHROMATOGRAPHIC PEAK INFORMATION: cols 33-36 (AH-AK)
    # RT EVALUATION:              col  37    (AL)
    #
    # Rows 14-19 (isomer_details through ms2_notes) share the
    # COMPOUND IDENTIFICATION SCORES header per the new format spec.

    _SECTION_SPANS = [
        (0,  8,  "COMPOUND ANNOTATION"),
        (9,  19, "COMPOUND IDENTIFICATION SCORES"),
        (20, 22, "MS1 INTENSITY INFORMATION"),
        (23, 26, "MSMS INFORMATION"),
        (27, 27, "MSMS EVALUATION"),
        (28, 30, "ION INFORMATION"),
        (31, 32, "M/Z EVALUATION"),
        (33, 36, "CHROMATOGRAPHIC PEAK INFORMATION"),
        (37, 37, "RT EVALUATION"),
    ]

    # Section background colours (applied to data rows too via conditional format)
    # Using the same colour logic as original but mapped to new spans
    _SECTION_COLORS = {
        "COMPOUND ANNOTATION":              "#DCEEFF",  # light blue
        "COMPOUND IDENTIFICATION SCORES":   "#DCFFFF",  # cyan
        "MS1 INTENSITY INFORMATION":        "#FFFFDC",  # yellow
        "MSMS INFORMATION":                 "#FFDCFF",  # rose
        "MSMS EVALUATION":                  "#FFDCFF",  # rose (continuation)
        "ION INFORMATION":                  "#FFFFDC",  # yellow
        "M/Z EVALUATION":                   "#FFFFDC",  # yellow
        "CHROMATOGRAPHIC PEAK INFORMATION": "#FFFFDC",  # yellow
        "RT EVALUATION":                    "#DCFFFF",  # cyan
    }

    def _col_letter(col_idx: int) -> str:
        """Convert 0-based column index to Excel column letter(s)."""
        result = ""
        n = col_idx + 1
        while n:
            n, remainder = divmod(n - 1, 26)
            result = chr(65 + remainder) + result
        return result

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        # Write data starting at row 4 (0-indexed) to leave room for 4 header rows
        final_df.to_excel(
            writer,
            sheet_name="Final_Identifications",
            index=False,
            header=False,
            startrow=4,
        )
        workbook = writer.book
        worksheet = writer.sheets["Final_Identifications"]

        nrows = len(final_df) + 4  # total rows including headers

        # ---- Formats ----
        f_header_base = {
            "bold": True,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "border": 1,
        }
        f_header = workbook.add_format(f_header_base)
        f_scientific = workbook.add_format({"num_format": "0.00E+00"})

        # One format per section colour
        section_formats: dict[str, object] = {}
        for label, color in _SECTION_COLORS.items():
            section_formats[label] = workbook.add_format({"bg_color": color})

        section_header_formats: dict[str, object] = {}
        for label, color in _SECTION_COLORS.items():
            section_header_formats[label] = workbook.add_format({
                **f_header_base,
                "bg_color": color,
            })

        # ---- Row heights ----
        worksheet.set_row(0, 30)   # section headers
        worksheet.set_row(1, 80)   # column names
        worksheet.set_row(2, 120)  # descriptions
        worksheet.set_row(3, 40)   # internal field names

        # ---- Column widths ----
        worksheet.set_column(0, 0, 10)   # index
        worksheet.set_column(1, 1, 30)   # identified_metabolite
        worksheet.set_column(2, 2, 30)   # label
        worksheet.set_column(3, 3, 30)   # overlapping_compound
        worksheet.set_column(4, 4, 30)   # overlapping_inchi_keys
        worksheet.set_column(5, 5, 14)   # formula
        worksheet.set_column(6, 6, 10)   # polarity
        worksheet.set_column(7, 7, 14)   # exact_mass
        worksheet.set_column(8, 8, 28)   # inchi_key
        worksheet.set_column(9, 13, 12)  # scores
        worksheet.set_column(14, 19, 25) # notes
        worksheet.set_column(20, 20, 16, f_scientific)  # max_intensity
        worksheet.set_column(21, 22, 35) # file + rt
        worksheet.set_column(23, 26, 35) # msms cols
        worksheet.set_column(27, 27, 12) # msms_score
        worksheet.set_column(28, 30, 14) # ion info
        worksheet.set_column(31, 32, 14) # mz eval
        worksheet.set_column(33, 37, 14) # rt cols

        # ---- Row 0: section header merges ----
        for start_idx, end_idx, label in _SECTION_SPANS:
            fmt = section_header_formats[label]
            start_letter = _col_letter(start_idx)
            end_letter = _col_letter(end_idx)
            if start_idx == end_idx:
                worksheet.write(f"{start_letter}1", label, fmt)
            else:
                worksheet.merge_range(
                    f"{start_letter}1:{end_letter}1", label, fmt
                )

        # ---- Row 1: column display names ----
        for col_idx, name in enumerate(COL_NAMES):
            # Find which section this column belongs to for background colour
            section_label = next(
                lbl for s, e, lbl in _SECTION_SPANS if s <= col_idx <= e
            )
            fmt = section_header_formats[section_label]
            worksheet.write(1, col_idx, name, fmt)

        # ---- Row 2: descriptions ----
        for col_idx, desc in enumerate(COL_DESCRIPTIONS):
            section_label = next(
                lbl for s, e, lbl in _SECTION_SPANS if s <= col_idx <= e
            )
            fmt = section_header_formats[section_label]
            worksheet.write(2, col_idx, desc, fmt)

        # ---- Row 3: internal field names ----
        for col_idx, field in enumerate(COL_FIELDS):
            section_label = next(
                lbl for s, e, lbl in _SECTION_SPANS if s <= col_idx <= e
            )
            fmt = section_header_formats[section_label]
            worksheet.write(3, col_idx, field, fmt)

        # ---- Conditional background colours for data rows ----
        for start_idx, end_idx, label in _SECTION_SPANS:
            color = _SECTION_COLORS[label]
            fmt = workbook.add_format({"bg_color": color})
            start_letter = _col_letter(start_idx)
            end_letter = _col_letter(end_idx)
            worksheet.conditional_format(
                f"{start_letter}5:{end_letter}{nrows}",
                {"type": "no_errors", "format": fmt},
            )

    logger.info("Exported final ID table to %s", excel_path)
    return

def _short_fname(file_path: str) -> str:
    """Return an abbreviated filename label (12th and 15th ``_``-separated parts)."""
    if not file_path:
        return "no data"
    stem = os.path.basename(file_path).split(".")[0]
    parts = stem.split("_")
    return f"{parts[12]}_{parts[15]}"

def _render_eic_thumbnail(
    ax,
    rt_arr: List,
    i_arr: List,
    rt_min: float,
    rt_peak: float,
    rt_max: float,
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
    _vline(rt_min, "red", "--")
    _vline(rt_peak, "black", ":" )
    _vline(rt_max, "black", "--")

    if y_max is not None and y_max > 0:
        ax.set_ylim(bottom=0, top=y_max * 1.05)

    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    ax.tick_params(axis="both", labelsize=10, length=2, pad=1)
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    ax.set_title(fname_short, fontsize=10, pad=2.5, loc="center")

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

    rt_min = mc_row.get("rt_min", np.nan)
    rt_max = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    if shared_y:
        all_intensities = [v for item in file_items for v in item["i_arr"] if v is not None]
        y_max_global = float(max(all_intensities)) if all_intensities else None
    else:
        y_max_global = None

    total_files = len(file_items)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)
    title_base = f"[{compound_idx:04d}] {_strip_non_chars(compound_name)} | {_strip_non_chars(adduct)}  (RT alignment {rt_alignment_num}, analysis {analysis_num})"

    _GRID_COLS = 5
    _GRID_ROWS = 5
    _PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS

    with PdfPages(pdf_path) as pdf:
        for page_idx in range(total_pages):
            start = page_idx * _PLOTS_PER_PAGE
            end = min(start + _PLOTS_PER_PAGE, total_files)
            page_items = file_items[start:end]
            n_on_page = len(page_items)

            fig, axes = plt.subplots(
                _GRID_ROWS, _GRID_COLS,
                figsize=(20, 16),
            )
            fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06, hspace=0.45, wspace=0.3)
            axes_flat = axes.flatten()

            for slot_idx, item in enumerate(page_items):
                ax = axes_flat[slot_idx]
                rt_arr = item["rt_arr"]
                i_arr = item["i_arr"]
                fname_s = _strip_non_chars(_short_fname(item["file_path"]))

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
            fig.suptitle(f"{title_base}  {page_label}\n", fontsize=12, y=1.005)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font", category=UserWarning)
                pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

def _compound_pdf_worker(kwargs: dict) -> str:
    """Worker: write shared-y and independent-y PDFs for one compound in a single page-loop.

    Both PDFs are produced from the same figure per page: the independent-scale
    version is saved first, then y-limits are adjusted to the global maximum and
    the shared-scale version is saved, halving the number of figures created.
    """

    _GRID_COLS = 5
    _GRID_ROWS = 5
    _PLOTS_PER_PAGE = _GRID_COLS * _GRID_ROWS

    import matplotlib
    matplotlib.use("Agg")

    path_shared = Path(kwargs["path_shared"])
    path_indep = Path(kwargs["path_indep"])

    mc_row = pd.Series(kwargs["mc_row_dict"])
    file_items = kwargs["file_items"]
    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    rt_alignment_num = kwargs["rt_alignment_num"]
    analysis_num = kwargs["analysis_num"]
    compound_idx = kwargs["compound_idx"]

    rt_min = mc_row.get("rt_min", np.nan)
    rt_max_val = mc_row.get("rt_max", np.nan)
    rt_peak = mc_row.get("atlas_rt_peak", np.nan)

    all_intensities = [v for item in file_items for v in item["i_arr"] if v is not None]
    y_max_global = float(max(all_intensities)) if all_intensities else None

    total_files = len(file_items)
    total_pages = max(1, (total_files + _PLOTS_PER_PAGE - 1) // _PLOTS_PER_PAGE)
    title_base = (
        f"[{compound_idx:04d}] {_strip_non_chars(compound_name)} | "
        f"{_strip_non_chars(adduct)}  (RT alignment {rt_alignment_num}, analysis {analysis_num})"
    )

    pdf_shared = PdfPages(path_shared)
    pdf_indep = PdfPages(path_indep)
    try:
        for page_idx in range(total_pages):
            start = page_idx * _PLOTS_PER_PAGE
            end = min(start + _PLOTS_PER_PAGE, total_files)
            page_items = file_items[start:end]
            n_on_page = len(page_items)
            page_label = f"({page_idx + 1}/{total_pages})" if total_pages > 1 else ""

            fig, axes = plt.subplots(_GRID_ROWS, _GRID_COLS, figsize=(20, 16))
            fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.06, hspace=0.45, wspace=0.3)
            axes_flat = axes.flatten()

            active_axes = []
            for slot_idx, item in enumerate(page_items):
                ax = axes_flat[slot_idx]
                fname_s = _strip_non_chars(_short_fname(item["file_path"]))
                y_max_indep = float(max(item["i_arr"])) if item["i_arr"] else None
                _render_eic_thumbnail(
                    ax, item["rt_arr"], item["i_arr"],
                    rt_min, rt_peak, rt_max_val, fname_s, y_max_indep,
                )
                active_axes.append(ax)

            for slot_idx in range(n_on_page, _PLOTS_PER_PAGE):
                axes_flat[slot_idx].set_visible(False)

            fig.suptitle(f"{title_base}  {page_label}\n", fontsize=12, y=1.005)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Glyph \d+ .* missing from font", category=UserWarning)
                if pdf_indep is not None:
                    pdf_indep.savefig(fig, bbox_inches="tight")
                if pdf_shared is not None:
                    if y_max_global is not None and y_max_global > 0:
                        for ax in active_axes:
                            ax.set_ylim(bottom=0, top=y_max_global * 1.05)
                    pdf_shared.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    finally:
        if pdf_shared is not None:
            pdf_shared.close()
        if pdf_indep is not None:
            pdf_indep.close()

    return kwargs["compound_name"]

def make_eic_thumbnails(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    """Generate per-compound EIC thumbnail PDFs in two output folders (parallelised)."""

    rt_alignment_num = summary_obj.rt_alignment_number
    analysis_num = summary_obj.analysis_number

    base_dir = Path(output_loc)
    dir_shared = base_dir / "eic_thumbnails_shared_y"
    dir_indep = base_dir / "eic_thumbnails_independent_y"
    for dir in (dir_shared, dir_indep):
        if overwrite and dir.exists():
            logger.info("Overwriting enabled: clearing existing contents of %s", dir)
            shutil.rmtree(dir)
        elif not overwrite and dir.exists():
            logger.info("Overwriting disabled: existing directory %s will be used (existing PDFs will be preserved).", dir)
            return
        logger.info("Creating directory %s", dir)
        dir.mkdir(parents=True, exist_ok=True)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms1_all_df = summary_obj.ms1_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    manual_curation_df = manual_curation_df.reset_index(drop=True)

    if not ms1_all_df.empty:
        ms1_groups: dict = {
            key: grp.reset_index(drop=True)
            for key, grp in ms1_all_df.groupby(["inchi_key", "adduct"], sort=False)
        }
    else:
        ms1_groups = {}

    n_compounds = len(manual_curation_df)
    logger.info("Building task list for %d compounds...", n_compounds)

    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")

        safe_stem = f"{cmp_idx:04d}_{compound_name}_{adduct}_{inchi_key}".replace("/", "-").replace(" ", "_")
        path_shared = dir_shared / f"{safe_stem}.pdf"
        path_indep = dir_indep  / f"{safe_stem}.pdf"

        ms1_cmp = ms1_groups.get((inchi_key, adduct))
        if ms1_cmp is None or ms1_cmp.empty:
            file_items = [{"file_path": "", "rt_arr": [], "i_arr": []}]
        else:
            file_items = [
                {
                    "file_path": str(file_row.get("file_path", "")),
                    "rt_arr": rt_arr,
                    "i_arr": i_arr,
                }
                for _, file_row in ms1_cmp.iterrows()
                for rt_arr, i_arr in (_parse_spectrum(file_row.get("raw_spectrum")),)
            ]

        tasks.append({
            "mc_row_dict": mc_row.to_dict(),
            "compound_name": compound_name,
            "adduct": adduct,
            "rt_alignment_num": rt_alignment_num,
            "analysis_num": analysis_num,
            "compound_idx": cmp_idx,
            "path_shared": str(path_shared),
            "path_indep": str(path_indep),
            "file_items": file_items,
        })

    if not tasks:
        logger.info("Nothing to generate (all PDFs already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(
        "Generating EIC thumbnail PDFs for %d compounds using %d workers...",
        len(tasks), n_workers,
    )

    pbar = tqdm(total=len(tasks), desc="Generating EIC thumbnails", unit="compound")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_compound_pdf_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Wrote EIC PDFs for %s", name)
            except Exception as exc:
                logger.error("Failed to write PDFs for %s: %s", name, exc)
            finally:
                pbar.update(1)

    pbar.close()

def _file_group(file_path: str) -> str:
    """Return the file group label: the segment at underscore-separated index 11 of the stem.

    Example: ``..._groupA_...mzML`` → ``"groupA"`` (assuming groupA is at position 11).
    Falls back to the full stem when the filename has fewer than 12 ``_`` segments.
    """
    if not file_path:
        return "unknown"
    stem = os.path.basename(file_path).split(".")[0]
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
            "peak_height", "peak_area", "rt_peak", "rt_centroid",
            "mz_peak", "mz_centroid",
        ])

    records = []
    for _, row in ms1_all_df.iterrows():
        rt_arr, i_arr = _parse_spectrum(row.get("raw_spectrum"))
        fp = str(row.get("file_path", ""))
        group = _file_group(fp)
        base = {
            "inchi_key": row["inchi_key"],
            "adduct": row["adduct"],
            "file_path": fp,
            "file_group": group,
        }

        if not i_arr or max(i_arr) <= 0:
            records.append({**base,
                            "peak_height": np.nan, "peak_area": np.nan,
                            "rt_peak": np.nan, "rt_centroid": np.nan,
                            "mz_peak": np.nan, "mz_centroid": np.nan})
            continue

        i_np = np.array(i_arr, dtype=float)
        rt_np = np.array(rt_arr, dtype=float)
        idx_max = int(np.argmax(i_np))
        peak_height = float(i_np[idx_max])
        rt_peak_val = float(rt_np[idx_max])
        total_i = i_np.sum()
        rt_centroid_val = float((rt_np * i_np).sum() / total_i)
        peak_area_val = float(np.trapz(i_np, rt_np))

        mz_centroid = np.nan
        mz_peak_val = np.nan
        mz_json = row.get("mz")
        if mz_json is not None and not (isinstance(mz_json, float) and np.isnan(mz_json)):
            try:
                mz_np = np.array(json.loads(mz_json), dtype=float)
                if len(mz_np) == len(i_np) and total_i > 0:
                    mz_centroid = float((mz_np * i_np).sum() / total_i)
                    mz_peak_val = float(mz_np[idx_max])
            except Exception:
                pass

        records.append({**base,
                        "peak_height": peak_height,
                        "peak_area": peak_area_val,
                        "rt_peak": rt_peak_val,
                        "rt_centroid": rt_centroid_val,
                        "mz_peak": mz_peak_val,
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
    rng = np.random.default_rng(seed=42)
    groups = sorted(compound_metrics["file_group"].dropna().unique()) if not compound_metrics.empty else []

    data_per_group: List[List[float]] = []
    valid_groups: List[str] = []
    for g in groups:
        group_rows = compound_metrics.loc[compound_metrics["file_group"] == g]
        vals = group_rows[metric].dropna().tolist()
        if log_scale:
            vals = [np.log10(v) for v in vals if isinstance(v, (int, float)) and v > 0]
        n_files = len(group_rows)
        data_per_group.append(vals if vals else [0.0] * max(n_files, 1))
        valid_groups.append(g)

    if not data_per_group:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", fontsize=8, color="gray")
        title_top = f"{compound_idx:04d}  {compound_name}  {adduct}"
        if metric == "mz_centroid":
            title_bottom = f"m/z centroid: {atlas_ref if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)) else 'N/A'}"
            ax.set_title(f"{title_top}\n{title_bottom}", fontsize=13, pad=2, loc="center", fontweight="bold")
        else:
            ax.set_title(title_top, fontsize=13, pad=2, loc="center", fontweight="bold")
        
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
        ax.legend(fontsize=10, loc="upper right", framealpha=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(valid_groups, rotation=45, ha="right", fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    # Title formatting: large top line, smaller below
    title_top = f"{compound_idx:04d}  {compound_name}  {adduct}"
    if metric == "mz_centroid":
        mz_val = atlas_ref if atlas_ref is not None and not (isinstance(atlas_ref, float) and np.isnan(atlas_ref)) else 'N/A'
        title_bottom = f"m/z centroid: {mz_val}"
        ax.set_title(f"{title_top}\n{title_bottom}", fontsize=13, pad=2, loc="center", fontweight="bold")
    else:
        ax.set_title(title_top, fontsize=13, pad=2, loc="center", fontweight="bold")
    

    # For linear scales, remove scientific notation multiplier and offset
    if not log_scale:
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

def _boxplot_compound_worker(kwargs: dict) -> str:
    """Worker: generate all metric boxplot PDFs for one compound."""
    import matplotlib
    matplotlib.use("Agg")

    compound_name = kwargs["compound_name"]
    adduct = kwargs["adduct"]
    inchi_key = kwargs["inchi_key"]
    cmp_idx = kwargs["compound_idx"]
    atlas_ref_dict = kwargs["atlas_ref_dict"]
    metric_configs = kwargs["metric_configs"]
    cmp_metrics = kwargs["cmp_metrics"]

    safe_stem = f"{cmp_idx:04d}_{compound_name}_{adduct}_{inchi_key}".replace("/", "-").replace(" ", "_")

    for metric, log_scale, ylabel, atlas_attr, metric_dir_str in metric_configs:
        pdf_path = Path(metric_dir_str) / f"{safe_stem}.pdf"

        atlas_ref = atlas_ref_dict.get(atlas_attr) if atlas_attr else None

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(bottom=0.2, left=0.12, right=0.97, top=0.88)
        try:
            _plot_compound_boxplot(
                ax, cmp_metrics, metric, log_scale,
                atlas_ref, compound_name, adduct, ylabel,
                compound_idx=cmp_idx,
            )
            with PdfPages(pdf_path) as pdf:
                pdf.savefig(fig, bbox_inches="tight")
        finally:
            plt.close(fig)

    return compound_name


def make_boxplots(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
    max_workers: Optional[int] = None,
) -> None:

    output_dir = Path(output_loc) / "boxplots"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info("Overwriting disabled: existing directory %s will be used (existing PDFs will be preserved).", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting identification figures to %s", output_dir)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df.reset_index(drop=True)
    per_file_df = summary_obj.per_file_metrics_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - nothing to plot.")
        return

    # ── 1. Build atlas lookup (unchanged) ────────────────────────────────────
    atlas_lookup: Dict[Tuple[str, str], Dict[str, float]] = {
        (mc.get("inchi_key", ""), mc.get("adduct", "")): {
            "atlas_mz":      float(mc.get("atlas_mz", np.nan)),
            "atlas_rt_peak": float(mc.get("atlas_rt_peak", np.nan)),
        }
        for _, mc in manual_curation_df.iterrows()
    }

    # ── 2. Create metric output dirs upfront, store paths as strings ──────────
    _METRIC_CONFIGS = [
        ("peak_height", False, "Peak Height (intensity)", None),
        ("peak_height", True,  "Peak Height (log₁₀)",     None),
        ("rt_peak",     False, "RT Peak (min)",           "atlas_rt_peak"),
        ("mz_centroid", False, "m/z Centroid",            "atlas_mz"),
    ]
    metric_configs_with_dirs: list[tuple] = []
    for metric, log_scale, ylabel, atlas_attr in _METRIC_CONFIGS:
        metric_dir = output_dir / f"{metric}_{'log' if log_scale else 'linear'}"
        metric_dir.mkdir(parents=True, exist_ok=True)
        metric_configs_with_dirs.append((metric, log_scale, ylabel, atlas_attr, str(metric_dir)))

    # ── 3. Pre-group per_file_df — replaces 4×n O(n) filters with O(1) lookup
    if per_file_df is not None and not per_file_df.empty:
        pf_groups: dict = {
            key: grp.reset_index(drop=True)
            for key, grp in per_file_df.groupby(["inchi_key", "adduct"], sort=False)
        }
    else:
        pf_groups = {}
    empty_df = pd.DataFrame()

    # ── 4. One task per compound — worker handles all 4 metric PDFs ───────────
    tasks: list[dict] = []
    for cmp_idx, mc_row in manual_curation_df.iterrows():
        compound_name = mc_row.get("compound_name") or "unknown"
        adduct = mc_row.get("adduct", "")
        inchi_key = mc_row.get("inchi_key", "")
        key = (inchi_key, adduct)

        tasks.append({
            "compound_name": compound_name,
            "adduct": adduct,
            "inchi_key": inchi_key,
            "compound_idx": cmp_idx,
            "atlas_ref_dict": atlas_lookup.get(key, {}),
            "metric_configs": metric_configs_with_dirs,
            "cmp_metrics": pf_groups.get(key, empty_df),
        })

    if not tasks:
        logger.info("Nothing to generate (all boxplot PDFs already exist).")
        return

    n_workers = max_workers or min(os.cpu_count() or 4, len(tasks))
    logger.info(
        "Generating boxplot PDFs for %d compounds (%d metric types) using %d workers...",
        len(tasks), len(_METRIC_CONFIGS), n_workers,
    )

    pbar = tqdm(total=len(tasks), desc="Generating boxplots", unit="compound")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_name = {
            executor.submit(_boxplot_compound_worker, task): task["compound_name"]
            for task in tasks
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                future.result()
                logger.debug("Wrote boxplot PDFs for %s", name)
            except Exception as exc:
                logger.error("Failed to generate boxplots for %s: %s", name, exc)
            finally:
                pbar.set_postfix(compound=name, refresh=False)
                pbar.update(1)

    pbar.close()

    logger.info("Boxplot PDFs complete → %s", output_dir)

def make_manual_curation_csv(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "manually_curated_compound_data.csv",
    overwrite: bool = True,
) -> None:
    """Write the ``manual_curation`` table to a CSV file (one row per compound).

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
        ``manual_curation_df`` is loaded automatically if not already cached.
    output_loc:
        Override output directory. Defaults to
        ``<project_directory>/analysis_tables/rt<N>_analysis<M>/``.
    output_filename:
        CSV filename (the ``.csv`` extension is appended automatically if absent).
    overwrite:
        When *False*, skips writing if the output file already exists.

    Returns
    -------
    pd.DataFrame
        The exported DataFrame (empty on error).
    """

    output_dir = Path(output_loc) / "data_sheets"
    output_file = output_dir / output_filename
    if not overwrite and output_file.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_file)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting manual curation CSV to %s", output_file)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - CSV not written.")
        return

    manual_curation_df.to_csv(output_file, index=False)
    logger.info("Exported manual curation CSV")
    return

def make_best_ms2_hit_fragment_ions_csv(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    output_filename: str = "best_ms2_hit_fragment_ions.csv",
    overwrite: bool = True,
    min_fragment_intensity: Optional[float] = 1e4,
) -> None:

    output_dir = Path(output_loc) / "data_sheets"
    output_file = output_dir / output_filename
    if not overwrite and output_file.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_file)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting best MS2 hit fragment ions CSV to %s", output_file)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    ms2_hits_all_df = summary_obj.ms2_hits_all_df

    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - best MS2 hit CSV not written.")
        return

    # Pre-compute best hit per (inchi_key, adduct)
    if ms2_hits_all_df is not None and not ms2_hits_all_df.empty:
        ms2_best: dict[tuple, pd.Series] = {
            key: grp.sort_values("score", ascending=False).iloc[0]
            for key, grp in ms2_hits_all_df.groupby(["inchi_key", "adduct"], sort=False)
        }
    else:
        ms2_best = {}

    rows: List[dict] = []
    for cmp_idx, mc_row in tqdm(
        manual_curation_df.reset_index(drop=True).iterrows(),
        total=len(manual_curation_df),
        desc="Finding best MS2 hits for compounds",
    ):
        inchi_key = mc_row.get("inchi_key", "")
        adduct = mc_row.get("adduct", "")
        compound_name = mc_row.get("compound_name") or f"compound_{cmp_idx}"

        best_hit = ms2_best.get((inchi_key, adduct))
        if best_hit is None:
            continue

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
            "compound_index": cmp_idx,
            "compound_name":  compound_name,
            "adduct": adduct,
            "file_name": os.path.basename(str(best_hit.get("file_path", ""))),
            "rt_peak": best_hit.get("rt", np.nan),
            "mz_peak": best_hit.get("mz_measured", np.nan),
            "spectrum": raw_spectrum,
        })

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        logger.warning("No MS2 hits found for any compound - best MS2 hit CSV not written.")
        return

    result_df.to_csv(output_file, index=False)
    logger.info(
        "Exported best MS2 hit fragment ions CSV (%d compounds) → %s",
        len(result_df), output_file,
    )
    return

def make_data_sheets(
    summary_obj: "AnalysisSummary",
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
) -> None:
    """Export per-compound, per-file quantitative metric tables as wide-format CSVs.

    Writes one CSV per metric into ``output_loc/data_sheets/``:

    - ``peak_height.csv``   — maximum EIC intensity per file
    - ``peak_area.csv``     — trapezoidal area under the EIC per file
    - ``rt_peak.csv``       — retention time at peak intensity per file
    - ``rt_centroid.csv``   — intensity-weighted mean retention time per file
    - ``mz_peak.csv``       — m/z at peak intensity per file
    - ``mz_centroid.csv``   — intensity-weighted mean m/z per file

    Each CSV is wide-format: rows = compounds (compound_name, inchi_key, adduct),
    columns = one per input file (file stem without extension).

    Parameters
    ----------
    summary_obj:
        Configured :class:`AnalysisSummary` object (call ``.setup(...)`` first).
    output_loc:
        Base output directory.  A ``data_sheets`` sub-directory is created inside it.
    overwrite:
        When *False*, existing CSVs are skipped rather than overwritten.
    """
    output_dir = Path(output_loc) / "data_sheets"
    if overwrite and output_dir.exists():
        logger.info("Overwriting enabled: clearing existing contents of %s", output_dir)
        shutil.rmtree(output_dir)
    elif not overwrite and output_dir.exists():
        logger.info("Overwriting disabled: existing directory %s will be used (existing PDFs will be preserved).", output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting data sheets to %s", output_dir)

    if summary_obj.manual_curation_df is None:
        summary_obj.load_data()
    manual_curation_df = summary_obj.manual_curation_df
    if manual_curation_df is None or manual_curation_df.empty:
        logger.error("No manual curation entries found - data sheets not written.")
        return

    per_file_df = summary_obj.per_file_metrics_df
    if per_file_df is None or per_file_df.empty:
        logger.warning("per_file_metrics_df is empty - data sheets not written.")
        return

    # Join compound_name and compound_index from manual_curation_df
    mc_reset = manual_curation_df.reset_index(drop=True)
    mc_slim = mc_reset[["inchi_key", "adduct", "compound_name"]].copy()
    mc_slim["compound_index"] = mc_reset.index
    mc_slim = mc_slim.drop_duplicates(subset=["inchi_key", "adduct"])
    pfm = per_file_df.merge(mc_slim, on=["inchi_key", "adduct"], how="left")

    # Column labels: file stem (basename without extension)
    pfm = pfm.copy()
    pfm["_file_col"] = pfm["file_path"].apply(
        lambda p: os.path.splitext(os.path.basename(str(p)))[0] if p else "unknown"
    )

    _DATA_SHEET_METRICS = [
        "peak_height",
        "peak_area",
        "rt_peak",
        "rt_centroid",
        "mz_peak",
        "mz_centroid",
    ]
    _INDEX_COLS = [
        "compound_index",
        "compound_name", 
        "inchi_key", 
        "adduct"
    ]
    for metric in _DATA_SHEET_METRICS:
        if metric not in pfm.columns:
            logger.warning("Metric '%s' not found in per_file_metrics_df — skipping.", metric)
            continue

        csv_path = output_dir / f"{metric}.csv"

        wide = (
            pfm[_INDEX_COLS + ["_file_col", metric]]
            .pivot_table(
                index=_INDEX_COLS,
                columns="_file_col",
                values=metric,
                aggfunc="first",
            )
            .reset_index()
        )
        wide.columns.name = None
        wide.to_csv(csv_path, index=False)
        logger.info(
            "Exported %s data sheet (%d compounds × %d files) → %s",
            metric, len(wide), len(wide.columns) - len(_INDEX_COLS), csv_path,
        )

    logger.info("Data sheets written to %s", output_dir)


def make_peak_height_filtered_csv(
    output_loc: Optional[Path] = None,
    overwrite: bool = True,
    control_fold_threshold: float = 3.0,
) -> None:
    """Filter and process the ``peak_height`` data sheet.

    Applies the following steps in order:

    1. **Control labeling** — adds a ``control_filter`` column (``"keep"`` or
       ``"remove"``) based on whether the maximum non-control peak height is at
       least ``control_fold_threshold`` × the maximum control peak height.
       Control files are identified by the presence of any of ``"ExCtrl"``,
       ``"TxCtrl"``, or ``"InjBL"`` in the column name.  All rows are retained
       regardless of this label.
       Compounds not detected in any non-control sample (all NaN) are labeled
       ``"remove"``.
    2. **Row deduplication** — drops the ``adduct`` and ``compound_index``
       columns, then merges rows sharing the same ``inchi_key`` /
       ``compound_name`` / ``control_filter`` by taking the column-wise maximum.
    3. **Column shortening & deduplication** — drops the last two
       ``_``-separated tokens from each data column name (e.g.
       ``…_MS1_POS`` → ``…``), then merges any resulting duplicate column
       names by taking the row-wise maximum.
    4. **Missing-value imputation** — fills remaining NaN cells with the
       global minimum measured value in the final matrix.
    5. **Export** — writes ``peak_height_filtered.csv`` into the same
       ``data_sheets`` sub-directory as ``peak_height.csv``.  Output columns
       are ordered: ``control_filter``, ``compound_name``, ``inchi_key``,
       followed by the data columns.

    Parameters
    ----------
    summary_obj:
        Configured :class:`AnalysisSummary` object (used only for the default
        output location).
    output_loc:
        Base output directory (same value passed to :func:`make_data_sheets`).
    overwrite:
        When *False*, skips writing if the output file already exists.
    control_fold_threshold:
        Fold-change required to retain a compound (default ``3.0``).

    Returns
    -------
    None
        The filtered, processed table (empty on error).
    """

    out_dir = Path(output_loc) / "data_sheets"
    source_csv = out_dir / "peak_height.csv"
    output_csv = out_dir / "peak_height_filtered.csv"
    if not source_csv.exists():
        logger.error("peak_height.csv not found at %s — run make_data_sheets first.", source_csv)
        return
    if not overwrite and output_csv.exists():
        logger.info("Overwriting disabled: existing file %s will be used.", output_csv)
        return

    logger.info("Creating filtered peak height CSV at %s", output_csv)

    df = pd.read_csv(source_csv)

    _META_COLS = {"compound_index", "compound_name", "inchi_key", "adduct"}
    data_cols = [c for c in df.columns if c not in _META_COLS]

    # ── 1. Control-signal filter ───────────────────────────────────────────────
    _CTRL_PATTERNS = ("ExCtrl", "TxCtrl", "InjBL")
    ctrl_cols = [c for c in data_cols if any(p in c for p in _CTRL_PATTERNS)]
    non_ctrl_cols = [c for c in data_cols if c not in ctrl_cols]

    logger.info(
        "Control filter: %d control columns, %d non-control columns.",
        len(ctrl_cols), len(non_ctrl_cols),
    )

    if not ctrl_cols:
        logger.info("No control columns found — marking all rows as 'keep'.")
        df["control_filter"] = "keep"
    else:
        max_ctrl = df[ctrl_cols].max(axis=1)         # NaN → not detected in any control
        max_non_ctrl = df[non_ctrl_cols].max(axis=1)  # NaN → not detected in any sample
        keep_mask = (
            max_non_ctrl.notna() &
            (max_ctrl.isna() | (max_non_ctrl >= control_fold_threshold * max_ctrl.fillna(0)))
        )
        df["control_filter"] = keep_mask.map({True: "keep", False: "remove"})

    n_keep = (df["control_filter"] == "keep").sum()
    logger.info("Control filter: %d keep, %d remove (of %d total).", n_keep, len(df) - n_keep, len(df))

    if df.empty:
        logger.warning("No compounds found — no output written.")
        return

    # ── 2. Drop extra metadata; merge identical inchi_key rows by max ──────────
    drop_cols = [c for c in ("compound_index", "adduct") if c in df.columns]
    df = df.drop(columns=drop_cols)

    n_before = len(df)
    df = df.groupby(["inchi_key", "compound_name", "control_filter"], sort=False).max().reset_index()
    logger.info("inchi_key row merge: %d → %d rows.", n_before, len(df))

    # ── 3. Shorten column names; merge duplicate columns by row-wise max ────────
    def _shorten(col: str) -> str:
        parts = col.split("_")
        return "_".join(parts[:-2]) if len(parts) > 2 else col

    _META_FIXED = {"inchi_key", "compound_name", "control_filter"}
    df.columns = [c if c in _META_FIXED else _shorten(c) for c in df.columns]

    if df.columns.duplicated().any():
        data_df = df[[c for c in df.columns if c not in _META_FIXED]]
        # Transpose so column names become the index, then groupby merges duplicates
        data_merged = data_df.T.groupby(level=0).max().T.reset_index(drop=True)
        df = pd.concat(
            [df[list(_META_FIXED)].reset_index(drop=True), data_merged], axis=1
        )
        logger.info(
            "Merged duplicate columns after shortening; %d data columns remain.",
            len(data_merged.columns),
        )

    # ── 4. Impute NaN cells with the global matrix minimum ────────────────────
    data_cols_final = [c for c in df.columns if c not in _META_FIXED]
    global_min = float(df[data_cols_final].min().min()) if data_cols_final else np.nan
    if not np.isnan(global_min):
        n_nan = int(df[data_cols_final].isna().sum().sum())
        df[data_cols_final] = df[data_cols_final].fillna(global_min)
        logger.info("Imputed %d NaN cells with global minimum: %g", n_nan, global_min)
    else:
        logger.warning("Global minimum is NaN — no imputation performed.")

    # ── 5. Reorder columns: control_filter first, then compound_name, inchi_key, data ──
    other_cols = [c for c in df.columns if c not in _META_FIXED]
    df = df[["control_filter", "compound_name", "inchi_key"] + other_cols]

    # ── 6. Export ──────────────────────────────────────────────────────────────
    df.to_csv(output_csv, index=False)
    logger.info(
        "Exported filtered peak height (%d compounds × %d files) → %s",
        len(df), len(df.columns) - len(_META_FIXED), output_csv,
    )
    return

def make_metabomap(
    summary_obj: "AnalysisSummary",
    overwrite: bool = True,
) -> None:
    """Merge pos/neg filtered peak-height tables and compute pairwise log2 fold-changes.

    Checks whether the sibling-polarity ``peak_height_filtered.csv`` (same analysis
    type, opposite polarity) exists alongside the current one.  When found, the two
    tables are merged on ``inchi_key``; for compounds shared between polarities the
    higher peak-height value is kept per matched replicate column.  QC and ISTD
    sample groups are excluded.  Two output CSVs are written to
    ``<analysis_output_dir>/metabomaps/``:

    ``merged_peak_heights.csv``
        One row per unique ``inchi_key``; columns are the sample-group name (index 12
        of the underscore-split column name, e.g. ``M-HighS-HighL-12h-HeatStr``).
        Replicate columns share the same group name.  Values are the element-wise
        maximum across the two polarities (positional matching after sorting within
        each group alphabetically).

    ``log_fold_changes.csv``
        One row per ``inchi_key``; columns are ``group1_vs_group2`` for every
        pairwise combination of unique sample groups.  LFC is
        ``log2(mean(group1_replicates) / mean(group2_replicates))``.  Rows or
        pairs where either mean is zero or NaN are set to ``NaN``.

    Parameters
    ----------
    summary_obj:
        Configured ``AnalysisSummary`` object (call ``.setup(...)`` first).
    overwrite:
        When *False*, skips writing if both output files already exist.
    """
    analysis_output_dir = Path(summary_obj.paths.get("analysis_output_dir"))
    current_polarity = summary_obj.post_autoid_atlas_obj.polarity   # e.g. "POS"
    analysis_type    = summary_obj.post_autoid_atlas_obj.analysis_type  # e.g. "EMA"

    if current_polarity.upper() == "POS":
        sibling_polarity = "NEG"
    elif current_polarity.upper() == "NEG":
        sibling_polarity = "POS"
    else:
        logger.error(
            "Unrecognised polarity '%s' — cannot determine sibling polarity for metabomap.",
            current_polarity,
        )
        return

    current_csv = (
        analysis_output_dir / f"{current_polarity}-{analysis_type}"
        / "data_sheets" / "peak_height_filtered.csv"
    )
    sibling_csv = (
        analysis_output_dir / f"{sibling_polarity}-{analysis_type}"
        / "data_sheets" / "peak_height_filtered.csv"
    )

    metabomaps_dir = analysis_output_dir / "metabomaps"
    merged_csv     = metabomaps_dir / "merged_peak_heights.csv"
    lfc_csv        = metabomaps_dir / "log_fold_changes.csv"

    if not overwrite and merged_csv.exists() and lfc_csv.exists():
        logger.info(
            "Overwriting disabled: metabomap files already exist in %s", metabomaps_dir
        )
        return

    if not current_csv.exists():
        logger.error("Current polarity filtered peak-height CSV not found: %s", current_csv)
        return

    if not sibling_csv.exists():
        logger.info(
            "Sibling polarity (%s-%s) peak_height_filtered.csv not yet available"
            " — skipping metabomap.",
            sibling_polarity, analysis_type,
        )
        return

    metabomaps_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Building metabomap from:\n  %s\n  %s", current_csv, sibling_csv)

    # ── Constants ──────────────────────────────────────────────────────────────
    _META_COLS_SET      = {"control_filter", "compound_name", "inchi_key"}
    _EXCLUDE_PATTERNS   = ("QC", "ISTD")

    def _col_group(col: str) -> Optional[str]:
        """Return sample-group label (underscore-split index 12) or None to drop."""
        parts = col.split("_")
        if len(parts) <= 12:
            return None
        grp = parts[12]
        if any(p in grp for p in _EXCLUDE_PATTERNS):
            return None
        return grp

    def _load_filtered(csv_path: Path) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        keep = [c for c in df.columns if c in _META_COLS_SET or _col_group(c) is not None]
        return df[keep].copy()

    cur_df = _load_filtered(current_csv)
    sib_df = _load_filtered(sibling_csv)

    # ── Build sorted data column lists (for positional replicate matching) ────
    cur_data_cols = [c for c in cur_df.columns if c not in _META_COLS_SET]
    sib_data_cols = [c for c in sib_df.columns if c not in _META_COLS_SET]

    # Sort by (group_name, full_col_name) so replicates are consistently ordered
    cur_sorted = sorted(cur_data_cols, key=lambda c: (_col_group(c), c))
    sib_sorted = sorted(sib_data_cols, key=lambda c: (_col_group(c), c))

    cur_groups = [_col_group(c) for c in cur_sorted]
    sib_groups = [_col_group(c) for c in sib_sorted]

    # ── Index both frames by inchi_key ────────────────────────────────────────
    cur_indexed = cur_df.set_index("inchi_key")
    sib_indexed = sib_df.set_index("inchi_key")
    all_inchi_keys = cur_indexed.index.union(sib_indexed.index)

    cur_matrix = cur_indexed[cur_sorted].reindex(all_inchi_keys)
    sib_matrix = sib_indexed[sib_sorted].reindex(all_inchi_keys)

    # ── Merge: element-wise max when groups match; group-max fallback ─────────
    if cur_groups == sib_groups:
        logger.info(
            "Column groups match between polarities (%d replicates) "
            "— using element-wise maximum.",
            len(cur_groups),
        )
        merged_vals = np.fmax(cur_matrix.values, sib_matrix.values)
        merged_data_df = pd.DataFrame(
            merged_vals,
            index=all_inchi_keys,
            columns=cur_groups,  # rename to group names
        )
    else:
        logger.warning(
            "Column groups differ between polarities (%d cur vs %d sib) "
            "— using per-group maximum (per-replicate resolution not available).",
            len(cur_groups), len(sib_groups),
        )
        all_groups_ordered = list(dict.fromkeys(cur_groups + sib_groups))
        merged_data_df = pd.DataFrame(index=all_inchi_keys)
        for grp in all_groups_ordered:
            cur_grp_cols = [c for c in cur_sorted if _col_group(c) == grp]
            sib_grp_cols = [c for c in sib_sorted if _col_group(c) == grp]
            grp_vals = pd.concat(
                [cur_matrix[cur_grp_cols], sib_matrix[sib_grp_cols]], axis=1
            )
            merged_data_df[grp] = grp_vals.max(axis=1)

    merged_data_df.index.name = "inchi_key"
    merged_data_df = merged_data_df.reset_index()

    # Restore compound_name (prefer current polarity, fall back to sibling)
    compound_names = (
        cur_indexed["compound_name"].combine_first(sib_indexed["compound_name"])
        if "compound_name" in cur_indexed.columns and "compound_name" in sib_indexed.columns
        else cur_indexed.get("compound_name", sib_indexed.get("compound_name", pd.Series(dtype=str)))
    )
    merged_data_df.insert(1, "compound_name", merged_data_df["inchi_key"].map(compound_names))

    logger.info(
        "Merged %d (cur) + %d (sib) → %d unique inchi_keys, %d sample columns",
        len(cur_df), len(sib_df), len(merged_data_df),
        len(merged_data_df.columns) - 2,
    )

    merged_data_df.to_csv(merged_csv, index=False)
    logger.info("Saved merged peak heights → %s", merged_csv)

    # ── Pairwise log2 fold-change table ───────────────────────────────────────
    lfc_meta = {"inchi_key", "compound_name"}
    data_cols_renamed = [c for c in merged_data_df.columns if c not in lfc_meta]
    unique_groups = list(dict.fromkeys(data_cols_renamed))

    if len(unique_groups) < 2:
        logger.warning(
            "Fewer than 2 unique sample groups found — pairwise LFC table not created."
        )
        return

    # Compute per-group mean across all replicate columns (duplicate column names)
    group_means: dict[str, np.ndarray] = {}
    for grp in unique_groups:
        grp_data = merged_data_df.loc[:, merged_data_df.columns == grp]
        group_means[grp] = grp_data.mean(axis=1).to_numpy(dtype=float)

    lfc_records: dict[str, np.ndarray] = {
        "inchi_key":     merged_data_df["inchi_key"].to_numpy(),
        "compound_name": merged_data_df["compound_name"].to_numpy(),
    }
    for grp1, grp2 in itertools.combinations(unique_groups, 2):
        m1 = group_means[grp1]
        m2 = group_means[grp2]
        valid = (m1 > 0) & (m2 > 0) & ~np.isnan(m1) & ~np.isnan(m2)
        lfc = np.full(len(m1), np.nan)
        lfc[valid] = np.log2(m1[valid] / m2[valid])
        lfc_records[f"{grp1}_vs_{grp2}"] = lfc

    lfc_df = pd.DataFrame(lfc_records)
    lfc_df.to_csv(lfc_csv, index=False)
    logger.info(
        "Saved pairwise LFC table (%d compounds × %d group pairs) → %s",
        len(lfc_df), len(lfc_df.columns) - 2, lfc_csv,
    )
    return


def run_all_summaries(
    summary_obj: "AnalysisSummary",
    overwrite: bool = False,
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
    output_loc = Path(summary_obj.paths.get("analysis_output_dir", None)) / f"{summary_obj.post_autoid_atlas_obj.polarity}-{summary_obj.post_autoid_atlas_obj.analysis_type}"
    os.makedirs(output_loc, exist_ok=True)
    if summary_obj.override_parameters.get("skip_outputs") is not None:
        skip_outputs = summary_obj.override_parameters.get("skip_outputs", [])
    else:
        skip_outputs = summary_obj.config.get("skip_outputs", [])
    
    if summary_obj.manual_curation_df is None or summary_obj.manual_curation_df.empty:
        logger.error("No manual curation entries found - aborting run_all_summaries.")
        return

    if "final_id_sheet" not in (skip_outputs or []):
        logger.info("Making Final Identification sheet...")
        make_final_id_sheet(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "id_figures" not in (skip_outputs or []):
        logger.info("Making Identification figures...")
        make_identification_figure(summary_obj, output_loc=output_loc, overwrite=overwrite, max_workers=8)

    if "eic_thumbnails" not in (skip_outputs or []):
        logger.info("Making EIC thumbnails...")
        make_eic_thumbnails(summary_obj, output_loc=output_loc, overwrite=overwrite, max_workers=8)

    if "boxplots" not in (skip_outputs or []):
        logger.info("Making Boxplots...")
        make_boxplots(summary_obj, output_loc=output_loc, overwrite=overwrite, max_workers=8)

    if "data_sheets" not in (skip_outputs or []):
        logger.info("Making quantitative data sheets...")
        make_data_sheets(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "manual_curation_csv" not in (skip_outputs or []):
        logger.info("Making Manual curation CSV...")
        make_manual_curation_csv(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "best_ms2_hits_csv" not in (skip_outputs or []):
        logger.info("Making best MS2 hit fragment ions CSV...")
        make_best_ms2_hit_fragment_ions_csv(summary_obj, output_loc=output_loc, overwrite=overwrite)

    if "peak_height_filtered_csv" not in (skip_outputs or []):
        logger.info("Making filtered peak height CSV...")
        make_peak_height_filtered_csv(output_loc=output_loc, overwrite=overwrite)

    if "metabomap" not in (skip_outputs or []):
        logger.info("Making metabomap (merged pos/neg peak heights + LFC table)...")
        make_metabomap(summary_obj, overwrite=overwrite)

    logger.info("Exporting post-auto-ID atlas data CSV...")
    summary_obj.post_autoid_atlas_obj.to_dataframe().to_csv(
        f"{summary_obj.paths['analysis_output_dir']}/{summary_obj.post_autoid_atlas_obj.atlas_uid}.csv", index=False
    )

    logger.info("Exporting post-curation atlas data CSV...")
    summary_obj.post_curation_atlas_obj.to_dataframe().to_csv(
        f"{summary_obj.paths['analysis_output_dir']}/{summary_obj.post_curation_atlas_obj.atlas_uid}.csv", index=False
    )

    logger.info("Saving input yaml config to analysis output directory...")
    with open(f"{summary_obj.paths['analysis_output_dir']}/RTA{summary_obj.rt_alignment_number}_TGA{summary_obj.analysis_number}_analysis_config.yaml", "w") as f:
        with open(summary_obj.config_path, "r") as original:
            f.write(original.read())

    logger.info("Uploading outputs to Google Drive...")
    rcl.copy_outputs_to_google_drive(summary_obj, overwrite=overwrite)