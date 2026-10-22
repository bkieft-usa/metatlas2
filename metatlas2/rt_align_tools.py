from __future__ import annotations

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
import json
import yaml

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import metatlas2.logging_config as lcf
from metatlas2.utils import should_disable_tqdm
import metatlas2.file_and_project_format as fpf
logger = lcf.get_logger('rt_align_tools')

def calculate_model_values_from_existing(model_dict: dict) -> dict:
    """Reconstruct callable sklearn objects from a serialised model dict.

    Supports ``polynomial`` and ``linear`` model types.  ``median_offset``
    models store only a scalar offset and do not need sklearn reconstruction.
    """
    model_type = model_dict.get('model_type', 'polynomial')
    metadata = model_dict.get('metadata', {})

    if model_type == 'median_offset':
        # Nothing to reconstruct — offset is stored directly.
        return model_dict

    # polynomial or linear — both use PolynomialFeatures + LinearRegression
    degree = metadata.get('poly_degree', model_dict.get('degree', 1))
    poly_features = PolynomialFeatures(
        degree=degree,
        include_bias=metadata.get('poly_include_bias', True),
        interaction_only=metadata.get('poly_interaction_only', False)
    )
    dummy_X = np.array([[0]]).reshape(-1, 1)
    poly_features.fit(dummy_X)

    model = LinearRegression()
    model.coef_ = np.array(metadata.get('model_coefficients', model_dict.get('coefficients', [])))
    model.intercept_ = metadata.get('model_intercept', model_dict.get('intercept', 0.0))

    model_dict['poly_features'] = poly_features
    model_dict['model'] = model
    model_dict['intercept'] = model.intercept_
    return model_dict


def build_polynomial_model(X: np.ndarray, y: np.ndarray, degree: int) -> dict:
    """Fit a polynomial regression model of the given *degree*.

    Args:
        X:      1-D array of atlas RT values (predictor).
        y:      1-D array of observed RT values (response).
        degree: Polynomial degree (≥ 1).

    Returns:
        Dict with keys ``model``, ``poly_features``, ``degree``, ``r2``,
        ``rmse``, ``y_pred``, ``coefficients``, ``intercept``.
    """
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))

    model = LinearRegression()
    model.fit(X_poly, y)

    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    return {
        'model_type': 'polynomial',
        'model': model,
        'poly_features': poly_features,
        'degree': degree,
        'r2': r2,
        'rmse': rmse,
        'y_pred': y_pred,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
    }


def build_linear_model(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a simple linear regression model (degree-1 polynomial).

    Convenience wrapper around :func:`build_polynomial_model` with
    ``degree=1``.  The returned dict is identical in structure so that
    :func:`_apply_rt_model` works unchanged.

    Args:
        X: 1-D array of atlas RT values.
        y: 1-D array of observed RT values.

    Returns:
        Same dict structure as :func:`build_polynomial_model` with
        ``model_type`` set to ``"linear"``.
    """
    result = build_polynomial_model(X, y, degree=1)
    result['model_type'] = 'linear'
    return result


def build_median_offset_model(X: np.ndarray, y: np.ndarray) -> dict:
    """Build a median-offset RT alignment model.

    Computes the median of ``(observed_RT − atlas_RT)`` across all compounds
    and applies that single scalar shift uniformly.  This is the most robust
    model when only a few QC compounds are available or when the RT drift is
    approximately constant across the gradient.

    Args:
        X: 1-D array of atlas RT values.
        y: 1-D array of observed RT values (medians per compound).

    Returns:
        Dict with keys ``model_type``, ``offset``, ``r2``, ``rmse``,
        ``y_pred``, ``degree`` (always 0), ``coefficients`` (empty array),
        ``intercept`` (the offset value).  ``poly_features`` and ``model``
        are ``None`` — :func:`_apply_rt_model` handles this case directly.
    """
    offsets = y - X
    offset = float(np.median(offsets))
    y_pred = X + offset

    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

    return {
        'model_type': 'median_offset',
        'offset': offset,
        'degree': 0,
        'r2': r2,
        'rmse': rmse,
        'y_pred': y_pred,
        'coefficients': np.array([]),
        'intercept': offset,
        'poly_features': None,
        'model': None,
    }


def format_model_equation(model_info: dict) -> str:
    """Return a human-readable equation string for any supported model type."""
    model_type = model_info.get('model_type', 'polynomial')

    if model_type == 'median_offset':
        offset = model_info.get('offset', model_info.get('intercept', 0.0))
        sign = '+' if offset >= 0 else '-'
        return f"RT_aligned = RT_atlas {sign} {abs(offset):.6f}"

    # polynomial / linear — use degree and coefficients
    degree = model_info.get('degree', 1)
    coeffs = model_info.get('coefficients', [])
    intercept = model_info.get('intercept', 0.0)

    if degree == 0 or len(coeffs) < 2:
        return f"RT_aligned = {intercept:.6f}"
    if degree == 1:
        return f"RT_aligned = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas"
    if degree == 2:
        return (
            f"RT_aligned = {intercept:.6f}"
            f" + {coeffs[1]:.6f} * RT_atlas"
            f" + {coeffs[2]:.6f} * RT_atlas^2"
        )
    if degree == 3:
        return (
            f"RT_aligned = {intercept:.6f}"
            f" + {coeffs[1]:.6f} * RT_atlas"
            f" + {coeffs[2]:.6f} * RT_atlas^2"
            f" + {coeffs[3]:.6f} * RT_atlas\u00b3"
        )
    return f"Polynomial degree {degree} (coefficients: {coeffs})"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_all_three_models(X: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Build linear, polynomial (degree-2), and median-offset models.

    Args:
        X: 1-D array of atlas RT values.
        y: 1-D array of observed RT values (one per compound).

    Returns:
        Dict keyed by ``"linear"``, ``"polynomial"``, ``"median_offset"``.
    """
    linear_model = build_linear_model(X, y)
    linear_model['equation'] = format_model_equation(linear_model)

    polynomial_model = build_polynomial_model(X, y, degree=2)
    polynomial_model['equation'] = format_model_equation(polynomial_model)

    median_offset_model = build_median_offset_model(X, y)
    median_offset_model['equation'] = format_model_equation(median_offset_model)

    return {
        'linear': linear_model,
        'polynomial': polynomial_model,
        'median_offset': median_offset_model,
    }


def _extract_per_file_peak_rts(
    ms1_df: pd.DataFrame,
    compound_mzrts: dict,
    exclude_inchikeys: set,
) -> dict[str, dict[str, dict]]:
    """Extract the RT at the intensity peak for each compound × file.

    For every QC atlas compound, and for every QC file that has MS1 data for
    that compound, this function finds the single scan point with the highest
    intensity among all ``in_feature=True`` points and records its RT.  This
    is the true chromatographic peak RT, which is the correct value to regress
    against the atlas RT peak.

    Args:
        ms1_df:           Wide-format MS1 DataFrame (one row per
                          ``mz_rt_uid × filename``) with list columns
                          ``spec_rts``, ``spec_ints``, ``in_feature``.
        compound_mzrts:   Dict of ``mz_rt_uid → CompoundMzRt`` objects from
                          the QC atlas.
        exclude_inchikeys: Set of InChI keys to skip.

    Returns:
        Nested dict ``{mz_rt_uid: {filename: {"rt_peak": float,
        "intensity_peak": float, "atlas_rt_peak": float,
        "atlas_rt_min": float, "atlas_rt_max": float,
        "atlas_mz": float, "compound_name": str,
        "inchi_key": str, "adduct": str,
        "compound_uid": str}}}``.
    """
    ms1_grouped = ms1_df.groupby('mz_rt_uid', sort=False)
    per_file_data: dict[str, dict[str, dict]] = {}

    for compound_mzrt in compound_mzrts.values():
        mz_rt_uid     = getattr(compound_mzrt, 'mz_rt_uid', None)
        inchi_key      = compound_mzrt.inchi_key
        adduct         = compound_mzrt.adduct
        compound_uid   = compound_mzrt.compound_uid
        compound_name  = compound_mzrt.compound_name
        atlas_rt_peak  = compound_mzrt.rt_peak
        atlas_rt_min   = compound_mzrt.rt_min
        atlas_rt_max   = compound_mzrt.rt_max
        atlas_mz       = compound_mzrt.mz

        if exclude_inchikeys and inchi_key in exclude_inchikeys:
            continue

        try:
            ms1_df_comp = ms1_grouped.get_group(mz_rt_uid)
        except KeyError:
            continue

        file_peaks: dict[str, dict] = {}
        for _, row in ms1_df_comp.iterrows():
            filename = row['filename']
            try:
                arr_rts  = np.asarray(row['spec_rts'],  dtype=np.float64)
                arr_ints = np.asarray(row['spec_ints'], dtype=np.float64)
                arr_mask = np.asarray(row['in_feature'], dtype=bool)
            except (TypeError, ValueError):
                continue

            min_len = min(arr_rts.size, arr_ints.size, arr_mask.size)
            if min_len == 0:
                continue
            arr_rts  = arr_rts[:min_len]
            arr_ints = arr_ints[:min_len]
            arr_mask = arr_mask[:min_len]

            if not arr_mask.any():
                continue

            in_rts  = arr_rts[arr_mask]
            in_ints = arr_ints[arr_mask]

            peak_idx = int(np.argmax(in_ints))
            file_peaks[filename] = {
                'rt_peak':        float(in_rts[peak_idx]),
                'intensity_peak': float(in_ints[peak_idx]),
                'atlas_rt_peak':  atlas_rt_peak,
                'atlas_rt_min':   atlas_rt_min,
                'atlas_rt_max':   atlas_rt_max,
                'atlas_mz':       atlas_mz,
                'compound_name':  compound_name,
                'inchi_key':      inchi_key,
                'adduct':         adduct,
                'compound_uid':   compound_uid,
            }

        if file_peaks:
            per_file_data[mz_rt_uid] = file_peaks

    return per_file_data


def _per_file_peak_rts_to_modeling_df(
    per_file_data: dict[str, dict[str, dict]],
    filename: str,
) -> pd.DataFrame:
    """Build a per-compound modeling DataFrame for a single QC file.

    Args:
        per_file_data: Output of :func:`_extract_per_file_peak_rts`.
        filename:      The QC filename to extract data for.

    Returns:
        DataFrame with one row per compound that has data in *filename*,
        containing ``atlas_rt_peak``, ``exp_rt_peak``, ``compound_name``,
        ``inchi_key``, ``adduct``, ``compound_uid``, ``mz_rt_uid``,
        ``intensity_peak`` columns.  Empty DataFrame if no data.
    """
    rows = []
    for mz_rt_uid, file_dict in per_file_data.items():
        if filename not in file_dict:
            continue
        d = file_dict[filename]
        rows.append({
            'mz_rt_uid':      mz_rt_uid,
            'compound_uid':   d['compound_uid'],
            'compound_name':  d['compound_name'],
            'inchi_key':      d['inchi_key'],
            'adduct':         d['adduct'],
            'atlas_rt_peak':  d['atlas_rt_peak'],
            'atlas_rt_min':   d['atlas_rt_min'],
            'atlas_rt_max':   d['atlas_rt_max'],
            'atlas_mz':       d['atlas_mz'],
            'exp_rt_peak':    d['rt_peak'],
            'intensity_peak': d['intensity_peak'],
        })
    return pd.DataFrame(rows)


def _composite_peak_rts(
    per_file_data: dict[str, dict[str, dict]],
    mode: str,
) -> pd.DataFrame:
    """Build a composite modeling DataFrame by aggregating across all QC files.

    For each compound, collects the per-file peak RTs and reduces them with
    *mode* (``"min"``, ``"median"``, or ``"max"``).

    Args:
        per_file_data: Output of :func:`_extract_per_file_peak_rts`.
        mode:          Aggregation mode — ``"min"``, ``"median"``, or
                       ``"max"``.

    Returns:
        DataFrame with one row per compound, same schema as
        :func:`_per_file_peak_rts_to_modeling_df`.
    """
    _AGG = {'min': np.min, 'median': np.median, 'max': np.max}
    agg_fn = _AGG[mode]

    rows = []
    for mz_rt_uid, file_dict in per_file_data.items():
        if not file_dict:
            continue
        # Use metadata from the first file entry
        first = next(iter(file_dict.values()))
        peak_rts = [v['rt_peak'] for v in file_dict.values()]
        rows.append({
            'mz_rt_uid':      mz_rt_uid,
            'compound_uid':   first['compound_uid'],
            'compound_name':  first['compound_name'],
            'inchi_key':      first['inchi_key'],
            'adduct':         first['adduct'],
            'atlas_rt_peak':  first['atlas_rt_peak'],
            'atlas_rt_min':   first['atlas_rt_min'],
            'atlas_rt_max':   first['atlas_rt_max'],
            'atlas_mz':       first['atlas_mz'],
            'exp_rt_peak':    float(agg_fn(peak_rts)),
            'intensity_peak': float(np.max([v['intensity_peak'] for v in file_dict.values()])),
        })
    return pd.DataFrame(rows)


def _build_models_for_df(
    df: pd.DataFrame,
    min_observations: int,
    min_compounds: int,
    compounds_used: list[str] | None = None,
) -> dict[str, dict] | None:
    """Fit all three model types for a single modeling DataFrame.

    Args:
        df:               Per-compound DataFrame with ``atlas_rt_peak`` and
                          ``exp_rt_peak`` columns.
        min_observations: Minimum number of files a compound must appear in
                          (only used for composite DataFrames; pass 1 for
                          per-file DataFrames).
        min_compounds:    Minimum number of compounds required to fit a model.
        compounds_used:   Optional list of compound UIDs to attach to each
                          model dict.

    Returns:
        Dict ``{"linear": ..., "polynomial": ..., "median_offset": ...}``
        or ``None`` if there are insufficient compounds.
    """
    if df.empty or len(df) < min_compounds:
        return None

    X = df['atlas_rt_peak'].values
    y = df['exp_rt_peak'].values

    uid_list = compounds_used if compounds_used is not None else df['compound_uid'].tolist()
    models = _build_all_three_models(X, y)
    for m in models.values():
        m['compounds_used_for_modeling'] = uid_list
    return models


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_rt_alignment_model(rt_align_obj: "RTAlign", save_plot: bool = True):
    """
    Visualize RT alignment model results using RTAlign object.

    Produces ``RT_Alignment_Summary.pdf`` showing the chosen model type
    (default: polynomial) fitted to the chosen rt_peak_source
    (default: median across QC files).
    """

    logger.info("Plotting RT alignment model results to figure...")

    modeling_results_df = rt_align_obj.modeling_data
    rt_alignment_model = rt_align_obj.rt_alignment_model
    output_dir = rt_align_obj.paths['rt_alignment_results_dir']

    modeling_results_df = modeling_results_df.sort_values('atlas_rt_peak').reset_index(drop=True)
    modeling_results_df['compound_num'] = modeling_results_df.index + 1

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Atlas RT vs Observed RT (with model fit)
    model_type = rt_alignment_model.get('model_type', 'polynomial')
    degree = rt_alignment_model.get('degree', 0)
    if model_type == 'median_offset':
        fit_label = 'Median Offset Fit'
    elif model_type == 'linear':
        fit_label = 'Linear Fit'
    else:
        fit_label = f'Polynomial Fit (degree {degree})'

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(modeling_results_df['atlas_rt_peak'], modeling_results_df['exp_rt_peak'],
                alpha=0.7, s=50, c='blue', label='Observed Data')
    ax1.plot(modeling_results_df['atlas_rt_peak'], modeling_results_df['predicted_rt'],
            'r-', linewidth=2, label=fit_label)
    ax1.plot([modeling_results_df['atlas_rt_peak'].min(), modeling_results_df['atlas_rt_peak'].max()],
            [modeling_results_df['atlas_rt_peak'].min(), modeling_results_df['atlas_rt_peak'].max()],
            'k--', alpha=0.5, label='Perfect Correlation')
    ax1.set_xlabel('Atlas RT (min)')
    ax1.set_ylabel('Observed RT (min)')
    ax1.set_title(f'RT Correlation (R² = {rt_alignment_model["r2"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for _, row in modeling_results_df.iterrows():
        ax1.annotate(str(int(row['compound_num'])), 
                    (row['atlas_rt_peak'], row['exp_rt_peak']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Residuals vs Atlas RT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(modeling_results_df['atlas_rt_peak'], modeling_results_df['residual'], 
                alpha=0.7, s=50, c='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=modeling_results_df['residual'].std(), color='orange', linestyle=':', alpha=0.7, label='+1 σ')
    ax2.axhline(y=-modeling_results_df['residual'].std(), color='orange', linestyle=':', alpha=0.7, label='-1 σ')
    ax2.set_xlabel('Atlas RT (min)')
    ax2.set_ylabel('Residual (min)')
    ax2.set_title(f'Residuals vs Atlas RT (RMSE = {rt_alignment_model["rmse"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    for _, row in modeling_results_df.iterrows():
        ax2.annotate(str(int(row['compound_num'])), 
                    (row['atlas_rt_peak'], row['residual']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Compound number to name mapping (ordered by RT)
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    table_data = modeling_results_df[['compound_num', 'compound_name', 'inchi_key', 'adduct', 'atlas_rt_peak', 'residual']].copy()
    table_data['atlas_rt_peak'] = table_data['atlas_rt_peak'].round(3)
    table_data['residual'] = table_data['residual'].round(4)
    table_data.columns = ['#', 'Compound Name', 'InChi Key', 'Adduct', 'Atlas RT (min)', 'Residual (min)']
    table = ax_table.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        loc='center',
                        cellLoc='left',
                        colLoc='left')

    col_widths = [0.2, 0.5, 0.5, 0.3, 0.3]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for j in range(len(table_data) + 1):
            cell = table[(j, i)]
            cell.set_width(width)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.suptitle('RT Alignment Model Validation', fontsize=16, fontweight='bold', y=1.02)

    if save_plot:
        pdf_path = None
        try:
            plot_save_dir = Path(output_dir)
            plot_save_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = plot_save_dir / "RT_Alignment_Summary.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"Plot saved to {pdf_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        finally:
            if pdf_path is not None:
                logger.info(f"RT alignment model plot saved to {pdf_path}")

    return


# Colour palette for the three model types — consistent across all subplots.
_MODEL_COLORS = {
    'linear':        '#e41a1c',   # red
    'polynomial':    '#377eb8',   # blue
    'median_offset': '#4daf4a',   # green
}
_MODEL_LABELS = {
    'linear':        'Linear',
    'polynomial':    'Polynomial (deg 2)',
    'median_offset': 'Median Offset',
}


def _draw_facet(
    ax: plt.Axes,
    df: pd.DataFrame,
    models: dict[str, dict] | None,
    title: str,
    show_legend: bool = True,
) -> None:
    """Draw one facet (subplot) of the RT_Alignment_All figure.

    Plots the scatter of atlas RT vs observed peak RT, then overlays a
    regression line for each of the three model types.

    Args:
        ax:          Matplotlib Axes to draw on.
        df:          Per-compound DataFrame with ``atlas_rt_peak`` and
                     ``exp_rt_peak`` columns, sorted by ``atlas_rt_peak``.
        models:      Dict returned by :func:`_build_models_for_df`, or
                     ``None`` when there are insufficient data.
        title:       Subplot title string.
        show_legend: Whether to draw the legend on this axes.
    """
    if df is None or df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='grey')
        ax.set_title(title, fontsize=9)
        return

    df_sorted = df.sort_values('atlas_rt_peak')
    x_vals = df_sorted['atlas_rt_peak'].values
    y_vals = df_sorted['exp_rt_peak'].values

    ax.scatter(x_vals, y_vals, s=30, c='black', alpha=0.7, zorder=3, label='RT Peak')

    # 1:1 reference line
    lo, hi = x_vals.min(), x_vals.max()
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.3, linewidth=1, label='1:1')

    if models is not None:
        x_line = np.linspace(lo, hi, 200)
        for mt in ('linear', 'polynomial', 'median_offset'):
            m = models.get(mt)
            if m is None:
                continue
            y_line = _apply_rt_model(x_line, m)
            r2_val = m.get('r2', float('nan'))
            label = f"{_MODEL_LABELS[mt]} (R²={r2_val:.3f})"
            ax.plot(x_line, y_line,
                    color=_MODEL_COLORS[mt], linewidth=1.5,
                    label=label, zorder=2)

    ax.set_xlabel('Atlas RT (min)', fontsize=8)
    ax.set_ylabel('RT Peak (min)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(fontsize=6, loc='upper left')


def visualize_rt_alignment_all(rt_align_obj: "RTAlign", save_plot: bool = True) -> None:
    """Produce ``RT_Alignment_All.pdf`` — one facet per QC file plus composites.

    Each facet shows the atlas RT vs observed peak RT scatter and overlays
    three regression lines (linear, polynomial degree-2, median-offset) so
    the user can compare model types and per-file vs composite behaviour at a
    glance.

    Facets are ordered as:
        1. One subplot per QC file (sorted alphabetically by filename).
        2. Min composite.
        3. Median composite.
        4. Max composite.

    Args:
        rt_align_obj: RTAlign object after :func:`build_rt_alignment_model`
                      has been called (must have ``per_file_models``,
                      ``composite_models``, ``per_file_modeling_data``,
                      ``composite_modeling_data`` attributes populated).
        save_plot:    When ``True`` (default) the figure is saved to
                      ``rt_alignment_results_dir/RT_Alignment_All.pdf``.
    """
    logger.info("Plotting RT_Alignment_All figure...")

    per_file_models: dict[str, dict] = getattr(rt_align_obj, 'per_file_models', {})
    composite_models: dict[str, dict] = getattr(rt_align_obj, 'composite_models', {})
    per_file_dfs: dict[str, pd.DataFrame] = getattr(rt_align_obj, 'per_file_modeling_data', {})
    composite_dfs: dict[str, pd.DataFrame] = getattr(rt_align_obj, 'composite_modeling_data', {})
    output_dir = rt_align_obj.paths['rt_alignment_results_dir']

    filenames_sorted = sorted(per_file_models.keys())
    composite_keys = ['min', 'median', 'max']

    panels = filenames_sorted + composite_keys
    n_panels = len(panels)

    if n_panels == 0:
        logger.warning("No per-file or composite models found; skipping RT_Alignment_All plot.")
        return

    # Layout: try to keep roughly square grid
    n_cols = min(n_panels, 4)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    for idx, panel_key in enumerate(panels):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        if panel_key in filenames_sorted:
            df = per_file_dfs.get(panel_key)
            models = per_file_models.get(panel_key)
            short_name = fpf.get_filename_header_labels(panel_key)
            title = f"{short_name.get('short_samplename', panel_key)}"
        else:
            df = composite_dfs.get(panel_key)
            models = composite_models.get(panel_key)
            title = f"Composite — {panel_key.capitalize()}"

        _draw_facet(ax, df, models, title, show_legend=(idx == 0))

    # Hide any unused axes
    for idx in range(n_panels, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    # Shared legend for model colours (placed outside the grid)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=_MODEL_COLORS['linear'],        linewidth=2, label='Linear'),
        Line2D([0], [0], color=_MODEL_COLORS['polynomial'],    linewidth=2, label='Polynomial (deg 2)'),
        Line2D([0], [0], color=_MODEL_COLORS['median_offset'], linewidth=2, label='Median Offset'),
        Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=5, label='RT Peak'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.4, linewidth=1, label='1:1'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, -0.02))

    rt_peak_source = rt_align_obj.rt_alignment_params.get('rt_peak_source', 'median')
    rt_model_source = rt_align_obj.rt_alignment_params.get('model_type', 'polynomial')
    fig.suptitle(
        f'All RT Alignment Modes (Current: {rt_model_source} fit to {rt_peak_source})',
        fontsize=13, fontweight='bold',
    )

    if save_plot:
        try:
            plot_save_dir = Path(output_dir)
            plot_save_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = plot_save_dir / "RT_Alignment_All.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"RT_Alignment_All plot saved to {pdf_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error saving RT_Alignment_All plot: {e}")


def build_rt_alignment_model(
    rt_align_obj: "RTAlign"
) -> None:
    """Build RT alignment models from ExperimentalData and Atlas.

    Computes the chromatographic peak RT (RT at maximum intensity) for each
    QC compound in each QC file, then builds three model types (linear,
    polynomial degree-2, median-offset) for:

    * Each individual QC file.
    * Three composites across all QC files: min, median, and max of the
      per-file peak RTs.

    The model selected for atlas alignment is determined by two parameters in
    ``rt_alignment_params``:

    * ``model_type`` (``"polynomial"`` / ``"linear"`` / ``"median_offset"``)
      — which regression type to use.
    * ``rt_peak_source`` (``"min"`` / ``"median"`` / ``"max"`` / *filename*)
      — which set of observed RT peaks to regress against.  Defaults to
      ``"median"``.

    All per-file and composite models are stored on *rt_align_obj* for the
    ``RT_Alignment_All.pdf`` visualisation.

    Args:
        rt_align_obj: RTAlign object with alignment settings.
    """
    logger.info("Building RT alignment model from experimental data and atlas...")
    exclude_inchikeys = set(rt_align_obj.rt_alignment_params.get('exclude_inchikeys', []))
    rt_align_settings = rt_align_obj.rt_alignment_params
    min_obs   = rt_align_settings.get('min_observations_per_compound', 1)
    min_comps = rt_align_settings.get('min_compounds_for_modeling', 2)

    ms1_df = rt_align_obj.experimental_data.ms1_df

    # ------------------------------------------------------------------
    # Step 1: extract per-file peak RTs for every compound × file
    # ------------------------------------------------------------------
    logger.info("Extracting per-file peak RTs (RT at max intensity within atlas window)...")
    per_file_data = _extract_per_file_peak_rts(
        ms1_df=ms1_df,
        compound_mzrts=rt_align_obj.align_atlas_obj.compound_mzrts,
        exclude_inchikeys=exclude_inchikeys,
    )

    if not per_file_data:
        raise ValueError("No compounds with MS1 data found for RT alignment model.")

    # ------------------------------------------------------------------
    # Step 2: build per-file models
    # ------------------------------------------------------------------
    all_filenames: set[str] = set()
    for file_dict in per_file_data.values():
        all_filenames.update(file_dict.keys())
    all_filenames_sorted = sorted(all_filenames)

    logger.info(f"Building models for {len(all_filenames_sorted)} QC file(s)...")
    per_file_models: dict[str, dict] = {}
    per_file_modeling_data: dict[str, pd.DataFrame] = {}

    for fname in tqdm(all_filenames_sorted, desc="Per-file RT models", disable=should_disable_tqdm()):
        df = _per_file_peak_rts_to_modeling_df(per_file_data, fname)
        per_file_modeling_data[fname] = df
        if df.empty or len(df) < min_comps:
            logger.warning(f"  {fname}: only {len(df)} compound(s) — skipping model fit (need ≥{min_comps}).")
            per_file_models[fname] = None
            continue
        models = _build_models_for_df(df, min_obs, min_comps)
        if models is not None:
            for mt, m in models.items():
                X = df['atlas_rt_peak'].values
                y = df['exp_rt_peak'].values
                m['y_pred'] = _apply_rt_model(X, m)
            per_file_models[fname] = models
            logger.info(
                f"  {fname}: {len(df)} compounds | "
                f"poly R²={models['polynomial']['r2']:.4f} | "
                f"linear R²={models['linear']['r2']:.4f} | "
                f"offset R²={models['median_offset']['r2']:.4f}"
            )
        else:
            per_file_models[fname] = None

    # ------------------------------------------------------------------
    # Step 3: build composite models (min / median / max)
    # ------------------------------------------------------------------
    logger.info("Building composite models (min / median / max across QC files)...")
    composite_models: dict[str, dict] = {}
    composite_modeling_data: dict[str, pd.DataFrame] = {}

    for mode in ('min', 'median', 'max'):
        df = _composite_peak_rts(per_file_data, mode)
        composite_modeling_data[mode] = df
        if df.empty or len(df) < min_comps:
            logger.warning(f"  Composite '{mode}': only {len(df)} compound(s) — skipping model fit.")
            composite_models[mode] = None
            continue
        models = _build_models_for_df(df, min_obs, min_comps)
        if models is not None:
            X = df['atlas_rt_peak'].values
            for mt, m in models.items():
                m['y_pred'] = _apply_rt_model(X, m)
            composite_models[mode] = models
            logger.info(
                f"  Composite '{mode}': {len(df)} compounds | "
                f"poly R²={models['polynomial']['r2']:.4f} | "
                f"linear R²={models['linear']['r2']:.4f} | "
                f"offset R²={models['median_offset']['r2']:.4f}"
            )
        else:
            composite_models[mode] = None

    # ------------------------------------------------------------------
    # Step 4: select the active model based on rt_peak_source + model_type
    # ------------------------------------------------------------------
    _VALID_MODEL_TYPES = frozenset({"polynomial", "linear", "median_offset"})
    model_type = rt_align_settings.get('model_type', 'polynomial')
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {sorted(_VALID_MODEL_TYPES)}"
        )

    rt_peak_source = rt_align_settings.get('rt_peak_source', 'median')

    # Resolve the source DataFrame and model dict
    _COMPOSITE_KEYS = frozenset({'min', 'median', 'max'})
    if rt_peak_source in _COMPOSITE_KEYS:
        source_df     = composite_modeling_data.get(rt_peak_source, pd.DataFrame())
        source_models = composite_models.get(rt_peak_source)
        source_label  = f"composite '{rt_peak_source}'"
    else:
        # Treat rt_peak_source as a filename (exact match or basename match)
        matched_fname = None
        for fname in all_filenames_sorted:
            if fname == rt_peak_source or Path(fname).name == rt_peak_source:
                matched_fname = fname
                break
        if matched_fname is None:
            raise ValueError(
                f"rt_peak_source='{rt_peak_source}' does not match any QC filename "
                f"or composite key ('min', 'median', 'max'). "
                f"Available filenames: {all_filenames_sorted}"
            )
        source_df     = per_file_modeling_data.get(matched_fname, pd.DataFrame())
        source_models = per_file_models.get(matched_fname)
        source_label  = f"file '{matched_fname}'"

    if source_models is None or source_df.empty:
        raise ValueError(
            f"No valid model could be built for rt_peak_source='{rt_peak_source}'. "
            f"Check that enough QC compounds have data in that source."
        )

    best_model = source_models[model_type]

    # Attach y_pred and residuals to the modeling DataFrame
    X_active = source_df['atlas_rt_peak'].values
    y_active  = source_df['exp_rt_peak'].values
    modeling_results_df = source_df.copy()
    modeling_results_df['predicted_rt'] = best_model['y_pred']
    modeling_results_df['residual']     = y_active - best_model['y_pred']
    modeling_results_df['abs_residual'] = np.abs(modeling_results_df['residual'])

    degree_str = (
        f"degree {best_model['degree']}"
        if model_type in ('polynomial', 'linear')
        else f"offset={best_model.get('offset', 0.0):.4f} min"
    )
    logger.info(f"Selected model: {model_type} ({degree_str}) from {source_label}")
    logger.info(f"  R² = {best_model['r2']:.4f}")
    logger.info(f"  RMSE = {best_model['rmse']:.4f} min")
    logger.info(f"  Max residual = {modeling_results_df['abs_residual'].max():.4f} min")
    logger.info(f"  Equation: {best_model['equation']}")

    for mt, m in source_models.items():
        if mt != model_type:
            logger.info(f"  [{mt}] R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  Eq: {m['equation']}")

    if best_model['r2'] < rt_align_settings.get('r2_threshold', 0.5):
        logger.warning(
            f"Model R² ({best_model['r2']:.4f}) is below threshold "
            f"({rt_align_settings.get('r2_threshold', 0.5)})"
        )

    logger.info("Compound RT Statistics (active source):")
    logger.info(
        f"\n{modeling_results_df[['compound_name', 'inchi_key', 'adduct', 'atlas_rt_peak', 'exp_rt_peak', 'residual', 'intensity_peak']].to_string()}"
    )

    # Attach all three candidate models from the active source for DB saving
    best_model['compounds_used_for_modeling'] = source_df['compound_uid'].tolist()

    rt_align_obj.rt_alignment_model      = best_model
    rt_align_obj.all_rt_alignment_models = source_models
    rt_align_obj.modeling_data           = modeling_results_df
    rt_align_obj.per_file_models         = per_file_models
    rt_align_obj.composite_models        = composite_models
    rt_align_obj.per_file_modeling_data  = per_file_modeling_data
    rt_align_obj.composite_modeling_data = composite_modeling_data

    return


def calculate_rt_shifts(rt_align_obj: "RTAlign") -> pd.DataFrame:
    """Apply RT shifts to the template atlas compound mzrt uids and save the new bounds to a dataframe."""
    all_rt_shifts = []
    per_compound_rt_shifts = []
    new_compound_mzrts = {}
    for mz_rt_uid, comp_ref in rt_align_obj.aligned_atlas_obj.compound_mzrts.items():
        aligned_rt_peak = float(_apply_rt_model([comp_ref.rt_peak], rt_align_obj.rt_alignment_model)[0])
        if aligned_rt_peak <= 0:
            aligned_rt_peak = 0.01
            #logger.warning(f"Aligned RT peak for {mz_rt_uid} is non-positive, setting to 0.01.")
        if rt_align_obj.rt_alignment_params['apply_model_to_min_max']:
            aligned_rt_min = float(_apply_rt_model([comp_ref.rt_min], rt_align_obj.rt_alignment_model)[0])
            aligned_rt_max = float(_apply_rt_model([comp_ref.rt_max], rt_align_obj.rt_alignment_model)[0])
        else:
            window = comp_ref.rt_max - comp_ref.rt_min
            aligned_rt_min = aligned_rt_peak - window / 2
            aligned_rt_max = aligned_rt_peak + window / 2

        rt_shift = aligned_rt_peak - comp_ref.rt_peak
        all_rt_shifts.append(rt_shift)
        per_compound_rt_shifts.append({
            'compound_name': comp_ref.compound_name,
            'inchi_key': comp_ref.inchi_key,
            'adduct': comp_ref.adduct,
            'mz_rt_uid': mz_rt_uid,
            'original_rt_peak': float(comp_ref.rt_peak),
            'aligned_rt_peak': aligned_rt_peak,
            'rt_shift': rt_shift,
        })
        new_compound_mzrts[mz_rt_uid] = {"rt_peak": aligned_rt_peak, "rt_min": aligned_rt_min, "rt_max": aligned_rt_max}

    atlas_obj = rt_align_obj.aligned_atlas_obj
    rt_shifts_filename_part = (
        f"{(atlas_obj.chromatography or '').upper()}"
        f"-{(atlas_obj.polarity or '').upper()}"
        f"-{(atlas_obj.atlas_name or '')}"
        f"-RTA{rt_align_obj.rt_alignment_number or ''}"
    )
    output_fname = f"RTShifts__{rt_shifts_filename_part}__{rt_align_obj.project_name}.json"
    _save_rt_aligned_stats_to_json(
        all_rt_shifts=all_rt_shifts,
        per_compound_rt_shifts=per_compound_rt_shifts,
        output_file=output_fname,
        output_dir=rt_align_obj.paths['rt_alignment_results_dir'],
    )

    return pd.DataFrame([{'mz_rt_uid': k, **v} for k, v in new_compound_mzrts.items()])

def _apply_rt_model(atlas_rt_values: list | np.ndarray, model_info: dict) -> np.ndarray:
    """Apply an RT alignment model to a list of atlas RT values.

    Dispatches to the correct implementation based on ``model_info['model_type']``:

    * ``"polynomial"`` / ``"linear"`` — uses the stored sklearn
      ``PolynomialFeatures`` + ``LinearRegression`` objects.
    * ``"median_offset"`` — adds the stored scalar offset to each value.

    Args:
        atlas_rt_values: Sequence of atlas RT values to transform.
        model_info:      Model dict returned by one of the ``build_*_model``
                         functions (must contain ``model_type``).

    Returns:
        1-D numpy array of aligned RT values.
    """
    model_type = model_info.get('model_type', 'polynomial')
    X_new = np.array(atlas_rt_values, dtype=float)

    if model_type == 'median_offset':
        offset = float(model_info.get('offset', model_info.get('intercept', 0.0)))
        return X_new + offset

    # polynomial or linear — use sklearn objects
    X_new_poly = model_info['poly_features'].transform(X_new.reshape(-1, 1))
    return model_info['model'].predict(X_new_poly)

def _save_rt_aligned_stats_to_json(
    all_rt_shifts: list[float], 
    per_compound_rt_shifts: dict[str, float],
    output_file: str,
    output_dir: str
) -> None:
    rt_shift_stats = {}
    if all_rt_shifts:
        rt_shift_stats = {
            'rt_shift_min': float(np.min(all_rt_shifts)),
            'rt_shift_max': float(np.max(all_rt_shifts)),
            'rt_shift_median': float(np.median(all_rt_shifts)),
        }

    # Save per-compound RT shift data with summary stats at the top
    rt_shift_output = {
        'stats': rt_shift_stats,
        'compounds': per_compound_rt_shifts,
    }
    rt_shift_stats_path = Path(output_dir) / f"{output_file}"
    rt_shift_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rt_shift_stats_path, "w") as f:
        json.dump(rt_shift_output, f, indent=4)
