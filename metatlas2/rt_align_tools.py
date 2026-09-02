from __future__ import annotations

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from tqdm.auto import tqdm
import json

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

import metatlas2.logging_config as lcf
from metatlas2.utils import should_disable_tqdm
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

def visualize_rt_alignment_model(rt_align_obj: "RTAlign", save_plot: bool = True):
    """
    Visualize RT alignment model results using RTAlign object.
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
    ax1.scatter(modeling_results_df['atlas_rt_peak'], modeling_results_df['exp_rt_median'],
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
                    (row['atlas_rt_peak'], row['exp_rt_median']),
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
            pdf_path = plot_save_dir / f"summary-for-{rt_alignment_model['rt_alignment_uid']}.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"Plot saved to {pdf_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        finally:
            if pdf_path is not None:
                logger.info(f"RT alignment model plot saved to {pdf_path}")

    return

def build_rt_alignment_model(
    rt_align_obj: "RTAlign"
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Build RT alignment model directly from ExperimentalData and Atlas.
    Args:
        rt_align_obj: RTAlign object with alignment settings
    Returns:
        Tuple of (rt_alignment_model, modeling_results_df, compound_rt_stats)
    """
    logger.info("Building RT alignment model from experimental data and atlas...")
    exclude_inchikeys = set(rt_align_obj.rt_alignment_params.get('exclude_inchikeys', []))

    ms1_df = rt_align_obj.experimental_data.ms1_df
    ms1_grouped = ms1_df.groupby('mz_rt_uid', sort=False)

    compound_stats = []
    for compound_mzrt in tqdm(
        rt_align_obj.align_atlas_obj.compound_mzrts.values(),
        desc="Building RT alignment model",
        disable=should_disable_tqdm(),
    ):
        mz_rt_uid   = getattr(compound_mzrt, 'mz_rt_uid', None)
        inchi_key   = compound_mzrt.inchi_key
        adduct      = compound_mzrt.adduct
        compound_uid  = compound_mzrt.compound_uid
        compound_name = compound_mzrt.compound_name
        atlas_rt_peak = compound_mzrt.rt_peak
        atlas_rt_min  = compound_mzrt.rt_min
        atlas_rt_max  = compound_mzrt.rt_max
        atlas_mz      = compound_mzrt.mz

        if exclude_inchikeys and inchi_key in exclude_inchikeys:
            continue

        try:
            ms1_df_comp = ms1_grouped.get_group(mz_rt_uid)
        except KeyError:
            continue  # no MS1 data for this compound

        rts_parts, mzs_parts, ints_parts = [], [], []
        for row_rts, row_ints, row_mzs, row_mask in zip(
            ms1_df_comp['spec_rts'],
            ms1_df_comp['spec_ints'],
            ms1_df_comp['spec_mzs'],
            ms1_df_comp['in_feature'],
        ):
            try:
                arr_rts  = np.asarray(row_rts,  dtype=np.float64)
                arr_ints = np.asarray(row_ints, dtype=np.float64)
                arr_mzs  = np.asarray(row_mzs,  dtype=np.float64)
                arr_mask = np.asarray(row_mask, dtype=bool)
            except (TypeError, ValueError):
                continue
            if arr_mask.size == 0 or not arr_mask.any():
                continue
            # Align mask length to data length (guard against ragged lists).
            min_len = min(arr_rts.size, arr_ints.size, arr_mzs.size, arr_mask.size)
            if min_len == 0:
                continue
            arr_mask = arr_mask[:min_len]
            rts_parts.append(arr_rts[:min_len][arr_mask])
            ints_parts.append(arr_ints[:min_len][arr_mask])
            mzs_parts.append(arr_mzs[:min_len][arr_mask])

        if not rts_parts:
            continue

        observed_rts  = np.concatenate(rts_parts)
        observed_ints = np.concatenate(ints_parts)
        observed_mzs  = np.concatenate(mzs_parts)

        if observed_rts.size == 0:
            continue

        rt_diffs  = observed_rts - atlas_rt_peak
        mz_errors = ((observed_mzs - atlas_mz) / atlas_mz * 1e6) if atlas_mz > 0 else np.zeros_like(observed_mzs)

        compound_stats.append({
            'compound_uid':        compound_uid,
            'mz_rt_uid':           mz_rt_uid,
            'compound_name':       compound_name,
            'inchi_key':           inchi_key,
            'adduct':              adduct,
            'atlas_rt_peak':       atlas_rt_peak,
            'atlas_rt_min':        atlas_rt_min,
            'atlas_rt_max':        atlas_rt_max,
            'atlas_mz':            atlas_mz,
            'exp_rt_mean':         float(np.mean(observed_rts)),
            'exp_rt_median':       float(np.median(observed_rts)),
            'exp_rt_std':          float(np.std(observed_rts)),
            'observation_count':   int(observed_rts.size),
            'exp_mz_mean':         float(np.mean(observed_mzs)),
            'exp_mz_std':          float(np.std(observed_mzs)),
            'exp_intensity_mean':  float(np.mean(observed_ints)),
            'exp_intensity_median':float(np.median(observed_ints)),
            'exp_intensity_max':   float(np.max(observed_ints)),
            'rt_diff_mean':        float(np.mean(rt_diffs)),
            'rt_diff_median':      float(np.median(rt_diffs)),
            'rt_diff_std':         float(np.std(rt_diffs)),
            'mz_error_mean':       float(np.mean(mz_errors)),
            'mz_error_std':        float(np.std(mz_errors)),
        })

    compound_rt_stats = pd.DataFrame(compound_stats)
    if compound_rt_stats.empty:
        raise ValueError("No compounds with matches found for RT alignment model.")

    logger.info(f"RT Statistics Summary:")
    logger.info(f"  Atlas RT range: {compound_rt_stats['atlas_rt_peak'].min():.2f} - {compound_rt_stats['atlas_rt_peak'].max():.2f} min")
    logger.info(f"  Observed RT range (median): {compound_rt_stats['exp_rt_median'].min():.2f} - {compound_rt_stats['exp_rt_median'].max():.2f} min")
    logger.info(f"  Mean RT difference (observed - atlas): {compound_rt_stats['rt_diff_median'].mean():.3f} ± {compound_rt_stats['rt_diff_median'].std():.3f} min")

    rt_align_settings = rt_align_obj.rt_alignment_params
    reliable_compounds = compound_rt_stats[
        compound_rt_stats['observation_count'] >= rt_align_settings['min_observations_per_compound']
    ]
    logger.info(f"Using {len(reliable_compounds)} compounds with ≥{rt_align_settings['min_observations_per_compound']} observations (QC files) for modeling")
    if len(reliable_compounds) < rt_align_settings['min_compounds_for_modeling']:
        raise ValueError(f"Insufficient compounds for modeling. Need at least {rt_align_settings['min_compounds_for_modeling']}, but found {len(reliable_compounds)}")

    X_atlas_rt = reliable_compounds['atlas_rt_peak'].values
    y_observed_rt = reliable_compounds['exp_rt_median'].values

    modeling_results_df = reliable_compounds.copy()

    # Select and build the requested model type
    _VALID_MODEL_TYPES = frozenset({"polynomial", "linear", "median_offset"})
    model_type = rt_align_settings.get('model_type', 'polynomial')
    if model_type not in _VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Valid options: {sorted(_VALID_MODEL_TYPES)}"
        )

    if model_type == 'median_offset':
        logger.info("Building median-offset model...")
        best_model = build_median_offset_model(X_atlas_rt, y_observed_rt)
    elif model_type == 'linear':
        logger.info("Building linear regression model...")
        best_model = build_linear_model(X_atlas_rt, y_observed_rt)
    else:  # polynomial (default)
        degree = rt_align_settings.get('polynomial_degree', 2)
        logger.info(f"Building polynomial model (degree {degree})...")
        best_model = build_polynomial_model(X_atlas_rt, y_observed_rt, degree)

    modeling_results_df['predicted_rt'] = best_model['y_pred']
    modeling_results_df['residual'] = y_observed_rt - best_model['y_pred']
    modeling_results_df['abs_residual'] = np.abs(modeling_results_df['residual'])

    degree_str = (
        f"degree {best_model['degree']}"
        if model_type in ('polynomial', 'linear')
        else f"offset={best_model.get('offset', 0.0):.4f} min"
    )
    logger.info(f"Model built successfully:")
    logger.info(f"  Model type: {model_type} ({degree_str})")
    logger.info(f"  R² = {best_model['r2']:.4f}")
    logger.info(f"  RMSE = {best_model['rmse']:.4f} min")
    logger.info(f"  Max residual = {modeling_results_df['abs_residual'].max():.4f} min")

    equation = format_model_equation(best_model)
    logger.info(f"  Equation: {equation}")
    best_model['equation'] = equation

    if best_model['r2'] < rt_align_settings['r2_threshold']:
        logger.warning(f"Model R² ({best_model['r2']:.4f}) is below threshold ({rt_align_settings['r2_threshold']})")

    best_model['compounds_used_for_modeling'] = reliable_compounds['compound_uid'].tolist()

    logger.info(f"Compound RT Statistics:")
    logger.info(f"\n{compound_rt_stats[['compound_name', 'inchi_key', 'adduct', 'atlas_rt_peak', 'exp_rt_median', 'rt_diff_median', 'observation_count', 'exp_rt_std']].to_string()}")

    rt_align_obj.rt_alignment_model = best_model
    rt_align_obj.modeling_data = modeling_results_df

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

    output_fname = f"rt_shifts_for_{rt_align_obj.aligned_atlas_obj.atlas_uid}.json" 
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