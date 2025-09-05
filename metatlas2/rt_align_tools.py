import pandas as pd
import numpy as np
import sys
import duckdb
import os
import json
from tqdm.notebook import tqdm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, Tuple

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import load_tools as ldt
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('rt_align_tools')

def build_polynomial_model(X, y, degree):
    """Build polynomial regression model."""
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Calculate predictions and metrics
    y_pred = model.predict(X_poly)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return {
        'model': model,
        'poly_features': poly_features,
        'degree': degree,
        'r2': r2,
        'rmse': rmse,
        'y_pred': y_pred,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }

def predict_rt_correction(atlas_rt_values, model_info):
    """Apply RT correction model to Atlas RT values."""
    X_new = np.array(atlas_rt_values).reshape(-1, 1)
    X_new_poly = model_info['poly_features'].transform(X_new)
    corrected_rt = model_info['model'].predict(X_new_poly)
    return corrected_rt

def format_polynomial_equation(model_info):
    """Format polynomial equation as string."""
    degree = model_info['degree']
    coeffs = model_info['coefficients']
    intercept = model_info['intercept']
    
    if degree == 1:
        return f"RT_corrected = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas"
    elif degree == 2:
        return f"RT_corrected = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas + {coeffs[2]:.6f} * RT_atlas^2"
    elif degree == 3:
        return f"RT_corrected = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas + {coeffs[2]:.6f} * RT_atlas^2 + {coeffs[3]:.6f} * RT_atlas³"
    else:
        return f"Polynomial degree {degree} (coefficients: {coeffs})"

def visualize_RT_model(modeling_results_df: pd.DataFrame, best_model: dict, output_dir: str, rtc_atlas_name: str, save_plot: bool = True):

    # Sort by Atlas RT before numbering and plotting
    modeling_results_df = modeling_results_df.sort_values('ref_rt_peak').reset_index(drop=True)
    modeling_results_df['compound_num'] = modeling_results_df.index + 1

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Plot 1: Atlas RT vs Observed RT (with model fit)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(modeling_results_df['ref_rt_peak'], modeling_results_df['exp_rt_median'], 
                alpha=0.7, s=50, c='blue', label='Observed Data')
    ax1.plot(modeling_results_df['ref_rt_peak'], modeling_results_df['predicted_rt'], 
            'r-', linewidth=2, label='Polynomial Fit (degree 2)')
    ax1.plot([modeling_results_df['ref_rt_peak'].min(), modeling_results_df['ref_rt_peak'].max()],
            [modeling_results_df['ref_rt_peak'].min(), modeling_results_df['ref_rt_peak'].max()],
            'k--', alpha=0.5, label='Perfect Correlation')
    ax1.set_xlabel('Atlas RT (min)')
    ax1.set_ylabel('Observed RT (min)')
    ax1.set_title(f'RT Correlation (R² = {best_model["r2"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add numbers to points
    for _, row in modeling_results_df.iterrows():
        ax1.annotate(str(int(row['compound_num'])), 
                    (row['ref_rt_peak'], row['exp_rt_median']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Plot 2: Residuals vs Atlas RT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(modeling_results_df['ref_rt_peak'], modeling_results_df['residual'], 
                alpha=0.7, s=50, c='green')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=modeling_results_df['residual'].std(), color='orange', linestyle=':', alpha=0.7, label='+1 σ')
    ax2.axhline(y=-modeling_results_df['residual'].std(), color='orange', linestyle=':', alpha=0.7, label='-1 σ')
    ax2.set_xlabel('Atlas RT (min)')
    ax2.set_ylabel('Residual (min)')
    ax2.set_title(f'Residuals vs Atlas RT (RMSE = {best_model["rmse"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add numbers to points
    for _, row in modeling_results_df.iterrows():
        ax2.annotate(str(int(row['compound_num'])), 
                    (row['ref_rt_peak'], row['residual']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Table: Compound number to name mapping (ordered by RT), now with residuals
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    table_data = modeling_results_df[['compound_num', 'compound_name', 'inchi_key', 'ref_rt_peak', 'residual']].copy()
    table_data['ref_rt_peak'] = table_data['ref_rt_peak'].round(3)
    table_data['residual'] = table_data['residual'].round(4)
    table_data.columns = ['#', 'Compound Name', 'InChi Key', 'Atlas RT (min)', 'Residual (min)']
    table = ax_table.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        loc='center',
                        cellLoc='left',
                        colLoc='left')

    # Set column widths as requested: 0.1, 0.3, 0.3, 0.15, 0.15
    col_widths = [0.2, 0.5, 0.5, 0.3, 0.3]
    for i, width in enumerate(col_widths):
        table.auto_set_column_width(i)
        for j in range(len(table_data) + 1):  # +1 for header
            cell = table[(j, i)]
            cell.set_width(width)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    plt.suptitle('RT Correction Model Validation', fontsize=16, fontweight='bold', y=1.02)

    if save_plot:
        # Save the plot as PDF
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / f"summary-for-{rtc_atlas_name}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.show()
        logger.info(f"Plot saved to {pdf_path}")


def create_RT_summary(compounds: pd.DataFrame, best_model: dict, qc_files: list, target_compounds: pd.DataFrame, modeling_data: list, 
                      output_dir: str, rtc_atlas_name: str, save_summary: bool = True):

    corrected_mz_rt_experimental_data = {}
    correction_stats = []

    for _, row in compounds.iterrows():
        entry = row.to_dict()
        if entry['rt_peak'] is not None:
            new_uid = dbi._generate_uid("mz_rt_experimental")
            corrected_rt = predict_rt_correction([entry['rt_peak']], best_model)[0]
            window = (entry['rt_max'] - entry['rt_min']) if entry['rt_min'] is not None and entry['rt_max'] is not None else None
            corrected_min = corrected_rt - window / 2 if window is not None else None
            corrected_max = corrected_rt + window / 2 if window is not None else None
            rt_shift = corrected_rt - entry['rt_peak']
            corrected_mz_rt_experimental_data[new_uid] = {
                "mz_rt_experimental_uid": new_uid,
                "compound_uid": entry['compound_uid'],
                "rt_peak": corrected_rt,
                "rt_min": corrected_min,
                "rt_max": corrected_max,
                "mz": entry['mz'],
                "mz_tolerance": entry['mz_tolerance'],
                "adduct": entry['adduct'],
                "last_modified": TIMESTAMP,
                "updated_from_ref": True,
                "data_source": "rt_correction",
                "source_mz_rt_reference_uid": entry['mz_rt_reference_uid'],
                "rt_correction_metadata": {
                    "correction_applied": True,
                    "rt_shift": rt_shift,
                    "model_degree": best_model['degree'],
                    "model_r2": best_model['r2'],
                    "model_rmse": best_model['rmse'],
                    "correction_timestamp": TIMESTAMP,
                    "correction_method": "polynomial_qc_based",
                    "source_qc_atlas": os.path.basename(QC_ATLAS_FILE_PATH),
                    "qc_files_count": len(qc_files),
                    "qc_compounds_used_for_modeling": len(modeling_data),
                    "model_equation": format_polynomial_equation(best_model)
                }
            }
            correction_stats.append({
                'compound_name': entry['compound_name'],
                'compound_uid': entry['compound_uid'],
                'mz_rt_reference_uid': entry['mz_rt_reference_uid'],
                'mz_rt_experimental_uid': new_uid,
                'original_rt': entry['rt_peak'],
                'corrected_rt': corrected_rt,
                'rt_shift': rt_shift
            })
        else:
            logger.warning(f"Original RT peak is None, skipping correction for compound: {entry['compound_name']}")

    correction_df = pd.DataFrame(correction_stats)
    summary = {
        'total_compounds': len(target_compounds),
        'corrected_compounds': len(correction_stats),
        'uncorrected_compounds': len(target_compounds) - len(correction_stats),
        'correction_stats': correction_stats,
        'mean_correction': correction_df['rt_shift'].mean() if not correction_df.empty else 0,
        'std_correction': correction_df['rt_shift'].std() if not correction_df.empty else 0,
        'min_correction': correction_df['rt_shift'].min() if not correction_df.empty else 0,
        'max_correction': correction_df['rt_shift'].max() if not correction_df.empty else 0
    }

    logger.info(f"RT correction completed: {summary['corrected_compounds']}/{summary['total_compounds']} compounds corrected")
    if summary['corrected_compounds']:
        logger.info(f"Correction statistics: mean = {summary['mean_correction']:.4f}, std = {summary['std_correction']:.4f} min")

    if save_summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        alignment_summary_file = output_dir / f"summary-for-{rtc_atlas_name}.json"
        with open(alignment_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"RT alignment summary saved to {alignment_summary_file}")

def build_rt_alignment_model(matches_df: pd.DataFrame, config: Dict) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Build RT alignment model from QC compound matches.
    
    Args:
        matches_df: DataFrame of QC compound matches
        config: Dictionary containing RT modeling settings
    
    Returns:
        Tuple of (best_model, modeling_results_df, compound_rt_stats)
    """
    logger.info("Building RT alignment model...")

    rt_settings = config['rt_alignment']['model']
    # Filter by excluded InChI keys if specified
    if rt_settings.get('exclude_inchikeys'):
        if 'inchi_key' not in matches_df.columns:
            raise ValueError("matches_df must contain an 'inchi_key' column to filter by InChI Key.")
        before_count = len(matches_df)
        matches_df = matches_df[~matches_df['inchi_key'].isin(rt_settings['exclude_inchikeys'])]
        after_count = len(matches_df)
        logger.info(f"Filtered out {before_count - after_count} matches by InChI Key.")
    
    # Aggregate matches to calculate median/mean observed RT values for each compound
    compound_rt_stats = matches_df.groupby(['compound_uid', 'compound_name', 'inchi_key']).agg({
        'atlas_rt_peak': 'first',
        'atlas_rt_min': 'first',
        'atlas_rt_max': 'first',
        'atlas_mz': 'first',
        'observed_rt': ['mean', 'median', 'std', 'count'],
        'observed_mz': ['mean', 'std'],
        'observed_intensity': ['mean', 'median', 'max'],
        'rt_difference': ['mean', 'median', 'std'],
        'mz_error_ppm': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    compound_rt_stats.columns = [
        'ref_rt_peak',           # Atlas RT peak (reference)
        'ref_rt_min',            # Atlas RT min
        'ref_rt_max',            # Atlas RT max  
        'ref_mz',                # Atlas m/z
        'exp_rt_mean',           # Mean observed RT across files
        'exp_rt_median',         # Median observed RT across files (more robust)
        'exp_rt_std',            # Standard deviation of observed RT
        'observation_count',     # Number of files where compound was observed
        'exp_mz_mean',           # Mean observed m/z
        'exp_mz_std',            # Std dev of observed m/z
        'exp_intensity_mean',    # Mean intensity
        'exp_intensity_median',  # Median intensity  
        'exp_intensity_max',     # Max intensity
        'rt_diff_mean',          # Mean RT difference (observed - atlas)
        'rt_diff_median',        # Median RT difference
        'rt_diff_std',           # Std dev of RT difference
        'mz_error_mean',         # Mean m/z error (ppm)
        'mz_error_std'           # Std dev of m/z error (ppm)
    ]
    
    # Reset index to get compound info as columns
    compound_rt_stats = compound_rt_stats.reset_index()
    
    # Display summary statistics
    logger.info(f"RT Statistics Summary:")
    logger.info(f"  Atlas RT range: {compound_rt_stats['ref_rt_peak'].min():.2f} - {compound_rt_stats['ref_rt_peak'].max():.2f} min")
    logger.info(f"  Observed RT range (median): {compound_rt_stats['exp_rt_median'].min():.2f} - {compound_rt_stats['exp_rt_median'].max():.2f} min")
    logger.info(f"  Mean RT difference (observed - atlas): {compound_rt_stats['rt_diff_median'].mean():.3f} ± {compound_rt_stats['rt_diff_median'].std():.3f} min")
    
    # Filter compounds with sufficient observations for reliable modeling
    reliable_compounds = compound_rt_stats[
        compound_rt_stats['observation_count'] >= rt_settings['min_observations_per_compound']
    ]
    logger.info(f"Using {len(reliable_compounds)} compounds with ≥{rt_settings['min_observations_per_compound']} observations (QC files) for modeling")
    
    if len(reliable_compounds) < rt_settings['min_compounds_for_modeling']:
        raise ValueError(f"Insufficient compounds for modeling. Need at least {rt_settings['min_compounds_for_modeling']}, but found {len(reliable_compounds)}")
    
    # Extract X (Atlas RT) and y (observed median RT) for modeling
    X_atlas_rt = reliable_compounds['ref_rt_peak'].values
    y_observed_rt = reliable_compounds['exp_rt_median'].values
    
    # Create modeling dataset
    modeling_results_df = reliable_compounds.copy()
    
    # Build polynomial model
    logger.info(f"Building polynomial model (degree {rt_settings['polynomial_degree']})...")
    best_model = build_polynomial_model(X_atlas_rt, y_observed_rt, rt_settings['polynomial_degree'])
    
    # Add predictions and residuals to modeling results
    modeling_results_df['predicted_rt'] = best_model['y_pred']
    modeling_results_df['residual'] = y_observed_rt - best_model['y_pred']
    modeling_results_df['abs_residual'] = np.abs(modeling_results_df['residual'])
    
    # Model evaluation
    logger.info(f"Model built successfully:")
    logger.info(f"  Model type: Polynomial degree {best_model['degree']}")
    logger.info(f"  R² = {best_model['r2']:.4f}")
    logger.info(f"  RMSE = {best_model['rmse']:.4f} min")
    logger.info(f"  Max residual = {modeling_results_df['abs_residual'].max():.4f} min")
    
    # Display polynomial equation
    equation = format_polynomial_equation(best_model)
    logger.info(f"  Equation: {equation}")
    best_model['equation'] = equation
    
    # Check model quality
    if best_model['r2'] < rt_settings['r2_threshold']:
        logger.warning(f"Model R² ({best_model['r2']:.4f}) is below threshold ({rt_settings['r2_threshold']})")
    
    # Add compounds used for modeling
    best_model['compounds_used_for_modeling'] = reliable_compounds['compound_uid'].tolist()
    
    # Display compound statistics table
    logger.info(f"Compound RT Statistics:")
    display(compound_rt_stats[['compound_name', 'inchi_key', 'ref_rt_peak', 'exp_rt_median', 'rt_diff_median', 
                            'observation_count', 'exp_rt_std']])

    return best_model, modeling_results_df, compound_rt_stats


def apply_rt_correction_to_target(
    project_db_path,
    target_atlas_uid,
    config,
    best_model,
    lcmsrun_files,
    modeling_results_df,
):
    """
    Clone target atlas to project database and apply RT correction.
    Uses database_interact functions for all database operations.
    """
    logger.info("Cloning target atlas and applying RT correction...")

    database_path = config['paths']['main_database']

    # Get target atlas and compounds from master database
    target_atlas_df = dbi.get_atlas_from_db(database_path, target_atlas_uid)
    target_compounds_df = dbi.get_atlas_compounds_table(database_path, target_atlas_uid)
    
    if target_atlas_df.empty:
        raise ValueError(f"Atlas {target_atlas_uid} not found in master database")
    
    atlas_info = target_atlas_df.iloc[0]

    # Create RT-corrected atlas using database function
    corrected_atlas_uid, correction_stats = dbi.create_rt_corrected_atlas(
        project_db_path=project_db_path,
        source_atlas_uid=target_atlas_uid,
        atlas_info=atlas_info,
        best_model=best_model,
        target_compounds_df=target_compounds_df
    )
    
    # Save RT alignment model to database
    rt_alignment_uid = dbi.save_rt_alignment_model_to_db(
        corrected_atlas_uid,
        project_db_path,
        best_model,
        [f for files in lcmsrun_files.values() for pol_files in files.values() for f in pol_files.get('qc', [])],
        modeling_results_df.to_dict('records')
    )
    
    # Create and save summary using database function
    summary = dbi.create_rt_alignment_summary(
        rtc_atlas_name=f"{atlas_info['atlas_name']}-RT-Corrected",
        correction_stats=correction_stats,
        total_compounds=len(target_compounds_df)
    )
    
    # Add additional summary fields
    summary.update({
        'rt_alignment_uid': rt_alignment_uid,
        'corrected_atlas_uid': corrected_atlas_uid,
        'corrected_atlas_name': f"{atlas_info['atlas_name']} (RT Corrected)"
    })

    logger.info(f"RT-corrected atlas created:")
    logger.info(f"  Atlas UID: {corrected_atlas_uid}")
    logger.info(f"  Atlas name: {summary['corrected_atlas_name']}")
    logger.info(f"  RT alignment model UID: {rt_alignment_uid}")
    logger.info(f"  Project database: {project_db_path}")

    logger.info(f"  Correction summary:")
    for key, value in summary.items():
        if key not in ['correction_stats']:  # Skip the detailed stats for brevity
            logger.info(f"    {key}: {value}")

    return corrected_atlas_uid