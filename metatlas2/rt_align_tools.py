import pandas as pd
import numpy as np
import sys
from pathlib import Path
from IPython.display import display

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import extract_data_from_parquet as edp
import ms1_ms2_summarizer as mss

# Initialize logger properly at module level
logger = lcf.get_logger('rt_align_tools')

def apply_rt_alignment_to_target(main_db_path: str, 
                                 target_atlas_uid: str, 
                                 best_model: dict, 
                                 rt_align_settings: dict
) -> Tuple[pd.DataFrame, List[Dict]]:
    
    # Get target atlas and compounds from master database
    target_compounds_df = dbi.get_atlas_compounds_table(main_db_path, target_atlas_uid)
    if target_compounds_df.empty:
        raise ValueError(f"Atlas {target_atlas_uid} not found in master database")
    else:
        logger.info("Successfully loaded target atlas metadata and compounds from database")
    compounds_verified = dbi.verify_compounds_exist_in_db(target_compounds_df['compound_uid'].tolist(), main_db_path)
    logger.debug(f"Compounds verified in database: {compounds_verified}")
    if compounds_verified is False:
        logger.error("One or more compounds in target_compounds_df do not exist in the main database. Aborting RT-aligned atlas creation.")
        return None, []
    else:
        logger.info(f"Loaded target atlas with {len(target_compounds_df)} compounds for RT alignment")

    # Ensure columns are float64 to avoid dtype warnings
    rt_aligned_compounds_df = target_compounds_df.copy()
    for col in ['rt_peak', 'rt_min', 'rt_max', 'rt_shift']:
        if col in rt_aligned_compounds_df.columns:
            rt_aligned_compounds_df[col] = rt_aligned_compounds_df[col].astype('float64')
        else:
            rt_aligned_compounds_df[col] = np.nan
            rt_aligned_compounds_df[col] = rt_aligned_compounds_df[col].astype('float64')
    rt_aligned_compounds_df['rt_shift'] = 0.0

    alignment_stats = []
    for i, row in target_compounds_df.iterrows():
        compound_uid = row['compound_uid']
        original_rt_peak = row.get('rt_peak')
        original_rt_min = row.get('rt_min', None)
        original_rt_max = row.get('rt_max', None)

        if pd.isna(original_rt_peak) or original_rt_peak is None:
            logger.warning(f"Skipping compound {row.get('compound_name', 'Unknown')} - no RT peak data")
            continue

        # Apply RT alignment using the model
        aligned_rt_peak = apply_rt_model([original_rt_peak], best_model)[0]
        if rt_align_settings['apply_model_to_min_max'] and pd.notna(original_rt_min) and pd.notna(original_rt_max):
            aligned_rt_min = apply_rt_model([original_rt_min], best_model)[0]
            aligned_rt_max = apply_rt_model([original_rt_max], best_model)[0]
        else:
            original_window = None
            if pd.notna(original_rt_min) and pd.notna(original_rt_max):
                original_window = original_rt_max - original_rt_min
            else:
                original_window = 1.0  # Default 1-minute window
            aligned_rt_min = aligned_rt_peak - original_window / 2
            aligned_rt_max = aligned_rt_peak + original_window / 2

        rt_shift = aligned_rt_peak - original_rt_peak

        # Update the RT aligned DataFrame
        rt_aligned_compounds_df.loc[i, 'rt_peak'] = float(aligned_rt_peak)
        rt_aligned_compounds_df.loc[i, 'rt_min'] = float(aligned_rt_min)
        rt_aligned_compounds_df.loc[i, 'rt_max'] = float(aligned_rt_max)
        rt_aligned_compounds_df.loc[i, 'rt_shift'] = float(rt_shift)

        # Track alignment statistics
        alignment_stats.append({
            'compound_name': row.get('compound_name'),
            'compound_inchi_key': row.get('inchi_key'),
            'compound_uid': compound_uid,
            'aligned_rt': aligned_rt_peak,
            'rt_shift': rt_shift
        })

    logger.info(f"Returning RT aligned atlas with {len(rt_aligned_compounds_df)} compounds and stats for {len(alignment_stats)} compounds")
    return rt_aligned_compounds_df, alignment_stats

def calculate_model_values_from_existing(model_dict: Dict) -> Dict:
    """
    Reconstruct sklearn model objects from database values and calculate predictions and metrics.
    """
    metadata = model_dict.get('metadata', {})
    
    # Reconstruct PolynomialFeatures
    poly_features = PolynomialFeatures(
        degree=metadata.get('poly_degree', model_dict.get('degree', 1)),
        include_bias=metadata.get('poly_include_bias', True),
        interaction_only=metadata.get('poly_interaction_only', False)
    )
    
    # Fit the PolynomialFeatures with dummy data matching expected input shape
    dummy_X = np.array([[0]]).reshape(-1, 1)
    poly_features.fit(dummy_X)
    
    # Reconstruct LinearRegression model
    model = LinearRegression()
    model.coef_ = np.array(metadata.get('model_coefficients', model_dict.get('coefficients', [])))
    model.intercept_ = metadata.get('model_intercept', model_dict.get('intercept', 0.0))
    
    # Update model_dict with reconstructed objects
    model_dict['poly_features'] = poly_features
    model_dict['model'] = model
    model_dict['intercept'] = model.intercept_
    
    return model_dict

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

def apply_rt_model(atlas_rt_values, model_info):
    """Apply RT alignment model to Atlas RT values."""
    X_new = np.array(atlas_rt_values).reshape(-1, 1)
    X_new_poly = model_info['poly_features'].transform(X_new)
    aligned_rt = model_info['model'].predict(X_new_poly)
    return aligned_rt

def format_polynomial_equation(model_info):
    """Format polynomial equation as string."""
    degree = model_info['degree']
    coeffs = model_info['coefficients']
    intercept = model_info['intercept']
    
    if degree == 1:
        return f"RT_aligned = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas"
    elif degree == 2:
        return f"RT_aligned = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas + {coeffs[2]:.6f} * RT_atlas^2"
    elif degree == 3:
        return f"RT_aligned = {intercept:.6f} + {coeffs[1]:.6f} * RT_atlas + {coeffs[2]:.6f} * RT_atlas^2 + {coeffs[3]:.6f} * RT_atlas³"
    else:
        return f"Polynomial degree {degree} (coefficients: {coeffs})"

def visualize_RT_model(modeling_results_df: pd.DataFrame, best_model: dict, output_dir: str, save_plot: bool = True):

    # Sort by Atlas RT before numbering and plotting
    modeling_results_df = modeling_results_df.sort_values('atlas_rt_peak').reset_index(drop=True)
    modeling_results_df['compound_num'] = modeling_results_df.index + 1

    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Plot 1: Atlas RT vs Observed RT (with model fit)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(modeling_results_df['atlas_rt_peak'], modeling_results_df['exp_rt_median'], 
                alpha=0.7, s=50, c='blue', label='Observed Data')
    ax1.plot(modeling_results_df['atlas_rt_peak'], modeling_results_df['predicted_rt'], 
            'r-', linewidth=2, label='Polynomial Fit (degree 2)')
    ax1.plot([modeling_results_df['atlas_rt_peak'].min(), modeling_results_df['atlas_rt_peak'].max()],
            [modeling_results_df['atlas_rt_peak'].min(), modeling_results_df['atlas_rt_peak'].max()],
            'k--', alpha=0.5, label='Perfect Correlation')
    ax1.set_xlabel('Atlas RT (min)')
    ax1.set_ylabel('Observed RT (min)')
    ax1.set_title(f'RT Correlation (R² = {best_model["r2"]:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add numbers to points
    for _, row in modeling_results_df.iterrows():
        ax1.annotate(str(int(row['compound_num'])), 
                    (row['atlas_rt_peak'], row['exp_rt_median']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Plot 2: Residuals vs Atlas RT
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(modeling_results_df['atlas_rt_peak'], modeling_results_df['residual'], 
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
                    (row['atlas_rt_peak'], row['residual']),
                    textcoords="offset points", xytext=(5, -10), ha='left', fontsize=9, color='black')

    # Table: Compound number to name mapping (ordered by RT), now with residuals
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    table_data = modeling_results_df[['compound_num', 'compound_name', 'inchi_key', 'atlas_rt_peak', 'residual']].copy()
    table_data['atlas_rt_peak'] = table_data['atlas_rt_peak'].round(3)
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

    plt.suptitle('RT Alignment Model Validation', fontsize=16, fontweight='bold', y=1.02)

    if save_plot:
        # Save the plot as PDF
        plot_save_dir = Path(output_dir) / "rt_alignment_results"
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = plot_save_dir / f"summary-for-{best_model['rt_alignment_uid']}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        #plt.show()
        logger.info(f"Plot saved to {pdf_path}")
        plt.close()

    return

def build_rt_alignment_model(experimental_data: Dict, rt_align_settings: Dict) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Build RT alignment model from QC compound matches in experimental_data dict.
    
    Args:
        experimental_data: Dictionary containing compound experimental data (with atlas metadata)
        rt_align_settings: Dictionary containing RT modeling settings
    
    Returns:
        Tuple of (best_model, modeling_results_df, compound_rt_stats)
    """
    logger.info("Building RT alignment model from experimental_data dict...")
    exclude_inchikeys = rt_align_settings.get('exclude_inchikeys', [])

    # Aggregate per-compound statistics
    compound_stats = []
    for compound_uid, compound_data in experimental_data.items():
        inchi_key = compound_data.get('inchi_key')
        compound_name = compound_data.get('compound_name')
        atlas_rt_peak = compound_data.get('rt_peak')
        atlas_rt_min = compound_data.get('rt_min')
        atlas_rt_max = compound_data.get('rt_max')
        atlas_mz = compound_data.get('mz')
        if exclude_inchikeys and inchi_key in exclude_inchikeys:
            logger.info(f"Excluding compound {compound_name} (InChI Key: {inchi_key}) from modeling")
            continue
        observed_rts = []
        observed_mzs = []
        observed_intensities = []
        rt_diffs = []
        mz_errors = []
        if 'eic_files' in compound_data:
            for file_data in compound_data['eic_files'].values():
                if file_data and 'intensity_peak' in file_data:
                    observed_rt = file_data.get('rt_peak')
                    observed_mz = file_data.get('mz_peak')
                    observed_intensity = file_data.get('intensity_peak')
                    ppm_diff = file_data.get('ppm_diff', 0)
                    rt_diff = observed_rt - atlas_rt_peak if observed_rt is not None and atlas_rt_peak is not None else None
                    observed_rts.append(observed_rt)
                    observed_mzs.append(observed_mz)
                    observed_intensities.append(observed_intensity)
                    rt_diffs.append(rt_diff)
                    mz_errors.append(ppm_diff)
        if len(observed_rts) == 0:
            continue
        compound_stats.append({
            'compound_uid': compound_uid,
            'compound_name': compound_name,
            'inchi_key': inchi_key,
            'atlas_rt_peak': atlas_rt_peak,
            'atlas_rt_min': atlas_rt_min,
            'atlas_rt_max': atlas_rt_max,
            'atlas_mz': atlas_mz,
            'exp_rt_mean': np.mean(observed_rts),
            'exp_rt_median': np.median(observed_rts),
            'exp_rt_std': np.std(observed_rts),
            'observation_count': len(observed_rts),
            'exp_mz_mean': np.mean(observed_mzs),
            'exp_mz_std': np.std(observed_mzs),
            'exp_intensity_mean': np.mean(observed_intensities),
            'exp_intensity_median': np.median(observed_intensities),
            'exp_intensity_max': np.max(observed_intensities),
            'rt_diff_mean': np.mean(rt_diffs),
            'rt_diff_median': np.median(rt_diffs),
            'rt_diff_std': np.std(rt_diffs),
            'mz_error_mean': np.mean(mz_errors),
            'mz_error_std': np.std(mz_errors)
        })
    compound_rt_stats = pd.DataFrame(compound_stats)
    if compound_rt_stats.empty:
        raise ValueError("No compounds with matches found for RT alignment model.")

    # Display summary statistics
    logger.info(f"RT Statistics Summary:")
    logger.info(f"  Atlas RT range: {compound_rt_stats['atlas_rt_peak'].min():.2f} - {compound_rt_stats['atlas_rt_peak'].max():.2f} min")
    logger.info(f"  Observed RT range (median): {compound_rt_stats['exp_rt_median'].min():.2f} - {compound_rt_stats['exp_rt_median'].max():.2f} min")
    logger.info(f"  Mean RT difference (observed - atlas): {compound_rt_stats['rt_diff_median'].mean():.3f} ± {compound_rt_stats['rt_diff_median'].std():.3f} min")

    # Filter compounds with sufficient observations for reliable modeling
    reliable_compounds = compound_rt_stats[
        compound_rt_stats['observation_count'] >= rt_align_settings['min_observations_per_compound']
    ]
    logger.info(f"Using {len(reliable_compounds)} compounds with ≥{rt_align_settings['min_observations_per_compound']} observations (QC files) for modeling")
    if len(reliable_compounds) < rt_align_settings['min_compounds_for_modeling']:
        raise ValueError(f"Insufficient compounds for modeling. Need at least {rt_align_settings['min_compounds_for_modeling']}, but found {len(reliable_compounds)}")

    # Extract X (Atlas RT) and y (observed median RT) for modeling
    X_atlas_rt = reliable_compounds['atlas_rt_peak'].values
    y_observed_rt = reliable_compounds['exp_rt_median'].values

    # Create modeling dataset
    modeling_results_df = reliable_compounds.copy()

    # Build polynomial model
    logger.info(f"Building polynomial model (degree {rt_align_settings['polynomial_degree']})...")
    best_model = build_polynomial_model(X_atlas_rt, y_observed_rt, rt_align_settings['polynomial_degree'])

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
    if best_model['r2'] < rt_align_settings['r2_threshold']:
        logger.warning(f"Model R² ({best_model['r2']:.4f}) is below threshold ({rt_align_settings['r2_threshold']})")

    # Add compounds used for modeling
    best_model['compounds_used_for_modeling'] = reliable_compounds['compound_uid'].tolist()

    # Display compound statistics table
    logger.info(f"Compound RT Statistics:")
    display(compound_rt_stats[['compound_name', 'inchi_key', 'atlas_rt_peak', 'exp_rt_median', 'rt_diff_median', 
                            'observation_count', 'exp_rt_std']])

    return best_model, modeling_results_df, compound_rt_stats

def extract_matches_from_qc_files(main_db_path: str,
                                  qc_atlas_uid: str,
                                  qc_files_df: pd.DataFrame,
                                  rt_align_settings: Dict) -> Dict: 
    """
    Use same approach as feature extraction to extract EIC data for QC compounds.
    Adds atlas metadata to each compound in experimental_data by inchi_key.
    """

    logger.info("Loading QC atlas...")
    atlas_dataframe = dbi.get_atlas_compounds_table(main_db_path, atlas_uid=qc_atlas_uid)
    if not atlas_dataframe.empty:
        logger.info(f"Preparing {len(atlas_dataframe)} compounds for RT alignment modeling...")
    else:
        raise ValueError(f"No compounds found in QC atlas")
    logger.info(f"Created Atlas dataframe with {len(atlas_dataframe)} compounds")

    logger.info("Preparing parquet file list...")
    project_qc_files = qc_files_df['file_path'].tolist()
    existing_parquet_files = [f for f in project_qc_files if Path(f).exists()]
    if len(existing_parquet_files) == 0:
        raise FileNotFoundError("No parquet files found for QC files")
    logger.info(f"Found {len(existing_parquet_files)} parquet files")

    logger.info("Extracting EIC data from QC parquet files...")
    parquet_results = edp.extract_eic_and_ms2_from_parquet(
        atlas_df=atlas_dataframe,
        parquet_files=existing_parquet_files,
        ppm_tolerance=rt_align_settings['ppm_error'],
        extra_time=rt_align_settings['extra_time'],
        use_parallel=True,
        only_ms_level=1,
    )
    parquet_results_with_summary = mss.create_ms_summaries(parquet_results, 
                                                           only_ms_level=1)

    # for inchi_key, file_data in parquet_results.items():
    #     for file, ms_level_data in file_data.items():
    #         for ms_level, df in ms_level_data.items():
    #             if not df.empty:
    #                 display(df.head(2))

    logger.info("Formatting results for RT alignment...")
    experimental_data = _format_parquet_results_for_rt_alignment(
        parquet_results_with_summary, 
        atlas_dataframe
    )

    return experimental_data


def _format_parquet_results_for_rt_alignment(
    parquet_results: Dict[str, Dict],
    atlas_dataframe: pd.DataFrame
) -> Dict:
    """
    Format parquet extraction results to match the structure expected by RT alignment.
    
    Args:
        parquet_results: Output from extract_eic_and_ms2_from_parquet
        atlas_dataframe: Atlas DataFrame with compound metadata
    
    Returns:
        Dict matching experimental_data structure with eic_files and atlas metadata
    """
    experimental_data = {}
    
    for _, atlas_row in atlas_dataframe.iterrows():
        inchi_key = atlas_row.get('inchi_key')
        
        if inchi_key not in parquet_results:
            continue
        
        # Initialize compound entry with atlas metadata
        compound_data = atlas_row.to_dict()
        compound_data['eic_files'] = {}
        
        # Process each file's data for this compound
        for parquet_file, file_data in parquet_results[inchi_key].items():
            ms1_summary = file_data.get('ms1_summary', pd.DataFrame())
            
            if ms1_summary.empty or ms1_summary['peak_height'].iloc[0] == 0:
                # No valid peak found in this file
                continue
            
            # Extract peak properties
            summary_row = ms1_summary.iloc[0]
            observed_mz = summary_row['mz_centroid']
            observed_rt = summary_row['rt_peak']
            observed_intensity = summary_row['peak_height']
            
            # Calculate PPM error
            atlas_mz = atlas_row['mz']
            ppm_diff = ((observed_mz - atlas_mz) / atlas_mz) * 1e6 if atlas_mz > 0 else 0
            
            # Store in eic_files format expected by RT alignment
            compound_data['eic_files'][parquet_file] = {
                'rt_peak': observed_rt,
                'mz_peak': observed_mz,
                'intensity_peak': observed_intensity,
                'ppm_diff': ppm_diff,
                'peak_area': summary_row['peak_area'],
                'num_datapoints': summary_row['num_datapoints']
            }
        
        # Only add compounds that have at least one match
        if compound_data['eic_files']:
            experimental_data[inchi_key] = compound_data
    
    logger.info(f"Formatted data for {len(experimental_data)} compounds with matches")
    
    return experimental_data

def evaluate_qc_matching_stats(experimental_data: Dict) -> Dict:
    """
    Evaluate QC compound matching statistics from experimental_data.
    
    Args:
        experimental_data: Dictionary containing compound experimental data from feature extraction
    
    Returns:
        Dictionary with comprehensive matching statistics
    """
    logger.info("Evaluating QC compound matching statistics...")
    
    total_compounds = len(experimental_data)
    compounds_with_matches = 0
    compounds_without_matches = 0
    total_peaks_extracted = 0
    file_match_counts = {}
    
    for compound_uid, compound_data in experimental_data.items():
        has_matches = False
        compound_peaks = 0
        compound_files_with_matches = set()
        
        # Check EIC files for matches
        if 'eic_files' in compound_data and compound_data['eic_files']:
            for file_path, file_data in compound_data['eic_files'].items():
                # Each file entry represents one peak/match for this compound
                if file_data and 'intensity_peak' in file_data:
                    has_matches = True
                    compound_peaks += 1  # Each file entry is one peak
                    compound_files_with_matches.add(file_path)
                    
                    # Track file-level statistics
                    if file_path not in file_match_counts:
                        file_match_counts[file_path] = {'compounds_matched': 0, 'total_peaks': 0}
                    file_match_counts[file_path]['compounds_matched'] += 1
                    file_match_counts[file_path]['total_peaks'] += 1
        
        # Update compound-level statistics
        if has_matches:
            compounds_with_matches += 1
        else:
            compounds_without_matches += 1
        
        total_peaks_extracted += compound_peaks
    
    # Calculate file-level statistics
    total_files_analyzed = len(file_match_counts)
    total_files_with_matches = sum(1 for stats in file_match_counts.values() if stats['compounds_matched'] > 0)
    total_files_without_matches = total_files_analyzed - total_files_with_matches
    
    # Calculate averages and percentages
    match_percentage = (compounds_with_matches / total_compounds * 100) if total_compounds > 0 else 0
    avg_peaks_per_compound = (total_peaks_extracted / compounds_with_matches) if compounds_with_matches > 0 else 0
    avg_compounds_per_file = sum(stats['compounds_matched'] for stats in file_match_counts.values()) / total_files_analyzed if total_files_analyzed > 0 else 0
    
    # Compile comprehensive statistics
    stats = {
        'total_compounds': total_compounds,
        'compounds_with_matches': compounds_with_matches,
        'compounds_without_matches': compounds_without_matches,
        'match_percentage': round(match_percentage, 2),
        'total_files_analyzed': total_files_analyzed,
        'total_files_with_matches': total_files_with_matches,
        'total_files_without_matches': total_files_without_matches,
        'total_peaks_extracted': total_peaks_extracted,
        'avg_peaks_per_compound_with_matches': round(avg_peaks_per_compound, 2),
        'avg_compounds_matched_per_file': round(avg_compounds_per_file, 2),
        'file_match_statistics': file_match_counts
    }
    
    # Log summary statistics
    logger.info(f"QC Compound Matching Summary:")
    logger.info(f"  Total compounds: {total_compounds}")
    logger.info(f"  Compounds with matches: {compounds_with_matches} ({match_percentage:.1f}%)")
    logger.info(f"  Compounds without matches: {compounds_without_matches}")
    logger.info(f"  Total files analyzed: {total_files_analyzed}")
    logger.info(f"  Files with matches: {total_files_with_matches}")
    logger.info(f"  Total peaks extracted: {total_peaks_extracted}")
    logger.info(f"  Average peaks per matched compound: {avg_peaks_per_compound:.1f}")
    logger.info(f"  Average compounds matched per file: {avg_compounds_per_file:.1f}")
    
    return stats


def create_rt_alignment_summary(
    rtc_atlas_uid: str,
    rt_alignment_uid: str,
    rtc_atlas_name: str,
    alignment_stats: List[Dict],
) -> Dict:
    """
    Create RT alignment summary dictionary.

    Args:
        rtc_atlas_name: Name of RT-aligned atlas
        alignment_stats: List of alignment statistics

    Returns:
        Dictionary with alignment summary
    """
    rt_shifts = [stat['rt_shift'] for stat in alignment_stats]
    summary = {
        'rt_alignment_uid': rt_alignment_uid,
        'rtc_atlas_name': rtc_atlas_name,
        'rtc_atlas_uid': rtc_atlas_uid,
        'total_compounds': len(alignment_stats),
        'aligned_compounds': len(alignment_stats),
        'mean_alignment': np.mean(rt_shifts),
        'std_alignment': np.std(rt_shifts),
        'min_alignment': np.min(rt_shifts),
        'max_alignment': np.max(rt_shifts)
    }
    
    return summary