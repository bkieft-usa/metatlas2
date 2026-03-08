import pandas as pd
import numpy as np
import sys
from pathlib import Path
from IPython.display import display
from typing import Dict, Tuple, List

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('rt_align_tools')

def apply_rt_alignment_to_target_atlases(
        rt_align_obj: "RTAlign"
) -> Tuple[Dict[str, "Atlas"], Dict[str, float]]:
    from workflow_objects import Atlas, CompoundMZRT

    logger.info("Applying RT alignment model to target atlases and generating RT-aligned Atlas objects...")

    main_db_path = rt_align_obj.paths['main_db_path']
    targeted_analyses = rt_align_obj.config['WORKFLOWS']['TARGETED_ANALYSES']
    rt_alignment_model = rt_align_obj.rt_alignment_model
    rt_align_settings = rt_align_obj.rt_alignment_params
    rt_alignment_number = rt_align_obj.rt_alignment_number

    aligned_atlases = {}
    all_rt_shifts = []
    for chrom, pol_dict in targeted_analyses.items():
        for pol, analysis_dict in pol_dict.items():
            for analysis_type, atlas_params_dict in analysis_dict.items():
                target_atlas_uid = atlas_params_dict.get('ATLAS', {}).get('uid', None)
                if target_atlas_uid is None:
                    logger.debug(f"Skipping {chrom} {pol} {analysis_type} - no target atlas UID found in parameters")
                    continue

                logger.info(f"Loading {chrom} {pol} {analysis_type} target atlas with UID {target_atlas_uid} for applying RT alignment model...")
                atlas_obj = Atlas.from_database(main_db_path, target_atlas_uid)

                # Create a new Atlas object for the RT-aligned version
                aligned_compound_mzrts = {}
                for inchi_key, comp_ref in atlas_obj.compound_mzrts.items():
                    # Apply RT alignment model
                    aligned_rt_peak = float(_apply_rt_model([comp_ref.rt_peak], rt_alignment_model)[0])
                    if rt_align_settings['apply_model_to_min_max']:
                        aligned_rt_min = float(_apply_rt_model([comp_ref.rt_min], rt_alignment_model)[0])
                        aligned_rt_max = float(_apply_rt_model([comp_ref.rt_max], rt_alignment_model)[0])
                    else:
                        window = comp_ref.rt_max - comp_ref.rt_min
                        aligned_rt_min = aligned_rt_peak - window / 2
                        aligned_rt_max = aligned_rt_peak + window / 2
                    rt_shift = aligned_rt_peak - comp_ref.rt_peak
                    all_rt_shifts.append(rt_shift)

                    # Create a new CompoundMZRT with updated RTs
                    mz_rt_uid = dbi._generate_uid("mz_rt", decorator="exp")
                    comp_dict = {k: v for k, v in comp_ref.__dict__.items() if k not in ['mz_rt_uid', 'rt_peak', 'rt_min', 'rt_max']}
                    aligned_comp_mzrt = CompoundMZRT(
                        **comp_dict,
                        mz_rt_uid=mz_rt_uid,
                        rt_peak=aligned_rt_peak,
                        rt_min=aligned_rt_min,
                        rt_max=aligned_rt_max,
                    )
                    aligned_compound_mzrts[inchi_key] = aligned_comp_mzrt

                # Generate new UID and name for the aligned atlas
                aligned_atlas_uid = dbi._generate_uid("rt_atlas", decorator=f"{analysis_type.lower()}-{chrom.lower()}-{pol.lower()}")
                aligned_atlas = Atlas(
                    atlas_uid=aligned_atlas_uid,
                    atlas_name=f"{atlas_obj.atlas_name} (post-rt-alignment)",
                    atlas_description=f"{atlas_obj.atlas_description} (post-rt-alignment)",
                    chromatography=chrom,
                    polarity=pol,
                    analysis_type=analysis_type,
                    atlas_type="RT-ALIGNED",
                    source_atlas_uid=atlas_obj.atlas_uid,
                    rt_alignment_number=rt_alignment_number,
                    analysis_number=None,
                    created_by=atlas_obj.created_by,
                    created_date=atlas_obj.created_date,
                    source=atlas_obj.source,
                    compound_mzrts=aligned_compound_mzrts
                )
                aligned_atlases[aligned_atlas_uid] = aligned_atlas

    # Calculate RT shift stats
    rt_shift_stats = {}
    if all_rt_shifts:
        rt_shift_stats = {
            'rt_shift_min': float(np.min(all_rt_shifts)),
            'rt_shift_max': float(np.max(all_rt_shifts)),
            'rt_shift_median': float(np.median(all_rt_shifts)),
        }

    logger.info(f"Applied RT alignment model to {len(aligned_atlases)} target atlases. RT shift stats: {rt_shift_stats}")
    rt_align_obj.rt_shift_stats = rt_shift_stats
    rt_align_obj.rt_aligned_atlases = aligned_atlases
    
    return

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

def _apply_rt_model(atlas_rt_values, model_info):
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

def visualize_rt_alignment_model(rt_align_obj: "RTAlign", save_plot: bool = True):
    """
    Visualize RT alignment model results using RTAlign object.
    """

    logger.info("Plotting RT alignment model results to figure...")

    modeling_results_df = rt_align_obj.modeling_data
    rt_alignment_model = rt_align_obj.rt_alignment_model
    output_dir = rt_align_obj.paths['rt_alignment_output_dir']

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
    ax1.set_title(f'RT Correlation (R² = {rt_alignment_model["r2"]:.4f})')
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
    ax2.set_title(f'Residuals vs Atlas RT (RMSE = {rt_alignment_model["rmse"]:.4f})')
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
    table_data = modeling_results_df[['compound_num', 'inchi_key', 'adduct', 'atlas_rt_peak', 'residual']].copy()
    table_data['atlas_rt_peak'] = table_data['atlas_rt_peak'].round(3)
    table_data['residual'] = table_data['residual'].round(4)
    table_data.columns = ['#', 'InChi Key', 'Adduct', 'Atlas RT (min)', 'Residual (min)']
    table = ax_table.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        loc='center',
                        cellLoc='left',
                        colLoc='left')

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
        try:
            plot_save_dir = Path(output_dir) / "rt_alignment_results"
            plot_save_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = plot_save_dir / f"summary-for-{rt_alignment_model['rt_alignment_uid']}.pdf"
            plt.savefig(pdf_path, bbox_inches='tight')
            logger.info(f"Plot saved to {pdf_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Error saving plot: {e}")
        finally:
            logger.info(f"RT alignment model plot saved to {pdf_path}")

    return

def build_rt_alignment_model(
    experimental_data: "ExperimentalData",
    atlas: "Atlas",
    rt_align: "RTAlign"
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Build RT alignment model directly from ExperimentalData and Atlas.
    Args:
        experimental_data: ExperimentalData object with extracted MS1 data
        atlas: Atlas object with compound references
        rt_align: RTAlign object with alignment settings
    Returns:
        Tuple of (rt_alignment_model, modeling_results_df, compound_rt_stats)
    """
    logger.info("Building RT alignment model from experimental data and atlas...")
    exclude_inchikeys = rt_align.rt_alignment_params.get('exclude_inchikeys', [])

    # Build a lookup for MS1Data by (inchi_key, adduct)
    ms1_lookup = {}
    for ms1 in experimental_data.ms1_data:
        key = (ms1.inchi_key, ms1.adduct)
        ms1_lookup.setdefault(key, []).append(ms1)

    compound_stats = []
    for compound_mzrt in atlas.compound_mzrts.values():
        inchi_key = compound_mzrt.inchi_key
        adduct = compound_mzrt.adduct
        compound_uid = compound_mzrt.compound_uid
        atlas_rt_peak = compound_mzrt.rt_peak
        atlas_rt_min = compound_mzrt.rt_min
        atlas_rt_max = compound_mzrt.rt_max
        atlas_mz = compound_mzrt.mz

        if exclude_inchikeys and inchi_key in exclude_inchikeys:
            continue

        ms1_list = ms1_lookup.get((inchi_key, adduct), [])
        if not ms1_list:
            continue

        observed_rts = []
        observed_mzs = []
        observed_intensities = []
        rt_diffs = []
        mz_errors = []

        for ms1 in ms1_list:
            ms1_data = ms1.data
            if ms1_data.empty:
                continue
            sum_intensity = ms1_data['i'].sum()
            if sum_intensity > 0:
                idx = ms1_data['i'].idxmax()
                observed_rt = float(ms1_data.loc[idx, 'rt'])
                observed_mz = float((ms1_data['i'] * ms1_data['mz']).sum() / sum_intensity)
                observed_intensity = float(ms1_data.loc[idx, 'i'])
                ppm_diff = ((observed_mz - atlas_mz) / atlas_mz) * 1e6 if atlas_mz > 0 else 0
                rt_diff = observed_rt - atlas_rt_peak if observed_rt is not None and atlas_rt_peak is not None else None
                observed_rts.append(observed_rt)
                observed_mzs.append(observed_mz)
                observed_intensities.append(observed_intensity)
                rt_diffs.append(rt_diff)
                mz_errors.append(ppm_diff)

        if not observed_rts:
            continue

        compound_stats.append({
            'compound_uid': compound_uid,
            'inchi_key': inchi_key,
            'adduct': adduct,
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

    logger.info(f"RT Statistics Summary:")
    logger.info(f"  Atlas RT range: {compound_rt_stats['atlas_rt_peak'].min():.2f} - {compound_rt_stats['atlas_rt_peak'].max():.2f} min")
    logger.info(f"  Observed RT range (median): {compound_rt_stats['exp_rt_median'].min():.2f} - {compound_rt_stats['exp_rt_median'].max():.2f} min")
    logger.info(f"  Mean RT difference (observed - atlas): {compound_rt_stats['rt_diff_median'].mean():.3f} ± {compound_rt_stats['rt_diff_median'].std():.3f} min")

    rt_align_settings = rt_align.rt_alignment_params
    reliable_compounds = compound_rt_stats[
        compound_rt_stats['observation_count'] >= rt_align_settings['min_observations_per_compound']
    ]
    logger.info(f"Using {len(reliable_compounds)} compounds with ≥{rt_align_settings['min_observations_per_compound']} observations (QC files) for modeling")
    if len(reliable_compounds) < rt_align_settings['min_compounds_for_modeling']:
        raise ValueError(f"Insufficient compounds for modeling. Need at least {rt_align_settings['min_compounds_for_modeling']}, but found {len(reliable_compounds)}")

    X_atlas_rt = reliable_compounds['atlas_rt_peak'].values
    y_observed_rt = reliable_compounds['exp_rt_median'].values

    modeling_results_df = reliable_compounds.copy()

    logger.info(f"Building polynomial model (degree {rt_align_settings['polynomial_degree']})...")
    best_model = build_polynomial_model(X_atlas_rt, y_observed_rt, rt_align_settings['polynomial_degree'])

    modeling_results_df['predicted_rt'] = best_model['y_pred']
    modeling_results_df['residual'] = y_observed_rt - best_model['y_pred']
    modeling_results_df['abs_residual'] = np.abs(modeling_results_df['residual'])

    logger.info(f"Model built successfully:")
    logger.info(f"  Model type: Polynomial degree {best_model['degree']}")
    logger.info(f"  R² = {best_model['r2']:.4f}")
    logger.info(f"  RMSE = {best_model['rmse']:.4f} min")
    logger.info(f"  Max residual = {modeling_results_df['abs_residual'].max():.4f} min")

    equation = format_polynomial_equation(best_model)
    logger.info(f"  Equation: {equation}")
    best_model['equation'] = equation

    if best_model['r2'] < rt_align_settings['r2_threshold']:
        logger.warning(f"Model R² ({best_model['r2']:.4f}) is below threshold ({rt_align_settings['r2_threshold']})")

    best_model['compounds_used_for_modeling'] = reliable_compounds['compound_uid'].tolist()

    logger.info(f"Compound RT Statistics:")
    display(compound_rt_stats[['inchi_key', 'adduct', 'atlas_rt_peak', 'exp_rt_median', 'rt_diff_median', 
                               'observation_count', 'exp_rt_std']])

    rt_align.rt_alignment_model = best_model
    rt_align.modeling_data = modeling_results_df

    return

def create_file_matching_summary(
    experimental_data: "ExperimentalData",
    atlas: "Atlas"
) -> None:
    """
    Evaluate QC compound matching statistics directly from ExperimentalData and Atlas.

    Args:
        experimental_data: ExperimentalData object with extracted MS1 data
        atlas: Atlas object with compound references

    Returns:
        None (logs statistics)
    """
    logger.info("Evaluating QC compound matching statistics from ExperimentalData...")

    total_compounds = 0
    compounds_with_matches = 0
    compounds_without_matches = 0
    total_peaks_extracted = 0
    file_match_counts = {}

    # Build a lookup for MS1Data by (inchi_key, adduct)
    ms1_lookup = {}
    for ms1 in experimental_data.ms1_data:
        key = (ms1.inchi_key, ms1.adduct)
        ms1_lookup.setdefault(key, []).append(ms1)

    for compound_mzrt in atlas.compound_mzrts.values():
        inchi_key = compound_mzrt.inchi_key
        adduct = compound_mzrt.adduct
        total_compounds += 1
        has_matches = False
        compound_peaks = 0

        ms1_list = ms1_lookup.get((inchi_key, adduct), [])
        if not ms1_list:
            compounds_without_matches += 1
            continue

        for ms1 in ms1_list:
            ms1_data = ms1.data
            if ms1_data.empty:
                continue

            sum_intensity = ms1_data['i'].sum()
            if float(sum_intensity) > 0:
                has_matches = True
                compound_peaks += 1
                # Track file-level statistics
                file_path = ms1.filename
                if file_path not in file_match_counts:
                    file_match_counts[file_path] = {'compounds_matched': 0, 'total_peaks': 0}
                file_match_counts[file_path]['compounds_matched'] += 1
                file_match_counts[file_path]['total_peaks'] += 1

        if has_matches:
            compounds_with_matches += 1
        else:
            compounds_without_matches += 1

        total_peaks_extracted += compound_peaks

    total_files_analyzed = len(file_match_counts)
    total_files_with_matches = sum(1 for stats in file_match_counts.values() if stats['compounds_matched'] > 0)
    total_files_without_matches = total_files_analyzed - total_files_with_matches

    match_percentage = (compounds_with_matches / total_compounds * 100) if total_compounds > 0 else 0
    avg_peaks_per_compound = (total_peaks_extracted / compounds_with_matches) if compounds_with_matches > 0 else 0
    avg_compounds_per_file = sum(stats['compounds_matched'] for stats in file_match_counts.values()) / total_files_analyzed if total_files_analyzed > 0 else 0

    logger.info(f"QC Compound Matching Summary:")
    logger.info(f"  Total compounds: {total_compounds}")
    logger.info(f"  Compounds with matches: {compounds_with_matches} ({match_percentage:.1f}%)")
    logger.info(f"  Compounds without matches: {compounds_without_matches}")
    logger.info(f"  Total files analyzed: {total_files_analyzed}")
    logger.info(f"  Files with matches: {total_files_with_matches}")
    logger.info(f"  Total peaks extracted: {total_peaks_extracted}")
    logger.info(f"  Average peaks per matched compound: {avg_peaks_per_compound:.1f}")
    logger.info(f"  Average compounds matched per file: {avg_compounds_per_file:.1f}")

    return

def display_rt_alignment_summary(rt_align_obj: "RTAlign") -> None:
    """
    Log a concise summary of the RT alignment model and RT shift statistics using RTAlign object.
    """

    logger.info("Generating RT alignment model summary...")

    model = rt_align_obj.rt_alignment_model
    stats = getattr(rt_align_obj, "rt_shift_stats", None)

    if model is None:
        logger.info("No RT alignment model available to summarize.")
        return

    r2 = model.get('r2', None)
    rmse = model.get('rmse', None)
    degree = model.get('degree', None)
    equation = model.get('equation', None)
    compounds = model.get('compounds_used_for_modeling', [])
    n_compounds = len(compounds) if compounds is not None else 0

    logger.info("RT Alignment Model Summary:")
    logger.info(f"  Polynomial degree: {degree}")
    logger.info(f"  R²: {r2:.4f}" if r2 is not None else "  R²: N/A")
    logger.info(f"  RMSE: {rmse:.4f} min" if rmse is not None else "  RMSE: N/A")
    logger.info(f"  Equation: {equation}" if equation else "  Equation: N/A")
    logger.info(f"  Compounds used for modeling: {n_compounds}")

    if stats:
        logger.info("RT Shift Statistics (across all aligned compounds):")
        logger.info(f"  Min RT shift: {stats.get('rt_shift_min', 'N/A'):.4f} min")
        logger.info(f"  Max RT shift: {stats.get('rt_shift_max', 'N/A'):.4f} min")
        logger.info(f"  Median RT shift: {stats.get('rt_shift_median', 'N/A'):.4f} min")