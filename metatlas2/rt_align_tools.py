import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

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
        print(f"Plot saved to {pdf_path}")


def create_RT_summary(compounds: pd.DataFrame, best_model: dict, qc_files: list, target_compounds: pd.DataFrame, modeling_data: list, 
                      output_dir: str, rtc_atlas_name: str, save_summary: bool = True):

    corrected_mz_rt_experimental_data = {}
    correction_stats = []

    for _, row in compounds.iterrows():
        entry = row.to_dict()
        if entry['rt_peak'] is not None:
            new_uid = f"mzrt-exp-{uuid.uuid4().hex[:32]}"
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
                "rt_units": "min",
                "mz": entry['mz'],
                "mz_tolerance": entry['mz_tolerance'],
                "mz_tolerance_units": "ppm",
                "adduct": entry['adduct'],
                "charge": 1 if '+' in str(entry['adduct']) else -1 if '-' in str(entry['adduct']) else 0,
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
            print(f"Warning! Original RT peak is None, skipping correction for compound: {entry['compound_name']}")

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

    print(f"RT correction completed: {summary['corrected_compounds']}/{summary['total_compounds']} compounds corrected")
    if summary['corrected_compounds']:
        print(f"Correction statistics: mean = {summary['mean_correction']:.4f}, std = {summary['std_correction']:.4f} min")

    if save_summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        alignment_summary_file = output_dir / f"summary-for-{rtc_atlas_name}.json"
        with open(alignment_summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"RT alignment summary saved to {alignment_summary_file}")

def find_qc_compounds_in_peaks(qc_compounds, ms1_data, default_mz_tol, rt_window_expand):
    print("Matching QC Atlas compounds with QC peak data")

    compound_matches = []
    matching_stats = {
        'total_compounds': len(qc_compounds),
        'compounds_with_matches': 0,
        'compounds_without_matches': 0,
        'total_matches': 0,
        'multiple_matches': 0
    }

    for idx, compound in tqdm(qc_compounds.iterrows(), 
                             total=len(qc_compounds), 
                             desc="Matching QC compounds"):
        compound_name = compound['compound_name']
        target_mz = compound['mz']
        atlas_rt = compound['rt_peak']

        mz_tolerance = compound['mz_tolerance'] if pd.notna(compound['mz_tolerance']) else default_mz_tol

        matching_peaks = find_peaks_in_rt_window(
            ms1_data, 
            target_mz, 
            mz_tolerance, 
            atlas_rt, 
            rt_window_expand
        )

        if len(matching_peaks) > 0:
            matching_stats['compounds_with_matches'] += 1
            matching_stats['total_matches'] += len(matching_peaks)
            if len(matching_peaks) > 1:
                matching_stats['multiple_matches'] += 1

            file_grouped = matching_peaks.groupby('filename')
            best_peaks_per_file = []
            for filename, file_peaks in file_grouped:
                best_peak = file_peaks.loc[file_peaks['i'].idxmax()]
                best_peaks_per_file.append({
                    'compound_uid': compound['compound_uid'],
                    'compound_name': compound_name,
                    'atlas_rt_peak': atlas_rt,
                    'atlas_rt_min': compound['rt_min'],
                    'atlas_rt_max': compound['rt_max'],
                    'atlas_mz': target_mz,
                    'observed_rt': best_peak['rt'],
                    'observed_mz': best_peak['mz'],
                    'observed_intensity': best_peak['i'],
                    'mz_error_ppm': best_peak['mz_error_ppm'],
                    'rt_difference': best_peak['rt_difference'],
                    'filename': filename,
                    'file_path': best_peak['file_path'],
                    'mz_tolerance_used': mz_tolerance
                })
            compound_matches.extend(best_peaks_per_file)
        else:
            matching_stats['compounds_without_matches'] += 1

    if compound_matches:
        matches_df = pd.DataFrame(compound_matches)
        matches_df['inchi_key'] = matches_df['compound_uid'].apply(
            lambda uid: qc_compounds[qc_compounds['compound_uid'] == uid]['inchi_key'].iloc[0] 
            if len(qc_compounds[qc_compounds['compound_uid'] == uid]) > 0 else 'unknown'
        )

        print(f"Compounds with matches: {matching_stats['compounds_with_matches']}")
        print(f"Compounds without matches: {matching_stats['compounds_without_matches']}")
        print(f"Total peak matches: {matching_stats['total_matches']}")
        print(f"Mean m/z error: {matches_df['mz_error_ppm'].mean():.2f} ± {matches_df['mz_error_ppm'].std():.2f} ppm")
        print(f"Mean RT difference: {matches_df['rt_difference'].mean():.3f} ± {matches_df['rt_difference'].std():.3f} min")
        return matches_df, matching_stats
    else:
        print("No QC compound matches found. Check Atlas compound definitions, m/z tolerance, RT window settings, and QC file data quality")
        raise ValueError("No compound matches found")