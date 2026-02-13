import sys
import copy
from datetime import datetime
from pathlib import Path
import getpass
import copy
import pandas as pd
import numpy as np
import json
import pickle
import re
import os
import glob
from typing import Dict, List, Optional, Any, Tuple
from tqdm.notebook import tqdm
import time
import duckdb
import uuid
import glob
import ast
import threading

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from IPython.display import display, HTML
import ipywidgets as widgets
from ipywidgets import Output
from scipy.signal import find_peaks, peak_widths, peak_prominences
from scipy.ndimage import gaussian_filter1d

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import workflow_objects as wfo
import logging_config as lcf
import load_tools as ldt

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_gui')


def create_gui_from_database(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> Tuple:
    """
    Create GUI by loading compound data directly from project database.
    Only loads metadata initially - full data loaded on-demand per compound.
    """
    with dbi.get_db_connection(project_db_path) as conn:
        # Load compound list with key metadata only
        compounds_meta = conn.execute("""
            SELECT 
                compound_uid,
                compound_name,
                inchi_key,
                chromatography,
                polarity,
                rt_alignment_number,
                analysis_number
            FROM mz_rt_experimental
            WHERE rt_alignment_number = ?
            AND analysis_number = ?
            ORDER BY compound_name
        """, [rt_alignment_number, analysis_number]).df()
    
    if compounds_meta.empty:
        logger.error("No compounds found in database")
        return None
    
    logger.info(f"Found {len(compounds_meta)} compounds for curation")
    
    # Pass metadata to GUI - full compound loaded on-demand
    return create_gui_with_database_backend(
        compounds_meta=compounds_meta,
        project_db_path=project_db_path,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
    )

def update_compound_curation(
    project_db_path: str,
    compound_uid: str,
    rt_alignment_number: int,
    analysis_number: int,
    updates: Dict[str, Any]
) -> None:
    """
    Update compound curation data in database in real-time.
    
    Args:
        project_db_path: Path to project database
        compound_uid: Unique identifier for the compound
        rt_alignment_number: RT alignment number
        analysis_number: Analysis number
        updates: Dict with any of:
            - post_rt_peak, post_rt_min, post_rt_max
            - is_rt_modified
            - ms1_notes, ms2_notes
            - analyst_notes, identification_notes
            - is_annotation_modified
    """
    if not updates:
        return
    
    prov = ldt.get_provenance()
    
    with dbi.get_db_connection(project_db_path) as conn:
        # Build dynamic UPDATE statement
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        
        # Always update timestamp
        set_clauses.append('analysis_timestamp = ?')
        params.append(prov["timestamp"])
        
        # Add WHERE clause params
        params.extend([compound_uid, rt_alignment_number, analysis_number])
        
        conn.execute(f"""
            UPDATE mz_rt_experimental
            SET {', '.join(set_clauses)}
            WHERE compound_uid = ?
            AND rt_alignment_number = ?
            AND analysis_number = ?
        """, params)
    
    logger.debug(f"Updated curation for compound {compound_uid}: {list(updates.keys())}")

def export_curation_results(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    only_modified: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export curation results from database.
    
    Returns:
        Tuple of (all_compounds_df, modified_compounds_df)
    """
    with get_db_connection(project_db_path) as conn:
        query = """
            SELECT 
                compound_name,
                inchi_key,
                compound_uid,
                chromatography,
                polarity,
                pre_rt_peak,
                pre_rt_min,
                pre_rt_max,
                post_rt_peak,
                post_rt_min,
                post_rt_max,
                is_rt_modified,
                ms1_notes,
                ms2_notes,
                analyst_notes,
                identification_notes,
                is_annotation_modified,
                analyst,
                analysis_timestamp
            FROM mz_rt_experimental
            WHERE rt_alignment_number = ?
            AND analysis_number = ?
            ORDER BY compound_name
        """
        all_df = conn.execute(query, [rt_alignment_number, analysis_number]).df()
    
    # Calculate RT deltas
    all_df['rt_peak_delta'] = all_df['post_rt_peak'] - all_df['pre_rt_peak']
    
    # Filter to only modified compounds
    if only_modified:
        modified_df = all_df[all_df['is_rt_modified'] | all_df['is_annotation_modified']].copy()
    else:
        modified_df = all_df.copy()
    
    logger.info(f"Exported {len(all_df)} total compounds, {len(modified_df)} modified")
    return all_df, modified_df

def load_full_compound_from_db(
    project_db_path: str,
    compound_uid: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> Optional[wfo.CompoundExperimental]:
    """
    Load full compound data from database on-demand for display in GUI.
    This includes EIC data, MS2 data, and all metadata.
    """
    with dbi.get_db_connection(project_db_path) as conn:
        # Get compound experimental data
        exp_data = conn.execute("""
            SELECT *
            FROM mz_rt_experimental
            WHERE compound_uid = ?
            AND rt_alignment_number = ?
            AND analysis_number = ?
        """, [compound_uid, rt_alignment_number, analysis_number]).fetchone()
        
        if not exp_data:
            logger.error(f"Compound {compound_uid} not found in database")
            return None
        
        # Convert to dict for easier access
        columns = [desc[0] for desc in conn.description]
        exp_dict = dict(zip(columns, exp_data))
    
    # Build CompoundExperimental object
    compound = wfo.CompoundExperimental(
        compound_name=exp_dict['compound_name'],
        compound_uid=exp_dict['compound_uid'],
        inchi_key=exp_dict['inchi_key'],
        mz=exp_dict['mz'],
        mz_tolerance=exp_dict['mz_tolerance'],
        adduct=exp_dict['adduct'],
        chromatography=exp_dict['chromatography'],
        polarity=exp_dict['polarity'],
        atlas_rt_peak=exp_dict['pre_rt_peak'],
        atlas_rt_min=exp_dict['pre_rt_min'],
        atlas_rt_max=exp_dict['pre_rt_max'],
        rt_peak=exp_dict['post_rt_peak'] if exp_dict['post_rt_peak'] else exp_dict['pre_rt_peak'],
        rt_min=exp_dict['post_rt_min'] if exp_dict['post_rt_min'] else exp_dict['pre_rt_min'],
        rt_max=exp_dict['post_rt_max'] if exp_dict['post_rt_max'] else exp_dict['pre_rt_max']
    )
    
    # Load annotations from database
    compound.ms1_notes = exp_dict.get('ms1_notes', 'keep')
    compound.ms2_notes = exp_dict.get('ms2_notes', 'no selection')
    compound.analyst_notes = exp_dict.get('analyst_notes', '')
    compound.identification_notes = exp_dict.get('identification_notes', '')
    compound.is_rt_modified = exp_dict.get('is_rt_modified', False)
    compound.is_annotation_modified = exp_dict.get('is_annotation_modified', False)
    
    # Load EIC and MS2 data from parquet files
    eic_data_path = exp_dict.get('eic_data_path')
    if eic_data_path and os.path.exists(eic_data_path):
        eic_df = pd.read_parquet(eic_data_path)
        compound.eic_data_files = _reconstruct_eic_data(eic_df)
    
    ms2_data_path = exp_dict.get('ms2_data_path')
    if ms2_data_path and os.path.exists(ms2_data_path):
        ms2_df = pd.read_parquet(ms2_data_path)
        compound.ms2_data_files = _reconstruct_ms2_data(ms2_df)
    
    # Load suggested RT bounds if available
    suggested_rt_min = exp_dict.get('suggested_rt_min')
    suggested_rt_max = exp_dict.get('suggested_rt_max')
    suggested_rt_peak = exp_dict.get('suggested_rt_peak')
    if all(v is not None for v in [suggested_rt_min, suggested_rt_max, suggested_rt_peak]):
        compound.suggested_rt_bounds = {
            'rt_min': suggested_rt_min,
            'rt_max': suggested_rt_max,
            'rt_peak': suggested_rt_peak
        }
    
    # Load isomers if available
    isomers_json = exp_dict.get('isomers')
    if isomers_json:
        try:
            compound.isomers = json.loads(isomers_json)
        except json.JSONDecodeError:
            compound.isomers = []
    
    logger.debug(f"Loaded full compound data for {compound.compound_name}")
    return compound

def _reconstruct_eic_data(eic_df: pd.DataFrame) -> Dict:
    """Reconstruct EIC data dictionary from parquet DataFrame."""
    eic_data = {}
    for file_name in eic_df['file_name'].unique():
        file_data = eic_df[eic_df['file_name'] == file_name]
        eic_data[file_name] = {
            'rt_vals': file_data['rt'].values,
            'i_vals': file_data['intensity'].values
        }
    return eic_data

def _reconstruct_ms2_data(ms2_df: pd.DataFrame) -> Dict:
    """Reconstruct MS2 data dictionary from parquet DataFrame."""
    ms2_data = {}
    for file_name in ms2_df['file_name'].unique():
        file_data = ms2_df[ms2_df['file_name'] == file_name].iloc[0]
        
        ms2_entry = {}
        
        # Reconstruct best_hit if present
        if pd.notna(file_data.get('best_hit_database')):
            ms2_entry['best_hit'] = {
                'database': file_data['best_hit_database'],
                'score': file_data['best_hit_score'],
                'num_matches': file_data['best_hit_num_matches'],
                'rt_measured': file_data['best_hit_rt_measured'],
                'mz_measured': file_data['best_hit_mz_measured'],
                'mz_theoretical': file_data['best_hit_mz_theoretical'],
                'qry_spectrum': (
                    json.loads(file_data['best_hit_qry_spectrum_mz']),
                    json.loads(file_data['best_hit_qry_spectrum_intensity'])
                ),
                'ref_spectrum': (
                    json.loads(file_data['best_hit_ref_spectrum_mz']),
                    json.loads(file_data['best_hit_ref_spectrum_intensity'])
                ),
                'qry_frag_colors': json.loads(file_data['best_hit_qry_frag_colors'])
            }
        
        # Reconstruct best_ms2 if present
        if pd.notna(file_data.get('best_ms2_precursor_mz')):
            ms2_entry['best_ms2'] = {
                'precursor_mz': file_data['best_ms2_precursor_mz'],
                'rt': file_data['best_ms2_rt'],
                'intensity_peak': file_data['best_ms2_intensity_peak'],
                'spectrum': (
                    json.loads(file_data['best_ms2_spectrum_mz']),
                    json.loads(file_data['best_ms2_spectrum_intensity'])
                )
            }
        
        ms2_data[file_name] = ms2_entry
    
    return ms2_data

def create_gui_with_database_backend(
    compounds_meta: pd.DataFrame,
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
):
    """
    Create enhanced RT editor with direct database connection.
    Loads full compound data on-demand and saves changes to database in real-time.
    """
    
    # Store database connection info
    db_info = {
        'project_db_path': project_db_path,
        'rt_alignment_number': rt_alignment_number,
        'analysis_number': analysis_number,
    }
    
    # Set file colors
    file_color_dict = {"istd": "green", "injblk": "red", "qc": "blue", "refstd": "black"}
    
    # Create compound list from metadata
    compound_list = list(range(len(compounds_meta)))
    all_compound_names = compounds_meta['compound_name'].tolist()
    all_compound_uids = compounds_meta['compound_uid'].tolist()
    
    # Cache for loaded compounds (only keep current compound in memory)
    current_compound_cache = [None]
    
    def get_current_compound_func() -> Optional[wfo.CompoundExperimental]:
        """Load current compound from database on-demand."""
        idx = current_compound_index[0]
        if not (0 <= idx < len(compound_list)):
            return None
        
        # Check if already cached
        if current_compound_cache[0] is not None:
            return current_compound_cache[0]
        
        # Load full compound data from database
        compound_uid = all_compound_uids[idx]
        compound = load_full_compound_from_db(
            project_db_path=db_info['project_db_path'],
            compound_uid=compound_uid,
            rt_alignment_number=db_info['rt_alignment_number'],
            analysis_number=db_info['analysis_number'],
        )
        
        # Cache it
        current_compound_cache[0] = compound
        return compound
    
    def save_compound_to_db(compound: wfo.CompoundExperimental):
        """Save compound changes to database immediately."""
        if compound is None:
            return
        
        updates = {}
        
        # Check for RT modifications
        if compound.is_rt_modified:
            updates['post_rt_peak'] = compound.rt_peak
            updates['post_rt_min'] = compound.rt_min
            updates['post_rt_max'] = compound.rt_max
            updates['is_rt_modified'] = True
        
        # Check for annotation modifications
        if compound.is_annotation_modified:
            updates['ms1_notes'] = compound.ms1_notes
            updates['ms2_notes'] = compound.ms2_notes
            updates['analyst_notes'] = compound.analyst_notes
            updates['identification_notes'] = compound.identification_notes
            updates['is_annotation_modified'] = True
        
        # Save to database if there are updates
        if updates:
            update_compound_curation(
                project_db_path=db_info['project_db_path'],
                compound_uid=compound.compound_uid,
                rt_alignment_number=db_info['rt_alignment_number'],
                analysis_number=db_info['analysis_number'],
                updates=updates
            )
            logger.debug(f"Saved changes to database for {compound.compound_name}")

    # Set starting status
    current_compound_index = [0]
    current_ms2_file_index = [0]
    updating = [False]
    rt_slider = [None]
    
    # Define annotation options (same as before)
    ms2_options = ['no selection',
                   '-1.0, poor match, should remove',
                   '0.0, no match or no MSMS collected',
                   '0.5, partial or putative match of fragments',
                   '1.0, good match',
                   '0.5, co-isolated precursor, partial match',
                   '1.0, co-isolated precursor, good match',
                   '0.5, single ion match, no evidence',
                   '1.0, single ion match, ISTD/ref evidence']
    
    ms1_options = ['keep', 
                   'remove', 
                   'unresolvable isomers', 
                   'poor peak shape']

    # Create dropdown options
    dropdown_options = []
    for i, compound_name in enumerate(all_compound_names):
        dropdown_options.append(f"{i+1}: {compound_name}")

    # Create main widgets (same as your original code)
    plot_output = widgets.Output(layout=widgets.Layout(width='70%', height='600px'))
    prev_button = widgets.Button(description="◀ Prev", button_style='primary', layout=widgets.Layout(width='80px'))
    next_button = widgets.Button(description="Next ▶", button_style='primary', layout=widgets.Layout(width='80px'))
    counter_label = widgets.HTML(value=f"Compound 1 of {len(compound_list)}", layout=widgets.Layout(width='150px'))
    
    compound_dropdown = widgets.Dropdown(
        options=dropdown_options,
        value=dropdown_options[0],
        description='Jump to:',
        layout=widgets.Layout(width='400px'),
        style={'description_width': '60px'}
    )

    ms2_prev_button = widgets.Button(description="◀ Prev MS2", button_style='info', layout=widgets.Layout(width='100px'))
    ms2_next_button = widgets.Button(description="Next MS2 ▶", button_style='info', layout=widgets.Layout(width='100px'))
    ms2_counter_label = widgets.HTML(value="", layout=widgets.Layout(width='150px'))
    
    reset_button = widgets.Button(description="Reset to Original", button_style='warning', layout=widgets.Layout(width='200px'))
    accept_suggestions_button = widgets.Button(description="Accept Suggestions", button_style='success', layout=widgets.Layout(width='200px'))

    ms2_radio = widgets.RadioButtons(
        options=ms2_options,
        value=ms2_options[0],
        description='MS2 Notes:',
        layout=widgets.Layout(width='100%', height='240px'),
        style={'description_width': '80px'}
    )
    
    ms1_radio = widgets.RadioButtons(
        options=ms1_options,
        value=ms1_options[0],
        description='MS1 Notes:',
        layout=widgets.Layout(width='100%', height='360px'),
        style={'description_width': '80px'}
    )

    slider_container = widgets.HBox([], layout=widgets.Layout(justify_content='center', width='71%', margin='-110px 0 0 0'))

    analyst_notes_box = widgets.Text(
        value='',
        placeholder='Enter analyst notes...',
        description='Analyst Notes:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='30px'),
        style={'description_width': '100px'}
    )

    id_notes_box = widgets.Text(
        value='',
        placeholder='Enter ID notes...',
        description='ID Notes:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='30px'),
        style={'description_width': '100px'}
    )
    
    identification_notes_display = widgets.HTML(
        value='',
        layout=widgets.Layout(width='100%', height='auto', margin='5px 0px')
    )

    def get_current_compound() -> Optional[wfo.CompoundExperimental]:
        """Get current CompoundExperimental object (from cache or database)."""
        return get_current_compound_func()

    def write_ms2_title(compound: wfo.CompoundExperimental, ms2_info, has_reference, ms2_file, current_compound_index=None):
        """Create MS2 title and subheadings"""
        name = compound.compound_name
        index = current_compound_index + 1 if current_compound_index is not None else 1
        heading = f"{name} {index}"

        if has_reference is True and ms2_info is not None:
            subheading1 = (
                f"Measured m/z: {ms2_info.get('mz_measured', 0.0):.4f}  |  Reference m/z: {ms2_info.get('mz_theoretical', 0.0):.4f}  |  "
                f"Reference DB: {ms2_info.get('database', 'N/A')}  |  Ref Matches: {ms2_info.get('num_matches', 0)}  |  Score: {ms2_info.get('score', 0.0):.4f}"
            )
        elif has_reference is False and ms2_info is not None:
            subheading1 = (
                f"Measured m/z: {ms2_info.get('precursor_mz', 0.0):.4f}  |  Theoretical m/z: {compound.mz:.4f}  |  "
                f"Reference DB: N/A  |  Ref Matches: N/A  |  Score: N/A"
            )
        else:
            subheading1 = 'No MS2 data available'

        if ms2_file:
            subheading2 = f'File: {ms2_file}'
        else:
            subheading2 = 'No file data'

        ms2_title = f'{heading}<br><sub>{subheading1}</sub><br><sub>{subheading2}</sub>'
        return ms2_title

    def write_eic_title(compound: wfo.CompoundExperimental, current_compound_index=None):
        """Create EIC title and subheadings"""
        eic_title_parts = []
        
        if current_compound_index is not None:
            name = compound.compound_name
            index = current_compound_index + 1
            eic_title_parts.append(f'{name} {index}')
        
        if compound.adduct:
            eic_title_parts.append(f'Adduct: {compound.adduct}')
        if compound.inchi_key:
            eic_title_parts.append(f'InChI Key: {compound.inchi_key}')
        eic_title_parts.append(f'LCMS files: {len(compound.eic_data_files)}')
        subheading1 = ' | '.join(eic_title_parts)

        rt_parts = [f'Atlas RT: {compound.atlas_rt_peak:.2f}']
        best_rt = compound.best_eic_rt
        if best_rt is not None and best_rt > 0:
            current_rt_diff = compound.atlas_rt_peak - best_rt
            rt_parts.extend([
                f'Measured RT: {best_rt:.2f} min',
                f'RT diff: {current_rt_diff:.2f} min'
            ])
        else:
            rt_parts.extend(['Measured RT: N/A', 'RT diff: N/A'])
        
        rt_parts.append(f'Atlas m/z: {compound.mz:.4f}')
        best_mz = compound.best_eic_mz
        if best_mz is not None and best_mz > 0:
            ppm_diff = compound.best_eic_ppm_error
            rt_parts.extend([
                f'Measured m/z: {best_mz:.4f}',
                f'ppm diff: {ppm_diff:.2f}' if ppm_diff is not None else 'ppm diff: N/A'
            ])
        else:
            rt_parts.extend(['Measured m/z: N/A', 'ppm diff: N/A'])
        
        subheading2 = '   |   '.join(rt_parts)

        isomers = compound.isomers
        isomer_strings = []
        for isomer in isomers:
            name = isomer.get('compound_name', '')
            rt_pk = isomer.get('rt', '')
            if rt_pk:
                isomer_strings.append(f"{name} {rt_pk:.2f}")
        
        subheading3 = f'Isomers: {", ".join(isomer_strings)}' if isomer_strings else 'No isomers'

        eic_title = f'<sub>{subheading1}</sub><br><sub>{subheading2}</sub><br><sub>{subheading3}</sub>'
        return eic_title

    def create_targeted_analysis_plot(compound: wfo.CompoundExperimental, 
                                      height=600, ms2_plot_data=None,
                                      initial_render=False, current_compound_index=None):
        """
        Create the plot using CompoundExperimental object.
        Args:
            ms2_plot_data: tuple (spec_data_dict, selected_file)
            initial_render: if True, update EIC x-axis bounds based on RT bounds
        """

        current_rt_min = compound.rt_min
        current_rt_max = compound.rt_max
        current_rt_peak = compound.rt_peak
        
        # Only update x-axis bounds on initial render (compound navigation)
        # Otherwise, keep the existing x-axis bounds to maintain slider alignment
        if initial_render:
            x_min = current_rt_min - 1.0
            x_max = current_rt_max + 1.0
            # Store the x-axis bounds in compound metadata for consistency
            compound._plot_x_bounds = (x_min, x_max)
        else:
            # Use stored bounds if available, otherwise calculate from current RT bounds
            if hasattr(compound, '_plot_x_bounds'):
                x_min, x_max = compound._plot_x_bounds
            else:
                x_min = current_rt_min - 1.0
                x_max = current_rt_max + 1.0
                compound._plot_x_bounds = (x_min, x_max)

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.4, 0.6],
            vertical_spacing=0.35,  # Increased from 0.3 to add more space between MS2 and EIC plots
            subplot_titles=('', ''),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        eic_row = 2
        ms2_row = 1

        ms2_plot_data_dict = ms2_plot_data[0]
        selected_file = ms2_plot_data[1]
        
        # Initialize defaults
        exp_max_intensity = 0
        max_mz = 500.0
        pmz = 0.0
        y_range = [-1000, 1000]  # Default range
        
        def add_experimental_spectrum(mz_vals, intensity_vals, fragment_colors):
            """Helper function to add experimental spectrum traces"""
            for mz, intensity, color in zip(mz_vals, intensity_vals, fragment_colors):
                fig.add_trace(go.Scatter(
                    x=[mz, mz], y=[0, intensity],
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip', name='Exp Frag'
                ), row=ms2_row, col=1)
            
            # Add invisible markers for hover interaction
            fig.add_trace(go.Scatter(
                x=list(mz_vals), y=list(intensity_vals),
                mode='markers', marker=dict(size=1, color="black", opacity=0.01),
                showlegend=False, name='Exp Frag',
                hovertemplate='m/z: %{x:.4f}<br>Intensity: %{y:e}'
            ), row=ms2_row, col=1)
        
        def add_reference_spectrum(mz_vals, intensity_vals, fragment_colors, exp_max_intensity):
            """Helper function to add reference spectrum traces (mirrored downward)"""
            ref_max_intensity = np.nanmax(intensity_vals) if len(intensity_vals) > 0 else 1
            scale_factor = exp_max_intensity / ref_max_intensity if ref_max_intensity > 0 else 1
            scaled_intensities = [i * scale_factor for i in intensity_vals]

            for mz, scaled_int, color in zip(mz_vals, scaled_intensities, fragment_colors):
                fig.add_trace(go.Scatter(
                    x=[mz, mz], y=[0, -scaled_int],  # Negative for downward
                    mode='lines', line=dict(color=color, width=2),
                    showlegend=False, hoverinfo='skip', name='Ref Frag'
                ), row=ms2_row, col=1)
            
            # Add invisible markers for hover interaction
            fig.add_trace(go.Scatter(
                x=list(mz_vals), y=[-s for s in scaled_intensities],
                mode='markers', marker=dict(size=1, color='red', opacity=0.01),
                showlegend=False, name='Ref Frag',
                hovertemplate='m/z: %{x:.4f}<br>Intensity: %{customdata:e}',
                customdata=list(intensity_vals)  # Use original intensities for hover
            ), row=ms2_row, col=1)

        if ms2_plot_data_dict is not None:
            # Determine data type and extract spectra
            has_reference = "database" in ms2_plot_data_dict
            
            if has_reference:
                # Database hit with reference spectrum
                query_spectrum = ms2_plot_data_dict['qry_spectrum']
                ref_spectrum = ms2_plot_data_dict['ref_spectrum']
                fragment_colors = ms2_plot_data_dict['qry_frag_colors']
                pmz = ms2_plot_data_dict.get('mz_theoretical', 0.0)
            else:
                # Extracted spectrum only
                query_spectrum = ms2_plot_data_dict['spectrum']
                ref_spectrum = None
                fragment_colors = ['red'] * len(ms2_plot_data_dict['spectrum'][0])
                pmz = ms2_plot_data_dict.get('precursor_mz', 0.0)
            
            # Plot experimental spectrum
            mz_vals, intensity_vals = query_spectrum[0], query_spectrum[1]
            if len(mz_vals) > 0 and len(intensity_vals) > 0:
                exp_max_intensity = np.nanmax(intensity_vals)
                add_experimental_spectrum(mz_vals, intensity_vals, fragment_colors)

                # Plot reference spectrum if available
                if ref_spectrum is not None:
                    ref_mz_vals, ref_intensity_vals = ref_spectrum[0], ref_spectrum[1]
                    if len(ref_mz_vals) > 0 and len(ref_intensity_vals) > 0:
                        max_mz = max(np.nanmax(mz_vals), np.nanmax(ref_mz_vals))
                        add_reference_spectrum(ref_mz_vals, ref_intensity_vals, fragment_colors, exp_max_intensity)
                
                # Set appropriate y-range based on data type
                if has_reference:
                    y_range = [-exp_max_intensity * 1.1, exp_max_intensity * 1.1]  # Symmetric for mirror plot
                else:
                    y_range = [0, exp_max_intensity * 1.1]  # Start at 0 for single spectrum

            ms2_title = write_ms2_title(compound, ms2_plot_data_dict, has_reference, selected_file)
        else:
            # No MS2 data - add invisible trace to initialize subplot
            ms2_title = write_ms2_title(compound, None, None, None)
            y_range = [-1000, 1000]  # Default for empty plot
            fig.add_trace(go.Scatter(
                x=[0, max_mz], y=[0, 0],
                mode='lines', line=dict(color='rgba(0,0,0,0)', width=0),
                showlegend=False, hoverinfo='skip', name='Empty'
            ), row=ms2_row, col=1)

        # Configure MS2 subplot
        fig.update_xaxes(
            title_text="m/z",
            showgrid=False,
            range=[0, max_mz*1.1],
            row=ms2_row, col=1
        )
        
        fig.update_yaxes(
            title_text="Intensity",
            showgrid=False,
            range=y_range,
            row=ms2_row, col=1,
            exponentformat="e",
            showexponent="all"
        )
        fig.add_hline(y=0, line_color="black", line_width=1, row=ms2_row, col=1)
        fig.add_vline(x=pmz, line_dash="dash", line_color="black",
                        line_width=1, row=ms2_row, col=1)

        # Add EIC traces
        def get_file_color(file_name, color_dict, default_color="gray"):
            """Get color for file based on color dictionary or default"""
            if not color_dict:
                return default_color
            for key, color in color_dict.items():
                if key.lower() in file_name.lower():
                    return color
            return default_color

        for file_name, trace_data in compound.eic_data_files.items():
            color = get_file_color(file_name, file_color_dict)
            
            # Sort rt_vals and i_vals by rt_vals
            rt_vals = np.array(trace_data['rt_vals'])
            i_vals = np.array(trace_data['i_vals'])
            sort_idx = np.argsort(rt_vals)
            rt_vals_sorted = rt_vals[sort_idx]
            i_vals_sorted = i_vals[sort_idx]
            
            fig.add_trace(go.Scatter(
            x=rt_vals_sorted,
            y=i_vals_sorted,
            mode='lines',
            name=file_name,
            line=dict(color=color, width=2),
            hovertemplate=f'{file_name}<extra></extra>',
            showlegend=False
            ), row=eic_row, col=1)

        # Add RT reference lines using CURRENT working values
        fig.add_vline(x=current_rt_peak, line_dash="dash", line_color="black", line_width=3, row=eic_row, col=1)
        fig.add_vline(x=current_rt_min, line_dash="dot", line_color="green", line_width=2, row=eic_row, col=1)
        fig.add_vline(x=current_rt_max, line_dash="dot", line_color="green", line_width=2, row=eic_row, col=1)

        # Add RT reference lines from original atlas
        fig.add_vline(x=compound.atlas_rt_peak, line_dash="dash", line_color="gray", line_width=3, row=eic_row, col=1)
        fig.add_vline(x=compound.atlas_rt_min, line_dash="dot", line_color="gray", line_width=2, row=eic_row, col=1)
        fig.add_vline(x=compound.atlas_rt_max, line_dash="dot", line_color="gray", line_width=2, row=eic_row, col=1)

        # Add suggested RT bounds as two light green vlines if available
        suggested_data = compound.suggested_rt_bounds
        if suggested_data is not None:
            rt_min = suggested_data['rt_min']
            rt_max = suggested_data['rt_max']
            # Add light green vlines for suggested RT min and max
            fig.add_vline(
                x=rt_min,
                line_dash="dot",
                line_color="lightgreen",
                line_width=3,
                row=eic_row,
                col=1
            )
            fig.add_vline(
                x=rt_max,
                line_dash="dot",
                line_color="lightgreen",
                line_width=3,
                row=eic_row,
                col=1
            )

        # Configure EIC subplot
        fig.update_xaxes(title_text='RT', 
                        range=[x_min, x_max],
                        showgrid=False, zeroline=False, showline=True,
                        linewidth=1, linecolor='black', ticks='outside',
                        tickwidth=1, tickcolor='black', row=eic_row, col=1)
        fig.update_yaxes(title_text='Intensity',
                        range=[0, None],
                        showgrid=False, zeroline=False, showline=True,
                        linewidth=1, linecolor='black', ticks='outside',
                        tickwidth=1, tickcolor='black', row=eic_row, col=1,
                        exponentformat="e",  # Use scientific notation
                        showexponent="all"
        )
        
        eic_title = write_eic_title(compound, current_compound_index)

        # Adjusted annotation positions
        fig.add_annotation(
            text=eic_title,
            x=0.5, y=0.53,  # Reduced from 0.58 to move EIC title closer to plot
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            xanchor="center"
        )

        fig.add_annotation(
            text=ms2_title,
            x=0.5, y=1.13,  # Increased from 1.1 to add more space above MS2 plot
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=12),
            xanchor="center"
        )

        # Final layout configuration
        fig.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),  # Reduced margins
            hovermode='closest',
            height=height,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
            yaxis=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
            xaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
            yaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True)
        )

        # Add scale toggle buttons for both plots
        updatemenus = [
            # EIC plot buttons (bottom left)
            dict(
                type="buttons", direction="right",
                buttons=[
                    dict(args=[{f"yaxis{eic_row}.type": "linear", f"yaxis{eic_row}.autorange": True}], 
                            label="Lin", method="relayout"),
                    dict(args=[{f"yaxis{eic_row}.type": "log", f"yaxis{eic_row}.autorange": True}], 
                            label="Log", method="relayout")
                ],
                x=0.02, xanchor="left", y=0.02, yanchor="bottom",
                bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
            ),
        ]
        
        fig.update_layout(updatemenus=updatemenus)

        return fig

    def get_selected_ms2_spectra(compound: wfo.CompoundExperimental, current_ms2_file_index, rt_min, rt_max):
        """
        Returns the selected MS2 spectra for plotting and info for the title.
        Uses CompoundExperimental structure and filters by RT bounds.
        """
        
        print(f"Getting MS2 spectra for compound: {compound.compound_name}")
        # Get MS2 data from CompoundExperimental structure
        ms2_data = compound.ms2_data_files
        if not ms2_data:
            return None, None, []
        print(ms2_data)
        
        # Filter files that have MS2 data within RT bounds and calculate scores
        files_in_rt_window = []
        for file_name, file_data in ms2_data.items():
            if not isinstance(file_data, dict):
                continue
                
            # Get RT from best_hit or best_ms2
            best_hit = file_data.get('best_hit', {})
            best_ms2 = file_data.get('best_ms2', {})
            
            rt_measured = None
            if best_hit and 'rt_measured' in best_hit:
                rt_measured = best_hit.get('rt_measured')
            elif best_ms2:
                rt_measured = best_ms2.get('rt_measured') or best_ms2.get('rt') or best_ms2.get('rt_peak')
            
            # Only include files where MS2 data falls within current RT bounds
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                # Prioritize score from best_hit, fallback to intensity from best_ms2
                if best_hit and 'score' in best_hit:
                    score = best_hit.get('score', 0.0)
                elif best_ms2 and 'intensity_peak' in best_ms2:
                    score = best_ms2.get('intensity_peak', 0.0)
                else:
                    score = 0.0
                files_in_rt_window.append((file_name, score))
        
        # If no files have MS2 data in the current RT window, return None
        if not files_in_rt_window:
            return None, None, []
        
        # Sort files by score (highest first)
        files_in_rt_window.sort(key=lambda x: x[1], reverse=True)
        sorted_file_names = [name for name, score in files_in_rt_window]
        
        # Clamp file index to available files within RT window
        idx = min(current_ms2_file_index, len(sorted_file_names) - 1)
        selected_file = sorted_file_names[idx]
        file_data = ms2_data[selected_file]
        
        # Check if we have a best_hit (with reference data)
        best_hit = file_data.get('best_hit', {})
        best_ms2 = file_data.get('best_ms2', {})

        if best_hit:  # We have reference hit data
            return best_hit, selected_file, files_in_rt_window
        elif not best_hit and best_ms2:  # Use experimental data only
            return best_ms2, selected_file, files_in_rt_window
        else:
            return None, selected_file, files_in_rt_window

    def create_plot_only(initial_render=False):
        """ONLY create and display the plot - no other side effects."""
        compound = get_current_compound()
        if compound is None:
            return

        plot_output.clear_output(wait=True)

        rt_min = compound.rt_min
        rt_max = compound.rt_max

        plot_spec_data, ms2_file, files_in_rt_window = get_selected_ms2_spectra(
            compound, current_ms2_file_index[0], rt_min, rt_max
        )

        if len(files_in_rt_window) > 0:
            current_file_idx = min(current_ms2_file_index[0], len(files_in_rt_window) - 1)
            ms2_counter_label.value = f"MS2 File {current_file_idx + 1} of {len(files_in_rt_window)}"
        else:
            ms2_counter_label.value = "No MS2 data"

        fig = create_targeted_analysis_plot(
            compound=compound,
            ms2_plot_data=(plot_spec_data, ms2_file),
            initial_render=initial_render,
            current_compound_index=current_compound_index[0]
        )

        if fig is not None:
            with plot_output:
                fig.show()

    def full_update():
        """Perform a complete update of all UI elements."""
        if updating[0]:
            return
            
        updating[0] = True
        try:
            compound = get_current_compound()
            if compound is None:
                return
            
            rt_min = compound.rt_min
            rt_max = compound.rt_max
            
            rt_range = rt_max - rt_min
            slider_min = rt_min - 1
            slider_max = rt_max + 1
            slider_step = max(0.01, rt_range / 100)
            
            new_slider = widgets.FloatRangeSlider(
                value=(rt_min, rt_max),
                min=slider_min,
                max=slider_max,
                step=slider_step,
                description='RT',
                continuous_update=True,
                layout=widgets.Layout(width='95%'),
                readout=True
            )
            
            new_slider.observe(on_rt_slider_change, names='value')
            slider_container.children = [new_slider]
            rt_slider[0] = new_slider
            
            update_annotation_widgets()
            update_dropdown_value()
            create_plot_only(initial_render=True)
            
        finally:
            updating[0] = False

    def update_annotation_widgets():
        """Update annotation radio buttons to reflect current compound's values"""
        compound = get_current_compound()
        if compound is None:
            return

        temp_updating = updating[0]
        updating[0] = True
        
        try:
            ms2_radio.value = compound.ms2_notes
            ms1_radio.value = compound.ms1_notes
            analyst_notes_box.value = compound.analyst_notes
            id_notes_box.value = compound.identification_notes
        finally:
            updating[0] = temp_updating

    def update_dropdown_value():
        """Update dropdown to reflect current compound without triggering observer"""
        idx = current_compound_index[0]
        if 0 <= idx < len(dropdown_options):
            temp_updating = updating[0]
            updating[0] = True
            try:
                compound_dropdown.value = dropdown_options[idx]
            finally:
                updating[0] = temp_updating

    def reset_ms2_file_index():
        """Reset MS2 file index to 0"""
        current_ms2_file_index[0] = 0

    # Event handlers that save to database
    def on_rt_slider_change(change):
        """Callback that fires whenever the user drags the slider - saves to database."""
        if updating[0]:
            return
        
        compound = get_current_compound()
        if compound is None:
            return

        new_min, new_max = change['new']
        current_rt_min = compound.rt_min
        current_rt_max = compound.rt_max
        
        tolerance = 1e-6
        if (abs(new_min - current_rt_min) < tolerance and 
            abs(new_max - current_rt_max) < tolerance):
            return

        new_peak = new_min + (new_max - new_min) / 2.0
        compound.update_rt_bounds(new_min, new_max, new_peak)
        
        # Save to database immediately
        save_compound_to_db(compound)
        reset_ms2_file_index()
        create_plot_only()

    def on_ms2_annotation_change(change):
        """Handle MS2 annotation changes - saves to database."""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        compound.update_annotations(ms2_notes=change['new'])
        save_compound_to_db(compound)

    def on_ms1_annotation_change(change):
        """Handle MS1 annotation changes - saves to database."""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        compound.update_annotations(ms1_notes=change['new'])
        save_compound_to_db(compound)

    def on_analyst_notes_change(change):
        """Handle analyst notes changes - saves to database."""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        compound.update_annotations(analyst_notes=change['new'])
        save_compound_to_db(compound)

    def on_id_notes_change(change):
        """Handle ID notes changes - saves to database."""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        compound.update_annotations(identification_notes=change['new'])
        save_compound_to_db(compound)
    
    def on_navigation(direction):
        """Handle compound navigation - clears cache and loads new compound."""
        if updating[0]:
            return

        # Save current compound before navigating
        current = get_current_compound()
        if current:
            save_compound_to_db(current)
        
        idx = current_compound_index[0] + direction
        if 0 <= idx < len(compound_list):
            current_compound_index[0] = idx
            counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
            
            # Clear cache to force reload from database
            current_compound_cache[0] = None
            reset_ms2_file_index()
            full_update()

    def on_ms2_navigation(direction):
        """Handle MS2 file navigation"""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return

        rt_min = compound.rt_min
        rt_max = compound.rt_max
        ms2_data = compound.ms2_data_files
        if not ms2_data:
            return
        
        files_in_rt_window = []
        for file_name, file_data in ms2_data.items():
            if not isinstance(file_data, dict):
                continue
            best_hit = file_data.get('best_hit', {})
            best_ms2 = file_data.get('best_ms2', {})
            
            rt_measured = None
            if best_hit and 'rt_measured' in best_hit:
                rt_measured = best_hit.get('rt_measured')
            elif best_ms2:
                rt_measured = best_ms2.get('rt_measured') or best_ms2.get('rt') or best_ms2.get('rt_peak')
            
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                if best_hit and 'score' in best_hit:
                    score = best_hit.get('score', 0.0)
                elif best_ms2 and 'intensity_peak' in best_ms2:
                    score = best_ms2.get('intensity_peak', 0.0)
                else:
                    score = 0.0
                files_in_rt_window.append((file_name, score))
        
        total_files = len(files_in_rt_window)
        if total_files == 0:
            return

        new_idx = current_ms2_file_index[0] + direction
        if 0 <= new_idx < total_files:
            current_ms2_file_index[0] = new_idx
            create_plot_only(initial_render=False)

    def on_reset(button):
        """Reset current compound to original RT bounds and annotations"""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        
        updating[0] = True
        try:
            compound.rt_min = compound.atlas_rt_min
            compound.rt_max = compound.atlas_rt_max
            compound.rt_peak = compound.atlas_rt_peak
            compound.is_rt_modified = False
            save_compound_to_db(compound)
            reset_ms2_file_index()
            full_update()
        finally:
            updating[0] = False

    def on_accept_suggestions(button):
        """Accept suggested RT bounds for current compound"""
        if updating[0]:
            return
        compound = get_current_compound()
        if compound is None:
            return
        suggested_data = compound.suggested_rt_bounds
        if suggested_data is None:
            return
        
        updating[0] = True
        try:
            reset_ms2_file_index()
            compound.update_rt_bounds(
                suggested_data['rt_min'],
                suggested_data['rt_max'],
                suggested_data['rt_peak']
            )
            save_compound_to_db(compound)
            create_plot_only()
            if rt_slider[0] is not None:
                rt_slider[0].value = (suggested_data['rt_min'], suggested_data['rt_max'])
        finally:
            updating[0] = False

    def on_dropdown_change(change):
        """Handle dropdown selection changes"""
        if updating[0]:
            return
        try:
            selected_value = change['new']
            idx = int(selected_value.split(':')[0]) - 1
            
            if 0 <= idx < len(compound_list) and idx != current_compound_index[0]:
                # Save current before switching
                current = get_current_compound()
                if current:
                    save_compound_to_db(current)
                
                current_compound_index[0] = idx
                counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
                current_compound_cache[0] = None
                reset_ms2_file_index()
                full_update()
        except (ValueError, IndexError):
            pass

    # Status button (shows database connection)
    status_button = widgets.Button(
        description="✓ DB Connected", 
        button_style='success', 
        layout=widgets.Layout(width='150px'),
        disabled=True
    )
    
    # Connect event handlers
    prev_button.on_click(lambda b: on_navigation(-1))
    next_button.on_click(lambda b: on_navigation(1))
    ms2_prev_button.on_click(lambda b: on_ms2_navigation(-1))
    ms2_next_button.on_click(lambda b: on_ms2_navigation(1))
    reset_button.on_click(on_reset)
    accept_suggestions_button.on_click(on_accept_suggestions)
    
    ms2_radio.observe(on_ms2_annotation_change, names='value')
    ms1_radio.observe(on_ms1_annotation_change, names='value')
    analyst_notes_box.observe(on_analyst_notes_change, names='value')
    id_notes_box.observe(on_id_notes_change, names='value')
    compound_dropdown.observe(on_dropdown_change, names='value')
    
    # Create layout (same as original)
    nav_row = widgets.HBox([prev_button, counter_label, next_button], layout=widgets.Layout(justify_content='center'))
    dropdown_row = widgets.HBox([compound_dropdown], layout=widgets.Layout(justify_content='center', margin='5px 0'))
    ms2_nav_row = widgets.HBox([ms2_prev_button, ms2_counter_label, ms2_next_button], layout=widgets.Layout(justify_content='center'))

    nav_box = widgets.HBox([nav_row, dropdown_row, ms2_nav_row], layout=widgets.Layout(width='100%', align_items='center'))
        
    plot_and_radios = widgets.HBox([
        plot_output,
        widgets.VBox([
            widgets.Box([ms2_radio], layout=widgets.Layout(margin='50px 0 50px 0')),
            ms1_radio
        ], layout=widgets.Layout(width='30%', justify_content='flex-start'))
    ], layout=widgets.Layout(width='100%'))
    
    button_row = widgets.HBox([reset_button, accept_suggestions_button, status_button], 
                             layout=widgets.Layout(justify_content='center', margin='0 0 0 0'))
    
    container = widgets.VBox([
        nav_box, 
        analyst_notes_box,
        id_notes_box,
        plot_and_radios,
        slider_container,
        button_row,
        identification_notes_display
    ], layout=widgets.Layout(width='100%', align_items='flex-start', height='fit-content'))
    
    # Initialize with the first compound
    full_update()
    
    logger.info(f"Database-connected GUI created with {len(compound_list)} compounds")
    return container, db_info