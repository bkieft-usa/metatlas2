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

def to_python_type(val):
    """Convert numpy types to native Python types for DB compatibility."""
    if isinstance(val, (np.generic,)):
        return val.item()
    return val

def update_manual_curation_entry(
    project_db_path: str,
    curation_uid: str,
    updates: Dict[str, Any]
) -> None:
    """Update manual_curation entry in DB."""
    if not updates:
        return
    set_clause = ', '.join([f"{k} = ?" for k in updates])
    # Convert all values to Python native types
    params = [to_python_type(v) for v in updates.values()] + [curation_uid]
    with dbi.get_db_connection(project_db_path) as conn:
        conn.execute(
            f"UPDATE manual_curation SET {set_clause} WHERE curation_uid = ?",
            params
        )

def run_analysis_gui(
    analysis_gui_obj: "AnalysisGUIWorkflowObject"
):
    # --- 1. Load all dataframes into memory ---
    manual_curation_df = dbi.get_manual_curation_entries(
        analysis_gui_obj.paths['project_db_path'], 
        analysis_gui_obj.rt_alignment_number, 
        analysis_gui_obj.analysis_number
    )

    ms1_data_df = dbi.get_ms1_data_for_compound(
        analysis_gui_obj.paths['project_db_path'], 
        None, 
        None, 
        analysis_gui_obj.rt_alignment_number, 
        analysis_gui_obj.analysis_number
    )

    ms2_hits_df = dbi.get_ms2_hits_for_compound(
        analysis_gui_obj.paths['project_db_path'], 
        None, 
        None, 
        analysis_gui_obj.rt_alignment_number, 
        analysis_gui_obj.analysis_number
    )

    ms2_data_df = dbi.get_ms2_data_for_compound(
        analysis_gui_obj.paths['project_db_path'], 
        None, 
        None, 
        analysis_gui_obj.rt_alignment_number, 
        analysis_gui_obj.analysis_number
    )

    compound_list = manual_curation_df.index.tolist()
    curation_buffer = {}
    current_compound_index = [0]
    updating = [False]
    rt_slider = [None]
    current_ms2_file_index = [0]

    # --- 2. Helper functions for current compound and data subsetting ---
    def get_current_curation_row():
        return manual_curation_df.iloc[current_compound_index[0]]

    def get_current_curation_uid():
        return get_current_curation_row()['curation_uid']

    def get_current_inchi_adduct():
        row = get_current_curation_row()
        return row['inchi_key'], row['adduct']

    def get_ms1_data_for_current():
        inchi_key, adduct = get_current_inchi_adduct()
        return ms1_data_df[
            (ms1_data_df['inchi_key'] == inchi_key) &
            (ms1_data_df['adduct'] == adduct)
        ]

    def get_ms2_hits_for_current():
        inchi_key, adduct = get_current_inchi_adduct()
        return ms2_hits_df[
            (ms2_hits_df['inchi_key'] == inchi_key) &
            (ms2_hits_df['adduct'] == adduct)
        ].sort_values('score', ascending=False).head(20)

    def get_ms2_data_for_current():
        inchi_key, adduct = get_current_inchi_adduct()
        return ms2_data_df[
            (ms2_data_df['inchi_key'] == inchi_key) &
            (ms2_data_df['adduct'] == adduct)
        ]

    # --- 3. Buffer/flush mechanism ---
    def buffer_update(field, value):
        curation_uid = get_current_curation_uid()
        if curation_uid not in curation_buffer:
            curation_buffer[curation_uid] = {}
        curation_buffer[curation_uid][field] = value

    def flush_buffer_for_current():
        curation_uid = get_current_curation_uid()
        if curation_uid in curation_buffer:
            update_manual_curation_entry(
                analysis_gui_obj.paths['project_db_path'], 
                curation_uid, 
                curation_buffer[curation_uid]
            )
            curation_buffer.pop(curation_uid)

    # --- 4. Widget setup ---
    ms2_options = [
        'no selection',
        '-1.0, poor match, should remove',
        '0.0, no match or no MSMS collected',
        '0.5, partial or putative match of fragments',
        '1.0, good match',
        '0.5, co-isolated precursor, partial match',
        '1.0, co-isolated precursor, good match',
        '0.5, single ion match, no evidence',
        '1.0, single ion match, ISTD/ref evidence'
    ]
    ms1_options = ['keep', 'remove', 'unresolvable isomers', 'poor peak shape']

    dropdown_options = [
        f"{i+1}: {get_current_curation_row()['compound_name']}" for i in range(len(compound_list))
    ]

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
    slider_container = widgets.HBox(
        [], 
        layout=widgets.Layout(
            justify_content='center',
             width='70%', 
             margin='0')
        )
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

    # --- 5. Event handlers ---
    def on_rt_slider_change(change):
        if updating[0]:
            return
        new_min, new_max = change['new']
        buffer_update('rt_min', new_min)
        buffer_update('rt_max', new_max)
        buffer_update('rt_peak', (new_min + new_max) / 2)
        manual_curation_df.at[get_current_curation_row().name, 'rt_min'] = new_min
        manual_curation_df.at[get_current_curation_row().name, 'rt_max'] = new_max
        manual_curation_df.at[get_current_curation_row().name, 'rt_peak'] = (new_min + new_max) / 2
        manual_curation_df.at[get_current_curation_row().name, 'is_rt_modified'] = True
        create_plot_only()

    def on_ms2_annotation_change(change):
        if updating[0]:
            return
        buffer_update('ms2_notes', change['new'])
        manual_curation_df.at[get_current_curation_row().name, 'ms2_notes'] = change['new']
        manual_curation_df.at[get_current_curation_row().name, 'is_annotation_modified'] = True

    def on_ms1_annotation_change(change):
        if updating[0]:
            return
        buffer_update('ms1_notes', change['new'])
        manual_curation_df.at[get_current_curation_row().name, 'ms1_notes'] = change['new']
        manual_curation_df.at[get_current_curation_row().name, 'is_annotation_modified'] = True

    def on_analyst_notes_change(change):
        if updating[0]:
            return
        buffer_update('analyst_notes', change['new'])
        manual_curation_df.at[get_current_curation_row().name, 'analyst_notes'] = change['new']

    def on_id_notes_change(change):
        if updating[0]:
            return
        buffer_update('identification_notes', change['new'])
        manual_curation_df.at[get_current_curation_row().name, 'identification_notes'] = change['new']

    def on_navigation(direction):
        if updating[0]:
            return
        flush_buffer_for_current()
        idx = current_compound_index[0] + direction
        if 0 <= idx < len(compound_list):
            current_compound_index[0] = idx
            counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
            current_ms2_file_index[0] = 0
            full_update()

    def on_ms2_navigation(direction):
        if updating[0]:
            return
        # Only update the MS2 scan index and plot
        ms2_df = get_ms2_data_for_current()
        ms2_hits = get_ms2_hits_for_current()
        if not ms2_df.empty:
            grouped = ms2_df.groupby(['inchi_key', 'adduct', 'file_path', 'rt']).agg({
                'mz': lambda x: list(x),
                'i': lambda x: list(x),
                'precursor_MZ': 'first'
            }).reset_index()
        else:
            grouped = ms2_df.copy()
        if grouped.empty and ms2_hits.empty:
            scans = grouped
        elif not grouped.empty and ms2_hits.empty:
            scans = grouped.sort_values('rt')
        elif grouped.empty and not ms2_hits.empty:
            scans = pd.DataFrame()
        else:
            merged = pd.merge(
                grouped,
                ms2_hits,
                left_on=['inchi_key', 'adduct', 'file_path', 'rt'],
                right_on=['inchi_key', 'adduct', 'file_path', 'rt_measured'],
                how='left'
            )
            merged['score'] = merged['score'].fillna(0)
            scans = merged.sort_values(['score', 'rt'], ascending=[False, True])
        total_scans = len(scans)
        # Clamp index
        current_ms2_file_index[0] += direction
        if total_scans == 0:
            current_ms2_file_index[0] = 0
        else:
            current_ms2_file_index[0] = max(0, min(current_ms2_file_index[0], total_scans - 1))
        create_plot_only()

    def on_accept_suggestions(button):
        if updating[0]:
            return
        row = get_current_curation_row()
        if pd.notnull(row['suggested_rt_min']) and pd.notnull(row['suggested_rt_max']) and pd.notnull(row['suggested_rt_peak']):
            buffer_update('rt_min', row['suggested_rt_min'])
            buffer_update('rt_max', row['suggested_rt_max'])
            buffer_update('rt_peak', row['suggested_rt_peak'])
            manual_curation_df.at[row.name, 'rt_min'] = row['suggested_rt_min']
            manual_curation_df.at[row.name, 'rt_max'] = row['suggested_rt_max']
            manual_curation_df.at[row.name, 'rt_peak'] = row['suggested_rt_peak']
            manual_curation_df.at[row.name, 'is_rt_modified'] = True
            create_plot_only()

    def on_dropdown_change(change):
        if updating[0]:
            return
        try:
            idx = int(change['new'].split(':')[0]) - 1
            if 0 <= idx < len(compound_list) and idx != current_compound_index[0]:
                flush_buffer_for_current()
                current_compound_index[0] = idx
                counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
                current_ms2_file_index[0] = 0
                full_update()
        except Exception:
            pass

    def get_plot_range(entry):
        if entry.is_rt_modified:
            rt_min = entry.rt_min
            rt_max = entry.rt_max
        else:
            rt_min = entry.atlas_rt_min
            rt_max = entry.atlas_rt_max
        return rt_min - 1, rt_max + 1

    def _create_plot_titles(
        manual_curation_entry: pd.DataFrame, 
        ms_level: int, 
        ms1_data: pd.DataFrame = None, 
        ms2_data: pd.DataFrame = None,
    ) -> str:

        title = None
        if ms_level == 1:
            if ms1_data is None or ms1_data.empty:
                title = "No MS1 data available"
            else:
                isomers = json.loads(manual_curation_entry['isomers']) if pd.notnull(manual_curation_entry['isomers']) else []
                # Format isomers as pretty strings, one per line
                if isomers:
                    isomer_lines = []
                    for iso in isomers:
                        line = (
                            f"Name: {iso.get('compound_name', '')}, "
                            f"InChi Key: {iso.get('inchi_key', '')}"
                            f"RT: {iso.get('rt', ''):.2f}, "
                            f"m/z: {iso.get('mz', ''):.4f}, "
                        )
                        isomer_lines.append(line)
                    isomer_str = "<br>".join(isomer_lines)
                else:
                    isomer_str = "None"

                title = (
                    f"<sub>Atlas RT: {manual_curation_entry['atlas_rt_peak']:.4f}  |  Measured RT: {manual_curation_entry['best_ms1_rt']:.4f}  |  RT Diff: {manual_curation_entry['best_ms1_rt_error']:.2f}</sub><br>"
                    f"<sub>Atlas m/z: {manual_curation_entry['atlas_mz']:.4f}  |  Measured m/z: {manual_curation_entry['best_ms1_mz']:.4f}  |  m/z Diff: {manual_curation_entry['best_ms1_ppm_error']:.4f}</sub><br>"
                    f"<sub>Isomers: {isomer_str.replace('<br>', ', ')}</sub>"
                )

        if ms_level == 2:
            if ms2_data is None or ms2_data.empty:
                title = (
                    f"{manual_curation_entry['compound_name']}  |  {manual_curation_entry['adduct']}  |  {manual_curation_entry['inchi_key']}<br>"
                    f"<sub>No MS2 data available</sub>"
                )
            else:
                file_name = "_".join(os.path.basename(ms2_data['file_path']).split('.')[0].split('_')[11:])
                title = (
                    f"{manual_curation_entry['compound_name']}  |  {manual_curation_entry['adduct']}  |  {manual_curation_entry['inchi_key']}<br>"
                    f"<sub>File: {file_name}</sub><br>"
                    f"<sub>Score: {ms2_data['score']:.4f}  |  Scan RT: {ms2_data['rt']:.4f}</sub><br>"
                    f"<sub>Precursor m/z: {ms2_data['precursor_MZ']:.4f}  |  Reference m/z: {ms2_data['mz_theoretical']:.4f}  |  Measured m/z: {ms2_data['mz_measured']:.4f}  |  m/z Diff: {ms2_data['ppm_error']:.2f}</sub><br>"
                )

        return title

    def create_plot_only(initial_render=False):
        """Create and display the plot for the current compound using in-memory DataFrames."""
        plot_output.clear_output(wait=True)
        manual_curation_entry = get_current_curation_row()
        ms1_df = get_ms1_data_for_current()
        ms2_hits = get_ms2_hits_for_current()
        ms2_df = get_ms2_data_for_current()

        # RT bounds for current compound
        rt_min = manual_curation_entry['rt_min']
        rt_max = manual_curation_entry['rt_max']
        rt_peak = manual_curation_entry['rt_peak']

        # --- EIC Plot ---
        sample_color_dict = {
            'istd': 'blue',
            'qc': 'blue',
            'exctrl': 'red',
            'txctrl': 'red',
            'refstd': 'black'
        }

        plot_start, plot_end = get_plot_range(manual_curation_entry)
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.4, 0.6],
            vertical_spacing=0.35,
            subplot_titles=('No MS2 data', 'No MS1 data'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        eic_row = 2
        ms2_row = 1
        fig.update_xaxes(range=[plot_start, plot_end], row=eic_row, col=1)

        # Plot all MS1 EIC traces for this compound
        ms1_title = _create_plot_titles(
            manual_curation_entry=manual_curation_entry,
            ms_level=1,
            ms1_data=ms1_df
        )
        if not ms1_df.empty:
            for file_path in ms1_df['file_path'].unique():
                trace = ms1_df[ms1_df['file_path'] == file_path]
                file_name = "_".join(os.path.basename(file_path).split('.')[0].split('_')[11:])
                file_color = 'gray'
                for key, color in sample_color_dict.items():
                    if key in file_name.lower():
                        file_color = color
                        break
                fig.add_trace(
                    go.Scatter(
                        x=trace['rt'],
                        y=trace['i'],
                        mode='lines',
                        name=file_name,
                        line=dict(width=2, color=file_color),
                        hovertemplate=f'<span style="font-size:9px; white-space:pre-wrap; max-width:400px;">{file_name}</span><extra></extra>',
                        showlegend=False
                    ),
                    row=eic_row, col=1
                )

        # Add RT reference lines (current and atlas)
        #fig.add_vline(x=rt_peak, line_dash="dash", line_color="black", line_width=3, row=eic_row, col=1)
        fig.add_vline(x=rt_min, line_dash="dot", line_color="purple", line_width=4, row=eic_row, col=1)
        fig.add_vline(x=rt_max, line_dash="dot", line_color="purple", line_width=4, row=eic_row, col=1)
        fig.add_vline(x=manual_curation_entry['atlas_rt_peak'], line_dash="dash", line_color="green", line_width=1.5, row=eic_row, col=1)
        fig.add_vline(x=manual_curation_entry['atlas_rt_min'], line_dash="dot", line_color="gray", line_width=1.5, row=eic_row, col=1)
        fig.add_vline(x=manual_curation_entry['atlas_rt_max'], line_dash="dot", line_color="gray", line_width=1.5, row=eic_row, col=1)
        if pd.notnull(manual_curation_entry.get('suggested_rt_min', None)):
            fig.add_vline(x=manual_curation_entry['suggested_rt_min'], line_dash="dot", line_color="orange", line_width=2, row=eic_row, col=1)
        if pd.notnull(manual_curation_entry.get('suggested_rt_max', None)):
            fig.add_vline(x=manual_curation_entry['suggested_rt_max'], line_dash="dot", line_color="orange", line_width=2, row=eic_row, col=1)

        # --- MS2 Plot: Merge ms2_data and ms2_hits ---
        # Group ms2_df by scan and aggregate m/z and i into arrays
        if not ms2_df.empty:
            grouped = ms2_df.groupby(['inchi_key', 'adduct', 'file_path', 'rt']).agg({
                'mz': lambda x: list(x),
                'i': lambda x: list(x),
                'precursor_MZ': 'first'
            }).reset_index()
            grouped['raw_spectrum'] = grouped.apply(lambda row: (row['mz'], row['i']), axis=1)
        else:
            grouped = ms2_df.copy()  # empty dataframe

        # Merge with ms2_hits if present
        if grouped.empty and ms2_hits.empty:
            scans = grouped  # empty dataframe
        elif not grouped.empty and ms2_hits.empty:
            scans = grouped.sort_values('rt')
        elif grouped.empty and not ms2_hits.empty:
            raise ValueError("MS2 hits exist but no MS2 data found - this should not happen")
        else:
            merged = pd.merge(
                grouped,
                ms2_hits,
                left_on=['inchi_key', 'adduct', 'file_path', 'rt'],
                right_on=['inchi_key', 'adduct', 'file_path', 'rt_measured'],
                how='left'
            )
            merged['score'] = merged['score'].fillna(0)
            scans = merged.sort_values(['score', 'rt'], ascending=[False, True])

        total_scans = len(scans)
        idx = min(current_ms2_file_index[0], total_scans - 1) if total_scans > 0 else 0
        scan = scans.iloc[idx] if total_scans > 0 else None

        ms2_counter_label.value = f"MS2 Scan {idx+1} of {total_scans}" if total_scans > 0 else "No MS2 data"

        # Plot MS2 spectrum for selected scan
        ms2_title = _create_plot_titles(
            manual_curation_entry=manual_curation_entry,
            ms_level=2,
            ms2_data=scan
        )
        if scan is not None:
            # Use fragment_colors if present, else default to red
            fragment_colors = None
            if 'aligned_fragment_colors' in scan and pd.notnull(scan['aligned_fragment_colors']):
                fragment_colors = json.loads(scan['aligned_fragment_colors'])
            else:
                fragment_colors = "red"

            if pd.notnull(scan.get('qry_spectrum')) and pd.notnull(scan.get('ref_spectrum')):
                mz_query, int_query = json.loads(scan['qry_spectrum'])
                mz_ref, int_ref = json.loads(scan['ref_spectrum'])

                def replace_nan_in_spectra(mz_arr, int_arr):
                    mz_min = min([x for x in mz_arr if isinstance(x, float) and not np.isnan(x)] + [x for x in mz_arr if isinstance(x, int)])
                    mz_arr_clean = [mz_min if (isinstance(x, float) and np.isnan(x)) else x for x in mz_arr]
                    int_arr_clean = [0 if (isinstance(x, float) and np.isnan(x)) else x for x in int_arr]
                    return mz_arr_clean, int_arr_clean

                mz_query, int_query = replace_nan_in_spectra(mz_query, int_query)
                mz_ref, int_ref = replace_nan_in_spectra(mz_ref, int_ref)

                # --- Scaling reference intensity ---
                max_query = max(int_query) if int_query else 1
                max_ref = max(int_ref) if int_ref else 1
                scale_factor = max_query / max_ref if max_ref > 0 else 1
                scaled_int_ref = [i * scale_factor for i in int_ref]

                fig.add_trace(
                    go.Bar(
                        x=mz_query,
                        y=int_query,
                        marker_color=fragment_colors,
                        width=0.5,
                        hovertemplate='<span style="font-size:9px; white-space:pre-wrap; max-width:400px;">'
                            'Query<br>m/z: %{x:.4f}<br>Intensity: %{y:.2f}</span><extra></extra>'
                    ),
                    row=ms2_row, col=1)

                fig.add_trace(
                    go.Bar(
                        x=mz_ref,
                        y=[-i for i in scaled_int_ref],
                        marker_color=fragment_colors,
                        width=0.5,
                        hovertemplate=(
                            '<span style="font-size:9px; white-space:pre-wrap; max-width:400px;">'
                            f'Reference<br>m/z: %{{x:.4f}}<br>Intensity: %{{y:.2f}}<br>Scale factor: {scale_factor:.2f}'
                            '</span><extra></extra>'
                        )
                    ),
                    row=ms2_row, col=1)

            else:
                # Plot raw_spectrum if no hit
                mz_raw, int_raw = scan['raw_spectrum']
                fig.add_trace(
                    go.Bar(
                        x=mz_raw,
                        y=int_raw,
                        marker_color=fragment_colors,
                        name='MS2',
                        width=0.5,
                        hovertemplate='<span style="font-size:9px; white-space:pre-wrap; max-width:400px;">m/z: %{x:.4f}<br>Intensity: %{y:.2f}</span><extra></extra>'
                    ),
                    row=ms2_row, col=1)

        fig.layout.annotations[0].update(text=ms2_title, font=dict(size=10))
        fig.layout.annotations[1].update(text=ms1_title, font=dict(size=10))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            hovermode='closest',
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        fig.add_hline(y=0, line_color="black", line_width=1, row=ms2_row, col=1)
        fig.update_xaxes(title_text='RT', row=eic_row, col=1)
        fig.update_yaxes(title_text='Intensity', row=eic_row, col=1)
        fig.update_xaxes(title_text='m/z', row=ms2_row, col=1)
        fig.update_yaxes(title_text='Intensity', row=ms2_row, col=1)

        with plot_output:
            fig.show()

    def full_update():
        """Update all widgets and plot for the current compound."""
        if updating[0]:
            return
        updating[0] = True
        try:
            row = get_current_curation_row()
            # Update slider
            rt_min = row['rt_min']
            rt_max = row['rt_max']
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
                layout=widgets.Layout(width='100%'),
                readout=True
            )
            new_slider.observe(on_rt_slider_change, names='value')
            slider_container.children = [new_slider]
            rt_slider[0] = new_slider

            # Update annotation widgets
            ms2_radio.value = row['ms2_notes']
            ms1_radio.value = row['ms1_notes']
            analyst_notes_box.value = row['analyst_notes']
            id_notes_box.value = row['identification_notes']

            # Update dropdown
            idx = current_compound_index[0]
            if 0 <= idx < len(dropdown_options):
                temp_updating = updating[0]
                updating[0] = True
                try:
                    compound_dropdown.value = dropdown_options[idx]
                finally:
                    updating[0] = temp_updating

            # Update plot
            create_plot_only(initial_render=True)
        finally:
            updating[0] = False

    # --- 7. Connect event handlers ---
    prev_button.on_click(lambda b: on_navigation(-1))
    next_button.on_click(lambda b: on_navigation(1))
    ms2_prev_button.on_click(lambda b: on_ms2_navigation(-1))
    ms2_next_button.on_click(lambda b: on_ms2_navigation(1))
    accept_suggestions_button.on_click(on_accept_suggestions)
    ms2_radio.observe(on_ms2_annotation_change, names='value')
    ms1_radio.observe(on_ms1_annotation_change, names='value')
    analyst_notes_box.observe(on_analyst_notes_change, names='value')
    id_notes_box.observe(on_id_notes_change, names='value')
    compound_dropdown.observe(on_dropdown_change, names='value')

    # --- 8. Layout ---
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
    button_row = widgets.HBox([accept_suggestions_button], layout=widgets.Layout(justify_content='center', margin='0 0 0 0'))
    container = widgets.VBox([
        nav_box, 
        analyst_notes_box,
        id_notes_box,
        plot_and_radios,
        slider_container,
        button_row,
        identification_notes_display
    ], layout=widgets.Layout(width='100%', align_items='flex-start', height='fit-content'))

    # --- 9. Initialize ---
    full_update()
    logger.info(f"Manual curation GUI created with {len(compound_list)} compounds")
    return container