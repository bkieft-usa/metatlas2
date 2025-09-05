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

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import ms1_ms2_analysis as msa
import data_classes as dcl
import logging_config as lcf
import simple_cache as scache

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_gui')

def create_gui(compound_metadata, config, project_db_path=None):
    """
    Create an enhanced RT editor with simple cache auto-save capability.
    Now loads existing AnalystModifications from cache if available.
    """
    
    # Remove compounds that do not have EIC data or MS2 data (empty dicts)
    compound_metadata = {k: v for k, v in compound_metadata.items() if v.get('eic_data') or v.get('ms2_data')}

    # Set starting status
    file_color_dict = config['plot_settings']['file_color_mapping'] if 'file_color_mapping' in config['plot_settings'] else None
    compound_list = list(compound_metadata.keys())
    all_compound_names = [meta['original_atlas_data']['compound_name'] for meta in compound_metadata.values()]
    current_compound_index = [0]
    current_ms2_file_index = [0]
    updating = [False]  # Prevent recursive updates
    rt_slider = [None]  # Store slider reference
    
    # Initialize AnalystModifications - try to load from cache first
    analyst_modifications = _load_or_create_analyst_modifications(config, project_db_path)

    # Get project directory for caching
    project_dir = str(Path(project_db_path).parent) if project_db_path else None
    last_save_time = [time.time()]
    save_interval = 30  # seconds between auto-saves
    
    if not compound_list:
        logger.error("No compounds found in metadata")
        return None

    # Define annotation options
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

    # Create dropdown options: format as "index: compound_name" for clarity
    dropdown_options = []
    for i, compound_name in enumerate(all_compound_names):
        dropdown_options.append(f"{i+1}: {compound_name}")

    # Create main widgets with better height management
    plot_output = widgets.Output(layout=widgets.Layout(width='70%', height='600px'))
    prev_button = widgets.Button(description="◀ Prev", button_style='primary', layout=widgets.Layout(width='80px'))
    next_button = widgets.Button(description="Next ▶", button_style='primary', layout=widgets.Layout(width='80px'))
    counter_label = widgets.HTML(value=f"Compound 1 of {len(compound_list)}", layout=widgets.Layout(width='150px'))
    
    # Add compound dropdown for direct navigation
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

    # Create annotation radio buttons with compact layouts
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

    # Create placeholder for slider that will be replaced dynamically
    slider_container = widgets.HBox([], layout=widgets.Layout(justify_content='center', width='75%', margin='-110px 0 0 0'))

    # Add analyst notes text box
    analyst_notes_box = widgets.Text(
        value='',
        placeholder='Enter analyst notes...',
        description='Analyst Notes:',
        disabled=False,
        layout=widgets.Layout(width='100%', height='30px'),
        style={'description_width': '100px'}
    )

    # Add ID notes text box
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

    def get_current_rt_bounds(compound_metadata, inchi_key):
        """
        Get the current RT bounds for a compound, checking modifications first.
        """
        # Check if there are analyst modifications first
        rt_mods = analyst_modifications.get_rt_bounds(inchi_key)
        if rt_mods:
            return rt_mods['rt_min'], rt_mods['rt_max'], rt_mods['rt_peak']
        
        # Fall back to original atlas data
        original_data = compound_metadata['original_atlas_data']
        return original_data['rt_min'], original_data['rt_max'], original_data['rt_peak']

    def get_current_annotations(inchi_key):
        """Get current annotations, checking modifications first."""
        # Check modifications first
        annotation_mods = analyst_modifications.get_annotations(inchi_key)
        
        # Get original data as fallback
        original_data = compound_metadata[inchi_key]['original_atlas_data']
        
        return {
            'ms1_notes': annotation_mods.get('ms1_notes', original_data.get('ms1_notes', 'keep')),
            'ms2_notes': annotation_mods.get('ms2_notes', original_data.get('ms2_notes', 'no selection')),
            'analyst_notes': annotation_mods.get('analyst_notes', original_data.get('analyst_notes', '')),
            'identification_notes': annotation_mods.get('identification_notes', original_data.get('identification_notes', ''))
        }

    def write_ms2_title(compound_metadata, ms2_info, ms2_data_type, ms2_file, current_compound_index=None):
        """Create MS2 title and subheadings"""
        
        # Create MS2 title (exact same format as original)
        name = compound_metadata['original_atlas_data']['compound_name']
        # Use current_compound_index if provided, otherwise default to 1
        index = current_compound_index + 1 if current_compound_index is not None else 1
        heading = f"{name} {index}"

        if ms2_data_type == 'hits' and ms2_info is not None:
            # ms2_info is now the reference spectrum dict directly
            subheading1 = (
                f"Measured m/z: {ms2_info.get('mz_measured', 0.0):.4f}  |  Reference m/z: {ms2_info.get('mz_theoretical', 0.0):.4f}  |  "
                f"Reference DB: {ms2_info.get('database', 'N/A')}  |  Ref Matches: {ms2_info.get('num_matches', 0)}  |  Score: {ms2_info.get('score', 0.0):.4f}"
            )
        elif ms2_data_type == 'extracted' and ms2_info is not None:
            subheading1 = (
                f"Measured m/z: {ms2_info.get('precursor_mz', 0.0):.4f}  |  Theoretical m/z: {compound_metadata['original_atlas_data']['mz']:.4f}  |  "
                f"Reference DB: N/A  |  Ref Matches: N/A  |  Score: N/A"
            )
        else:
            # Handle case when there's no MS2 data at all
            subheading1 = 'No MS2 data available'

        if ms2_file:
            subheading2 = f'File: {ms2_file}'
        else:
            subheading2 = 'No file data'

        ms2_title = f'{heading}<br><sub>{subheading1}</sub><br><sub>{subheading2}</sub>'

        return ms2_title

    def write_eic_title(compound_metadata, current_compound_index=None):
        # Create EIC title (exact same format as original)
        eic_title_parts = []
        
        # Add compound index to the beginning if provided
        if current_compound_index is not None:
            name = compound_metadata['original_atlas_data']['compound_name']
            index = current_compound_index + 1
            eic_title_parts.append(f'{name} {index}')
        
        # Basic info
        if compound_metadata['original_atlas_data']['adduct']:
            eic_title_parts.append(f'Adduct: {compound_metadata["original_atlas_data"]["adduct"]}')
        if compound_metadata['original_atlas_data']['inchi_key']:
            eic_title_parts.append(f'InChI Key: {compound_metadata["original_atlas_data"]["inchi_key"]}')
        eic_title_parts.append(f'LCMS files: {len(compound_metadata["eic_data"])}')
        subheading1 = ' | '.join(eic_title_parts)

        # RT and m/z comparison using CURRENT RT values
        rt_parts = [f'Atlas RT: {compound_metadata["original_atlas_data"]["rt_peak"]:.2f}']

        best_rt = compound_metadata['best_eic']['rt_peak']
        if best_rt is not None:
            # Recalculate RT diff using current RT peak
            current_rt_diff = compound_metadata["original_atlas_data"]["rt_peak"] - best_rt
            rt_parts.extend([
                f'Measured RT: {best_rt:.2f} min',
                f'RT diff: {current_rt_diff:.2f} min'
            ])
        else:
            rt_parts.extend(['Measured RT: N/A', 'RT diff: N/A'])
        
        rt_parts.append(f'Atlas m/z: {compound_metadata["original_atlas_data"]["mz"]:.4f}')
        
        best_mz = compound_metadata['best_eic']['mz_peak']
        if best_mz is not None:
            ppm_diff = compound_metadata['best_eic']['ppm_diff']
            rt_parts.extend([
                f'Measured m/z: {best_mz:.4f}',
                f'ppm diff: {ppm_diff:.2f}' if ppm_diff is not None else 'ppm diff: N/A'
            ])
        else:
            rt_parts.extend(['Measured m/z: N/A', 'ppm diff: N/A'])
        
        subheading2 = '   |   '.join(rt_parts)

        # Isomers
        isomers = compound_metadata['original_atlas_data']['isomers']
        isomer_strings = []
        for isomer in isomers:
            name = isomer.get('compound_name', '')
            isomer_inchi_key = isomer.get('inchi_key', '')
            rt_pk = isomer.get('rt', '')
            isomer_strings.append(f"{name} {rt_pk:.2f}")
        
        subheading3 = f'Isomers: {", ".join(isomer_strings)}' if isomer_strings else 'No isomers'

        eic_title = f'<sub>{subheading1}</sub><br><sub>{subheading2}</sub><br><sub>{subheading3}</sub>'

        return eic_title

    def create_targeted_analysis_plot(compound_metadata, 
                                      height=600, file_color_dict=None, 
                                      ms2_data_type=None, ms2_plot_data=None,
                                      initial_render=False, current_compound_index=None):
        """
        Create the plot using pre-calculated metadata and selected MS2 spectra.
        Args:
            ms2_plot_data: tuple (has_hits, has_data, selected_query, selected_ref)
            initial_render: if True, update EIC x-axis bounds based on RT bounds
        """

        current_rt_min, current_rt_max, current_rt_peak = get_current_rt_bounds(compound_metadata, compound_metadata['original_atlas_data']['inchi_key'])
        
        # Only update x-axis bounds on initial render (compound navigation)
        # Otherwise, keep the existing x-axis bounds to maintain slider alignment
        if initial_render:
            x_min = current_rt_min - 1.0
            x_max = current_rt_max + 1.0
            # Store the x-axis bounds in metadata for consistency
            compound_metadata['_plot_x_bounds'] = (x_min, x_max)
        else:
            # Use stored bounds if available, otherwise calculate from current RT bounds
            if '_plot_x_bounds' in compound_metadata:
                x_min, x_max = compound_metadata['_plot_x_bounds']
            else:
                x_min = current_rt_min - 1.0
                x_max = current_rt_max + 1.0
                compound_metadata['_plot_x_bounds'] = (x_min, x_max)

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.4, 0.6],
            vertical_spacing=0.35,  # Increased from 0.3 to add more space between MS2 and EIC plots
            subplot_titles=('', ''),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        eic_row = 2
        ms2_row = 1

        # MS2 plot logic
        selected_query, selected_ref, selected_file = ms2_plot_data
        exp_max_intensity = 0
        ref_max_intensity = 0
        ms2_title = ""

        # Handle case when there's no MS2 data at all
        if ms2_data_type is None or (selected_query is None and selected_ref is None):
            # Create empty MS2 plot with proper structure
            ms2_title = write_ms2_title(compound_metadata, None, None, None)
            pmz = 0.0
            max_mz = 500.0
            # Add an invisible trace to properly initialize the subplot
            fig.add_trace(go.Scatter(
                x=[0, max_mz],
                y=[0, 0],
                mode='lines',
                line=dict(color='rgba(0,0,0,0)', width=0),  # Completely transparent
                showlegend=False,
                hoverinfo='skip',
                name='Empty'
            ), row=ms2_row, col=1)
            
            # Set y-range for MS2 plot - handle empty plot case
            y_range = [-1000, 1000]
        elif ms2_data_type == 'hits':
            # Plot query (top) with hover info
            mz_vals = selected_query.get('mz', [])
            pmz = selected_query.get('precursor_mz', 0.0)
            mz_colors = selected_query.get('qry_frag_colors', [])
            intensity_vals = selected_query.get('intensity', [])

            if len(mz_vals) > 0 and len(intensity_vals) > 0:
                max_mz = np.max(mz_vals)
                exp_max_intensity = max(exp_max_intensity, np.max(intensity_vals))
                
                # Create stick plot for experimental spectrum with per-stick color
                for mz, intensity, color in zip(mz_vals, intensity_vals, mz_colors):
                    fig.add_trace(go.Scatter(
                        x=[mz, mz],
                        y=[0, intensity],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False, hoverinfo='skip', name='Experimental'
                    ), row=ms2_row, col=1)
                
                # Add invisible markers at peak tops for hover interaction
                fig.add_trace(go.Scatter(
                    x=list(mz_vals), y=list(intensity_vals),
                    mode='markers',
                    marker=dict(size=1, color="black", opacity=0.01),
                    showlegend=False, name='Experimental Peaks',
                    hovertemplate='m/z: %{x:.4f}<br>Intensity: %{y:.0f}<extra>Experimental</extra>'
                ), row=ms2_row, col=1)
            
            # Plot reference (bottom) with hover info
            mz_vals = selected_ref.get('mz', [])
            mz_colors = selected_ref.get('qry_frag_colors', [])
            intensity_vals = selected_ref.get('intensity', [])
            if len(mz_vals) > 0 and len(intensity_vals) > 0:
                max_mz = np.max(mz_vals)
                ref_max_intensity = max(ref_max_intensity, np.max(intensity_vals))
                # Scale reference intensities to match experimental scale
                if exp_max_intensity > 0 and ref_max_intensity > 0:
                    scaled_intensities = [i * (exp_max_intensity / ref_max_intensity) for i in intensity_vals]
                else:
                    scaled_intensities = intensity_vals

                # Create stick plot for reference spectrum with per-stick color
                for mz, scaled_int, color in zip(mz_vals, scaled_intensities, mz_colors):
                    fig.add_trace(go.Scatter(
                        x=[mz, mz],
                        y=[0, -scaled_int],  # Negative for downward
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False, hoverinfo='skip', name='Reference'
                    ), row=ms2_row, col=1)

                # Add invisible markers at peak tops for hover interaction
                fig.add_trace(go.Scatter(
                    x=list(mz_vals), y=[-s for s in scaled_intensities],
                    mode='markers',
                    marker=dict(size=1, color='red', opacity=0.01),
                    showlegend=False, name='Reference Peaks',
                    hovertemplate='m/z: %{x:.4f}<br>Intensity: %{customdata:.0f}<extra>Reference</extra>',
                    customdata=list(intensity_vals)  # Use original intensities for hover
                ), row=ms2_row, col=1)
            
            ms2_title = write_ms2_title(compound_metadata, selected_ref, ms2_data_type, selected_file)
            # Set y-range for MS2 plot to show both positive and negative
            if exp_max_intensity > 0:
                y_range = [-exp_max_intensity * 1.1, exp_max_intensity * 1.1]
            else:
                y_range = [-1000, 1000]
        elif ms2_data_type == 'extracted':
            # Plot only experimental spectrum (top) with hover info
            mz_vals = selected_query.get('mz', [])
            pmz = selected_query.get('precursor_mz', 0.0)
            intensity_vals = selected_query.get('intensity', [])
            colors = ['red'] * len(mz_vals)
            
            if len(mz_vals) > 0 and len(intensity_vals) > 0:
                max_mz = np.max(mz_vals)
                exp_max_intensity = max(exp_max_intensity, np.max(intensity_vals))
                
                # Create stick plot for experimental spectrum with per-stick color
                for mz, intensity, color in zip(mz_vals, intensity_vals, colors):
                    fig.add_trace(go.Scatter(
                        x=[mz, mz],
                        y=[0, intensity],
                        mode='lines',
                        line=dict(color=color, width=2),
                        showlegend=False, hoverinfo='skip', name='Experimental'
                    ), row=ms2_row, col=1)
                
                # Add invisible markers at peak tops for hover interaction
                fig.add_trace(go.Scatter(
                    x=list(mz_vals), y=list(intensity_vals),
                    mode='markers',
                    marker=dict(size=1, color="red", opacity=0.01),
                    showlegend=False, name='Experimental Peaks',
                    hovertemplate='m/z: %{x:.4f}<br>Intensity: %{y:.0f}<extra>Experimental</extra>'
                ), row=ms2_row, col=1)
            
            ms2_title = write_ms2_title(compound_metadata, selected_query, ms2_data_type, selected_file)
            # Set y-range for MS2 plot to start at 0 and go to max intensity
            if exp_max_intensity > 0:
                y_range = [0, exp_max_intensity * 1.1]
            else:
                y_range = [0, 1000]
        else:
            raise ValueError(f"Unknown MS2 data type: {ms2_data_type}")

        # Configure MS2 subplot
        fig.update_xaxes(
            title_text="m/z",
            showgrid=False,
            range=[0, max_mz*1.1] if 'max_mz' in locals() and max_mz > 0 else [0, 500],
            row=ms2_row, col=1
        )
        
        fig.update_yaxes(
            title_text="Intensity",
            showgrid=False,
            range=y_range,
            row=ms2_row, col=1,
            exponentformat="e",  # Use scientific notation
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

        for file_name, trace_data in compound_metadata['eic_data'].items():
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
        fig.add_vline(x=compound_metadata['original_atlas_data']['rt_peak'], line_dash="dash", line_color="gray", line_width=3, row=eic_row, col=1)
        fig.add_vline(x=compound_metadata['original_atlas_data']['rt_min'], line_dash="dot", line_color="gray", line_width=2, row=eic_row, col=1)
        fig.add_vline(x=compound_metadata['original_atlas_data']['rt_max'], line_dash="dot", line_color="gray", line_width=2, row=eic_row, col=1)

        # Add suggested RT bounds as two light green vlines if available
        suggested_data = compound_metadata.get('suggested_rt_bounds_data', None)
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
        
        eic_title = write_eic_title(compound_metadata, current_compound_index)

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
            # # MS2 plot buttons (top left)
            # dict(
            #     type="buttons", direction="right",
            #     buttons=[
            #         dict(args=[{f"yaxis{ms2_row}.type": "linear", f"yaxis{ms2_row}.autorange": True}], 
            #                 label="Lin", method="relayout"),
            #         dict(args=[{f"yaxis{ms2_row}.type": "log", f"yaxis{ms2_row}.autorange": True}], 
            #                 label="Log", method="relayout")
            #     ],
            #     x=0.02, xanchor="left", y=0.65, yanchor="bottom",  # Positioned for MS2 plot
            #     bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1
            # )
        ]
        
        fig.update_layout(updatemenus=updatemenus)

        return fig

    def auto_save_if_needed():
        """Auto-save AnalystModifications if changes have been made and enough time has passed."""
        if (project_dir and 
            len(analyst_modifications.modified_compounds) > 0 and
            time.time() - last_save_time[0] > save_interval):
            
            try:
                # Save current AnalystModifications to progress cache
                scache.save_gui_cache(container, project_dir, "progress")
                last_save_time[0] = time.time()
                logger.info("Auto-saved AnalystModifications to progress cache")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def create_plot_only(initial_render=False):
        """
        ONLY create and display the plot - no other side effects.
        """
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return

        plot_output.clear_output(wait=True)

        # Get RT bounds for MS2 filtering
        rt_min, rt_max, rt_peak = get_current_rt_bounds(meta, compound_inchi)

        # Get MS2 selection info (now filtered by RT bounds and sorted by score)
        ms2_data_type, query_spec, ref_spec, ms2_file = get_selected_ms2_spectra(meta, 
                                                                                current_ms2_file_index[0],
                                                                                rt_min,
                                                                                rt_max)

        # Update MS2 counter label - count only files within RT window
        ms2_files_data = meta.get('ms2_data', {})
        files_in_rt_window = []
        for file_name in ms2_files_data.keys():
            file_data = ms2_files_data[file_name]
            best_hit = file_data.get('best_hit', {})
            best_ms2 = file_data.get('best_ms2', {})
            
            rt_measured = None
            if best_hit and 'rt_measured' in best_hit:
                rt_measured = best_hit.get('rt_measured', 0)
            elif best_ms2 and 'rt' in best_ms2:
                rt_measured = best_ms2.get('rt', 0)
            
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                files_in_rt_window.append(file_name)
        
        total_files_in_window = len(files_in_rt_window)
        if total_files_in_window > 0:
            current_file_idx = min(current_ms2_file_index[0], total_files_in_window - 1)
            ms2_counter_label.value = f"MS2 File {current_file_idx + 1} of {total_files_in_window}"
        else:
            ms2_counter_label.value = "No MS2 data"

        fig = create_targeted_analysis_plot(
            compound_metadata=meta,
            file_color_dict=file_color_dict,
            ms2_data_type=ms2_data_type,
            ms2_plot_data=(query_spec, ref_spec, ms2_file),
            initial_render=initial_render,
            current_compound_index=current_compound_index[0]
        )

        if fig is not None:
            with plot_output:
                fig.show()

    def full_update():
        """
        Perform a complete update of all UI elements.
        This is the main orchestrator function.
        """
        if updating[0]:
            return
            
        updating[0] = True
        try:
            compound_inchi, compound_name, meta = get_current_compound()
            if compound_inchi is None or meta is None:
                return
            
            rt_min, rt_max, rt_peak = get_current_rt_bounds(meta, compound_inchi)
            
            # Calculate dynamic slider bounds based on compound's RT range
            rt_range = rt_max - rt_min
            slider_min = rt_min - 1
            slider_max = rt_max + 1
            slider_step = max(0.01, rt_range / 100)  # 1% of range or 0.01 min minimum
            
            # Create new RT slider with dynamic bounds
            new_slider = widgets.FloatRangeSlider(
                value=(rt_min, rt_max),
                min=slider_min,
                max=slider_max,
                step=slider_step,
                description='RT',
                continuous_update=True,
                layout=widgets.Layout(width='95%'),
                slider_style='continuous',
                readout=True
            )
            
            # Attach observer to new slider
            new_slider.observe(on_rt_slider_change, names='value')
            
            # Replace slider in container
            slider_container.children = [new_slider]
            rt_slider[0] = new_slider  # Store reference
            
            # Update all UI elements WITHOUT any observers active
            update_annotation_widgets()
            update_dropdown_value()  # Update dropdown to match current compound
            create_plot_only(initial_render=True)  # Use initial_render=True for compound navigation
            
        finally:
            updating[0] = False

    def on_navigation(direction):
        """Handle compound navigation"""
        if updating[0]:
            return

        idx = current_compound_index[0] + direction
        if 0 <= idx < len(compound_list):
            current_compound_index[0] = idx
            counter_label.value = f"Compound {idx+1} of {len(compound_list)}"

            # Reset MS2 file index when switching compounds
            reset_ms2_file_index()

            # Do a full update
            full_update()
            
            # Trigger auto-save check
            auto_save_if_needed()

    def on_ms2_navigation(direction):
        """Handle MS2 file navigation (filtered by current RT bounds and sorted by score)"""
        if updating[0]:
            return
            
        compound_inchi, _, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return

        # Get current RT bounds for filtering
        rt_min, rt_max, rt_peak = get_current_rt_bounds(meta, compound_inchi)
        
        # Count files within current RT window
        ms2_files_data = meta.get('ms2_data', {})
        files_in_rt_window = []
        for file_name in ms2_files_data.keys():
            file_data = ms2_files_data[file_name]
            best_hit = file_data.get('best_hit', {})
            best_ms2 = file_data.get('best_ms2', {})
            
            rt_measured = None
            if best_hit and 'rt_measured' in best_hit:
                rt_measured = best_hit.get('rt_measured', 0)
            elif best_ms2 and 'rt' in best_ms2:
                rt_measured = best_ms2.get('rt', 0)
            
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                files_in_rt_window.append(file_name)
        
        total_files_in_window = len(files_in_rt_window)
        if total_files_in_window == 0:
            return

        new_idx = current_ms2_file_index[0] + direction
        # Bounds checking for files within RT window only
        if 0 <= new_idx < total_files_in_window:
            current_ms2_file_index[0] = new_idx
            create_plot_only(initial_render=False)  # Don't update x-axis bounds for MS2 navigation

    def on_reset(button):
        """Reset current compound to original RT bounds and annotations"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None:
            return
        
        updating[0] = True
        try:
            # Reset all modifications for this compound
            analyst_modifications.reset_compound(compound_inchi)
            
            # Get original values for slider update
            original_rt_min = meta['original_atlas_data']['rt_min']
            original_rt_max = meta['original_atlas_data']['rt_max']
            
            # Reset MS2 file index since RT bounds changed
            reset_ms2_file_index()
            
            # Update all UI elements (this will recreate the slider with correct bounds)
            full_update()
            
        finally:
            updating[0] = False

    def update_dropdown_value():
        """Update dropdown to reflect current compound without triggering observer"""
        idx = current_compound_index[0]
        if 0 <= idx < len(dropdown_options):
            # Temporarily disable updating to prevent recursion
            temp_updating = updating[0]
            updating[0] = True
            try:
                compound_dropdown.value = dropdown_options[idx]
            finally:
                updating[0] = temp_updating

    def on_dropdown_change(change):
        """Handle dropdown selection changes"""
        if updating[0]:
            return
        
        # Extract index from dropdown value (format: "index: compound_name")
        try:
            selected_value = change['new']
            idx = int(selected_value.split(':')[0]) - 1  # Convert from 1-based to 0-based index
            
            if 0 <= idx < len(compound_list) and idx != current_compound_index[0]:
                current_compound_index[0] = idx
                counter_label.value = f"Compound {idx+1} of {len(compound_list)}"
                
                # Reset MS2 file index when switching compounds
                reset_ms2_file_index()
                
                # Do a full update
                full_update()
                
                # Trigger auto-save check
                auto_save_if_needed()
                
        except (ValueError, IndexError):
            # If parsing fails, ignore the change
            pass

    def on_accept_suggestions(button):
        """Accept suggested RT bounds for current compound"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        suggested_data = meta.get('suggested_rt_bounds_data', None)
        if suggested_data is None:
            return
        
        updating[0] = True
        try:
            # Reset MS2 file index since RT bounds will change
            reset_ms2_file_index()
            
            # Update using AnalystModifications
            analyst_modifications.update_rt_bounds(
                compound_inchi,
                suggested_data['rt_min'],
                suggested_data['rt_max'],
                suggested_data['rt_peak']
            )

            # Update all UI elements
            create_plot_only()
            
            # Update slider to match accepted suggestions
            if rt_slider[0] is not None:
                rt_slider[0].value = (suggested_data['rt_min'], suggested_data['rt_max'])
            
        finally:
            updating[0] = False

    def on_rt_slider_change(change):
        """Callback that fires whenever the user drags the slider."""
        if updating[0]:
            return
        
        compound_inchi, compound_name, meta = get_current_compound()
        if meta is None:
            return

        new_min, new_max = change['new']
        current_rt_min, current_rt_max, current_rt_peak = get_current_rt_bounds(meta, compound_inchi)
        
        tolerance = 1e-6
        if (abs(new_min - current_rt_min) < tolerance and 
            abs(new_max - current_rt_max) < tolerance):
            return

        # Calculate new peak as midpoint
        new_peak = new_min + (new_max - new_min) / 2.0
        
        # Update using AnalystModifications
        analyst_modifications.update_rt_bounds(compound_inchi, new_min, new_max, new_peak)

        # Reset MS2 file index since RT bounds changed
        reset_ms2_file_index()
        
        create_plot_only()

        # Trigger auto-save check
        auto_save_if_needed()

    def get_current_compound():
        """Get current compound name and metadata"""
        idx = current_compound_index[0]
        if 0 <= idx < len(compound_list):
            compound_inchi = compound_list[idx]
            compound_name = compound_metadata[compound_inchi]['original_atlas_data']['compound_name']
            return compound_inchi, compound_name, compound_metadata[compound_inchi]
        return None, None, None
    
    def reset_ms2_file_index():
        """Reset MS2 file index to 0 when switching compounds"""
        current_ms2_file_index[0] = 0

    def get_selected_ms2_spectra(meta, current_ms2_file_index, rt_min, rt_max):
        """
        Returns the selected MS2 spectra for plotting and info for the title.
        Updated to work with new metadata structure and properly filter by RT bounds.
        """
        # Get MS2 files data using new structure
        ms2_files_data = meta.get('ms2_data', {})
        if not ms2_files_data:
            return None, None, None, None
        
        # Get file names and filter by RT bounds FIRST
        file_names = list(ms2_files_data.keys())
        if not file_names:
            return None, None, None, None
        
        # Filter files that have MS2 data within RT bounds and calculate scores
        files_in_rt_window = []
        for file_name in file_names:
            file_data = ms2_files_data[file_name]
            
            # Check both best_hit and best_ms2 for RT data
            best_hit = file_data.get('best_hit', {})
            best_ms2 = file_data.get('best_ms2', {})
            
            rt_measured = None
            if best_hit and 'rt_measured' in best_hit:
                rt_measured = best_hit.get('rt_measured', 0)
            elif best_ms2 and 'rt' in best_ms2:
                rt_measured = best_ms2.get('rt', 0)
            
            # Only include files where MS2 data falls within current RT bounds
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                # Prioritize score from best_hit, fallback to intensity from best_ms2
                if best_hit:
                    score = best_hit.get('score', 0.0)
                else:
                    score = best_ms2.get('intensity_peak', 0.0)
                files_in_rt_window.append((file_name, score))
        
        # If no files have MS2 data in the current RT window, return None
        if not files_in_rt_window:
            return None, None, None, None
        
        # Sort files by score (highest first)
        files_in_rt_window.sort(key=lambda x: x[1], reverse=True)
        sorted_file_names = [name for name, score in files_in_rt_window]
        
        # Clamp file index to available files within RT window
        idx = min(current_ms2_file_index, len(sorted_file_names) - 1)
        selected_file = sorted_file_names[idx]
        file_data = ms2_files_data[selected_file]
        
        # Check if we have a best_hit (with reference data)
        best_hit = file_data.get('best_hit', {})
        if best_hit:  # best_hit is not empty
            # Extract spectrum data from best_hit
            qry_spectrum = best_hit.get('qry_spectrum', None)
            ref_spectrum = best_hit.get('ref_spectrum', None)
            
            # Check if we have any spectrum data
            if qry_spectrum is None and ref_spectrum is None:
                return None, None, None, selected_file
            
            query_spec = None
            ref_spec = None
            
            # Process query spectrum if available
            if qry_spectrum is not None and len(qry_spectrum) >= 2:
                qry_mz = np.array(qry_spectrum[0])
                qry_intensity = np.array(qry_spectrum[1])

                # Get fragment colors (default to red if not available)
                qry_colors = best_hit.get('qry_frag_colors', ['red'] * len(qry_mz))
                
                query_spec = {
                    'mz': qry_mz,
                    'intensity': qry_intensity,
                    'precursor_mz': best_hit.get('mz_measured', 0.0),
                    'rt': best_hit.get('rt_measured', 0.0),
                    'qry_frag_colors': qry_colors
                }
            
            # Process reference spectrum if available
            if ref_spectrum is not None and len(ref_spectrum) >= 2:
                ref_mz = np.array(ref_spectrum[0])
                ref_intensity = np.array(ref_spectrum[1])
                
                # Get fragment colors (use same as query if available, otherwise red)
                ref_colors = best_hit.get('qry_frag_colors', ['red'] * len(ref_mz))
                
                ref_spec = {
                    'mz': ref_mz,
                    'intensity': ref_intensity,
                    'score': best_hit.get('score', 0.0),
                    'database': best_hit.get('database', 'N/A'),
                    'num_matches': best_hit.get('num_matches', 0),
                    'ref_id': best_hit.get('ref_id', 'N/A'),
                    'mz_measured': best_hit.get('mz_measured', 0.0),
                    'mz_theoretical': best_hit.get('mz_theoretical', 0.0),
                    'rt_measured': best_hit.get('rt_measured', 0.0),
                    'qry_frag_colors': ref_colors
                }
            
            # Determine data type
            if query_spec is not None and ref_spec is not None:
                return 'hits', query_spec, ref_spec, selected_file
            elif query_spec is not None:
                return 'extracted', query_spec, None, selected_file
            else:
                return None, None, None, selected_file
                
        else:  # best_hit is empty, use best_ms2
            best_ms2 = file_data.get('best_ms2', {})
            if not best_ms2:
                return None, None, None, selected_file
            
            # Extract spectrum data from best_ms2
            spectrum_data = best_ms2.get('spectrum', None)
            if spectrum_data is None or len(spectrum_data) < 2:
                return None, None, None, selected_file
            
            # Parse spectrum format
            try:
                if isinstance(spectrum_data, (list, tuple)) and len(spectrum_data) == 2:
                    mz_values = np.array(spectrum_data[0])
                    intensity_values = np.array(spectrum_data[1])
                elif isinstance(spectrum_data, np.ndarray) and spectrum_data.shape[0] == 2:
                    mz_values = np.array(spectrum_data[0])
                    intensity_values = np.array(spectrum_data[1])
                else:
                    return None, None, None, selected_file
            except:
                return None, None, None, selected_file
            
            if len(mz_values) == 0 or len(intensity_values) == 0:
                return None, None, None, selected_file
            
            # Create query spec for experimental data only
            query_spec = {
                'mz': mz_values,
                'intensity': intensity_values,
                'precursor_mz': best_ms2.get('precursor_mz', 0.0),
                'rt': best_ms2.get('rt', 0.0),
                'qry_frag_colors': ['red'] * len(mz_values)  # All red for experimental-only
            }
            
            return 'extracted', query_spec, None, selected_file

    def update_annotation_widgets():
        """Update annotation radio buttons to reflect current compound's values"""
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        # Get current annotations (from modifications or original data)
        current_annotations = get_current_annotations(compound_inchi)

        # Temporarily disable updating flag to prevent recursion
        temp_updating = updating[0]
        updating[0] = True
        
        try:
            ms2_radio.value = current_annotations['ms2_notes']
            ms1_radio.value = current_annotations['ms1_notes']
            analyst_notes_box.value = current_annotations['analyst_notes']
            id_notes_box.value = current_annotations['identification_notes']
        finally:
            updating[0] = temp_updating

    def on_ms2_annotation_change(change):
        """Handle MS2 annotation changes"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
                
        # Update using AnalystModifications
        analyst_modifications.update_annotations(compound_inchi, ms2_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_ms1_annotation_change(change):
        """Handle MS1 annotation changes"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        # Update using AnalystModifications
        analyst_modifications.update_annotations(compound_inchi, ms1_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_analyst_notes_change(change):
        """Handle analyst notes changes"""
        if updating[0]:
            return
            
        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return
        
        # Update using AnalystModifications
        analyst_modifications.update_annotations(compound_inchi, analyst_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_id_notes_change(change):
        """Handle ID notes changes"""
        if updating[0]:
            return

        compound_inchi, compound_name, meta = get_current_compound()
        if compound_inchi is None or meta is None:
            return

        # Update using AnalystModifications
        analyst_modifications.update_annotations(compound_inchi, identification_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()
    
    # Add manual save button
    manual_save_button = widgets.Button(
        description="Save Session", 
        button_style='info', 
        layout=widgets.Layout(width='150px')
    )
    
    def on_manual_save(button):
        """Manual save handler for AnalystModifications"""
        if project_dir:
            try:
                # Save current AnalystModifications to progress cache
                timestamp = scache.save_gui_cache(container, project_dir, "progress")
                button.description = f"Saved {timestamp[-8:-3]}"
                last_save_time[0] = time.time()
                
                # Reset button text after 2 seconds
                import threading
                def reset_button():
                    time.sleep(2)
                    button.description = "Save Session"
                threading.Thread(target=reset_button).start()
            except Exception as e:
                logger.error(f"Manual save failed: {e}")
                button.description = "Save Failed"
    
    manual_save_button.on_click(on_manual_save)

    # Connect all event handlers to their widgets
    prev_button.on_click(lambda b: on_navigation(-1))
    next_button.on_click(lambda b: on_navigation(1))
    ms2_prev_button.on_click(lambda b: on_ms2_navigation(-1))
    ms2_next_button.on_click(lambda b: on_ms2_navigation(1))
    reset_button.on_click(on_reset)
    accept_suggestions_button.on_click(on_accept_suggestions)
    
    # Connect annotation widgets to their handlers
    ms2_radio.observe(on_ms2_annotation_change, names='value')
    ms1_radio.observe(on_ms1_annotation_change, names='value')
    analyst_notes_box.observe(on_analyst_notes_change, names='value')
    id_notes_box.observe(on_id_notes_change, names='value')
    compound_dropdown.observe(on_dropdown_change, names='value')
    
    # Create layout structure
    nav_row = widgets.HBox([prev_button, counter_label, next_button], layout=widgets.Layout(justify_content='center'))
    
    # Add dropdown to navigation - put it in its own row for better layout
    dropdown_row = widgets.HBox([compound_dropdown], layout=widgets.Layout(justify_content='center', margin='5px 0'))
    
    # MS2 browser row
    ms2_nav_row = widgets.HBox([ms2_prev_button, ms2_counter_label, ms2_next_button], layout=widgets.Layout(justify_content='center'))

    # Combine all navigation elements horizontally
    nav_box = widgets.HBox([
        nav_row,
        dropdown_row,
        ms2_nav_row
    ], layout=widgets.Layout(width='100%', align_items='center'))
        
    # Create the main plot and radio button container
    plot_and_radios = widgets.HBox([
        plot_output,
        widgets.VBox([
            widgets.Box([ms2_radio], layout=widgets.Layout(margin='50px 0 50px 0')),
            ms1_radio
        ], layout=widgets.Layout(width='30%', justify_content='flex-start'))
    ], layout=widgets.Layout(width='100%'))
    
    # Create metadata rows
    button_row = widgets.HBox([reset_button, accept_suggestions_button, manual_save_button], 
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
    
    # Add helper methods
    container.metadata = compound_metadata

    # Initialize with the first compound
    full_update()
    
    # Add method to access modifications from the container
    container.get_modifications = lambda: analyst_modifications
    container.get_plot_data = lambda: analyst_modifications.to_plot_data_format(compound_metadata)
    
    # Store project data reference in container for auto-save
    container._project_data = None  # Will be set externally
    
    return container

def _load_or_create_analyst_modifications(config: Dict, project_db_path: Optional[str]) -> dcl.AnalystModifications:
    """
    Load existing AnalystModifications from cache or create new instance.
    
    Args:
        config: Configuration dictionary
        project_db_path: Path to project database
        
    Returns:
        AnalystModifications instance (either loaded from cache or new)
    """
    # Check for session cache setting
    use_session_cache = config.get('analysis_settings', {}).get('use_session_cache', False)
    
    if use_session_cache is not False and project_db_path:
        logger.info("Attempting to load existing AnalystModifications from session cache...")
        project_dir = str(Path(project_db_path).parent)
        
        try:
            # Try to load from progress cache first (most recent session)
            cached_gui = scache.load_gui_cache(project_dir, use_session_cache, "progress")
            if cached_gui is not None:
                logger.info(f"Loaded AnalystModifications from progress cache with {len(cached_gui.get_modifications().modified_compounds)} modified compounds")
                return cached_gui.get_modifications()
            
            # Fallback to complete cache if no progress cache
            cached_gui = scache.load_gui_cache(project_dir, use_session_cache, "complete")
            if cached_gui is not None:
                logger.info(f"Loaded AnalystModifications from complete cache with {len(cached_gui.get_modifications().modified_compounds)} modified compounds")
                return cached_gui.get_modifications()
                
        except Exception as e:
            logger.warning(f"Failed to load AnalystModifications from cache: {e}")
    
    # Create new instance if no cache or cache loading failed
    logger.info("Creating new AnalystModifications instance")
    return dcl.AnalystModifications()