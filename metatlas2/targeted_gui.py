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

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import ms1_ms2_analysis as msa
import workflow_objects as wfo
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('targeted_gui')

# def create_gui_from_analysis_projects(analysis_projects: List[Tuple], config: Dict, project_dir: str = None):
#     """
#     Create an enhanced RT editor working with AnalysisProject objects that contain full experimental data.
    
#     Args:
#         analysis_projects: List of tuples (atlas_type, chrom_pol, AnalysisProject)
#         config: Configuration dictionary
#         project_dir: Project directory for caching
#     """
    
#     if not analysis_projects:
#         logger.error("No analysis projects provided")
#         return None
    
#     # Flatten all compounds from all analysis projects
#     all_compounds = []
#     for atlas_type, chrom_pol, analysis_project in analysis_projects:
#         for inchi_key, compound in analysis_project.compounds.items():
#             # Add metadata to track which atlas/method this compound came from
#             compound._atlas_type = atlas_type
#             compound._chromatography_polarity = chrom_pol
#             all_compounds.append(compound)
    
#     if not all_compounds:
#         logger.error("No compounds found in analysis projects")
#         return None
    
#     logger.info(f"Creating GUI with {len(all_compounds)} compounds from {len(analysis_projects)} analysis projects")
    
#     # Use the existing create_gui function but with compound objects that have full data
#     return create_gui_with_compounds(all_compounds, config, project_dir)

def create_gui_with_compounds(compounds: List, config: Dict, analysis_dir: str = None):
    """
    Create an enhanced RT editor working directly with CompoundExperimental objects.
    
    Args:
        compounds: List of CompoundExperimental objects with full experimental data
        config: Configuration dictionary
        analysis_dir: Project directory for caching
    """
    
    if not compounds:
        logger.error("No compounds provided")
        return None
    
    # Set starting status
    file_color_dict = config['plot_settings']['file_color_mapping'] if 'file_color_mapping' in config['plot_settings'] else None
    
    compound_list = list(range(len(compounds)))  # Use indices
    all_compound_names = [compound.compound_name for compound in compounds]
    
    def get_current_compound_func():
        idx = current_compound_index[0]
        if 0 <= idx < len(compounds):
            return compounds[idx]
        return None

    # Set starting status
    current_compound_index = [0]
    current_ms2_file_index = [0]
    updating = [False]  # Prevent recursive updates
    rt_slider = [None]  # Store slider reference
    
    # Auto-save tracking
    last_save_time = [time.time()]
    save_interval = 30  # seconds between auto-saves
    
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
    slider_container = widgets.HBox([], layout=widgets.Layout(justify_content='center', width='71%', margin='-110px 0 0 0'))

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

    def get_current_compound() -> Optional[wfo.CompoundExperimental]:
        """Get current CompoundExperimental object"""
        return get_current_compound_func()

    def auto_save_if_needed():
        """Auto-save compound changes if enough time has passed."""
        if (analysis_dir and 
            any(compound.is_rt_modified or compound.is_annotation_modified for compound in compounds) and
            time.time() - last_save_time[0] > save_interval):
            
            try:
                # Save compounds state to cache
                save_compounds_progress(compounds, analysis_dir)
                last_save_time[0] = time.time()
                logger.info("Auto-saved compounds progress to cache")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")

    def save_compounds_progress(compounds, analysis_dir):
        """Save current compounds state to cache"""
        try:
            cache_dir = Path(analysis_dir) / "cache" / "compounds_progress"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_file = cache_dir / f"compounds_{timestamp}.pkl"
            latest_file = cache_dir / "compounds_latest.pkl"
            
            # Save timestamped version
            with open(cache_file, 'wb') as f:
                pickle.dump(compounds, f)
            
            # Update latest symlink
            if latest_file.exists():
                latest_file.unlink()
            latest_file.symlink_to(cache_file.name)
            
            logger.info(f"Saved compounds progress: {cache_file}")
            return timestamp
        except Exception as e:
            logger.error(f"Failed to save compounds progress: {e}")
            return None

    def write_ms2_title(compound: wfo.CompoundExperimental, ms2_info, has_reference, ms2_file, current_compound_index=None):
        """Create MS2 title and subheadings"""
        
        # Create MS2 title (exact same format as original)
        name = compound.compound_name
        # Use current_compound_index if provided, otherwise default to 1
        index = current_compound_index + 1 if current_compound_index is not None else 1
        heading = f"{name} {index}"

        if has_reference is True and ms2_info is not None:
            # ms2_info is now the reference spectrum dict directly
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
            # Handle case when there's no MS2 data at all
            subheading1 = 'No MS2 data available'

        if ms2_file:
            subheading2 = f'File: {ms2_file}'
        else:
            subheading2 = 'No file data'

        ms2_title = f'{heading}<br><sub>{subheading1}</sub><br><sub>{subheading2}</sub>'

        return ms2_title

    def write_eic_title(compound: wfo.CompoundExperimental, current_compound_index=None):
        # Create EIC title (exact same format as original)
        eic_title_parts = []
        
        # Add compound index to the beginning if provided
        if current_compound_index is not None:
            name = compound.compound_name
            index = current_compound_index + 1
            eic_title_parts.append(f'{name} {index}')
        
        # Basic info
        if compound.adduct:
            eic_title_parts.append(f'Adduct: {compound.adduct}')
        if compound.inchi_key:
            eic_title_parts.append(f'InChI Key: {compound.inchi_key}')
        eic_title_parts.append(f'LCMS files: {len(compound.eic_data_files)}')
        subheading1 = ' | '.join(eic_title_parts)

        # RT and m/z comparison using CURRENT RT values
        rt_parts = [f'Atlas RT: {compound.atlas_rt_peak:.2f}']

        best_rt = compound.best_eic_rt
        if best_rt is not None and best_rt > 0:
            # Recalculate RT diff using current RT peak
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

        # Isomers
        isomers = compound.isomers
        isomer_strings = []
        for isomer in isomers:
            name = isomer.get('compound_name', '')
            isomer_inchi_key = isomer.get('inchi_key', '')
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

    def create_plot_only(initial_render=False):
        """
        ONLY create and display the plot - no other side effects.
        """
        compound = get_current_compound()
        if compound is None:
            return
        print(f"Compound: {compound}")

        plot_output.clear_output(wait=True)

        # Get RT bounds for MS2 filtering
        rt_min = compound.rt_min
        rt_max = compound.rt_max
        rt_peak = compound.rt_peak

        # Get MS2 selection info (now filtered by RT bounds and sorted by score)
        plot_spec_data, ms2_file, files_in_rt_window = get_selected_ms2_spectra(compound, 
                                                            current_ms2_file_index[0],
                                                            rt_min,
                                                            rt_max)

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
        """
        Perform a complete update of all UI elements.
        This is the main orchestrator function.
        """
        if updating[0]:
            return
            
        updating[0] = True
        try:
            compound = get_current_compound()
            if compound is None:
                return
            
            rt_min = compound.rt_min
            rt_max = compound.rt_max
            rt_peak = compound.rt_peak
            
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
            print("Running create_plot_only() on first compound")
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
            
        compound = get_current_compound()
        if compound is None:
            return

        # Get current RT bounds for filtering
        rt_min = compound.rt_min
        rt_max = compound.rt_max
        
        # Get MS2 data using corrected structure access
        ms2_data = compound.ms2_data_files
        if not ms2_data:
            return
        
        # Handle both possible structures
        if 'files' in ms2_data and isinstance(ms2_data['files'], dict):
            ms2_files_data = ms2_data['files']
        else:
            ms2_files_data = ms2_data
        
        # Count files within current RT window
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
            elif best_ms2 and 'rt_peak' in best_ms2:
                rt_measured = best_ms2.get('rt_peak', 0)
            
            if rt_measured is not None and rt_min <= rt_measured <= rt_max:
                files_in_rt_window.append(file_name)
        
        total_files_in_window = len(files_in_rt_window)
        if total_files_in_window == 0:
            return

        new_idx = current_ms2_file_index[0] + direction
        # Bounds checking for files within RT window only
        if 0 <= new_idx < total_files_in_window:
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
            # Reset compound to original values using CompoundExperimental methods
            compound.rt_min = compound.atlas_rt_min
            compound.rt_max = compound.atlas_rt_max
            compound.rt_peak = compound.atlas_rt_peak
            compound.is_rt_modified = False
            
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
            
        compound = get_current_compound()
        if compound is None:
            return
        
        suggested_data = compound.suggested_rt_bounds
        if suggested_data is None:
            return
        
        updating[0] = True
        try:
            # Reset MS2 file index since RT bounds will change
            reset_ms2_file_index()
            
            # Update using CompoundExperimental method
            compound.update_rt_bounds(
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

        # Calculate new peak as midpoint
        new_peak = new_min + (new_max - new_min) / 2.0
        
        # Update using CompoundExperimental method
        compound.update_rt_bounds(new_min, new_max, new_peak)

        # Reset MS2 file index since RT bounds changed
        reset_ms2_file_index()
        
        create_plot_only()

        # Trigger auto-save check
        auto_save_if_needed()

    def reset_ms2_file_index():
        """Reset MS2 file index to 0 when switching compounds"""
        current_ms2_file_index[0] = 0

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

    def update_annotation_widgets():
        """Update annotation radio buttons to reflect current compound's values"""
        compound = get_current_compound()
        if compound is None:
            return

        # Temporarily disable updating flag to prevent recursion
        temp_updating = updating[0]
        updating[0] = True
        
        try:
            ms2_radio.value = compound.ms2_notes
            ms1_radio.value = compound.ms1_notes
            analyst_notes_box.value = compound.analyst_notes
            id_notes_box.value = compound.identification_notes
        finally:
            updating[0] = temp_updating

    def on_ms2_annotation_change(change):
        """Handle MS2 annotation changes"""
        if updating[0]:
            return
            
        compound = get_current_compound()
        if compound is None:
            return
                
        # Update using CompoundExperimental method
        compound.update_annotations(ms2_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_ms1_annotation_change(change):
        """Handle MS1 annotation changes"""
        if updating[0]:
            return
            
        compound = get_current_compound()
        if compound is None:
            return
        
        # Update using CompoundExperimental method
        compound.update_annotations(ms1_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_analyst_notes_change(change):
        """Handle analyst notes changes"""
        if updating[0]:
            return
            
        compound = get_current_compound()
        if compound is None:
            return
        
        # Update using CompoundExperimental method
        compound.update_annotations(analyst_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()

    def on_id_notes_change(change):
        """Handle ID notes changes"""
        if updating[0]:
            return

        compound = get_current_compound()
        if compound is None:
            return

        # Update using CompoundExperimental method
        compound.update_annotations(identification_notes=change['new'])

        # Trigger auto-save check
        auto_save_if_needed()
    
    # Add manual save button
    manual_save_button = widgets.Button(
        description="Save Session", 
        button_style='info', 
        layout=widgets.Layout(width='150px')
    )
    
    def on_manual_save(button):
        """Manual save handler for CompoundExperimental objects"""
        if analysis_dir:
            try:
                # Save compounds state to cache
                timestamp = save_compounds_progress(compounds, analysis_dir)
                if timestamp:
                    button.description = f"Saved {timestamp[-8:-3]}"
                    last_save_time[0] = time.time()
                    
                    # Reset button text after 2 seconds
                    import threading
                    def reset_button():
                        time.sleep(2)
                        button.description = "Save Session"
                    threading.Thread(target=reset_button).start()
                else:
                    button.description = "Save Failed"
            except Exception as e:
                logger.error(f"Manual save failed: {e}")
                button.description = "Save Failed"
    
    # Connect all event handlers to their widgets
    manual_save_button.on_click(on_manual_save)
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
    
    # Initialize with the first compound
    full_update()
    
    return container, compounds

def export_compound_changes(compounds: List) -> pd.DataFrame:
    """
    Export all compound changes to a pandas DataFrame for review.
    
    Args:
        compounds: List of CompoundExperimental objects
        
    Returns:
        DataFrame with all compound changes
    """
    changes_data = []
    
    for i, compound in enumerate(compounds):
        row = {
            'compound_index': i + 1,
            'compound_name': compound.compound_name,
            'inchi_key': getattr(compound, 'inchi_key', ''),
            'atlas_type': getattr(compound, '_atlas_type', ''),
            'chromatography_polarity': getattr(compound, '_chromatography_polarity', ''),
            'rt_modified': compound.is_rt_modified if hasattr(compound, 'is_rt_modified') else False,
            'annotation_modified': compound.is_annotation_modified if hasattr(compound, 'is_annotation_modified') else False,
            'atlas_rt_min': getattr(compound, 'atlas_rt_min', None),
            'atlas_rt_max': getattr(compound, 'atlas_rt_max', None),
            'atlas_rt_peak': getattr(compound, 'atlas_rt_peak', None),
            'current_rt_min': getattr(compound, 'rt_min', None),
            'current_rt_max': getattr(compound, 'rt_max', None),
            'current_rt_peak': getattr(compound, 'rt_peak', None),
            'rt_peak_delta': None,
            'ms1_notes': getattr(compound, 'ms1_notes', ''),
            'ms2_notes': getattr(compound, 'ms2_notes', ''),
            'analyst_notes': getattr(compound, 'analyst_notes', ''),
            'identification_notes': getattr(compound, 'identification_notes', '')
        }
        
        # Calculate RT peak delta if both values exist
        if row['atlas_rt_peak'] is not None and row['current_rt_peak'] is not None:
            row['rt_peak_delta'] = row['current_rt_peak'] - row['atlas_rt_peak']
            
        changes_data.append(row)
    
    df = pd.DataFrame(changes_data)
    
    # Filter to only show compounds with changes
    modified_df = df[df['rt_modified'] | df['annotation_modified']].copy()
    
    return df, modified_df