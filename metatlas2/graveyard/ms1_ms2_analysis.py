def calculate_mz_tolerance_range(mz: float, tolerance_ppm: float) -> Tuple[float, float]:
    """Calculate m/z tolerance range in Daltons."""
    tolerance_da = mz * tolerance_ppm / 1e6
    return mz - tolerance_da, mz + tolerance_da

def extract_eic_and_ms2_data(input_data_list: List[Dict], atlas_dataframe: pd.DataFrame, 
                             config: Dict, ms_levels: List[str] = ['ms1', 'ms2']) -> Dict[str, Dict]:
    """
    Extract EIC and MS2 data using simplified approach - returns raw data only.
    AnalysisProject will handle object creation and management.
    
    Returns:
        Dict keyed by inchi_key containing experimental data structures
    """
    
    # Create a mapping from compound label to metadata
    compound_metadata = {}
    for _, row in atlas_dataframe.iterrows():
        compound_metadata[row['label']] = {
            'inchi_key': row.get('inchi_key', ''),
            'compound_uid': row.get('compound_uid', ''),
            'adduct': row.get('adduct', ''),
            'mz': row.get('mz', 0.0),
            'rt_min': row.get('rt_min', 0.0),
            'rt_max': row.get('rt_max', 0.0),
            'rt_peak': row.get('rt_peak', 0.0)
        }

    # Extract experimental data using simplified approach
    max_workers = min(mp.cpu_count(), len(input_data_list), 8)
    use_parallel = max_workers > 1 and len(input_data_list) > 1
    if use_parallel:
        logger.info(f"Using parallel processing with {max_workers} workers...")
    else:
        logger.info("Using sequential processing...")
    experimental_data = _extract_data(input_data_list, compound_metadata, config, ms_levels, use_parallel, max_workers)
    
    # Print summary
    logger.info(f"Extraction complete:")
    logger.info(f"  Total compounds with data: {len(experimental_data)}")
    compounds_with_eic = sum(1 for data in experimental_data.values() if data.get('eic_files'))
    compounds_with_ms2 = sum(1 for data in experimental_data.values() if data.get('ms2_files'))
    logger.info(f"  Compounds with EIC data: {compounds_with_eic}")
    logger.info(f"  Compounds with MS2 data: {compounds_with_ms2}")
    
    return experimental_data

def _extract_data(input_data_list: List[Dict], compound_metadata: Dict[str, Dict], 
                        config: Dict, ms_levels: List[str], 
                        use_parallel: bool, max_workers: int) -> Dict[str, Dict]:
    """Extract data using either parallel or sequential processing based on flag."""
    experimental_data = {}
    
    if use_parallel:
        # Parallel processing
        logger.info(f"Setting up {max_workers} workers for parallel processing...")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, file_input in enumerate(input_data_list):
                filename = Path(file_input['lcmsrun']).name
                future = executor.submit(_process_file_data, file_input, compound_metadata, config, ms_levels, filename)
                futures.append((future, file_input['lcmsrun'], i))
            
            # Collect results
            for future, file_path, i in tqdm(futures, desc=f"Extracting {ms_levels} data from files"):
                try:
                    file_experimental_data = future.result()
                    
                    # Merge file data into experimental_data -  merging
                    _merge_file_data_into_experimental_data(experimental_data, file_experimental_data)
                    
                    # Log progress
                    if len(file_experimental_data) > 0:
                        logger.info(f"  Completed {i+1}/{len(input_data_list)}: {Path(file_path).name} - {len(file_experimental_data)} compounds")
                    else:
                        logger.warning(f"  Completed {i+1}/{len(input_data_list)}: {Path(file_path).name} - No data extracted")
                        
                except Exception as e:
                    logger.error(f"  Error in parallel processing for file {i+1}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    continue
    
    else:
        # Sequential processing
        for i, file_input in enumerate(tqdm(input_data_list, desc="Processing files")):
            file_path = file_input['lcmsrun']
            filename = Path(file_path).name
            
            try:
                # Process single file and get results
                file_experimental_data = _process_file_data(file_input, compound_metadata, config, ms_levels, filename)
                
                # Merge file data into experimental_data
                _merge_file_data_into_experimental_data(experimental_data, file_experimental_data)
                
                logger.info(f"  File {filename} processed successfully: {len(file_experimental_data)} compounds found")
                
            except Exception as e:
                logger.error(f"  Error processing {filename}: {e}")
                continue

    # Post-process to add summary statistics
    for inchi_key, compound_data in experimental_data.items():
        _add_summary_statistics(compound_data)
    
    processing_type = "parallel" if use_parallel else "sequential"
    logger.info(f"{processing_type.capitalize()} processing complete: {len(experimental_data)} compounds with data")
    return experimental_data


def _process_file_data(file_input: Dict, compound_metadata: Dict[str, Dict], 
                      config: Dict, ms_levels: List[str], filename: str) -> Dict[str, Dict]:
    """Process a single file's data - shared logic for both parallel and sequential processing."""
    # Extract raw data
    data = ftt.get_data(file_input, ms_levels, save_file=False, return_data=True, ms1_feature_filter=False)
    
    file_experimental_data = {}
    
    # Process EIC data -  extraction only
    if not data['ms1_data'].empty:
        adduct_eics = ftt.group_duplicates(data['ms1_data'], 'label', make_string=False)
        
        if not adduct_eics.empty and 'label' in adduct_eics.columns:
            # Add metadata mapping
            adduct_eics['inchi_key'] = adduct_eics['label'].map(
                lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
            )
            adduct_eics['compound_uid'] = adduct_eics['label'].map(
                lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
            )
            adduct_eics['adduct'] = adduct_eics['label'].map(
                lambda x: compound_metadata.get(x, {}).get('adduct', '')
            )
            
            for idx, eic_row in adduct_eics.iterrows():
                inchi_key = eic_row.get('inchi_key', '')
                if not inchi_key:
                    continue
                
                # Initialize  data structure
                if inchi_key not in file_experimental_data:
                    file_experimental_data[inchi_key] = {
                        'eic_files': {},
                        'ms2_files': {}
                    }
                
                # Extract  EIC data
                eic_data = _extract_eic_data(eic_row, filename, config)
                if eic_data:
                    file_experimental_data[inchi_key]['eic_files'][filename] = eic_data
    
    # Process MS2 data -  extraction only
    if not data['ms2_data'].empty:
        ms2_summary = ftt.calculate_ms2_summary(data['ms2_data'])
        
        if not ms2_summary.empty and 'label' in ms2_summary.columns:
            # Add metadata mapping
            ms2_summary['inchi_key'] = ms2_summary['label'].map(
                lambda x: compound_metadata.get(x, {}).get('inchi_key', '')
            )
            ms2_summary['compound_uid'] = ms2_summary['label'].map(
                lambda x: compound_metadata.get(x, {}).get('compound_uid', '')
            )
            ms2_summary['adduct'] = ms2_summary['label'].map(
                lambda x: compound_metadata.get(x, {}).get('adduct', '')
            )
            
            for idx, ms2_row in ms2_summary.iterrows():
                inchi_key = ms2_row.get('inchi_key', '')
                if not inchi_key:
                    continue
                
                # Initialize  data structure
                if inchi_key not in file_experimental_data:
                    file_experimental_data[inchi_key] = {
                        'eic_files': {},
                        'ms2_files': {}
                    }
                
                # Extract raw MS2 data without hits
                ms2_data = _extract_ms2_data(ms2_row, filename, config)
                if ms2_data:
                    if filename not in file_experimental_data[inchi_key]['ms2_files']:
                        file_experimental_data[inchi_key]['ms2_files'][filename] = {
                            "ms2_entries": []
                        }
                    
                    file_experimental_data[inchi_key]['ms2_files'][filename]["ms2_entries"].append(ms2_data)
    
    return file_experimental_data

def _extract_ms2_data(ms2_row: pd.Series, filename: str, config: Dict) -> Optional[Dict]:
    """Extract raw MS2 data from a row - pure data extraction without hit detection."""
    try:
        inchi_key = ms2_row.get('inchi_key', '')

        # Parse spectrum data
        spectrum_data = ms2_row.get('spectrum', [[], []])
        mz_values = np.array(spectrum_data[0]) if len(spectrum_data) == 2 else np.array([])
        intensity_values = np.array(spectrum_data[1]) if len(spectrum_data) == 2 else np.array([])

        # Check for empty arrays
        if (isinstance(mz_values, np.ndarray) and mz_values.size == 0) or \
           (isinstance(intensity_values, np.ndarray) and intensity_values.size == 0) or \
           (not isinstance(mz_values, np.ndarray) and len(mz_values) == 0) or \
           (not isinstance(intensity_values, np.ndarray) and len(intensity_values) == 0):
            return None

        # Find peak values
        max_idx = np.argmax(intensity_values)
        intensity_peak = intensity_values[max_idx]
        mz_peak = mz_values[max_idx]
        rt_peak = float(ms2_row.get('rt', 0.0))
        precursor_mz = ms2_row.get('precursor_mz', 0.0)
        
        # Extract measured values from the MS2 scan
        mz_measured = float(ms2_row.get('precursor_mz', precursor_mz))  # Use precursor_mz as measured m/z
        rt_measured = float(ms2_row.get('rt', rt_peak))  # Use rt as measured RT

        # Calculate ppm error for precursor
        ppm_error = abs(mz_peak - precursor_mz) / precursor_mz * 1e6 if precursor_mz > 0 else 0.0

        ms2_entry = {
            "inchi_key": inchi_key,
            "spectrum": [mz_values.tolist(), intensity_values.tolist()],
            "intensity_peak": float(intensity_peak),
            "rt_peak": float(rt_peak),
            "rt_measured": rt_measured,  # Add measured RT
            "mz_peak": float(mz_peak),
            "mz_measured": mz_measured,  # Add measured m/z
            "precursor_mz": float(precursor_mz),
            "ppm_diff": float(ppm_error),
        }

        return ms2_entry

    except Exception as e:
        logger.error(f"Error extracting MS2 data for {filename}: {e}")
        return None

def _merge_file_data_into_experimental_data(experimental_data: Dict[str, Dict], file_experimental_data: Dict[str, Dict]):
    """Merge file data into experimental_data -  merging logic."""
    for inchi_key, compound_data in file_experimental_data.items():
        if inchi_key not in experimental_data:
            experimental_data[inchi_key] = {
                'eic_files': {},
                'ms2_files': {}
            }
        
        # Merge EIC files
        if 'eic_files' in compound_data:
            experimental_data[inchi_key]['eic_files'].update(compound_data['eic_files'])
        
        # Merge MS2 files
        if 'ms2_files' in compound_data:
            experimental_data[inchi_key]['ms2_files'].update(compound_data['ms2_files'])

def _extract_eic_data(eic_row: pd.Series, filename: str, config: Dict) -> Optional[Dict]:
    """Extract  EIC data from a row - pure data extraction."""
    try:
        intensities = eic_row.get('i', np.array([]))
        rts = eic_row.get('rt', np.array([]))
        mzs = eic_row.get('mz', np.array([]))

        if (isinstance(intensities, np.ndarray) and intensities.size == 0) or \
           (isinstance(rts, np.ndarray) and rts.size == 0) or \
           (not isinstance(intensities, np.ndarray) and len(intensities) == 0) or \
           (not isinstance(rts, np.ndarray) and len(rts) == 0):
            return None
        
        # Find peak values
        max_idx = np.argmax(intensities)
        rt_peak = rts[max_idx]
        intensity_peak = intensities[max_idx]
        mz_peak = mzs[max_idx] if (isinstance(mzs, (np.ndarray, list)) and len(mzs) > 0) else 0.0

        # Filter by intensity peak
        #if intensity_peak < config['']
        
        # Calculate  errors
        atlas_rt_peak = eic_row.get('rt_peak', 0.0)
        atlas_mz = eic_row.get('mz', 0.0)
        # Ensure atlas_mz is a scalar
        if isinstance(atlas_mz, (np.ndarray, list, pd.Series)):
            if len(atlas_mz) > 0:
                atlas_mz_val = float(atlas_mz[0])
            else:
                atlas_mz_val = 0.0
        else:
            atlas_mz_val = float(atlas_mz)
        ppm_error = abs(mz_peak - atlas_mz_val) / atlas_mz_val * 1e6 if atlas_mz_val > 0 else 0.0
        rt_error = rt_peak - atlas_rt_peak
        
        result = {
            "rt_vals": rts.tolist(),
            "i_vals": intensities.tolist(),
            "mz_vals": mzs.tolist(),
            "intensity_peak": float(intensity_peak),
            "rt_peak": float(rt_peak),
            "mz_peak": float(mz_peak),
            "ppm_diff": float(ppm_error),
            "rt_diff": float(rt_error),
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting EIC data for {filename}: {e}")
        return None


def _add_summary_statistics(compound_data: Dict):
    """Add summary statistics to compound data - basic version without hits."""
    try:
        # Add EIC summary statistics
        eic_files = compound_data.get('eic_files', {})
        if eic_files:
            compound_data['total_files_detected'] = len(eic_files)
        
        # Add MS2 summary statistics without hit information
        ms2_files = compound_data.get('ms2_files', {})
        if ms2_files:
            # Count files with MS2 data
            files_with_data = len([f for f in ms2_files.values() if f.get('ms2_entries')])
            compound_data['ms2_files_with_data'] = files_with_data
            
            # Ensure each file has basic structure
            for filename, file_data in ms2_files.items():
                entries = file_data.get('ms2_entries', [])
                
                # Best MS2 by intensity for this file (no hits yet)
                if entries:
                    best_ms2 = max(entries, key=lambda e: e.get('intensity_peak', 0.0))
                    file_data['best_ms2'] = best_ms2
                    file_data['num_ms2_entries'] = len(entries)
                else:
                    file_data['best_ms2'] = {}
                    file_data['num_ms2_entries'] = 0
        
    except Exception as e:
        logger.error(f"Error adding summary statistics: {e}")

# def prepare_feature_tools_inputs(atlas_df: pd.DataFrame, h5_files: List[str], 
#                                 ppm_tolerance: float = 20, extra_time: float = 0.1) -> List[Dict]:
#     """
#     Prepare input parameters for feature_tools.get_data() function using setup_file_slicing_parameters
    
#     Args:
#         atlas_df: Atlas DataFrame with required columns
#         h5_files: List of H5 file paths
#         ppm_tolerance: m/z tolerance in ppm (default: 20)
#         extra_time: Additional time window in minutes (default: 0.1)
    
#     Returns:
#         List of input dictionaries for feature_tools.get_data()
#     """
#     # Auto-detect polarity from atlas
#     polarity = 'positive'
#     if 'polarity' in atlas_df.columns:
#         polarity = atlas_df['polarity'].iloc[0] if not atlas_df['polarity'].empty else 'positive'

#     # Use setup_file_slicing_parameters to prepare inputs
#     input_data_list = ftt.setup_file_slicing_parameters(
#         atlas=atlas_df,
#         filenames=h5_files,
#         extra_time=extra_time,
#         ppm_tolerance=ppm_tolerance,
#         polarity=polarity,
#         project_dir=False,
#         overwrite=True
#     )
    
#     return input_data_list