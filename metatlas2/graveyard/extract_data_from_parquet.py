
# def extract_single_file_from_parquet(
#     parquet_file: str,
#     atlas_df: pd.DataFrame,
#     polarity: str,
#     ppm_tolerance: float = 20.0,
#     extra_time: float = 0.1,
#     ms_levels: List[str] = ['ms1', 'ms2']
# ) -> Dict[str, Dict]:
#     """
#     Extract data from a single file (useful for parallel processing).
    
#     Returns:
#         Dict mapping label to compound data:
#         {
#             'label1': {
#                 'ms1_data': DataFrame,
#                 'ms1_summary': DataFrame,
#                 'ms2_data': DataFrame
#             }
#         }
#     """
#     results = {}
    
#     # Get parquet paths
#     ms1_parquet = get_parquet_path_from_h5(parquet_file, 1, polarity)
#     ms2_parquet = get_parquet_path_from_h5(h5_file, 2, polarity)
    
#     # Check if MS1 parquet exists
#     if not Path(ms1_parquet).exists() and 'ms1' in ms_levels:
#         logger.warning(f"MS1 parquet not found: {ms1_parquet}")
#         return results
    
#     # Extract features for each compound
#     for _, row in atlas_df.iterrows():
#         label = row.get('label', row.get('inchi_key', ''))
        
#         compound_data = {
#             'ms1_data': pd.DataFrame(),
#             'ms1_summary': pd.DataFrame(),
#             'ms2_data': pd.DataFrame()
#         }
        
#         # Extract MS1 data
#         if 'ms1' in ms_levels and Path(ms1_parquet).exists():
#             ms1_data = extract_ms1_from_parquet(
#                 ms1_parquet,
#                 label=label,
#                 mz=row['mz'],
#                 rt_min=row['rt_min'],
#                 rt_max=row['rt_max'],
#                 ppm_tolerance=ppm_tolerance,
#                 extra_time=extra_time
#             )
#             compound_data['ms1_data'] = ms1_data
            
#             # Calculate summary
#             if not ms1_data.empty:
#                 ms1_summary = calculate_ms1_summary(ms1_data, feature_filter=True).reset_index(drop=True)
#                 if ms1_summary.shape[0] == 0:
#                     for c in ['num_datapoints', 'peak_area', 'peak_height', 'mz_centroid', 'rt_peak']:
#                         ms1_summary[c] = 0
#                 compound_data['ms1_summary'] = ms1_summary
        
#         # Extract MS2 data
#         if 'ms2' in ms_levels and Path(ms2_parquet).exists():
#             ms2_data = extract_ms2_feature_from_parquet(
#                 ms2_parquet,
#                 label=label,
#                 mz=row['mz'],
#                 rt_min=row['rt_min'],
#                 rt_max=row['rt_max'],
#                 ppm_tolerance=ppm_tolerance,
#                 extra_time=extra_time
#             )
#             compound_data['ms2_data'] = ms2_data
        
#         results[label] = compound_data
    
#     return results

# def get_parquet_path_from_h5(h5_path: str, ms_level: int, polarity: str) -> str:
#     """
#     Construct parquet file path from H5 path.
    
#     Args:
#         h5_path: Path to H5 file (e.g., /path/to/file.h5)
#         ms_level: 1 or 2
#         polarity: 'positive' or 'negative'
    
#     Returns:
#         Path to parquet file
#     """
#     base_path = Path(h5_path).with_suffix('')
#     polarity_short = polarity[:3].lower()  # 'pos' or 'neg'
#     ms_str = f'ms{ms_level}'
    
#     return str(base_path) + f'_{ms_str}_{polarity_short}.parquet'