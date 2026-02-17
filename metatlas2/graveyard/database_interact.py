# def save_atlas_to_database(atlas_obj, db_path: str, db_type: str = "main") -> None:
#     """
#     Save an Atlas object to the database (typing not included to avoid circular imports).
#     Only creates new mz_rt_references entries for references that don't already exist.
    
#     Args:
#         atlas_obj: Atlas object to save
#     """
#     # Verify all compounds exist in database
#     if not verify_compounds_exist_in_db([comp.compound_uid for comp in atlas_obj.compound_references.values()], db_path):
#         raise ValueError(f"Some compounds in atlas {atlas_obj.atlas_uid} don't exist in database")

#     prov = ldt.get_provenance()
#     with get_db_connection(db_path) as conn:
#         if db_type == "main":
#             # Create atlas entry
#             conn.execute("""
#                 INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 atlas_obj.atlas_uid,
#                 atlas_obj.atlas_name,
#                 atlas_obj.atlas_description,
#                 atlas_obj.chromatography,
#                 atlas_obj.polarity,
#                 "REFERENCE",  # atlas_type
#                 prov["analyst"],
#                 prov["timestamp"]
#             ))
            
#             # Process each CompoundReference
#             association_order = 0
#             references_created = 0
#             references_reused = 0
#             for inchi_key, compound_ref in atlas_obj.compound_references.items():
#                 # Check if this reference already exists in database
#                 existing_check = conn.execute("""
#                     SELECT mz_rt_reference_uid FROM mz_rt_references 
#                     WHERE mz_rt_reference_uid = ?
#                 """, [compound_ref.mz_rt_reference_uid]).fetchone()
                
#                 mz_rt_reference_uid = compound_ref.mz_rt_reference_uid
                
#                 # Create new reference if it doesn't exist
#                 if not existing_check:
#                     conn.execute("""
#                         INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                     """, (
#                         mz_rt_reference_uid,
#                         compound_ref.compound_uid,
#                         compound_ref.rt_peak,
#                         compound_ref.rt_min,
#                         compound_ref.rt_max,
#                         compound_ref.mz,
#                         compound_ref.mz_tolerance,
#                         compound_ref.adduct,
#                         compound_ref.chromatography,
#                         compound_ref.polarity,
#                         compound_ref.confidence,
#                         'atlas_creation',  # source
#                         prov["analyst"],
#                         prov["timestamp"]
#                     ))
#                     references_created += 1
#                 else:
#                     references_reused += 1
                
#                 # Create atlas-compound association
#                 assoc_uid = _generate_uid("association")
#                 conn.execute("""
#                     INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     assoc_uid,
#                     atlas_obj.atlas_uid,
#                     compound_ref.compound_uid,
#                     mz_rt_reference_uid,
#                     association_order,
#                     prov["analyst"],
#                     prov["timestamp"]
#                 ))
                
#                 association_order += 1
        
#             logger.info(f"Saved atlas {atlas_obj.atlas_name} to database with UID: {atlas_obj.atlas_uid}")
#             logger.info(f"  References created: {references_created}")
#             logger.info(f"  References reused: {references_reused}")
#             logger.info(f"  Total associations: {association_order}")

#         association_order += 1

# def get_rt_aligned_atlases_from_db(
#     project_db_path: str,
#     rt_alignment_number: int,
# ) -> List[Dict]:
#     """
#     Get all RT-aligned atlases for given RT alignment number.
    
#     """
#     params = [rt_alignment_number]
#     query = """
#         SELECT 
#             atlas_uid,
#             atlas_name,
#             atlas_description,
#             chromatography,
#             polarity,
#             atlas_type,
#             source_atlas_uid,
#             rt_alignment_number,
#             created_by,
#             created_date
#         FROM atlases
#         WHERE rt_alignment_number = ?
#     """
    
#     query += " ORDER BY atlas_type, polarity"
    
#     with get_db_connection(project_db_path) as conn:
#         results = conn.execute(query, params).fetchall()
#         atlases = []
#         for row in results:
#             # Infer workflow from chromatography
#             workflow = row[3].upper()  # chromatography -> workflow
            
#             atlases.append({
#                 'atlas_uid': row[0],
#                 'atlas_name': row[1],
#                 'atlas_description': row[2],
#                 'chromatography': row[3],
#                 'polarity': row[4],
#                 'atlas_type': row[5],
#                 'source_atlas_uid': row[6],
#                 'rt_alignment_number': row[7],
#                 'created_by': row[8],
#                 'created_date': row[9],
#                 'workflow': workflow
#             })
        
#         if not atlases:
#             raise ValueError(f"No RT-aligned atlases found in project database for RT alignment number {rt_alignment_number}")
#         else:
#             logger.info(f"Found {len(atlases)} RT-aligned atlases in project database for RT alignment number {rt_alignment_number}")
#         return atlases

# def save_rt_aligned_atlas_to_db(
#     project_db_path: str,
#     main_db_path: str,
#     target_atlas_uid: str,
#     best_model: dict,
#     aligned_compounds_df: pd.DataFrame,
#     rt_alignment_number: int,
#     analysis_number: Optional[int] = None
# ) -> Tuple[str, str]:
#     """
#     Create RT-aligned atlas in project database and apply RT alignment to compounds.
    
#     Args:
#         project_db_path: Path to project database
#         main_db_path: Path to main database
#         target_atlas_uid: UID of the target atlas
#         best_model: RT alignment model dictionary
#         aligned_compounds_df: DataFrame of RT aligned compounds
#         rt_alignment_number: RT alignment number
#         analysis_number: Optional analysis number for this RT alignment (if not provided, will be NULL in database)
#     Returns:
#         Tuple of (aligned_atlas_uid, aligned_atlas_name)
#     """
#     logger.info("Creating RT-aligned atlas in project database...")

#     prov = ldt.get_provenance()

#     # Use some info from the target atlas for the new RT-aligned target atlas
#     target_atlas_info = get_atlas_metadata_from_db(main_db_path, target_atlas_uid)

#     # Generate new atlas UID for RT-aligned version
#     target_atlas_decorator = get_decorator_from_uid(target_atlas_uid)
#     aligned_atlas_uid = _generate_uid("rt_atlas", decorator=target_atlas_decorator)
#     aligned_atlas_name = f"{target_atlas_info['atlas_name']} (RT aligned)"
#     aligned_atlas_description = f"RT-aligned version of {target_atlas_info['atlas_name']} using polynomial model (R²={best_model['r2']:.4f})"
#     logger.info(f"From the {target_atlas_info['atlas_uid']} template, generated new atlas UID: {aligned_atlas_uid} for RT-aligned atlas: {aligned_atlas_name}")

#     logger.info("Saving RT-aligned atlas and compounds to project database...")
#     display(aligned_compounds_df.head())
#     with get_db_connection(project_db_path) as conn:
        
#         # Create new atlas entry
#         conn.execute("""
#             INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#         """, (
#             aligned_atlas_uid,
#             aligned_atlas_name,
#             aligned_atlas_description,
#             target_atlas_info['chromatography'],
#             target_atlas_info['polarity'],
#             target_atlas_info['atlas_type'],
#             target_atlas_info['atlas_uid'],
#             rt_alignment_number,
#             analysis_number,
#             prov["analyst"],
#             prov["timestamp"]
#         ))

#         association_order = 0
#         for _, row in aligned_compounds_df.iterrows():
#             compound_uid = row['compound_uid']
#             aligned_rt_peak = row.get('rt_peak')
#             aligned_rt_min = row.get('rt_min')
#             aligned_rt_max = row.get('rt_max')
#             rt_shift = row.get('rt_shift')
#             exp_uid = _generate_uid("mz_rt_experimental")
#             assoc_uid = _generate_uid("association")
#             mz_rt_reference_uid = row.get('mz_rt_reference_uid')

#             conn.execute("""
#                 INSERT INTO mz_rt_experimental VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 exp_uid,
#                 compound_uid,
#                 rt_alignment_number,
#                 analysis_number,
#                 aligned_rt_peak,
#                 aligned_rt_min,
#                 aligned_rt_max,
#                 '',  # ms1_notes
#                 '',  # ms2_notes
#                 row.get('mz'),
#                 row.get('mz_tolerance', 5.0),
#                 row.get('adduct', ''),
#                 target_atlas_info['chromatography'],
#                 target_atlas_info['polarity'],
#                 True,  # rt_alignment_applied
#                 rt_shift,
#                 mz_rt_reference_uid,
#                 prov["analyst"],
#                 prov["timestamp"]
#             ))
            
#             # Create atlas-compound association
#             conn.execute("""
#                 INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 assoc_uid,
#                 aligned_atlas_uid,
#                 compound_uid,
#                 exp_uid,
#                 association_order,
#                 prov["analyst"],
#                 prov["timestamp"]
#             ))

#             association_order += 1

#     logger.info(f"Created and deposited RT-aligned atlas: {aligned_atlas_uid} with name: {aligned_atlas_name}")

#     return aligned_atlas_uid, aligned_atlas_name


# def get_atlas_compounds_table(database_path: str, atlas_uid: str, main_db_path: str = None) -> pd.DataFrame:
#     """
#     Extract all compound information for a given atlas UID from the database.
#     Handles both main database (with mz_rt_references) and project database (with mz_rt_experimental).
#     """
#     with get_db_connection(database_path) as conn:
        
#         # Detect database type
#         try:
#             result = conn.execute("""
#                 SELECT name FROM sqlite_master 
#                 WHERE type='table' AND name='mz_rt_experimental'
#             """).fetchall()
#             is_project_db = len(result) > 0
#         except Exception as e:
#             logger.error(f"Error checking database type: {e}")
#             return pd.DataFrame()

#         # Attach main database if needed for compound metadata
#         if is_project_db and main_db_path:
#             try:
#                 conn.execute(f"ATTACH '{main_db_path}' AS main_db")
#                 logger.info("Attached main database for compound metadata")
#             except Exception as e:
#                 logger.error(f"Error attaching main database: {e}")
#                 return pd.DataFrame()

#         try:
#             if is_project_db:
#                 # Project DB: Join through associations to experimental entries
#                 query = """
#                     SELECT
#                         a.atlas_uid,
#                         a.atlas_name,
#                         a.atlas_description,
#                         a.chromatography,
#                         a.polarity,
#                         aca.compound_uid,
#                         COALESCE(main_db.compounds.name, '') AS compound_name,
#                         COALESCE(main_db.compounds.inchi_key, '') AS inchi_key,
#                         COALESCE(main_db.compounds.inchi, '') AS inchi,
#                         mzrt_exp.adduct,
#                         mzrt_exp.mz,
#                         mzrt_exp.rt_peak,
#                         mzrt_exp.rt_min,
#                         mzrt_exp.rt_max,
#                         mzrt_exp.mz_tolerance,
#                         mzrt_exp.mz_rt_experimental_uid AS mz_rt_reference_uid,
#                         mzrt_exp.rt_alignment_applied,
#                         mzrt_exp.rt_shift
#                     FROM atlases a
#                     JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
#                     LEFT JOIN main_db.compounds ON aca.compound_uid = main_db.compounds.compound_uid
#                     LEFT JOIN mz_rt_experimental mzrt_exp 
#                         ON aca.mz_rt_reference_uid = mzrt_exp.mz_rt_experimental_uid
#                     WHERE a.atlas_uid = ?
#                     ORDER BY aca.association_order
#                 """
#             else:
#                 # Main DB: Join through associations to reference entries
#                 query = """
#                     SELECT
#                         a.atlas_uid,
#                         a.atlas_name,
#                         a.atlas_description,
#                         a.chromatography,
#                         a.polarity,
#                         c.compound_uid,
#                         c.name AS compound_name,
#                         c.inchi_key,
#                         c.inchi,
#                         mzrt.adduct,
#                         mzrt.mz,
#                         mzrt.rt_peak,
#                         mzrt.rt_min,
#                         mzrt.rt_max,
#                         mzrt.mz_tolerance,
#                         mzrt.mz_rt_reference_uid,
#                         FALSE AS rt_alignment_applied,
#                         NULL AS rt_shift
#                     FROM atlases a
#                     JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
#                     JOIN compounds c ON aca.compound_uid = c.compound_uid
#                     LEFT JOIN mz_rt_references mzrt 
#                         ON aca.mz_rt_reference_uid = mzrt.mz_rt_reference_uid
#                     WHERE a.atlas_uid = ?
#                     ORDER BY aca.association_order
#                 """

#             df = conn.execute(query, [atlas_uid]).df()

#         except Exception as e:
#             logger.error(f"Error querying atlas {atlas_uid}: {e}")
#             return pd.DataFrame()
        
#         finally:
#             # Detach if attached
#             if is_project_db and main_db_path:
#                 try:
#                     conn.execute("DETACH main_db")
#                 except:
#                     pass

#     if df.empty:
#         logger.warning(f"No compounds found for atlas {atlas_uid}")
#     else:
#         df['label'] = df['compound_name'] if 'compound_name' in df.columns else ''
#         logger.info(f"Retrieved {len(df)} compounds for atlas: {df['atlas_name'].iloc[0]}")

#     return df



# def create_atlas_from_compounds(atlas_compounds_df: pd.DataFrame, atlas_name: str, 
#                                atlas_description: str, atlas_type: str,
#                                chromatography: str, polarity: str,
#                                config: Dict) -> Tuple[str, str]:
#     """
#     Create a new atlas from a DataFrame of compounds.
    
#     Args:
#         atlas_compounds_df: DataFrame with compound and RT/MZ reference data
#         atlas_name: Name for the new atlas
#         atlas_description: Description for the new atlas
#         config: Configuration dictionary
    
#     Returns:
#         Tuple of (atlas_uid, atlas_name)
#     """
#     db_path = config["ENV"]["PATHS"]["main_database"]
    
#     # Generate atlas UID
#     atlas_uid = _generate_uid("atlas", decorator=atlas_type.lower())
    
#     prov = ldt.get_provenance()
    
#     with get_db_connection(db_path) as conn:
#         # Create atlas
#         conn.execute("""
#             INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?)
#         """, (
#             atlas_uid,
#             atlas_name,
#             atlas_description,
#             chromatography,
#             polarity,
#             atlas_type,
#             prov["analyst"],
#             prov["timestamp"]
#         ))
        
#         # Get existing compounds
#         existing_compounds = {}
#         existing_compound_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
#         for row in existing_compound_result:
#             existing_compounds[row[0]] = row[1]
#         logger.info(f"Found {len(existing_compounds)} existing compounds in database")
        
#         # Process compounds and create/find RT/MZ references
#         association_order = 0
#         compounds_processed = 0
#         compounds_skipped = 0
#         references_created = 0
#         references_reused = 0
        
#         for _, row in atlas_compounds_df.iterrows():
#             inchi_key = row.get('inchi_key', '')
#             if not inchi_key:
#                 continue
            
#             # Check if compound exists in database
#             compound_uid = existing_compounds.get(inchi_key)
#             if not compound_uid:
#                 logger.warning(f"Compound {inchi_key} not found in database, skipping")
#                 compounds_skipped += 1
#                 continue
            
#             # Extract RT/MZ data from input
#             rt_peak = row.get('rt_peak')
#             rt_min = row.get('rt_min')
#             rt_max = row.get('rt_max')
#             mz = row.get('mz')
#             adduct = row.get('adduct', '')
#             mz_tolerance = row.get('mz_tolerance', 5.0)
            
#             # Validate that we have the required RT/MZ data
#             if pd.isna(rt_peak) or pd.isna(mz):
#                 logger.warning(f"Compound {inchi_key} missing RT or MZ data, skipping")
#                 compounds_skipped += 1
#                 continue
            
#             # Check for existing reference with identical data
#             existing_ref_query = """
#                 SELECT mz_rt_reference_uid 
#                 FROM mz_rt_references 
#                 WHERE compound_uid = ? 
#                 AND chromatography = ? 
#                 AND polarity = ? 
#                 AND adduct = ?
#                 AND ABS(rt_peak - ?) < 0.001 
#                 AND ABS(mz - ?) < 0.001
#                 AND ABS(mz_tolerance - ?) < 0.001
#             """
            
#             existing_ref = conn.execute(existing_ref_query, [
#                 compound_uid, chromatography, polarity, adduct,
#                 float(rt_peak), float(mz), float(mz_tolerance)
#             ]).fetchone()
            
#             if existing_ref:
#                 # Use existing identical reference
#                 mz_rt_reference_uid = existing_ref[0]
#                 references_reused += 1
#                 logger.debug(f"Reusing existing RT/MZ reference for compound {inchi_key}")
#             else:
#                 # Create new RT/MZ reference
#                 mz_rt_reference_uid = _generate_uid("mz_rt_reference")
                
#                 # Prepare RT bounds
#                 if pd.isna(rt_min):
#                     rt_min = float(rt_peak) - 0.5
#                 if pd.isna(rt_max):
#                     rt_max = float(rt_peak) + 0.5
                
#                 conn.execute("""
#                     INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     mz_rt_reference_uid,
#                     compound_uid,
#                     float(rt_peak),
#                     float(rt_min),
#                     float(rt_max),
#                     float(mz),
#                     float(mz_tolerance),
#                     adduct,
#                     chromatography,
#                     polarity,
#                     row.get('confidence', 'Unknown'),
#                     'atlas_creation',  # source
#                     prov["analyst"],
#                     prov["timestamp"]
#                 ))
                
#                 references_created += 1
#                 logger.debug(f"Created new RT/MZ reference for compound {inchi_key}")
            
#             # Create atlas-compound association
#             assoc_uid = _generate_uid("association")
#             conn.execute("""
#                 INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 assoc_uid,
#                 atlas_uid,
#                 compound_uid,
#                 mz_rt_reference_uid,
#                 association_order,
#                 prov["analyst"],
#                 prov["timestamp"]
#             ))
            
#             association_order += 1
#             compounds_processed += 1
    
    
#     logger.info(f"Created atlas '{atlas_name}' with UID: {atlas_uid}")
#     logger.info(f"  Processed {compounds_processed} compounds successfully")
#     logger.info(f"  Skipped {compounds_skipped} compounds (missing from database or invalid data)")
#     logger.info(f"  Created {references_created} new RT/MZ references")
#     logger.info(f"  Reused {references_reused} existing RT/MZ references")
#     logger.info(f"  Added {association_order} compound associations")
    
#     return atlas_uid, atlas_name

# def generate_targeted_analysis_summary(project_db_path: str, config: Dict, analysis_uid: str) -> pd.DataFrame:
#     """
#     Generate summary of targeted analysis results from database.
    
#     Args:
#         project_db_path: Path to project database
#         config: Configuration dictionary
#         analysis_uid: UID of targeted analysis
    
#     Returns:
#         DataFrame with analysis summary
#     """
#     with get_db_connection(project_db_path) as conn:
#         # Get targeted analysis data
#         query = """
#             SELECT 
#                 ta.*,
#                 c.formula,
#                 c.mono_isotopic_molecular_weight
#             FROM targeted_analysis ta
#             LEFT JOIN compounds c ON ta.compound_uid = c.compound_uid
#             WHERE ta.analysis_uid = ?
#             ORDER BY ta.pre_rt_peak
#         """
        
#         try:
#             df = conn.execute(query, [analysis_uid]).df()
#         except Exception as e:
#             # Fallback query if compounds table doesn't exist in project DB
#             logger.warning(f"Could not join with compounds table: {e}")
#             query = """
#                 SELECT * FROM targeted_analysis 
#                 WHERE analysis_uid = ?
#                 ORDER BY pre_rt_peak
#             """
#             df = conn.execute(query, [analysis_uid]).df()
    
#     if df.empty:
#         logger.warning(f"No targeted analysis results found for analysis_uid: {analysis_uid}")
#     else:
#         logger.info(f"Retrieved {len(df)} compounds from targeted analysis {analysis_uid}")
    
#     return df

# def _calculate_msms_quality_score_from_notes(row: pd.Series) -> float:
#     """Calculate MS/MS quality score from analyst notes."""
#     ms2_notes = str(row.get('ms2_notes', 'no selection')).lower()
    
#     # Extract numeric score from notes if present
#     if '1.0' in ms2_notes:
#         return 3.0
#     elif '0.5' in ms2_notes:
#         return 1.5
#     elif '0.0' in ms2_notes:
#         return 0.0
#     elif '-1.0' in ms2_notes:
#         return 0.0
#     elif 'no selection' in ms2_notes:
#         return 0.0
#     else:
#         # Fallback to score if available
#         score = row.get('best_ms2_score', 0.0)
#         if pd.notna(score) and score > 0:
#             return min(3.0, float(score) * 3.0)
#         return 0.0

# def _calculate_mz_quality_score(row: pd.Series) -> float:
#     """Calculate m/z quality score based on PPM error."""
#     ppm_error = row.get('best_eic_ppm_error', None)
    
#     if pd.isna(ppm_error) or ppm_error is None:
#         return 0.0
    
#     ppm_error = abs(float(ppm_error))
    
#     # Score based on PPM error
#     if ppm_error <= 2.0:
#         return 2.0
#     elif ppm_error <= 5.0:
#         return 1.5
#     elif ppm_error <= 10.0:
#         return 1.0
#     elif ppm_error <= 20.0:
#         return 0.5
#     else:
#         return 0.0

# def _calculate_rt_quality_score(row: pd.Series) -> float:
#     """Calculate RT quality score based on RT error."""
#     rt_error = row.get('best_eic_rt_error', None)
    
#     if pd.isna(rt_error) or rt_error is None:
#         return 0.0
    
#     rt_error = abs(float(rt_error))
    
#     # Score based on RT error (in minutes)
#     if rt_error <= 0.1:
#         return 2.0
#     elif rt_error <= 0.2:
#         return 1.5
#     elif rt_error <= 0.5:
#         return 1.0
#     elif rt_error <= 1.0:
#         return 0.5
#     else:
#         return 0.0

# def _determine_msi_level(msms_quality: float, mz_quality: float, rt_quality: float) -> str:
#     """Determine MSI identification level based on quality scores."""
#     total_score = msms_quality + mz_quality + rt_quality
    
#     if total_score >= 6.0 and msms_quality >= 2.0:
#         return "MSI Level 2"
#     elif total_score >= 4.0 and mz_quality >= 1.0:
#         return "MSI Level 3"
#     elif total_score >= 2.0:
#         return "MSI Level 4"
#     else:
#         return "MSI Level 5"

# def _create_empty_summary_from_atlas(atlas_dataframe: pd.DataFrame, analysis_uid: str, config: Dict) -> pd.DataFrame:
#     """Create empty summary rows for all atlas compounds."""
#     empty_rows = []
    
#     for _, row in atlas_dataframe.iterrows():
#         empty_row = {
#             'analysis_uid': analysis_uid,
#             'compound_uid': row.get('compound_uid', ''),
#             'inchi_key': row.get('inchi_key', ''),
#             'compound_name': row.get('compound_name', row.get('label', '')),
#             'formula': row.get('formula', ''),
#             'mono_isotopic_molecular_weight': row.get('mono_isotopic_molecular_weight', ''),
#             'adduct': row.get('adduct', ''),
#             'pre_rt_peak': row.get('rt_peak', 0.0),
#             'pre_rt_min': row.get('rt_min', 0.0),
#             'pre_rt_max': row.get('rt_max', 0.0),
#             'pre_mz': row.get('mz', 0.0),
#             'ms1_notes': 'keep',
#             'ms2_notes': 'no selection',
#             'analyst_notes': '',
#             'identification_notes': '',
#             # Empty detection fields
#             'best_eic_intensity': 0.0,
#             'best_eic_file': '',
#             'best_eic_rt': 0.0,
#             'best_eic_mz': 0.0,
#             'best_eic_ppm_error': 0.0,
#             'best_eic_rt_error': 0.0,
#             'best_ms2_file': '',
#             'best_ms2_score': 0.0,
#             'best_ms2_num_matches': 0,
#             'best_ms2_matched_fragments': '',
#             'post_rt_peak': row.get('rt_peak', 0.0),
#             'post_rt_min': row.get('rt_min', 0.0),
#             'post_rt_max': row.get('rt_max', 0.0)
#         }
#         empty_rows.append(empty_row)
    
#     return pd.DataFrame(empty_rows)

# def _add_missing_compounds_to_summary(base_summary: pd.DataFrame, atlas_dataframe: pd.DataFrame, 
#                                      analysis_uid: str, config: Dict) -> pd.DataFrame:
#     """Add missing compounds from atlas to summary as empty rows."""
#     # Find compounds in atlas but not in summary
#     atlas_inchi_keys = set(atlas_dataframe['inchi_key'].unique())
#     summary_inchi_keys = set(base_summary['inchi_key'].unique())
#     missing_inchi_keys = atlas_inchi_keys - summary_inchi_keys
    
#     if missing_inchi_keys:
#         logger.info(f"Adding {len(missing_inchi_keys)} missing compounds to report")
        
#         missing_compounds = atlas_dataframe[atlas_dataframe['inchi_key'].isin(missing_inchi_keys)]
#         empty_summary = _create_empty_summary_from_atlas(missing_compounds, analysis_uid, config)
        
#         # Combine with existing summary
#         combined_summary = pd.concat([base_summary, empty_summary], ignore_index=True)
#         return combined_summary
    
#     return base_summary

# def _save_report_with_grouped_headers(report_df: pd.DataFrame, output_path: str, atlas_uid: str):
#     """Save report to Excel with grouped headers."""
#     try:
#         import openpyxl
#         from openpyxl.utils.dataframe import dataframe_to_rows
#         from openpyxl.styles import Font, Alignment, PatternFill
#         from openpyxl.utils import get_column_letter
        
#         # Create workbook and worksheet
#         wb = openpyxl.Workbook()
#         ws = wb.active
#         ws.title = "Targeted Analysis Report"
        
#         # Define column groups and their headers
#         column_groups = [
#             ('Compound Information', ['index', 'identified_metabolite', 'label', 'isomer_compound', 'isomer_inchi_keys', 'formula', 'polarity', 'mono_isotopic_molecular_weight', 'inchi_key', 'adduct']),
#             ('Quality Scores', ['msms_quality', 'mz_quality', 'rt_quality', 'total_score', 'msi_level']),
#             ('Analyst Annotations', ['ms1_notes', 'ms2_notes', 'analyst_notes', 'identification_notes']),
#             ('Detection Results', ['max_intensity', 'max_intensity_file', 'max_intensity_rt', 'best_msms_file', 'best_msms_rt', 'best_msms_num_matching_ions', 'best_msms_matching_ions', 'best_msms_score']),
#             ('Measurement Accuracy', ['mz_theoretical', 'mz_measured', 'mz_error', 'rt_peak_theoretical', 'rt_peak_measured', 'rt_min_measured', 'rt_max_measured', 'rt_error'])
#         ]
        
#         # Add title and metadata
#         ws['A1'] = f"Targeted Analysis Report - Atlas: {atlas_uid}"
#         ws['A1'].font = Font(bold=True, size=14)
#         ws['A2'] = f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
#         ws['A3'] = f"Total Compounds: {len(report_df)}"
        
#         # Start data at row 5
#         start_row = 5
#         current_col = 1
        
#         # Create grouped headers
#         for group_name, columns in column_groups:
#             # Check which columns exist in the dataframe
#             existing_columns = [col for col in columns if col in report_df.columns]
#             if not existing_columns:
#                 continue
            
#             # Group header
#             start_col = current_col
#             end_col = current_col + len(existing_columns) - 1
            
#             if start_col == end_col:
#                 ws.cell(row=start_row, column=start_col).value = group_name
#             else:
#                 ws.merge_cells(start_row=start_row, start_column=start_col, end_row=start_row, end_column=end_col)
#                 ws.cell(row=start_row, column=start_col).value = group_name
            
#             ws.cell(row=start_row, column=start_col).font = Font(bold=True)
#             ws.cell(row=start_row, column=start_col).fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
#             ws.cell(row=start_row, column=start_col).alignment = Alignment(horizontal="center")
            
#             # Column headers
#             for i, col in enumerate(existing_columns):
#                 ws.cell(row=start_row + 1, column=current_col + i).value = col
#                 ws.cell(row=start_row + 1, column=current_col + i).font = Font(bold=True)
            
#             current_col += len(existing_columns)
        
#         # Add data rows
#         data_start_row = start_row + 2
#         for row_idx, (_, row) in enumerate(report_df.iterrows()):
#             current_col = 1
#             for group_name, columns in column_groups:
#                 existing_columns = [col for col in columns if col in report_df.columns]
#                 for col in existing_columns:
#                     ws.cell(row=data_start_row + row_idx, column=current_col).value = row[col]
#                     current_col += 1
        
#         # Auto-adjust column widths
#         for col in ws.columns:
#             max_length = 0
#             column = col[0].column_letter
#             for cell in col:
#                 try:
#                     if len(str(cell.value)) > max_length:
#                         max_length = len(str(cell.value))
#                 except:
#                     pass
#             adjusted_width = min(max_length + 2, 50)
#             ws.column_dimensions[column].width = adjusted_width
        
#         # Save workbook
#         wb.save(output_path)
        
#     except ImportError:
#         logger.warning("openpyxl not available, saving as CSV instead")
#         report_df.to_csv(output_path.replace('.xlsx', '.csv'), index=False)
#     except Exception as e:
#         logger.error(f"Error saving Excel file: {e}")
#         # Fallback to CSV
#         report_df.to_csv(output_path.replace('.xlsx', '.csv'), index=False)

# def validate_targeted_analysis_data(project_db_path: str, analysis_uid: str = None) -> bool:
#     """
#     Validate targeted analysis results in database.
#     """
#     with get_db_connection(project_db_path) as conn:
        
#         # Get latest analysis if no specific UID provided
#         if analysis_uid:
#             condition = "WHERE analysis_uid = ?"
#             params = [analysis_uid]
#         else:
#             # Get most recent analysis
#             latest_result = conn.execute("""
#                 SELECT analysis_uid FROM targeted_analysis 
#                 ORDER BY analysis_timestamp DESC LIMIT 1
#             """).fetchone()
#             if not latest_result:
#                 return False
#             analysis_uid = latest_result[0]
#             condition = "WHERE analysis_uid = ?"
#             params = [analysis_uid]
        
#         # Validation queries
#         validation_query = f"""
#         SELECT 
#             COUNT(*) as total_records,
#             COUNT(CASE WHEN pre_rt_peak IS NOT NULL THEN 1 END) as records_with_rt,
#             COUNT(CASE WHEN pre_mz IS NOT NULL THEN 1 END) as records_with_mz,
#             COUNT(CASE WHEN best_eic_file IS NOT NULL AND best_eic_file != '' THEN 1 END) as records_with_eic,
#             COUNT(CASE WHEN avg_eic_rt IS NOT NULL THEN 1 END) as records_with_avg_eic_rt,
#             COUNT(CASE WHEN ms2_files_with_data > 0 THEN 1 END) as records_with_ms2,
#             COUNT(CASE WHEN is_rt_modified = true THEN 1 END) as modified_records
#         FROM targeted_analysis
#         {condition}
#         """
        
#         result = conn.execute(validation_query, params).fetchone()
    
#     if result and result[0] > 0:  # At least one record exists
#         total_records = result[0]
#         logger.info(f"Validation successful for analysis {analysis_uid}:")
#         logger.info(f"  Total records: {total_records}")
#         logger.info(f"  Records with RT data: {result[1]}")
#         logger.info(f"  Records with m/z data: {result[2]}")
#         logger.info(f"  Records with EIC data: {result[3]}")
#         logger.info(f"  Records with MS2 data: {result[5]}")
#         logger.info(f"  Modified records: {result[6]}")
#         return True
#     else:
#         logger.error(f"Validation failed for analysis {analysis_uid}: No records found")
#         return False

# def clone_and_modify_atlas(source_db_path: str, dest_db_path: str, source_atlas_uid: str,
#                           config: Dict, compound_updates: Dict[str, Dict],
#                           use_experimental_table: bool = True,
#                           new_atlas_description: str = "Modified Atlas") -> str:
#     """
#     Clone an atlas and apply compound modifications.
    
#     Args:
#         source_db_path: Path to source database
#         dest_db_path: Path to destination database  
#         source_atlas_uid: UID of source atlas to clone
#         config: Configuration dictionary
#         compound_updates: Dict mapping compound_uid to update dictionary
#         use_experimental_table: Whether to use experimental or reference table
#         new_atlas_description: Description for new atlas
    
#     Returns:
#         UID of new atlas
#     """
#     logger.info(f"Cloning atlas {source_atlas_uid} with {len(compound_updates)} modifications...")
    
#     # Generate new atlas UID
#     new_atlas_uid = _generate_uid("analyzed_atlas")
#     prov = ldt.get_provenance()
    
#     with get_db_connection(source_db_path) as source_conn:
#         with get_db_connection(dest_db_path) as dest_conn:
            
#             # Get source atlas metadata
#             source_atlas = source_conn.execute("""
#                 SELECT atlas_name, chromatography, polarity 
#                 FROM atlases WHERE atlas_uid = ?
#             """, [source_atlas_uid]).fetchone()
            
#             if not source_atlas:
#                 raise ValueError(f"Source atlas {source_atlas_uid} not found")
            
#             atlas_name, chromatography, polarity = source_atlas
#             new_atlas_name = f"{atlas_name} - {new_atlas_description}"
            
#             # Create new atlas
#             dest_conn.execute("""
#                 INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 new_atlas_uid,
#                 new_atlas_name,
#                 new_atlas_description,
#                 chromatography,
#                 polarity,
#                 prov["analyst"],
#                 prov["timestamp"]
#             ))
            
#             # Get source compounds and associations
#             source_data = source_conn.execute("""
#                 SELECT 
#                     aca.compound_uid,
#                     aca.association_order,
#                     c.name as compound_name,
#                     c.inchi_key,
#                     mzrt.rt_peak,
#                     mzrt.rt_min,
#                     mzrt.rt_max,
#                     mzrt.mz,
#                     mzrt.mz_tolerance,
#                     mzrt.adduct,
#                     mzrt.chromatography,
#                     mzrt.polarity
#                 FROM atlas_compound_associations aca
#                 LEFT JOIN compounds c ON aca.compound_uid = c.compound_uid
#                 LEFT JOIN mz_rt_experimental mzrt ON aca.mz_rt_reference_uid = mzrt.mz_rt_experimental_uid
#                 WHERE aca.atlas_uid = ?
#                 ORDER BY aca.association_order
#             """, [source_atlas_uid]).fetchall()
            
#             # Create modified experimental entries and associations
#             for row in source_data:
#                 compound_uid = row[0]
#                 association_order = row[1]
                
#                 # Apply modifications if they exist
#                 if compound_uid in compound_updates:
#                     updates = compound_updates[compound_uid]
#                     rt_peak = updates.get('rt_peak', row[4])
#                     rt_min = updates.get('rt_min', row[5]) 
#                     rt_max = updates.get('rt_max', row[6])
#                     ms1_notes = updates.get('ms1_notes', 'keep')
#                     ms2_notes = updates.get('ms2_notes', 'no selection')
#                 else:
#                     rt_peak = row[4]
#                     rt_min = row[5]
#                     rt_max = row[6]
#                     ms1_notes = 'keep'
#                     ms2_notes = 'no selection'
                
#                 # Create new experimental entry
#                 exp_uid = _generate_uid("mz_rt_experimental")
#                 dest_conn.execute("""
#                     INSERT INTO mz_rt_experimental VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     exp_uid,
#                     compound_uid,
#                     rt_peak,
#                     rt_min,
#                     rt_max,
#                     ms1_notes,
#                     ms2_notes,
#                     row[7],  # mz
#                     row[8],  # mz_tolerance
#                     row[9],  # adduct
#                     row[10], # chromatography
#                     row[11], # polarity
#                     True,    # rt_correction_applied
#                     0.0,     # rt_shift
#                     None,    # source_mz_rt_reference_uid
#                     prov["analyst"],
#                     prov["timestamp"]
#                 ))
                
#                 # Create association
#                 assoc_uid = _generate_uid("association")
#                 dest_conn.execute("""
#                     INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
#                 """, (
#                     assoc_uid,
#                     new_atlas_uid,
#                     compound_uid,
#                     exp_uid,
#                     association_order,
#                     prov["analyst"],
#                     prov["timestamp"]
#                 ))
    
#     logger.info(f"Created new atlas: {new_atlas_uid}")
#     return new_atlas_uid

# def get_lcmsruns_from_db(project_db_path: str, file_types: List[str] = None) -> List[str]:
#     """Get experimental files from project database."""
    
#     if file_types is None:
#         file_types = ['experimental', 'istd', 'exctrl']
    
#     with get_db_connection(project_db_path) as conn:

#         # Get experimental files (excluding QC)
#         placeholders = ','.join(['?' for _ in file_types])
#         result = conn.execute(f"""
#             SELECT file_path 
#             FROM lcmsruns 
#             WHERE file_type IN ({placeholders})
#             ORDER BY filename
#         """, file_types).fetchall()
        
#     file_paths = [row[0] for row in result]
#     logger.info(f"Retrieved {len(file_paths)} files in database")
    
#     return file_paths

# def get_atlas_compounds_with_metadata(project_db_path: str, main_db_path: str, atlas_uid: str) -> pd.DataFrame:
#     """
#     Get atlas compounds from project database and enrich with metadata from main database.
#     This is specifically for RT-corrected atlases in project databases.
#     Uses external database attachment for compound metadata.
#     """
#     # Use the updated get_atlas_compounds_table with external attachment
#     enriched_df = get_atlas_compounds_table(project_db_path, atlas_uid, main_db_path)
    
#     if enriched_df.empty:
#         return pd.DataFrame()
    
#     # Add label column for compatibility with feature_tools
#     enriched_df['label'] = enriched_df['compound_name']
    
#     logger.info(f"Retrieved {len(enriched_df)} compounds with metadata using external database attachment")
    
#     return enriched_df

# def save_targeted_analysis_from_analysis_project(
#     analysis_project,
#     project_name: str,
#     atlas_uid: str
# ) -> str:
#     """
#     Save targeted analysis results using AnalysisProject class.
#     This replaces the old deposit_targeted_analysis_from_plot_data function.
#     """
#     logger.info(f"Saving targeted analysis results for project '{project_name}' using class-based approach...")
    
#     # Use the class method to save to database
#     analysis_uid = analysis_project.save_to_database(project_name, atlas_uid)
    
#     # Validate the saved data
#     validated = validate_targeted_analysis_data(analysis_project.project_db_path, analysis_uid)
#     if not validated:
#         logger.warning(f"Validation failed for targeted analysis entry {analysis_uid}")
#         return None
    
#     logger.info(f"Successfully saved targeted analysis with UID: {analysis_uid}")
#     return analysis_uid

# def generate_comprehensive_targeted_analysis_report(project_db_path: str, config: Dict, 
#                                                    analysis_uid: str, atlas_dataframe: pd.DataFrame,
#                                                    post_analysis_atlas_uid: str,
#                                                    output_path: str = None, include_missing_compounds: bool = False) -> pd.DataFrame:
#     """
#     Generate a comprehensive targeted analysis report matching the specified Excel format.
#     Updated to work with consistent per-file MS2 data structure.
#     """
#     main_db_path = config["ENV"]["PATHS"]["main_database"]
    
#     # Get the base targeted analysis summary
#     base_summary = generate_targeted_analysis_summary(project_db_path, config, analysis_uid)
    
#     if base_summary.empty:
#         logger.warning(f"No targeted analysis results found for analysis_uid {analysis_uid}")
#         if atlas_dataframe is not None and include_missing_compounds:
#             logger.info("Creating empty rows for all atlas compounds...")
#             base_summary = _create_empty_summary_from_atlas(atlas_dataframe, analysis_uid, config)
#         else:
#             return pd.DataFrame()
#     elif atlas_dataframe is not None and include_missing_compounds:
#         # Add missing compounds from atlas as empty rows
#         base_summary = _add_missing_compounds_to_summary(base_summary, atlas_dataframe, analysis_uid, config)
    
#     # Connect to both databases
#     with get_db_connection(project_db_path) as conn_proj:
#         with get_db_connection(main_db_path) as conn_main:
            
#             report_rows = []
            
#             logger.info(f"Generating comprehensive report for {len(base_summary)} compounds...")
            
#             for idx, row in tqdm(base_summary.iterrows(), total=len(base_summary), desc="Processing compounds"):
#                 try:
#                     # Basic compound information
#                     compound_uid = row['compound_uid']
#                     inchi_key = row['inchi_key']
#                     compound_name = row['compound_name']
                    
#                     # Calculate stats using simplified methods
#                     msms_quality = _calculate_msms_quality_score_from_notes(row)
#                     mz_quality = _calculate_mz_quality_score(row)
#                     rt_quality = _calculate_rt_quality_score(row)
#                     total_score = msms_quality + mz_quality + rt_quality
#                     msi_level = _determine_msi_level(msms_quality, mz_quality, rt_quality)
                    
#                     # Find isomers using simplified approach
#                     isomer_info = _find_overlapping_compounds_simple(conn_main, row, base_summary)
                    
#                     # Build the report row with consistent field names
#                     report_row = {
#                         'index': idx,
#                         'identified_metabolite': compound_name,
#                         'label': compound_name,
#                         'isomer_compound': isomer_info['compound_names'],
#                         'isomer_inchi_keys': isomer_info['inchi_keys'],
#                         'formula': row.get('formula', ''),
#                         'polarity': row.get('polarity', ''),
#                         'mono_isotopic_molecular_weight': row.get('mono_isotopic_molecular_weight', ''),
#                         'inchi_key': inchi_key,
#                         'adduct': row.get('adduct', ''),
#                         'msms_quality': msms_quality,
#                         'mz_quality': mz_quality,
#                         'rt_quality': rt_quality,
#                         'total_score': total_score,
#                         'msi_level': msi_level,
#                         'ms1_notes': row.get('ms1_notes', 'keep'),
#                         'ms2_notes': row.get('ms2_notes', 'no selection'),
#                         'analyst_notes': row.get('analyst_notes', ''),
#                         'identification_notes': row.get('identification_notes', ''),
#                         'max_intensity': row.get('best_eic_intensity', ''),
#                         'max_intensity_file': row.get('best_eic_file', ''),
#                         'max_intensity_rt': row.get('best_eic_rt', ''),
#                         'best_msms_file': row.get('best_ms2_file', ''),
#                         'best_msms_rt': row.get('best_ms2_rt_peak', ''),
#                         'best_msms_num_matching_ions': row.get('best_ms2_num_matches', ''),
#                         'best_msms_matching_ions': row.get('best_ms2_matched_fragments', ''),
#                         'best_msms_score': row.get('best_ms2_score', ''),
#                         'mz_theoretical': row.get('pre_mz', ''),
#                         'mz_measured': row.get('best_eic_mz', ''),
#                         'mz_error': row.get('best_eic_ppm_error', ''),
#                         'rt_peak_theoretical': row.get('pre_rt_peak', ''),
#                         'rt_peak_measured': row.get('post_rt_peak', ''),
#                         'rt_min_measured': row.get('post_rt_min', ''),
#                         'rt_max_measured': row.get('post_rt_max', ''),
#                         'rt_error': row.get('best_eic_rt_error', '')
#                     }
                    
#                     report_rows.append(report_row)
                    
#                 except Exception as e:
#                     logger.error(f"Error processing compound {compound_name} ({inchi_key}): {e}")
#                     continue
    
#     # Create DataFrame and sort
#     report_df = pd.DataFrame(report_rows)
    
#     if not report_df.empty:
#         # Convert rt_peak_theoretical to numeric for proper sorting
#         report_df['rt_peak_theoretical_num'] = pd.to_numeric(report_df['rt_peak_theoretical'], errors='coerce')
#         report_df = report_df.sort_values('rt_peak_theoretical_num', ascending=True, na_position='last')
#         report_df = report_df.drop(columns=['rt_peak_theoretical_num'])
#         report_df = report_df.reset_index(drop=True)
#         report_df['index'] = range(len(report_df))
    
#     # Save to Excel if path provided
#     if output_path is not None and not report_df.empty:
#         try:
#             _save_report_with_grouped_headers(report_df, output_path, post_analysis_atlas_uid)
#             logger.info(f"Report saved to {output_path}")
#         except Exception as e:
#             logger.error(f"Error saving Excel file: {e}")
    
#     return report_df

# def _find_overlapping_compounds_simple(conn_main, current_row: pd.Series, all_compounds: pd.DataFrame) -> Dict[str, str]:
#     """
#     Find overlapping compounds using simplified approach.
#     Uses the isomers field if available, otherwise falls back to database lookup.
#     """
#     # First try to use the isomers field if it exists and is populated
#     isomers_json = current_row.get('isomers', None)
#     if isomers_json and isomers_json != 'null':
#         try:
#             import json
#             isomers = json.loads(isomers_json) if isinstance(isomers_json, str) else isomers_json
#             if isinstance(isomers, list) and isomers:
#                 compound_names = []
#                 inchi_keys = []
#                 for iso in isomers:
#                     name = iso.get('compound_name', 'Unknown')
#                     inchi = iso.get('inchi_key', '')
#                     if inchi and inchi != current_row.get('inchi_key', ''):
#                         compound_names.append(name)
#                         inchi_keys.append(inchi)
#                 return {
#                     'compound_names': '; '.join(compound_names),
#                     'inchi_keys': '; '.join(inchi_keys),
#                 }
#         except (json.JSONDecodeError, TypeError):
#             pass
    
#     # Fallback to empty if no isomers data
#     return {
#         'compound_names': '',
#         'inchi_keys': '',
#     }
