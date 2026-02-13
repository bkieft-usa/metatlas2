import pandas as pd
import numpy as np
import duckdb
import uuid
import sys
import os
import json
import sys
from pathlib import Path
from typing import Dict
from tqdm.notebook import tqdm
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
from IPython.display import display

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import lcmsruns_tools as lrt
import pubchem_retrieval as pcr
import load_tools as ldt
import logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('database_interact')

@contextmanager
def get_db_connection(db_path: str):
    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()

def get_atlas_compounds_table(database_path: str, atlas_uid: str, main_db_path: str = None) -> pd.DataFrame:
    """
    Extract all compound information for a given atlas UID from the database and return as a pandas DataFrame.
    Handles both main database (with mz_rt_references) and project database (with mz_rt_experimental) structures.
    Uses external database attachment for compound metadata when needed.
    """
    with get_db_connection(database_path) as conn:
    
        try:
            # Check if this is a project database by looking for mz_rt_experimental table
            is_project_db = conn.execute("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_name = 'mz_rt_experimental'
            """).fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking database type at {database_path}: {e}")
            return pd.DataFrame()

        # Attach main database if this is a project database and main_db_path provided
        try:
            if is_project_db and main_db_path:
                conn.execute(f"ATTACH '{main_db_path}' AS main_db")
                logger.info("Attached main database for compound metadata")
        except Exception as e:
            logger.error(f"Error attaching main database at {main_db_path}: {e}")

        try:
            if is_project_db:
                # Project database query - uses experimental data with external compound references
                query = """
                    SELECT
                        a.atlas_uid,
                        a.atlas_name,
                        a.atlas_description,
                        a.chromatography,
                        a.polarity,
                        aca.compound_uid,
                        COALESCE(main_db.compounds.name, '') AS compound_name,
                        COALESCE(main_db.compounds.inchi_key, '') AS inchi_key,
                        COALESCE(main_db.compounds.inchi, '') AS inchi,
                        mzrt_exp.adduct,
                        mzrt_exp.mz,
                        mzrt_exp.rt_peak,
                        mzrt_exp.rt_min,
                        mzrt_exp.rt_max,
                        mzrt_exp.mz_tolerance,
                        mzrt_exp.mz_rt_experimental_uid AS mz_rt_reference_uid,
                        mzrt_exp.rt_alignment_applied,
                        mzrt_exp.rt_shift,
                        mzrt_exp.source_mz_rt_reference_uid
                    FROM atlases a
                    JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                    LEFT JOIN main_db.compounds ON aca.compound_uid = main_db.compounds.compound_uid
                    LEFT JOIN mz_rt_experimental mzrt_exp ON aca.mz_rt_reference_uid = mzrt_exp.mz_rt_experimental_uid
                    WHERE a.atlas_uid = ?
                    ORDER BY aca.association_order, mzrt_exp.rt_peak
                """
            else:
                # Main database query - uses reference data with full compound info
                query = """
                    SELECT
                        a.atlas_uid,
                        a.atlas_name,
                        a.atlas_description,
                        a.chromatography,
                        a.polarity,
                        c.compound_uid,
                        c.name AS compound_name,
                        c.inchi_key,
                        c.inchi,
                        mzrt.adduct,
                        mzrt.mz,
                        mzrt.rt_peak,
                        mzrt.rt_min,
                        mzrt.rt_max,
                        mzrt.mz_tolerance,
                        mzrt.mz_rt_reference_uid,
                        FALSE AS rt_alignment_applied,
                        NULL AS rt_shift,
                        NULL AS source_mz_rt_reference_uid
                    FROM atlases a
                    JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                    JOIN compounds c ON aca.compound_uid = c.compound_uid
                    LEFT JOIN mz_rt_references mzrt ON aca.mz_rt_reference_uid = mzrt.mz_rt_reference_uid
                    WHERE a.atlas_uid = ?
                    ORDER BY aca.association_order, mzrt.rt_peak
                """

            df = conn.execute(query, [atlas_uid]).df()

        except Exception as e:
            logger.error(f"Error preparing query for atlas {atlas_uid}: {e}")
            return pd.DataFrame()
        
        # Detach main database if attached
        if is_project_db and main_db_path:
            conn.execute("DETACH main_db")

    if df.empty:
        logger.warning(f"No compounds found for atlas {atlas_uid}")
        return pd.DataFrame()
    else:
        db_type = "project" if is_project_db else "main"
        chromatography = ldt.detect_atlas_input_chromatography(df)
        polarity = ldt.detect_atlas_input_polarity(df)
        df['label'] = df['compound_name'] if 'compound_name' in df.columns else ''
        logger.info(f"Retrieved {len(df)} compounds from {db_type} database for atlas: {df['atlas_name'].iloc[0]} ({df['atlas_uid'].iloc[0]})")

    return df

def check_rt_and_analysis_numbers(project_db_path: str, rt_alignment_number: int, analysis_number: int) -> None:
    """
    Check if the provided RT alignment number and analysis number already exist in the project database.
    If they do, raise an error to prevent overwriting existing data.
    """
    with get_db_connection(project_db_path) as conn:
        rt_exists = conn.execute("""
            SELECT COUNT(*) 
            FROM rt_alignment 
            WHERE rt_alignment_number = ?
        """, [rt_alignment_number]).fetchone()[0] > 0

        analysis_exists = conn.execute("""
            SELECT COUNT(*) 
            FROM mz_rt_experimental 
            WHERE analysis_number = ?
            AND rt_alignment_number = ?
        """, [analysis_number, rt_alignment_number]).fetchone()[0] > 0

    if rt_exists and analysis_exists:
        raise ValueError(f"RT alignment number {rt_alignment_number} and analysis number {analysis_number} already exists in the project database. Please increment one.")
    elif rt_exists and not analysis_exists:
        logger.info(f"Creating new analysis ({analysis_number}) for existing RT alignment ({rt_alignment_number}).")
    elif not rt_exists and analysis_exists:
        pass
    elif not rt_exists and not analysis_exists:
        logger.info(f"Creating new RT alignment ({rt_alignment_number}) and new analysis ({analysis_number}).")

    return

def create_project_database(project_db_path: str, rt_alignment_number: int, analysis_number: int) -> None:
    """
    Create project-specific database with required tables.
    Never overwrites existing databases - requires analyst to increment analysis number.
    """
    project_db_path = Path(project_db_path)
    project_db_path.parent.mkdir(parents=True, exist_ok=True)

    if project_db_path.exists():
        check_rt_and_analysis_numbers(project_db_path, rt_alignment_number, analysis_number)
        logger.info(f"Project database already exists at {project_db_path}. Proceeding with existing database and new RT alignment {rt_alignment_number} and/or analysis {analysis_number}.")

    with get_db_connection(project_db_path) as conn:
        _create_database_tables(conn, db_type="project")
    logger.info(f"Project database created at {project_db_path}")
    return

def create_metatlas_database(db_path: str, overwrite_existing: bool) -> None:
    """
    Create main metatlas database with required tables.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if overwrite_existing and db_path.exists():
        logger.warning("Overwriting existing main database (overwrite_existing=True)")
        db_path.unlink()
        logger.info(f"Deleted existing database at {db_path}")
    elif not overwrite_existing and db_path.exists():
        logger.warning(f"Database already exists at {db_path}. Use overwrite_existing=True to replace it.")
        return

    with get_db_connection(db_path) as conn:
        _create_database_tables(conn, db_type="main")

    logger.info(f"Main metatlas database created at {db_path}")
    return

def save_lcmsruns_to_db(
    project_db_path: str,
    project_name: str,
    project_lcmsruns_path: str,
    new_lcmsruns: str = False
) -> Dict:
    """
    Save LCMS run files to project database and return file paths grouped by chromatography/polarity/analysis type.
    Chromatography and polarity are inferred from filenames.
    If the lcmsruns table exists and has rows, do not overwrite unless new_lcmsruns is true
    """

    with get_db_connection(project_db_path) as conn:

        files_by_group = lrt.get_project_files(project_lcmsruns_path)
        prov = ldt.get_provenance()

        # Check if lcmsruns table exists
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'lcmsruns'"
        ).fetchone()[0] > 0

        # Check if lcmsruns table has rows
        has_rows = False
        if table_exists:
            has_rows = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0] > 0
            if has_rows:
                if not new_lcmsruns:
                    logger.info("LCMS runs already exist in the database and new_lcmsruns is False. Proceeding with these files.")
                    conn.close()
                    print_files_summary(files_by_group)
                    return files_by_group
                elif new_lcmsruns:
                    logger.info(f"LCMS runs already exist in the database for {project_name} and new_lcmsruns is True. Creating a new table...")
                    conn.execute("DELETE FROM lcmsruns")
            else:
                logger.info(f"No LCMS runs detected in project database for {project_name}. Creating a new table...")
        else:
            logger.info(f"No LCMS runs detected in project database for {project_name}. Creating a new table...")
        
        total_files = 0
        for file_format, chrom_dict in files_by_group.items():
            for chrom, ms_level_dict in chrom_dict.items():
                for ms_level, pol_dict in ms_level_dict.items():
                    for pol, analysis_dict in pol_dict.items():
                        for file_type, file_list in analysis_dict.items():
                            for file_path in file_list:
                                filename = os.path.basename(file_path)
                                conn.execute(
                                    "INSERT INTO lcmsruns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        file_path,
                                        filename,
                                        file_format,
                                        file_type,
                                        chrom,
                                        ms_level,
                                        pol,
                                        prov["analyst"],
                                        prov["timestamp"],
                                    ),
                                )
                                total_files += 1

    logger.info(f"Saved {total_files} LCMS runs to database:")
    print_files_summary(files_by_group)

    return files_by_group

def print_files_summary(file_list: Dict[str, Any]):
    for file_format, chrom_dict in file_list.items():
        logger.info(f"File format: {file_format}")
        for chrom, ms_level_dict in chrom_dict.items():
            logger.info(f"  Chromatography: {chrom}")
            for ms_level, pol_dict in ms_level_dict.items():
                logger.info(f"    MS level: {ms_level}")
                for pol, filetype_dict in pol_dict.items():
                    logger.info(f"      Polarity: {pol}")
                    for file_type, files_list in filetype_dict.items():
                        logger.info(f"          {file_type}: {len(files_list)} files")

def list_available_atlases(db_path: str) -> pd.DataFrame:
    """List all available atlases with optional filtering."""

    try:
        with get_db_connection(db_path) as conn:
            conditions = ["1=1"]
            params = []
            
            query = f"""
            SELECT 
                a.atlas_uid,
                a.atlas_name,
                a.atlas_description,
                a.chromatography,
                a.polarity,
                a.created_date,
                COUNT(aca.compound_uid) as compound_count
            FROM atlases a
            LEFT JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
            WHERE {' AND '.join(conditions)}
            GROUP BY a.atlas_uid, a.atlas_name, a.atlas_description, 
                    a.chromatography, a.polarity, a.created_date
            ORDER BY a.atlas_name
            """
            
            df = conn.execute(query, params).df()
        
        return df
    except:
        #logger.warning(f"Did not find any atlases in database at {db_path}")
        return pd.DataFrame()

def get_most_recent_QC_atlas_id(config: Dict, database_path: str = "main"):
    with get_db_connection(database_path) as conn:

        query = """
        SELECT *
        FROM atlases
        WHERE atlas_name LIKE '%QC%' OR atlas_description LIKE '%QC%'
        ORDER BY created_date DESC
        LIMIT 1
        """
        df = conn.execute(query).df()

    if df.empty:
        logger.warning("No QC atlas found.")
        return None

    return df.iloc[0]['atlas_uid']

def validate_database(database_path: str, database_type: str = "main") -> None:
    """
    Validate the DuckDB database and print summary statistics.
    If database_path is "main", validates main DB structure.
    If database_path is a custom project DB, validates project DB structure.
    """
    logger.info("Database Validation:")

    db_path = Path(database_path)
    if not os.path.exists(db_path):
        logger.error(f"Database not found at: {db_path}")
        return

    with get_db_connection(db_path) as conn:
        if database_type == "main":
            # Main DB: compounds, references, atlases, associations
            compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
            references_count = conn.execute("SELECT COUNT(*) FROM mz_rt_references").fetchone()[0]
            atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
            atlas_compound_associations_count = conn.execute("SELECT COUNT(*) FROM atlas_compound_associations").fetchone()[0]
            atlas_info = list_available_atlases(db_path)

            method_combinations = conn.execute("""
                SELECT chromatography, polarity, COUNT(*) as reference_count 
                FROM mz_rt_references 
                GROUP BY chromatography, polarity
            """).fetchall()

            logger.info(f"   Compounds: {compounds_count}")
            logger.info(f"   RT/MZ References: {references_count}")
            logger.info(f"   Atlases: {atlases_count}")
            logger.info(f"   Atlas-Compound associations: {atlas_compound_associations_count}")

            if method_combinations:
                logger.info("   Method combinations:")
                for combo in method_combinations:
                    logger.info(f"      {combo[0]}/{combo[1]}: {combo[2]} references")

            if not atlas_info.empty:
                logger.info("   Available atlases:")
                for _, row in atlas_info.iterrows():
                    logger.info(f"      {row['atlas_uid']}")
                    logger.debug(f"            {row['atlas_name']}")
                    logger.debug(f"            {row['chromatography']} {row['polarity']}")
                    logger.debug(f"            {row['compound_count']} compounds")
                    logger.debug(f"            {row['created_date']}")
        else:
            # Project DB: atlases, targeted_analysis, rt_alignment, mz_rt_experimental
            atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
            # Count unique analysis_uid values in targeted_analysis table
            #targeted_count = len(conn.execute("SELECT DISTINCT analysis_uid FROM targeted_analysis").fetchall())
            rt_alignment_count = conn.execute("SELECT COUNT(*) FROM rt_alignment").fetchone()[0]
            mz_rt_exp_count = conn.execute("SELECT COUNT(*) FROM mz_rt_experimental").fetchone()[0]

            logger.info(f"   Atlases: {atlases_count}")
            #logger.info(f"   Targeted analyses: {targeted_count}")
            logger.info(f"   RT alignment models: {rt_alignment_count}")
            logger.info(f"   Experimental RT/MZ entries: {mz_rt_exp_count}")

            # List atlases
            atlas_info = list_available_atlases(db_path)
            if not atlas_info.empty:
                logger.info("   Available atlases:")
                for _, row in atlas_info.iterrows():
                    logger.info(f"      {row['atlas_uid']}")
                    logger.debug(f"            {row['atlas_name']}")
                    logger.debug(f"            {row['chromatography']} {row['polarity']}")
                    logger.debug(f"            {row['compound_count']} compounds")
                    logger.debug(f"            {row['created_date']}")

            # List targeted analyses
            # targeted_df = conn.execute("""
            #     SELECT analysis_uid, project_name, atlas_uid, COUNT(*) as compound_count
            #     FROM targeted_analysis
            #     GROUP BY analysis_uid, project_name, atlas_uid
            #     ORDER BY analysis_uid
            # """).df()
            # if not targeted_df.empty:
            #     logger.info("   Targeted analysis entries:")
            #     for _, row in targeted_df.iterrows():
            #         logger.info(f"      {row['analysis_uid']} ({row['project_name']}) - Atlas: {row['atlas_uid']} - {row['compound_count']} compounds")

            # List RT alignment models
            rt_df = conn.execute("""
                SELECT rt_alignment_uid, atlas_uid, model_type, polynomial_degree, r_squared, rmse, created_date
                FROM rt_alignment
                ORDER BY created_date DESC
            """).df()
            if not rt_df.empty:
                logger.info("   RT alignment models:")
                for _, row in rt_df.iterrows():
                    logger.info(f"      {row['rt_alignment_uid']} - Atlas: {row['atlas_uid']} - {row['model_type']} (deg={row['polynomial_degree']}, r2={row['r_squared']}, rmse={row['rmse']})")

            # List experimental RT/MZ entries summary
            exp_df = conn.execute("""
                SELECT chromatography, polarity, COUNT(*) as entry_count
                FROM mz_rt_experimental
                GROUP BY chromatography, polarity
            """).df()
            if not exp_df.empty:
                logger.info("   Experimental RT/MZ entries by method:")
                for _, row in exp_df.iterrows():
                    logger.info(f"      {row['chromatography']}/{row['polarity']}: {row['entry_count']} entries")

    return

def save_rt_alignment_model_to_db(qc_atlas_uid: str,
                                  project_db_path: Path, 
                                  rt_alignment_number: int,
                                  best_model: dict, 
                                  qc_files_info: pd.DataFrame, 
                                  modeling_data: list) -> str:
    """Save RT alignment model to project database."""
    rt_alignment_uid = _generate_uid("rt_alignment")
    
    # Extract project name from path
    qc_files = qc_files_info['file_path'].tolist()
    project_db_path = Path(project_db_path)
    project_name = project_db_path.stem.replace('.duckdb', '')
    prov = ldt.get_provenance()

    model_metadata = {
        "qc_files": [os.path.basename(f) for f in qc_files],
        "compounds_used": [d.get('compound_uid', '') for d in modeling_data],
        "alignment_timestamp": prov["timestamp"],
        "alignment_method": "polynomial_qc_based",
        "analyst": prov["analyst"],
        # Store PolynomialFeatures parameters
        "poly_degree": int(best_model['degree']),
        "poly_include_bias": bool(best_model.get('poly_features').include_bias if best_model.get('poly_features') else True),
        "poly_interaction_only": bool(best_model.get('poly_features').interaction_only if best_model.get('poly_features') else False),
        # Store LinearRegression parameters
        "model_intercept": float(best_model.get('intercept', 0.0)),
        "model_coefficients": best_model['coefficients'].tolist() if hasattr(best_model['coefficients'], 'tolist') else list(best_model['coefficients'])
    }
    
    with get_db_connection(project_db_path) as conn:
        conn.execute("""
            INSERT INTO rt_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rt_alignment_uid,
            project_name,
            rt_alignment_number,
            qc_atlas_uid,
            "polynomial",
            best_model['degree'],
            best_model['r2'],
            best_model['rmse'],
            json.dumps(best_model['coefficients'].tolist()),
            best_model['equation'],
            len(qc_files),
            len(modeling_data),
            prov["analyst"],
            prov["timestamp"],
            json.dumps(model_metadata)
        ))
    
    logger.info(f"RT alignment model saved to database with UID: {rt_alignment_uid}")

    best_model['rt_alignment_uid'] = rt_alignment_uid

    return rt_alignment_uid

def get_rt_alignment_model_from_db(project_db_path: Path, qc_atlas_uid: str, rt_alignment_number: int) -> dict:
    """
    Retrieve RT alignment model from project database and reconstruct sklearn objects.
    
    Args:
        project_db_path: Path to project database
        qc_atlas_uid: QC atlas UID to retrieve
        rt_alignment_number: RT alignment number to retrieve
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    
    project_db_path = Path(project_db_path)
    
    with get_db_connection(project_db_path) as conn:
        results = conn.execute("""
            SELECT *
            FROM rt_alignment
            WHERE rt_alignment_number = ? AND qc_atlas_uid = ?
        """, [rt_alignment_number, qc_atlas_uid]).fetchall()

        if len(results) == 0:
            return None
        elif len(results) > 1:
            raise ValueError(f"Multiple RT alignment models found for rt_alignment_number {rt_alignment_number} and qc_atlas_uid {qc_atlas_uid}. Something is wrong.")
        elif len(results) == 1:
            result = results[0]

    # Unpack result tuple and metadata
    metadata = json.loads(result[14])
    
    model_dict = {
        'rt_alignment_uid': result[0],
        'project_name': result[1],
        'rt_alignment_number': result[2],
        'qc_atlas_uid': result[3],
        'model_type': result[4],
        'degree': result[5],
        'r2': result[6],
        'rmse': result[7],
        'coefficients': np.array(json.loads(result[8])),
        'equation': result[9],
        'num_qc_files': result[10],
        'num_compounds': result[11],
        'analyst': result[12],
        'timestamp': result[13],
        'metadata': metadata,
    }
    
    logger.info(f"Retrieved and reconstructed RT alignment model {model_dict['rt_alignment_uid']} (RT alignment number {rt_alignment_number})")
    logger.info(f"  Model: {model_dict['model_type']} (degree={model_dict['degree']})")
    logger.info(f"  Performance: R²={model_dict['r2']:.4f}, RMSE={model_dict['rmse']:.4f}")
    logger.info(f"  Training data: {model_dict['num_qc_files']} QC files, {model_dict['num_compounds']} compounds")
    
    return model_dict

def add_compounds_to_db(input_df: pd.DataFrame, db_path: str, pubchem_cache_path: str, input_file_path: str = ""):
    """Add compounds and RT/MZ references to database using batch operations."""

    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}. Check path or create it first using create_metatlas_database().")
        raise FileNotFoundError(f"Database not found at {db_path}")

    unique_inchi_keys = input_df['inchi_key'].dropna().unique()
    logger.info(f"Adding {len(unique_inchi_keys)} compounds to database: {db_path}")
    
    pubchem_cache = pcr.load_or_create_pubchem_cache(pubchem_cache_path)
    prov = ldt.get_provenance()

    with get_db_connection(db_path) as conn:
        # Get existing compounds in one query
        existing_inchi_keys = set()
        existing_compounds_map = {}
        existing_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
        existing_inchi_keys = {row[0] for row in existing_result}
        existing_compounds_map = {row[0]: row[1] for row in existing_result}
        if existing_inchi_keys:
            logger.warning(f"Found {len(existing_inchi_keys)} existing compounds in database. Not creating duplicates.")

        # Prepare batch data
        compound_records = []
        reference_records = []
        compounds_created = 0
        compounds_skipped = 0

        logger.info("Preparing batch data...")
        for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Preparing compounds"):
            inchi_key = row.get('inchi_key')
            if pd.isna(inchi_key):
                continue
            
            chromatography = str(row.get('chromatography', 'Unknown'))
            polarity = str(row.get('polarity', 'Unknown'))
            compound_uid = None
            
            # Check if compound exists
            if inchi_key in existing_inchi_keys:
                compounds_skipped += 1
                compound_uid = existing_compounds_map.get(inchi_key)
            else:
                # Prepare new compound record
                compound_uid = _generate_uid("compound")
                compound_record = _prepare_compound_record(row, compound_uid, pubchem_cache, prov)
                compound_records.append(compound_record)
                compounds_created += 1
                
                # Add to existing sets for future duplicate checking in this session
                existing_inchi_keys.add(inchi_key)
                existing_compounds_map[inchi_key] = compound_uid
            
            # Prepare RT/MZ reference if data available and compound_uid exists
            if compound_uid and pd.notna(row.get('rt_peak')) and pd.notna(row.get('mz')):
                reference_record = _prepare_reference_record(
                    row, compound_uid, chromatography, polarity, input_file_path, prov
                )
                if reference_record:
                    reference_records.append(reference_record)

        # Batch insert compounds
        references_created = 0
        if compound_records:
            logger.info(f"Batch inserting {len(compound_records)} compounds...")
            conn.executemany("""
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, compound_records)

        # Check for existing references and batch insert if necessary
        if reference_records:
            logger.info(f"Checking for existing references and batch inserting if necessary...")
            # Get existing references to check for duplicates
            existing_refs = set()
            existing_ref_result = conn.execute("""
                SELECT compound_uid, chromatography, polarity, adduct 
                FROM mz_rt_references
            """).fetchall()
            existing_refs = {(row[0], row[1], row[2], row[3]) for row in existing_ref_result}

            # Filter out duplicates
            filtered_reference_records = []
            references_skipped = 0
            
            for record in reference_records:
                # Extract compound_uid, chromatography, polarity, adduct from record
                compound_uid = record[1]
                chromatography = record[8]
                polarity = record[9]
                adduct = record[7]
                
                ref_key = (compound_uid, chromatography, polarity, adduct)
                if ref_key not in existing_refs:
                    filtered_reference_records.append(record)
                    existing_refs.add(ref_key)  # Add to set to prevent duplicates within this batch
                    references_created += 1
                else:
                    references_skipped += 1

            # Batch insert filtered references
            if filtered_reference_records:
                conn.executemany("""
                    INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, filtered_reference_records)
        
        # Get final counts
        compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        references_count = conn.execute("SELECT COUNT(*) FROM mz_rt_references").fetchone()[0]
    
    logger.info("Compounds added successfully!")
    logger.info(f"   Total compounds in database: {compounds_count}")
    logger.info(f"   New compounds created: {compounds_created}")
    if compounds_skipped > 0:
        logger.info(f"   Compounds skipped (already existed): {compounds_skipped}")
    logger.info(f"   Total RT/MZ references in database: {references_count}")
    logger.info(f"   New RT/MZ references created: {references_created}")
    if 'references_skipped' in locals() and references_skipped > 0:
        logger.info(f"   RT/MZ references skipped (duplicates): {references_skipped}")
    
    return

def _prepare_compound_record(row: pd.Series, compound_uid: str, pubchem_cache: Dict, prov: Dict) -> tuple:
    """Prepare a compound record tuple for batch insertion."""
    # Extract compound data from row
    name = str(row.get('label', row.get('compound_name', 'Unknown')))
    inchi_key = str(row.get('inchi_key', ''))
    pubchem_cache_data = pubchem_cache.get(inchi_key, {})

    inchi = str(pubchem_cache_data.get('inchi', '')) if pd.notna(pubchem_cache_data.get('inchi')) else None
    smiles = str(pubchem_cache_data.get('smiles', '')) if pd.notna(pubchem_cache_data.get('smiles')) else None
    formula = str(pubchem_cache_data.get('formula', '')) if pd.notna(pubchem_cache_data.get('formula')) else None
    mono_isotopic_molecular_weight = pubchem_cache_data.get('mono_isotopic_molecular_weight') if pd.notna(pubchem_cache_data.get('mono_isotopic_molecular_weight')) else None
    iupac_name = str(pubchem_cache_data.get('iupac_name', '')) if pd.notna(pubchem_cache_data.get('iupac_name')) else None
    pubchem_cid = str(pubchem_cache_data.get('pubchem_cid', '')) if pd.notna(pubchem_cache_data.get('pubchem_cid')) else None
    cas_number = str(pubchem_cache_data.get('cas_number', '')) if pd.notna(pubchem_cache_data.get('cas_number')) else None
    synonyms_val = pubchem_cache_data.get('synonyms', '')
    if synonyms_val is None or synonyms_val == '':
        synonyms = None
    elif isinstance(synonyms_val, (list, tuple, np.ndarray)):
        synonyms = '; '.join(map(str, synonyms_val))
    else:
        synonyms = str(synonyms_val)
    compound_classes = str(row.get('compound_classes', '')) if pd.notna(row.get('compound_classes')) else None
    compound_pathways = str(row.get('compound_pathways', '')) if pd.notna(row.get('compound_pathways')) else None
    compound_tags = str(row.get('compound_tags', '')) if pd.notna(row.get('compound_tags')) else None

    return (
        compound_uid,
        name,
        inchi_key,
        inchi,
        smiles,
        formula,
        compound_classes,
        compound_pathways,
        compound_tags,
        mono_isotopic_molecular_weight,
        iupac_name,
        pubchem_cid,
        cas_number,
        synonyms,
        prov["analyst"],
        prov["timestamp"]
    )

def _prepare_reference_record(row: pd.Series, compound_uid: str, chromatography: str, 
                            polarity: str, input_file_path: str, prov: Dict) -> tuple:
    """Prepare a reference record tuple for batch insertion."""
    try:
        uid = _generate_uid("mz_rt_reference")
        
        # Extract RT/MZ data
        rt_peak = float(row.get('rt_peak'))
        rt_min = float(row.get('rt_min', rt_peak - 0.5))
        rt_max = float(row.get('rt_max', rt_peak + 0.5))
        mz = float(row.get('mz'))
        mz_tolerance = float(row.get('mz_tolerance', 5.0))
        adduct = str(row.get('adduct', ''))
        
        confidence = str(row.get('confidence', 'Unknown')) if pd.notna(row.get('confidence')) else 'Unknown'
        source = input_file_path if input_file_path else 'Unknown'
        
        return (
            uid,
            compound_uid,
            rt_peak,
            rt_min,
            rt_max,
            mz,
            mz_tolerance,
            adduct,
            chromatography,
            polarity,
            confidence,
            source,
            prov["analyst"],
            prov["timestamp"]
        )
    except (ValueError, TypeError) as e:
        # Skip records with invalid data
        return None

def _prepare_compound_record_from_dict(compound_data: Dict) -> Optional[Tuple]:
    """Prepare compound record from dictionary data."""
    try:
        prov = ldt.get_provenance()
        
        # Use provided compound_uid (should be set by calling function)
        compound_uid = compound_data.get('compound_uid')
        if not compound_uid:
            logger.error("Compound data missing compound_uid - this should be set by calling function")
            return None
        
        # Ensure we have a valid inchi_key
        inchi_key = compound_data.get('inchi_key', '')
        if not inchi_key:
            logger.warning("Compound data missing inchi_key - skipping")
            return None
        
        return (
            compound_uid,
            compound_data.get('name', 'Unknown'),
            inchi_key,
            compound_data.get('inchi'),
            compound_data.get('smiles'),
            compound_data.get('formula'),
            compound_data.get('compound_classes'),
            compound_data.get('compound_pathways'),
            compound_data.get('compound_tags'),
            compound_data.get('mono_isotopic_molecular_weight'),
            compound_data.get('iupac_name'),
            compound_data.get('pubchem_cid'),
            compound_data.get('cas_number'),
            compound_data.get('synonyms'),
            prov["analyst"],
            prov["timestamp"]
        )
    except Exception as e:
        logger.error(f"Error preparing compound record: {e}")
        return None

def _prepare_reference_record_from_dict(reference_data: Dict) -> Optional[Tuple]:
    """Prepare reference record from dictionary data."""
    try:
        prov = ldt.get_provenance()
        
        return (
            _generate_uid("mz_rt_reference"),
            reference_data.get('compound_uid'),
            float(reference_data.get('rt_peak')),
            float(reference_data.get('rt_min')),
            float(reference_data.get('rt_max')),
            float(reference_data.get('mz')),
            float(reference_data.get('mz_tolerance', 5.0)),
            reference_data.get('adduct', ''),
            reference_data.get('chromatography'),
            reference_data.get('polarity'),
            reference_data.get('confidence', 'Unknown'),
            reference_data.get('source', 'Unknown'),
            prov["analyst"],
            prov["timestamp"]
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Error preparing reference record: {e}")
        return None

def get_atlas_metadata_from_db(db_path: str, atlas_uid: str, validation: bool = False) -> pd.DataFrame:
    """
    Retrieve atlas metadata from database for a specific atlas.
    Returns DataFrame with atlas information including name, description, chromatography, polarity, etc.
    """
    
    try:
        logger.debug(f"Retrieving atlas metadata for UID: {atlas_uid} from database: {db_path}")
        with get_db_connection(db_path) as conn: 
            query = """
            SELECT 
                atlas_uid,
                atlas_name,
                atlas_description,
                chromatography,
                polarity,
                atlas_type,
                created_by,
                created_date
            FROM atlases 
            WHERE atlas_uid = ?
            """
            
            df = conn.execute(query, [atlas_uid]).df()
            logger.debug(f"Query executed successfully, retrieved {len(df)} records")
        
        if df.empty:
            raise ValueError(f"Atlas not found in main database")
        if len(df) > 1:
            raise ValueError(f"Multiple entries found for atlas in the database.")
        if validation is False:
            logger.info(f"Retrieved atlas metadata for: {df['atlas_name'].iloc[0]}")
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.warning(f"Did not find atlas {atlas_uid} in database {db_path}: {e}")
        return {}

def _generate_uid(entity_type: str, decorator: str = None) -> str:
    """Generate a unique identifier for database entities."""
    if entity_type == "ref_atlas":
        return f"atl-ref-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-ref-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_atlas":
        return f"atl-rta-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "analyzed_atlas":
        return f"atl-tga-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-tga-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_experimental":
        return f"mzrt-exp-{uuid.uuid4().hex[:32]}"
    elif entity_type == "compound":
        return f"cmp-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_reference":
        return f"mzrt-ref-{uuid.uuid4().hex[:32]}"
    elif entity_type == "association":
        return f"assoc-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_alignment":
        return f"rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "analysis":
        return f"tga-{uuid.uuid4().hex[:32]}"
    elif entity_type == "ms1_data":
        return f"ms1-{uuid.uuid4().hex[:32]}"
    elif entity_type == "ms2_data":
        return f"ms2-{uuid.uuid4().hex[:32]}"
    elif entity_type == "ms1_summary":
        return f"ms1-sum-{uuid.uuid4().hex[:32]}"
    elif entity_type == "ms2_summary":
        return f"ms2-sum-{uuid.uuid4().hex[:32]}"
    elif entity_type == "ms2_hits":
        return f"ms2-hits-{uuid.uuid4().hex[:32]}"
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")


def get_files_by_type_from_db(project_db_path: str, 
                              file_types: List[str], 
                              file_format: str = "parquet",
                              chromatography: str = None,
                              polarity: str = None
) -> pd.DataFrame:
    """
    Get files of a specific type from the project database.
    
    Args:
        project_db_path: Path to project database
        file_types: List of file types to retrieve ('qc', 'experimental', 'istd', 'injbl', 'exctrl')
        file_format: Format of files to retrieve ('parquet', 'raw', 'mzML')
        chromatography: Optional chromatography filter (e.g., 'HILIC', 'C18')
        polarity: Optional polarity filter (e.g., 'positive', 'pos', 'negative', 'neg')
    
    Returns:
        DataFrame with columns: file_path, chromatography, polarity, file_type
    """

    database_files = {}

    # If specific category requested, return only that category
    for type in file_types:
        if type not in ['qc', 'experimental', 'istd', 'exctrl']:
            raise ValueError(f"Invalid file category: {type}. Must be one of ['qc', 'experimental', 'istd', 'exctrl']")

        with get_db_connection(project_db_path) as conn:
            placeholders = ','.join(['?' for _ in file_types])
            params = file_types + [file_format]
            
            where_conditions = [f"file_type IN ({placeholders})", "file_format = ?"]
            
            if chromatography:
                if chromatography == "HILICZ":
                    chromatography = "HILIC"
                where_conditions.append("UPPER(chromatography) = UPPER(?)")
                params.append(chromatography)
            
            if polarity:
                if polarity.lower() in ["pos", "positive"]:
                    polarity = "positive"
                elif polarity.lower() in ["neg", "negative"]:
                    polarity = "negative"
                where_conditions.append("UPPER(polarity) = UPPER(?)")
                params.append(polarity)
            
            where_clause = " AND ".join(where_conditions)

            files_df = conn.execute(f"""
                SELECT file_path, chromatography, ms_level, polarity, file_type 
                FROM lcmsruns 
                WHERE {where_clause}
                ORDER BY chromatography, ms_level, polarity, file_path
            """, params).df()
        
        if files_df.empty:
            raise ValueError(f"No {file_types} files found in the project database with specified filters")

        database_files[type] = files_df

    return database_files

def batch_save_compounds_and_references(
    compounds_data: List[Dict],
    references_data: List[Dict],
    db_path: str
) -> Tuple[int, int]:
    """
    Schema-compliant batch save for compounds and references from raw input data.
    
    Schema Logic:
    1. For each compound_data: check if inchi_key exists in database
       - If exists: get existing compound_uid, skip compound creation
       - If not exists: create new compound with new compound_uid
    2. For each reference_data: find corresponding compound_uid, then check if identical reference exists
       - If identical: skip reference creation
       - If different: create new reference entry
    
    Args:
        compounds_data: List of compound dictionaries
        references_data: List of reference dictionaries
        db_path: str

    Returns:
        Tuple of (compounds_created, references_created)
    """
    compounds_created = 0
    references_created = 0
    references_skipped_identical = 0
    compounds_skipped_existing = 0
    
    with get_db_connection(db_path) as conn:
        # Step 1: Get ALL existing compounds (inchi_key -> compound_uid mapping)
        existing_compounds = {}  # inchi_key -> compound_uid
        existing_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
        for row in existing_result:
            existing_compounds[row[0]] = row[1]
        
        logger.info(f"Found {len(existing_compounds)} existing compounds in database")
        
        # Step 2: Process compound data and build inchi_key -> compound_uid mapping
        compound_records = []
        inchi_to_compound_uid = {}  # This will contain both existing and new compound mappings
        
        for compound_data in compounds_data:
            inchi_key = compound_data.get('inchi_key')
            if not inchi_key:
                logger.warning("Skipping compound with missing inchi_key")
                continue
                
            if inchi_key in existing_compounds:
                # Compound already exists - use existing UID
                inchi_to_compound_uid[inchi_key] = existing_compounds[inchi_key]
                compounds_skipped_existing += 1
                logger.debug(f"Compound already exists: {inchi_key} -> {existing_compounds[inchi_key]}")
            else:
                # New compound - generate UID and prepare for creation
                new_compound_uid = _generate_uid("compound")
                compound_record = _prepare_compound_record_from_dict({
                    **compound_data,  # Raw data from input
                    'compound_uid': new_compound_uid  # Add the generated UID
                })
                
                if compound_record:
                    compound_records.append(compound_record)
                    inchi_to_compound_uid[inchi_key] = new_compound_uid
                    existing_compounds[inchi_key] = new_compound_uid  # Prevent duplicates within batch
                    compounds_created += 1
                    logger.debug(f"New compound prepared: {inchi_key} -> {new_compound_uid}")
        
        # Step 3: Batch insert new compounds
        if compound_records:
            logger.info(f"Creating {len(compound_records)} new compounds...")
            conn.executemany("""
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, compound_records)
        
        # Step 4: Process reference data - match to compounds and check for duplicates
        if references_data:
            logger.info(f"Processing {len(references_data)} reference entries...")
            reference_records = []
            
            for reference_data in references_data:
                # Find the compound_uid for this reference using inchi_key
                ref_inchi_key = reference_data.get('inchi_key')
                if not ref_inchi_key:
                    logger.warning("Skipping reference with missing inchi_key")
                    continue
                
                compound_uid = inchi_to_compound_uid.get(ref_inchi_key)
                if not compound_uid:
                    logger.warning(f"No compound found for reference with inchi_key: {ref_inchi_key}")
                    continue
                
                # Prepare reference data with compound_uid
                reference_with_uid = {
                    **reference_data,
                    'compound_uid': compound_uid
                }
                
                # Check if IDENTICAL reference already exists
                identical_ref_exists = _check_identical_reference_exists(conn, reference_with_uid)
                
                if identical_ref_exists:
                    references_skipped_identical += 1
                    logger.debug(f"Identical reference already exists for compound {compound_uid}")
                    continue
                
                # Create new reference entry
                reference_record = _prepare_reference_record_from_dict(reference_with_uid)
                if reference_record:
                    reference_records.append(reference_record)
                    references_created += 1
                    logger.debug(f"New reference prepared for compound {compound_uid}")
            
            # Step 5: Batch insert new references
            if reference_records:
                logger.info(f"Creating {len(reference_records)} new references...")
                conn.executemany("""
                    INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, reference_records)
    
    # Log summary
    logger.info("Batch save completed:")
    logger.info(f"  Compounds created: {compounds_created}")
    logger.info(f"  Compounds skipped (already exist): {compounds_skipped_existing}")
    logger.info(f"  References created: {references_created}")
    logger.info(f"  References skipped (identical data): {references_skipped_identical}")
    
    return compounds_created, references_created


def _check_identical_reference_exists(conn, reference_data: Dict) -> bool:
    """
    Check if an IDENTICAL reference already exists in the database.
    This implements the schema rule: only create new references if data is different.
    
    Args:
        conn: Database connection
        reference_data: Dictionary with reference data
        
    Returns:
        True if identical reference exists, False otherwise
    """
    try:
        compound_uid = reference_data.get('compound_uid')
        rt_peak = float(reference_data.get('rt_peak', 0.0))
        rt_min = float(reference_data.get('rt_min', 0.0))
        rt_max = float(reference_data.get('rt_max', 0.0))
        mz = float(reference_data.get('mz', 0.0))
        mz_tolerance = float(reference_data.get('mz_tolerance', 5.0))
        adduct = str(reference_data.get('adduct', ''))
        chromatography = str(reference_data.get('chromatography', ''))
        polarity = str(reference_data.get('polarity', ''))
        
        # Check for IDENTICAL reference (using small tolerance for floating point comparison)
        existing_ref = conn.execute("""
            SELECT mz_rt_reference_uid 
            FROM mz_rt_references 
            WHERE compound_uid = ? 
            AND chromatography = ? 
            AND polarity = ? 
            AND adduct = ?
            AND ABS(rt_peak - ?) < 0.0001
            AND ABS(rt_min - ?) < 0.0001  
            AND ABS(rt_max - ?) < 0.0001
            AND ABS(mz - ?) < 0.0001
            AND ABS(mz_tolerance - ?) < 0.0001
        """, [
            compound_uid, chromatography, polarity, adduct,
            rt_peak, rt_min, rt_max, mz, mz_tolerance
        ]).fetchone()
        
        return existing_ref is not None
        
    except (ValueError, TypeError) as e:
        logger.warning(f"Error checking reference data: {e}")
        return False

def _create_database_tables(conn, db_type: str = "main"):
    """
    Create database tables based on database type.
    
    Args:
        conn: Database connection
        db_type: Either "main" or "project"
    """
    if db_type == "main":
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compounds (
                compound_uid TEXT PRIMARY KEY,
                name TEXT,
                inchi_key TEXT,
                inchi TEXT,
                smiles TEXT,
                formula TEXT,
                compound_classes TEXT,
                compound_pathways TEXT,
                compound_tags TEXT,
                mono_isotopic_molecular_weight REAL,
                iupac_name TEXT,
                pubchem_cid TEXT,
                cas_number TEXT,
                synonyms TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mz_rt_references (
                mz_rt_reference_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                mz REAL,
                mz_tolerance REAL,
                adduct TEXT,
                chromatography TEXT,
                polarity TEXT,
                confidence TEXT,
                source TEXT,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlases (
                atlas_uid TEXT PRIMARY KEY,
                atlas_name TEXT,
                atlas_description TEXT,
                chromatography TEXT,
                polarity TEXT,
                atlas_type TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid TEXT PRIMARY KEY,
                atlas_uid TEXT,
                compound_uid TEXT,
                mz_rt_reference_uid TEXT,
                association_order INTEGER,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (atlas_uid) REFERENCES atlases (atlas_uid),
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid),
                FOREIGN KEY (mz_rt_reference_uid) REFERENCES mz_rt_references (mz_rt_reference_uid)
            )
        """)
        
    elif db_type == "project":
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lcmsruns (
                file_path TEXT PRIMARY KEY,
                filename TEXT,
                file_format TEXT,
                file_type TEXT,
                chromatography TEXT,
                ms_level INTEGER,
                polarity TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlases (
                atlas_uid TEXT PRIMARY KEY,
                atlas_name TEXT,
                atlas_description TEXT,
                chromatography TEXT,
                polarity TEXT,
                atlas_type TEXT,
                source_atlas_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compounds (
                compound_uid TEXT PRIMARY KEY,
                name TEXT,
                inchi_key TEXT,
                inchi TEXT,
                smiles TEXT,
                formula TEXT,
                compound_classes TEXT,
                compound_pathways TEXT,
                compound_tags TEXT,
                mono_isotopic_molecular_weight REAL,
                iupac_name TEXT,
                pubchem_cid TEXT,
                cas_number TEXT,
                synonyms TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mz_rt_experimental (
                mz_rt_experimental_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                ms1_notes TEXT,
                ms2_notes TEXT,
                mz REAL,
                mz_tolerance REAL,
                adduct TEXT,
                chromatography TEXT,
                polarity TEXT,
                rt_alignment_applied BOOLEAN,
                rt_shift REAL,
                source_mz_rt_reference_uid TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid TEXT PRIMARY KEY,
                atlas_uid TEXT,
                compound_uid TEXT,
                mz_rt_reference_uid TEXT,
                association_order INTEGER,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (atlas_uid) REFERENCES atlases (atlas_uid)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rt_alignment (
                rt_alignment_uid TEXT PRIMARY KEY,
                project_name TEXT,
                rt_alignment_number INTEGER,
                qc_atlas_uid TEXT,
                model_type TEXT,
                polynomial_degree INTEGER,
                r_squared REAL,
                rmse REAL,
                coefficients TEXT,
                equation TEXT,
                num_qc_files INTEGER,
                num_compounds INTEGER,
                created_by TEXT,
                created_date TEXT,
                metadata TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms1_data (
                ms1_data_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                file_path TEXT,
                label TEXT,
                rt REAL,
                mz REAL,
                i REAL,
                in_feature BOOLEAN,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_data (
                ms2_data_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                file_path TEXT,
                label TEXT,
                rt REAL,
                mz REAL,
                i REAL,
                precursor_MZ REAL,
                precursor_intensity REAL,
                collision_energy REAL,
                in_feature BOOLEAN,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_hits (
                ms2_hit_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                file_path TEXT,
                inchi_key TEXT,
                database TEXT,
                ref_id TEXT,
                ref_name TEXT,
                score REAL,
                num_matches INTEGER,
                mz_theoretical REAL,
                mz_measured REAL,
                rt_measured REAL,
                qry_intensity_peak REAL,
                ref_frags INTEGER,
                data_frags INTEGER,
                matched_fragments TEXT,
                qry_frag_colors TEXT,
                qry_spectrum TEXT,
                ref_spectrum TEXT,
                qry_spectrum_original TEXT,
                ref_spectrum_original TEXT,
                created_by TEXT,
                created_date TEXT,
                analyst_notes TEXT,
                curation_status TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms1_summary (
                ms1_summary_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                file_path TEXT,
                label TEXT,
                num_datapoints INTEGER,
                peak_area REAL,
                peak_height REAL,
                mz_centroid REAL,
                rt_peak REAL,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)
        
        # Add MS2 summary table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_summary (
                ms2_summary_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                file_path TEXT,
                num_scans INTEGER,
                num_fragments INTEGER,
                best_ms2_rt REAL,
                best_ms2_mz REAL,
                best_ms2_intensity REAL,
                num_hits INTEGER,
                best_hit_score REAL,
                best_hit_database TEXT,
                best_hit_ref_id TEXT,
                best_hit_ref_name TEXT,
                best_hit_num_matches INTEGER,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS compound_analysis_metadata (
                metadata_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                
                -- Computed metadata (not raw data)
                suggested_rt_min REAL,
                suggested_rt_max REAL,
                suggested_rt_peak REAL,
                rt_bounds_confidence REAL,
                
                best_eic_file TEXT,
                best_eic_rt REAL,
                best_eic_mz REAL,
                best_eic_intensity REAL,
                best_eic_ppm_error REAL,
                
                best_ms2_file TEXT,
                best_ms2_database TEXT,
                best_ms2_score REAL,
                best_ms2_num_matches INTEGER,
                best_ms2_matched_fragments TEXT,
                
                total_files_detected INTEGER,
                ms2_files_with_data INTEGER,
                
                -- Isomers stored as JSON list of dicts
                isomers TEXT,
                
                -- Curation fields
                ms1_notes TEXT DEFAULT 'keep',
                ms2_notes TEXT DEFAULT 'no selection',
                analyst_notes TEXT,
                identification_notes TEXT,
                curation_status TEXT DEFAULT 'pending',
                
                created_by TEXT,
                created_date TEXT,
                
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid)
            )
        """)

        # conn.execute("""
        #     CREATE TABLE IF NOT EXISTS targeted_analysis (
        #         analysis_uid TEXT,
        #         project_name TEXT,
        #         rt_alignment_number INTEGER,
        #         analysis_number INTEGER,
        #         atlas_uid TEXT,
        #         atlas_type TEXT,
        #         chromatography_polarity TEXT,
        #         compound_uid TEXT,
        #         inchi_key TEXT,
        #         compound_name TEXT,
        #         pre_rt_peak REAL,
        #         pre_rt_min REAL,
        #         pre_rt_max REAL,
        #         pre_mz REAL,
        #         mz_tolerance REAL,
        #         adduct TEXT,
        #         isomers TEXT,
        #         post_rt_peak REAL,
        #         post_rt_min REAL,
        #         post_rt_max REAL,
        #         is_rt_modified BOOLEAN,
        #         best_eic_file TEXT,
        #         best_eic_rt REAL,
        #         best_eic_mz REAL,
        #         best_eic_intensity REAL,
        #         best_eic_ppm_error REAL,
        #         best_eic_rt_error REAL,
        #         avg_eic_rt REAL,
        #         avg_eic_intensity REAL,
        #         avg_eic_mz REAL,
        #         best_ms2_file TEXT,
        #         best_ms2_database TEXT,
        #         best_ms2_ref_id TEXT,
        #         best_ms2_rt_peak REAL,
        #         best_ms2_intensity_peak REAL,
        #         best_ms2_mz_peak REAL,
        #         best_ms2_score REAL,
        #         best_ms2_num_matches INTEGER,
        #         best_ms2_ref_frags INTEGER,
        #         best_ms2_data_frags INTEGER,
        #         best_ms2_matched_fragments TEXT,
        #         avg_ms2_score REAL,
        #         total_files_detected INTEGER,
        #         ms2_files_with_data INTEGER,
        #         ms2_best_score REAL,
        #         ms2_best_database TEXT,
        #         ms2_total_matches INTEGER,
        #         ms1_notes TEXT,
        #         ms2_notes TEXT,
        #         analyst_notes TEXT,
        #         identification_notes TEXT,
        #         curation_status TEXT,
        #         analyst TEXT,
        #         analysis_timestamp TEXT,
        #         PRIMARY KEY (analysis_uid, compound_uid, rt_alignment_number, analysis_number)
        #     )
        # """)

def verify_compounds_exist_in_db(compound_uids: list, db_path: str) -> bool:
    """
    Verify that all compound_uids exist in the database at db_path.
    Returns a DataFrame of compounds that exist in the database.
    Logs warnings for any missing compound_uids.
    """
    if not compound_uids:
        logger.warning("No compound_uids provided for verification.")
        return False

    logger.info("Verifying that all compounds exist in the database...")
    with get_db_connection(db_path) as conn:
        placeholders = ','.join(['?'] * len(compound_uids))
        query = f"SELECT compound_uid FROM compounds WHERE compound_uid IN ({placeholders})"
        existing = conn.execute(query, compound_uids).fetchall()
        existing_uids = {row[0] for row in existing}
    missing_uids = set(compound_uids) - existing_uids
    if missing_uids:
        for uid in missing_uids:
            logger.warning(f"Compound {uid} not found in database {db_path}")
        return False
    return True

def get_compound_uids_by_inchi_keys(db_path: str, inchi_keys: list) -> dict:
    """Return {inchi_key: compound_uid} for all inchi_keys found in the database."""
    with get_db_connection(db_path) as conn:
        if not inchi_keys:
            return {}
        placeholders = ','.join(['?'] * len(inchi_keys))
        rows = conn.execute(
            f"SELECT inchi_key, compound_uid FROM compounds WHERE inchi_key IN ({placeholders})",
            inchi_keys
        ).fetchall()
        return {row[0]: row[1] for row in rows}

def get_or_create_mz_rt_reference_uid(
    db_path: str,
    compound_uid: str,
    chromatography: str,
    polarity: str,
    adduct: str,
    rt_peak: float,
    mz: float,
    mz_tolerance: float
) -> tuple[str, bool]:
    """
    Return (mz_rt_reference_uid, reused_flag). If not found, generate a new UID (do not insert).
    """
    with get_db_connection(db_path) as conn:
        existing = conn.execute("""
            SELECT mz_rt_reference_uid FROM mz_rt_references
            WHERE compound_uid = ?
            AND chromatography = ? AND polarity = ? AND adduct = ?
            AND ABS(rt_peak - ?) < 0.0001 AND ABS(mz - ?) < 0.0001
            AND ABS(mz_tolerance - ?) < 0.0001
        """, [
            compound_uid, chromatography, polarity, adduct,
            rt_peak, mz, mz_tolerance
        ]).fetchone()
        if existing:
            return existing[0], True
        else:
            return _generate_uid("mz_rt_reference"), False

def save_ms1_data_to_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str,
    ms1_df: pd.DataFrame
) -> None:
    """Save MS1 data for a compound/file to database."""
    prov = ldt.get_provenance()
    
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database. Skipping MS1 data.")
            return
        
        compound_uid = compound_result[0]
        
        records = []
        for _, row in ms1_df.iterrows():
            records.append((
                _generate_uid("ms1_data"),
                compound_uid,
                rt_alignment_number,
                analysis_number,
                file_path,
                row.get('label', ''),
                float(row.get('rt', 0.0)),
                float(row.get('mz', 0.0)),
                float(row.get('i', 0.0)),
                bool(row.get('in_feature', False)),
                prov["analyst"],
                prov["timestamp"]
            ))
        
        if records:
            conn.executemany("""
                INSERT INTO ms1_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    logger.info(f"Saved {len(records)} MS1 data points for compound {inchi_key}")

def save_ms2_data_to_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str,
    ms2_df: pd.DataFrame
) -> None:
    """Save MS2 data for a compound/file to database."""
    prov = ldt.get_provenance()
    
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database. Skipping MS2 data.")
            return
        
        compound_uid = compound_result[0]
        
        records = []
        for _, row in ms2_df.iterrows():
            records.append((
                _generate_uid("ms2_data"),
                compound_uid,
                rt_alignment_number,
                analysis_number,
                file_path,
                row.get('label', ''),
                float(row.get('rt', 0.0)),
                float(row.get('mz', 0.0)),
                float(row.get('i', 0.0)),
                float(row.get('precursor_MZ', 0.0)),
                float(row.get('precursor_intensity', 0.0)),
                float(row.get('collision_energy', 0.0)),
                bool(row.get('in_feature', False)),
                prov["analyst"],
                prov["timestamp"]
            ))
        
        if records:
            conn.executemany("""
                INSERT INTO ms2_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    logger.info(f"Saved {len(records)} MS2 data points for compound {inchi_key}")

def save_ms2_hits_to_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str,
    ms2_hits_df: pd.DataFrame
) -> None:
    """Save MS2 spectral matching hits to database."""
    prov = ldt.get_provenance()
    
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database. Skipping MS2 hits.")
            return
        
        compound_uid = compound_result[0]
        
        records = []
        for _, row in ms2_hits_df.iterrows():
            records.append((
                _generate_uid("ms2_hits"),
                compound_uid,
                rt_alignment_number,
                analysis_number,
                file_path,
                row.get('inchi_key', ''),
                row.get('database', ''),
                row.get('ref_id', ''),
                row.get('ref_name', ''),
                float(row.get('score', 0.0)),
                int(row.get('num_matches', 0)),
                float(row.get('mz_theoretical', 0.0)),
                float(row.get('mz_measured', 0.0)),
                float(row.get('rt_measured', 0.0)),
                float(row.get('qry_intensity_peak', 0.0)),
                int(row.get('ref_frags', 0)),
                int(row.get('data_frags', 0)),
                json.dumps(row.get('matched_fragments', [])),
                json.dumps(row.get('qry_frag_colors', [])),
                json.dumps(row.get('qry_spectrum', [])),
                json.dumps(row.get('ref_spectrum', [])),
                json.dumps(row.get('qry_spectrum_original', [])),
                json.dumps(row.get('ref_spectrum_original', [])),
                prov["analyst"],
                prov["timestamp"],
                '',  # analyst_notes
                'pending'  # curation_status
            ))
        
        if records:
            conn.executemany("""
                INSERT INTO ms2_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    logger.info(f"Saved {len(records)} MS2 hits for compound {inchi_key}")

def save_ms1_summary_to_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str,
    ms1_summary_df: pd.DataFrame
) -> None:
    """Save MS1 summary data for a compound/file to database."""
    prov = ldt.get_provenance()
    
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database. Skipping MS1 summary.")
            return
        
        compound_uid = compound_result[0]
        
        records = []
        for _, row in ms1_summary_df.iterrows():
            records.append((
                _generate_uid("ms1_summary"),
                compound_uid,
                rt_alignment_number,
                analysis_number,
                file_path,
                row.get('label', ''),
                int(row.get('num_datapoints', 0)),
                float(row.get('peak_area', 0.0)),
                float(row.get('peak_height', 0.0)),
                float(row.get('mz_centroid', 0.0)),
                float(row.get('rt_peak', 0.0)),
                prov["analyst"],
                prov["timestamp"]
            ))
        
        if records:
            conn.executemany("""
                INSERT INTO ms1_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    logger.info(f"Saved {len(records)} MS1 summary records for compound {inchi_key}")

def save_ms2_summary_to_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str,
    ms2_summary_df: pd.DataFrame
) -> None:
    """Save MS2 summary data for a compound/file to database."""
    prov = ldt.get_provenance()
    
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database. Skipping MS2 summary.")
            return
        
        compound_uid = compound_result[0]
        
        records = []
        for _, row in ms2_summary_df.iterrows():
            records.append((
                _generate_uid("ms2_summary"),
                compound_uid,
                rt_alignment_number,
                analysis_number,
                file_path,
                int(row.get('num_scans', 0)),
                int(row.get('num_fragments', 0)),
                float(row.get('best_ms2_rt', 0.0)),
                float(row.get('best_ms2_mz', 0.0)),
                float(row.get('best_ms2_intensity', 0.0)),
                int(row.get('num_hits', 0)),
                float(row.get('best_hit_score', 0.0)),
                row.get('best_hit_database', ''),
                row.get('best_hit_ref_id', ''),
                row.get('best_hit_ref_name', ''),
                int(row.get('best_hit_num_matches', 0)),
                prov["analyst"],
                prov["timestamp"]
            ))
        
        if records:
            conn.executemany("""
                INSERT INTO ms2_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
    
    logger.info(f"Saved {len(records)} MS2 summary records for compound {inchi_key}")

def get_ms1_data_from_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str = None
) -> pd.DataFrame:
    """Retrieve MS1 data for a compound from database using inchi_key."""
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database.")
            return pd.DataFrame()
        
        compound_uid = compound_result[0]
        
        if file_path:
            df = conn.execute("""
                SELECT label, rt, mz, i, in_feature
                FROM ms1_data
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ? AND file_path = ?
                ORDER BY rt
            """, [compound_uid, rt_alignment_number, analysis_number, file_path]).df()
        else:
            df = conn.execute("""
                SELECT label, rt, mz, i, in_feature, file_path
                FROM ms1_data
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ?
                ORDER BY file_path, rt
            """, [compound_uid, rt_alignment_number, analysis_number]).df()
    
    return df

def get_ms2_data_from_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str = None
) -> pd.DataFrame:
    """Retrieve MS2 data for a compound from database using inchi_key."""
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database.")
            return pd.DataFrame()
        
        compound_uid = compound_result[0]
        
        if file_path:
            df = conn.execute("""
                SELECT label, rt, mz, i, precursor_MZ, precursor_intensity, 
                       collision_energy, in_feature
                FROM ms2_data
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ? AND file_path = ?
                ORDER BY rt
            """, [compound_uid, rt_alignment_number, analysis_number, file_path]).df()
        else:
            df = conn.execute("""
                SELECT label, rt, mz, i, precursor_MZ, precursor_intensity, 
                       collision_energy, in_feature, file_path
                FROM ms2_data
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ?
                ORDER BY file_path, rt
            """, [compound_uid, rt_alignment_number, analysis_number]).df()
    
    return df

def get_ms2_hits_from_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str = None
) -> pd.DataFrame:
    """Retrieve MS2 hits for a compound from database using inchi_key."""
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database.")
            return pd.DataFrame()
        
        compound_uid = compound_result[0]
        
        if file_path:
            df = conn.execute("""
                SELECT * FROM ms2_hits
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ? AND file_path = ?
                ORDER BY score DESC
            """, [compound_uid, rt_alignment_number, analysis_number, file_path]).df()
        else:
            df = conn.execute("""
                SELECT * FROM ms2_hits
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ?
                ORDER BY file_path, score DESC
            """, [compound_uid, rt_alignment_number, analysis_number]).df()
    
    # Convert JSON text columns back to Python objects
    json_columns = ['matched_fragments', 'qry_frag_colors', 'qry_spectrum', 
                    'ref_spectrum', 'qry_spectrum_original', 'ref_spectrum_original']
    
    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else None)
    
    return df

def get_ms1_summary_from_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str = None
) -> pd.DataFrame:
    """Retrieve MS1 summary for a compound from database using inchi_key."""
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database.")
            return pd.DataFrame()
        
        compound_uid = compound_result[0]
        
        if file_path:
            df = conn.execute("""
                SELECT label, num_datapoints, peak_area, peak_height, 
                       mz_centroid, rt_peak
                FROM ms1_summary
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ? AND file_path = ?
            """, [compound_uid, rt_alignment_number, analysis_number, file_path]).df()
        else:
            df = conn.execute("""
                SELECT label, num_datapoints, peak_area, peak_height, 
                       mz_centroid, rt_peak, file_path
                FROM ms1_summary
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ?
                ORDER BY file_path
            """, [compound_uid, rt_alignment_number, analysis_number]).df()
    
    return df

def get_ms2_summary_from_db(
    project_db_path: str,
    inchi_key: str,
    rt_alignment_number: int,
    analysis_number: int,
    file_path: str = None
) -> pd.DataFrame:
    """Retrieve MS2 summary for a compound from database using inchi_key."""
    with get_db_connection(project_db_path) as conn:
        # Look up compound_uid from inchi_key
        compound_result = conn.execute("""
            SELECT compound_uid FROM compounds WHERE inchi_key = ?
        """, [inchi_key]).fetchone()
        
        if not compound_result:
            logger.warning(f"Compound with inchi_key {inchi_key} not found in database.")
            return pd.DataFrame()
        
        compound_uid = compound_result[0]
        
        if file_path:
            df = conn.execute("""
                SELECT num_scans, num_fragments, best_ms2_rt, best_ms2_mz, 
                       best_ms2_intensity, num_hits, best_hit_score, 
                       best_hit_database, best_hit_ref_id, best_hit_ref_name, 
                       best_hit_num_matches
                FROM ms2_summary
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ? AND file_path = ?
            """, [compound_uid, rt_alignment_number, analysis_number, file_path]).df()
        else:
            df = conn.execute("""
                SELECT num_scans, num_fragments, best_ms2_rt, best_ms2_mz, 
                       best_ms2_intensity, num_hits, best_hit_score, 
                       best_hit_database, best_hit_ref_id, best_hit_ref_name, 
                       best_hit_num_matches, file_path
                FROM ms2_summary
                WHERE compound_uid = ? AND rt_alignment_number = ? 
                AND analysis_number = ?
                ORDER BY file_path
            """, [compound_uid, rt_alignment_number, analysis_number]).df()
    
    return df

def save_atlas_to_database(atlas_obj, db_path: str, db_type: str = "main") -> None:
    """
    Save an Atlas object to the database (typing not included to avoid circular imports).
    Only creates new mz_rt_references entries for references that don't already exist.
    
    Args:
        atlas_obj: Atlas object to save
    """
    # Verify all compounds exist in database
    if not verify_compounds_exist_in_db([comp.compound_uid for comp in atlas_obj.compound_references.values()], db_path):
        raise ValueError(f"Some compounds in atlas {atlas_obj.atlas_uid} don't exist in database")

    prov = ldt.get_provenance()
    with get_db_connection(db_path) as conn:
        if db_type == "main":
            # Create atlas entry
            conn.execute("""
                INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atlas_obj.atlas_uid,
                atlas_obj.atlas_name,
                atlas_obj.atlas_description,
                atlas_obj.chromatography,
                atlas_obj.polarity,
                "REFERENCE",  # atlas_type
                prov["analyst"],
                prov["timestamp"]
            ))
            
            # Process each CompoundReference
            association_order = 0
            references_created = 0
            references_reused = 0
            for inchi_key, compound_ref in atlas_obj.compound_references.items():
                # Check if this reference already exists in database
                existing_check = conn.execute("""
                    SELECT mz_rt_reference_uid FROM mz_rt_references 
                    WHERE mz_rt_reference_uid = ?
                """, [compound_ref.mz_rt_reference_uid]).fetchone()
                
                mz_rt_reference_uid = compound_ref.mz_rt_reference_uid
                
                # Create new reference if it doesn't exist
                if not existing_check:
                    conn.execute("""
                        INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mz_rt_reference_uid,
                        compound_ref.compound_uid,
                        compound_ref.rt_peak,
                        compound_ref.rt_min,
                        compound_ref.rt_max,
                        compound_ref.mz,
                        compound_ref.mz_tolerance,
                        compound_ref.adduct,
                        compound_ref.chromatography,
                        compound_ref.polarity,
                        compound_ref.confidence,
                        'atlas_creation',  # source
                        prov["analyst"],
                        prov["timestamp"]
                    ))
                    references_created += 1
                else:
                    references_reused += 1
                
                # Create atlas-compound association
                assoc_uid = _generate_uid("association")
                conn.execute("""
                    INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    assoc_uid,
                    atlas_obj.atlas_uid,
                    compound_ref.compound_uid,
                    mz_rt_reference_uid,
                    association_order,
                    prov["analyst"],
                    prov["timestamp"]
                ))
                
                association_order += 1
        
            logger.info(f"Saved atlas {atlas_obj.atlas_name} to database with UID: {atlas_obj.atlas_uid}")
            logger.info(f"  References created: {references_created}")
            logger.info(f"  References reused: {references_reused}")
            logger.info(f"  Total associations: {association_order}")

        association_order += 1

def get_rt_aligned_atlases_from_db(
    project_db_path: str,
    rt_alignment_number: int,
) -> List[Dict]:
    """
    Get all RT-aligned atlases for given RT alignment number.
    
    """
    params = [rt_alignment_number]
    query = """
        SELECT 
            atlas_uid,
            atlas_name,
            atlas_description,
            chromatography,
            polarity,
            atlas_type,
            source_atlas_uid,
            rt_alignment_number,
            created_by,
            created_date
        FROM atlases
        WHERE rt_alignment_number = ?
    """
    
    query += " ORDER BY atlas_type, polarity"
    
    with get_db_connection(project_db_path) as conn:
        results = conn.execute(query, params).fetchall()
        
        atlases = []
        for row in results:
            # Infer workflow from chromatography
            workflow = row[3].upper()  # chromatography -> workflow
            
            atlases.append({
                'atlas_uid': row[0],
                'atlas_name': row[1],
                'atlas_description': row[2],
                'chromatography': row[3],
                'polarity': row[4],
                'atlas_type': row[5],
                'source_atlas_uid': row[6],
                'rt_alignment_number': row[7],
                'created_by': row[8],
                'created_date': row[9],
                'workflow': workflow
            })
        
        if not atlases:
            raise ValueError(f"No RT-aligned atlases found in project database for RT alignment number {rt_alignment_number}")
        else:
            logger.info(f"Found {len(atlases)} RT-aligned atlases in project database for RT alignment number {rt_alignment_number}")
        return atlases

def save_rt_aligned_atlas_to_db(
    project_db_path: str,
    main_db_path: str,
    target_atlas_uid: str,
    best_model: dict,
    corr_compounds_df: pd.DataFrame,
    rt_alignment_number: int,
    analysis_number: Optional[int] = None
) -> Tuple[str, str]:
    """
    Create RT-aligned atlas in project database and apply RT alignment to compounds.
    
    Args:
        project_db_path: Path to project database
        main_db_path: Path to main database
        target_atlas_uid: UID of the target atlas
        best_model: RT alignment model dictionary
        corr_compounds_df: DataFrame of RT aligned compounds
        rt_alignment_number: RT alignment number
        analysis_number: Optional analysis number for this RT alignment (if not provided, will be NULL in database)
    Returns:
        Tuple of (aligned_atlas_uid, aligned_atlas_name)
    """
    logger.info("Creating RT-aligned atlas in project database...")

    prov = ldt.get_provenance()

    # Use some info from the target atlas for the new RT-aligned target atlas
    target_atlas_info = get_atlas_metadata_from_db(main_db_path, target_atlas_uid)

    # Generate new atlas UID for RT-aligned version
    aligned_atlas_uid = _generate_uid("rt_atlas")
    aligned_atlas_name = f"{target_atlas_info['atlas_name']} (RT aligned)"
    aligned_atlas_description = f"RT-aligned version of {target_atlas_info['atlas_name']} using polynomial model (R²={best_model['r2']:.4f})"
    logger.info(f"From the {target_atlas_info['atlas_uid']} template, generated new atlas UID: {aligned_atlas_uid} for RT-aligned atlas: {aligned_atlas_name}")

    logger.info("Saving RT-aligned atlas and compounds to project database...")
    with get_db_connection(project_db_path) as conn:
        
        # Create new atlas entry
        conn.execute("""
            INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            aligned_atlas_uid,
            aligned_atlas_name,
            aligned_atlas_description,
            target_atlas_info['chromatography'],
            target_atlas_info['polarity'],
            target_atlas_info['atlas_type'],
            target_atlas_info['atlas_uid'],
            rt_alignment_number,
            analysis_number,
            prov["analyst"],
            prov["timestamp"]
        ))

        association_order = 0
        for _, row in corr_compounds_df.iterrows():
            compound_uid = row['compound_uid']
            aligned_rt_peak = row.get('rt_peak')
            aligned_rt_min = row.get('rt_min')
            aligned_rt_max = row.get('rt_max')
            rt_shift = row.get('rt_shift')
            exp_uid = _generate_uid("mz_rt_experimental")
            assoc_uid = _generate_uid("association")
            source_mz_rt_reference_uid = row.get('mz_rt_reference_uid')

            conn.execute("""
                INSERT INTO mz_rt_experimental VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                exp_uid,
                compound_uid,
                rt_alignment_number,
                analysis_number,
                aligned_rt_peak,
                aligned_rt_min,
                aligned_rt_max,
                '',  # ms1_notes
                '',  # ms2_notes
                row.get('mz'),
                row.get('mz_tolerance', 5.0),
                row.get('adduct', ''),
                target_atlas_info['chromatography'],
                target_atlas_info['polarity'],
                True,  # rt_alignment_applied
                rt_shift,
                source_mz_rt_reference_uid,
                prov["analyst"],
                prov["timestamp"]
            ))
            
            # Create atlas-compound association
            conn.execute("""
                INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                assoc_uid,
                aligned_atlas_uid,
                compound_uid,
                source_mz_rt_reference_uid,
                association_order,
                prov["analyst"],
                prov["timestamp"]
            ))

            association_order += 1

    logger.info(f"Created and deposited RT-aligned atlas: {aligned_atlas_uid} with name: {aligned_atlas_name}")

    return aligned_atlas_uid, aligned_atlas_name

def verify_project_db_holdings(project_db_path: str, rt_alignment_number: int = None, analysis_number: int = None) -> Dict[str, pd.DataFrame]:
    """
    Verify all expected data from putative identification workflow.
    Returns a dictionary of DataFrames for external display.
    
    Args:
        project_db_path: Path to project database
        rt_alignment_number: Optional RT alignment number to filter results
        analysis_number: Optional analysis number to filter results
    """
    
    results = {}
    
    with duckdb.connect(project_db_path) as conn:
        # 1. LCMS Runs (not filtered - shows all runs)
        results['lcms_runs'] = conn.execute("""
            SELECT *
            FROM lcmsruns
            ORDER BY file_type, chromatography, polarity
        """).df()
        
        # 2. RT Alignment Models
        if rt_alignment_number is not None:
            results['rt_models'] = conn.execute("""
                SELECT *
                FROM rt_alignment
                WHERE rt_alignment_number = ?
                ORDER BY rt_alignment_number DESC
            """, [rt_alignment_number]).df()
        else:
            results['rt_models'] = conn.execute("""
                SELECT *
                FROM rt_alignment
                ORDER BY rt_alignment_number DESC
            """).df()
        
        # 3. Atlases Created
        where_clauses = []
        params = []
        if rt_alignment_number is not None:
            where_clauses.append("rt_alignment_number = ?")
            params.append(rt_alignment_number)
        if analysis_number is not None:
            where_clauses.append("analysis_number = ?")
            params.append(analysis_number)
        
        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        results['atlases'] = conn.execute(f"""
            SELECT *
            FROM atlases
            {where_str}
            ORDER BY rt_alignment_number DESC, analysis_number DESC
        """, params).df()
        
        # 4. Atlas-Compound Associations (filtered by atlases from step 3)
        if not results['atlases'].empty:
            atlas_uids = results['atlases']['atlas_uid'].tolist()
            placeholders = ','.join(['?' for _ in atlas_uids])
            results['associations'] = conn.execute(f"""
                SELECT *
                FROM atlas_compound_associations
                WHERE atlas_uid IN ({placeholders})
                ORDER BY atlas_uid, compound_uid
            """, atlas_uids).df()
        else:
            results['associations'] = pd.DataFrame()
        
        # 5. Experimental Entries (sample)
        where_clauses = []
        params = []
        if rt_alignment_number is not None:
            where_clauses.append("rt_alignment_number = ?")
            params.append(rt_alignment_number)
        if analysis_number is not None:
            where_clauses.append("analysis_number = ?")
            params.append(analysis_number)
        
        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        results['exp_entries'] = conn.execute(f"""
            SELECT *
            FROM mz_rt_experimental
            {where_str}
            ORDER BY rt_alignment_number DESC, analysis_number DESC
            LIMIT 100
        """, params).df()
        
        # 6. Summary Statistics
        results['lcms_summary'] = conn.execute("""
            SELECT file_type, file_format, chromatography, polarity, ms_level, COUNT(*) as count
            FROM lcmsruns
            WHERE file_format = 'parquet'
            GROUP BY file_type, file_format, chromatography, polarity, ms_level
            ORDER BY file_type, file_format, chromatography, polarity, ms_level
        """).df()
        
        # Compounds per atlas (filtered by atlas_uids from step 3)
        if not results['atlases'].empty:
            atlas_uids = results['atlases']['atlas_uid'].tolist()
            placeholders = ','.join(['?' for _ in atlas_uids])
            results['compounds_per_atlas'] = conn.execute(f"""
                SELECT a.atlas_name, a.atlas_type, a.chromatography, a.polarity,
                       COUNT(DISTINCT aca.compound_uid) as compound_count
                FROM atlases a
                LEFT JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                WHERE a.atlas_uid IN ({placeholders})
                GROUP BY a.atlas_name, a.atlas_type, a.chromatography, a.polarity
                ORDER BY a.atlas_name
            """, atlas_uids).df()
        else:
            results['compounds_per_atlas'] = pd.DataFrame()
        
        # Experimental summary (filtered)
        where_clauses = []
        params = []
        if rt_alignment_number is not None:
            where_clauses.append("rt_alignment_number = ?")
            params.append(rt_alignment_number)
        if analysis_number is not None:
            where_clauses.append("analysis_number = ?")
            params.append(analysis_number)
        
        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        results['exp_summary'] = conn.execute(f"""
            SELECT chromatography, polarity, rt_alignment_applied,
                   COUNT(*) as entry_count,
                   ROUND(AVG(ABS(rt_shift)), 3) as avg_abs_rt_shift,
                   ROUND(MIN(rt_shift), 3) as min_rt_shift,
                   ROUND(MAX(rt_shift), 3) as max_rt_shift
            FROM mz_rt_experimental
            {where_str}
            GROUP BY chromatography, polarity, rt_alignment_applied
            ORDER BY chromatography, polarity
        """, params).df()
        
        # 7. Data Integrity Checks (filtered by atlas_uids from step 3)
        integrity_results = []
        
        if not results['atlases'].empty:
            atlas_uids = results['atlases']['atlas_uid'].tolist()
            placeholders = ','.join(['?' for _ in atlas_uids])
            
            orphaned = conn.execute(f"""
                SELECT COUNT(*) as orphaned_associations
                FROM atlas_compound_associations aca
                WHERE aca.atlas_uid IN ({placeholders})
                AND NOT EXISTS (SELECT 1 FROM atlases WHERE atlas_uid = aca.atlas_uid)
            """, atlas_uids).fetchone()[0]
            integrity_results.append({"Check": "Orphaned associations", "Count": orphaned})
        
        # RT-aligned entries (filtered)
        where_clauses = []
        params = []
        where_clauses.append("rt_alignment_applied = TRUE")
        if rt_alignment_number is not None:
            where_clauses.append("rt_alignment_number = ?")
            params.append(rt_alignment_number)
        if analysis_number is not None:
            where_clauses.append("analysis_number = ?")
            params.append(analysis_number)
        
        where_str = f"WHERE {' AND '.join(where_clauses)}"
        
        rt_aligned = conn.execute(f"""
            SELECT COUNT(*) FROM mz_rt_experimental {where_str}
        """, params).fetchone()[0]
        integrity_results.append({"Check": "RT-aligned entries", "Count": rt_aligned})
        
        results['integrity_checks'] = pd.DataFrame(integrity_results)
    
    return results