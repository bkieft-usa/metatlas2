import pandas as pd
import numpy as np
import duckdb
import uuid
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict
from tqdm.notebook import tqdm
from typing import Dict, List, Optional, Any, Tuple

from contextlib import contextmanager

sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.lcmsruns_tools as lrt
import metatlas2.pubchem_retrieval as pcr
import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf

# Initialize logger properly at module level
logger = lcf.get_logger('database_interact')

@contextmanager
def get_db_connection(db_path: str):
    conn = duckdb.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()

def get_files_by_type_from_db(project_db_path: str, file_type: str) -> pd.DataFrame:
    """
    Get files of a specific type from the project database.
    
    Args:
        project_db_path: Path to project database
        file_type: Type of files to retrieve ('qc', 'experimental', 'istd', 'injbl', 'exctrl')
    
    Returns:
        DataFrame with columns: file_path, chromatography, polarity, file_type
    """

    with get_db_connection(project_db_path) as conn:
        files_df = conn.execute("""
            SELECT file_path, chromatography, polarity, file_type 
            FROM lcmsruns 
            WHERE file_type = ?
            ORDER BY chromatography, polarity, file_path
        """, [file_type]).df()
    
    if files_df.empty:
        raise ValueError(f"No {file_type} files found in the project database")
    
    return files_df

def get_atlas_compounds_table(database_path: str, atlas_uid: str) -> pd.DataFrame:
    """
    Extract all compound information for a given atlas UID from the database and return as a pandas DataFrame.
    Handles both main database (with mz_rt_references) and project database (with mz_rt_experimental) structures.
    """

    with get_db_connection(database_path) as conn:
    
        # Check if this is a project database by looking for mz_rt_experimental table
        is_project_db = conn.execute("""
            SELECT COUNT(*) 
            FROM information_schema.tables 
            WHERE table_name = 'mz_rt_experimental'
        """).fetchone()[0] > 0
        
        if is_project_db:
            # Project database query - uses experimental data and limited compound info
            # Note: In project databases, compounds table might be minimal/empty since 
            # full compound data is in the main database
            query = """
                SELECT
                    a.atlas_uid,
                    a.atlas_name,
                    a.atlas_description,
                    a.chromatography,
                    a.polarity,
                    aca.compound_uid,
                    COALESCE(c.name, '') AS compound_name,
                    COALESCE(c.inchi_key, '') AS inchi_key,
                    COALESCE(c.inchi, '') AS inchi,
                    mzrt_exp.adduct,
                    mzrt_exp.mz,
                    mzrt_exp.rt_peak,
                    mzrt_exp.rt_min,
                    mzrt_exp.rt_max,
                    mzrt_exp.mz_tolerance,
                    mzrt_exp.mz_rt_experimental_uid AS mz_rt_reference_uid,
                    mzrt_exp.rt_correction_applied,
                    mzrt_exp.rt_shift,
                    mzrt_exp.source_mz_rt_reference_uid
                FROM atlases a
                JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                LEFT JOIN compounds c ON aca.compound_uid = c.compound_uid
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
                    FALSE AS rt_correction_applied,
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

    if df.empty:
        logger.warning(f"No compounds found for atlas {atlas_uid}")
        return pd.DataFrame()
    else:
        db_type = "project" if is_project_db else "main"
        chromatography = ldt.detect_atlas_input_chromatography(df)
        polarity = ldt.detect_atlas_input_polarity(df)
        logger.info(f"Retrieved {len(df)} compounds from {db_type} database for atlas: {df['atlas_name'].iloc[0]} ({df['atlas_uid'].iloc[0]})")
        logger.debug(f"Atlas chromatography: {chromatography}, polarity: {polarity}")
        if is_project_db and df['rt_correction_applied'].any():
            logger.info("RT-corrected atlas detected with experimental data")

    return df

def create_project_database(project_db_path: str, config: Dict) -> None:
    """Create project-specific database with required tables.
    If overwrite_existing is True, delete the database file and remake it.
    """
    project_db_path = Path(project_db_path)
    project_db_path.parent.mkdir(parents=True, exist_ok=True)
    overwrite_existing = config["database_options"]["overwrite_existing_project_db"] if "overwrite_existing_project_db" in config["database_options"] else False

    if overwrite_existing and project_db_path.exists():
        project_db_path.unlink()
        logger.info(f"Overwrite is set to True; Deleted existing database at {project_db_path} and proceeding...")
    elif not overwrite_existing and project_db_path.exists():
        logger.warning(f"Overwrite is set to False but database already exists at {project_db_path}. Use overwrite_existing=True to replace it.")
        return

    with get_db_connection(project_db_path) as conn:
        _create_database_tables(conn, db_type="project")

    logger.info(f"Project database created at {project_db_path}")

def create_metatlas_database(config: Dict) -> None:
    """
    Create main metatlas database with required tables.
    """
    db_path = Path(config["paths"]["main_database"])
    overwrite_existing = config["database_options"]["overwrite_existing_main_db"]

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
    project_path: str,
    config: Dict
) -> Dict:
    """
    Save LCMS run files to project database and return file paths grouped by chromatography/polarity/analysis type.
    Chromatography and polarity are inferred from filenames.
    If the lcmsruns table exists and has rows, do not overwrite unless overwrite_existing=True.
    """
    overwrite_existing = config["database_options"]["overwrite_existing_project_db"] if "overwrite_existing_project_db" in config["database_options"] else False

    files_by_group = lrt.get_project_files(project_path)

    with get_db_connection(project_db_path) as conn:

        # Check if lcmsruns table exists
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'lcmsruns'"
        ).fetchone()[0] > 0

        # Check if lcmsruns table has rows
        has_rows = False
        if table_exists:
            has_rows = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0] > 0

        if table_exists and has_rows and not overwrite_existing:
            logger.info("LCMS runs already exist in the database. Skipping overwrite.")
            conn.close()
            print_files_summary(files_by_group)
            return files_by_group

        if table_exists and has_rows and overwrite_existing:
            logger.info(f"LCMS runs already exist in the database for {project_name} but overwrite is set to True. Creating new table...")
            conn.execute("DELETE FROM lcmsruns")

        prov = ldt.get_provenance()
        total_files = 0
        for chrom, pol_dict in files_by_group.items():
            for pol, analysis_dict in pol_dict.items():
                for file_type, file_list in analysis_dict.items():
                    for file_path in file_list:
                        filename = os.path.basename(file_path)
                        conn.execute(
                            "INSERT INTO lcmsruns VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                file_path,
                                filename,
                                file_type,
                                chrom,
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
    for chrom, pol_dict in file_list.items():
        logger.info(f"Chromatography: {chrom}")
        for pol, filetype_dict in pol_dict.items():
            logger.info(f"  Polarity: {pol}")
            for file_type, files_list in filetype_dict.items():
                logger.info(f"    {file_type}: {len(files_list)} files")

def list_available_atlases(db_path: str) -> pd.DataFrame:
    """List all available atlases with optional filtering."""

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
            a.last_modified,
            COUNT(aca.compound_uid) as compound_count
        FROM atlases a
        LEFT JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
        WHERE {' AND '.join(conditions)}
        GROUP BY a.atlas_uid, a.atlas_name, a.atlas_description, 
                a.chromatography, a.polarity, a.last_modified
        ORDER BY a.atlas_name
        """
        
        df = conn.execute(query, params).df()
    
    return df

def get_most_recent_QC_atlas_id(config: Dict, database_path: str = "main"):
    with get_db_connection(database_path) as conn:

        query = """
        SELECT *
        FROM atlases
        WHERE atlas_name LIKE '%QC%' OR atlas_description LIKE '%QC%'
        ORDER BY last_modified DESC
        LIMIT 1
        """
        df = conn.execute(query).df()

    if df.empty:
        logger.warning("No QC atlas found.")
        return None

    return df.iloc[0]['atlas_uid']

def validate_database(config: Dict, database_path: str = "main") -> None:
    """
    Validate the DuckDB database and print summary statistics.
    If database_path is "main", validates main DB structure.
    If database_path is a custom project DB, validates project DB structure.
    """
    logger.info("Database Validation:")

    if database_path == "main":
        db_path = config["paths"]["main_database"]
    else:
        db_path = database_path

    if not os.path.exists(Path(db_path)):
        logger.error(f"Database not found at: {db_path}")
        return

    with get_db_connection(db_path) as conn:
        if database_path == "main":
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
                    logger.debug(f"            {row['last_modified']}")
        else:
            # Project DB: atlases, targeted_analysis, rt_alignment, mz_rt_experimental
            atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
            # Count unique analysis_uid values in targeted_analysis table
            targeted_count = len(conn.execute("SELECT DISTINCT analysis_uid FROM targeted_analysis").fetchall())
            rt_alignment_count = conn.execute("SELECT COUNT(*) FROM rt_alignment").fetchone()[0]
            mz_rt_exp_count = conn.execute("SELECT COUNT(*) FROM mz_rt_experimental").fetchone()[0]

            logger.info(f"   Atlases: {atlases_count}")
            logger.info(f"   Targeted analyses: {targeted_count}")
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
                    logger.debug(f"            {row['last_modified']}")

            # List targeted analyses
            targeted_df = conn.execute("""
                SELECT analysis_uid, project_name, atlas_uid, COUNT(*) as compound_count
                FROM targeted_analysis
                GROUP BY analysis_uid, project_name, atlas_uid
                ORDER BY analysis_uid
            """).df()
            if not targeted_df.empty:
                logger.info("   Targeted analysis entries:")
                for _, row in targeted_df.iterrows():
                    logger.info(f"      {row['analysis_uid']} ({row['project_name']}) - Atlas: {row['atlas_uid']} - {row['compound_count']} compounds")

            # List RT alignment models
            rt_df = conn.execute("""
                SELECT rt_alignment_uid, atlas_uid, model_type, polynomial_degree, r_squared, rmse, last_modified
                FROM rt_alignment
                ORDER BY last_modified DESC
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

def save_rt_alignment_model_to_db(corrected_atlas_uid: str, project_db_path: Path, best_model: dict, qc_files: list, modeling_data: list) -> str:
    """Save RT alignment model to project database."""
    rt_alignment_uid = _generate_uid("rt_alignment")
    
    # Extract project name from path
    project_db_path = Path(project_db_path)
    project_name = project_db_path.stem.replace('.duckdb', '')
    prov = ldt.get_provenance()

    model_metadata = {
        "qc_files": [os.path.basename(f) for f in qc_files],
        "compounds_used": [d.get('compound_uid', '') for d in modeling_data],
        "correction_timestamp": prov["timestamp"],
        "correction_method": "polynomial_qc_based",
        "analyst": prov["analyst"]
    }
    
    with get_db_connection(project_db_path) as conn:
        conn.execute("""
            INSERT INTO rt_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rt_alignment_uid,
            project_name,
            corrected_atlas_uid,
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
    return rt_alignment_uid

def add_compounds_to_db(input_df: pd.DataFrame, config: Dict, input_file_path: str = ""):
    """Add compounds and RT/MZ references to database using batch operations."""

    pubchem_cache_path = config["paths"]["pubchem_cache"]
    db_path = config["paths"]["main_database"]
    create_duplicates = config["database_options"]["add_compound_duplicates"] if "add_compound_duplicates" in config["database_options"] else False

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
        if not create_duplicates:
            existing_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
            existing_inchi_keys = {row[0] for row in existing_result}
            existing_compounds_map = {row[0]: row[1] for row in existing_result}
            if existing_inchi_keys:
                logger.debug(f"Found {len(existing_inchi_keys)} existing compounds in database. Not creating duplicates.")

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
            if not create_duplicates and inchi_key in existing_inchi_keys:
                compounds_skipped += 1
                compound_uid = existing_compounds_map.get(inchi_key)
            else:
                # Prepare new compound record
                compound_uid = _generate_uid("compound")
                compound_record = _prepare_compound_record(row, compound_uid, pubchem_cache, prov)
                compound_records.append(compound_record)
                compounds_created += 1
                
                # Add to existing sets for future duplicate checking in this session
                if not create_duplicates:
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
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, compound_records)

        # Check for duplicate references and batch insert
        if reference_records:
            logger.info(f"Checking for duplicate references and batch inserting...")
            # Get existing references to check for duplicates
            existing_refs = set()
            if not create_duplicates:
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
    inchi = str(row.get('inchi', '')) if pd.notna(row.get('inchi')) else None
    smiles = str(row.get('smiles', '')) if pd.notna(row.get('smiles')) else None
    formula = str(row.get('formula', '')) if pd.notna(row.get('formula')) else None
    
    # Handle optional fields
    compound_classes = str(row.get('compound_classes', '')) if pd.notna(row.get('compound_classes')) else None
    compound_pathways = str(row.get('compound_pathways', '')) if pd.notna(row.get('compound_pathways')) else None
    compound_tags = str(row.get('compound_tags', '')) if pd.notna(row.get('compound_tags')) else None
    molecular_weight = row.get('molecular_weight') if pd.notna(row.get('molecular_weight')) else None
    mono_isotopic_molecular_weight = row.get('mono_isotopic_molecular_weight') if pd.notna(row.get('mono_isotopic_molecular_weight')) else None
    iupac_name = str(row.get('iupac_name', '')) if pd.notna(row.get('iupac_name')) else None
    pubchem_cid = str(row.get('pubchem_cid', '')) if pd.notna(row.get('pubchem_cid')) else None
    cas_number = str(row.get('cas_number', '')) if pd.notna(row.get('cas_number')) else None
    synonyms = str(row.get('synonyms', '')) if pd.notna(row.get('synonyms')) else None
    
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
        molecular_weight,
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

def create_atlas_from_compounds(
    atlas_input_table: pd.DataFrame,
    atlas_name: str,
    atlas_description: str,
    config: Dict
):
    """
    Create an atlas from compounds, validating they exist in the database first.
    Uses adducts from the input table to find the correct mz_rt_references.
    """

    db_path = config["paths"]["main_database"]

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Please check path.")

    prov = ldt.get_provenance()
    chromatography = ldt.detect_atlas_input_chromatography(atlas_input_table)
    polarity = ldt.detect_atlas_input_polarity(atlas_input_table)

    # Validate all compounds exist in database
    logger.info(f"Validating compounds exist in database: {db_path}")

    with get_db_connection(db_path) as conn:

        # Get all compounds currently in database
        existing_compounds = conn.execute("SELECT inchi_key, compound_uid, name FROM compounds").fetchall()
        existing_inchi_keys = {row[0]: {'compound_uid': row[1], 'label': row[2]} for row in existing_compounds}
        
        logger.info(f"Database contains {len(existing_inchi_keys)} compounds, searching for missing compounds in input.")
        
        # Check which atlas compounds are missing from database
        missing_compounds = []
        available_compounds = []
        
        for inchi_key in atlas_input_table['inchi_key'].dropna().unique():
            if inchi_key in existing_inchi_keys:
                available_compounds.append({
                    'inchi_key': inchi_key,
                    'compound_uid': existing_inchi_keys[inchi_key]['compound_uid'],
                    'label': existing_inchi_keys[inchi_key]['label']
                })
            else:
                # Get compound name from input data if available
                compound_name = atlas_input_table[atlas_input_table['inchi_key'] == inchi_key]['label'].iloc[0] if 'label' in atlas_input_table.columns else 'Unknown'
                missing_compounds.append({
                    'inchi_key': inchi_key,
                    'label': compound_name
                })
        
        logger.info(f"Compounds available for atlas: {len(available_compounds)}")
        logger.info(f"Compounds missing from database: {len(missing_compounds)}")
        
        if missing_compounds:
            logger.warning("Missing compounds:")
            for compound in missing_compounds[:10]:
                logger.warning(f"  - {compound['label']} ({compound['inchi_key']})")
            if len(missing_compounds) > 10:
                logger.warning(f"  ... and {len(missing_compounds) - 10} more")
            
            conn.close()
            raise ValueError(f"Cannot create atlas: {len(missing_compounds)} compounds not found in database. "
                            f"Please add these compounds first.")
        
        logger.info(f"All {len(available_compounds)} compounds found in database!")

        # Create the atlas
        atlas_uid = _generate_uid("atlas")
        
        conn.execute("""
            INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            atlas_uid,
            atlas_name,
            atlas_description,
            chromatography,
            polarity,
            prov["analyst"],
            prov["timestamp"]
        ))

        logger.info(f"Created new atlas: {atlas_name} (UID: {atlas_uid}). Adding compounds...")

        # Create associations for each row in the input table (each compound-adduct combination)
        atlas_associations_created = 0
        
        for idx, row in atlas_input_table.iterrows():
            inchi_key = row.get('inchi_key')
            if pd.isna(inchi_key) or inchi_key not in existing_inchi_keys:
                continue
                
            compound_uid = existing_inchi_keys[inchi_key]['compound_uid']
            adduct = str(row.get('adduct', ''))
            
            # Find the mz_rt_reference for this specific compound, chromatography, polarity, and adduct
            mz_rt_ref_result = conn.execute("""
                SELECT mz_rt_reference_uid 
                FROM mz_rt_references 
                WHERE compound_uid = ? 
                AND chromatography = ? 
                AND polarity = ? 
                AND adduct = ?
                LIMIT 1
            """, [compound_uid, chromatography, polarity, adduct]).fetchone()
            
            mz_rt_reference_uid = mz_rt_ref_result[0] if mz_rt_ref_result else None
            
            # Create association
            association_uid = _generate_uid("association")
            
            conn.execute("""
                INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                association_uid,
                atlas_uid,
                compound_uid,
                mz_rt_reference_uid,
                idx + 1,
                prov["analyst"],
                prov["timestamp"]
            ))
            atlas_associations_created += 1

    logger.info(f"Atlas creation complete!")
    logger.info(f"   Atlas Name: {atlas_name}")
    logger.info(f"   Atlas UID: {atlas_uid}")
    logger.info(f"   Atlas associations created: {atlas_associations_created}")
    logger.info(f"   Method: {chromatography}/{polarity}")

    return atlas_uid, atlas_name

def _create_database_tables(conn, db_type: str = "main"):
    """Create all required database tables.
    
    Parameters:
        conn: DuckDB connection object
        db_type: Either "main" or "project" to determine which tables to create
    """
    # Core tables (common to both main and project databases)
    
    # Create compounds table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS compounds (
            compound_uid VARCHAR PRIMARY KEY,
            name VARCHAR NOT NULL,
            inchi_key VARCHAR(27) NOT NULL UNIQUE,
            inchi TEXT,
            smiles TEXT,
            formula VARCHAR,
            compound_classes TEXT,
            compound_pathways TEXT,
            compound_tags TEXT,
            molecular_weight DOUBLE,
            mono_isotopic_molecular_weight DOUBLE,
            iupac_name TEXT,
            pubchem_cid VARCHAR,
            cas_number VARCHAR,
            synonyms TEXT,
            created_by VARCHAR,
            last_modified TIMESTAMP
        )
    """)
    
    # Create atlases table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS atlases (
            atlas_uid VARCHAR PRIMARY KEY,
            atlas_name VARCHAR NOT NULL,
            atlas_description TEXT,
            chromatography VARCHAR,
            polarity VARCHAR,
            created_by VARCHAR,
            last_modified TIMESTAMP
        )
    """)
    
    # Create mz_rt_references table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mz_rt_references (
            mz_rt_reference_uid VARCHAR PRIMARY KEY,
            compound_uid VARCHAR NOT NULL,
            rt_peak DOUBLE NOT NULL,
            rt_min DOUBLE NOT NULL,
            rt_max DOUBLE NOT NULL,
            mz DOUBLE NOT NULL,
            mz_tolerance DOUBLE NOT NULL,
            adduct VARCHAR NOT NULL,
            chromatography VARCHAR NOT NULL,
            polarity VARCHAR NOT NULL,
            confidence VARCHAR,
            source VARCHAR,
            created_by VARCHAR,
            last_modified TIMESTAMP,
            FOREIGN KEY (compound_uid) REFERENCES compounds(compound_uid)
        )
    """)
    
    # Create atlas_compound_associations table with conditional foreign keys
    if db_type == "main":
        # Main database: enforce foreign key constraints
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid VARCHAR PRIMARY KEY,
                atlas_uid VARCHAR NOT NULL,
                compound_uid VARCHAR NOT NULL,
                mz_rt_reference_uid VARCHAR,
                association_order INTEGER,
                created_by VARCHAR,
                last_modified TIMESTAMP,
                FOREIGN KEY (atlas_uid) REFERENCES atlases(atlas_uid),
                FOREIGN KEY (compound_uid) REFERENCES compounds(compound_uid),
                FOREIGN KEY (mz_rt_reference_uid) REFERENCES mz_rt_references(mz_rt_reference_uid)
            )
        """)
    else:
        # Project database: no foreign key constraints (lightweight)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid VARCHAR PRIMARY KEY,
                atlas_uid VARCHAR NOT NULL,
                compound_uid VARCHAR NOT NULL,
                mz_rt_reference_uid VARCHAR,
                association_order INTEGER,
                created_by VARCHAR,
                last_modified TIMESTAMP
            )
        """)
    
    # Project-specific tables
    if db_type == "project":
        # Create lcmsruns table (project-specific)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lcmsruns (
                file_path VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                file_type VARCHAR NOT NULL,
                chromatography VARCHAR,
                polarity VARCHAR,
                created_by VARCHAR,
                last_modified TIMESTAMP
            )
        """)
        
        # Create mz_rt_experimental table (project-specific)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mz_rt_experimental (
                mz_rt_experimental_uid VARCHAR PRIMARY KEY,
                compound_uid VARCHAR NOT NULL,
                rt_peak DOUBLE NOT NULL,
                rt_min DOUBLE NOT NULL,
                rt_max DOUBLE NOT NULL,
                ms1_notes VARCHAR,
                ms2_notes VARCHAR,
                mz DOUBLE NOT NULL,
                mz_tolerance DOUBLE NOT NULL,
                adduct VARCHAR NOT NULL,
                chromatography VARCHAR NOT NULL,
                polarity VARCHAR NOT NULL,
                rt_correction_applied BOOLEAN DEFAULT FALSE,
                rt_shift DOUBLE,
                source_mz_rt_reference_uid VARCHAR,
                created_by VARCHAR,
                last_modified TIMESTAMP
            )
        """)
        
        # Create rt_alignment table without atlas foreign key constraint
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rt_alignment (
                rt_alignment_uid VARCHAR PRIMARY KEY,
                project_name VARCHAR NOT NULL,
                atlas_uid VARCHAR,
                model_type VARCHAR NOT NULL,
                polynomial_degree INTEGER,
                r_squared DOUBLE,
                rmse DOUBLE,
                coefficients TEXT,
                equation TEXT,
                qc_files_count INTEGER,
                compounds_used_count INTEGER,
                created_by VARCHAR,
                last_modified TIMESTAMP,
                model_metadata TEXT
            )
        """)
    
        # Create targeted_analysis table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS targeted_analysis (
                analysis_uid VARCHAR NOT NULL,
                project_name VARCHAR NOT NULL,
                atlas_uid VARCHAR NOT NULL,
                compound_uid VARCHAR NOT NULL,
                inchi_key VARCHAR(27) NOT NULL,
                compound_name VARCHAR NOT NULL,

                -- Pre targeted analysis atlas data
                pre_rt_peak DOUBLE,
                pre_rt_min DOUBLE,
                pre_rt_max DOUBLE,
                pre_mz DOUBLE,
                mz_tolerance DOUBLE,
                adduct VARCHAR,
                isomers TEXT,

                -- Post targeted analysis atlas data
                post_rt_peak DOUBLE,
                post_rt_min DOUBLE,
                post_rt_max DOUBLE,
                is_rt_modified BOOLEAN DEFAULT FALSE,
                
                -- Best EIC match across all files
                best_eic_file VARCHAR,
                best_eic_rt DOUBLE,
                best_eic_mz DOUBLE,
                best_eic_intensity DOUBLE,
                best_eic_ppm_error DOUBLE,
                best_eic_rt_error DOUBLE,

                -- Average EIC
                avg_eic_rt DOUBLE,
                avg_eic_intensity DOUBLE,
                avg_eic_mz DOUBLE,

                -- Best MS2 match across all files
                best_ms2_file VARCHAR,
                best_ms2_database VARCHAR,
                best_ms2_ref_id VARCHAR,
                best_ms2_rt_peak DOUBLE,
                best_ms2_intensity_peak DOUBLE,
                best_ms2_mz_peak DOUBLE,
                best_ms2_score DOUBLE,
                best_ms2_num_matches INTEGER,
                best_ms2_ref_frags INTEGER,
                best_ms2_data_frags INTEGER,
                best_ms2_frags_matching TEXT,

                -- Average MS2
                avg_ms2_score DOUBLE,

                -- File detection summary
                total_files_detected INTEGER DEFAULT 0,
                
                -- MS2 analysis results
                ms2_files_with_data INTEGER DEFAULT 0,
                ms2_best_score DOUBLE,
                ms2_best_database VARCHAR,
                ms2_total_matches INTEGER DEFAULT 0,
                
                -- User annotations
                ms1_notes VARCHAR DEFAULT 'keep',
                ms2_notes VARCHAR DEFAULT 'no selection',
                
                -- Analysis metadata
                analyst VARCHAR,
                analysis_timestamp TIMESTAMP,
            
            PRIMARY KEY (analysis_uid, compound_uid)
            )
        """)

    # Create indexes (common to both database types)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_compounds_inchi_key ON compounds(inchi_key)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mzrt_ref_compound ON mz_rt_references(compound_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_atlas_associations_atlas ON atlas_compound_associations(atlas_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_atlas_associations_compound ON atlas_compound_associations(compound_uid)")

    # Project-specific indexes
    if db_type == "project":
        conn.execute("CREATE INDEX IF NOT EXISTS idx_lcmsruns_analysis ON lcmsruns(file_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mzrt_exp_compound ON mz_rt_experimental(compound_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_alignment_project ON rt_alignment(project_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_alignment_atlas ON rt_alignment(atlas_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_project ON targeted_analysis(project_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_atlas ON targeted_analysis(atlas_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_compound ON targeted_analysis(compound_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_inchi ON targeted_analysis(inchi_key)")

def delete_atlas_from_db(config: Dict, atlas_uid: str) -> str:
    """
    Delete an atlas and all its associated metadata from the database.
    This includes cascading deletions from related tables.
    
    Parameters:
        db_path (str): Path to the DuckDB database file
        atlas_uid (str): UID of the atlas to delete
        
    Returns:
        str: Status message indicating success or failure
    """

    db_path = config["paths"]["main_database"]
    
    with get_db_connection(db_path) as conn: 
        try:
            # First verify the atlas exists and get its name for confirmation
            atlas_result = conn.execute("""
                SELECT atlas_name, atlas_description 
                FROM atlases 
                WHERE atlas_uid = ?
            """, [atlas_uid]).fetchone()
            
            if not atlas_result:
                logger.warning(f"Atlas with UID {atlas_uid} not found in database")
                return "No deletion!"
            
            atlas_name = atlas_result[0]
            atlas_description = atlas_result[1]
            
            logger.info(f"Found atlas: {atlas_name}")
            logger.info(f"Description: {atlas_description}")
            
            # Track deletions for summary
            deletion_summary = {}
            
            # Check and delete from atlas_compound_associations
            associations_count = conn.execute("""
                SELECT COUNT(*) 
                FROM atlas_compound_associations 
                WHERE atlas_uid = ?
            """, [atlas_uid]).fetchone()[0]
            
            if associations_count > 0:
                conn.execute("""
                    DELETE FROM atlas_compound_associations 
                    WHERE atlas_uid = ?
                """, [atlas_uid])
                deletion_summary['atlas_compound_associations'] = associations_count
                logger.info(f"Removed {associations_count} compound associations")
            
            # Check and delete from rt_alignment table (if it exists and has atlas_uid column)
            try:
                # First check if table exists
                table_exists = conn.execute("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_name = 'rt_alignment'
                """).fetchone()[0] > 0
                
                if table_exists:
                    # Check if atlas_uid column exists in rt_alignment table
                    column_exists = conn.execute("""
                        SELECT COUNT(*) 
                        FROM information_schema.columns 
                        WHERE table_name = 'rt_alignment' AND column_name = 'atlas_uid'
                    """).fetchone()[0] > 0
                    
                    if column_exists:
                        rt_alignment_count = conn.execute("""
                            SELECT COUNT(*) 
                            FROM rt_alignment 
                            WHERE atlas_uid = ?
                        """, [atlas_uid]).fetchone()[0]
                        
                        if rt_alignment_count > 0:
                            conn.execute("""
                                DELETE FROM rt_alignment 
                                WHERE atlas_uid = ?
                            """, [atlas_uid])
                            deletion_summary['rt_alignment'] = rt_alignment_count
                            logger.info(f"Removed {rt_alignment_count} RT alignment records")
            except Exception as e:
                # Table or column might not exist, which is fine
                pass
            
            # Check for any other tables that might reference atlas_uid
            # This is a generic approach to find tables with atlas_uid columns
            try:
                tables_with_atlas_uid = conn.execute("""
                    SELECT DISTINCT table_name 
                    FROM information_schema.columns 
                    WHERE column_name = 'atlas_uid' 
                    AND table_name NOT IN ('atlases', 'atlas_compound_associations', 'rt_alignment')
                """).fetchall()
                
                for (table_name,) in tables_with_atlas_uid:
                    try:
                        count = conn.execute(f"""
                            SELECT COUNT(*) 
                            FROM {table_name} 
                            WHERE atlas_uid = ?
                        """, [atlas_uid]).fetchone()[0]
                        
                        if count > 0:
                            conn.execute(f"""
                                DELETE FROM {table_name} 
                                WHERE atlas_uid = ?
                            """, [atlas_uid])
                            deletion_summary[table_name] = count
                            logger.info(f"Removed {count} records from {table_name}")
                    except Exception as e:
                        logger.warning(f"Could not delete from table {table_name}: {e}")
            except Exception as e:
                # Information schema queries might fail in some contexts
                pass
            
            # Finally, delete the atlas record itself
            conn.execute("""
                DELETE FROM atlases 
                WHERE atlas_uid = ?
            """, [atlas_uid])
            
            logger.info(f"Successfully deleted atlas '{atlas_name}' (UID: {atlas_uid})")
            
            if deletion_summary:
                logger.info("Cascade deletions performed:")
                for table, count in deletion_summary.items():
                    logger.info(f"  - {table}: {count} records")
            
            return "Deletion successful!"
            
        except Exception as e:
            logger.error(f"Error deleting atlas: {e}")
            return "Deletion failed!"
    
    return

def get_atlas_from_db(db_path: str, atlas_uid: str) -> pd.DataFrame:
    """
    Retrieve atlas metadata from database for a specific atlas.
    Returns DataFrame with atlas information including name, description, chromatography, polarity, etc.
    """
    
    with get_db_connection(db_path) as conn: 
        query = """
        SELECT 
            atlas_uid,
            atlas_name,
            atlas_description,
            chromatography,
            polarity,
            created_by,
            last_modified
        FROM atlases 
        WHERE atlas_uid = ?
        """
        
        df = conn.execute(query, [atlas_uid]).df()
    
    if df.empty:
        logger.warning(f"No atlas found with UID: {atlas_uid}")
        return pd.DataFrame()
    else:
        logger.info(f"Retrieved atlas metadata for: {df['atlas_name'].iloc[0]}")
        return df

def get_rt_correction_table_entry(db_path: Path, atlas_uid: str) -> Optional[dict]:
    """
    Retrieve the RT correction table entry for a given atlas UID from the rt_alignment table.
    Returns a dictionary with the entry if found, else None.
    """

    with get_db_connection(db_path) as conn:

        query = """
            SELECT *
            FROM rt_alignment
            WHERE atlas_uid = ?
            ORDER BY last_modified DESC
            LIMIT 1
        """
        result = conn.execute(query, [atlas_uid]).fetchone()
        columns = [desc[0] for desc in conn.description] if result else []

    if result and columns:
        rt_dict = dict(zip(columns, result))
        rt_df = pd.DataFrame([rt_dict])
        return rt_df
    else:
        logger.warning(f"No RT correction entry found for atlas UID: {atlas_uid}")
        return None


def _generate_uid(entity_type: str) -> str:
    """Generate a unique identifier for database entities."""
    if entity_type == "atlas":
        return f"atl-raw-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_atlas":
        return f"atl-rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "analyzed_atlas":
        return f"atl-tga-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_experimental":
        return f"mzrt-exp-{uuid.uuid4().hex[:32]}"
    elif entity_type == "compound":
        return f"cmp-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_reference":
        return f"mzrt-{uuid.uuid4().hex[:32]}"
    elif entity_type == "association":
        return f"assoc-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_alignment":
        return f"rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "analysis":
        return f"tga-{uuid.uuid4().hex[:32]}"
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

def get_experimental_files_from_db(project_db_path: str, file_types: List[str] = None) -> List[str]:
    """Get experimental files from project database."""
    
    if file_types is None:
        file_types = ['experimental', 'istd', 'exctrl']  # Exclude QC files
    
    with get_db_connection(project_db_path) as conn:

        # Get experimental files (excluding QC)
        placeholders = ','.join(['?' for _ in file_types])
        result = conn.execute(f"""
            SELECT file_path 
            FROM lcmsruns 
            WHERE file_type IN ({placeholders})
            ORDER BY filename
        """, file_types).fetchall()
        
    file_paths = [row[0] for row in result]
    logger.info(f"Found {len(file_paths)} experimental files in database")
    
    return file_paths

def get_atlas_compounds_with_metadata(project_db_path: str, main_db_path: str, atlas_uid: str) -> pd.DataFrame:
    """
    Get atlas compounds from project database and enrich with metadata from main database.
    This is specifically for RT-corrected atlases in project databases.
    """
    # Get basic atlas structure from project database
    project_df = get_atlas_compounds_table(project_db_path, atlas_uid)
    
    if project_df.empty:
        return pd.DataFrame()
    
    # Get compound UIDs that need metadata
    compound_uids = project_df['compound_uid'].unique().tolist()
    
    # Fetch compound metadata from main database
    with get_db_connection(main_db_path) as conn_main:

        placeholders = ','.join(['?' for _ in compound_uids])
        metadata_query = f"""
            SELECT 
                compound_uid,
                name AS compound_name,
                inchi_key,
                inchi,
                formula,
                mono_isotopic_molecular_weight AS exact_mass
            FROM compounds 
            WHERE compound_uid IN ({placeholders})
        """
        
        metadata_df = conn_main.execute(metadata_query, compound_uids).df()
    
    # Merge project data with metadata
    enriched_df = project_df.merge(
        metadata_df, 
        on='compound_uid', 
        how='left',
        suffixes=('', '_meta')
    )
    
    # Update compound_name and other fields with metadata
    enriched_df['compound_name'] = enriched_df['compound_name_meta'].fillna(enriched_df['compound_name'])
    enriched_df['inchi_key'] = enriched_df['inchi_key_meta'].fillna(enriched_df['inchi_key'])
    enriched_df['inchi'] = enriched_df['inchi_meta'].fillna(enriched_df['inchi'])
    
    # Add label column for compatibility with feature_tools
    enriched_df['label'] = enriched_df['compound_name']
    
    # Clean up duplicate columns
    cols_to_drop = [col for col in enriched_df.columns if col.endswith('_meta')]
    enriched_df = enriched_df.drop(columns=cols_to_drop)
    
    logger.info(f"Enriched {len(enriched_df)} compounds with metadata from main database")
    
    return enriched_df


def validate_targeted_analysis_data(project_db_path: str, analysis_uid: str = None) -> Dict:
    """Validate targeted analysis results in database."""
    
    with get_db_connection(project_db_path) as conn: 
        
        # Get latest analysis if no specific UID provided
        if analysis_uid:
            condition = "WHERE analysis_uid = ?"
            params = [analysis_uid]
        
        # Validation queries
        validation_query = f"""
        SELECT 
            COUNT(*) as total_records,
            COUNT(CASE WHEN pre_rt_peak IS NOT NULL THEN 1 END) as records_with_rt,
            COUNT(CASE WHEN pre_mz IS NOT NULL THEN 1 END) as records_with_mz,
            COUNT(CASE WHEN best_eic_file IS NOT NULL THEN 1 END) as records_with_eic,
            COUNT(CASE WHEN avg_eic_rt IS NOT NULL THEN 1 END) as records_with_avg_eic_rt,
            COUNT(CASE WHEN ms2_files_with_data > 0 THEN 1 END) as records_with_ms2,
            COUNT(CASE WHEN is_rt_modified = true THEN 1 END) as modified_records,
        FROM targeted_analysis
        {condition}
        """
        
        result = conn.execute(validation_query, params).fetchone()
    
    if result:
        return True
    else:
        return False

def clone_and_modify_atlas(
    from_db: str,
    to_db: str,
    source_atlas_uid: str,
    config: Dict,
    compound_updates: Dict[str, Dict[str, Any]],
    new_atlas_name: Optional[str] = None,
    new_atlas_description: Optional[str] = None,
    use_experimental_table: bool = False,
    rt_correction_metadata: Optional[Dict] = None
) -> str:
    """
    Clone an atlas and modify its compound associations and RT/MZ/annotation data.
    Supports both RT correction and targeted analysis workflows.
    Clones from one database (from_db) to another (to_db).

    Args:
        from_db: Path to source database (main).
        to_db: Path to destination database (project).
        source_atlas_uid: UID of the atlas to clone.
        config: Metatlas config dictionary.
        compound_updates: Dict keyed by compound_uid or inchi_key, with update fields.
        new_atlas_name: Optional new atlas name.
        new_atlas_description: Optional new atlas description.
        use_experimental_table: If True, create new mz_rt_experimental entries.
        rt_correction_metadata: Optional metadata for RT correction.

    Returns:
        str: UID of the new cloned atlas.
    """
    # Read source atlas and associations from from_db
    with get_db_connection(from_db) as conn_src:
        atlas_row = conn_src.execute("""
            SELECT atlas_name, atlas_description, chromatography, polarity, created_by
            FROM atlases WHERE atlas_uid = ?
        """, [source_atlas_uid]).fetchone()
        if not atlas_row:
            conn_src.close()
            raise ValueError(f"Atlas UID {source_atlas_uid} not found in database {from_db}.")

        associations = conn_src.execute("""
            SELECT compound_uid, mz_rt_reference_uid, association_order
            FROM atlas_compound_associations WHERE atlas_uid = ?
        """, [source_atlas_uid]).fetchall()

    # Write new atlas and associations to to_db
    prov = ldt.get_provenance()
    timestamp = prov["timestamp"]
    analyst = prov["analyst"]
    new_atlas_uid = _generate_uid("analyzed_atlas")
    atlas_name = new_atlas_name if new_atlas_name else atlas_row[0]
    atlas_description = new_atlas_description if new_atlas_description else atlas_row[1]

    with get_db_connection(to_db) as conn_dst:

        conn_dst.execute("""
            INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            new_atlas_uid,
            atlas_name,
            atlas_description,
            atlas_row[2],
            atlas_row[3],
            analyst,
            timestamp
        ))

        for compound_uid, mz_rt_reference_uid, association_order in associations:
            update_fields = compound_updates.get(compound_uid) or compound_updates.get(mz_rt_reference_uid)
            if update_fields and use_experimental_table:
                new_mz_rt_exp_uid = _generate_uid("mz_rt_experimental")
                conn_dst.execute("""
                    INSERT INTO mz_rt_experimental (
                        mz_rt_experimental_uid, compound_uid, rt_peak, rt_min, rt_max,
                        mz, mz_tolerance, adduct, chromatography, polarity, ms1_notes, ms2_notes,
                        rt_correction_applied, rt_shift, source_mz_rt_reference_uid,
                        created_by, last_modified
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_mz_rt_exp_uid,
                    compound_uid,
                    update_fields.get('rt_peak'),
                    update_fields.get('rt_min'),
                    update_fields.get('rt_max'),
                    update_fields.get('mz'),
                    update_fields.get('mz_tolerance'),
                    update_fields.get('adduct'),
                    update_fields.get('chromatography'),
                    update_fields.get('polarity'),
                    update_fields.get('ms1_notes'),
                    update_fields.get('ms2_notes'),
                    update_fields.get('rt_correction_applied', False),
                    update_fields.get('rt_shift'),
                    mz_rt_reference_uid,
                    analyst,
                    timestamp
                ))
                assoc_mz_uid = new_mz_rt_exp_uid
            else:
                assoc_mz_uid = mz_rt_reference_uid

            new_assoc_uid = _generate_uid("association")
            conn_dst.execute("""
                INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                new_assoc_uid,
                new_atlas_uid,
                compound_uid,
                assoc_mz_uid,
                association_order,
                analyst,
                timestamp
            ))

            if update_fields and not use_experimental_table:
                fields = []
                values = []
                for key in ['rt_min', 'rt_max', 'rt_peak', 'ms2_notes', 'ms1_notes']:
                    if key in update_fields and update_fields[key] is not None:
                        fields.append(f"{key} = ?")
                        values.append(update_fields[key])
                if fields:
                    conn_dst.execute(
                        f"UPDATE mz_rt_experimental SET {', '.join(fields)} WHERE compound_uid = ?",
                        values + [compound_uid]
                    )

        if rt_correction_metadata:
            rt_alignment_uid = _generate_uid("rt_alignment")
            conn_dst.execute("""
                INSERT INTO rt_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rt_alignment_uid,
                Path(to_db).stem.replace('.duckdb', ''),
                new_atlas_uid,
                rt_correction_metadata.get('model_type', 'polynomial'),
                rt_correction_metadata.get('degree'),
                rt_correction_metadata.get('r2'),
                rt_correction_metadata.get('rmse'),
                json.dumps(rt_correction_metadata.get('coefficients', [])),
                rt_correction_metadata.get('equation'),
                rt_correction_metadata.get('qc_files_count', 0),
                rt_correction_metadata.get('compounds_used_count', 0),
                analyst,
                timestamp,
                json.dumps(rt_correction_metadata)
            ))

    logger.info(f"Cloned and modified atlas {source_atlas_uid} from {from_db} to new atlas {new_atlas_uid} in {to_db}")
    return new_atlas_uid

def deposit_targeted_analysis_from_plot_data(
    atlas_df,
    project_db_path,
    plot_data,
    atlas_uid,
    project_name
):
    """
    Deposit all relevant targeted analysis info from plot_data into the targeted_analysis table.
    Uses a single analysis_uid for all compounds in the project (not per compound).
    Updated to handle enhanced best_ms2 selection logic.
    """
    prov = ldt.get_provenance()
    analysis_uid = _generate_uid('analysis')

    rows = []
    for inchi_key, meta in plot_data.items():
        input_data = meta.get('original_atlas_data', {})
        new_data = meta.get('new_atlas_data', {})
        best_eic = meta.get('best_eic', {})
        avg_eic = meta.get('avg_eic', {})
        best_ms2 = meta.get('best_ms2', {})
        avg_ms2 = meta.get('avg_ms2', {})

        compound_rows = atlas_df[atlas_df['inchi_key'] == inchi_key]
        if compound_rows.empty:
            continue
        compound_uid = compound_rows.iloc[0]['compound_uid']
        compound_name = compound_rows.iloc[0]['compound_name']
        adduct = compound_rows.iloc[0]['adduct']

        isomers = input_data.get('isomers', [])
        isomers_serialized = json.dumps(isomers) if isomers else None

        pre_rt_peak = input_data.get('rt_peak', None)
        pre_rt_min = input_data.get('rt_min', None)
        pre_rt_max = input_data.get('rt_max', None)
        pre_mz = input_data.get('mz', None)
        mz_tolerance = input_data.get('mz_tolerance', None)

        post_rt_peak = new_data.get('rt_peak', None)
        post_rt_min = new_data.get('rt_min', None)
        post_rt_max = new_data.get('rt_max', None)
        is_rt_modified = meta.get('is_modified', False)
        ms1_notes = new_data.get('ms1_notes', 'keep')
        ms2_notes = new_data.get('ms2_notes', 'no selection')
        
        best_eic_file = best_eic.get('file_peak', None)
        best_eic_rt = best_eic.get('rt_peak', None)
        best_eic_mz = best_eic.get('mz_peak', None)
        best_eic_intensity = best_eic.get('intensity_peak', None)
        best_eic_ppm_error = best_eic.get('ppm_diff', None)
        best_eic_rt_error = best_eic.get('rt_diff', None)

        avg_eic_rt = avg_eic.get('rt_peak', None)
        avg_eic_mz = avg_eic.get('mz_peak', None)
        avg_eic_intensity = avg_eic.get('intensity_peak', None)

        best_ms2_file = best_ms2.get('file_peak', None)
        best_ms2_database = best_ms2.get('database', None)
        best_ms2_ref_id = best_ms2.get('ref_id', None)
        best_ms2_rt_peak = best_ms2.get('rt_peak', None)
        best_ms2_intensity_peak = best_ms2.get('intensity_peak', None)
        best_ms2_mz_peak = best_ms2.get('mz_peak', None)
        best_ms2_score = best_ms2.get('score', None)
        best_ms2_num_matches = best_ms2.get('num_matches', None)
        best_ms2_ref_frags = best_ms2.get('ref_frags', None)
        best_ms2_data_frags = best_ms2.get('data_frags', None)
        best_ms2_frags_matching = best_ms2.get('frags_matching', None)

        avg_ms2_score = avg_ms2.get('avg_score', None)

        # Enhanced tracking of MS2 selection method and metadata
        ms2_selection_method = best_ms2.get('selection_method', 'unknown')
        total_files_detected = len(meta.get('eic_data', {}))
        ms2_files_with_data = len(meta.get('ms2_data', {}).get('files', {}))
        ms2_best_score = best_ms2_score
        ms2_best_database = best_ms2_database
        ms2_total_matches = best_ms2_num_matches

        row = (
            analysis_uid,
            project_name,
            atlas_uid,
            compound_uid,
            inchi_key,
            compound_name,
            pre_rt_peak,
            pre_rt_min,
            pre_rt_max,
            pre_mz,
            mz_tolerance,
            adduct,
            isomers_serialized,
            post_rt_peak,
            post_rt_min,
            post_rt_max,
            is_rt_modified,
            best_eic_file,
            best_eic_rt,
            best_eic_mz,
            best_eic_intensity,
            best_eic_ppm_error,
            best_eic_rt_error,
            avg_eic_rt,
            avg_eic_intensity,
            avg_eic_mz,
            best_ms2_file,
            best_ms2_database,
            best_ms2_ref_id,
            best_ms2_rt_peak,
            best_ms2_intensity_peak,
            best_ms2_mz_peak,
            best_ms2_score,
            best_ms2_num_matches,
            best_ms2_ref_frags,
            best_ms2_data_frags,
            best_ms2_frags_matching,
            avg_ms2_score,
            total_files_detected,
            ms2_files_with_data,
            ms2_best_score,
            ms2_best_database,
            ms2_total_matches,
            ms1_notes,
            ms2_notes,
            prov["analyst"],
            prov["timestamp"]
        )
        rows.append(row)

    with get_db_connection(project_db_path) as conn: 
        insert_sql = '''
            INSERT INTO targeted_analysis (
                analysis_uid, project_name, atlas_uid, compound_uid, inchi_key, compound_name,
                pre_rt_peak, pre_rt_min, pre_rt_max, pre_mz, mz_tolerance, adduct, isomers,
                post_rt_peak, post_rt_min, post_rt_max, is_rt_modified,
                best_eic_file, best_eic_rt, best_eic_mz, best_eic_intensity, best_eic_ppm_error, best_eic_rt_error,
                avg_eic_rt, avg_eic_intensity, avg_eic_mz,
                best_ms2_file, best_ms2_database, best_ms2_ref_id, best_ms2_rt_peak, best_ms2_intensity_peak, best_ms2_mz_peak,
                best_ms2_score, best_ms2_num_matches, best_ms2_ref_frags, best_ms2_data_frags, best_ms2_frags_matching,
                avg_ms2_score,
                total_files_detected, ms2_files_with_data, ms2_best_score, ms2_best_database, ms2_total_matches,
                ms1_notes, ms2_notes, analyst, analysis_timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        conn.executemany(insert_sql, rows)

    logger.info(f"Inserted {len(rows)} rows into targeted_analysis table with analysis_uid {analysis_uid}.")
    logger.info(f"Enhanced MS2 selection: reference_hits={sum(1 for row in rows if rows[rows.index(row)][31] is not None)}, intensity_based={sum(1 for row in rows if rows[rows.index(row)][31] is None and rows[rows.index(row)][25] is not None)}")

    validated = validate_targeted_analysis_data(project_db_path, analysis_uid)
    if validated is False:
        logger.warning(f"Validation failed for targeted analysis entry {analysis_uid}")
        return None
    elif validated is True:
        return analysis_uid

def verify_single_atlas_for_targeted_analysis(targeted_df: pd.DataFrame):
    """Ensure that only one atlas is used for the targeted analysis."""
    if targeted_df['atlas_uid'].nunique() > 1:
        raise ValueError("Multiple atlases found in targeted analysis results.")
       
    else:
        return targeted_df['atlas_uid'].iloc[0]
    

def generate_targeted_analysis_summary(project_db_path, config, analysis_uid):
    """
    Generate a per-compound targeted analysis summary table for a given analysis_uid.
    Uses the targeted_analysis table and compounds metadata.
    Args:
        project_db_path: Path to project database.
        config: Configuration dictionary with database paths.
        analysis_uid: UID of the targeted analysis.
    Returns:
        pd.DataFrame with one row per compound, including all targeted analysis results and compound metadata.
    """
    main_db_path = config["paths"]["main_database"]

    # Get targeted analysis results for this analysis_uid
    conn_proj = duckdb.connect(project_db_path)
    targeted_df = conn_proj.execute(
        "SELECT * FROM targeted_analysis WHERE analysis_uid = ?", [analysis_uid]
    ).df()

    if targeted_df.empty:
        conn_proj.close()
        logger.warning(f"No targeted analysis results found for analysis_uid {analysis_uid}")
        return pd.DataFrame()

    # Infer atlas_uid from targeted analysis
    atlas_uid = verify_single_atlas_for_targeted_analysis(targeted_df)

    # Get compound metadata from main DB
    compound_uids = targeted_df['compound_uid'].unique().tolist()
    conn_main = duckdb.connect(main_db_path)
    placeholders = ','.join(['?' for _ in compound_uids])
    compounds_df = conn_main.execute(
        f"SELECT * FROM compounds WHERE compound_uid IN ({placeholders})", compound_uids
    ).df()
    conn_main.close()

    # Merge targeted analysis results with compound metadata
    merged = targeted_df.merge(compounds_df, on='compound_uid', how='left', suffixes=('', '_compound'))

    # Optionally, add atlas metadata columns
    atlas_meta = conn_proj.execute("SELECT * FROM atlases WHERE atlas_uid = ?", [atlas_uid]).df()
    conn_proj.close()
    for col in atlas_meta.columns:
        merged[col] = atlas_meta[col].iloc[0] if not atlas_meta.empty else None

    # Final output: one row per compound with all targeted analysis and compound info
    return merged

def generate_comprehensive_targeted_analysis_report(project_db_path: str, config: Dict, 
                                                   analysis_uid: str, atlas_df_ft: pd.DataFrame,
                                                   output_path: str = None, include_missing_compounds: bool = False) -> pd.DataFrame:
    """
    Generate a comprehensive targeted analysis report matching the specified Excel format.
    Extracts data from targeted_analysis table and enriches with compound metadata.
    
    Args:
        project_db_path: Path to project database
        config: Configuration dictionary with database paths
        analysis_uid: UID of the targeted analysis
        atlas_df_ft: Optional atlas DataFrame to include missing compounds
        output_path: Optional path to save Excel file
        include_missing_compounds: Whether to include compounds from atlas that weren't detected
    
    Returns:
        pd.DataFrame with comprehensive report data
    """
    main_db_path = config["paths"]["main_database"]
    
    # Get the base targeted analysis summary
    base_summary = generate_targeted_analysis_summary(project_db_path, config, analysis_uid)
    
    if base_summary.empty:
        logger.warning(f"No targeted analysis results found for analysis_uid {analysis_uid}")
        if atlas_df_ft is not None and include_missing_compounds:
            logger.info("Creating empty rows for all atlas compounds...")
            base_summary = _create_empty_summary_from_atlas(atlas_df_ft, analysis_uid, config)
        else:
            return pd.DataFrame()
    elif atlas_df_ft is not None and include_missing_compounds:
        # Add missing compounds from atlas as empty rows
        base_summary = _add_missing_compounds_to_summary(base_summary, atlas_df_ft, analysis_uid, config)
    
    # Connect to both databases
    conn_proj = duckdb.connect(project_db_path)
    conn_main = duckdb.connect(main_db_path)
    
    report_rows = []
    
    logger.info(f"Generating comprehensive report for {len(base_summary)} compounds...")
    
    for idx, row in tqdm(base_summary.iterrows(), total=len(base_summary), desc="Processing compounds"):
        try:
            # Basic compound information
            compound_uid = row['compound_uid']
            inchi_key = row['inchi_key']
            compound_name = row['compound_name']
            
            # Calculate stats
            msms_quality = _calculate_msms_quality_score_from_notes(row)
            mz_quality = _calculate_mz_quality_score(row)
            rt_quality = _calculate_rt_quality_score(row)
            total_score = msms_quality + mz_quality + rt_quality
            msi_level = _determine_msi_level(msms_quality, mz_quality, rt_quality)
            isomer_info = _find_overlapping_compounds(conn_main, row, base_summary)
            
            # Build the report row
            report_row = {
                'index': idx,
                'identified_metabolite': compound_name,
                'label': compound_name,
                'isomer_compound': isomer_info['compound_names'],
                'isomer_inchi_keys': isomer_info['inchi_keys'],
                'formula': row.get('formula', ''),
                'polarity': row.get('polarity', ''),
                'exact_mass': row.get('mono_isotopic_molecular_weight', ''),
                'inchi_key': inchi_key,
                'adduct': row.get('adduct', ''),
                'msms_quality': msms_quality,
                'mz_quality': mz_quality,
                'rt_quality': rt_quality,
                'total_score': total_score,
                'msi_level': msi_level,
                'ms1_notes': row.get('ms1_notes', 'keep'),
                'ms2_notes': row.get('ms2_notes', 'no selection'),
                'analyst_notes': row.get('analyst_notes', ''),
                'identification_notes': row.get('identification_notes', ''),
                'max_intensity': row.get('best_eic_intensity', ''),
                'max_intensity_file': row.get('best_eic_file', ''),
                'max_intensity_rt': row.get('best_eic_rt', ''),
                'best_msms_file': row.get('best_ms2_file', ''),
                'best_msms_rt': row.get('best_ms2_rt_peak', ''),
                'best_msms_num_matching_ions': row.get('best_ms2_num_matches', ''),
                'best_msms_matching_ions': row.get('best_ms2_frags_matching', ''),
                'best_msms_score': row.get('best_ms2_score', ''),
                'mz_theoretical': row.get('pre_mz', ''),
                'mz_measured': row.get('best_eic_mz', ''),
                'mz_error': row['best_eic_ppm_error'],
                'rt_peak_theoretical': row.get('pre_rt_peak', ''),
                'rt_peak_measured': row.get('post_rt_peak', ''),
                'rt_min_measured': row.get('post_rt_min', ''),
                'rt_max_measured': row.get('post_rt_max', ''),
                'rt_error': row.get('best_eic_rt_error', '')
            }
            
            report_rows.append(report_row)
            
        except Exception as e:
            logger.error(f"Error processing compound {compound_name} ({inchi_key}): {e}")
            continue
    
    conn_proj.close()
    conn_main.close()
    
    # Create DataFrame
    report_df = pd.DataFrame(report_rows)
    
    # Sort by MS1 retention time (rt_theoretical) low to high
    if not report_df.empty:
        # Convert rt_peak_theoretical to numeric for proper sorting
        report_df['rt_peak_theoretical_num'] = pd.to_numeric(report_df['rt_peak_theoretical'], errors='coerce')
        report_df = report_df.sort_values('rt_peak_theoretical_num', ascending=True, na_position='last')
        report_df = report_df.drop(columns=['rt_peak_theoretical_num'])
        report_df = report_df.reset_index(drop=True)
        report_df['index'] = range(len(report_df))
    
    # Save to Excel if path provided with grouped headers
    if output_path and not report_df.empty:
        try:
            _save_report_with_grouped_headers(report_df, output_path)
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
    
    return report_df

def _calculate_msms_quality_score_from_notes(row: pd.Series) -> float:
    """Calculate MSMS quality score from ms2_notes selection"""
    ms2_notes = str(row.get('ms2_notes', 'no selection')).lower()
    
    if 'no selection' in ms2_notes:
        return 0.0
    
    numeric_match = re.match(r'^(\d*\.?\d+)', ms2_notes)
    if numeric_match:
        try:
            return float(numeric_match.group(1))
        except ValueError:
            return 0.0

    return 0.0

def _calculate_mz_quality_score(row: pd.Series) -> float:
    """Calculate m/z quality score based on ppm error"""
    ppm_error = row.get('best_eic_ppm_error', None)
    
    if pd.isna(ppm_error) or ppm_error is None:
        return 0.0
    
    ppm_error = abs(float(ppm_error))
    
    if ppm_error <= 5:
        return 1.0
    elif ppm_error <= 10:
        return 0.5
    else:
        return 0.0

def _calculate_rt_quality_score(row: pd.Series) -> float:
    """Calculate RT quality score based on retention time difference"""
    rt_error = row.get('best_eic_rt_error', None)
    
    if pd.isna(rt_error) or rt_error is None:
        return 0.0
    
    rt_error = abs(float(rt_error))
    
    if rt_error <= 0.5:
        return 1.0
    elif rt_error <= 2.0:
        return 0.5
    else:
        return 0.0

def _determine_msi_level(msms_quality: float, mz_quality: float, rt_quality: float) -> str:
    """Determine MSI identification level"""
    total_score = msms_quality + mz_quality + rt_quality
    
    if total_score >= 2.5 and msms_quality >= 0.5:
        return "Exceeds Level 1"
    elif total_score >= 2.0:
        return "Level 1"
    elif total_score >= 1.5:
        return "Level 2"
    elif total_score >= 1.0:
        return "Level 3"
    else:
        return "Level 4"

def _find_overlapping_compounds(conn_main, current_row: pd.Series, all_compounds: pd.DataFrame) -> Dict[str, str]:
    """Find overlapping compounds using the 'isomers' field in the current row."""
    isomers = current_row.get('isomers', [])
    if not isinstance(isomers, list) or not isomers:
        return {
            'compound_names': '',
            'inchi_keys': '',
        }
    compound_names = []
    inchi_keys = []
    for iso in isomers:
        name = iso.get('compound_name', 'Unknown')
        inchi = iso.get('inchi_key', '')
        if inchi and inchi != current_row.get('inchi_key', ''):
            compound_names.append(name)
            inchi_keys.append(inchi)
    return {
        'compound_names': '; '.join(compound_names),
        'inchi_keys': '; '.join(inchi_keys),
    }

def _create_empty_summary_from_atlas(atlas_df_ft: pd.DataFrame, analysis_uid: str, config: Dict) -> pd.DataFrame:
    """Create empty targeted analysis summary from atlas compounds"""
    empty_rows = []
    
    for _, row in atlas_df_ft.iterrows():
        empty_row = {
            'analysis_uid': analysis_uid,
            'compound_uid': row.get('compound_uid', ''),
            'inchi_key': row.get('inchi_key', ''),
            'compound_name': row.get('compound_name', row.get('label', '')),
            'formula': row.get('formula', ''),
            'polarity': row.get('polarity', ''),
            'mono_isotopic_molecular_weight': row.get('exact_mass', ''),
            'adduct': row.get('adduct', ''),
            'pre_rt_peak': row.get('rt_peak', ''),
            'pre_rt_min': row.get('rt_min', ''),
            'pre_rt_max': row.get('rt_max', ''),
            'pre_mz': row.get('mz', ''),
            'mz_tolerance': row.get('mz_tolerance', ''),
            'ms1_notes': 'keep',
            'ms2_notes': 'no selection',
            'best_eic_intensity': None,
            'best_eic_file': None,
            'best_eic_rt': None,
            'best_eic_mz': None,
            'best_eic_ppm_error': None,
            'best_eic_rt_error': None,
            'avg_eic_rt': None,
            'best_ms2_score': None,
            'best_ms2_file': None,
            'best_ms2_rt_peak': None,
            'best_ms2_num_matches': None,
            'best_ms2_frags_matching': None
        }
        empty_rows.append(empty_row)
    
    return pd.DataFrame(empty_rows)

def _add_missing_compounds_to_summary(base_summary: pd.DataFrame, atlas_df_ft: pd.DataFrame, 
                                     analysis_uid: str, config: Dict) -> pd.DataFrame:
    """Add missing compounds from atlas to existing summary"""
    existing_inchi_keys = set(base_summary['inchi_key'].dropna())
    atlas_inchi_keys = set(atlas_df_ft['inchi_key'].dropna())
    missing_inchi_keys = atlas_inchi_keys - existing_inchi_keys
    
    if missing_inchi_keys:
        logger.info(f"Adding {len(missing_inchi_keys)} missing compounds from atlas as empty rows")
        missing_atlas_df = atlas_df_ft[atlas_df_ft['inchi_key'].isin(missing_inchi_keys)]
        empty_summary = _create_empty_summary_from_atlas(missing_atlas_df, analysis_uid, config)
        base_summary = pd.concat([base_summary, empty_summary], ignore_index=True)
    
    return base_summary

def _save_report_with_grouped_headers(report_df: pd.DataFrame, output_path: str):
    """Save report to Excel with grouped column headers"""
    try:
        import openpyxl
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.styles import Font, Alignment, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.warning("openpyxl not available, saving as CSV instead")
        report_df.to_csv(output_path.replace('.xlsx', '.csv'), index=False)
        return
    
    # Create workbook and worksheet
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Targeted Analysis Report"
    
    # Define column groups and their ranges
    column_groups = [
        ("Compound Information", ["index", "identified_metabolite", "label", "isomer_compound", 
                                "isomer_inchi_keys", "formula", "polarity", "exact_mass", "inchi_key", "adduct"]),
        ("Identification Scores", ["msms_quality", "mz_quality", "rt_quality", "total_score", 
                                 "msi_level"]),
        ("Identification Notes", ["ms1_notes", "ms2_notes", "analyst_notes", "identification_notes"]),
        ("MS1 Information", ["max_intensity", "max_intensity_file", "max_intensity_rt"]),
        ("MS2 Information", ["best_msms_file", "best_msms_rt", "best_msms_num_matching_ions", "best_msms_matching_ions", "best_msms_score"]),
        ("MZ/RT Information", ["mz_theoretical", "mz_measured", "mz_error", "mz_ppmerror",
                              "rt_peak_theoretical", "rt_peak_measured", "rt_min_measured", "rt_max_measured", "rt_error"])
    ]
    
    # Create header mapping
    column_to_group = {}
    group_ranges = {}
    col_num = 1
    
    for group_name, columns in column_groups:
        start_col = col_num
        for col in columns:
            if col in report_df.columns:
                column_to_group[col] = group_name
                col_num += 1
        end_col = col_num - 1
        if end_col >= start_col:
            group_ranges[group_name] = (start_col, end_col)
    
    # Write grouped headers (row 1)
    for group_name, (start_col, end_col) in group_ranges.items():
        if start_col == end_col:
            ws.cell(row=1, column=start_col, value=group_name)
        else:
            ws.merge_cells(start_row=1, start_column=start_col, end_row=1, end_column=end_col)
            ws.cell(row=1, column=start_col, value=group_name)
        
        # Style the group header
        for col in range(start_col, end_col + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = Font(bold=True, size=12)
            cell.alignment = Alignment(horizontal='center', vertical='center')
            cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
    
    # Write column headers (row 2)
    col_num = 1
    for col in report_df.columns:
        ws.cell(row=2, column=col_num, value=col)
        cell = ws.cell(row=2, column=col_num)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center')
        col_num += 1
    
    # Write data starting from row 3
    for r_idx, row in enumerate(dataframe_to_rows(report_df, index=False, header=False), 3):
        for c_idx, value in enumerate(row, 1):
            # Handle NaN values
            if pd.isna(value):
                value = ''
            ws.cell(row=r_idx, column=c_idx, value=value)
    
    # Adjust column widths - fix for merged cells
    max_col = len(report_df.columns)
    for col_idx in range(1, max_col + 1):
        column_letter = get_column_letter(col_idx)
        max_length = 0
        
        # Check all cells in this column
        for row_idx in range(1, ws.max_row + 1):
            cell = ws.cell(row=row_idx, column=col_idx)
            try:
                cell_value = str(cell.value) if cell.value is not None else ''
                if len(cell_value) > max_length:
                    max_length = len(cell_value)
            except:
                pass
        
        # Set column width with reasonable limits
        adjusted_width = min(max(max_length + 2, 10), 50)  # Min 10, max 50 characters
        ws.column_dimensions[column_letter].width = adjusted_width
    
    # Save workbook
    wb.save(output_path)

def create_rt_corrected_atlas(
    project_db_path: str,
    source_atlas_uid: str,
    atlas_info: Dict,
    best_model: Dict,
    target_compounds_df: pd.DataFrame
) -> Tuple[str, List[Dict]]:
    """
    Create RT-corrected atlas in project database.
    
    Args:
        project_db_path: Path to project database
        source_atlas_uid: UID of source atlas
        atlas_info: Dictionary containing atlas information
        best_model: RT correction model dictionary
        target_compounds_df: DataFrame of compounds to correct
    
    Returns:
        Tuple of (corrected_atlas_uid, correction_stats_list)
    """
    prov = ldt.get_provenance()
    source_atlas_name = atlas_info["atlas_name"]
    source_chromatography = atlas_info["chromatography"]
    source_polarity = atlas_info["polarity"]

    corrected_atlas_name = f"{source_atlas_name} (RT Corrected)"
    corrected_atlas_uid = _generate_uid("rt_atlas")
    
    correction_stats = []
    
    with get_db_connection(project_db_path) as conn:
        # Clear any existing experimental data
        conn.execute("DELETE FROM mz_rt_experimental")
        
        # Create new atlas
        conn.execute("""
            INSERT INTO atlases (
                atlas_uid, atlas_name, atlas_description, chromatography, 
                polarity, created_by, last_modified
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            corrected_atlas_uid,
            corrected_atlas_name,
            f"RT-corrected version of {source_atlas_uid} using QC-based polynomial model",
            source_chromatography,
            source_polarity,
            prov["analyst"],
            prov["timestamp"]
        ])
        
        # Process each compound
        for _, compound in target_compounds_df.iterrows():
            compound_uid = compound['compound_uid']
            original_rt_peak = compound['rt_peak']
            original_rt_min = compound['rt_min']
            original_rt_max = compound['rt_max']

            # Apply RT correction using the polynomial model
            from metatlas2.rt_align_tools import predict_rt_correction
            corrected_rt_peak = predict_rt_correction([original_rt_peak], best_model)[0]
            corrected_rt_min = predict_rt_correction([original_rt_min], best_model)[0]
            corrected_rt_max = predict_rt_correction([original_rt_max], best_model)[0]
            rt_shift = corrected_rt_peak - original_rt_peak

            # Create new mz_rt_experimental entry
            mz_rt_experimental_uid = _generate_uid("mz_rt_experimental")
            
            conn.execute("""
                INSERT INTO mz_rt_experimental (
                    mz_rt_experimental_uid, compound_uid, rt_peak, rt_min, rt_max,
                    mz, mz_tolerance, adduct, chromatography, polarity,
                    created_by, last_modified, rt_correction_applied, rt_shift,
                    source_mz_rt_reference_uid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                mz_rt_experimental_uid, compound_uid, corrected_rt_peak, corrected_rt_min, 
                corrected_rt_max, compound['mz'], compound['mz_tolerance'],
                compound['adduct'], source_chromatography, source_polarity, 
                prov["analyst"], prov["timestamp"], True, rt_shift, compound['mz_rt_reference_uid']
            ])
            
            # Create atlas association
            association_uid = _generate_uid("association")
            conn.execute("""
                INSERT INTO atlas_compound_associations (
                    association_uid, atlas_uid, compound_uid, mz_rt_reference_uid, 
                    association_order, created_by, last_modified
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                association_uid, corrected_atlas_uid, compound_uid, mz_rt_experimental_uid, 
                len(correction_stats) + 1, prov["analyst"], prov["timestamp"]
            ])
            
            correction_stats.append({
                'compound_name': compound['compound_name'],
                'compound_uid': compound_uid,
                'mz_rt_reference_uid': compound['mz_rt_reference_uid'],
                'mz_rt_experimental_uid': mz_rt_experimental_uid,
                'original_rt': original_rt_peak,
                'corrected_rt': corrected_rt_peak,
                'rt_shift': rt_shift
            })
        
        conn.commit()
    
    logger.info(f"Created RT-corrected atlas '{corrected_atlas_name}' with {len(correction_stats)} associations")
    
    return corrected_atlas_uid, correction_stats

def create_rt_alignment_summary(
    rtc_atlas_name: str,
    correction_stats: List[Dict],
    total_compounds: int
) -> Dict:
    """    
    Args:
        rtc_atlas_name: Name of RT-corrected atlas
        correction_stats: List of correction statistics
        total_compounds: Total number of compounds
    
    Returns:
        Dictionary with summary statistics
    """
    
    correction_df = pd.DataFrame(correction_stats)
    summary = {
        'total_compounds': total_compounds,
        'corrected_compounds': len(correction_stats),
        'uncorrected_compounds': total_compounds - len(correction_stats),
        'correction_stats': correction_stats,
        'mean_correction': correction_df['rt_shift'].mean() if not correction_df.empty else 0,
        'std_correction': correction_df['rt_shift'].std() if not correction_df.empty else 0,
        'min_correction': correction_df['rt_shift'].min() if not correction_df.empty else 0,
        'max_correction': correction_df['rt_shift'].max() if not correction_df.empty else 0
    }
        
    return summary