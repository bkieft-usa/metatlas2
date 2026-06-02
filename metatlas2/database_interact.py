import pandas as pd
import numpy as np
import duckdb
import uuid
import grp
import re
import os
import sys
import json
import copy
import getpass
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager
from tqdm.auto import tqdm

import metatlas2.rt_align_tools as rat
import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
logger = lcf.get_logger('database_interact')

def should_disable_tqdm():
    return (
        "SLURM_JOB_ID" in os.environ
        or not sys.stdout.isatty()
    )

@contextmanager
def get_db_connection(db_path: str, read_only: bool = False, max_retries: int = 10, initial_retry_delay: float = 0.5):
    """
    Context manager for DuckDB connections with automatic retry on lock conflicts.
    
    Args:
        db_path: Path to the database file
        read_only: If True, open in read-only mode (allows concurrent readers)
        max_retries: Maximum number of retry attempts on lock conflicts (default 10 for writes)
        initial_retry_delay: Initial delay in seconds before first retry (doubles with each attempt)
    
    Raises:
        duckdb.IOException: If database lock cannot be acquired after all retries
    
    Notes:
        - Read operations should use read_only=True to allow concurrent access
        - Default retry parameters (~4 min total wait) handle typical multi-analyst conflicts
        - For heavy writes (bulk inserts, GUI updates), use max_retries=20, initial_retry_delay=1.0
    """
    retry_delay = initial_retry_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            conn = duckdb.connect(str(db_path), read_only=read_only)
            try:
                # Begin explicit transaction for write connections
                if not read_only:
                    conn.execute("BEGIN TRANSACTION")
                
                yield conn
                
                # Commit transaction and checkpoint WAL to release locks promptly
                if not read_only:
                    conn.execute("COMMIT")
                    conn.execute("CHECKPOINT")
                
                return  # Success - exit the retry loop
            except Exception as e:
                # Rollback on any error during the transaction
                if not read_only:
                    try:
                        conn.execute("ROLLBACK")
                    except:
                        pass  # Rollback may fail if connection is broken
                raise
            finally:
                conn.close()
        except duckdb.IOException as e:
            last_error = e
            # Check if this is a lock-related error
            if "lock" in str(e).lower() or "conflicting lock" in str(e).lower():
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database locked at {db_path}, retrying in {retry_delay:.2f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        f"Failed to acquire database lock after {max_retries} attempts. "
                        f"Another process may be holding the lock. Error: {e}"
                    )
                    raise
            else:
                # Not a lock error, raise immediately
                raise
    
    # Should not reach here, but just in case
    if last_error:
        raise last_error

def get_provenance():
    """Get provenance information for database records."""
    return {
        "analyst": getpass.getuser(),
        "timestamp": datetime.now().isoformat()
    }

def _generate_uid(entity_type: str, decorator: str = None) -> str:
    """Generate a unique identifier for database entities."""
    if entity_type == "ref_atlas":
        return f"atl-ref-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-ref-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_atlas":
        return f"atl-rta-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "autoid_atlas":
        return f"atl-aid-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-aid-{uuid.uuid4().hex[:32]}"
    elif entity_type == "curated_atlas":
        return f"atl-mcr-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-mcr-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt":
        return f"mzrt-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"mzrt-{uuid.uuid4().hex[:32]}"
    elif entity_type == "compound":
        return f"cmp-{uuid.uuid4().hex[:32]}"
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
    elif entity_type == "ms2_hits":
        return f"ms2-hits-{uuid.uuid4().hex[:32]}"
    elif entity_type == "manual_curation":
        return f"mcr-{uuid.uuid4().hex[:32]}"
    elif entity_type == "project":
        return f"prj-{uuid.uuid4().hex[:32]}"
    else:
        raise ValueError(f"Unknown entity type: {entity_type}")

def create_new_atlas_from_dataframe(
    atlas_df: pd.DataFrame, 
    atlas_name: str, 
    atlas_description: str, 
    analysis_type: str,
    chromatography: str = None, 
    polarity: str = None,
    atlas_file_path: str = None,
    main_db_path: str = None
) -> "Atlas":
    
    from metatlas2.workflow_objects import Atlas, CompoundMZRT

    if not chromatography:
        chromatography = ldt.detect_atlas_input_chromatography(atlas_df)
    if not polarity:
        polarity = ldt.detect_atlas_input_polarity(atlas_df)        

    # Required columns for constructing CompoundMZRTs (mz_rt_uid will be generated and is the unique key)
    required_cols = ['inchi_key', 'adduct', 'rt_peak', 'mz']
    missing_cols = [col for col in required_cols if col not in atlas_df.columns]
    if missing_cols:
        raise ValueError(f"Atlas input dataframe is missing required columns: {missing_cols}. Found columns: {atlas_df.columns.tolist()}")

    # inchi_key and adduct are attributes, not unique identifiers; mz_rt_uid is the unique key for all compounds
    inchi_keys = atlas_df['inchi_key'].dropna().unique().tolist()
    compound_lookup = get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)

    # Use mz_rt_uid as the unique key for each CompoundMZRT
    compound_mzrts = {}
    for _, row in atlas_df.iterrows():
        inchi_key = row.get('inchi_key', '')
        adduct = str(row.get('adduct', None))
        if not inchi_key or inchi_key not in compound_lookup:
            logger.warning(f"Compound with inchi_key {inchi_key} missing from metatlas database, skipping. (Note: mz_rt_uid is the unique identifier for all compounds.)")
            continue
        compound_uid = compound_lookup[inchi_key]
        compound_name = str(row.get('compound_name', row.get('label', 'Unknown Compound')))
        rt_peak = row.get('rt_peak', None)
        rt_min = row.get('rt_min', rt_peak - 0.5)
        rt_max = row.get('rt_max', rt_peak + 0.5)
        mz = row.get('mz', None)
        mz_tolerance = row.get('mz_tolerance', 5.0)
        rt_space = row.get('rt_space', 'HF_Aug2019')
        if rt_peak is None or mz is None or adduct is None:
            raise ValueError(f"Compound {inchi_key} missing essential data (rt_peak: {rt_peak}, mz: {mz}, adduct: {adduct}), cannot create reference. (mz_rt_uid is the unique identifier for all compounds.)")
        confidence_level = row.get('confidence_level', None)
        identification_notes = row.get('identification_notes', '')

        mz_rt_uid = _generate_uid("mz_rt", decorator="ref")
        compound_mzrt = CompoundMZRT(
            mz_rt_uid=mz_rt_uid,
            compound_uid=compound_uid,
            compound_name=compound_name,
            inchi_key=inchi_key,
            adduct=adduct,
            rt_space=rt_space,
            rt_peak=rt_peak,
            rt_min=rt_min,
            rt_max=rt_max,
            mz=mz,
            mz_tolerance=mz_tolerance,
            chromatography=chromatography,
            polarity=polarity,
            confidence=confidence_level,
            source=atlas_file_path,
            identification_notes=identification_notes,
        )
        compound_mzrts[mz_rt_uid] = compound_mzrt

    atlas_uid = _generate_uid("ref_atlas", decorator=f"{analysis_type.lower()}-{chromatography.lower()}-{polarity.lower()}")
    atlas_obj = Atlas(
        atlas_uid=atlas_uid,
        atlas_name=atlas_name,
        atlas_description=atlas_description,
        chromatography=chromatography,
        polarity=polarity,
        analysis_type=analysis_type,
        atlas_type="REFERENCE",
        compound_mzrts=compound_mzrts,
        source=atlas_file_path
    )
    return atlas_obj

def get_atlas_compounds_table(database_path: str, atlas_uid: str, main_db_path: str = None) -> pd.DataFrame:
    """
    Extract all compound information for a given atlas UID from the database.
    Handles both main database and project database.
    """
    with get_db_connection(database_path, read_only=True) as conn:

        # Attach main database if needed for compound metadata
        if main_db_path:
            try:
                conn.execute(f"ATTACH '{main_db_path}' AS main_db (READ_ONLY)")
                logger.info("Attached main database for compound metadata")
            except Exception as e:
                logger.error(f"Error attaching main database: {e}")
                return pd.DataFrame()

        try:
            if main_db_path:
                # Project DB: Join through associations to experimental entries
                query = """
                    SELECT
                        a.atlas_uid,
                        a.atlas_name,
                        a.atlas_description,
                        a.chromatography,
                        a.polarity,
                        a.analysis_type,
                        a.atlas_type,
                        aca.compound_uid,
                        COALESCE(main_db.compounds.compound_name, '') AS compound_name,
                        COALESCE(main_db.compounds.inchi_key, '') AS inchi_key,
                        COALESCE(main_db.compounds.inchi, '') AS inchi,
                        mzrt.adduct,
                        mzrt.mz,
                        mzrt.rt_peak,
                        mzrt.rt_min,
                        mzrt.rt_max,
                        mzrt.mz_tolerance,
                        mzrt.mz_rt_uid AS mz_rt_uid,
                    FROM atlases a
                    JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                    LEFT JOIN main_db.compounds ON aca.compound_uid = main_db.compounds.compound_uid
                    LEFT JOIN compound_mzrt mzrt 
                        ON aca.mz_rt_uid = mzrt.mz_rt_uid
                    WHERE a.atlas_uid = ?
                    ORDER BY aca.association_order
                """
            else:
                # Main DB: Join through associations to reference entries
                query = """
                    SELECT
                        a.atlas_uid,
                        a.atlas_name,
                        a.atlas_description,
                        a.chromatography,
                        a.polarity,
                        a.analysis_type,
                        a.atlas_type,
                        c.compound_uid,
                        c.compound_name,
                        c.inchi_key,
                        c.inchi,
                        mzrt.adduct,
                        mzrt.mz,
                        mzrt.rt_peak,
                        mzrt.rt_min,
                        mzrt.rt_max,
                        mzrt.mz_tolerance,
                        mzrt.mz_rt_uid,
                    FROM atlases a
                    JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
                    JOIN compounds c ON aca.compound_uid = c.compound_uid
                    LEFT JOIN compound_mzrt mzrt 
                        ON aca.mz_rt_uid = mzrt.mz_rt_uid
                    WHERE a.atlas_uid = ?
                    ORDER BY aca.association_order
                """

            df = conn.execute(query, [atlas_uid]).df()

        except Exception as e:
            logger.error(f"Error querying atlas {atlas_uid}: {e}")
            return pd.DataFrame()
        
        finally:
            if main_db_path:
                conn.execute("DETACH main_db")

    if df.empty:
        raise ValueError(f"No compounds found for atlas UID {atlas_uid} in database at {database_path}")
    else:
        df['compound_name'] = df['compound_name'] if 'compound_name' in df.columns else ''
        logger.info(f"Retrieved {len(df)} compounds for atlas {atlas_uid} ({df['atlas_name'].iloc[0]})")

    return df

def create_project_database(
    project_db_path: str, 
    rt_align_path: str,
    log_file_path: str = None,
    overwrite: bool = False
) -> bool:
    """
    Create project-specific database with required tables.
    Never overwrites existing databases - requires analyst to increment analysis number.
    """

    project_db_path = Path(project_db_path)
    rt_alignment_path = Path(rt_align_path)
    log_file_path = Path(log_file_path) if log_file_path else None

    if project_db_path.exists() and not overwrite:
        logger.info(f"Project database already exists at {project_db_path}. Not overwriting.")
        return True
    elif project_db_path.exists() and overwrite:
        logger.warning(f"Overwriting existing project log and database at {project_db_path} (overwrite=True)")
        log_file_path.unlink() if log_file_path and Path(log_file_path).exists() else None
        project_db_path.unlink() if project_db_path.exists() else None
        logger.info(f"Deleted existing database at {project_db_path}")
        logger.warning(f"Overwriting existing RT alignment output at {rt_alignment_path} (overwrite=True)")
        tga_pattern = re.compile(r'^TGA\d+$')
        for item in Path(rt_align_path).iterdir():
            if item.is_dir() and tga_pattern.match(item.name):
                for sub_item in item.iterdir():
                    shutil.rmtree(sub_item) if sub_item.is_dir() else sub_item.unlink()
            else:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
        logger.info(f"Cleared existing RT alignment output at {rt_alignment_path}")
    elif not project_db_path.exists():
        logger.info(f"No existing project database found at {project_db_path}. Creating new database.")

    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        _create_database_tables(conn, db_type="project")
    logger.info(f"Project database created at {project_db_path}")

    return False

def create_metatlas_database(db_path: str, overwrite: bool = False) -> None:
    """
    Create main metatlas database with required tables.
    If database already exists and overwrite=False, does nothing (allows adding to existing DB).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if overwrite and db_path.exists():
        logger.warning("Overwriting existing main database because overwrite=True")
        db_path.unlink()
        logger.info(f"Deleted existing database at {db_path} and creating new database.")
    elif overwrite and not db_path.exists():
        logger.info(f"No existing main database found at {db_path}. Creating new database.")
    elif not overwrite and db_path.exists():
        logger.info(f"Database already exists at {db_path}. Skipping database creation.")
        return
    elif not overwrite and not db_path.exists():
        logger.info(f"No existing main database found at {db_path}. Creating new database.")

    with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        _create_database_tables(conn, db_type="main")

    db_path.chmod(0o660)
    try:
        metatlas_gid = grp.getgrnam("metatlas").gr_gid
        os.chown(db_path, -1, metatlas_gid)
    except KeyError:
        logger.info("Group 'metatlas' not found. Skipping group ownership change.")

    logger.info(f"Main metatlas database created at {db_path}")

def save_rt_alignment_model_to_db(
    rt_align_obj: "RTAlign",
) -> str:
    """Save RT alignment model to project database using RTAlign object and LCMSRun list."""

    logger.info("Saving RT alignment model to project database...")

    rt_alignment_uid = _generate_uid("rt_alignment")
    qc_files = [run.file_path for run in rt_align_obj.aligner_lcmsruns]
    project_db_path = Path(rt_align_obj.paths['project_db_path'])
    project_name = project_db_path.stem.replace('.duckdb', '')
    prov = get_provenance()

    rt_alignment_model = rt_align_obj.rt_alignment_model
    modeling_data = (
        rt_align_obj.modeling_data.to_dict('records')
        if hasattr(rt_align_obj.modeling_data, 'to_dict')
        else rt_align_obj.modeling_data
    )

    model_metadata = {
        "qc_files": [os.path.basename(f) for f in qc_files],
        "compounds_used": [d.get('compound_uid', '') for d in modeling_data],
        "alignment_timestamp": prov["timestamp"],
        "alignment_method": "polynomial_qc_based",
        "analyst": prov["analyst"],
        "poly_degree": int(rt_alignment_model['degree']),
        "poly_include_bias": bool(rt_alignment_model.get('poly_features').include_bias if rt_alignment_model.get('poly_features') else True),
        "poly_interaction_only": bool(rt_alignment_model.get('poly_features').interaction_only if rt_alignment_model.get('poly_features') else False),
        "model_intercept": float(rt_alignment_model.get('intercept', 0.0)),
        "model_coefficients": rt_alignment_model['coefficients'].tolist() if hasattr(rt_alignment_model['coefficients'], 'tolist') else list(rt_alignment_model['coefficients'])
    }

    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        conn.execute("""
            INSERT INTO rt_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rt_alignment_uid,
            project_name,
            rt_align_obj.rt_alignment_number,
            rt_align_obj.qc_atlas_uid,
            "polynomial",
            rt_alignment_model['degree'],
            rt_alignment_model['r2'],
            rt_alignment_model['rmse'],
            json.dumps(rt_alignment_model['coefficients'].tolist()),
            rt_alignment_model['equation'],
            len(qc_files),
            len(modeling_data),
            prov["analyst"],
            prov["timestamp"],
            json.dumps(model_metadata)
        ))

    logger.info(f"RT alignment model saved to database with UID: {rt_alignment_uid}")

    rt_alignment_model['rt_alignment_uid'] = rt_alignment_uid

    return rt_alignment_uid

def get_rt_alignment_model_from_db(
    rt_align_obj: "RTAlign"
) -> dict:
    """
    Retrieve RT alignment model from project database and reconstruct sklearn objects.
    
    Args:
        project_db_path: Path to project database
        qc_atlas_uid: QC atlas UID to retrieve
        rt_alignment_number: RT alignment number to retrieve
    """
    
    project_db_path = Path(rt_align_obj.paths['project_db_path'])
    rt_alignment_number = rt_align_obj.rt_alignment_number
    qc_atlas_uid = rt_align_obj.qc_atlas_uid
    
    with get_db_connection(project_db_path, read_only=True) as conn:
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

# def add_compounds_to_db(input_df: pd.DataFrame, db_path: str, pubchem_cache_path: str, input_file_path: str = ""):
#     """Add compounds and RT/MZ references to database using batch operations."""

#     if not os.path.exists(db_path):
#         logger.error(f"Database not found at {db_path}. Check path or create it first!")
#         raise FileNotFoundError(f"Database not found at {db_path}")

#     unique_inchi_keys = input_df['inchi_key'].dropna().unique()
#     logger.info(f"Adding {len(unique_inchi_keys)} compounds to database: {db_path}")
    
#     pubchem_cache = pcr.load_or_create_pubchem_cache(pubchem_cache_path)
#     prov = get_provenance()

#     with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
#         # Get existing compounds in one query
#         existing_inchi_keys = set()
#         existing_compounds_map = {}
#         existing_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
#         existing_inchi_keys = {row[0] for row in existing_result}
#         existing_compounds_map = {row[0]: row[1] for row in existing_result}
#         if existing_inchi_keys:
#             logger.warning(f"Found {len(existing_inchi_keys)} existing compounds in database. Not creating duplicates.")

#         # Prepare batch data
#         compound_records = []
#         reference_records = []
#         compounds_created = 0
#         compounds_skipped = 0

#         logger.info("Preparing batch data...")
#         for idx, row in input_df.iterrows():
#             inchi_key = row.get('inchi_key')
#             if pd.isna(inchi_key):
#                 continue
            
#             chromatography = str(row.get('chromatography', 'Unknown'))
#             polarity = str(row.get('polarity', 'Unknown'))
#             compound_uid = None
            
#             # Check if compound exists
#             if inchi_key in existing_inchi_keys:
#                 compounds_skipped += 1
#                 compound_uid = existing_compounds_map.get(inchi_key)
#             else:
#                 # Prepare new compound record
#                 compound_uid = _generate_uid("compound")
#                 compound_record = _prepare_compound_record(row, compound_uid, pubchem_cache, prov)
#                 compound_records.append(compound_record)
#                 compounds_created += 1
                
#                 # Add to existing sets for future duplicate checking in this session
#                 existing_inchi_keys.add(inchi_key)
#                 existing_compounds_map[inchi_key] = compound_uid
            
#             # Prepare RT/MZ reference if data available and compound_uid exists
#             if compound_uid and pd.notna(row.get('rt_peak')) and pd.notna(row.get('mz')):
#                 reference_record = _prepare_reference_record(
#                     row, compound_uid, chromatography, polarity, input_file_path, prov
#                 )
#                 if reference_record:
#                     reference_records.append(reference_record)

#         # Batch insert compounds
#         mzrts_created = 0
#         if compound_records:
#             logger.info(f"Batch inserting {len(compound_records)} compounds...")
#             conn.executemany("""
#                 INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#             """, compound_records)

#         # Check for existing references and batch insert if necessary
#         if reference_records:
#             logger.info(f"Checking for existing references and batch inserting if necessary...")
#             # Get existing references to check for duplicates
#             existing_refs = set()
#             existing_ref_result = conn.execute("""
#                 SELECT compound_uid, chromatography, polarity, adduct 
#                 FROM compound_mzrt
#             """).fetchall()
#             existing_refs = {(row[0], row[1], row[2], row[3]) for row in existing_ref_result}

#             # Filter out duplicates
#             filtered_reference_records = []
#             mzrts_skipped = 0
            
#             for record in reference_records:
#                 # Extract compound_uid, chromatography, polarity, adduct from record
#                 compound_uid = record[1]
#                 chromatography = record[8]
#                 polarity = record[9]
#                 adduct = record[7]
                
#                 ref_key = (compound_uid, chromatography, polarity, adduct)
#                 if ref_key not in existing_refs:
#                     filtered_reference_records.append(record)
#                     existing_refs.add(ref_key)  # Add to set to prevent duplicates within this batch
#                     mzrts_created += 1
#                 else:
#                     mzrts_skipped += 1

#             # Batch insert filtered references
#             if filtered_reference_records:
#                 conn.executemany("""
#                     INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
#                 """, filtered_reference_records)
        
#         # Get final counts
#         compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
#         mzrts_count = conn.execute("SELECT COUNT(*) FROM compound_mzrt").fetchone()[0]
    
#     logger.info("Compounds added successfully!")
#     logger.info(f"   Total compounds in database: {compounds_count}")
#     logger.info(f"   New compounds created: {compounds_created}")
#     if compounds_skipped > 0:
#         logger.info(f"   Compounds skipped (already existed): {compounds_skipped}")
#     logger.info(f"   Total RT/MZ references in database: {mzrts_count}")
#     logger.info(f"   New RT/MZ references created: {mzrts_created}")
#     if 'mzrts_skipped' in locals() and mzrts_skipped > 0:
#         logger.info(f"   RT/MZ references skipped (duplicates): {mzrts_skipped}")
    
#     return

def _prepare_compound_record(row: pd.Series, compound_uid: str, pubchem_cache: Dict, prov: Dict) -> tuple:
    """Prepare a compound record tuple for batch insertion."""
    # Extract compound data from row
    compound_name = str(row.get('compound_name', 'Unknown Compound'))
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
    classes = str(row.get('classes', '')) if pd.notna(row.get('classes')) else None
    pathways = str(row.get('pathways', '')) if pd.notna(row.get('pathways')) else None
    tags = str(row.get('tags', '')) if pd.notna(row.get('tags')) else None

    return (
        compound_uid,
        compound_name,
        inchi_key,
        inchi,
        smiles,
        formula,
        classes,
        pathways,
        tags,
        mono_isotopic_molecular_weight,
        iupac_name,
        pubchem_cid,
        cas_number,
        synonyms,
        prov["analyst"],
        prov["timestamp"]
    )

def _prepare_compound_record_from_dict(compound_data: Dict) -> Optional[Tuple]:
    """Prepare compound record from dictionary data."""
    try:
        prov = get_provenance()
        
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
            compound_data.get('compound_name', 'Unknown Compound'),
            inchi_key,
            compound_data.get('inchi'),
            compound_data.get('smiles'),
            compound_data.get('formula'),
            compound_data.get('classes'),
            compound_data.get('pathways'),
            compound_data.get('tags'),
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

def _prepare_reference_record(row: pd.Series, compound_uid: str, chromatography: str, 
                            polarity: str, input_file_path: str, prov: Dict) -> tuple:
    """Prepare a reference record tuple for batch insertion."""
    try:
        uid = _generate_uid("mz_rt")
        
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

def _prepare_reference_record_from_dict(reference_data: Dict) -> Optional[Tuple]:
    """Prepare reference record from dictionary data."""
    try:
        prov = get_provenance()
        
        return (
            _generate_uid("mz_rt", decorator="ref"),
            reference_data.get('compound_uid'),
            reference_data.get('compound_name'),
            reference_data.get('inchi_key'),
            reference_data.get('adduct'),
            reference_data.get('rt_space'),
            float(reference_data.get('rt_peak')),
            float(reference_data.get('rt_min')),
            float(reference_data.get('rt_max')),
            float(reference_data.get('mz')),
            float(reference_data.get('mz_tolerance', 5.0)),
            reference_data.get('chromatography'),
            reference_data.get('polarity'),
            reference_data.get('confidence', 'Unknown'),
            reference_data.get('source', 'Unknown'),
            reference_data.get('identification_notes', ''),
            prov["analyst"],
            prov["timestamp"]
        )
    except (ValueError, TypeError) as e:
        logger.error(f"Error preparing reference record: {e}")
        return None

def save_project_to_main_db(
    main_db_path: str,
    project_name: str,
    project_db_path: str
) -> str:
    """
    Register a project in the main database for meta-analysis tracking.
    
    Args:
        main_db_path: Path to the main database
        project_name: Name of the project
        project_db_path: Absolute path to the project database
        
    Returns:
        project_uid: Unique identifier for the project
    """
    with get_db_connection(main_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        prov = get_provenance()
        
        # Check if project already exists
        existing = conn.execute(
            "SELECT project_uid FROM projects WHERE project_name = ? AND project_db_path = ?",
            (project_name, project_db_path)
        ).fetchone()
        
        if existing:
            logger.info(f"Project '{project_name}' already registered in main database with UID {existing[0]}.")
            return existing[0]
        
        # Generate new project UID and insert
        project_uid = _generate_uid("project")
        conn.execute(
            "INSERT INTO projects VALUES (?, ?, ?, ?, ?)",
            (
                project_uid,
                project_name,
                project_db_path,
                prov["analyst"],
                prov["timestamp"],
            ),
        )
        
        logger.info(f"Registered project '{project_name}' in main database with UID {project_uid}.")
        return project_uid

def save_lcmsruns_to_db(
    project_db_path: str,
    project_name: str,
    lcmsruns_list: List[Dict],
    overwrite_existing: bool = False
) -> int:
    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        prov = get_provenance()
        
        # Simplified check: try to count rows; if table doesn't exist, it will fail or return 0
        # Note: For SQLite, use 'sqlite_master'. For others, information_schema is fine.
        try:
            has_rows = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0] > 0
        except Exception:
            has_rows = False

        if has_rows:
            if not overwrite_existing:
                logger.info("LCMS runs already exist. Skipping save.")
                return 0
            logger.info(f"Overwriting existing runs for {project_name}.")
            conn.execute("DELETE FROM lcmsruns")

        # EFFICIENT: Bulk insertion using executemany
        # Prepare data as a list of tuples
        data_to_insert = [
            (
                run["file_path"], run["filename"], run["file_format"], run["file_type"],
                run["chromatography"], run["ms_level"], run["polarity"],
                prov["analyst"], prov["timestamp"]
            )
            for run in lcmsruns_list
        ]
        
        conn.executemany(
            "INSERT INTO lcmsruns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
            data_to_insert
        )

    logger.info(f"Saved {len(lcmsruns_list)} LCMS runs to database.")
    return len(lcmsruns_list)


def get_lcmsruns_from_db(
    project_db_path: str,
    file_types: Optional[List[str]] = None,
    file_format: str = "h5",
    chromatography: str = None,
    polarity: str = None
) -> List["LCMSRun"]:
    from metatlas2.workflow_objects import LCMSRun

    # Normalization
    if chromatography == "HILICZ": chromatography = "HILIC"
    if polarity:
        polarity = "positive" if polarity.lower() in ["pos", "positive"] else \
                   "negative" if polarity.lower() in ["neg", "negative"] else polarity

    where_conditions = ["file_format = ?"]
    params = [file_format]

    if chromatography:
        where_conditions.append("UPPER(chromatography) = UPPER(?)")
        params.append(chromatography)
    if polarity:
        where_conditions.append("UPPER(polarity) = UPPER(?)")
        params.append(polarity)
    if file_types:
        where_conditions.append(f"file_type IN ({','.join(['?']*len(file_types))})")
        params.extend(file_types)

    where_clause = " AND ".join(where_conditions)
    
    with get_db_connection(project_db_path, read_only=True) as conn:
        # EFFICIENT: Use cursor directly instead of Pandas .df()
        cursor = conn.execute(f"""
            SELECT file_path, filename, file_format, file_type, chromatography, ms_level, polarity, created_by, created_date
            FROM lcmsruns
            WHERE {where_clause}
            ORDER BY chromatography, ms_level, polarity, file_path
        """, params)
        
        # Use a list comprehension directly on the cursor
        # This avoids the overhead of creating a DataFrame and iterating via Series
        lcmsruns_list = [LCMSRun(*row) for row in cursor.fetchall()]

    logger.info(f"Retrieved {len(lcmsruns_list)} {file_format}-formatted runs from DB.")
    return lcmsruns_list

def batch_save_compounds(
    db_path: str,
    compounds: List["Compound"],
) -> int:
    """
    Schema-compliant batch save for compounds and references from raw input data.
    
    Schema Logic:
    1. For each compound_data: check if inchi_key exists in database
       - If exists: get existing compound_uid, skip compound creation
       - If not exists: create new compound with new compound_uid
    2. For each mzrt_data: find corresponding compound_uid, then check if identical mzrt exists
       - If identical: skip mzrt creation
       - If different: create new mzrt entry
    
    Args:
        new_compound_obj: NewCompound object containing compounds and mzrts data

    Returns:
        Tuple of (compounds_created)
    """

    compounds_data = [compound.to_dict() for compound in compounds]

    compounds_created = 0
    compounds_skipped_existing = 0

    with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        # Get ALL existing compounds (inchi_key -> compound_uid mapping)
        existing_compounds = {}  # inchi_key -> compound_uid
        existing_result = conn.execute("SELECT inchi_key, compound_uid FROM compounds").fetchall()
        for row in existing_result:
            existing_compounds[row[0]] = row[1]

        logger.info(f"Found {len(existing_compounds)} existing compounds in database")

        # Process compound data and build inchi_key -> compound_uid mapping
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

        # Batch insert new compounds
        if compound_records:
            logger.info(f"Creating {len(compound_records)} new compounds...")
            conn.executemany("""
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, compound_records)

    # Log summary
    logger.info("Batch save completed:")
    logger.info(f"  Compounds created: {compounds_created}")
    logger.info(f"  Compounds skipped (already exist): {compounds_skipped_existing}")

    return

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
            SELECT mz_rt_uid 
            FROM compound_mzrt 
            WHERE compound_uid = ? 
            AND chromatography = ? 
            AND polarity = ? 
            AND adduct = ?
            AND rt_peak = ?
            AND rt_min = ?
            AND rt_max = ?
            AND mz = ?
            AND mz_tolerance = ?
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
                compound_name TEXT,
                inchi_key TEXT,
                inchi TEXT,
                smiles TEXT,
                formula TEXT,
                classes TEXT,
                pathways TEXT,
                tags TEXT,
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
            CREATE TABLE IF NOT EXISTS compound_mzrt (
                mz_rt_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                compound_name TEXT,
                inchi_key TEXT,
                adduct TEXT,
                rt_space TEXT,
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                mz REAL,
                mz_tolerance REAL,
                chromatography TEXT,
                polarity TEXT,
                confidence TEXT,
                source TEXT,
                identification_notes TEXT,
                created_by TEXT,
                created_date TEXT,
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlases (
                atlas_uid TEXT PRIMARY KEY,
                atlas_name TEXT,
                atlas_description TEXT,
                chromatography TEXT,
                polarity TEXT,
                analysis_type TEXT,
                atlas_type TEXT,
                created_by TEXT,
                created_date TEXT,
                source TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid TEXT PRIMARY KEY,
                atlas_uid TEXT,
                compound_uid TEXT,
                mz_rt_uid TEXT,
                association_order INTEGER,
                created_by TEXT,
                created_date TEXT,
                FOREIGN KEY (atlas_uid) REFERENCES atlases (atlas_uid),
                FOREIGN KEY (compound_uid) REFERENCES compounds (compound_uid),
                FOREIGN KEY (mz_rt_uid) REFERENCES compound_mzrt (mz_rt_uid)
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_uid TEXT PRIMARY KEY,
                project_name TEXT,
                project_db_path TEXT,
                created_by TEXT,
                created_date TEXT
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
                ms_level TEXT,
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
                analysis_type TEXT,
                atlas_type TEXT,
                source_atlas_uid TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                created_by TEXT,
                created_date TEXT,
                source TEXT
            )
        """)
        
        # conn.execute("""
        #     CREATE TABLE IF NOT EXISTS compounds (
        #         compound_uid TEXT PRIMARY KEY,
        #         compound_name TEXT,
        #         inchi_key TEXT,
        #         inchi TEXT,
        #         smiles TEXT,
        #         formula TEXT,
        #         classes TEXT,
        #         pathways TEXT,
        #         tags TEXT,
        #         mono_isotopic_molecular_weight REAL,
        #         iupac_name TEXT,
        #         pubchem_cid TEXT,
        #         cas_number TEXT,
        #         synonyms TEXT,
        #         created_by TEXT,
        #         created_date TEXT
        #     )
        # """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compound_mzrt (
                mz_rt_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                compound_name TEXT,
                inchi_key TEXT,
                adduct TEXT,
                rt_space TEXT,
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                mz REAL,
                mz_tolerance REAL,
                chromatography TEXT,
                polarity TEXT,
                confidence TEXT,
                source TEXT,
                identification_notes TEXT,
                created_by TEXT,
                created_date TEXT,
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS atlas_compound_associations (
                association_uid TEXT PRIMARY KEY,
                atlas_uid TEXT,
                compound_uid TEXT,
                mz_rt_uid TEXT,
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
                mz_rt_uid VARCHAR,
                filename VARCHAR,
                inchi_key VARCHAR,
                adduct VARCHAR,
                spec_rts REAL[],           
                spec_ints REAL[],   
                spec_mzs REAL[],  
                in_feature BOOLEAN[],
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type VARCHAR,      
                created_by VARCHAR,
                created_date VARCHAR
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_data (
                mz_rt_uid VARCHAR,
                filename VARCHAR,
                inchi_key VARCHAR,
                adduct VARCHAR,
                scan_rt REAL,
                frag_mzs REAL[],           
                frag_ints REAL[],          
                precursor_MZ REAL,
                precursor_intensity REAL,
                collision_energy REAL,
                in_feature BOOLEAN,
                hits VARCHAR,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type VARCHAR,
                created_by VARCHAR,
                created_date VARCHAR
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_curation (
                mz_rt_uid VARCHAR,
                compound_uid VARCHAR,
                inchi_key VARCHAR,
                adduct VARCHAR,
                compound_name VARCHAR,
                auto_ided BOOLEAN,
                polarity VARCHAR,
                chromatography VARCHAR,
                mz_tolerance REAL,
                atlas_mz REAL,
                atlas_rt_peak REAL,
                atlas_rt_min REAL,
                atlas_rt_max REAL,
                mz REAL,
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                initial_rt_min REAL,
                initial_rt_max REAL,
                rt_error REAL,
                mz_error REAL,
                ms1_notes VARCHAR,
                ms2_notes VARCHAR,
                other_notes VARCHAR,
                identification_notes VARCHAR,
                analyst_notes VARCHAR,
                best_ms1_file VARCHAR,
                best_ms1_rt REAL,
                best_ms1_mz REAL,
                best_ms1_intensity REAL,
                best_ms1_ppm_error REAL,
                best_ms1_rt_error REAL,
                max_eic_rt REAL[],
                max_eic_intensity REAL[],
                isomers VARCHAR,
                suggested_rt_min REAL,
                suggested_rt_max REAL,
                suggested_rt_peak REAL,
                rt_suggestion_confidence REAL,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type VARCHAR,
                created_by VARCHAR,
                created_date VARCHAR,
            )
        """)

def save_atlas_to_database(atlas_obj: "Atlas", db_path: str, main_db_path: str = None) -> None:
    """
    Save an Atlas object to the database (typing not included to avoid circular imports).
    Only creates new compound_mzrt entries for references that don't already exist.
    
    Args:
        atlas_obj: Atlas object to save
    """

    logger.info(f"Saving atlas {atlas_obj.atlas_name} to database at {db_path}...")
    prov = get_provenance()
    with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        if not main_db_path: # This will be a main database save, since db_path will be main
            
            # Verify all compounds exist in main database
            if not _verify_compounds_exist_in_db([comp.compound_uid for comp in atlas_obj.compound_mzrts.values()], conn):
                raise ValueError(f"Some compounds in atlas {atlas_obj.atlas_uid} don't exist in database")

            # Create atlas entry
            conn.execute("""
                INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atlas_obj.atlas_uid,
                atlas_obj.atlas_name,
                atlas_obj.atlas_description,
                atlas_obj.chromatography,
                atlas_obj.polarity,
                atlas_obj.analysis_type,
                atlas_obj.atlas_type,
                prov["analyst"],
                prov["timestamp"],
                atlas_obj.source
            ))
            
            # Process each CompoundMZRT
            association_order = 0
            mzrts_created = 0
            mzrts_reused = 0
            for inchi_key, compound_mzrt in atlas_obj.compound_mzrts.items():
                # Check if this reference already exists in database
                existing_check = conn.execute("""
                    SELECT mz_rt_uid FROM compound_mzrt 
                    WHERE mz_rt_uid = ?
                """, [compound_mzrt.mz_rt_uid]).fetchone()
                
                mz_rt_uid = compound_mzrt.mz_rt_uid
                
                # Create new reference if it doesn't exist
                if not existing_check:
                    conn.execute("""
                        INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mz_rt_uid,
                        compound_mzrt.compound_uid,
                        compound_mzrt.compound_name,
                        compound_mzrt.inchi_key,
                        compound_mzrt.adduct,
                        compound_mzrt.rt_space,
                        compound_mzrt.rt_peak,
                        compound_mzrt.rt_min,
                        compound_mzrt.rt_max,
                        compound_mzrt.mz,
                        compound_mzrt.mz_tolerance,
                        compound_mzrt.chromatography,
                        compound_mzrt.polarity,
                        compound_mzrt.confidence,
                        compound_mzrt.source,
                        compound_mzrt.identification_notes,
                        prov["analyst"],
                        prov["timestamp"]
                    ))
                    mzrts_created += 1
                else:
                    mzrts_reused += 1
                
                # Create atlas-compound association
                assoc_uid = _generate_uid("association")
                conn.execute("""
                    INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    assoc_uid,
                    atlas_obj.atlas_uid,
                    compound_mzrt.compound_uid,
                    mz_rt_uid,
                    association_order,
                    prov["analyst"],
                    prov["timestamp"]
                ))
                
                association_order += 1

        else: # This will be a project database save, so we need to attach the main db for verification of compounds

            # Attach main database for compound verification (read-only: it lives on a ro mount)
            conn.execute(f"ATTACH '{main_db_path}' AS main_db (READ_ONLY)")

            # Verify all compounds exist in database
            if not _verify_compounds_exist_in_db([comp.compound_uid for comp in atlas_obj.compound_mzrts.values()], conn):
                raise ValueError(f"Some compounds in atlas {atlas_obj.atlas_uid} don't exist in main database")

            # Create new atlas entry
            conn.execute("""
                INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atlas_obj.atlas_uid,
                atlas_obj.atlas_name,
                atlas_obj.atlas_description,
                atlas_obj.chromatography,
                atlas_obj.polarity,
                atlas_obj.analysis_type,
                atlas_obj.atlas_type,
                atlas_obj.source_atlas_uid,
                atlas_obj.rt_alignment_number,
                atlas_obj.analysis_number,
                prov["analyst"],
                prov["timestamp"],
                atlas_obj.source
            ))

            # Process each CompoundMZRT
            association_order = 0
            mzrts_created = 0
            mzrts_reused = 0
            for inchi_key, compound_mzrt in atlas_obj.compound_mzrts.items():
                # Check if this reference already exists in database
                existing_check = conn.execute("""
                    SELECT mz_rt_uid FROM compound_mzrt 
                    WHERE mz_rt_uid = ?
                """, [compound_mzrt.mz_rt_uid]).fetchone()
                
                mz_rt_uid = compound_mzrt.mz_rt_uid

                # Create new reference if it doesn't exist
                if not existing_check:
                    conn.execute("""
                        INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mz_rt_uid,
                        compound_mzrt.compound_uid,
                        compound_mzrt.compound_name,
                        compound_mzrt.inchi_key,
                        compound_mzrt.adduct,
                        compound_mzrt.rt_space,
                        compound_mzrt.rt_peak,
                        compound_mzrt.rt_min,
                        compound_mzrt.rt_max,
                        compound_mzrt.mz,
                        compound_mzrt.mz_tolerance,
                        compound_mzrt.chromatography,
                        compound_mzrt.polarity,
                        compound_mzrt.confidence,
                        compound_mzrt.source,
                        compound_mzrt.identification_notes,
                        prov["analyst"],
                        prov["timestamp"]
                    ))
                    mzrts_created += 1
                else:
                    mzrts_reused += 1
                
                # Create atlas-compound association
                assoc_uid = _generate_uid("association")
                conn.execute("""
                    INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    assoc_uid,
                    atlas_obj.atlas_uid,
                    compound_mzrt.compound_uid,
                    mz_rt_uid,
                    association_order,
                    prov["analyst"],
                    prov["timestamp"]
                ))
                
                association_order += 1
        
            conn.execute("DETACH main_db")

    logger.info(f"Saved atlas {atlas_obj.atlas_name} to database with UID {atlas_obj.atlas_uid}, chromatography {atlas_obj.chromatography}, polarity {atlas_obj.polarity}, and analysis type {atlas_obj.analysis_type}")
    logger.info(f"  References created: {mzrts_created}")
    logger.info(f"  References reused: {mzrts_reused}")
    logger.info(f"  Total associations: {association_order}")

def _verify_compounds_exist_in_db(compound_uids: list, conn: duckdb.DuckDBPyConnection) -> bool:
    """
    Verify that all compound_uids exist in the database at db_path or attached main_db.
    """
    if not compound_uids:
        logger.info("No compound_uids provided for verification; skipping check.")
        return True
    # Check if 'main_db' is attached
    attached_dbs = {row[1] for row in conn.execute("PRAGMA database_list").fetchall()}
    table_prefix = "main_db." if "main_db" in attached_dbs else ""
    placeholders = ','.join(['?'] * len(compound_uids))
    query = f"SELECT compound_uid FROM {table_prefix}compounds WHERE compound_uid IN ({placeholders})"
    existing = conn.execute(query, compound_uids).fetchall()
    existing_uids = {row[0] for row in existing}
    missing_uids = set(compound_uids) - existing_uids
    if missing_uids:
        for uid in missing_uids:
            logger.warning(f"Compound {uid} not found in main database")
        return False
    logger.info("All compounds verified to exist in the main database.")
    return True

def get_compound_uids_by_inchi_keys(db_path: str, inchi_keys: list) -> dict:
    """Return {inchi_key: compound_uid} for all inchi_keys found in the database."""
    if not inchi_keys:
        return {}
    with get_db_connection(db_path, read_only=True) as conn:
        placeholders = ','.join(['?'] * len(inchi_keys))
        rows = conn.execute(
            f"SELECT inchi_key, compound_uid FROM compounds WHERE inchi_key IN ({placeholders})",
            inchi_keys
        ).fetchall()
        if not rows:
            logger.error(f"No compounds found in database {db_path} for provided inchi_keys.")
            return {}
        return {row[0]: row[1] for row in rows}

def check_existing_auto_identification(auto_id_obj: "AutoIdentification") -> None:
    """
    Check for existing AutoIdentification results in the project database for the given
    RT alignment number and analysis number. If found, raise an error.
    """
    project_db_path = auto_id_obj.paths['project_db_path']
    rt_alignment_number = auto_id_obj.rt_alignment_number
    analysis_number = auto_id_obj.analysis_number

    if not Path(project_db_path).exists():
        raise FileNotFoundError(f"Project database not found: {project_db_path}")

    with get_db_connection(project_db_path, read_only=True) as conn:
        # Check MS1 and MS2 summary tables for existing entries

        ms1_data_exists = conn.execute(
            "SELECT COUNT(*) FROM ms1_data WHERE rt_alignment_number = ? AND analysis_number = ?",
            [rt_alignment_number, analysis_number]
        ).fetchone()[0] > 0

        ms2_data_exists = conn.execute(
            "SELECT COUNT(*) FROM ms2_data WHERE rt_alignment_number = ? AND analysis_number = ?",
            [rt_alignment_number, analysis_number]
        ).fetchone()[0] > 0

        manual_curation_exists = conn.execute(
            "SELECT COUNT(*) FROM manual_curation WHERE rt_alignment_number = ? AND analysis_number = ?",
            [rt_alignment_number, analysis_number]
        ).fetchone()[0] > 0

    if ms1_data_exists or ms2_data_exists or manual_curation_exists:
        raise ValueError(
            f"AutoIdentification results already exist for RT alignment number {rt_alignment_number} "
            f"and analysis number {analysis_number}. Please increment the analysis number or run a new RT alignment."
        )
    else:
        logger.info(f"No existing AutoIdentification results found for RT alignment number {rt_alignment_number} and analysis number {analysis_number}. Proceeding.")


def create_new_atlas_after_manual_curation(
    summary_obj: "AnalysisSummary",
) -> "Atlas":
    """
    Create a new Atlas object after manual curation.
    Applies manual RT adjustments and removes flagged compounds.
    """
    prov = get_provenance()
    source_atlas = summary_obj.post_autoid_atlas_obj

    # 1. Efficiently prepare atlas compounds for DB query
    # Instead of a full DataFrame, just provide the minimal required structure
    atlas_compounds = pd.DataFrame({"mz_rt_uid": list(source_atlas.compound_mzrts.keys())})
    
    curation_df = get_manual_curation_entries(
        summary_obj.paths['project_db_path'],
        summary_obj.rt_alignment_number,
        summary_obj.analysis_number,
        remove_unidentified_compounds=None,
        atlas_compounds=atlas_compounds,
        analysis_type=source_atlas.analysis_type,
    )

    # 2. Optimization: Use to_dict('records') instead of iterrows()
    # This is significantly faster for creating lookup maps
    curation_lookup = {row['mz_rt_uid']: row for row in curation_df.to_dict('records')}

    # Resolve overrides and workflow params once
    overrides = getattr(summary_obj, 'override_parameters', {}) or {}
    wp = summary_obj.workflow_params
    
    require_eval = overrides.get('gui_require_all_evaluated', wp.get('gui_require_all_evaluated', True))
    remove_flagged = overrides.get('remove_flagged_compounds', wp.get('remove_flagged_compounds', True))

    new_compound_mzrts = {}

    # 3. Main Processing Loop
    for mz_rt_uid, cmzrt in source_atlas.compound_mzrts.items():
        curation_row = curation_lookup.get(mz_rt_uid)
        
        if curation_row is not None:
            # Cache values to avoid repeated dict lookups
            ms1_notes = str(curation_row.get('ms1_notes', '')).strip()
            ms2_notes = str(curation_row.get('ms2_notes', '')).strip()
            
            # A. Requirement Validation (Evaluated flag)
            if require_eval:
                if not ms1_notes:
                    raise ValueError(
                        f"Compound {cmzrt.compound_uid} ({cmzrt.inchi_key} / {cmzrt.adduct}) "
                        "has empty ms1_notes. Please update in GUI."
                    )
                if not ms2_notes:
                    raise ValueError(
                        f"Compound {cmzrt.compound_uid} ({cmzrt.inchi_key} / {cmzrt.adduct}) "
                        "has empty ms2_notes. Please update in GUI."
                    )

            # B. Flagged Removal
            if remove_flagged and 'remove' in ms1_notes.lower():
                logger.info(f"Removing {cmzrt.compound_uid} ({cmzrt.compound_name}) based on 'remove' flag.")
                continue

            # C. Apply Updates
            # Shallow copy is sufficient as we only update immutable floats
            new_cmzrt = copy.copy(cmzrt)
            new_cmzrt.rt_peak = float(curation_row.get('rt_peak', cmzrt.rt_peak))
            new_cmzrt.rt_min  = float(curation_row.get('rt_min',  cmzrt.rt_min))
            new_cmzrt.rt_max  = float(curation_row.get('rt_max',  cmzrt.rt_max))
            new_compound_mzrts[mz_rt_uid] = new_cmzrt
            
        else:
            # No curation entry: keep original, but still copy to avoid shared references
            logger.warning(f"No curation entry for {mz_rt_uid}, keeping original values.")
            new_compound_mzrts[mz_rt_uid] = copy.copy(cmzrt)

    # 4. Atlas Generation
    new_atlas_uid = _generate_uid(
        "curated_atlas",
        decorator=f"{source_atlas.analysis_type.lower()}-{source_atlas.chromatography.lower()}-{source_atlas.polarity.lower()}"
    )

    new_atlas = copy.copy(source_atlas)
    new_atlas.atlas_uid = new_atlas_uid
    new_atlas.compound_mzrts = new_compound_mzrts
    new_atlas.source_atlas_uid = source_atlas.atlas_uid
    new_atlas.atlas_name = f"{source_atlas.atlas_name} (post-manual-curation)"
    new_atlas.atlas_description = f"{source_atlas.atlas_description} (post-manual-curation)"
    new_atlas.atlas_type = "MANUALLY_CURATED"
    new_atlas.created_by = prov["analyst"]
    new_atlas.created_date = prov["timestamp"]

    save_atlas_to_database(new_atlas, summary_obj.paths['project_db_path'], summary_obj.paths['main_db_path'])

    logger.info(f"Saved curated atlas {new_atlas_uid} with {len(new_compound_mzrts)} compounds.")
    return new_atlas

def create_new_atlas_after_auto_id(
    auto_id_obj: "AutoIdentification"
) -> "Atlas":
    """
    Create a new atlas after auto-identification.
    Surviving compounds are those present in the filtered manual_curation list.
    """
    prov = get_provenance()
    source_atlas = auto_id_obj.pre_autoid_atlas_obj

    # 1. Get the set of surviving UIDs
    surviving_uids = auto_id_obj.experimental_data.manual_curation_df['mz_rt_uid'].unique().tolist()

    # 2. Filter and copy surviving compounds using a dictionary comprehension.
    new_compound_mzrts = {
        uid: copy.copy(cmzrt)
        for uid, cmzrt in source_atlas.compound_mzrts.items() 
        if uid in surviving_uids
    }

    # 3. Generate a new atlas UID
    new_atlas_uid = _generate_uid(
        "autoid_atlas",
        decorator=(
            f"{source_atlas.analysis_type.lower()}-"
            f"{source_atlas.chromatography.lower()}-"
            f"{source_atlas.polarity.lower()}"
        )
    )

    # 4. Create the new atlas object
    new_atlas = copy.copy(source_atlas)
    new_atlas.atlas_uid = new_atlas_uid
    new_atlas.compound_mzrts = new_compound_mzrts
    new_atlas.source_atlas_uid = source_atlas.atlas_uid
    new_atlas.atlas_name = f"{source_atlas.atlas_name} (post-auto-identification)"
    new_atlas.atlas_description = f"{source_atlas.atlas_description} (post-auto-identification)"
    new_atlas.atlas_type = "AUTO_IDED"
    new_atlas.created_by = prov["analyst"]
    new_atlas.created_date = prov["timestamp"]

    # 5. Persistence
    save_atlas_to_database(new_atlas, auto_id_obj.paths['project_db_path'], auto_id_obj.paths['main_db_path'])

    logger.info(
        f"Created and saved post-auto-identification atlas {new_atlas_uid} "
        f"(from source {source_atlas.atlas_uid}) with {len(new_compound_mzrts)} compounds."
    )

    return new_atlas

def update_atlas_with_rt_alignment(
    rt_align_obj: "RTAlignment"
) -> tuple[dict[str, "Atlas"], list[float]]:
    """Apply RT alignment model to target atlases and return new Atlas objects with updated RTs, 
    along with a list of all RT shifts applied.
    """
    
    from metatlas2.workflow_objects import Atlas, CompoundMZRT

    prov = get_provenance()
    main_db_path = rt_align_obj.paths['main_db_path']
    targeted_analyses = rt_align_obj.config['WORKFLOWS']['TARGETED_ANALYSES']


    aligned_atlases = {}
    all_rt_shifts = []
    atlas_jobs = [
        (chrom, pol, analysis_type, atlas_params_dict)
        for chrom, pol_dict in targeted_analyses.items()
        for pol, analysis_dict in pol_dict.items()
        for analysis_type, atlas_params_dict in analysis_dict.items()
    ]

    for chrom, pol, analysis_type, atlas_params_dict in tqdm(atlas_jobs, desc="Updating atlases with RT alignment", disable=should_disable_tqdm()):
        target_atlas_uid = atlas_params_dict.get('ATLAS', {}).get('uid', None)
        if target_atlas_uid is None:
            logger.info(f"Skipping {chrom} {pol} {analysis_type} - no target atlas UID found in parameters")
            continue

        logger.info(f"Loading {chrom} {pol} {analysis_type} target atlas with UID {target_atlas_uid} for applying RT alignment model...")
        atlas_obj = Atlas.from_database(main_db_path, target_atlas_uid)

        # Create a new Atlas object for the RT-aligned version
        aligned_compound_mzrts = {}
        for mz_rt_uid, comp_ref in atlas_obj.compound_mzrts.items():
            # Apply RT alignment model
            aligned_rt_peak = float(rat.apply_rt_model([comp_ref.rt_peak], rt_align_obj.rt_alignment_model)[0])
            if rt_align_obj.rt_alignment_params['apply_model_to_min_max']:
                aligned_rt_min = float(rat.apply_rt_model([comp_ref.rt_min], rt_align_obj.rt_alignment_model)[0])
                aligned_rt_max = float(rat.apply_rt_model([comp_ref.rt_max], rt_align_obj.rt_alignment_model)[0])
            else:
                window = comp_ref.rt_max - comp_ref.rt_min
                aligned_rt_min = aligned_rt_peak - window / 2
                aligned_rt_max = aligned_rt_peak + window / 2
            rt_shift = aligned_rt_peak - comp_ref.rt_peak
            all_rt_shifts.append(rt_shift)

            # Create a new CompoundMZRT with updated RTs
            new_mz_rt_uid = _generate_uid("mz_rt", decorator="exp")
            comp_dict = {k: v for k, v in comp_ref.__dict__.items() if k not in ['mz_rt_uid', 'rt_peak', 'rt_min', 'rt_max']}
            aligned_comp_mzrt = CompoundMZRT(
                **comp_dict,
                mz_rt_uid=new_mz_rt_uid,
                rt_peak=aligned_rt_peak,
                rt_min=aligned_rt_min,
                rt_max=aligned_rt_max,
            )
            aligned_compound_mzrts[new_mz_rt_uid] = aligned_comp_mzrt

        # Generate new UID and name for the aligned atlas
        aligned_atlas_uid = _generate_uid("rt_atlas", decorator=f"{analysis_type.lower()}-{chrom.lower()}-{pol.lower()}")
        aligned_atlas = Atlas(
            atlas_uid=aligned_atlas_uid,
            atlas_name=f"{atlas_obj.atlas_name} (post-rt-alignment)",
            atlas_description=f"{atlas_obj.atlas_description} (post-rt-alignment)",
            chromatography=chrom,
            polarity=pol,
            analysis_type=analysis_type,
            atlas_type="RT-ALIGNED",
            source_atlas_uid=atlas_obj.atlas_uid,
            rt_alignment_number=rt_align_obj.rt_alignment_number,
            analysis_number=None,
            created_by=prov["analyst"],
            created_date=prov["timestamp"],
            source=atlas_obj.source,
            compound_mzrts=aligned_compound_mzrts
        )
        aligned_atlases[aligned_atlas_uid] = aligned_atlas

    return aligned_atlases, all_rt_shifts

def to_python_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val
 
def write_gui_updates_to_db(
    project_db_path: str,
    mz_rt_uid: str,
    rt_alignment_number: int,
    analysis_number: int,
    updated_fields: dict
) -> None:
    """
    Update a manual curation entry with new values for specified fields.
    
    Args:
        project_db_path: Path to the project database
        mz_rt_uid: mz_rt_uid of the manual curation entry to update
        rt_alignment_number: RT alignment number
        analysis_number: Analysis number
        updated_fields: Dictionary of fields to update with their new values
    """
    if not updated_fields:
        return

    try:
        mz_rt_uid = to_python_type(mz_rt_uid)
        rt_alignment_number = to_python_type(rt_alignment_number)
        analysis_number = to_python_type(analysis_number)
        set_clause = ", ".join([f"{k} = ?" for k in updated_fields])
        params = [to_python_type(v) for v in updated_fields.values()] + [mz_rt_uid, rt_alignment_number, analysis_number]
        with get_db_connection(project_db_path, max_retries=5, initial_retry_delay=0.5) as conn:
            conn.execute(
                f"UPDATE manual_curation SET {set_clause} WHERE mz_rt_uid = ? AND rt_alignment_number = ? AND analysis_number = ?",
                params
            )
    except Exception as e:
        logger.error(f"Error updating manual curation entry {mz_rt_uid}, RTA{rt_alignment_number}, TGA{analysis_number}: {e}")
        raise ValueError(f"Failed to update manual curation entry {mz_rt_uid}, RTA{rt_alignment_number}, TGA{analysis_number}. See logs for details.")

def display_auto_id_summary(auto_id_obj: "AutoIdentification") -> None:
    """
    Display a summary table of auto identification results to the logger.
    Shows number of compounds, MS1/MS2 datapoints, MS2 hits, etc.
    """

    summary_rows = []
    for ci in auto_id_obj.experimental_data.manual_curation_df:
        mz_rt_uid = ci.mz_rt_uid

        ms1_count = sum(
            len(ms1.data) for ms1 in auto_id_obj.experimental_data.ms1_data
            if ms1.mz_rt_uid == mz_rt_uid
        )
        ms2_count = sum(
            len(ms2.data) for ms2 in auto_id_obj.experimental_data.ms2_data
            if ms2.mz_rt_uid == mz_rt_uid
        )
        ms2_hits_count = sum(
            len(ms2_hit.data) for ms2_hit in auto_id_obj.experimental_data.ms2_hits
            if ms2_hit.mz_rt_uid == mz_rt_uid
        )

        summary_rows.append({
            "mz_rt_uid": mz_rt_uid,
            "compound_name": getattr(ci, 'compound_name', ''),
            "MS1 datapoints": ms1_count,
            "MS2 datapoints": ms2_count,
            "MS2 hits": ms2_hits_count,
        })

    return


########################################################
############## Saving ExperimentalData to the database
########################################################

def save_auto_identification_results_to_db(auto_id_obj):
    """
    Saves Tidy DataFrames directly to pre-defined DuckDB tables.
    """
    logger.info("Saving results to database...")
    project_db_path = auto_id_obj.paths['project_db_path']
    dataset = auto_id_obj.experimental_data
    curation_df = dataset.manual_curation_df
    ms1_df = dataset.ms1_df
    ms2_df = dataset.ms2_df
    prov = get_provenance()
    
    constants = {
        'rt_alignment_number': auto_id_obj.rt_alignment_number,
        'analysis_number': auto_id_obj.analysis_number,
        'analysis_type': auto_id_obj.pre_autoid_atlas_obj.analysis_type,
        'created_by': prov["analyst"],
        'created_date': prov["timestamp"]
    }

    # Add constants to all DataFrames
    for df in [ms1_df, ms2_df, curation_df]:
        if not df.empty:
            for k, v in constants.items():
                df[k] = v

    # Ensure hits are persisted as JSON strings for the VARCHAR ms2_data.hits column.
    logger.info("Serializing MS2 hits for database storage...")
    ms2_df_to_save = ms2_df
    if not ms2_df.empty and "hits" in ms2_df.columns:
        ms2_df_to_save = ms2_df.copy()
        ms2_df_to_save["hits"] = ms2_df_to_save["hits"].apply(_serialize_hits_value)

    # write ms1_df, ms2_df, and curation_df to a file on disk for debugging
    debug_dir = Path(project_db_path).parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    ms1_df.to_csv(debug_dir / f"ms1_data_{auto_id_obj.rt_alignment_number}_{auto_id_obj.analysis_number}.csv", index=False)
    ms2_df_to_save.to_csv(debug_dir / f"ms2_data_{auto_id_obj.rt_alignment_number}_{auto_id_obj.analysis_number}.csv", index=False)
    curation_df.to_csv(debug_dir / f"manual_curation_{auto_id_obj.rt_alignment_number}_{auto_id_obj.analysis_number}.csv", index=False)

    with get_db_connection(project_db_path) as conn:
        if not curation_df.empty:
            logger.info("Saving manual curation entries to database...")
            conn.execute("INSERT INTO manual_curation SELECT * FROM curation_df")
        
        if not ms1_df.empty:
            logger.info("Saving MS1 data entries to database...")
            conn.execute("INSERT INTO ms1_data SELECT * FROM ms1_df")
            
        if not ms2_df.empty:
            logger.info("Saving MS2 data entries to database...")
            conn.execute("INSERT INTO ms2_data SELECT * FROM ms2_df_to_save")

    logger.info("Database save complete.")

################################################################
############## Re-extracting ExperimentalData from the database
################################################################


def get_istd_curation_for_polarity(
    project_db_path: str,
    polarity: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> pd.DataFrame:
    """
    Retrieve manual_curation entries for the ISTD analysis matching polarity,
    rt_alignment_number, and analysis_number.

    Returns a DataFrame with columns: inchi_key, adduct, rt_peak, rt_min, rt_max,
    ms1_notes, ms2_notes, other_notes, analyst_notes.  Returns an empty DataFrame (no error) when
    no matching ISTD run is found.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute("""
                SELECT inchi_key, adduct, rt_peak, rt_min, rt_max,
                       ms1_notes, ms2_notes, other_notes, analyst_notes
                FROM manual_curation
                WHERE analysis_type = 'ISTD'
                  AND polarity = ?
                  AND rt_alignment_number = ?
                  AND analysis_number = ?
            """, [polarity, rt_alignment_number, analysis_number]).df()
    except Exception as e:
        logger.warning(f"Could not retrieve ISTD curation entries: {e}")
        df = pd.DataFrame()
    if df.empty:
        logger.info(f"No ISTD curation found for polarity={polarity}, RTA{rt_alignment_number}, TGA{analysis_number}.")
    else:
        logger.info(f"Found {len(df)} ISTD curation entries for polarity={polarity}, RTA{rt_alignment_number}, TGA{analysis_number}.")
    return df


def apply_istd_curation_to_ema(
    manual_curation_df: pd.DataFrame,
    istd_df: pd.DataFrame,
    post_autoid_atlas_obj,
) -> tuple:
    """
    Apply ISTD curation (RT bounds + notes) to matching EMA compounds.

    Only updates EMA rows where ms1_notes equals default_ms1_note (i.e. the row
    has not yet been manually curated).  Matching is on (inchi_key, adduct).

    Also updates RT bounds on the post_autoid_atlas_obj CompoundMZRT objects so
    that EIC plot windows reflect the curated values when the GUI opens.

    Returns (updated_manual_curation_df, n_transferred).
    """
    if istd_df.empty:
        return manual_curation_df, 0

    istd_lookup = {
        (row["inchi_key"], row["adduct"]): row
        for _, row in istd_df.iterrows()
    }

    atlas_lookup = {
        (cmzrt.inchi_key, cmzrt.adduct): cmzrt
        for cmzrt in post_autoid_atlas_obj.compound_mzrts.values()
    }

    df = manual_curation_df.copy()
    n_transferred = 0

    for idx, row in df.iterrows():
        pair = (row["inchi_key"], row["adduct"])
        if pair not in istd_lookup:
            continue
        
        ms2_note_val = row.get("ms2_notes", "")
        if ms2_note_val != "":
            logger.info(f"Skipping ISTD info transfer to EMA for {pair} because ms2_notes already has value: '{ms2_note_val}'")
            continue  # already curated — do not overwrite

        istd_row = istd_lookup[pair]
        df.at[idx, "rt_peak"]    = istd_row["rt_peak"]
        df.at[idx, "rt_min"]     = istd_row["rt_min"]
        df.at[idx, "rt_max"]     = istd_row["rt_max"]
        df.at[idx, "ms1_notes"]  = istd_row["ms1_notes"]
        df.at[idx, "ms2_notes"]  = istd_row["ms2_notes"]
        df.at[idx, "analyst_notes"] = istd_row["analyst_notes"]
        df.at[idx, "other_notes"] = istd_row["other_notes"]

        if pair in atlas_lookup:
            atlas_lookup[pair].rt_peak = float(istd_row["rt_peak"])
            atlas_lookup[pair].rt_min  = float(istd_row["rt_min"])
            atlas_lookup[pair].rt_max  = float(istd_row["rt_max"])

        n_transferred += 1

    return df, n_transferred

def _transfer_istd_curation(ema_df, db_path, gui_obj, atlas_obj):
    """Vectorized ISTD -> EMA transfer."""
    with get_db_connection(db_path, read_only=True) as conn:
        istd_df = conn.execute("""
            SELECT inchi_key, adduct, rt_peak, rt_min, rt_max, 
                   ms1_notes, ms2_notes, other_notes, analyst_notes
            FROM manual_curation 
            WHERE analysis_type = 'ISTD' AND polarity = ? AND rt_alignment_number = ? AND analysis_number = ?
        """, [atlas_obj.polarity, gui_obj.rt_alignment_number, gui_obj.analysis_number]).df()

    if istd_df.empty: return ema_df

    # Merge ISTD values onto EMA based on (inchi_key, adduct)
    # Only update if EMA notes are empty (not yet curated)
    merged = ema_df.merge(istd_df, on=['inchi_key', 'adduct'], how='left', suffixes=('', '_istd'))
    
    for col in ['rt_peak', 'rt_min', 'rt_max', 'ms1_notes', 'ms2_notes', 'analyst_notes', 'other_notes']:
        istd_col = f"{col}_istd"
        if istd_col in merged.columns:
            # Update EMA value only if original is empty/default
            merged[col] = np.where(
                (merged[col] == '') | (merged[col] == 0.0), 
                merged[istd_col], 
                merged[col]
            )
    
    # Drop the temporary _istd columns
    return merged[[c for c in merged.columns if not c.endswith('_istd')]]

def validate_override_parameters(override_parameters):
    if not isinstance(override_parameters, dict):
        raise ValueError("override_parameters must be a dict")
    if not isinstance(override_parameters["gui_lcmsruns_colors"], (type(None), dict)):
        raise ValueError("override_parameters['gui_lcmsruns_colors'] must be a dict mapping LCMS run identifiers to color strings or None")
    if not isinstance(override_parameters["gui_require_all_evaluated"], (type(None), bool)):
        raise ValueError("override_parameters['gui_require_all_evaluated'] must be a boolean or None")
    if not isinstance(override_parameters["ms1_min_peak_intensity"], (type(None), (int, float))):
        raise ValueError("override_parameters['ms1_min_peak_intensity'] must be a number or None")
    if not isinstance(override_parameters["ms1_min_num_points"], (type(None), int)):
        raise ValueError("override_parameters['ms1_min_num_points'] must be an integer or None")
    if not isinstance(override_parameters["ms2_min_score"], (type(None), (int, float))):
        raise ValueError("override_parameters['ms2_min_score'] must be a number or None")
    if not isinstance(override_parameters["ms2_min_matching_frags"], (type(None), int)):
        raise ValueError("override_parameters['ms2_min_matching_frags'] must be an integer or None")
    if not isinstance(override_parameters.get("remove_unided_compounds"), (type(None), bool)):
        raise ValueError("override_parameters['remove_unided_compounds'] must be a boolean or None")
    if not isinstance(override_parameters.get("apply_istd_to_ema"), (type(None), bool)):
        raise ValueError("override_parameters['apply_istd_to_ema'] must be a boolean or None")
    if not isinstance(override_parameters.get("remove_flagged_compounds"), (type(None), bool)):
        raise ValueError("override_parameters['remove_flagged_compounds'] must be a boolean or None")
    if not isinstance(override_parameters.get("gui_top_n_hits"), (type(None), int)):
        raise ValueError("override_parameters['gui_top_n_hits'] must be an integer or None")
    if not isinstance(override_parameters["note_options_overrides"], (type(None), dict)):
        raise ValueError("override_parameters['note_options_overrides'] must be a dict mapping note types to option dicts or None")
    if isinstance(override_parameters["note_options_overrides"], dict):
        for note_type, options in override_parameters["note_options_overrides"].items():
            if note_type not in ["ms1_notes", "ms2_notes", "other_notes"]:
                raise ValueError(f"Invalid note type in note_options_overrides: {note_type}")
            if not isinstance(options, dict):
                raise ValueError(f"Options for {note_type} in note_options_overrides must be a dict mapping option text to hotkeys")
            for opt_text, hotkey in options.items():
                if not isinstance(opt_text, str) or not isinstance(hotkey, str):
                    raise ValueError(f"Invalid option in note_options_overrides for {note_type}: {opt_text}: {hotkey}")

def get_manual_curation_entries(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    remove_unidentified_compounds: bool = True,
    atlas_compounds: pd.DataFrame = None,
    analysis_type: str = None,
) -> pd.DataFrame:
    """
    Get all manual_curation entries for the given RT alignment and analysis number.

    If ``atlas_compounds`` (a DataFrame with 'inchi_key' and 'adduct' columns) is
    provided the query is restricted to those pairs and the result is returned in
    atlas order.  Otherwise rows are ordered by ascending RT peak with unlabeled
    compounds placed after their non-unlabeled counterparts.
    
    If ``analysis_type`` is provided, only records matching that analysis_type are returned.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            query = """
                SELECT *
                FROM manual_curation
                WHERE rt_alignment_number = ? AND analysis_number = ?
            """
            params = [rt_alignment_number, analysis_number]
            if analysis_type is not None:
                query += " AND analysis_type = ?"
                params.append(analysis_type)
            if remove_unidentified_compounds:
                query += " AND auto_ided = TRUE"
            if atlas_compounds is not None and not atlas_compounds.empty:
                mz_rt_uids = list(atlas_compounds["mz_rt_uid"].drop_duplicates())
                placeholders = ", ".join(["?"] * len(mz_rt_uids))
                query += f" AND mz_rt_uid IN ({placeholders})"
                params.extend(mz_rt_uids)
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving manual curation entries: {e}")
        df = pd.DataFrame()

    if df.empty:
        raise ValueError(f"No manual curation entries found for RT alignment number {rt_alignment_number} and analysis number {analysis_number}.")

    df['compound_name'] = df['compound_name'].fillna('')

    if atlas_compounds is not None and not atlas_compounds.empty:
        _order = (
            atlas_compounds[["mz_rt_uid"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(_atlas_order=lambda d: d.index)
        )
        df = (
            df.merge(_order, on=["mz_rt_uid"], how="left")
            .sort_values("_atlas_order")
            .drop(columns=["_atlas_order"])
            .reset_index(drop=True)
        )
    else:
        df['has_unlabeled'] = df['compound_name'].str.contains('unlabeled', case=False, na=False).astype(int)
        df = (
            df.sort_values(by=['rt_peak', 'has_unlabeled'], ascending=[True, True])
            .drop('has_unlabeled', axis=1)
            .reset_index(drop=True)
        )

    return df

def get_ms1_data_for_compound(
    project_db_path: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    atlas_compounds: pd.DataFrame = None,
    analysis_type: str = None,
) -> pd.DataFrame:
    """
    Get all MS1 data for a compound (inchi_key+adduct) for plotting EIC.
    If inchi_key or adduct is None, do not filter on that field.
    If atlas_compounds (DataFrame with 'inchi_key' and 'adduct' cols) is provided,
    only rows matching those pairs are returned.
    If analysis_type is provided, only rows matching that analysis_type are returned.
    """
    query = """
        SELECT *
        FROM ms1_data
        WHERE 1=1
    """
    try:
        params = []
        if rt_alignment_number is not None:
            query += " AND rt_alignment_number = ?"
            params.append(rt_alignment_number)
        if analysis_number is not None:
            query += " AND analysis_number = ?"
            params.append(analysis_number)
        if analysis_type is not None:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        if atlas_compounds is not None and not atlas_compounds.empty:
            mz_rt_uids = list(atlas_compounds["mz_rt_uid"].drop_duplicates())
            placeholders = ", ".join(["?"] * len(mz_rt_uids))
            query += f" AND mz_rt_uid IN ({placeholders})"
            params.extend(mz_rt_uids)
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving MS1 data: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No MS1 data found for RT alignment number {rt_alignment_number}, and analysis number {analysis_number}.")

    return df

def get_ms2_data_for_compound(
    project_db_path: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    atlas_compounds: pd.DataFrame = None,
    analysis_type: str = None,
) -> pd.DataFrame:
    """
    Get all MS2 data for a compound (inchi_key+adduct) for plotting query spectrum if no hits.
    If inchi_key or adduct is None, do not filter on that field.
    If atlas_compounds (DataFrame with 'inchi_key' and 'adduct' cols) is provided,
    only rows matching those pairs are returned.
    If analysis_type is provided, only rows matching that analysis_type are returned.
    """
    query = """
        SELECT *
        FROM ms2_data
        WHERE 1=1
    """
    try:
        params = []
        if rt_alignment_number is not None:
            query += " AND rt_alignment_number = ?"
            params.append(rt_alignment_number)
        if analysis_number is not None:
            query += " AND analysis_number = ?"
            params.append(analysis_number)
        if analysis_type is not None:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        if atlas_compounds is not None and not atlas_compounds.empty:
            mz_rt_uids = list(atlas_compounds["mz_rt_uid"].drop_duplicates())
            placeholders = ", ".join(["?"] * len(mz_rt_uids))
            query += f" AND mz_rt_uid IN ({placeholders})"
            params.extend(mz_rt_uids)
        query += " ORDER BY file_path, rt"
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving MS2 data: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No MS2 data found for RT alignment number {rt_alignment_number}, and analysis number {analysis_number}.")

    return df

def _deserialize_hits_value(hits_value):
    """Convert strict JSON hits values into Python hit objects."""
    if hits_value is None:
        return []
    if isinstance(hits_value, float) and np.isnan(hits_value):
        return []
    if isinstance(hits_value, list):
        return hits_value
    if isinstance(hits_value, tuple):
        return list(hits_value)
    if isinstance(hits_value, dict):
        return [hits_value]
    if isinstance(hits_value, str):
        text = hits_value.strip()
        if not text or text.lower() in {"none", "null"}:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return []
    return []


def _to_json_safe(value):
    """Recursively convert objects to strict-JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _to_json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_to_json_safe(v) for v in value.tolist()]
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    return value


def _serialize_hits_value(hits_value):
    """Serialize hit objects for storage in VARCHAR columns."""
    normalized = _to_json_safe(_deserialize_hits_value(hits_value))
    try:
        return json.dumps(normalized, allow_nan=False)
    except Exception:
        logger.warning("Failed to serialize hits value. Saving empty list instead.")
        return "[]"


def _get_max_score(hits_data):
    """Helper to find max score from serialized or in-memory hits."""
    hits_list = _deserialize_hits_value(hits_data)
    if not hits_list:
        return -1.0
    return max((float(h.get('score', -1.0)) for h in hits_list if isinstance(h, dict)), default=-1.0)


def _get_max_frags(hits_data):
    """Helper to find max fragment count from serialized or in-memory hits."""
    hits_list = _deserialize_hits_value(hits_data)
    if not hits_list:
        return -1
    return max((int(h.get('num_matches', -1)) for h in hits_list if isinstance(h, dict)), default=-1)

def load_and_filter_for_gui(analysis_gui_obj):
    """
    THE NEW ENTRY POINT FOR THE GUI.
    Replaces load_experimental_data_from_db and filter_experimental_data_in_memory.
    """
    from metatlas2.workflow_objects import ExperimentalData

    with get_db_connection(analysis_gui_obj.paths["project_db_path"], read_only=True) as conn:
        atlas_uids = list(analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts.keys())
        uid_filter = f"mz_rt_uid IN {tuple(atlas_uids)}" if atlas_uids else "1=0"
        
        curation_df = conn.execute(f"""
            SELECT * FROM manual_curation 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()
        
        ms1_df = conn.execute(f"""
            SELECT * FROM ms1_data 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()
        
        ms2_df = conn.execute(f"""
            SELECT * FROM ms2_data 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()

    if not ms2_df.empty and "hits" in ms2_df.columns:
        ms2_df = ms2_df.copy()
        ms2_df["hits"] = ms2_df["hits"].apply(_deserialize_hits_value)

    logger.info(f"Loaded {len(curation_df)} manual curation entries, {len(ms1_df)} MS1 rows, and {len(ms2_df)} MS2 rows for GUI display.")

    # check if the user has provided any override parameters, and if so, validate them
    _overrides = analysis_gui_obj.override_parameters or {}
    wp = analysis_gui_obj.workflow_params
    ms1_min_pts = _overrides.get("ms1_min_num_points")
    ms1_min_int = _overrides.get("ms1_min_peak_intensity")
    ms2_min_score = _overrides.get("ms2_min_score")
    ms2_min_frags = _overrides.get("ms2_min_matching_frags")

    # Determine if any MS1/MS2 filter is set
    ms1_filter = ms1_min_pts is not None or ms1_min_int is not None
    ms2_filter = ms2_min_score is not None or ms2_min_frags is not None

    # MS1 filtering
    if not ms1_df.empty and ms1_filter:
        max_ints = ms1_df['spec_ints'].apply(lambda x: np.max(x) if (isinstance(x, list) and len(x)>0) else 0)
        num_pts = ms1_df['spec_ints'].apply(lambda x: len(x) if isinstance(x, list) else 0)
        mask = pd.Series([True] * len(ms1_df))
        if ms1_min_pts is not None:
            mask &= num_pts >= ms1_min_pts
        if ms1_min_int is not None:
            mask &= max_ints >= ms1_min_int
        passing_ms1 = set(ms1_df.loc[mask, "mz_rt_uid"])
    else:
        passing_ms1 = set(ms1_df["mz_rt_uid"])

    # MS2 filtering
    if not ms2_df.empty and ms2_filter:
        max_scores = ms2_df['hits'].apply(_get_max_score)
        max_frags = ms2_df['hits'].apply(_get_max_frags)
        mask = pd.Series([True] * len(ms2_df))
        if ms2_min_score is not None:
            mask &= max_scores >= ms2_min_score
        if ms2_min_frags is not None:
            mask &= max_frags >= ms2_min_frags
        passing_ms2 = set(ms2_df.loc[mask, "mz_rt_uid"])
    else:
        passing_ms2 = set(ms2_df["mz_rt_uid"])

    # check whether to remove ids with no MS1 or MS2 data from curation df
    remove_unided = analysis_gui_obj.workflow_params.get("remove_unided_compounds", True)
    if _overrides.get("remove_unided_compounds") is not None:
        remove_unided = _overrides.get("remove_unided_compounds", None)
    if remove_unided:
        keep_mask = (curation_df['mz_rt_uid'].isin(passing_ms1) | curation_df['mz_rt_uid'].isin(passing_ms2))
        curation_df = curation_df[keep_mask].copy()
        surviving_uids = set(curation_df['mz_rt_uid'])
        ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(surviving_uids)]
        ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(surviving_uids)]    

    # ISTD transfer
    apply_istd_override = analysis_gui_obj.workflow_params.get("apply_istd_to_ema", True)
    if _overrides.get("apply_istd_to_ema") is not None:
        apply_istd_override = _overrides.get("apply_istd_to_ema", None)
    if apply_istd_override:
        if analysis_gui_obj.post_autoid_atlas_obj.analysis_type == "EMA":
            curation_df = _transfer_istd_curation(
                curation_df, analysis_gui_obj.paths["project_db_path"],
                analysis_gui_obj, analysis_gui_obj.post_autoid_atlas_obj
            )

    # Assign to GUI
    analysis_gui_obj.experimental_data = ExperimentalData()
    analysis_gui_obj.experimental_data.curation_df = curation_df
    analysis_gui_obj.experimental_data.ms1_df = ms1_df
    analysis_gui_obj.experimental_data.ms2_df = ms2_df
    return