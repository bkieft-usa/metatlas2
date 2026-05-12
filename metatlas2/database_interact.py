import pandas as pd
import numpy as np
import duckdb
import uuid
import re
import os
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
import metatlas2.pubchem_retrieval as pcr
import metatlas2.load_tools as ldt
import metatlas2.logging_config as lcf
logger = lcf.get_logger('database_interact')

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

# def get_all_atlases_for_autoid(
#     auto_id_obj: "AutoIdentification",
# ) -> List[str]:
#     """
#     Given a project database and an RT alignment number, return all atlases that match the RT alignment number, 
#     and return a list of Atlas uids
#     """

#     project_db_path = auto_id_obj.paths['project_db_path']
#     rt_alignment_number = auto_id_obj.rt_alignment_number
#     analysis_subset = auto_id_obj.analysis_subset
#     atlases = []

#     # Try to get RT-aligned atlases from database first
#     chromatography = getattr(auto_id_obj, 'chromatography', None)
#     with get_db_connection(project_db_path) as conn:
#         if analysis_subset:
#             # Build query with polarity and analysis_type filters
#             where_clauses = ["rt_alignment_number = ?", "atlas_type = 'RT-ALIGNED'"]
#             params = [rt_alignment_number]
#             if chromatography:
#                 where_clauses.append("chromatography = ?")
#                 params.append(chromatography)
#             pol_analysis_pairs = [tuple(subset.split('-')) for subset in analysis_subset]
#             pair_clauses = []
#             for pol, analysis_type in pol_analysis_pairs:
#                 pair_clauses.append("(polarity = ? AND analysis_type = ?)")
#                 params.extend([pol, analysis_type])
#             where_clauses.append("(" + " OR ".join(pair_clauses) + ")")
#             query = f"""
#                 SELECT atlas_uid, chromatography, polarity, analysis_type
#                 FROM atlases
#                 WHERE {' AND '.join(where_clauses)}
#             """
#             results = conn.execute(query, params).fetchall()
#             atlases = [row[0] for row in results]
#         else: # get all atlases that can be found for that rt align number
#             where_clauses = ["rt_alignment_number = ?", "atlas_type = 'RT-ALIGNED'"]
#             params = [rt_alignment_number]
#             if chromatography:
#                 where_clauses.append("chromatography = ?")
#                 params.append(chromatography)
#             query = f"""
#                 SELECT atlas_uid, chromatography, polarity, analysis_type
#                 FROM atlases
#                 WHERE {' AND '.join(where_clauses)}
#             """
#             results = conn.execute(query, params).fetchall()
#             atlases = [row[0] for row in results]

#     if atlases:
#         logger.info(f"Retrieved {len(atlases)} RT-aligned atlases for RT alignment number {rt_alignment_number} from database at {project_db_path}")
#         return atlases

#     # Fallback: Use atlas UIDs from config file attribute (e.g., analysis.yaml)
#     logger.warning(f"No RT-aligned atlases found for RT alignment number {rt_alignment_number} in database at {project_db_path}. Falling back to Atlas UIDs in config.")
#     config = auto_id_obj.config
#     atlas_entries = []
#     # Traverse config to collect all atlas UIDs
#     workflows = config.get('WORKFLOWS', {})
#     targeted = workflows.get('TARGETED_ANALYSES', {})
#     for chrom, chrom_dict in targeted.items():
#         for pol, pol_dict in chrom_dict.items():
#             for analysis_type, analysis_dict in pol_dict.items():
#                 atlas_uid = analysis_dict.get('ATLAS', {}).get('uid', None)
#                 if atlas_uid:
#                     atlas_entries.append({
#                         'uid': atlas_uid,
#                         'chrom': chrom,
#                         'pol': pol,
#                         'analysis_type': analysis_type
#                     })
#     # Filter if analysis_subset is provided
#     if analysis_subset:
#         analysis_filters = [tuple(subset.split('-')) for subset in analysis_subset]
#         for entry in atlas_entries:
#             if (entry['pol'], entry['analysis_type']) in analysis_filters:
#                 atlases.append(entry['uid'])
#     else:
#         atlases = [entry['uid'] for entry in atlas_entries]
#     logger.info(f"Retrieved {len(atlases)} atlas UIDs from config file.")
#     return atlases

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

    required_cols = ['inchi_key', 'adduct', 'rt_peak', 'mz']
    missing_cols = [col for col in required_cols if col not in atlas_df.columns]
    if missing_cols:
        raise ValueError(f"Atlas input dataframe is missing required columns: {missing_cols}. Found columns: {atlas_df.columns.tolist()}")

    inchi_keys = atlas_df['inchi_key'].dropna().unique().tolist()
    compound_lookup = get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)

    compound_mzrts = {}
    for _, row in atlas_df.iterrows():
        # main key
        inchi_key = row.get('inchi_key', '')
        adduct = str(row.get('adduct', None))
        if not inchi_key or inchi_key not in compound_lookup:
            logger.warning(f"Compound with inchi_key {inchi_key} missing from metatlas database, skipping.")
            continue
        # required fields
        compound_uid = compound_lookup[inchi_key]
        compound_name = str(row.get('compound_name', row.get('label', 'Unknown Compound')))
        rt_peak = row.get('rt_peak', None)
        rt_min = row.get('rt_min', rt_peak - 0.5)
        rt_max = row.get('rt_max', rt_peak + 0.5)
        mz = row.get('mz', None)
        mz_tolerance = row.get('mz_tolerance', 5.0)
        rt_space = row.get('rt_space', 'HF')
        if rt_peak is None or mz is None or adduct is None:
            raise ValueError(f"Compound {inchi_key} missing essential data (rt_peak: {rt_peak}, mz: {mz}, adduct: {adduct}), cannot create reference.")
        confidence_level = row.get('confidence_level', None)
        identification_notes = row.get('identification_notes', '')

        mz_rt_uid, _ = get_or_create_compound_mz_rt_uid(
            main_db_path,
            compound_uid,
            chromatography,
            polarity,
            adduct,
            rt_peak,
            mz,
            mz_tolerance,
            decorator="ref"
        )
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
        compound_mzrts[inchi_key] = compound_mzrt

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

def add_compounds_to_db(input_df: pd.DataFrame, db_path: str, pubchem_cache_path: str, input_file_path: str = ""):
    """Add compounds and RT/MZ references to database using batch operations."""

    if not os.path.exists(db_path):
        logger.error(f"Database not found at {db_path}. Check path or create it first!")
        raise FileNotFoundError(f"Database not found at {db_path}")

    unique_inchi_keys = input_df['inchi_key'].dropna().unique()
    logger.info(f"Adding {len(unique_inchi_keys)} compounds to database: {db_path}")
    
    pubchem_cache = pcr.load_or_create_pubchem_cache(pubchem_cache_path)
    prov = get_provenance()

    with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
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
        for idx, row in input_df.iterrows():
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
        mzrts_created = 0
        if compound_records:
            logger.info(f"Batch inserting {len(compound_records)} compounds...")
            conn.executemany("""
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, compound_records)

        # Check for existing references and batch insert if necessary
        if reference_records:
            logger.info(f"Checking for existing references and batch inserting if necessary...")
            # Get existing references to check for duplicates
            existing_refs = set()
            existing_ref_result = conn.execute("""
                SELECT compound_uid, chromatography, polarity, adduct 
                FROM compound_mzrt
            """).fetchall()
            existing_refs = {(row[0], row[1], row[2], row[3]) for row in existing_ref_result}

            # Filter out duplicates
            filtered_reference_records = []
            mzrts_skipped = 0
            
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
                    mzrts_created += 1
                else:
                    mzrts_skipped += 1

            # Batch insert filtered references
            if filtered_reference_records:
                conn.executemany("""
                    INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, filtered_reference_records)
        
        # Get final counts
        compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        mzrts_count = conn.execute("SELECT COUNT(*) FROM compound_mzrt").fetchone()[0]
    
    logger.info("Compounds added successfully!")
    logger.info(f"   Total compounds in database: {compounds_count}")
    logger.info(f"   New compounds created: {compounds_created}")
    if compounds_skipped > 0:
        logger.info(f"   Compounds skipped (already existed): {compounds_skipped}")
    logger.info(f"   Total RT/MZ references in database: {mzrts_count}")
    logger.info(f"   New RT/MZ references created: {mzrts_created}")
    if 'mzrts_skipped' in locals() and mzrts_skipped > 0:
        logger.info(f"   RT/MZ references skipped (duplicates): {mzrts_skipped}")
    
    return

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
    compound_classes = str(row.get('compound_classes', '')) if pd.notna(row.get('compound_classes')) else None
    compound_pathways = str(row.get('compound_pathways', '')) if pd.notna(row.get('compound_pathways')) else None
    compound_tags = str(row.get('compound_tags', '')) if pd.notna(row.get('compound_tags')) else None

    return (
        compound_uid,
        compound_name,
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

def save_lcmsruns_to_db(
    project_db_path: str,
    project_name: str,
    lcmsruns_list: List[Dict],
    overwrite_existing: bool = False
) -> int:
    """
    Save LCMS run files to project database from a flat list of run dicts.
    If the lcmsruns table exists and has rows, do not overwrite unless overwrite_existing is True.
    Returns the number of files saved.
    """
    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        prov = get_provenance()

        # Check if lcmsruns table exists and has rows
        table_exists = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'lcmsruns'"
        ).fetchone()[0] > 0

        has_rows = False
        if table_exists:
            has_rows = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0] > 0
            if has_rows:
                if not overwrite_existing:
                    logger.info("LCMS runs already exist in the database and overwrite_existing is False. Skipping save.")
                    return 0
                else:
                    logger.info(f"Overwriting existing LCMS runs in the database for {project_name}.")
                    conn.execute("DELETE FROM lcmsruns")
        else:
            logger.info(f"No LCMS runs detected in project database for {project_name}. Creating a new table...")

        total_files = 0
        for run in lcmsruns_list:
            conn.execute(
                "INSERT INTO lcmsruns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run["file_path"],
                    run["filename"],
                    run["file_format"],
                    run["file_type"],
                    run["chromatography"],
                    run["ms_level"],
                    run["polarity"],
                    prov["analyst"],
                    prov["timestamp"],
                ),
            )
            total_files += 1

    logger.info(f"Saved {total_files} LCMS runs to database.")
    return total_files

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

def get_lcmsruns_from_db(
    project_db_path: str,
    file_types: Optional[List[str]] = None,
    file_format: str = "parquet",
    chromatography: str = None,
    polarity: str = None
) -> List["LCMSRun"]:
    """
    Retrieve LCMS runs from the database, optionally filtering by file_types, file_format, chromatography, and polarity.
    Returns a flat list of LCMSRun objects.
    """

    from metatlas2.workflow_objects import LCMSRun

    where_conditions = ["file_format = ?"]
    params = [file_format]

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

    if file_types:
        for ft in file_types:
            if ft not in ['qc', 'experimental', 'istd', 'exctrl']:
                raise ValueError(f"Invalid file category: {ft}. Must be one of ['qc', 'experimental', 'istd', 'exctrl']")
        where_conditions.append(f"file_type IN ({','.join(['?']*len(file_types))})")
        params.extend(file_types)

    where_clause = " AND ".join(where_conditions)
    with get_db_connection(project_db_path, read_only=True) as conn:
        files_df = conn.execute(f"""
            SELECT file_path, filename, file_format, file_type, chromatography, ms_level, polarity, created_by, created_date
            FROM lcmsruns
            WHERE {where_clause}
            ORDER BY chromatography, ms_level, polarity, file_path
        """, params).df()

    lcmsruns_list = [LCMSRun(**row) for _, row in files_df.iterrows()]

    logger.info(f"Retrieved {len(lcmsruns_list)} LC/MS run files from database with filters: file_types={file_types}, chromatography={chromatography}, polarity={polarity}, format={file_format}")
    return lcmsruns_list

def batch_save_compounds_and_mzrts(
    db_path: str,
    compounds: List["Compound"],
    compound_mzrts: List["CompoundMZRT"]
) -> Tuple[int, int]:
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
        Tuple of (compounds_created, mzrts_created)
    """

    compounds_data = [compound.to_dict() for compound in compounds]
    mzrts_data = [compound_mzrt.to_dict() for compound_mzrt in compound_mzrts]

    compounds_created = 0
    mzrts_created = 0
    mzrts_skipped_identical = 0
    compounds_skipped_existing = 0
    
    with get_db_connection(db_path, max_retries=10, initial_retry_delay=0.5) as conn:
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
        if mzrts_data:
            logger.info(f"Processing {len(mzrts_data)} reference entries...")
            reference_records = []
            
            for reference_data in mzrts_data:
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
                
                # Check if the same reference already exists
                identical_ref_exists = _check_identical_reference_exists(conn, reference_with_uid)
                
                if identical_ref_exists:
                    mzrts_skipped_identical += 1
                    logger.debug(f"Identical reference already exists for compound {compound_uid}")
                    continue
                
                # Create new reference entry
                reference_record = _prepare_reference_record_from_dict(reference_with_uid)
                if reference_record:
                    reference_records.append(reference_record)
                    mzrts_created += 1
                    logger.debug(f"New reference prepared for compound {compound_uid}")
            
            # Step 5: Batch insert new references
            if reference_records:
                logger.info(f"Creating {len(reference_records)} new references...")
                conn.executemany("""
                    INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, reference_records)
    
    # Log summary
    logger.info("Batch save completed:")
    logger.info(f"  Compounds created: {compounds_created}")
    logger.info(f"  Compounds skipped (already exist): {compounds_skipped_existing}")
    logger.info(f"  References created: {mzrts_created}")
    logger.info(f"  References skipped (identical data): {mzrts_skipped_identical}")
    
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
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compounds (
                compound_uid TEXT PRIMARY KEY,
                compound_name TEXT,
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
                ms1_data_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                inchi_key TEXT,
                adduct TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type TEXT,
                file_path TEXT,
                mz TEXT,
                raw_spectrum TEXT,
                created_by TEXT,
                created_date TEXT,
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_data (
                ms2_data_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                inchi_key TEXT,
                adduct TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type TEXT,
                file_path TEXT,
                rt REAL,
                raw_spectrum TEXT,
                precursor_MZ REAL,
                precursor_intensity REAL,
                collision_energy REAL,
                created_by TEXT,
                created_date TEXT,
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ms2_hits (
                ms2_hit_uid TEXT PRIMARY KEY,
                compound_uid TEXT,
                inchi_key TEXT,
                adduct TEXT,
                rt_alignment_number INTEGER,
                analysis_number INTEGER,
                analysis_type TEXT,
                file_path TEXT,
                database TEXT,
                ref_id TEXT,
                ref_name TEXT,
                score REAL,
                num_matches INTEGER,
                mz_theoretical REAL,
                mz_measured REAL,
                ppm_error REAL,
                rt REAL,
                qry_intensity_peak REAL,
                ref_frags INTEGER,
                data_frags INTEGER,
                matched_fragments TEXT,
                aligned_fragment_colors TEXT,
                qry_spectrum TEXT,
                ref_spectrum TEXT,
                created_by TEXT,
                created_date TEXT,
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_curation (
            curation_uid TEXT PRIMARY KEY,
            compound_uid TEXT,
            inchi_key TEXT,
            adduct TEXT,
            rt_alignment_number INTEGER,
            analysis_number INTEGER,
            compound_name TEXT,
            auto_ided BOOLEAN,
            polarity TEXT,
            chromatography TEXT,
            analysis_type TEXT,
            mz_tolerance REAL,
            atlas_mz REAL,
            atlas_rt_peak REAL,
            atlas_rt_min REAL,
            atlas_rt_max REAL,
            rt_peak REAL,
            rt_min REAL,
            rt_max REAL,
            ms1_notes TEXT,
            ms2_notes TEXT,
            other_notes TEXT,
            identification_notes TEXT,
            analyst_notes TEXT,
            best_ms1_file TEXT,
            best_ms1_rt REAL,
            best_ms1_mz REAL,
            best_ms1_intensity REAL,
            best_ms1_ppm_error REAL,
            best_ms1_rt_error REAL,
            isomers TEXT,
            suggested_rt_min REAL,
            suggested_rt_max REAL,
            suggested_rt_peak REAL,
            rt_suggestion_confidence REAL,
            created_by TEXT,
            created_date TEXT
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

def get_or_create_compound_mz_rt_uid(
    db_path: str,
    compound_uid: str,
    chromatography: str,
    polarity: str,
    adduct: str,
    rt_peak: float,
    mz: float,
    mz_tolerance: float,
    decorator: str = None
) -> tuple[str, bool]:
    """
    Return (mz_rt_uid, reused_flag). If not found, generate a new UID (do not insert).
    """
    with get_db_connection(db_path, read_only=True) as conn:
        existing = conn.execute("""
            SELECT mz_rt_uid FROM compound_mzrt
            WHERE compound_uid = ?
            AND chromatography = ? AND polarity = ? AND adduct = ?
            AND rt_peak = ? AND mz = ?
            AND mz_tolerance = ?
        """, [
            compound_uid, chromatography, polarity, adduct,
            rt_peak, mz, mz_tolerance
        ]).fetchone()
        if existing:
            return existing[0], True
        else:
            return _generate_uid("mz_rt", decorator), False

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


def load_and_filter_gui_inputs(
        analysis_gui_obj,
        override_parameters: dict = None
):
    """
    Load GUI inputs from database and optionally apply second-stage filtering.
    
    In the two-stage funnel architecture:
    1. Auto-ID applies first-stage filters and saves to database
    2. GUI can apply stricter second-stage filters via override_parameters
    
    If override_parameters are provided and differ from the original workflow_params,
    a second round of filtering is applied to narrow the funnel further. This allows
    analysts to iteratively refine their data based on visual inspection in the GUI.
    
    Second-stage filtering is applied in-memory only and does not modify the database.
    To permanently save these stricter filters, the analyst should update the config
    and run a new analysis with incremented analysis_number.
    """
    _overrides = override_parameters or {}
    
    apply_istd_to_ema = (
        _overrides.get("apply_istd_to_ema")
        if _overrides.get("apply_istd_to_ema") is not None
        else analysis_gui_obj.workflow_params.get("apply_istd_to_ema", True)
    )

    # Create atlas_compounds DataFrame to scope queries to this specific atlas
    # This prevents loading compounds from other polarities/analyses with the same analysis_type
    atlas_compounds = pd.DataFrame([
        {"inchi_key": cmzrt.inchi_key, "adduct": cmzrt.adduct}
        for cmzrt in analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts.values()
    ])

    # Load pre-filtered data from database (first-stage filtering already applied)
    exp_data = load_experimental_data_from_db(
        project_db_path=analysis_gui_obj.paths["project_db_path"],
        rt_alignment_number=analysis_gui_obj.rt_alignment_number,
        analysis_number=analysis_gui_obj.analysis_number,
        atlas_compounds=atlas_compounds,
        analysis_type=analysis_gui_obj.post_autoid_atlas_obj.analysis_type,
        remove_unidentified_compounds=False,  # Already filtered during auto-ID
    )
    
    # Apply second-stage filtering if override parameters differ from workflow params
    if override_parameters:
        # Check if any override parameters actually differ
        params_differ = False
        override_values = {}
        
        for param in ["ms1_min_peak_intensity", "ms1_min_num_points", 
                      "ms2_min_score", "ms2_min_matching_frags", "remove_unided_compounds"]:
            override_val = _overrides.get(param)
            workflow_val = analysis_gui_obj.workflow_params.get(param)
            if override_val is not None and override_val != workflow_val:
                params_differ = True
                override_values[param] = override_val
        
        if params_differ:
            logger.info(
                "Override parameters differ from saved workflow parameters. "
                "Applying second-stage filtering (in-memory only). "
                f"Modified parameters: {override_values}"
            )
            
            # Apply second-stage filter with override parameters
            exp_data, n_removed = filter_experimental_data(
                exp_data,
                ms1_min_pts=_overrides.get("ms1_min_num_points", 
                                          analysis_gui_obj.workflow_params.get("ms1_min_num_points")),
                ms1_min_int=_overrides.get("ms1_min_peak_intensity",
                                          analysis_gui_obj.workflow_params.get("ms1_min_peak_intensity")),
                ms2_min_score=_overrides.get("ms2_min_score",
                                            analysis_gui_obj.workflow_params.get("ms2_min_score")),
                ms2_min_frags=_overrides.get("ms2_min_matching_frags",
                                            analysis_gui_obj.workflow_params.get("ms2_min_matching_frags")),
                remove_unided=_overrides.get("remove_unided_compounds",
                                            analysis_gui_obj.workflow_params.get("remove_unided_compounds", True)),
            )
            
            if n_removed > 0:
                logger.info(
                    f"Second-stage filtering removed {n_removed} additional compound(s). "
                    f"GUI will display {len(exp_data.manual_curation)} compounds."
                )
            
            # Trim the atlas to match the filtered data
            surviving_pairs = {(mc.inchi_key, mc.adduct) for mc in exp_data.manual_curation}
            keys_to_remove = [
                k for k, cmzrt in analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts.items()
                if (cmzrt.inchi_key, cmzrt.adduct) not in surviving_pairs
            ]
            for k in keys_to_remove:
                del analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts[k]
            
            if keys_to_remove:
                logger.info(
                    f"Trimmed post-auto-ID atlas to {len(analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts)} compounds "
                    f"({len(keys_to_remove)} removed by second-stage filtering)."
                )
    
    # Convert to flat DataFrames for GUI
    manual_curation_df, ms1_df, ms2_df, ms2_hits_df = experimental_data_to_dataframes(exp_data)

    if analysis_gui_obj.post_autoid_atlas_obj.analysis_type == "EMA":
        if apply_istd_to_ema:
            logger.info("EMA analysis detected — checking for matching ISTD curation to pre-populate...")
            istd_df = get_istd_curation_for_polarity(
                project_db_path=analysis_gui_obj.paths["project_db_path"],
                polarity=analysis_gui_obj.post_autoid_atlas_obj.polarity,
                rt_alignment_number=analysis_gui_obj.rt_alignment_number,
                analysis_number=analysis_gui_obj.analysis_number,
            )
            if not istd_df.empty:
                manual_curation_df, n_transferred = apply_istd_curation_to_ema(
                    manual_curation_df=manual_curation_df,
                    istd_df=istd_df,
                    post_autoid_atlas_obj=analysis_gui_obj.post_autoid_atlas_obj,
                )
                logger.info(f"Transferred ISTD curation to {n_transferred} EMA compounds (inchi_key+adduct matches).")
            else:
                logger.info("No matching ISTD curation found — loading EMA data without pre-population.")

    analysis_gui_obj.experimental_data = exp_data
    analysis_gui_obj.manual_curation_df = manual_curation_df
    analysis_gui_obj.ms1_df = ms1_df
    analysis_gui_obj.ms2_df = ms2_df
    analysis_gui_obj.ms2_hits_df = ms2_hits_df
    analysis_gui_obj.override_parameters = override_parameters
    return

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
                pairs = list(
                    atlas_compounds[["inchi_key", "adduct"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                placeholders = ", ".join(["(?, ?)"] * len(pairs))
                query += f" AND (inchi_key, adduct) IN (VALUES {placeholders})"
                for ik, ad in pairs:
                    params.extend([ik, ad])
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving manual curation entries: {e}")
        df = pd.DataFrame()

    if df.empty:
        raise ValueError(f"No manual curation entries found for RT alignment number {rt_alignment_number} and analysis number {analysis_number}.")

    df['compound_name'] = df['compound_name'].fillna('')

    if atlas_compounds is not None and not atlas_compounds.empty:
        _order = (
            atlas_compounds[["inchi_key", "adduct"]]
            .drop_duplicates()
            .reset_index(drop=True)
            .assign(_atlas_order=lambda d: d.index)
        )
        df = (
            df.merge(_order, on=["inchi_key", "adduct"], how="left")
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
    inchi_key: str = None,
    adduct: str = None,
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
        if inchi_key is not None:
            query += " AND inchi_key = ?"
            params.append(inchi_key)
        if adduct is not None:
            query += " AND adduct = ?"
            params.append(adduct)
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
            pairs = list(
                atlas_compounds[["inchi_key", "adduct"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            placeholders = ", ".join(["(?, ?)"] * len(pairs))
            query += f" AND (inchi_key, adduct) IN (VALUES {placeholders})"
            for ik, ad in pairs:
                params.extend([ik, ad])
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving MS1 data: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No MS1 data found for inchi_key {inchi_key}, adduct {adduct}, RT alignment number {rt_alignment_number}, and analysis number {analysis_number}.")

    return df

def get_ms2_hits_for_compound(
    project_db_path: str,
    inchi_key: str = None,
    adduct: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    atlas_compounds: pd.DataFrame = None,
    analysis_type: str = None,
) -> pd.DataFrame:
    """
    Get MS2 hits for a compound (inchi_key+adduct), ordered by score descending.
    If inchi_key or adduct is None, do not filter on that field.
    If atlas_compounds (DataFrame with 'inchi_key' and 'adduct' cols) is provided,
    only rows matching those pairs are returned.
    If analysis_type is provided, only rows matching that analysis_type are returned.
    """
    query = """
        SELECT *
        FROM ms2_hits
        WHERE 1=1
    """
    try:
        params = []
        if inchi_key is not None:
            query += " AND inchi_key = ?"
            params.append(inchi_key)
        if adduct is not None:
            query += " AND adduct = ?"
            params.append(adduct)
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
            pairs = list(
                atlas_compounds[["inchi_key", "adduct"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            placeholders = ", ".join(["(?, ?)"] * len(pairs))
            query += f" AND (inchi_key, adduct) IN (VALUES {placeholders})"
            for ik, ad in pairs:
                params.extend([ik, ad])
        query += " ORDER BY score DESC"
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving MS2 hits: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No MS2 hits found for inchi_key {inchi_key}, adduct {adduct}, RT alignment number {rt_alignment_number}, and analysis number {analysis_number}.")
    else:
        # Ensure critical fields have default values if missing/NULL
        default_fields = {
            'mz_theoretical': 0.0,
            'mz_measured': 0.0,
            'ppm_error': 0.0,
            'rt': 0.0,
            'score': 0.0,
            'precursor_MZ': 0.0
        }
        for field, default_val in default_fields.items():
            if field in df.columns:
                df[field] = df[field].fillna(default_val)
            else:
                df[field] = default_val

    return df

def get_ms2_data_for_compound(
    project_db_path: str,
    inchi_key: str = None,
    adduct: str = None,
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
        if inchi_key is not None:
            query += " AND inchi_key = ?"
            params.append(inchi_key)
        if adduct is not None:
            query += " AND adduct = ?"
            params.append(adduct)
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
            pairs = list(
                atlas_compounds[["inchi_key", "adduct"]]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )
            placeholders = ", ".join(["(?, ?)"] * len(pairs))
            query += f" AND (inchi_key, adduct) IN (VALUES {placeholders})"
            for ik, ad in pairs:
                params.extend([ik, ad])
        query += " ORDER BY file_path, rt"
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute(query, params).df()
    except Exception as e:
        logger.error(f"Error retrieving MS2 data: {e}")
        df = pd.DataFrame()

    if df.empty:
        logger.warning(f"No MS2 data found for inchi_key {inchi_key}, adduct {adduct}, RT alignment number {rt_alignment_number}, and analysis number {analysis_number}.")

    return df


def load_experimental_data_from_db(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    atlas_compounds: pd.DataFrame = None,
    analysis_type: str = None,
    remove_unidentified_compounds: bool = False,
) -> "ExperimentalData":
    """Reconstruct an ExperimentalData object from the four project DB analysis tables.

    Loads ``manual_curation``, ``ms1_data``, ``ms2_data``, and ``ms2_hits`` and
    wraps each row (or group of rows) in the corresponding workflow object.  The
    per-row DataFrames stored on each object preserve every DB column—including
    serialised ``raw_spectrum`` strings—so that
    :func:`experimental_data_to_dataframes` can reconstruct the original flat
    DataFrames without any additional JSON re-serialisation.

    Parameters
    ----------
    atlas_compounds:
        Optional DataFrame with ``inchi_key`` and ``adduct`` columns.  When
        provided the queries are restricted to those (inchi_key, adduct) pairs.
    remove_unidentified_compounds:
        Passed through to :func:`get_manual_curation_entries`.
    """
    from metatlas2.workflow_objects import ExperimentalData, ManualCuration, MS1Data, MS2Data, MS2Hit

    mc_flat = get_manual_curation_entries(
        project_db_path, rt_alignment_number, analysis_number,
        remove_unidentified_compounds=remove_unidentified_compounds,
        atlas_compounds=atlas_compounds,
        analysis_type=analysis_type,
    )
    ms1_flat = get_ms1_data_for_compound(
        project_db_path, None, None, rt_alignment_number, analysis_number,
        atlas_compounds=atlas_compounds, analysis_type=analysis_type,
    )
    ms2_flat = get_ms2_data_for_compound(
        project_db_path, None, None, rt_alignment_number, analysis_number,
        atlas_compounds=atlas_compounds, analysis_type=analysis_type,
    )
    ms2_hits_flat = get_ms2_hits_for_compound(
        project_db_path, None, None, rt_alignment_number, analysis_number,
        atlas_compounds=atlas_compounds, analysis_type=analysis_type,
    )

    exp_data = ExperimentalData()

    # ManualCuration — one object per compound
    for _, row in mc_flat.iterrows():
        exp_data.manual_curation.append(
            ManualCuration(
                inchi_key=row["inchi_key"],
                adduct=row["adduct"],
                data=pd.DataFrame([row.to_dict()]),
            )
        )

    # MS1Data — one object per (compound, file) DB row
    for _, row in ms1_flat.iterrows():
        exp_data.ms1_data.append(
            MS1Data(
                inchi_key=row["inchi_key"],
                adduct=row["adduct"],
                filename=row["file_path"],
                data=pd.DataFrame([row.to_dict()]),
            )
        )

    # MS2Data — one object per (compound, file) group of scan rows
    if not ms2_flat.empty:
        for (ik, ad, fp), grp in ms2_flat.groupby(["inchi_key", "adduct", "file_path"], sort=False):
            exp_data.ms2_data.append(
                MS2Data(
                    inchi_key=ik,
                    adduct=ad,
                    filename=fp,
                    data=grp.reset_index(drop=True),
                )
            )

    # MS2Hit — one object per (compound, file) group of hit rows
    if not ms2_hits_flat.empty:
        for (ik, ad, fp), grp in ms2_hits_flat.groupby(["inchi_key", "adduct", "file_path"], sort=False):
            exp_data.ms2_hits.append(
                MS2Hit(
                    inchi_key=ik,
                    adduct=ad,
                    filename=fp,
                    data=grp.reset_index(drop=True),
                )
            )

    logger.info(
        "load_experimental_data_from_db: loaded %d compounds, %d MS1 file-groups, "
        "%d MS2 file-groups, %d MS2 hit file-groups.",
        len(exp_data.manual_curation),
        len(exp_data.ms1_data),
        len(exp_data.ms2_data),
        len(exp_data.ms2_hits),
    )
    return exp_data


def filter_experimental_data(
    exp_data: "ExperimentalData",
    ms1_min_pts: int = None,
    ms1_min_int: float = None,
    ms2_min_score: float = None,
    ms2_min_frags: int = None,
    remove_unided: bool = False,
    remove_flagged: bool = False,
) -> tuple:
    """Filter an ExperimentalData object by quality thresholds and return a filtered copy.

    This is the core filtering function used in the two-stage funnel architecture:
    
    - **Stage 1 (Auto-ID)**: Called by :func:`apply_auto_id_filters` to permanently
      filter data before saving to the database.
    
    - **Stage 2 (GUI)**: Called by :func:`load_and_filter_gui_inputs` to apply
      optional stricter thresholds via override_parameters for in-memory filtering.
    
    A compound is *removed* when any of the following conditions holds:

    * ``remove_unided`` is True and the compound's ``auto_ided`` flag is False.
    * ``remove_flagged`` is True and ``ms1_notes`` is ``'remove'``.
    * ``ms1_min_pts`` is set and no MS1 file for the compound has >=
      ``ms1_min_pts`` data points in its raw EIC.
    * ``ms1_min_int`` is set and ``best_ms1_intensity`` stored in the
      ManualCuration row (or the raw spectrum max when not available) is below
      the threshold.
    * ``ms2_min_score`` or ``ms2_min_frags`` is set and no MS2 hit for the
      compound passes both thresholds.

    Within kept MS2Hit objects, individual hit rows that do not satisfy the
    score / fragment thresholds are pruned so downstream code sees only passing
    hits.

    Parameters
    ----------
    exp_data:
        Source ExperimentalData object.  Not modified in place.

    Returns
    -------
    tuple of (filtered_ExperimentalData, n_compounds_removed)
    """
    from metatlas2.workflow_objects import ExperimentalData

    # ── Step 1: passing pairs by MS2 thresholds ─────────────────────────────
    passing_ms2: Optional[set] = None
    # Only enforce MS2 filtering if thresholds are meaningful (> 0).
    # A threshold of 0 means "no filtering", not "filter with threshold of 0".
    ms2_score_active = ms2_min_score is not None and ms2_min_score > 0
    ms2_frags_active = ms2_min_frags is not None and ms2_min_frags > 0
    
    if ms2_score_active or ms2_frags_active:
        passing_ms2 = set()
        n_ms2_hit_objs = len(exp_data.ms2_hits)
        for hit_obj in exp_data.ms2_hits:
            hits = hit_obj.data
            if hits is None or hits.empty:
                continue
            if ms2_score_active and "score" in hits.columns:
                hits = hits[hits["score"] >= ms2_min_score]
            if ms2_frags_active and "num_matches" in hits.columns:
                hits = hits[hits["num_matches"] >= ms2_min_frags]
            if not hits.empty:
                passing_ms2.add((hit_obj.inchi_key, hit_obj.adduct))
        logger.info(
            f"MS2 filtering: {len(passing_ms2)} compounds passed thresholds "
            f"(ms2_min_score={ms2_min_score}, ms2_min_frags={ms2_min_frags}) "
            f"out of {n_ms2_hit_objs} compounds with MS2 hits"
        )

    # ── Step 2: passing pairs by MS1 thresholds ──────────────────────────────
    passing_ms1: Optional[set] = None
    if ms1_min_pts is not None or ms1_min_int is not None:
        mc_best_int = {
            (mc.inchi_key, mc.adduct): (
                float(mc.data.iloc[0].get("best_ms1_intensity", 0.0) or 0.0)
            )
            for mc in exp_data.manual_curation
        }
        max_pts: Dict[tuple, int] = {}
        fallback_int: Dict[tuple, float] = {}
        for ms1_obj in exp_data.ms1_data:
            key = (ms1_obj.inchi_key, ms1_obj.adduct)
            raw = ms1_obj.data
            if raw is None or raw.empty:
                continue
            if "raw_spectrum" in raw.columns:
                # DB-loaded format: raw_spectrum = JSON "[rt_list, i_list]"
                try:
                    rt_arr, i_arr = json.loads(raw.iloc[0]["raw_spectrum"])
                    if ms1_min_pts is not None:
                        max_pts[key] = max(max_pts.get(key, 0), len(rt_arr))
                    if ms1_min_int is not None and key not in mc_best_int:
                        clean_i = [0.0 if (isinstance(v, float) and np.isnan(v)) else v for v in i_arr]
                        fallback_int[key] = max(fallback_int.get(key, 0.0), max(clean_i) if clean_i else 0.0)
                except Exception:
                    pass
            elif "i" in raw.columns:
                # AutoID format: per-timepoint DataFrame
                if ms1_min_pts is not None:
                    max_pts[key] = max(max_pts.get(key, 0), len(raw))
                if ms1_min_int is not None and key not in mc_best_int:
                    fallback_int[key] = max(fallback_int.get(key, 0.0), float(raw["i"].max()) if len(raw) > 0 else 0.0)

        all_pairs = {(mc.inchi_key, mc.adduct) for mc in exp_data.manual_curation}
        passing_ms1 = set()
        ms1_failure_details = {}
        for pair in all_pairs:
            failure_reasons = []
            if ms1_min_pts is not None:
                actual_pts = max_pts.get(pair, 0)
                if actual_pts < ms1_min_pts:
                    failure_reasons.append(f"num_points={actual_pts} < {ms1_min_pts}")
            if ms1_min_int is not None:
                eff_int = mc_best_int.get(pair, fallback_int.get(pair, 0.0))
                if eff_int < ms1_min_int:
                    failure_reasons.append(f"intensity={eff_int:.1f} < {ms1_min_int:.1f}")
            if failure_reasons:
                ms1_failure_details[pair] = "; ".join(failure_reasons)
            else:
                passing_ms1.add(pair)

    # ── Step 3: filter ManualCuration list ───────────────────────────────────
    n_total = len(exp_data.manual_curation)
    kept_pairs: set = set()
    new_mc = []
    removed_by_unided = 0
    removed_by_flagged = 0
    removed_by_ms1 = 0
    removed_by_ms2 = 0
    removed_compounds_details = []
    for mc in exp_data.manual_curation:
        pair = (mc.inchi_key, mc.adduct)
        row0 = mc.data.iloc[0]
        compound_name = row0.get("compound_name", "unknown")
        if remove_unided and not bool(row0.get("auto_ided", True)):
            removed_by_unided += 1
            removed_compounds_details.append(f"{compound_name} ({mc.inchi_key}/{mc.adduct}): not auto-identified")
            continue
        if remove_flagged and "remove" in str(row0.get("ms1_notes", "")).strip().lower():
            removed_by_flagged += 1
            removed_compounds_details.append(f"{compound_name} ({mc.inchi_key}/{mc.adduct}): flagged for removal")
            continue
        if passing_ms1 is not None and pair not in passing_ms1:
            removed_by_ms1 += 1
            reason = ms1_failure_details.get(pair, "unknown MS1 reason")
            removed_compounds_details.append(f"{compound_name} ({mc.inchi_key}/{mc.adduct}): failed MS1 - {reason}")
            continue
        if passing_ms2 is not None and pair not in passing_ms2:
            removed_by_ms2 += 1
            removed_compounds_details.append(f"{compound_name} ({mc.inchi_key}/{mc.adduct}): failed MS2 thresholds")
            continue
        new_mc.append(mc)
        kept_pairs.add(pair)
    n_removed = n_total - len(new_mc)
    
    # Log filtering breakdown
    if n_removed > 0:
        logger.info(
            f"Filtering breakdown: {n_removed} total removed - "
            f"unidentified: {removed_by_unided}, "
            f"flagged: {removed_by_flagged}, "
            f"MS1 thresholds: {removed_by_ms1}, "
            f"MS2 thresholds: {removed_by_ms2}"
        )
        # Log details of each removed compound
        for detail in removed_compounds_details:
            logger.debug(f"  Removed: {detail}")

    # ── Step 4: propagate to other lists ─────────────────────────────────────
    new_ms1 = [obj for obj in exp_data.ms1_data if (obj.inchi_key, obj.adduct) in kept_pairs]
    new_ms2 = [obj for obj in exp_data.ms2_data if (obj.inchi_key, obj.adduct) in kept_pairs]
    new_hits = []
    for hit_obj in exp_data.ms2_hits:
        if (hit_obj.inchi_key, hit_obj.adduct) not in kept_pairs:
            continue
        # Only filter individual hit rows if thresholds are meaningful (> 0)
        if ms2_score_active or ms2_frags_active:
            hits = hit_obj.data.copy()
            if hits.empty:
                continue
            if ms2_score_active and "score" in hits.columns:
                hits = hits[hits["score"] >= ms2_min_score]
            if ms2_frags_active and "num_matches" in hits.columns:
                hits = hits[hits["num_matches"] >= ms2_min_frags]
            hits = hits.reset_index(drop=True)
            if not hits.empty:
                new_obj = copy.copy(hit_obj)
                new_obj.data = hits
                new_hits.append(new_obj)
        else:
            new_hits.append(hit_obj)

    filtered = ExperimentalData(
        manual_curation=new_mc,
        ms1_data=new_ms1,
        ms2_data=new_ms2,
        ms2_hits=new_hits,
    )
    return filtered, n_removed


def experimental_data_to_dataframes(
    exp_data: "ExperimentalData",
) -> tuple:
    """Convert a (filtered) ExperimentalData object to four flat DataFrames.

    This is the inverse of :func:`load_experimental_data_from_db`.  Each
    attribute list is concatenated back into the same column format that the
    individual ``get_*`` DB query helpers return, so existing GUI and summary
    code that consumes those DataFrames continues to work without modification.

    Returns
    -------
    tuple of (manual_curation_df, ms1_df, ms2_df, ms2_hits_df)
    """
    mc_df = (
        pd.concat([mc.data for mc in exp_data.manual_curation], ignore_index=True)
        if exp_data.manual_curation
        else pd.DataFrame()
    )
    ms1_df = (
        pd.concat([obj.data for obj in exp_data.ms1_data], ignore_index=True)
        if exp_data.ms1_data
        else pd.DataFrame()
    )
    ms2_df = (
        pd.concat([obj.data for obj in exp_data.ms2_data], ignore_index=True)
        if exp_data.ms2_data
        else pd.DataFrame()
    )
    ms2_hits_df = (
        pd.concat([obj.data for obj in exp_data.ms2_hits], ignore_index=True)
        if exp_data.ms2_hits
        else pd.DataFrame()
    )
    return mc_df, ms1_df, ms2_df, ms2_hits_df


def apply_auto_id_filters(
    auto_id_obj: "AutoIdentification"
) -> "ExperimentalData":
    """
    Apply first-stage quality filters to ExperimentalData before saving to database.
    
    This is the first stage in the two-stage funnel architecture:
    
    Stage 1 (Auto-ID): Applies configured quality thresholds and permanently saves
                       filtered data to the database.
    
    Stage 2 (GUI): Optionally applies stricter thresholds via override_parameters
                   for in-memory filtering during interactive analysis.
    
    Only data that passes Stage 1 filters is persisted to the database. Stage 2
    filtering (if used) is non-destructive and can be adjusted interactively in
    the GUI without requiring a new analysis run. To permanently apply stricter
    filters, update the config and increment analysis_number for a new Stage 1 run.
    
    Parameters
    ----------
    auto_id_obj : AutoIdentification
        The AutoIdentification object containing experimental_data to filter
        and workflow_params with filter thresholds.
    
    Returns
    -------
    ExperimentalData
        The filtered ExperimentalData object containing only compounds that
        passed all Stage 1 quality thresholds. This is what will be saved to
        the database.
    """
    source_atlas = auto_id_obj.pre_autoid_atlas_obj
    
    ms2_min_score = auto_id_obj.workflow_params.get('ms2_min_score')
    ms2_min_frags = auto_id_obj.workflow_params.get('ms2_min_matching_frags')
    ms1_min_pts   = auto_id_obj.workflow_params.get('ms1_min_num_points')
    ms1_min_int   = auto_id_obj.workflow_params.get('ms1_min_peak_intensity')
    remove_unided = auto_id_obj.workflow_params.get('remove_unided_compounds', True)

    # Log filtering parameters for debugging
    logger.info(
        f"Applying quality filters for {source_atlas.analysis_type}: "
        f"ms2_min_score={ms2_min_score}, ms2_min_frags={ms2_min_frags}, "
        f"ms1_min_pts={ms1_min_pts}, ms1_min_int={ms1_min_int}, "
        f"remove_unided={remove_unided}"
    )
    
    # Warn if MS2 filtering is disabled for non-QC/non-ISTD analyses
    if source_atlas.analysis_type not in ["QC", "ISTD"]:
        if ms2_min_score is None or ms2_min_score == 0:
            logger.warning(
                f"MS2 score filtering is disabled for {source_atlas.analysis_type} analysis "
                f"(ms2_min_score={ms2_min_score}). Compounds without MS2 hits will NOT be filtered out."
            )
        if ms2_min_frags is None or ms2_min_frags == 0:
            logger.warning(
                f"MS2 fragment filtering is disabled for {source_atlas.analysis_type} analysis "
                f"(ms2_min_frags={ms2_min_frags}). Compounds without sufficient MS2 fragments will NOT be filtered out."
            )

    filtered_exp, n_removed = filter_experimental_data(
        auto_id_obj.experimental_data,
        ms1_min_pts=ms1_min_pts,
        ms1_min_int=ms1_min_int,
        ms2_min_score=ms2_min_score,
        ms2_min_frags=ms2_min_frags,
        remove_unided=remove_unided,
    )
    
    if n_removed > 0:
        logger.info(
            f"Quality filters removed {n_removed} compound(s). "
            f"Only {len(filtered_exp.manual_curation)} compounds will be saved to database."
        )
    else:
        logger.info(
            f"All {len(filtered_exp.manual_curation)} compounds passed quality filters."
        )
    
    return filtered_exp


def create_new_atlas_after_manual_curation(
    summary_obj: "AnalysisSummary",
) -> "Atlas":
    """
    Create a new Atlas object after manual curation.
    
    In the funnel architecture, this works with data that has already been
    filtered during auto-ID and saved to the database. It applies manual
    curation updates (RT adjustments, removal of flagged compounds) and
    creates a new curated atlas.
    """
    prov = get_provenance()
    source_atlas = summary_obj.post_autoid_atlas_obj

    # Load curation entries from database for compounds in the post-auto-ID atlas
    atlas_compounds = pd.DataFrame([
        {"inchi_key": cmzrt.inchi_key, "adduct": cmzrt.adduct}
        for cmzrt in source_atlas.compound_mzrts.values()
    ])
    curation_df = get_manual_curation_entries(
        summary_obj.paths['project_db_path'],
        summary_obj.rt_alignment_number,
        summary_obj.analysis_number,
        remove_unidentified_compounds=None,
        atlas_compounds=atlas_compounds,
        analysis_type=source_atlas.analysis_type,
    )
    # Index by (inchi_key, adduct) for O(1) lookup
    curation_lookup = {
        (row['inchi_key'], row['adduct']): row
        for _, row in curation_df.iterrows()
    }

    # Resolve override vs. workflow_params once before iterating
    _overrides = getattr(summary_obj, 'override_parameters', {}) or {}
    _require_eval = (
        _overrides['gui_require_all_evaluated']
        if _overrides.get('gui_require_all_evaluated') is not None
        else summary_obj.workflow_params.get('gui_require_all_evaluated', True)
    )
    _remove_flagged = (
        _overrides['remove_flagged_compounds']
        if _overrides.get('remove_flagged_compounds') is not None
        else summary_obj.workflow_params.get('remove_flagged_compounds', True)
    )

    # Deep-copy every CompoundMZRT and apply curation updates
    new_compound_mzrts = {}
    for dict_key, cmzrt in source_atlas.compound_mzrts.items():
        new_cmzrt = copy.deepcopy(cmzrt)
        curation_row = curation_lookup.get((cmzrt.inchi_key, cmzrt.adduct))
        # Check if curation_row has ms2_notes still as '' and error out and print message to address it
        if _require_eval and curation_row is not None and str(curation_row.get('ms2_notes', '')) == '':
            raise ValueError(
                f"Compound {cmzrt.compound_uid} ({cmzrt.inchi_key} / {cmzrt.adduct}) has ms2_notes as '' in manual curation. "
                "Please update ms2_notes in manual curation GUI before creating the post-curation atlas."
            )
        if _require_eval and curation_row is not None and str(curation_row.get('ms1_notes', '')) == '':
            raise ValueError(
                f"Compound {cmzrt.compound_uid} ({cmzrt.inchi_key} / {cmzrt.adduct}) has ms1_notes as '' in manual curation. "
                "Please update ms1_notes in manual curation GUI before creating the post-curation atlas."
            )
        # Remove compounds from original atlas if the ms1 note has 'remove' in it
        if _remove_flagged and curation_row is not None and 'remove' in str(curation_row.get('ms1_notes', '')).lower():
            logger.info(f"Removing compound {cmzrt.compound_uid} ({cmzrt.compound_name} / {cmzrt.inchi_key} / {cmzrt.adduct}) from atlas because ms1_notes contains 'remove' and 'remove_flagged_compounds' is set to True in config.")
            continue
        if curation_row is not None:
            new_cmzrt.rt_peak = float(curation_row.get('rt_peak', cmzrt.rt_peak))
            new_cmzrt.rt_min  = float(curation_row.get('rt_min',  cmzrt.rt_min))
            new_cmzrt.rt_max  = float(curation_row.get('rt_max',  cmzrt.rt_max))
        else:
            logger.warning(
                f"No manual curation entry found for {cmzrt.inchi_key} / {cmzrt.adduct}, "
                "keeping original RT values."
            )
        new_compound_mzrts[dict_key] = new_cmzrt

    # Generate a new atlas UID
    new_atlas_uid = _generate_uid(
        "curated_atlas",
        decorator=(
            f"{source_atlas.analysis_type.lower()}-"
            f"{source_atlas.chromatography.lower()}-"
            f"{source_atlas.polarity.lower()}"
        )
    )

    # Shallow-copy the atlas, then replace the fields that must change
    new_atlas = copy.copy(source_atlas)
    new_atlas.atlas_uid        = new_atlas_uid
    new_atlas.compound_mzrts   = new_compound_mzrts
    new_atlas.source_atlas_uid = source_atlas.atlas_uid
    new_atlas.atlas_name = source_atlas.atlas_name + " (post-manual-curation)"
    new_atlas.atlas_description = source_atlas.atlas_description + " (post-manual-curation)"
    new_atlas.atlas_type = "MANUALLY_CURATED"
    new_atlas.created_by = prov["analyst"]
    new_atlas.created_date = prov["timestamp"]
    save_atlas_to_database(new_atlas, summary_obj.paths['project_db_path'], summary_obj.paths['main_db_path'])

    logger.info(
        f"Created and saved post-curation atlas {new_atlas_uid} (from source atlas {source_atlas.atlas_uid}) "
        f"with ({len(new_compound_mzrts)} compounds)."
    )

    return new_atlas

def create_new_atlas_after_auto_id(
    auto_id_obj: "AutoIdentification"
) -> "Atlas":
    """
    Create a new atlas after auto-identification.

    In the funnel architecture, filtering has already been applied to
    auto_id_obj.experimental_data before saving to the database. This function
    simply creates the new Atlas object from the already-filtered data.
    """
    prov = get_provenance()
    source_atlas = auto_id_obj.pre_autoid_atlas_obj
    
    # experimental_data has already been filtered by apply_auto_id_filters()
    survived_pairs = {(mc.inchi_key, mc.adduct) for mc in auto_id_obj.experimental_data.manual_curation}

    # Deep-copy every surviving CompoundMZRT into the new atlas
    new_compound_mzrts = {}
    for dict_key, cmzrt in tqdm(source_atlas.compound_mzrts.items(), desc="Creating new Atlas from curated compounds"):
        if (cmzrt.inchi_key, cmzrt.adduct) not in survived_pairs:
            continue
        new_compound_mzrts[dict_key] = copy.deepcopy(cmzrt)

    # Generate a new atlas UID
    new_atlas_uid = _generate_uid(
        "autoid_atlas",
        decorator=(
            f"{source_atlas.analysis_type.lower()}-"
            f"{source_atlas.chromatography.lower()}-"
            f"{source_atlas.polarity.lower()}"
        )
    )

    # Shallow-copy the atlas, then replace the fields that must change
    new_atlas = copy.copy(source_atlas)
    new_atlas.atlas_uid        = new_atlas_uid
    new_atlas.compound_mzrts   = new_compound_mzrts
    new_atlas.source_atlas_uid = source_atlas.atlas_uid
    new_atlas.atlas_name = source_atlas.atlas_name + " (post-auto-identification)"
    new_atlas.atlas_description = source_atlas.atlas_description + " (post-auto-identification)"
    new_atlas.atlas_type = "AUTO_IDED"
    new_atlas.created_by = prov["analyst"]
    new_atlas.created_date = prov["timestamp"]
    save_atlas_to_database(new_atlas, auto_id_obj.paths['project_db_path'], auto_id_obj.paths['main_db_path'])

    logger.info(
        f"Created and saved post-auto-identification atlas {new_atlas_uid} (from source atlas {source_atlas.atlas_uid}) "
        f"with ({len(new_compound_mzrts)} compounds)."
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
    for chrom, pol_dict in targeted_analyses.items():
        for pol, analysis_dict in pol_dict.items():
            for analysis_type, atlas_params_dict in analysis_dict.items():
                target_atlas_uid = atlas_params_dict.get('ATLAS', {}).get('uid', None)
                if target_atlas_uid is None:
                    logger.info(f"Skipping {chrom} {pol} {analysis_type} - no target atlas UID found in parameters")
                    continue

                logger.info(f"Loading {chrom} {pol} {analysis_type} target atlas with UID {target_atlas_uid} for applying RT alignment model...")
                atlas_obj = Atlas.from_database(main_db_path, target_atlas_uid)

                # Create a new Atlas object for the RT-aligned version
                aligned_compound_mzrts = {}
                for inchi_key, comp_ref in atlas_obj.compound_mzrts.items():
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
                    mz_rt_uid = _generate_uid("mz_rt", decorator="exp")
                    comp_dict = {k: v for k, v in comp_ref.__dict__.items() if k not in ['mz_rt_uid', 'rt_peak', 'rt_min', 'rt_max']}
                    aligned_comp_mzrt = CompoundMZRT(
                        **comp_dict,
                        mz_rt_uid=mz_rt_uid,
                        rt_peak=aligned_rt_peak,
                        rt_min=aligned_rt_min,
                        rt_max=aligned_rt_max,
                    )
                    aligned_compound_mzrts[inchi_key] = aligned_comp_mzrt

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

def save_auto_identification_results_to_db(
    auto_id_obj: "AutoIdentification"
) -> None:
    """
    Save complete analysis results to project database from AutoIdentification object.
    Handles both raw experimental data and MS summaries in one unified function.
    """
    logger.info("Preparing AutoIdentification results for database save...")

    # Extract paths and numbers from auto_id_obj
    project_db_path = auto_id_obj.paths['project_db_path']
    main_db_path = auto_id_obj.paths['main_db_path']
    rt_alignment_number = auto_id_obj.rt_alignment_number
    analysis_number = auto_id_obj.analysis_number
    analysis_type = auto_id_obj.pre_autoid_atlas_obj.analysis_type
    exp_data_obj = auto_id_obj.experimental_data

    # Get time and analyst for provenance
    prov = get_provenance()

    # Get compound_uid mappings
    inchi_keys = [ci.inchi_key for ci in exp_data_obj.manual_curation]
    compound_uid_map = get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)

    # Collect all records
    manual_curation_records = []
    ms2_hits_records = []
    ms1_data_records = []
    ms2_data_records = []

    # CompoundInfo
    logger.info(f"Processing manual curation entries...")
    for ci in tqdm(exp_data_obj.manual_curation, desc="Processing manual curation entries"):
        compound_uid = compound_uid_map.get(ci.inchi_key)
        if not compound_uid:
            logger.warning(f"Could not find compound_uid for {ci.inchi_key}, skipping")
            continue
        metadata_record = _prepare_manual_curation_record(
            ci.data.iloc[0:1], compound_uid, ci.inchi_key, ci.adduct,
            rt_alignment_number, analysis_number, analysis_type, prov
        )
        if metadata_record:
            manual_curation_records.append(metadata_record)

    # MS1Data
    logger.info(f"Processing MS1 data entries...")
    for ms1 in tqdm(exp_data_obj.ms1_data, desc="Processing MS1 data entries"):
        compound_uid = compound_uid_map.get(ms1.inchi_key)
        if not compound_uid or ms1.data.empty:
            continue
        else:
            filename = ms1.filename
            ms1.data['filename'] = filename
            ms1.data = ms1.data[['filename', 'mz', 'i', 'rt']]
            ms1_data_grouped = ms1.data.groupby('filename', sort=False).agg({
                'rt': list,
                'mz': list,
                'i': list,
            }).reset_index()
            ms1_data_grouped['raw_spectrum'] = list(zip(ms1_data_grouped['rt'], ms1_data_grouped['i']))

        for _, row in ms1_data_grouped.iterrows():
            ms1_data_record = _prepare_ms1_data_record(
                row, compound_uid, ms1.inchi_key, ms1.adduct, ms1.filename,
                rt_alignment_number, analysis_number, analysis_type, prov
            )
            if ms1_data_record:
                ms1_data_records.append(ms1_data_record)

    # MS2Data
    logger.info(f"Processing MS2 data entries...")
    for ms2 in tqdm(exp_data_obj.ms2_data, desc="Processing MS2 data entries"):
        compound_uid = compound_uid_map.get(ms2.inchi_key)
        if not compound_uid or ms2.data.empty:
            continue
        else:
            ms2.data = ms2.data[['mz', 'i', 'rt', 'precursor_MZ', 'precursor_intensity', 'collision_energy']]
            ms2_data_grouped = ms2.data.groupby('rt', sort=False).agg({
                'precursor_MZ': 'first',
                'precursor_intensity': 'first',
                'collision_energy': 'first',
                'mz': list,
                'i': list,
            }).reset_index()
            ms2_data_grouped['raw_spectrum'] = list(zip(ms2_data_grouped['mz'], ms2_data_grouped['i']))

        for _, row in ms2_data_grouped.iterrows():
            ms2_data_record = _prepare_ms2_data_record(
                row, compound_uid, ms2.inchi_key, ms2.adduct, rt_alignment_number, analysis_number, analysis_type, ms2.filename, prov
            )
            if ms2_data_record:
                ms2_data_records.append(ms2_data_record)

    # MS2Hits
    logger.info(f"Processing MS2 hit entries...")
    for ms2_hit in tqdm(exp_data_obj.ms2_hits, desc="Processing MS2 hit entries"):
        compound_uid = compound_uid_map.get(ms2_hit.inchi_key)
        if not compound_uid or ms2_hit.data.empty:
            continue
        for _, hit in ms2_hit.data.iterrows():
            ms2_hit_record = _prepare_ms2_hit_record(
                hit, compound_uid, ms2_hit.inchi_key, ms2_hit.adduct, ms2_hit.filename,
                rt_alignment_number, analysis_number, analysis_type, prov
            )
            if ms2_hit_record:
                ms2_hits_records.append(ms2_hit_record)

    # Check for identical MS2 summary records before inserting any data
    logger.info("Checking for identical ManualCuration records before database insert...")
    with get_db_connection(project_db_path, read_only=True) as conn:
        manual_curation_records, duplicate_keys = _check_identical_manual_curation_exists(conn, manual_curation_records)
    
    # Filter out associated records for duplicate compounds
    if duplicate_keys:
        logger.info(f"Filtering {len(duplicate_keys)} duplicate compound(s) from ms2_hits, ms1_data, and ms2_data records...")
        ms2_hits_records = [
            rec for rec in ms2_hits_records 
            if (rec[1], rec[2], rec[3], rec[6]) not in duplicate_keys  # compound_uid, inchi_key, adduct, analysis_type
        ]
        ms1_data_records = [
            rec for rec in ms1_data_records 
            if (rec[1], rec[2], rec[3], rec[6]) not in duplicate_keys  # compound_uid, inchi_key, adduct, analysis_type
        ]
        ms2_data_records = [
            rec for rec in ms2_data_records 
            if (rec[1], rec[2], rec[3], rec[6]) not in duplicate_keys  # compound_uid, inchi_key, adduct, analysis_type
        ]

    # Bulk insert all records
    logger.info("Performing bulk database inserts...")
    _bulk_insert_analysis_data(
        project_db_path,
        manual_curation_records,
        ms2_hits_records,
        ms1_data_records,
        ms2_data_records
    )

    logger.info("Database save complete. ")

    return

def to_python_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val
 
def write_gui_updates_to_db(
    project_db_path: str,
    curation_uid: str,
    updated_fields: dict
) -> None:
    """
    Update a manual curation entry with new values for specified fields.
    
    Args:
        project_db_path: Path to the project database
        curation_uid: UID of the manual curation entry to update
        updated_fields: Dictionary of fields to update with their new values
    """
    if not updated_fields:
        logger.warning("No fields provided for update.")
        return

    try:
        curation_uid = to_python_type(curation_uid)
        set_clause = ", ".join([f"{k} = ?" for k in updated_fields])
        params = [to_python_type(v) for v in updated_fields.values()] + [curation_uid]
        # Reduced retries for GUI operations - fail faster to avoid blocking user
        # With 5 retries and 0.5s initial delay: 0.5s + 1s + 2s + 4s = ~8s max wait
        with get_db_connection(project_db_path, max_retries=5, initial_retry_delay=0.5) as conn:
            conn.execute(f"UPDATE manual_curation SET {set_clause} WHERE curation_uid = ?", params)
    except Exception as e:
        logger.error(f"Error updating manual curation entry {curation_uid}: {e}")
        raise ValueError(f"Failed to update manual curation entry {curation_uid}. See logs for details.")

def _check_identical_manual_curation_exists(conn, manual_curation_records: List[tuple]) -> tuple:
    """
    Check for identical manual curation records in the database before inserting new ones.
    If an identical record exists (same compound_uid, inchi_key, adduct, rt_alignment_number, analysis_number, analysis_type),
    filter it out to prevent duplicate entries.
    
    Returns:
        tuple: (filtered_records, duplicate_keys) where duplicate_keys is a set of 
               (compound_uid, inchi_key, adduct, analysis_type) tuples that were already in the database
    """
    filtered_records = []
    duplicate_keys = set()
    
    for record in manual_curation_records:
        compound_uid = record[1]
        inchi_key = record[2]
        adduct = record[3]
        rt_alignment_number = record[4]
        analysis_number = record[5]
        # analysis_type is at index 10 (after compound_name, auto_ided, polarity, chromatography)
        analysis_type = record[10]

        existing = conn.execute("""
            SELECT curation_uid FROM manual_curation
            WHERE compound_uid = ? AND inchi_key = ? AND adduct = ?
            AND rt_alignment_number = ? AND analysis_number = ? AND analysis_type = ?
        """, [
            compound_uid, inchi_key, adduct, rt_alignment_number, analysis_number, analysis_type
        ]).fetchone()

        if existing:
            duplicate_keys.add((compound_uid, inchi_key, adduct, analysis_type))
            logger.info(
                f"Skipping duplicate manual curation record for compound_uid {compound_uid}, "
                f"inchi_key {inchi_key}, adduct {adduct}, analysis_type {analysis_type} (already exists for rt_alignment_number {rt_alignment_number}, "
                f"analysis_number {analysis_number})"
            )
        else:
            filtered_records.append(record)
    
    return filtered_records, duplicate_keys

def _prepare_manual_curation_record(
    manual_curation: pd.DataFrame,
    compound_uid: str,
    inchi_key: str,
    adduct: str,
    rt_alignment_number: int,
    analysis_number: int,
    analysis_type: str,
    prov: Dict
) -> Optional[tuple]:
    """Prepare compound metadata record for database insertion."""
    try:
        row = manual_curation.iloc[0]
        return (
            _generate_uid("manual_curation"),
            compound_uid,
            inchi_key,
            adduct,
            rt_alignment_number,
            analysis_number,
            row.get('compound_name', ''),
            bool(row.get('auto_ided', False)),
            row.get('polarity', ''),
            row.get('chromatography', ''),
            analysis_type,
            float(row.get('mz_tolerance', 5.0)),
            float(row.get('atlas_mz', 0.0)),
            float(row.get('atlas_rt_peak', 0.0)),
            float(row.get('atlas_rt_min', 0.0)),
            float(row.get('atlas_rt_max', 0.0)),
            float(row.get('rt_peak', 0.0)),
            float(row.get('rt_min', 0.0)),
            float(row.get('rt_max', 0.0)),
            row.get('identification_notes', ''),
            row.get('ms1_notes', ''),
            row.get('ms2_notes', ''),
            row.get('analyst_notes', ''),
            row.get('other_notes', ''),
            row.get('best_ms1_file', ''),
            float(row.get('best_ms1_rt', 0.0)),
            float(row.get('best_ms1_mz', 0.0)),
            float(row.get('best_ms1_intensity', 0.0)),
            float(row.get('best_ms1_ppm_error', 0.0)),
            float(row.get('best_ms1_rt_error', 0.0)),
            json.dumps(row['isomers']) if row['isomers'] is not None else '[]',
            float(row.get('suggested_rt_min', 0.0)),
            float(row.get('suggested_rt_max', 0.0)),
            float(row.get('suggested_rt_peak', 0.0)),
            float(row.get('rt_suggestion_confidence', 0.0)),
            prov["analyst"],
            prov["timestamp"]
        )
    except Exception as e:
        logger.error(f"Error preparing compound metadata record: {e}")
        return None

def _prepare_ms1_data_record(
    row: pd.Series,
    compound_uid: str,
    inchi_key: str,
    adduct: str,
    filename: str,
    rt_alignment_number: int,
    analysis_number: int,
    analysis_type: str,
    prov: Dict
) -> Optional[tuple]:
    """Prepare MS1 raw data record for database insertion."""
    try:
        return (
            _generate_uid("ms1_data"),
            compound_uid,
            inchi_key,
            adduct,
            rt_alignment_number,
            analysis_number,
            analysis_type,
            filename,
            json.dumps(row.get('mz', [])),
            json.dumps(row.get('raw_spectrum', ('[]', '[]'))),
            prov["analyst"],
            prov["timestamp"]
        )
    except Exception as e:
        logger.error(f"Error preparing MS1 data record: {e}")
        return None

def _prepare_ms2_data_record(
    row: pd.Series,
    compound_uid: str,
    inchi_key: str,
    adduct: str,
    rt_alignment_number: int,
    analysis_number: int,
    analysis_type: str,
    filename: str,
    prov: Dict
) -> Optional[tuple]:
    """Prepare MS2 raw data record for database insertion."""
    try:
        return (
            _generate_uid("ms2_data"),
            compound_uid,
            inchi_key,
            adduct,
            rt_alignment_number,
            analysis_number,
            analysis_type,
            filename,
            float(row.get('rt', 0.0)),
            json.dumps(row.get('raw_spectrum', ('[]', '[]'))),
            float(row.get('precursor_MZ', 0.0)),
            float(row.get('precursor_intensity', 0.0)),
            float(row.get('collision_energy', 0.0)),
            prov["analyst"],
            prov["timestamp"]
        )
    except Exception as e:
        logger.error(f"Error preparing MS2 data record: {e}")
        return None

def _prepare_ms2_hit_record(
    hit: pd.Series,
    compound_uid: str,
    inchi_key: str,
    adduct: str,
    filename: str,
    rt_alignment_number: int,
    analysis_number: int,
    analysis_type: str,
    prov: Dict
) -> Optional[tuple]:
    """Prepare MS2 hit record for database insertion."""
    try:
        return (
            _generate_uid("ms2_hits"),
            compound_uid,
            inchi_key,
            adduct,
            rt_alignment_number,
            analysis_number,
            analysis_type,
            filename,
            hit.get('database', ''),
            hit.get('ref_id', ''),
            hit.get('ref_name', ''),
            float(hit.get('score', 0.0)),
            int(hit.get('num_matches', 0)),
            float(hit.get('mz_theoretical', 0.0)),
            float(hit.get('mz_measured', 0.0)),
            float(hit.get('ppm_error', 0.0)),
            float(hit.get('rt', 0.0)),
            float(hit.get('qry_intensity_peak', 0.0)),
            int(hit.get('ref_frags', 0)),
            int(hit.get('data_frags', 0)),
            json.dumps(hit.get('matched_fragments', [])),
            json.dumps(hit.get('aligned_fragment_colors', [])),
            json.dumps(hit.get('qry_spectrum', [[], []])),
            json.dumps(hit.get('ref_spectrum', [[], []])),
            prov["analyst"],
            prov["timestamp"],
        )
    except Exception as e:
        logger.error(f"Error preparing MS2 hit record: {e}")
        return None

def _bulk_insert_analysis_data(
    project_db_path: str,
    manual_curation_records: List[tuple],
    ms2_hits_records: List[tuple],
    ms1_data_records: List[tuple],
    ms2_data_records: List[tuple],
    batch_size: int = 500
) -> None:
    """Perform bulk inserts for all analysis data types with progress tracking.
    
    Args:
        project_db_path: Path to project database
        manual_curation_records: List of manual curation record tuples
        ms2_hits_records: List of MS2 hits record tuples
        ms1_data_records: List of MS1 data record tuples
        ms2_data_records: List of MS2 data record tuples
        batch_size: Number of records to insert per batch (default 500)
    """
    
    def _insert_in_batches(conn, query: str, records: List[tuple], desc: str):
        """Helper to insert records in batches with progress bar."""
        total_records = len(records)
        num_batches = (total_records + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc=f"{desc} in {num_batches} batches ({total_records} records)"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_records)
            batch = records[start_idx:end_idx]
            conn.executemany(query, batch)
    
    with get_db_connection(project_db_path, max_retries=20, initial_retry_delay=1.0) as conn:
        if manual_curation_records:
            logger.info(f"Inserting {len(manual_curation_records)} compound metadata records...")
            _insert_in_batches(
                conn,
                """INSERT INTO manual_curation VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                manual_curation_records,
                "Inserting manual curation records"
            )
        
        if ms1_data_records:
            logger.info(f"Inserting {len(ms1_data_records)} MS1 raw data records...")
            _insert_in_batches(
                conn,
                """INSERT INTO ms1_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ms1_data_records,
                "Inserting MS1 data records"
            )
        
        if ms2_data_records:
            logger.info(f"Inserting {len(ms2_data_records)} MS2 raw data records...")
            _insert_in_batches(
                conn,
                """INSERT INTO ms2_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ms2_data_records,
                "Inserting MS2 data records"
            )

        if ms2_hits_records:
            logger.info(f"Inserting {len(ms2_hits_records)} MS2 hits records...")
            _insert_in_batches(
                conn,
                """INSERT INTO ms2_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ms2_hits_records,
                "Inserting MS2 hits records"
            )

def display_auto_id_summary(auto_id_obj: "AutoIdentification") -> None:
    """
    Display a summary table of auto identification results to the logger.
    Shows number of compounds, MS1/MS2 datapoints, MS2 hits, etc.
    """

    summary_rows = []
    for ci in auto_id_obj.experimental_data.manual_curation:
        inchi_key = ci.inchi_key
        adduct = ci.adduct

        ms1_count = sum(
            len(ms1.data) for ms1 in auto_id_obj.experimental_data.ms1_data
            if ms1.inchi_key == inchi_key and ms1.adduct == adduct
        )
        ms2_count = sum(
            len(ms2.data) for ms2 in auto_id_obj.experimental_data.ms2_data
            if ms2.inchi_key == inchi_key and ms2.adduct == adduct
        )
        ms2_hits_count = sum(
            len(ms2_hit.data) for ms2_hit in auto_id_obj.experimental_data.ms2_hits
            if ms2_hit.inchi_key == inchi_key and ms2_hit.adduct == adduct
        )

        summary_rows.append({
            "inchi_key": inchi_key,
            "adduct": adduct,
            "MS1 datapoints": ms1_count,
            "MS2 datapoints": ms2_count,
            "MS2 hits": ms2_hits_count,
        })

    return