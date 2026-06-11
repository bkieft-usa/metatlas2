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
    elif entity_type == "config":
        return f"cfg-{uuid.uuid4().hex[:32]}"
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
    elif entity_type == "workflow_stage_run":
        return f"wfr-{uuid.uuid4().hex[:32]}"
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
        confidence = row.get('confidence', None)
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
            confidence=confidence,
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
        source=atlas_file_path,
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
                logger.debug("Attached main database for compound metadata")
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
                        a.analysis_name,
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
                        a.analysis_name,
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
        logger.debug(f"Retrieved {len(df)} compounds for atlas {atlas_uid} ({df['atlas_name'].iloc[0]})")

    return df

def query_atlases_metadata(
    database_path: str,
    chromatography: Optional[str] = None,
    polarity: Optional[str] = None,
    analysis_type: Optional[str] = None,
    analysis_name: Optional[str] = None,
    created_by: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query the atlases table and return metadata rows matching the supplied filters.

    All filter arguments are optional.  When provided, matching is
    case-insensitive and uses SQL ILIKE so partial values (e.g. ``'C18'``)
    will match any atlas whose field contains that substring.

    Args:
        database_path: Path to the main (or project) database file.
        chromatography: Filter by chromatography method (partial match).
        polarity:       Filter by polarity (partial match).
        analysis_type:  Filter by analysis type (partial match).
        analysis_name:  Filter by analysis name (partial match).
        created_by:     Filter by creator username (partial match).

    Returns:
        A :class:`pandas.DataFrame` with one row per matching atlas, containing
        all columns from the ``atlases`` table.  Returns an empty DataFrame
        when no atlases match.
    """
    conditions: List[str] = []
    params: List[str] = []

    filter_map = {
        "chromatography": chromatography,
        "polarity": polarity,
        "analysis_type": analysis_type,
        "analysis_name": analysis_name,
        "created_by": created_by,
    }
    for column, value in filter_map.items():
        if value is not None:
            conditions.append(f"{column} ILIKE ?")
            params.append(f"%{value}%")

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
        SELECT
            atlas_uid,
            atlas_name,
            atlas_description,
            chromatography,
            polarity,
            analysis_type,
            analysis_name,
            atlas_type,
            created_by,
            created_date,
            source
        FROM atlases
        {where_clause}
        ORDER BY created_date DESC, atlas_name
    """

    with get_db_connection(database_path, read_only=True) as conn:
        try:
            df = conn.execute(query, params).df()
        except Exception as e:
            logger.error(f"Error querying atlases metadata: {e}")
            return pd.DataFrame()

    logger.info(
        "Found %d atlas(es) matching filters in %s",
        len(df),
        database_path,
    )
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
        "r_squared": float(rt_alignment_model.get('r2', 0.0)),
        "rmse": float(rt_alignment_model.get('rmse', 0.0)),
        "equation": rt_alignment_model.get('equation', ''),
        "poly_degree": int(rt_alignment_model['degree']),
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

    # save RT alignment model metadata to JSON file in RT alignment output directory for easy reference
    rt_alignment_output_dir = Path(rt_align_obj.paths['rt_alignment_results_dir'])
    rt_alignment_output_dir.mkdir(parents=True, exist_ok=True)
    rt_alignment_metadata_path = rt_alignment_output_dir / f"rt_alignment_model_{rt_alignment_uid}.json"
    with open(rt_alignment_metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=4)
    logger.info(f"RT alignment model metadata saved to {rt_alignment_metadata_path}")

    # display model metadata in log for easy reference
    logger.info(f"RT Alignment Model Metadata for UID {rt_alignment_uid}:")
    logger.info(f"  QC files used: {model_metadata['qc_files']}")
    logger.info(f"  Number of compounds used (by UID): {len(model_metadata['compounds_used'])}")
    logger.info(f"  R²: {model_metadata['r_squared']:.4f}")
    logger.info(f"  RMSE: {model_metadata['rmse']:.4f}")
    logger.info(f"  Polynomial degree: {model_metadata['poly_degree']}")
    logger.info(f"  Polynomial interaction_only: {model_metadata['poly_interaction_only']}")
    logger.info(f"  Model intercept: {model_metadata['model_intercept']}")
    logger.info(f"  Model coefficients: {model_metadata['model_coefficients']}")

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
                analysis_name TEXT,
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
                analysis_name TEXT,
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
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_uid TEXT PRIMARY KEY,
                rt_alignment_number INTEGER NOT NULL,
                analysis_number INTEGER,
                chromatography TEXT NOT NULL,
                polarity TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                analysis_name TEXT NOT NULL,
                stage TEXT NOT NULL,
                atlas_uid TEXT NOT NULL,
                source_atlas_uid TEXT,
                override_params TEXT,
                created_by TEXT,
                created_date TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_config (
                config_uid          TEXT PRIMARY KEY,
                rt_alignment_number INTEGER,
                analysis_number     INTEGER,
                config_yaml         TEXT NOT NULL,
                paths_json          TEXT,
                config_path         TEXT,
                created_by          TEXT,
                created_date        TEXT
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
                passed_autoid BOOLEAN,
                passed_curation BOOLEAN,
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

def save_config_to_db(
    project_db_path: str,
    config_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    paths: Optional[Dict] = None,
) -> str:
    """
    Snapshot the raw YAML text and computed paths dict into the
    ``project_config`` table.

    Called once at project setup with the RTA and TGA numbers from the CLI.
    All subsequent workflow stages (RT alignment, auto-ID) read the config and
    paths back from this single row via :func:`load_config_from_db` and
    :func:`load_paths_from_db` — no YAML file or ``set_up_paths()`` call is
    needed after project setup.

    If a row for the same ``(rt_alignment_number, analysis_number)`` already
    exists the call is a no-op and the existing ``config_uid`` is returned.

    Args:
        project_db_path:      Path to the project DuckDB database.
        config_path:          Absolute path to the YAML config file on disk.
        rt_alignment_number:  RTA index from the CLI call.
        analysis_number:      TGA index from the CLI call.
        paths:                Optional dict returned by ``set_up_paths()``.
                              Serialised as JSON into ``paths_json``.

    Returns:
        The ``config_uid`` of the stored (or pre-existing) row.
    """
    from pathlib import Path as _Path
    config_yaml = _Path(config_path).read_text()
    paths_json = json.dumps(paths) if paths is not None else None
    prov = get_provenance()

    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        # Idempotent: skip if a row for this RTA+TGA already exists
        existing = conn.execute("""
            SELECT config_uid FROM project_config
            WHERE rt_alignment_number IS NOT DISTINCT FROM ?
              AND analysis_number     IS NOT DISTINCT FROM ?
        """, [rt_alignment_number, analysis_number]).fetchone()

        if existing:
            logger.info(
                f"Config snapshot already exists for RTA={rt_alignment_number}, "
                f"TGA={analysis_number} (uid={existing[0]}). Skipping."
            )
            return existing[0]

        config_uid = _generate_uid("config")
        conn.execute("""
            INSERT INTO project_config VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config_uid,
            rt_alignment_number,
            analysis_number,
            config_yaml,
            paths_json,
            config_path,
            prov["analyst"],
            prov["timestamp"],
        ))

    logger.info(
        f"Saved config snapshot {config_uid} to project_config "
        f"(RTA={rt_alignment_number}, TGA={analysis_number}, path={config_path})"
    )
    return config_uid


def load_config_from_db(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> Optional["Metatlas2Config"]:
    """
    Load and parse a :class:`Metatlas2Config` from the ``project_config`` table.

    Looks up the single row stored for this ``(rt_alignment_number,
    analysis_number)`` pair — the one written by :func:`save_config_to_db`
    during project setup.

    Returns ``None`` if no matching row is found.

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  RTA index from the CLI call.
        analysis_number:      TGA index from the CLI call.

    Returns:
        Parsed :class:`Metatlas2Config` or ``None``.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            row = conn.execute("""
                SELECT config_yaml, config_path FROM project_config
                WHERE rt_alignment_number IS NOT DISTINCT FROM ?
                  AND analysis_number     IS NOT DISTINCT FROM ?
                ORDER BY created_date DESC
                LIMIT 1
            """, [rt_alignment_number, analysis_number]).fetchone()
            if row:
                config_yaml, config_path = row
                logger.info(
                    f"Loaded config from project_config table "
                    f"(RTA={rt_alignment_number}, TGA={analysis_number}, "
                    f"original_path={config_path})"
                )
                return ldt.load_metatlas2_config_from_string(config_yaml)
    except Exception as e:
        raise RuntimeError(
            f"No config found in project database at {project_db_path} for reason {e} "
            f"for RTA={rt_alignment_number}, TGA={analysis_number}. "
            "Ensure run_project_setup was called with a config_path."
        )


def load_paths_from_db(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> Optional[Dict]:
    """
    Load the serialised ``paths`` dict from the ``project_config`` table.

    Returns the dict that was passed to :func:`save_config_to_db` as the
    ``paths`` argument (the output of ``set_up_paths()``), or ``None`` if no
    matching row or no ``paths_json`` value is found.

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  RTA index from the CLI call.
        analysis_number:      TGA index from the CLI call.

    Returns:
        Paths dict or ``None``.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            row = conn.execute("""
                SELECT paths_json FROM project_config
                WHERE rt_alignment_number IS NOT DISTINCT FROM ?
                  AND analysis_number     IS NOT DISTINCT FROM ?
                ORDER BY created_date DESC
                LIMIT 1
            """, [rt_alignment_number, analysis_number]).fetchone()
            if row and row[0]:
                paths = json.loads(row[0])
                logger.info(
                    f"Loaded paths from project_config table "
                    f"(RTA={rt_alignment_number}, TGA={analysis_number})"
                )
                return paths
    except Exception as e:
        raise RuntimeError(
            f"No paths found in project database at {project_db_path} for reason {e} "
            f"for RTA={rt_alignment_number}, TGA={analysis_number}. "
            "Ensure run_project_setup was called with config_path and paths."
        )



def update_config_overrides(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    atlas_uid: str,
    override_params: Dict,
    stage: str = 'MANUALLY_CURATED',
) -> None:
    """
    Persist analyst override parameters onto the matching ``workflow_runs`` row.

    Called after the GUI is launched (``stage='AUTO_IDED'``) to record the
    override parameters the analyst chose, and again after
    ``run_analysis_summary`` (``stage='MANUALLY_CURATED'``) to record the
    parameters used to produce the final curated output.

    The ``override_params`` dict is serialised as JSON into the
    ``workflow_runs.override_params`` column.  ``None`` values are stripped so
    only analyst-set overrides are stored.

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  RTA index of the run.
        analysis_number:      TGA index of the run.
        atlas_uid:            UID of the atlas whose ``workflow_runs`` row
                              should be updated.
        override_params:      Dict of override parameters.
        stage:                ``'AUTO_IDED'`` (GUI) or ``'MANUALLY_CURATED'``
                              (summary).  Defaults to ``'MANUALLY_CURATED'``.
    """
    clean = {k: v for k, v in override_params.items() if v is not None}
    override_json = json.dumps(clean)

    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        conn.execute("""
            UPDATE workflow_runs
            SET override_params = ?
            WHERE rt_alignment_number = ?
              AND analysis_number = ?
              AND atlas_uid = ?
              AND stage = ?
        """, [override_json, rt_alignment_number, analysis_number, atlas_uid, stage])

    logger.info(
        f"Saved override_params to workflow_runs for atlas {atlas_uid} "
        f"(RTA={rt_alignment_number}, TGA={analysis_number}, stage={stage}): {clean}"
    )


def register_workflow_run(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: Optional[int],
    atlas_obj: "Atlas",
    stage: str,
) -> str:
    """
    Register a workflow atlas handoff in the ``workflow_runs`` table.

    This is the durable, database-backed replacement for the flat CSV files
    that previously tracked which atlas UID to use at each stage
    (``RT_ALIGNED``, ``AUTO_IDED``, ``MANUALLY_CURATED``).

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  Integer RT alignment run index (RTA#).
        analysis_number:      Integer analysis run index (TGA#); ``None`` at
                              the ``RT_ALIGNED`` stage (before TGA is assigned).
        atlas_obj:            The :class:`Atlas` that was just saved.
        stage:                One of ``'RT_ALIGNED'``, ``'AUTO_IDED'``,
                              ``'MANUALLY_CURATED'``.

    Returns:
        The newly generated ``run_uid`` string.
    """
    prov = get_provenance()
    run_uid = _generate_uid("workflow_stage_run")
    with get_db_connection(project_db_path, max_retries=10, initial_retry_delay=0.5) as conn:
        conn.execute("""
            INSERT INTO workflow_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_uid,
            rt_alignment_number,
            analysis_number,
            atlas_obj.chromatography,
            atlas_obj.polarity,
            atlas_obj.analysis_type,
            atlas_obj.analysis_name,
            stage,
            atlas_obj.atlas_uid,
            atlas_obj.source_atlas_uid,
            None,  # override_params (to be updated during GUI)
            prov["analyst"],
            prov["timestamp"],
        ))
    logger.info(
        f"Registered workflow run {run_uid}: stage={stage}, "
        f"RTA={rt_alignment_number}, TGA={analysis_number}, "
        f"atlas_uid={atlas_obj.atlas_uid} "
        f"({atlas_obj.chromatography}/{atlas_obj.polarity}/"
        f"{atlas_obj.analysis_type}/{atlas_obj.analysis_name})"
    )
    return run_uid


def count_workflow_runs(
    project_db_path: str,
    rt_alignment_number: int,
    stage: str,
    analysis_number: Optional[int] = None,
) -> int:
    """
    Return the number of ``workflow_runs`` rows matching the given filters.

    Used by :meth:`RTAlign.setup` to decide whether RT alignment has already
    been completed for a given RTA number.

    RT-aligned atlases are registered with ``analysis_number=NULL`` (since the
    TGA is not yet assigned at alignment time), so the ``analysis_number``
    filter uses ``IS NOT DISTINCT FROM ?`` for correct NULL-safe comparison.
    When ``analysis_number=None`` is passed (the default), this matches rows
    where ``analysis_number IS NULL``.  Pass an explicit integer to match a
    specific TGA.

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  RT alignment run index to filter on.
        stage:                Workflow stage to filter on (``'RT_ALIGNED'``,
                              ``'AUTO_IDED'``, ``'MANUALLY_CURATED'``).
        analysis_number:      TGA index; ``None`` matches rows where
                              ``analysis_number IS NULL`` (RT_ALIGNED stage).

    Returns:
        Integer row count (0 if the table does not yet exist).
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            count = conn.execute("""
                SELECT COUNT(*) FROM workflow_runs
                WHERE rt_alignment_number = ?
                  AND analysis_number IS NOT DISTINCT FROM ?
                  AND stage = ?
            """, [rt_alignment_number, analysis_number, stage]).fetchone()[0]
        return int(count)
    except Exception as e:
        logger.debug(f"count_workflow_runs: {e}")
        return 0


def get_atlases_for_stage(
    project_db_path: str,
    rt_alignment_number: int,
    stage: str,
    analysis_number: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of atlas identity columns for a given workflow stage.

    This is the database-backed replacement for
    ``ldt.load_atlas_data_from_csv(aligned_atlases_store_file)``.

    Columns returned: ``atlas_uid``, ``chromatography``, ``polarity``,
    ``analysis_type``, ``analysis_name``, ``rt_alignment_number``,
    ``analysis_number``.

    Args:
        project_db_path:      Path to the project DuckDB database.
        rt_alignment_number:  RT alignment run index to filter on.
        stage:                Workflow stage (``'RT_ALIGNED'``, ``'AUTO_IDED'``,
                              ``'MANUALLY_CURATED'``).
        analysis_number:      Optional TGA index; when ``None`` the filter is
                              omitted.

    Returns:
        :class:`pandas.DataFrame` ordered by ``created_date``.  Empty
        DataFrame when no rows match.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute("""
                SELECT atlas_uid, chromatography, polarity,
                       analysis_type, analysis_name,
                       rt_alignment_number, analysis_number
                FROM workflow_runs
                WHERE rt_alignment_number = ?
                  AND analysis_number IS NOT DISTINCT FROM ?
                  AND stage = ?
                ORDER BY created_date
            """, [rt_alignment_number, analysis_number, stage]).df()
    except Exception as e:
        logger.error(f"get_atlases_for_stage: {e}")
        df = pd.DataFrame()
    if df.empty:
        logger.warning(
            f"No workflow_runs entries found for stage={stage}, "
            f"RTA={rt_alignment_number}, TGA={analysis_number}."
        )
    else:
        logger.info(
            f"Found {len(df)} atlas(es) in workflow_runs for stage={stage}, "
            f"RTA={rt_alignment_number}, TGA={analysis_number}."
        )
    return df


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
                INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atlas_obj.atlas_uid,
                atlas_obj.atlas_name,
                atlas_obj.atlas_description,
                atlas_obj.chromatography,
                atlas_obj.polarity,
                atlas_obj.analysis_type,
                atlas_obj.analysis_name,
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
                INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                atlas_obj.atlas_uid,
                atlas_obj.atlas_name,
                atlas_obj.atlas_description,
                atlas_obj.chromatography,
                atlas_obj.polarity,
                atlas_obj.analysis_type,
                atlas_obj.analysis_name,
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

def create_new_atlas_after_manual_curation(
    summary_obj: "AnalysisSummary",
) -> "Atlas":
    """
    Create a new Atlas object after manual curation.
    Applies manual RT adjustments and removes flagged compounds.
    """
    prov = get_provenance()
    source_atlas = summary_obj.post_autoid_atlas_obj

    remove_unided_compounds = False
    if summary_obj.workflow_params.get('remove_unided_compounds', True) or summary_obj.override_parameters.get('remove_unided_compounds', True):
        remove_unided_compounds = True
        logger.info("Compounds without manual curation entries will be removed from the curated atlas.")

    remove_flagged_compounds = False
    if summary_obj.workflow_params.get('remove_flagged_compounds', True) or summary_obj.override_parameters.get('remove_flagged_compounds', True):
        remove_flagged_compounds = True
        logger.info("Compounds with 'remove' flag in manual curation ms1 notes will be removed from the curated atlas.")
    
    require_eval = True
    if summary_obj.workflow_params.get('gui_require_all_evaluated', True) or summary_obj.override_parameters.get('gui_require_all_evaluated', True):
        require_eval = True
        logger.info("All compounds are required to have manual curation entries with ms1 and ms2 notes. Missing notes will cause an error.")

    # 1. Construct manual curation object based on filtered atlas
    atlas_compounds = pd.DataFrame({"mz_rt_uid": list(source_atlas.compound_mzrts.keys())})
    curation_df = get_manual_curation_entries(
        summary_obj.paths['project_db_path'],
        summary_obj.rt_alignment_number,
        summary_obj.analysis_number,
        remove_unidentified_compounds=remove_unided_compounds,
        remove_flagged_compounds=remove_flagged_compounds,
        atlas_compounds=atlas_compounds,
        analysis_type=source_atlas.analysis_type,
    )

    # Main Processing Loop
    curation_lookup = {row['mz_rt_uid']: row for row in curation_df.to_dict('records')}
    new_compound_mzrts = {}
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
                        "has empty ms1_notes. Please update in GUI before generating summary."
                    )
                if not ms2_notes:
                    raise ValueError(
                        f"Compound {cmzrt.compound_uid} ({cmzrt.inchi_key} / {cmzrt.adduct}) "
                        "has empty ms2_notes. Please update in GUI before generating summary."
                    )

            # C. Apply Updates
            # Shallow copy is sufficient as we only update immutable floats
            new_cmzrt = copy.copy(cmzrt)
            new_cmzrt.rt_peak = float(curation_row.get('rt_peak', cmzrt.rt_peak))
            new_cmzrt.rt_min  = float(curation_row.get('rt_min',  cmzrt.rt_min))
            new_cmzrt.rt_max  = float(curation_row.get('rt_max',  cmzrt.rt_max))
            new_compound_mzrts[mz_rt_uid] = new_cmzrt
            
        else:
            # No curation entry: keep original, but still copy to avoid shared references
            new_compound_mzrts[mz_rt_uid] = copy.copy(cmzrt)

    # 4. Atlas Generation
    new_atlas_uid = _generate_uid(
        "curated_atlas",
        decorator=f"{source_atlas.analysis_type.lower()}-{source_atlas.analysis_name.lower()}-{source_atlas.chromatography.lower()}-{source_atlas.polarity.lower()}"
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

    logger.info(f"Attaching curated atlas {new_atlas_uid} with {len(new_compound_mzrts)} compounds to summary object.")
    summary_obj.post_curation_atlas_obj = new_atlas

    return

def save_curated_atlaes_to_db(summary_obj: "AnalysisSummary") -> None:
    logger.info("Saving curated Atlas to database...")
    save_atlas_to_database(
        summary_obj.post_curation_atlas_obj,
        summary_obj.paths['project_db_path'],
        summary_obj.paths['main_db_path']
    )
    logger.info("Registering curated Atlas in workflow_runs table...")
    register_workflow_run(
        project_db_path=summary_obj.paths['project_db_path'],
        rt_alignment_number=summary_obj.rt_alignment_number,
        analysis_number=summary_obj.analysis_number,
        atlas_obj=summary_obj.post_curation_atlas_obj,
        stage='MANUALLY_CURATED',
    )
    logger.info("Saving curated Atlas data to TSV...")
    ldt.save_atlas_data_to_tsv(
        atlas_obj=summary_obj.post_curation_atlas_obj,
        output_path=summary_obj.paths['analysis_output_dir']
    )

def create_new_atlas_after_auto_id(
    auto_id_obj: "AutoIdentification"
) -> None:
    """
    Create a new atlas after auto-identification.
    Surviving compounds are those present in the filtered manual_curation list.
    """
    source_atlas = auto_id_obj.pre_autoid_atlas_obj

    # 1. Get the set of surviving UIDs
    surviving_uids = auto_id_obj.experimental_data.curation_df['mz_rt_uid'].unique().tolist()

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
            f"{source_atlas.analysis_name.lower()}-"
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
    new_atlas.created_by = get_provenance()["analyst"]
    new_atlas.created_date = get_provenance()["timestamp"]

    logger.info(
        f"Created and saved post-auto-identification atlas {new_atlas_uid} "
        f"(from source {source_atlas.atlas_uid}) with {len(new_compound_mzrts)} compounds."
    )
    
    auto_id_obj.post_autoid_atlas_obj = new_atlas

    save_autoided_atlas_to_database(auto_id_obj)

    return

def save_autoided_atlas_to_database(auto_id_obj: "AutoIdentification") -> None:
    logger.info("Saving post-autoid Atlas to database...")
    save_atlas_to_database(
        auto_id_obj.post_autoid_atlas_obj,
        auto_id_obj.paths['project_db_path'],
        auto_id_obj.paths['main_db_path']
    )
    logger.info("Registering post-autoid Atlas in workflow_runs table...")
    register_workflow_run(
        project_db_path=auto_id_obj.paths['project_db_path'],
        rt_alignment_number=auto_id_obj.rt_alignment_number,
        analysis_number=auto_id_obj.analysis_number,
        atlas_obj=auto_id_obj.post_autoid_atlas_obj,
        stage='AUTO_IDED',
    )
    logger.info("Saving post-autoid Atlas data to TSV...")
    ldt.save_atlas_data_to_tsv(
        atlas_obj=auto_id_obj.post_autoid_atlas_obj,
        output_path=auto_id_obj.paths['analysis_output_dir']
    )

def create_new_atlases_after_rt_alignment(
    rt_align_obj: "RTAlignment"
) -> tuple[dict[str, "Atlas"], list[float]]:
    """Apply RT alignment model to target atlases and return new Atlas objects with updated RTs, 
    along with a list of all RT shifts applied.
    """

    from metatlas2.workflow_objects import Atlas

    def apply_rt_model(atlas_rt_values, model_info):
        """Apply RT alignment model to Atlas RT values."""
        X_new = np.array(atlas_rt_values).reshape(-1, 1)
        X_new_poly = model_info['poly_features'].transform(X_new)
        aligned_rt = model_info['model'].predict(X_new_poly)
        return aligned_rt

    aligned_atlases = {}
    for ta in tqdm(rt_align_obj.config.targeted_analyses, desc="Updating atlases with RT alignment", disable=should_disable_tqdm()):
        chrom = ta.chromatography
        pol = ta.polarity
        analysis_type = ta.analysis_type
        analysis_name = ta.analysis_name
        target_atlas_uid = ta.atlas_uid
        if not target_atlas_uid:
            logger.debug(f"Skipping RT alignment for {chrom} {pol} {analysis_type}-{analysis_name} - no target atlas UID found in parameters")
            continue

        logger.debug(f"Loading {chrom} {pol} {analysis_type}-{analysis_name} target atlas with UID {target_atlas_uid} for applying RT alignment model...")
        atlas_obj = Atlas.from_database(rt_align_obj.paths['main_db_path'] , target_atlas_uid)

        # Copy each CompoundMZRT and overwrite only the RT fields
        all_rt_shifts = []
        per_compound_rt_shifts = []
        aligned_compound_mzrts = {}
        for mz_rt_uid, comp_ref in atlas_obj.compound_mzrts.items():
            aligned_rt_peak = float(apply_rt_model([comp_ref.rt_peak], rt_align_obj.rt_alignment_model)[0])
            if rt_align_obj.rt_alignment_params['apply_model_to_min_max']:
                aligned_rt_min = float(apply_rt_model([comp_ref.rt_min], rt_align_obj.rt_alignment_model)[0])
                aligned_rt_max = float(apply_rt_model([comp_ref.rt_max], rt_align_obj.rt_alignment_model)[0])
            else:
                window = comp_ref.rt_max - comp_ref.rt_min
                aligned_rt_min = aligned_rt_peak - window / 2
                aligned_rt_max = aligned_rt_peak + window / 2

            rt_shift = aligned_rt_peak - comp_ref.rt_peak
            all_rt_shifts.append(rt_shift)
            per_compound_rt_shifts.append({
                'compound_name': comp_ref.compound_name,
                'inchi_key': comp_ref.inchi_key,
                'adduct': comp_ref.adduct,
                'mz_rt_uid': mz_rt_uid,
                'original_rt_peak': float(comp_ref.rt_peak),
                'aligned_rt_peak': aligned_rt_peak,
                'rt_shift': rt_shift,
            })

            # Copy the compound_mzrt entry and update RT fields and UID
            new_comp_mzrt = copy.copy(comp_ref)
            new_mz_rt_uid = _generate_uid("mz_rt", decorator="rta")
            new_comp_mzrt.mz_rt_uid = new_mz_rt_uid
            new_comp_mzrt.rt_peak = aligned_rt_peak
            new_comp_mzrt.rt_min = aligned_rt_min
            new_comp_mzrt.rt_max = aligned_rt_max
            aligned_compound_mzrts[new_mz_rt_uid] = new_comp_mzrt

        # Copy the atlas and overwrite only the fields that change
        aligned_atlas = copy.copy(atlas_obj)
        aligned_atlas_uid = _generate_uid("rt_atlas", decorator=f"{analysis_type.lower()}-{chrom.lower()}-{pol.lower()}-{analysis_name.lower()}")
        aligned_atlas.atlas_uid = aligned_atlas_uid
        aligned_atlas.atlas_name = f"{atlas_obj.atlas_name} (post-rt-alignment)"
        aligned_atlas.atlas_description = f"{atlas_obj.atlas_description} (post-rt-alignment)"
        aligned_atlas.chromatography = chrom
        aligned_atlas.polarity = pol
        aligned_atlas.analysis_type = analysis_type
        aligned_atlas.atlas_type = "RT-ALIGNED"
        aligned_atlas.rt_alignment_number = rt_align_obj.rt_alignment_number
        aligned_atlas.analysis_number = None
        aligned_atlas.analysis_name = analysis_name
        aligned_atlas.compound_mzrts = aligned_compound_mzrts
        aligned_atlas.source_atlas_uid = atlas_obj.atlas_uid
        aligned_atlas.created_by = get_provenance()["analyst"]
        aligned_atlas.created_date = get_provenance()["timestamp"]
        
        aligned_atlases[aligned_atlas_uid] = aligned_atlas

        # Save and print RT shift stats per-compound for this atlas
        save_rt_aligned_stats(
            all_rt_shifts=all_rt_shifts,
            per_compound_rt_shifts=per_compound_rt_shifts,
            aligned_atlas_uid=aligned_atlas_uid,
            output_dir=rt_align_obj.paths['rt_alignment_results_dir'],
        )

    rt_align_obj.rt_aligned_atlases = aligned_atlases

    save_aligned_atlases_to_db(rt_align_obj)

    return

def save_aligned_atlases_to_db(rt_align_obj: "RTAlignment") -> None:
    for uid, aligned_atlas_obj in rt_align_obj.rt_aligned_atlases.items():
        logger.info(f"Saving aligned Atlas {uid} to database...")
        save_atlas_to_database(
            atlas_obj=aligned_atlas_obj,
            db_path=rt_align_obj.paths['project_db_path'],
            main_db_path=rt_align_obj.paths['main_db_path']
        )
        logger.info(f"Saving aligned Atlas {uid} metadata to database...")
        register_workflow_run(
            project_db_path=rt_align_obj.paths['project_db_path'],
            rt_alignment_number=rt_align_obj.rt_alignment_number,
            analysis_number=None,
            atlas_obj=aligned_atlas_obj,
            stage='RT_ALIGNED',
        )
        logger.info(f"Saving aligned Atlas {uid} data to TSV...")
        ldt.save_atlas_data_to_tsv(
            atlas_obj=aligned_atlas_obj,
            output_path=rt_align_obj.paths['rt_alignment_results_dir']
        )

def save_rt_aligned_stats(
    all_rt_shifts: list[float], 
    per_compound_rt_shifts: dict[str, float],
    aligned_atlas_uid: str,
    output_dir: str
) -> None:
    rt_shift_stats = {}
    if all_rt_shifts:
        rt_shift_stats = {
            'rt_shift_min': float(np.min(all_rt_shifts)),
            'rt_shift_max': float(np.max(all_rt_shifts)),
            'rt_shift_median': float(np.median(all_rt_shifts)),
        }

    # Save per-compound RT shift data with summary stats at the top
    rt_shift_output = {
        'stats': rt_shift_stats,
        'compounds': per_compound_rt_shifts,
    }
    rt_shift_stats_path = Path(output_dir) / f"{aligned_atlas_uid}_rt_shift_stats.json"
    with open(rt_shift_stats_path, "w") as f:
        json.dump(rt_shift_output, f, indent=4)

def to_python_type(val):
    if isinstance(val, np.generic):
        return val.item()
    return val
 
def write_curation_updates_to_db(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int,
    rows: list[dict],  # Each dict must have 'mz_rt_uid' and the fields to update
    updated_field_keys: list[str]  # Ordered list of field names to update
) -> None:
    """
    Batch update multiple manual curation entries in a single transaction.
    All rows must share the same set of fields to update.

    Args:
        project_db_path: Path to the project database
        rt_alignment_number: RT alignment number
        analysis_number: Analysis number
        rows: List of dicts, each containing 'mz_rt_uid' and values for updated_field_keys
        updated_field_keys: Ordered list of field names to update (must be consistent across all rows)
    """
    if not rows or not updated_field_keys:
        return

    rt_alignment_number = to_python_type(rt_alignment_number)
    analysis_number = to_python_type(analysis_number)
    set_clause = ", ".join([f"{k} = ?" for k in updated_field_keys])
    sql = (
        f"UPDATE manual_curation SET {set_clause} "
        f"WHERE mz_rt_uid = ? AND rt_alignment_number = ? AND analysis_number = ?"
    )

    try:
        params_list = []
        for row in rows:
            mz_rt_uid = to_python_type(row["mz_rt_uid"])
            field_values = [to_python_type(row[k]) for k in updated_field_keys]
            params_list.append(field_values + [mz_rt_uid, rt_alignment_number, analysis_number])

        with get_db_connection(project_db_path, max_retries=5, initial_retry_delay=0.5) as conn:
            conn.executemany(sql, params_list)

    except Exception as e:
        logger.error(
            f"Error batch updating manual curation entries "
            f"RTA{rt_alignment_number}, TGA{analysis_number}: {e}"
        )
        raise ValueError(
            f"Failed to batch update manual curation entries "
            f"RTA{rt_alignment_number}, TGA{analysis_number}. See logs for details."
        )

def display_auto_id_summary(auto_id_obj: "AutoIdentification") -> None:
    """
    Display a summary table of auto identification results to the logger.
    Shows number of compounds, MS1/MS2 datapoints, MS2 hits, etc.
    """

    summary_rows = []
    for ci in auto_id_obj.experimental_data.curation_df:
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

    logger.info("Loading experimental data from AutoIdentification object...")
    project_db_path = auto_id_obj.paths['project_db_path']
    dataset = auto_id_obj.experimental_data
    curation_df = dataset.curation_df
    ms1_df = dataset.ms1_df
    ms2_df = dataset.ms2_df
    
    constants = {
        'rt_alignment_number': auto_id_obj.rt_alignment_number,
        'analysis_number': auto_id_obj.analysis_number,
        'analysis_type': auto_id_obj.pre_autoid_atlas_obj.analysis_type,
        'created_by': get_provenance()["analyst"],
        'created_date': get_provenance()["timestamp"]
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

    Returns a DataFrame with columns: compound_name, inchi_key, adduct,
    atlas_rt_peak, rt_peak, rt_min, rt_max, ms1_notes, ms2_notes, other_notes,
    analyst_notes.  Returns an empty DataFrame (no error) when no matching ISTD
    run is found.
    """
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute("""
                SELECT compound_name, inchi_key, adduct, atlas_rt_peak,
                       rt_peak, rt_min, rt_max,
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


def get_opposite_polarity_curation(
    project_db_path: str,
    analysis_type: str,
    polarity: str,
    rt_alignment_number: int,
    analysis_number: int,
) -> pd.DataFrame:
    """
    Retrieve manual_curation entries for the *opposite* polarity that have been
    manually curated (passed_curation = 1), matching the same analysis_type,
    rt_alignment_number, and analysis_number.

    Matching across polarities is done on (compound_name, inchi_key, atlas_rt_peak)
    since adducts differ between positive and negative mode.  Returns a DataFrame
    with columns: compound_name, inchi_key, atlas_rt_peak, rt_peak, rt_min, rt_max,
    ms1_notes, ms2_notes, other_notes, analyst_notes.  Returns an empty DataFrame
    (no error) when no matching entries are found.
    """
    opposite = "negative" if polarity.lower() == "positive" else "positive"
    try:
        with get_db_connection(project_db_path, read_only=True) as conn:
            df = conn.execute("""
                SELECT compound_name, inchi_key, atlas_rt_peak,
                       rt_peak, rt_min, rt_max,
                       ms1_notes, ms2_notes, other_notes, analyst_notes
                FROM manual_curation
                WHERE analysis_type = ?
                  AND polarity = ?
                  AND rt_alignment_number = ?
                  AND analysis_number = ?
                  AND passed_curation = 1
            """, [analysis_type, opposite, rt_alignment_number, analysis_number]).df()
    except Exception as e:
        logger.warning(f"Could not retrieve opposite-polarity curation entries: {e}")
        df = pd.DataFrame()
    if df.empty:
        logger.info(
            f"No curated {opposite}-polarity {analysis_type} entries found for "
            f"RTA{rt_alignment_number}, TGA{analysis_number}."
        )
    else:
        logger.info(
            f"Found {len(df)} curated {opposite}-polarity {analysis_type} entries for "
            f"RTA{rt_alignment_number}, TGA{analysis_number}."
        )
    return df


def apply_cross_polarity_curation(
    manual_curation_df: pd.DataFrame,
    opposite_polarity_df: pd.DataFrame,
    post_autoid_atlas_obj,
) -> tuple:
    """
    Apply RT bounds (and notes) from opposite-polarity curated entries to
    matching compounds in the current analysis.

    Matching is on (compound_name, inchi_key, atlas_rt_peak) since adducts
    differ across polarities.  Only updates rows that have not yet been manually
    curated (ms2_notes == '').  Also updates RT bounds on the
    post_autoid_atlas_obj CompoundMZRT objects (keyed by mz_rt_uid) so that EIC
    plot windows reflect the transferred values when the GUI opens.

    Returns (updated_manual_curation_df, n_transferred).
    """
    if opposite_polarity_df.empty:
        return manual_curation_df, 0

    # Build lookup keyed on (compound_name, inchi_key, atlas_rt_peak).
    # Adducts differ across polarities so they are excluded from the key.
    opp_lookup = {}
    for _, row in opposite_polarity_df.iterrows():
        key = (row["compound_name"], row["inchi_key"], row["atlas_rt_peak"])
        if key not in opp_lookup:
            opp_lookup[key] = row

    # Atlas is keyed by mz_rt_uid — use that as the unique identifier.
    compound_mzrts = post_autoid_atlas_obj.compound_mzrts

    df = manual_curation_df.copy()
    n_transferred = 0

    for idx, row in df.iterrows():
        key = (row.get("compound_name"), row["inchi_key"], row.get("atlas_rt_peak"))
        if key not in opp_lookup:
            continue

        ms2_note_val = row.get("ms2_notes", "")
        if ms2_note_val != "":
            logger.info(
                f"Skipping cross-polarity RT transfer for {row['inchi_key']} / {row.get('adduct')} "
                f"because ms2_notes already has value: '{ms2_note_val}'"
            )
            continue  # already curated — do not overwrite

        opp_row = opp_lookup[key]
        df.at[idx, "rt_peak"]       = opp_row["rt_peak"]
        df.at[idx, "rt_min"]        = opp_row["rt_min"]
        df.at[idx, "rt_max"]        = opp_row["rt_max"]
        df.at[idx, "ms1_notes"]     = opp_row["ms1_notes"]
        df.at[idx, "ms2_notes"]     = opp_row["ms2_notes"]
        df.at[idx, "analyst_notes"] = opp_row["analyst_notes"]
        df.at[idx, "other_notes"]   = opp_row["other_notes"]

        uid = row.get("mz_rt_uid")
        if uid and uid in compound_mzrts:
            compound_mzrts[uid].rt_peak = float(opp_row["rt_peak"])
            compound_mzrts[uid].rt_min  = float(opp_row["rt_min"])
            compound_mzrts[uid].rt_max  = float(opp_row["rt_max"])

        n_transferred += 1

    return df, n_transferred


def apply_istd_curation_to_ema(
    manual_curation_df: pd.DataFrame,
    istd_df: pd.DataFrame,
    post_autoid_atlas_obj,
) -> tuple:
    """
    Apply ISTD curation (RT bounds + notes) to matching EMA compounds.

    Only updates EMA rows that have not yet been manually curated
    (ms2_notes == '').  Matching is on (compound_name, inchi_key, adduct,
    atlas_rt_peak) — a composite key that uniquely identifies the same compound
    entry across ISTD and EMA atlases within the same polarity.

    Also updates RT bounds on the post_autoid_atlas_obj CompoundMZRT objects
    (keyed by mz_rt_uid) so that EIC plot windows reflect the curated values
    when the GUI opens.

    Returns (updated_manual_curation_df, n_transferred).
    """
    if istd_df.empty:
        return manual_curation_df, 0

    # Source lookup: ISTD rows keyed by (compound_name, inchi_key, adduct, atlas_rt_peak).
    istd_lookup = {
        (row["compound_name"], row["inchi_key"], row["adduct"], row["atlas_rt_peak"]): row
        for _, row in istd_df.iterrows()
    }

    # Atlas is keyed by mz_rt_uid — use that as the unique identifier.
    compound_mzrts = post_autoid_atlas_obj.compound_mzrts

    df = manual_curation_df.copy()
    n_transferred = 0

    for idx, row in df.iterrows():
        key = (row.get("compound_name"), row["inchi_key"], row["adduct"], row.get("atlas_rt_peak"))
        if key not in istd_lookup:
            continue
        
        ms2_note_val = row.get("ms2_notes", "")
        if ms2_note_val != "":
            logger.info(f"Skipping ISTD info transfer to EMA for {key} because ms2_notes already has value: '{ms2_note_val}'")
            continue  # already curated — do not overwrite

        istd_row = istd_lookup[key]
        df.at[idx, "rt_peak"]       = istd_row["rt_peak"]
        df.at[idx, "rt_min"]        = istd_row["rt_min"]
        df.at[idx, "rt_max"]        = istd_row["rt_max"]
        df.at[idx, "ms1_notes"]     = istd_row["ms1_notes"]
        df.at[idx, "ms2_notes"]     = istd_row["ms2_notes"]
        df.at[idx, "analyst_notes"] = istd_row["analyst_notes"]
        df.at[idx, "other_notes"]   = istd_row["other_notes"]

        uid = row.get("mz_rt_uid")
        if uid and uid in compound_mzrts:
            compound_mzrts[uid].rt_peak = float(istd_row["rt_peak"])
            compound_mzrts[uid].rt_min  = float(istd_row["rt_min"])
            compound_mzrts[uid].rt_max  = float(istd_row["rt_max"])

        n_transferred += 1

    return df, n_transferred

def _transfer_istd_curation(ema_df, db_path, gui_obj, atlas_obj):
    """Vectorized ISTD -> EMA transfer."""
    logger.info("Finding ISTD curation entries to transfer to EMA...")
    with get_db_connection(db_path, read_only=True) as conn:
        istd_df = conn.execute("""
            SELECT mz_rt_uid, inchi_key, adduct, rt_peak, rt_min, rt_max, 
                   passed_curation, ms1_notes, ms2_notes, other_notes, analyst_notes
            FROM manual_curation 
            WHERE analysis_type = 'ISTD' AND polarity = ? AND rt_alignment_number = ? AND analysis_number = ? AND passed_curation = 1
        """, [atlas_obj.polarity, gui_obj.rt_alignment_number, gui_obj.analysis_number]).df()

    if istd_df.empty: 
        logger.info("No ISTD curation entries found to transfer to EMA.")
        return ema_df

    logger.info(f"Merging ISTD curation with EMA for {len(istd_df)} compounds...")
    merged = ema_df.merge(istd_df, on='mz_rt_uid', how='left', suffixes=('', '_istd'))
    
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
    if not isinstance(override_parameters.get("apply_istd_curation_to_ema"), (type(None), bool)):
        raise ValueError("override_parameters['apply_istd_curation_to_ema'] must be a boolean or None")
    if not isinstance(override_parameters.get("apply_cross_polarity_curation"), (type(None), bool)):
        raise ValueError("override_parameters['apply_cross_polarity_curation'] must be a boolean or None")
    if not isinstance(override_parameters.get("remove_flagged_compounds"), (type(None), bool)):
        raise ValueError("override_parameters['remove_flagged_compounds'] must be a boolean or None")
    if not isinstance(override_parameters.get("gui_top_n_hits"), (type(None), int)):
        raise ValueError("override_parameters['gui_top_n_hits'] must be an integer or None")
    if not isinstance(override_parameters.get("upload_to_gdrive"), (type(None), bool)):
        raise ValueError("override_parameters['upload_to_gdrive'] must be a boolean or None")
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
    remove_flagged_compounds: bool = True,
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
                query += " AND passed_autoid = TRUE"
            if remove_flagged_compounds:
                query += " AND passed_curation = TRUE"
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

def load_and_filter_for_summary(summary_obj, update_raw_in_feature=False):
    """Load MS1/MS2/curation data, apply curated RT bounds to in_feature, and
    populate summary_obj.experimental_data with filtered DataFrames.

    Args:
        summary_obj: The summary workflow object.
        update_raw_in_feature: If True, persist the recalculated in_feature
            values back to the ms1_data and ms2_data database tables (a
            DELETE + INSERT per table).  If False (default), in_feature is
            updated only in memory — the DB tables are left unchanged.  The
            default is False because the in_feature values are always derived
            from manual_curation.rt_min/rt_max at load time, so writing them
            back to the raw tables is redundant for normal summary runs.
    """
    from metatlas2.workflow_objects import ExperimentalData

    atlas_uids = list(summary_obj.post_curation_atlas_obj.compound_mzrts.keys())
    logger.info(f"Connecting to the database and loading MS1 and MS2 data for {len(atlas_uids)} compounds...")
    with get_db_connection(summary_obj.paths["project_db_path"], read_only=True) as conn:
        uid_filter = f"mz_rt_uid IN {tuple(atlas_uids)}" if atlas_uids else "1=0"
        
        curation_df = conn.execute(f"""
            SELECT * FROM manual_curation
            WHERE rt_alignment_number = {summary_obj.rt_alignment_number}
            AND analysis_number = {summary_obj.analysis_number}
            AND {uid_filter}
        """).df()
        
        ms1_df = conn.execute(f"""
            SELECT * FROM ms1_data
            WHERE rt_alignment_number = {summary_obj.rt_alignment_number}
            AND analysis_number = {summary_obj.analysis_number}
            AND {uid_filter}
        """).df()
        
        ms2_df = conn.execute(f"""
            SELECT * FROM ms2_data
            WHERE rt_alignment_number = {summary_obj.rt_alignment_number}
            AND analysis_number = {summary_obj.analysis_number}
            AND {uid_filter}
        """).df()

    if not ms2_df.empty and "hits" in ms2_df.columns:
        ms2_df = ms2_df.copy()
        ms2_df["hits"] = ms2_df["hits"].apply(_deserialize_hits_value)

    logger.info(f"Loaded {len(curation_df)} manual curation entries,"
                f" {len(ms1_df)} MS1 data points,"
                f" and {len(ms2_df)} MS2 data points for summary.")

    # Create RT bounds lookup for efficient access
    rt_bounds = {
        row['mz_rt_uid']: (row['rt_min'], row['rt_max'])
        for _, row in curation_df.iterrows()
    }

    # Update in_feature for MS1 data (always done in memory)
    logger.info(f"Updating in_feature for MS1 data based on manual curation RT bounds...")
    def update_ms1_in_feature(row):
        bounds = rt_bounds.get(row['mz_rt_uid'])
        if bounds is None:
            raise ValueError(f"No RT bounds found for mz_rt_uid {row['mz_rt_uid']} in curation_df.")
        rt_min, rt_max = bounds
        return [bool(rt_min <= rt <= rt_max) for rt in row['spec_rts']]

    ms1_df['in_feature'] = ms1_df.apply(update_ms1_in_feature, axis=1)

    # Update in_feature for MS2 data (always done in memory)
    logger.info(f"Updating in_feature for MS2 data based on manual curation RT bounds...")
    if not ms2_df.empty:
        rt_min_series = ms2_df['mz_rt_uid'].map(lambda uid: rt_bounds.get(uid, (None, None))[0])
        rt_max_series = ms2_df['mz_rt_uid'].map(lambda uid: rt_bounds.get(uid, (None, None))[1])
        has_bounds = rt_min_series.notna()
        ms2_df['in_feature'] = np.where(
            has_bounds,
            (ms2_df['scan_rt'] >= rt_min_series) & (ms2_df['scan_rt'] <= rt_max_series),
            ms2_df['in_feature']
        )
        ms2_df['in_feature'] = ms2_df['in_feature'].astype(bool)

    # Optionally persist the updated in_feature values back to the raw DB tables.
    if update_raw_in_feature:
        logger.info("Saving updated in_feature values back to database (update_raw_in_feature=True)...")
        with get_db_connection(summary_obj.paths["project_db_path"]) as conn:
            # Delete old entries and insert updated ones for MS1
            if not ms1_df.empty:
                logger.info("  Saving MS1 data entries to database...")
                conn.execute(f"""
                    DELETE FROM ms1_data
                    WHERE rt_alignment_number = {summary_obj.rt_alignment_number}
                    AND analysis_number = {summary_obj.analysis_number}
                    AND {uid_filter}
                """)
                conn.execute("INSERT INTO ms1_data SELECT * FROM ms1_df")
            
            # Delete old entries and insert updated ones for MS2
            if not ms2_df.empty:
                logger.info("  Saving MS2 data entries to database...")
                # Re-serialize hits for database storage
                ms2_df_to_save = ms2_df.copy()
                ms2_df_to_save["hits"] = ms2_df_to_save["hits"].apply(_serialize_hits_value)
                
                conn.execute(f"""
                    DELETE FROM ms2_data
                    WHERE rt_alignment_number = {summary_obj.rt_alignment_number}
                    AND analysis_number = {summary_obj.analysis_number}
                    AND {uid_filter}
                """)
                conn.execute("INSERT INTO ms2_data SELECT * FROM ms2_df_to_save")
        logger.info("Updated in_feature values saved to database.")
    else:
        logger.info("Skipping DB write of in_feature (update_raw_in_feature=False); values updated in memory only.")

    # Create filtered copies with only in_feature=True data
    logger.info(f"Creating filtered copies with only in_feature=True data...")
    if not ms1_df.empty:
        def filter_ms1_row(row):
            mask = row['in_feature']
            if isinstance(mask, list) and isinstance(row['spec_rts'], list):
                filtered = {k: row[k] for k in ms1_df.columns}
                filtered['spec_rts'] = [v for v, m in zip(row['spec_rts'], mask) if m]
                filtered['spec_mzs'] = [v for v, m in zip(row['spec_mzs'], mask) if m]
                filtered['spec_ints'] = [v for v, m in zip(row['spec_ints'], mask) if m]
                filtered['in_feature'] = [True] * len(filtered['spec_rts'])
                return filtered
            return row.to_dict()

        ms1_df_filtered = pd.DataFrame(ms1_df.apply(filter_ms1_row, axis=1).tolist())

    if not ms2_df.empty:
        ms2_df_filtered = ms2_df[ms2_df['in_feature'] == True].copy()
    else:
        ms2_df_filtered = ms2_df.copy()

    # Store filtered dataframes in experimental_data object
    summary_obj.experimental_data = ExperimentalData()
    summary_obj.experimental_data.curation_df = curation_df
    summary_obj.experimental_data.ms1_df = ms1_df_filtered
    summary_obj.experimental_data.ms2_df = ms2_df_filtered
    logger.info(f"Summary object experimental_data populated with {len(curation_df)} curation entries")
    logger.info(f"Added {len(ms1_df_filtered)} MS1 rows ({len(ms1_df_filtered)/len(ms1_df)*100:.1f} percent of total) and {len(ms2_df_filtered)} MS2 rows ({len(ms2_df_filtered)/len(ms2_df)*100:.1f} percent of total) with in_feature=True to the summary object.")

    return

def load_and_filter_for_gui(analysis_gui_obj):

    from metatlas2.workflow_objects import ExperimentalData

    atlas_uids = list(analysis_gui_obj.post_autoid_atlas_obj.compound_mzrts.keys())
    logger.info(f"Connecting to the database and loading MS1 and MS2 data for {len(atlas_uids)} compounds...")
    with get_db_connection(analysis_gui_obj.paths["project_db_path"], read_only=True) as conn:
        uid_filter = f"mz_rt_uid IN {tuple(atlas_uids)}" if atlas_uids else "1=0"
        
        logger.info("Loading manual curation table from database...")
        curation_df = conn.execute(f"""
            SELECT * FROM manual_curation 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()
        logger.info(f"Loaded {len(curation_df)} manual curation entries ({curation_df['mz_rt_uid'].nunique()}) unique compounds) for GUI display.")
        
        logger.info("Loading MS1 data from database...")
        ms1_df = conn.execute(f"""
            SELECT * FROM ms1_data 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()
        logger.info(f"Loaded {len(ms1_df)} MS1 data points ({ms1_df['mz_rt_uid'].nunique()} unique compounds) for GUI display.")
        
        logger.info("Loading MS2 data from database...")
        ms2_df = conn.execute(f"""
            SELECT * FROM ms2_data 
            WHERE rt_alignment_number = {analysis_gui_obj.rt_alignment_number} 
            AND analysis_number = {analysis_gui_obj.analysis_number}
            AND {uid_filter}
        """).df()
        if not ms2_df.empty and "hits" in ms2_df.columns:
            ms2_df = ms2_df.copy()
            ms2_df["hits"] = ms2_df["hits"].apply(_deserialize_hits_value)
        logger.info(f"Loaded {len(ms2_df)} MS2 data points ({ms2_df['mz_rt_uid'].nunique()} unique compounds) for GUI display.")

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
        logger.info(f"Applying MS1 override filters: min_num_points={ms1_min_pts}, min_peak_intensity={ms1_min_int}...")
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
        logger.info(f"Applying MS2 override filters: min_score={ms2_min_score}, min_matching_frags={ms2_min_frags}...")
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
        logger.info(f"Removing unidentified compounds from curation object (those that do not have passing MS1 or MS2 data)...")
        keep_mask = (curation_df['mz_rt_uid'].isin(passing_ms1) | curation_df['mz_rt_uid'].isin(passing_ms2))
        curation_df = curation_df[keep_mask].copy()
        surviving_uids = set(curation_df['mz_rt_uid'])
        ms1_df = ms1_df[ms1_df['mz_rt_uid'].isin(surviving_uids)]
        ms2_df = ms2_df[ms2_df['mz_rt_uid'].isin(surviving_uids)]    

    # ISTD transfer
    apply_istd_override = analysis_gui_obj.workflow_params.get("apply_istd_curation_to_ema", True)
    if _overrides.get("apply_istd_curation_to_ema") is not None:
        apply_istd_override = _overrides.get("apply_istd_curation_to_ema", None)
    if apply_istd_override:
        if analysis_gui_obj.post_autoid_atlas_obj.analysis_type == "EMA":
            logger.info("Applying ISTD transfer...")
            curation_df = _transfer_istd_curation(
                curation_df, analysis_gui_obj.paths["project_db_path"],
                analysis_gui_obj, analysis_gui_obj.post_autoid_atlas_obj
            )

    # Cross-polarity RT transfer: apply curated RT bounds from the opposite polarity
    # (same analysis_type, rt_alignment_number, analysis_number) to uncurated compounds.
    apply_cross_polarity = analysis_gui_obj.workflow_params.get("apply_cross_polarity_curation", True)
    if _overrides.get("apply_cross_polarity_curation") is not None:
        apply_cross_polarity = _overrides.get("apply_cross_polarity_curation")
    if apply_cross_polarity:
        atlas_obj = analysis_gui_obj.post_autoid_atlas_obj
        logger.info(
            f"Checking for curated opposite-polarity {atlas_obj.analysis_type} entries "
            f"to pre-populate RT bounds..."
        )
        opp_polarity_df = get_opposite_polarity_curation(
            project_db_path=analysis_gui_obj.paths["project_db_path"],
            analysis_type=atlas_obj.analysis_type,
            polarity=atlas_obj.polarity,
            rt_alignment_number=analysis_gui_obj.rt_alignment_number,
            analysis_number=analysis_gui_obj.analysis_number,
        )
        if not opp_polarity_df.empty:
            curation_df, n_cross = apply_cross_polarity_curation(
                manual_curation_df=curation_df,
                opposite_polarity_df=opp_polarity_df,
                post_autoid_atlas_obj=atlas_obj,
            )
            logger.info(
                f"Transferred opposite-polarity RT bounds to {n_cross} compound(s) "
                f"(inchi_key matches, ms2_notes was empty)."
            )
        else:
            logger.info("No curated opposite-polarity entries found — skipping cross-polarity RT transfer.")

    # Assign to GUI
    logger.info(f"Assigning filtered data to GUI experimental_data object...")
    analysis_gui_obj.experimental_data = ExperimentalData()
    analysis_gui_obj.experimental_data.curation_df = curation_df
    analysis_gui_obj.experimental_data.ms1_df = ms1_df
    analysis_gui_obj.experimental_data.ms2_df = ms2_df

    return