import pandas as pd
import numpy as np
import duckdb
import uuid
import sys
import os
import json
import sys
from pathlib import Path
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

def _generate_uid(entity_type: str, decorator: str = None) -> str:
    """Generate a unique identifier for database entities."""
    if entity_type == "ref_atlas":
        return f"atl-ref-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-ref-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_atlas":
        return f"atl-rta-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-rta-{uuid.uuid4().hex[:32]}"
    elif entity_type == "analyzed_atlas":
        return f"atl-tga-{decorator}-{uuid.uuid4().hex[:32]}" if decorator else f"atl-tga-{uuid.uuid4().hex[:32]}"
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
        return f"cmp-info-{uuid.uuid4().hex[:32]}"
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
    from workflow_objects import Atlas, CompoundMZRT

    if not chromatography:
        chromatography = ldt.detect_atlas_input_chromatography(atlas_df)
    if not polarity:
        polarity = ldt.detect_atlas_input_polarity(atlas_df)        

    inchi_keys = atlas_df['inchi_key'].dropna().unique().tolist()
    compound_lookup = get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)

    compound_mzrts = {}
    for _, row in atlas_df.iterrows():
        inchi_key = row.get('inchi_key', '')
        if not inchi_key or inchi_key not in compound_lookup:
            logger.warning(f"Compound with inchi_key {inchi_key} missing from metatlas database, skipping.")
            continue
        compound_uid = compound_lookup[inchi_key]
        rt_peak = row.get('rt_peak', None)
        rt_min = row.get('rt_min', rt_peak - 0.5)
        rt_max = row.get('rt_max', rt_peak + 0.5)
        mz = row.get('mz', None)
        mz_tolerance = row.get('mz_tolerance', 5.0)
        adduct = str(row.get('adduct', None))
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
            inchi_key=inchi_key,
            rt_peak=rt_peak,
            rt_min=rt_min,
            rt_max=rt_max,
            mz=mz,
            mz_tolerance=mz_tolerance,
            adduct=adduct,
            chromatography=chromatography,
            polarity=polarity,
            confidence=confidence_level,
            identification_notes=identification_notes,
            source=atlas_file_path
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
    with get_db_connection(database_path) as conn:

        # Attach main database if needed for compound metadata
        if main_db_path:
            try:
                conn.execute(f"ATTACH '{main_db_path}' AS main_db")
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
                        mzrt.rt_alignment_applied,
                        mzrt.manual_curation_applied
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
                        mzrt.rt_alignment_applied,
                        mzrt.manual_curation_applied
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

    display(df.head())
    return df

def create_project_database(project_db_path: str, overwrite: bool = False) -> None:
    """
    Create project-specific database with required tables.
    Never overwrites existing databases - requires analyst to increment analysis number.
    """
    project_db_path = Path(project_db_path)

    if project_db_path.exists() and not overwrite:
        logger.info(f"Project database already exists at {project_db_path}. Not overwriting.")
        return
    elif project_db_path.exists() and overwrite:
        logger.warning(f"Overwriting existing project database at {project_db_path} (overwrite=True)")
        project_db_path.unlink()
        logger.info(f"Deleted existing database at {project_db_path}")
    elif not project_db_path.exists():
        logger.info(f"No existing project database found at {project_db_path}. Creating new database.")

    with get_db_connection(project_db_path) as conn:
        _create_database_tables(conn, db_type="project")
    logger.info(f"Project database created at {project_db_path}")
    return

def create_metatlas_database(db_path: str, overwrite: bool = False) -> None:
    """
    Create main metatlas database with required tables.
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
        raise ValueError(f"Database already exists at {db_path}. Use overwrite=True to replace it.")
    elif not overwrite and not db_path.exists():
        logger.info(f"No existing main database found at {db_path}. Creating new database.")

    with get_db_connection(db_path) as conn:
        _create_database_tables(conn, db_type="main")

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
    prov = ldt.get_provenance()

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

    with get_db_connection(project_db_path) as conn:
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
        logger.error(f"Database not found at {db_path}. Check path or create it first!")
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
                INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        prov = ldt.get_provenance()
        
        return (
            _generate_uid("mz_rt", decorator="ref"),
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
            reference_data.get('identification_notes', ''),
            reference_data.get('source', 'Unknown'),
            reference_data.get('rt_alignment_applied', False),
            reference_data.get('manual_curation_applied', False),
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
    with get_db_connection(project_db_path) as conn:
        prov = ldt.get_provenance()

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

    from workflow_objects import LCMSRun

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
    with get_db_connection(project_db_path) as conn:
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
                    INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                mz REAL,
                mz_tolerance REAL,
                adduct TEXT,
                chromatography TEXT,
                polarity TEXT,
                confidence TEXT,
                identification_notes TEXT,
                source TEXT,
                rt_alignment_applied BOOL,
                manual_curation_applied BOOL,
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
                rt_peak REAL,
                rt_min REAL,
                rt_max REAL,
                mz REAL,
                mz_tolerance REAL,
                adduct TEXT,
                chromatography TEXT,
                polarity TEXT,
                confidence TEXT,
                identification_notes TEXT,
                source TEXT,
                rt_alignment_applied BOOL,
                manual_curation_applied BOOL,
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
                file_path TEXT,
                rt REAL,
                mz REAL,
                i REAL,
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
                analyst_notes TEXT,
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
            formula TEXT,
            polarity TEXT,
            chromatography TEXT,
            mz_tolerance REAL,
            atlas_mz REAL,
            atlas_rt_peak REAL,
            atlas_rt_min REAL,
            atlas_rt_max REAL,
            original_rt_peak REAL,
            original_rt_min REAL,
            original_rt_max REAL,
            rt_peak REAL,
            rt_min REAL,
            rt_max REAL,
            identification_notes TEXT,
            ms1_notes TEXT DEFAULT 'keep',
            ms2_notes TEXT DEFAULT 'no selection',
            analyst_notes TEXT,
            is_rt_modified BOOLEAN DEFAULT FALSE,
            is_annotation_modified BOOLEAN DEFAULT FALSE,
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
    prov = ldt.get_provenance()
    with get_db_connection(db_path) as conn:
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
                        INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mz_rt_uid,
                        compound_mzrt.compound_uid,
                        compound_mzrt.rt_peak,
                        compound_mzrt.rt_min,
                        compound_mzrt.rt_max,
                        compound_mzrt.mz,
                        compound_mzrt.mz_tolerance,
                        compound_mzrt.adduct,
                        compound_mzrt.chromatography,
                        compound_mzrt.polarity,
                        compound_mzrt.confidence,
                        compound_mzrt.identification_notes,
                        compound_mzrt.source,
                        'False',  # rt_alignment_applied
                        'False',  # manual_curation_applied
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

            # Attach main database for compound verification
            conn.execute(f"ATTACH '{main_db_path}' AS main_db")

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
                        INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        mz_rt_uid,
                        compound_mzrt.compound_uid,
                        compound_mzrt.rt_peak,
                        compound_mzrt.rt_min,
                        compound_mzrt.rt_max,
                        compound_mzrt.mz,
                        compound_mzrt.mz_tolerance,
                        compound_mzrt.adduct,
                        compound_mzrt.chromatography,
                        compound_mzrt.polarity,
                        compound_mzrt.confidence,
                        compound_mzrt.identification_notes,
                        compound_mzrt.source,
                        compound_mzrt.rt_alignment_applied,
                        compound_mzrt.manual_curation_applied,
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
    with get_db_connection(db_path) as conn:
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
    with get_db_connection(db_path) as conn:
        existing = conn.execute("""
            SELECT mz_rt_uid FROM compound_mzrt
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

    with get_db_connection(project_db_path) as conn:
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

def get_manual_curation_entries(
    project_db_path: str,
    rt_alignment_number: int,
    analysis_number: int
) -> pd.DataFrame:
    """
    Get all manual_curation entries for the given RT alignment and analysis number,
    ordered by ascending RT peak.
    """
    with get_db_connection(project_db_path) as conn:
        df = conn.execute("""
            SELECT *
            FROM manual_curation
            WHERE rt_alignment_number = ? AND analysis_number = ?
            ORDER BY rt_peak ASC
        """, [rt_alignment_number, analysis_number]).df()
    return df

def get_ms1_data_for_compound(
    project_db_path: str,
    inchi_key: str = None,
    adduct: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None
) -> pd.DataFrame:
    """
    Get all MS1 data for a compound (inchi_key+adduct) for plotting EIC.
    If inchi_key or adduct is None, do not filter on that field.
    """
    query = """
        SELECT *
        FROM ms1_data
        WHERE 1=1
    """
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
    query += " ORDER BY file_path, rt"
    with get_db_connection(project_db_path) as conn:
        df = conn.execute(query, params).df()
    return df

def get_ms2_hits_for_compound(
    project_db_path: str,
    inchi_key: str = None,
    adduct: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None,
) -> pd.DataFrame:
    """
    Get MS2 hits for a compound (inchi_key+adduct), ordered by score descending.
    If inchi_key or adduct is None, do not filter on that field.
    """
    query = """
        SELECT *
        FROM ms2_hits
        WHERE 1=1
    """
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
    query += " ORDER BY score DESC"
    with get_db_connection(project_db_path) as conn:
        df = conn.execute(query, params).df()
    return df

def get_ms2_data_for_compound(
    project_db_path: str,
    inchi_key: str = None,
    adduct: str = None,
    rt_alignment_number: int = None,
    analysis_number: int = None
) -> pd.DataFrame:
    """
    Get all MS2 data for a compound (inchi_key+adduct) for plotting query spectrum if no hits.
    If inchi_key or adduct is None, do not filter on that field.
    """
    query = """
        SELECT *
        FROM ms2_data
        WHERE 1=1
    """
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
    query += " ORDER BY file_path, rt"
    with get_db_connection(project_db_path) as conn:
        df = conn.execute(query, params).df()
    return df

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
    exp_data_obj = auto_id_obj.experimental_data

    # Get time and analyst for provenance
    prov = ldt.get_provenance()

    # Get compound_uid mappings
    inchi_keys = [ci.inchi_key for ci in exp_data_obj.manual_curation]
    compound_uid_map = get_compound_uids_by_inchi_keys(main_db_path, inchi_keys)

    # Collect all records
    manual_curation_records = []
    ms2_hits_records = []
    ms1_data_records = []
    ms2_data_records = []

    # CompoundInfo
    for ci in exp_data_obj.manual_curation:
        compound_uid = compound_uid_map.get(ci.inchi_key)
        if not compound_uid:
            logger.warning(f"Could not find compound_uid for {ci.inchi_key}, skipping")
            continue
        metadata_record = _prepare_manual_curation_record(
            ci.data.iloc[0:1], compound_uid, ci.inchi_key, ci.adduct,
            rt_alignment_number, analysis_number, prov
        )
        if metadata_record:
            manual_curation_records.append(metadata_record)

    # MS1Data
    for ms1 in exp_data_obj.ms1_data:
        compound_uid = compound_uid_map.get(ms1.inchi_key)
        if not compound_uid or ms1.data.empty:
            continue
        for _, row in ms1.data.iterrows():
            ms1_data_record = _prepare_ms1_data_record(
                row, compound_uid, ms1.inchi_key, ms1.adduct, ms1.filename,
                rt_alignment_number, analysis_number, prov
            )
            if ms1_data_record:
                ms1_data_records.append(ms1_data_record)

    # MS2Data
    for ms2 in exp_data_obj.ms2_data:
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
                row, compound_uid, ms2.inchi_key, ms2.adduct, rt_alignment_number, analysis_number, ms2.filename, prov
            )
            if ms2_data_record:
                ms2_data_records.append(ms2_data_record)

    # MS2Hits
    for ms2_hit in exp_data_obj.ms2_hits:
        compound_uid = compound_uid_map.get(ms2_hit.inchi_key)
        if not compound_uid or ms2_hit.data.empty:
            continue
        for _, hit in ms2_hit.data.iterrows():
            ms2_hit_record = _prepare_ms2_hit_record(
                hit, compound_uid, ms2_hit.inchi_key, ms2_hit.adduct, ms2_hit.filename,
                rt_alignment_number, analysis_number, prov
            )
            if ms2_hit_record:
                ms2_hits_records.append(ms2_hit_record)

    # Check for identical MS2 summary records before inserting any data
    logger.info("Checking for identical ManualCuration records before database insert...")
    with get_db_connection(project_db_path) as conn:
        _check_identical_manual_curation_exists(conn, manual_curation_records)

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
    
def _check_identical_manual_curation_exists(conn, manual_curation_records: List[tuple]) -> None:
    """
    Check for identical manual curation records in the database before inserting new ones.
    If an identical record exists (same compound_uid, inchi_key, adduct, rt_alignment_number, analysis_number),
    raise a ValueError to prevent duplicate entries.
    """
    for record in manual_curation_records:
        compound_uid = record[1]
        inchi_key = record[2]
        adduct = record[3]
        rt_alignment_number = record[4]
        analysis_number = record[5]

        existing = conn.execute("""
            SELECT curation_uid FROM manual_curation
            WHERE compound_uid = ? AND inchi_key = ? AND adduct = ?
            AND rt_alignment_number = ? AND analysis_number = ?
        """, [
            compound_uid, inchi_key, adduct, rt_alignment_number, analysis_number
        ]).fetchone()

        if existing:
            raise ValueError(
                f"Identical manual curation record already exists for compound_uid {compound_uid}, "
                f"inchi_key {inchi_key}, adduct {adduct}, rt_alignment_number {rt_alignment_number}, "
                f"and analysis_number {analysis_number}. Please incremement the analysis number or run a new RT alignment to avoid duplicates."
            )

def _prepare_manual_curation_record(
    manual_curation: pd.DataFrame,
    compound_uid: str,
    inchi_key: str,
    adduct: str,
    rt_alignment_number: int,
    analysis_number: int,
    prov: Dict
) -> Optional[tuple]:
    """Prepare compound metadata record for database insertion."""
    try:
        row = manual_curation.iloc[0]
        isomers_json = json.dumps(row['isomers']) if row['isomers'] is not None else '[]'
        return (
            _generate_uid("manual_curation"),
            compound_uid,
            inchi_key,
            adduct,
            rt_alignment_number,
            analysis_number,
            row.get('compound_name', ''),
            row.get('formula', ''),
            row.get('polarity', ''),
            row.get('chromatography', ''),
            float(row.get('mz_tolerance', 5.0)),
            float(row.get('atlas_mz', 0.0)),
            float(row.get('atlas_rt_peak', 0.0)),
            float(row.get('atlas_rt_min', 0.0)),
            float(row.get('atlas_rt_max', 0.0)),
            float(row.get('original_rt_peak', 0.0)),
            float(row.get('original_rt_min', 0.0)),
            float(row.get('original_rt_max', 0.0)),
            float(row.get('rt_peak', 0.0)),
            float(row.get('rt_min', 0.0)),
            float(row.get('rt_max', 0.0)),
            row.get('identification_notes', ''),
            row.get('ms1_notes', ''),
            row.get('ms2_notes', ''),
            row.get('analyst_notes', ''),
            bool(row.get('is_rt_modified', False)),
            bool(row.get('is_annotation_modified', False)),
            row.get('best_ms1_file', ''),
            float(row.get('best_ms1_rt', 0.0)),
            float(row.get('best_ms1_mz', 0.0)),
            float(row.get('best_ms1_intensity', 0.0)),
            float(row.get('best_ms1_ppm_error', 0.0)),
            float(row.get('best_ms1_rt_error', 0.0)),
            isomers_json,
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
            filename,
            float(row.get('rt', 0.0)),
            float(row.get('mz', 0.0)),
            float(row.get('i', 0.0)),
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
            ''  # analyst_notes
        )
    except Exception as e:
        logger.error(f"Error preparing MS2 hit record: {e}")
        return None

def _bulk_insert_analysis_data(
    project_db_path: str,
    manual_curation_records: List[tuple],
    ms2_hits_records: List[tuple],
    ms1_data_records: List[tuple],
    ms2_data_records: List[tuple]
) -> None:
    """Perform bulk inserts for all analysis data types."""
    
    with get_db_connection(project_db_path) as conn:
        if manual_curation_records:
            logger.info(f"Inserting {len(manual_curation_records)} compound metadata records...")
            conn.executemany("""
                INSERT INTO manual_curation VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, manual_curation_records)
        
        if ms2_hits_records:
            logger.info(f"Inserting {len(ms2_hits_records)} MS2 hits records...")
            conn.executemany("""
                INSERT INTO ms2_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ms2_hits_records)
        
        if ms1_data_records:
            logger.info(f"Inserting {len(ms1_data_records)} MS1 raw data records...")
            conn.executemany("""
                INSERT INTO ms1_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ms1_data_records)
        
        if ms2_data_records:
            logger.info(f"Inserting {len(ms2_data_records)} MS2 raw data records...")
            conn.executemany("""
                INSERT INTO ms2_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, ms2_data_records)

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

    display(pd.DataFrame(summary_rows))
    return