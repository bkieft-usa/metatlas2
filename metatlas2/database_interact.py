import pandas as pd
import duckdb
import uuid
import sys
import os
import json
from pathlib import Path
from typing import Dict
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.lcmsruns_tools as lrt

def get_atlas_compounds_table(database_path: str, atlas_uid: str) -> pd.DataFrame:
    """
    Extract all compound information for a given atlas UID from the database and return as a pandas DataFrame.
    Handles both main database (with mz_rt_references) and project database (with mz_rt_experimental) structures.
    """
    conn = duckdb.connect(database_path)
    
    # Check if this is a project database by looking for mz_rt_experimental table
    is_project_db = conn.execute("""
        SELECT COUNT(*) 
        FROM information_schema.tables 
        WHERE table_name = 'mz_rt_experimental'
    """).fetchone()[0] > 0
    
    if is_project_db:
        # Project database query - uses experimental data and limited compound info
        query = """
            SELECT
                a.atlas_uid,
                a.atlas_name,
                a.atlas_description,
                a.chromatography,
                a.polarity,
                aca.compound_uid,
                'Compound_' || aca.compound_uid AS compound_name,
                '' AS inchi_key,
                '' AS inchi,
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
    conn.close()

    if df.empty:
        print(f"No compounds found for atlas {atlas_uid}")
        return pd.DataFrame()
    else:
        db_type = "project" if is_project_db else "main"
        print(f"Retrieved {len(df)} compounds from {db_type} database for atlas: {df['atlas_name'].iloc[0]}")
        print(f"Atlas chromatography: {df['chromatography'].iloc[0]}, polarity: {df['polarity'].iloc[0]}")
        if is_project_db and df['rt_correction_applied'].any():
            print(f"RT-corrected atlas detected with experimental data")

    return df

# def get_atlas_compounds_from_db(db_path: str, atlas_uid: str = None) -> pd.DataFrame:
#     """
#     Retrieve compounds and their RT/MZ reference data from database for a specific atlas.
#     Updated to work with both atlas_name and atlas_uid parameters.
#     """
#     conn = duckdb.connect(str(db_path))
    
#     if atlas_uid:
#         # Get compounds by specific atlas UID
#         query = """
#             SELECT
#                 a.atlas_uid,
#                 a.atlas_name,
#                 a.atlas_description,
#                 a.chromatography,
#                 a.polarity,
#                 c.compound_uid,
#                 c.name AS compound_name,
#                 c.inchi_key,
#                 c.inchi,
#                 mzrt.adduct,
#                 mzrt.mz,
#                 mzrt.rt_peak,
#                 mzrt.rt_min,
#                 mzrt.rt_max,
#                 mzrt.mz_tolerance,
#                 mzrt.mz_rt_reference_uid
#             FROM atlases a
#             JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
#             JOIN compounds c ON aca.compound_uid = c.compound_uid
#             LEFT JOIN mz_rt_references mzrt ON aca.mz_rt_reference_uid = mzrt.mz_rt_reference_uid
#             WHERE a.atlas_uid = ?
#             ORDER BY aca.association_order, mzrt.rt_peak
#         """
#         df = conn.execute(query, [atlas_uid]).df()
#     else:
#         print("Must provide atlas_uid")
#         conn.close()
#         return pd.DataFrame()

#     conn.close()
    
#     if df.empty:
#         print(f"No compounds found for atlas criteria")
#         return pd.DataFrame()
#     else:
#         print(f"Retrieved {len(df)} compounds for atlas: {df['atlas_name'].iloc[0]}")
#         print(f"Atlas chromatography: {df['chromatography'].iloc[0]}, polarity: {df['polarity'].iloc[0]}")

#         return df

def create_project_database(project_db_path: Path, overwrite_existing: bool = False) -> None:
    """Create project-specific database with required tables.
    If overwrite_existing is True, delete the database file and remake it.
    """
    project_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if overwrite_existing and project_db_path.exists():
        project_db_path.unlink()
        print(f"Overwrite is set to True; Deleted existing database at {project_db_path} and proceeding...")
    elif not overwrite_existing and project_db_path.exists():
        print(f"Overwrite is set to False but database already exists at {project_db_path}. Use overwrite_existing=True to replace it.")
        return

    conn = duckdb.connect(str(project_db_path))
    
    # Create all tables for project database
    _create_database_tables(conn, db_type="project")
    
    conn.close()
    print(f"Project database created at {project_db_path}")

def save_lcmsruns_to_db(
    project_db_path: Path,
    project_path: Path,
    project_metadata: dict,
    overwrite_existing: bool = False
) -> Dict:
    """
    Save LCMS run files to project database and return file paths grouped by chromatography/polarity/analysis type.
    Chromatography and polarity are inferred from filenames.
    If the lcmsruns table exists and has rows, do not overwrite unless overwrite_existing=True.
    """
    project_name = os.path.basename(project_path)
    files_by_group = lrt.get_project_files(project_path, "lcmsruns")

    conn = duckdb.connect(str(project_db_path))

    # Check if lcmsruns table exists
    table_exists = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'lcmsruns'"
    ).fetchone()[0] > 0

    # Check if lcmsruns table has rows
    has_rows = False
    if table_exists:
        has_rows = conn.execute("SELECT COUNT(*) FROM lcmsruns").fetchone()[0] > 0

    if table_exists and has_rows and not overwrite_existing:
        print("LCMS runs already exist in the database. Skipping overwrite.")
        conn.close()
        print_files_summary(files_by_group)
        return files_by_group

    if table_exists and has_rows and overwrite_existing:
        print(f"LCMS runs already exist in the database for {project_name} but overwrite is set to True. Creating new table...")
        conn.execute("DELETE FROM lcmsruns")

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
                            project_metadata["analyst"],
                            project_metadata["timestamp"],
                        ),
                    )
                    total_files += 1

    conn.close()
    print(f"Saved {total_files} LCMS runs to database:")
    print_files_summary(files_by_group)
    return files_by_group

def print_files_summary(file_list: Dict[str, Any]):
    for chrom, pol_dict in file_list.items():
        print(f"\nChromatography: {chrom}")
        for pol, filetype_dict in pol_dict.items():
            print(f"  Polarity: {pol}")
            for file_type, files_list in filetype_dict.items():
                print(f"    {file_type}: {len(files_list)} files")

def list_available_atlases(db_path: str, chromatography: str = None, polarity: str = None) -> pd.DataFrame:
    """List all available atlases with optional filtering."""
    conn = duckdb.connect(str(db_path))
    
    conditions = ["1=1"]
    params = []
    
    if chromatography:
        # Handle HILIC/HILICZ interchangeability
        if chromatography.upper() in ['HILIC', 'HILICZ']:
            conditions.append("(UPPER(a.chromatography) = 'HILIC' OR UPPER(a.chromatography) = 'HILICZ')")
        else:
            conditions.append("UPPER(a.chromatography) = UPPER(?)")
            params.append(chromatography)
    if polarity:
        conditions.append("a.polarity = ?")
        params.append(polarity)
    
    query = f"""
    SELECT 
        a.atlas_uid,
        a.atlas_name,
        a.atlas_description,
        a.chromatography,
        a.polarity,
        a.creation_time,
        COUNT(aca.compound_uid) as compound_count
    FROM atlases a
    LEFT JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
    WHERE {' AND '.join(conditions)}
    GROUP BY a.atlas_uid, a.atlas_name, a.atlas_description, 
             a.chromatography, a.polarity, a.creation_time
    ORDER BY a.atlas_name
    """
    
    df = conn.execute(query, params).df()
    conn.close()
    
    return df

def validate_database(db_path: Path) -> None:
    """
    Validate the DuckDB database and print summary statistics.
    
    Parameters:
        db_path (Path): Path to the DuckDB database file.
    """
    print(f"\nDatabase Validation:")

    try:
        conn = duckdb.connect(str(db_path))
        
        # Show table counts
        compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
        references_count = conn.execute("SELECT COUNT(*) FROM mz_rt_references").fetchone()[0]
        atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
        atlas_compound_associations_count = conn.execute("SELECT COUNT(*) FROM atlas_compound_associations").fetchone()[0]
        atlas_info = list_available_atlases(str(db_path))

        # Show method combinations
        method_combinations = conn.execute("""
            SELECT chromatography, polarity, COUNT(*) as reference_count 
            FROM mz_rt_references 
            GROUP BY chromatography, polarity
        """).fetchall()

        print(f"   Compounds: {compounds_count}")
        print(f"   RT/MZ References: {references_count}")
        print(f"   Atlases: {atlases_count}")
        print(f"   Atlas-Compound associations: {atlas_compound_associations_count}")

        if method_combinations:
            print(f"   Method combinations:")
            for combo in method_combinations:
                print(f"      {combo[0]}/{combo[1]}: {combo[2]} references")
        else:
            print(f"   No method combinations found")

        if not atlas_info.empty:
            print(f"   Available atlases:")
            for _, row in atlas_info.iterrows():
                print(f"      {row['atlas_uid']}")
                print(f"            {row['atlas_name']}")
                print(f"            {row['chromatography']} {row['polarity']}")
                print(f"            {row['compound_count']} compounds")
                print(f"            {row['creation_time']}")
        else:
            print(f"\n   No atlases found")

        conn.close()
        return
        
    except Exception as e:
        print(f"Error validating database: {e}")

def save_rt_alignment_model_to_db(corrected_atlas_uid: str, project_db_path: Path, best_model: dict, qc_files: list, modeling_data: list, project_metadata: dict) -> str:
    """Save RT alignment model to project database."""
    rt_alignment_uid = _generate_uid("rt_alignment")
    
    # Extract project name from path
    project_name = project_db_path.stem.replace('.duckdb', '')
    
    model_metadata = {
        "qc_files": [os.path.basename(f) for f in qc_files],
        "compounds_used": [d.get('compound_uid', '') for d in modeling_data],
        "correction_timestamp": project_metadata["timestamp"],
        "correction_method": "polynomial_qc_based",
        "analyst": project_metadata["analyst"]
    }
    
    conn = duckdb.connect(str(project_db_path))
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
        project_metadata["analyst"],
        project_metadata["timestamp"],
        json.dumps(model_metadata)
    ))
    conn.close()
    
    print(f"RT alignment model saved to database with UID: {rt_alignment_uid}")
    return rt_alignment_uid

def add_compounds_to_db(input_df: pd.DataFrame, pubchem_cache: Dict, db_path: Path, 
                       creator_name: str, input_file_path: str = "", duplicate_compounds: bool = False):
    """Add compounds and RT/MZ references to database without creating an atlas."""
    print(f"Adding compounds to database: {db_path}")
    
    conn = duckdb.connect(str(db_path))
    timestamp = datetime.now()
    
    # Create tables if they don't exist
    _create_database_tables(conn, db_type="main")
    
    # Get existing compounds
    existing_inchi_keys = set()
    if not duplicate_compounds:
        existing_result = conn.execute("SELECT inchi_key FROM compounds").fetchall()
        existing_inchi_keys = {row[0] for row in existing_result}
        print(f"Found {len(existing_inchi_keys)} existing compounds in database")
    
    # Process compounds and references
    compounds_created = 0
    compounds_skipped = 0
    references_created = 0
    references_skipped = 0
    
    for idx, row in tqdm(input_df.iterrows(), total=len(input_df), desc="Processing compounds"):
        inchi_key = row.get('inchi_key')
        if pd.isna(inchi_key):
            continue
        
        # Get method information from input data
        chromatography = str(row.get('chromatography', 'Unknown'))
        polarity = str(row.get('polarity', 'Unknown'))
        
        compound_uid = None
        
        # Check if compound exists
        if not duplicate_compounds and inchi_key in existing_inchi_keys:
            compounds_skipped += 1
            
            # Get existing compound_uid
            existing_compound_result = conn.execute(
                "SELECT compound_uid FROM compounds WHERE inchi_key = ?", 
                [inchi_key]
            ).fetchone()
            
            if existing_compound_result:
                compound_uid = existing_compound_result[0]
        else:
            # Create new compound
            compound_uid = _create_compound_record(conn, row, pubchem_cache, input_df, idx, 
                                                 creator_name, timestamp)
            compounds_created += 1
            
            # Add to existing sets for future duplicate checking in this session
            if not duplicate_compounds:
                existing_inchi_keys.add(inchi_key)
        
        # Create RT/MZ reference if data available and compound_uid exists
        if compound_uid and pd.notna(row.get('rt_peak')) and pd.notna(row.get('mz')):
            ref_uid = _create_mz_rt_reference(conn, row, compound_uid, chromatography, 
                                            polarity, input_file_path, creator_name, timestamp,
                                            table_name="mz_rt_references")
            if ref_uid:
                references_created += 1
            else:
                references_skipped += 1
    
    # Final counts
    compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    references_count = conn.execute("SELECT COUNT(*) FROM mz_rt_references").fetchone()[0]
    
    print(f"\nCompounds added successfully!")
    print(f"   Total compounds in database: {compounds_count}")
    print(f"   New compounds created: {compounds_created}")
    if compounds_skipped > 0:
        print(f"   Compounds skipped (already existed): {compounds_skipped}")
    print(f"   Total RT/MZ references in database: {references_count}")
    print(f"   New RT/MZ references created: {references_created}")
    if references_skipped > 0:
        print(f"   RT/MZ references skipped (duplicates): {references_skipped}")
    
    conn.close()
    return compounds_created, references_created, compounds_skipped, references_skipped

def _create_compound_record(conn, row, pubchem_cache: Dict, input_df: pd.DataFrame, idx: int, 
                           creator_name: str, timestamp) -> str:
    """Create a new compound record in the database."""
    compound_uid = _generate_uid("compound")
    
    # Extract compound data from row
    name = str(row.get('name', row.get('compound_name', f'Compound_{idx}')))
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
    
    # Insert compound record
    conn.execute("""
        INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
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
        creator_name,
        timestamp,
        timestamp
    ))
    
    return compound_uid

def _create_mz_rt_reference(conn, row, compound_uid: str, chromatography: str, 
                           polarity: str, input_file_path: str, creator_name: str, timestamp,
                           table_name: str = "mz_rt_references", rt_correction_applied: bool = False,
                           rt_shift: float = None, source_mz_rt_reference_uid: str = None) -> str:
    """Create a new mz_rt_reference or mz_rt_experimental record in the database."""
    
    if table_name == "mz_rt_references":
        # Check for duplicate reference (same compound, chromatography, polarity, adduct)
        adduct = str(row.get('adduct', ''))
        
        existing_ref = conn.execute("""
            SELECT mz_rt_reference_uid 
            FROM mz_rt_references 
            WHERE compound_uid = ? AND chromatography = ? AND polarity = ? AND adduct = ?
        """, [compound_uid, chromatography, polarity, adduct]).fetchone()
        
        if existing_ref:
            print(f"Warning! Attempting to add an exact duplicate to the database: {existing_ref}. Skipping")
            return None  # Skip duplicate
        
        uid = _generate_uid("mz_rt_reference")
        
        # Extract RT/MZ data
        rt_peak = float(row.get('rt_peak'))
        rt_min = float(row.get('rt_min', rt_peak - 0.5))  # Default to ±0.5 min if not provided
        rt_max = float(row.get('rt_max', rt_peak + 0.5))
        mz = float(row.get('mz'))
        mz_tolerance = float(row.get('mz_tolerance', 5.0))
        adduct = str(row.get('adduct', ''))
        
        confidence = str(row.get('confidence', 'Unknown')) if pd.notna(row.get('confidence')) else 'Unknown'
        source = input_file_path if input_file_path else 'Unknown'
        
        # Insert mz_rt_reference record
        conn.execute("""
            INSERT INTO mz_rt_references VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
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
            creator_name,
            timestamp
        ))
        
    elif table_name == "mz_rt_experimental":
        uid = _generate_uid("mz_rt_experimental")
        
        # Extract RT/MZ data
        rt_peak = float(row.get('rt_peak'))
        rt_min = float(row.get('rt_min', rt_peak - 0.5))
        rt_max = float(row.get('rt_max', rt_peak + 0.5))
        mz = float(row.get('mz'))
        mz_tolerance = float(row.get('mz_tolerance', 5.0))
        adduct = str(row.get('adduct', ''))
        
        # Insert mz_rt_experimental record
        conn.execute("""
            INSERT INTO mz_rt_experimental VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            uid,
            compound_uid,
            rt_peak,
            rt_min,
            rt_max,
            mz,
            mz_tolerance,
            adduct,
            rt_correction_applied,
            rt_shift,
            source_mz_rt_reference_uid,
            creator_name,
            timestamp
        ))
    else:
        raise ValueError(f"Invalid table_name: {table_name}. Must be 'mz_rt_references' or 'mz_rt_experimental'")
    
    return uid

def create_atlas_from_compounds(
    db_path: Path,
    atlas_name: str,
    atlas_description: str,
    chromatography: str,
    polarity: str,
    creator_name: str,
    compound_uids: list = None,
    compound_filter: dict = None,
    target_adducts: list = None,
    experimental_uid_map: dict = None
):
    """
    Create an atlas and associate compounds with it.
    """
    conn = duckdb.connect(str(db_path))
    timestamp = datetime.now()

    atlas_uid = _generate_uid("atlas")

    conn.execute("""
        INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        atlas_uid,
        atlas_name,
        atlas_description,
        chromatography,
        polarity,
        creator_name,
        timestamp,
        timestamp
    ))

    print(f"Created new atlas: {atlas_name} (UID: {atlas_uid})")

    # Get compounds to associate
    if compound_uids:
        # Build the query to filter by chromatography, polarity, and adduct
        placeholders = ','.join(['?' for _ in compound_uids])
        adduct_condition = ""
        params = [chromatography, polarity]
        if target_adducts:
            adduct_placeholders = ','.join(['?' for _ in target_adducts])
            adduct_condition = f"AND mzrt2.adduct IN ({adduct_placeholders})"
            params.extend(target_adducts)
        params.extend(compound_uids)
        query = f"""
        SELECT c.compound_uid, 
                (SELECT mz_rt_reference_uid 
                FROM mz_rt_references mzrt2 
                WHERE mzrt2.compound_uid = c.compound_uid 
                AND mzrt2.chromatography = ? 
                AND mzrt2.polarity = ? 
                {adduct_condition}
                LIMIT 1) as mz_rt_reference_uid
        FROM compounds c
        WHERE c.compound_uid IN ({placeholders})
        ORDER BY c.compound_uid
        """
        result = conn.execute(query, params).fetchall()
        compounds_to_associate = result
    elif compound_filter:
        # Filter compounds based on criteria
        conditions = ["1=1"]
        params = []
        
        if compound_filter.get('chromatography'):
            if compound_filter['chromatography'].upper() in ['HILIC', 'HILICZ']:
                conditions.append("(UPPER(mzrt.chromatography) = 'HILIC' OR UPPER(mzrt.chromatography) = 'HILICZ')")
            else:
                conditions.append("UPPER(mzrt.chromatography) = UPPER(?)")
                params.append(compound_filter['chromatography'])
        
        if compound_filter.get('polarity'):
            conditions.append("mzrt.polarity = ?")
            params.append(compound_filter['polarity'])
        
        query = f"""
        SELECT DISTINCT c.compound_uid, mzrt.mz_rt_reference_uid
        FROM compounds c
        LEFT JOIN mz_rt_references mzrt ON c.compound_uid = mzrt.compound_uid
        WHERE {' AND '.join(conditions)}
        ORDER BY c.compound_uid
        """
        
        result = conn.execute(query, params).fetchall()
        compounds_to_associate = result
    else:
        # Get all compounds
        result = conn.execute("""
            SELECT DISTINCT c.compound_uid, mzrt.mz_rt_reference_uid
            FROM compounds c
            LEFT JOIN mz_rt_references mzrt ON c.compound_uid = mzrt.compound_uid
            ORDER BY c.compound_uid
        """).fetchall()
        compounds_to_associate = result
    
    # Create associations - each atlas gets its own unique associations
    atlas_associations_created = 0
    
    for idx, (compound_uid, ref_or_exp_uid) in enumerate(compounds_to_associate):
        # Always create new association UID for each atlas
        association_uid = _generate_uid("association")
        
        conn.execute("""
            INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            association_uid,
            atlas_uid,
            compound_uid,
            ref_or_exp_uid,
            idx + 1,
            creator_name,
            timestamp
        ))
        atlas_associations_created += 1

    conn.close()
    return atlas_uid, atlas_associations_created

def create_new_atlas(conn, atlas_name: str, atlas_description: str, 
                    chromatography: str, polarity: str, creator_name: str, timestamp) -> str:
    """Create a new atlas with a new UID. No duplicate checking - always creates new."""
    
    # Always create new atlas with unique UID
    atlas_uid = _generate_uid("atlas")
    
    conn.execute("""
        INSERT INTO atlases VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        atlas_uid,
        atlas_name,
        atlas_description,
        chromatography,
        polarity,
        creator_name,
        timestamp,
        timestamp
    ))
    
    print(f"Created new atlas: {atlas_name} (UID: {atlas_uid})")
    return atlas_uid

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
            creation_time TIMESTAMP,
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
            creation_time TIMESTAMP,
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
            creation_time TIMESTAMP,
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
                creation_time TIMESTAMP,
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
                creation_time TIMESTAMP
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
                creation_time TIMESTAMP
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
                mz DOUBLE NOT NULL,
                mz_tolerance DOUBLE NOT NULL,
                adduct VARCHAR NOT NULL,
                rt_correction_applied BOOLEAN DEFAULT FALSE,
                rt_shift DOUBLE,
                source_mz_rt_reference_uid VARCHAR,
                created_by VARCHAR,
                creation_time TIMESTAMP
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
                creation_time TIMESTAMP,
                model_metadata TEXT
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
    
def delete_atlas_from_db(db_path: str, atlas_uid: str) -> str:
    """
    Delete an atlas and all its associated metadata from the database.
    This includes cascading deletions from related tables.
    
    Parameters:
        db_path (str): Path to the DuckDB database file
        atlas_uid (str): UID of the atlas to delete
        
    Returns:
        str: Status message indicating success or failure
    """
    conn = duckdb.connect(str(db_path))
    
    try:
        # First verify the atlas exists and get its name for confirmation
        atlas_result = conn.execute("""
            SELECT atlas_name, atlas_description 
            FROM atlases 
            WHERE atlas_uid = ?
        """, [atlas_uid]).fetchone()
        
        if not atlas_result:
            print(f"Atlas with UID {atlas_uid} not found in database")
            return "No deletion!"
        
        atlas_name = atlas_result[0]
        atlas_description = atlas_result[1]
        
        print(f"Found atlas: {atlas_name}")
        print(f"Description: {atlas_description}")
        
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
            print(f"Removed {associations_count} compound associations")
        
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
                        print(f"Removed {rt_alignment_count} RT alignment records")
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
                        print(f"Removed {count} records from {table_name}")
                except Exception as e:
                    print(f"Warning: Could not delete from table {table_name}: {e}")
        except Exception as e:
            # Information schema queries might fail in some contexts
            pass
        
        # Finally, delete the atlas record itself
        conn.execute("""
            DELETE FROM atlases 
            WHERE atlas_uid = ?
        """, [atlas_uid])
        
        print(f"\nSuccessfully deleted atlas '{atlas_name}' (UID: {atlas_uid})")
        
        if deletion_summary:
            print("Cascade deletions performed:")
            for table, count in deletion_summary.items():
                print(f"  - {table}: {count} records")
        
        return "Deletion successful!"
        
    except Exception as e:
        print(f"Error deleting atlas: {e}")
        return "Deletion failed!"

    finally:
        conn.close()

def get_atlas_from_db(db_path: str, atlas_uid: str) -> pd.DataFrame:
    """
    Retrieve atlas metadata from database for a specific atlas.
    Returns DataFrame with atlas information including name, description, chromatography, polarity, etc.
    """
    conn = duckdb.connect(str(db_path))
    
    query = """
    SELECT 
        atlas_uid,
        atlas_name,
        atlas_description,
        chromatography,
        polarity,
        created_by,
        creation_time,
        last_modified
    FROM atlases 
    WHERE atlas_uid = ?
    """
    
    df = conn.execute(query, [atlas_uid]).df()
    conn.close()
    
    if df.empty:
        print(f"No atlas found with UID: {atlas_uid}")
        return pd.DataFrame()
    else:
        print(f"Retrieved atlas metadata for: {df['atlas_name'].iloc[0]}")
        return df

def get_rt_correction_table_entry(db_path: Path, atlas_uid: str) -> Optional[dict]:
    """
    Retrieve the RT correction table entry for a given atlas UID from the rt_alignment table.
    Returns a dictionary with the entry if found, else None.
    """
    conn = duckdb.connect(str(db_path))
    query = """
        SELECT *
        FROM rt_alignment
        WHERE atlas_uid = ?
        ORDER BY creation_time DESC
        LIMIT 1
    """
    result = conn.execute(query, [atlas_uid]).fetchone()
    columns = [desc[0] for desc in conn.description] if result else []
    conn.close()
    if result and columns:
        return dict(zip(columns, result))
    else:
        print(f"No RT correction entry found for atlas UID: {atlas_uid}")
        return None


def _generate_uid(entity_type: str, rt_atlas: bool = False) -> str:
    """Generate a unique identifier for database entities."""
    if entity_type == "atlas":
        if rt_atlas is False:
            return f"atlas-{uuid.uuid4().hex[:32]}"
        elif rt_atlas is True:
            return f"atlas-rt-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_experimental":
        return f"mzrt-exp-{uuid.uuid4().hex[:32]}"
    elif entity_type == "compound":
        return f"compound-{uuid.uuid4().hex[:32]}"
    elif entity_type == "mz_rt_reference":
        return f"mzrt-{uuid.uuid4().hex[:32]}"
    elif entity_type == "association":
        return f"assoc-{uuid.uuid4().hex[:32]}"
    elif entity_type == "rt_alignment":
        return f"rta-{uuid.uuid4().hex[:32]}"
    else:
        return f"{entity_type}-{uuid.uuid4().hex[:32]}"

def get_experimental_files_from_db(project_db_path: str, file_types: List[str] = None) -> List[str]:
    """Get experimental files from project database."""
    
    if file_types is None:
        file_types = ['experimental', 'istd', 'exctrl']  # Exclude QC files
    
    conn = duckdb.connect(str(project_db_path))
    
    # Get experimental files (excluding QC)
    placeholders = ','.join(['?' for _ in file_types])
    result = conn.execute(f"""
        SELECT file_path 
        FROM lcmsruns 
        WHERE file_type IN ({placeholders})
        ORDER BY filename
    """, file_types).fetchall()
    
    conn.close()
    
    file_paths = [row[0] for row in result]
    print(f"Found {len(file_paths)} experimental files in database")
    
    return file_paths

def create_targeted_analysis_table(project_db_path: str) -> None:
    """Create targeted_analysis table in project database to store analysis results."""
    
    conn = duckdb.connect(str(project_db_path))
    
    # Create targeted_analysis table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS targeted_analysis (
            analysis_uid VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            atlas_uid VARCHAR NOT NULL,
            compound_uid VARCHAR NOT NULL,
            inchi_key VARCHAR(27) NOT NULL,
            compound_name VARCHAR NOT NULL,
            
            -- Original atlas data
            original_rt_peak DOUBLE,
            original_rt_min DOUBLE,
            original_rt_max DOUBLE,
            original_mz DOUBLE,
            original_mz_tolerance DOUBLE,
            adduct VARCHAR,
            
            -- Modified/corrected atlas data (if user adjusted)
            final_rt_peak DOUBLE,
            final_rt_min DOUBLE,
            final_rt_max DOUBLE,
            is_rt_modified BOOLEAN DEFAULT FALSE,
            
            -- Suggested RT bounds from EIC analysis
            suggested_rt_peak DOUBLE,
            suggested_rt_min DOUBLE,
            suggested_rt_max DOUBLE,
            suggestion_confidence DOUBLE,
            suggestion_source_file VARCHAR,
            
            -- Best EIC match across all files
            best_eic_file VARCHAR,
            best_eic_rt DOUBLE,
            best_eic_mz DOUBLE,
            best_eic_intensity DOUBLE,
            best_eic_ppm_error DOUBLE,
            best_eic_rt_error DOUBLE,
            
            -- File detection summary
            total_files_detected INTEGER DEFAULT 0,
            file_detection_rate DOUBLE,
            
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
            analysis_version VARCHAR DEFAULT '1.0'
        )
    """)
    
    # Create indexes for efficient querying
    conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_project ON targeted_analysis(project_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_atlas ON targeted_analysis(atlas_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_compound ON targeted_analysis(compound_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_targeted_analysis_inchi ON targeted_analysis(inchi_key)")
    
    conn.close()
    print("Created targeted_analysis table in project database")


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
    conn_main = duckdb.connect(main_db_path)
    
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
    conn_main.close()
    
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
    
    print(f"Enriched {len(enriched_df)} compounds with metadata from main database")
    
    return enriched_df


def validate_targeted_analysis_data(project_db_path: str, analysis_uid: str = None) -> Dict:
    """Validate targeted analysis results in database."""
    
    conn = duckdb.connect(str(project_db_path))
    
    # Get latest analysis if no specific UID provided
    if analysis_uid:
        condition = "WHERE analysis_uid = ?"
        params = [analysis_uid]
    else:
        condition = "WHERE analysis_timestamp = (SELECT MAX(analysis_timestamp) FROM targeted_analysis)"
        params = []
    
    # Validation queries
    validation_query = f"""
    SELECT 
        COUNT(*) as total_records,
        COUNT(CASE WHEN original_rt_peak IS NOT NULL THEN 1 END) as records_with_rt,
        COUNT(CASE WHEN original_mz IS NOT NULL THEN 1 END) as records_with_mz,
        COUNT(CASE WHEN best_eic_file IS NOT NULL THEN 1 END) as records_with_eic,
        COUNT(CASE WHEN ms2_files_with_data > 0 THEN 1 END) as records_with_ms2,
        COUNT(CASE WHEN is_rt_modified = true THEN 1 END) as modified_records,
        AVG(file_detection_rate) as avg_detection_rate,
        MIN(file_detection_rate) as min_detection_rate,
        MAX(file_detection_rate) as max_detection_rate
    FROM targeted_analysis
    {condition}
    """
    
    result = conn.execute(validation_query, params).fetchone()
    conn.close()
    
    if result:
        return {
            'total_records': result[0],
            'records_with_rt': result[1],
            'records_with_mz': result[2],
            'records_with_eic': result[3],
            'records_with_ms2': result[4],
            'modified_records': result[5],
            'avg_detection_rate': result[6],
            'min_detection_rate': result[7],
            'max_detection_rate': result[8]
        }
    else:
        return {}