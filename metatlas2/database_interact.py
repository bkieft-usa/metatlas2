import pandas as pd
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

sys.path.append('/Users/BKieft/Metabolomics/metatlas2')
import metatlas2.lcmsruns_tools as lrt
import metatlas2.pubchem_retrieval as pcr
import metatlas2.load_tools as ldt

def get_files_by_type_from_db(project_db_path: str, file_type: str) -> pd.DataFrame:
    """
    Get files of a specific type from the project database.
    
    Args:
        project_db_path: Path to project database
        file_type: Type of files to retrieve ('qc', 'experimental', 'istd', 'injbl', 'exctrl')
    
    Returns:
        DataFrame with columns: file_path, chromatography, polarity, file_type
    """
    conn = duckdb.connect(str(project_db_path))
    files_df = conn.execute("""
        SELECT file_path, chromatography, polarity, file_type 
        FROM lcmsruns 
        WHERE file_type = ?
        ORDER BY chromatography, polarity, file_path
    """, [file_type]).df()
    conn.close()
    
    if files_df.empty:
        raise ValueError(f"No {file_type} files found in the project database")
    
    return files_df

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
    conn.close()

    if df.empty:
        print(f"No compounds found for atlas {atlas_uid}")
        return pd.DataFrame()
    else:
        db_type = "project" if is_project_db else "main"
        chromatography = ldt.detect_atlas_input_chromatography(df)
        polarity = ldt.detect_atlas_input_polarity(df)
        print(f"Retrieved {len(df)} compounds from {db_type} database for atlas: {df['atlas_name'].iloc[0]} ({df['atlas_uid'].iloc[0]})")
        print(f"Atlas chromatography: {chromatography}, polarity: {polarity}")
        if is_project_db and df['rt_correction_applied'].any():
            print(f"RT-corrected atlas detected with experimental data")

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
        print(f"Overwrite is set to True; Deleted existing database at {project_db_path} and proceeding...")
    elif not overwrite_existing and project_db_path.exists():
        print(f"Overwrite is set to False but database already exists at {project_db_path}. Use overwrite_existing=True to replace it.")
        return

    conn = duckdb.connect(str(project_db_path))
    
    # Create all tables for project database
    _create_database_tables(conn, db_type="project")
    
    conn.close()
    print(f"Project database created at {project_db_path}")

def create_metatlas_database(config: Dict) -> None:
    """
    Create main metatlas database with required tables.
    """
    db_path = Path(config["paths"]["main_database"])
    overwrite_existing = config["database_options"]["overwrite_existing_main_db"]

    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if overwrite_existing and db_path.exists():
        print("Warning! Overwrite is True and database exists. Giving 30 seconds before overwriting to abort.")
        #time.sleep(30)
        db_path.unlink()
        print(f"Deleted existing database at {db_path} and proceeding...")
    elif not overwrite_existing and db_path.exists():
        print(f"Overwrite is set to False but database already exists at {db_path}. Use overwrite_existing=True to replace it.")
        return

    conn = duckdb.connect(str(db_path))
    
    # Create all tables for main database
    _create_database_tables(conn, db_type="main")
    
    conn.close()
    print(f"Main metatlas database created at {db_path}")
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

    conn = duckdb.connect(project_db_path)

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

def list_available_atlases(db_path: str) -> pd.DataFrame:
    """List all available atlases with optional filtering."""

    conn = duckdb.connect(db_path)
    
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
    conn.close()
    
    return df

def validate_database(config: Dict, database_path: str = "main") -> None:
    """
    Validate the DuckDB database and print summary statistics.
    
    Parameters:
        db_path (Path): Path to the DuckDB database file.
    """
    print(f"\nDatabase Validation:")

    if database_path == "main":
        db_path = config["paths"]["main_database"]
    else:
        db_path = database_path

    if not os.path.exists(Path(db_path)):
        print(f"Database not found at: {db_path}")
        return

    conn = duckdb.connect(db_path)
    
    # Show table counts
    compounds_count = conn.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]
    references_count = conn.execute("SELECT COUNT(*) FROM mz_rt_references").fetchone()[0]
    atlases_count = conn.execute("SELECT COUNT(*) FROM atlases").fetchone()[0]
    atlas_compound_associations_count = conn.execute("SELECT COUNT(*) FROM atlas_compound_associations").fetchone()[0]
    atlas_info = list_available_atlases(db_path)

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

    if not atlas_info.empty:
        print(f"   Available atlases:")
        for _, row in atlas_info.iterrows():
            print(f"      {row['atlas_uid']}")
            print(f"            {row['atlas_name']}")
            print(f"            {row['chromatography']} {row['polarity']}")
            print(f"            {row['compound_count']} compounds")
            print(f"            {row['last_modified']}")

    conn.close()
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
        prov["analyst"],
        prov["timestamp"],
        json.dumps(model_metadata)
    ))
    conn.close()
    
    print(f"RT alignment model saved to database with UID: {rt_alignment_uid}")
    return rt_alignment_uid

def add_compounds_to_db(input_df: pd.DataFrame, config: Dict, input_file_path: str = ""):
    """Add compounds and RT/MZ references to database without creating an atlas."""

    pubchem_cache_path = config["paths"]["pubchem_cache"]
    db_path = config["paths"]["main_database"]
    create_duplicates = config["database_options"]["add_compound_duplicates"] if "add_compound_duplicates" in config["database_options"] else False

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}. Check path or create it first using create_metatlas_database().")

    unique_inchi_keys = input_df['inchi_key'].dropna().unique()
    print(f"Adding {len(unique_inchi_keys)} compounds to database: {db_path}")
    
    pubchem_cache = pcr.load_or_create_pubchem_cache(pubchem_cache_path)

    conn = duckdb.connect(db_path)
    prov = ldt.get_provenance()

    # Get existing compounds
    existing_inchi_keys = set()
    if not create_duplicates:
        existing_result = conn.execute("SELECT inchi_key FROM compounds").fetchall()
        existing_inchi_keys = {row[0] for row in existing_result}
        if existing_inchi_keys:
            print(f"Found {len(existing_inchi_keys)} existing compounds in database. Not creating duplicates.")

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
        if not create_duplicates and inchi_key in existing_inchi_keys:
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
            compound_uid = _create_compound_record(conn, row, pubchem_cache, input_df, idx, prov)
            compounds_created += 1
            
            # Add to existing sets for future duplicate checking in this session
            if not create_duplicates:
                existing_inchi_keys.add(inchi_key)
        
        # Create RT/MZ reference if data available and compound_uid exists
        if compound_uid and pd.notna(row.get('rt_peak')) and pd.notna(row.get('mz')):
            ref_uid = _create_mz_rt_reference(conn, row, compound_uid, chromatography, 
                                            polarity, input_file_path, prov,
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

def _create_compound_record(conn, row, pubchem_cache: Dict, input_df: pd.DataFrame, idx: int, prov: Dict) -> str:
    """Create a new compound record in the database."""
    compound_uid = _generate_uid("compound")
    
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
    
    # Insert compound record
    conn.execute("""
        INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        prov["analyst"],
        prov["timestamp"]
    ))
    
    return compound_uid

def _create_mz_rt_reference(conn, row, compound_uid: str, chromatography: str, 
                           polarity: str, input_file_path: str, prov: Dict,
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
            #print(f"Warning! Attempting to add an exact duplicate to the database: {existing_ref}. Skipping")
            return None
        
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
            prov["analyst"],
            prov["timestamp"]
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
            prov["analyst"],
            prov["timestamp"]
        ))
    else:
        raise ValueError(f"Invalid table_name: {table_name}. Must be 'mz_rt_references' or 'mz_rt_experimental'")
    
    return uid

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

    conn = duckdb.connect(str(db_path))

    prov = ldt.get_provenance()
    chromatography = ldt.detect_atlas_input_chromatography(atlas_input_table)
    polarity = ldt.detect_atlas_input_polarity(atlas_input_table)

    # Validate all compounds exist in database
    print(f"Validating compounds exist in database: {db_path}")
    
    # Get all compounds currently in database
    existing_compounds = conn.execute("SELECT inchi_key, compound_uid, name FROM compounds").fetchall()
    existing_inchi_keys = {row[0]: {'compound_uid': row[1], 'label': row[2]} for row in existing_compounds}
    
    print(f"Database contains {len(existing_inchi_keys)} compounds, searching for missing compounds in input.")
    
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
    
    print(f"Compounds available for atlas: {len(available_compounds)}")
    print(f"Compounds missing from database: {len(missing_compounds)}")
    
    if missing_compounds:
        print("\nMissing compounds:")
        for compound in missing_compounds[:10]:
            print(f"  - {compound['label']} ({compound['inchi_key']})")
        if len(missing_compounds) > 10:
            print(f"  ... and {len(missing_compounds) - 10} more")
        
        conn.close()
        raise ValueError(f"Cannot create atlas: {len(missing_compounds)} compounds not found in database. "
                        f"Please add these compounds first.")
    
    print(f"\nAll {len(available_compounds)} compounds found in database!")

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

    print(f"Created new atlas: {atlas_name} (UID: {atlas_uid}). Adding compounds...")

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

    conn.close()

    print(f"\nAtlas creation complete!")
    print(f"   Atlas Name: {atlas_name}")
    print(f"   Atlas UID: {atlas_uid}")
    print(f"   Atlas associations created: {atlas_associations_created}")
    print(f"   Method: {chromatography}/{polarity}")

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
        ORDER BY last_modified DESC
        LIMIT 1
    """
    result = conn.execute(query, [atlas_uid]).fetchone()
    columns = [desc[0] for desc in conn.description] if result else []
    conn.close()
    if result and columns:
        rt_dict = dict(zip(columns, result))
        rt_df = pd.DataFrame([rt_dict])
        return rt_df
    else:
        print(f"No RT correction entry found for atlas UID: {atlas_uid}")
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
    conn.close()
    
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
    conn_src = duckdb.connect(from_db)
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
    conn_src.close()

    # Write new atlas and associations to to_db
    conn_dst = duckdb.connect(to_db)
    prov = ldt.get_provenance()
    timestamp = prov["timestamp"]
    analyst = prov["analyst"]

    new_atlas_uid = _generate_uid("analyzed_atlas")
    atlas_name = new_atlas_name if new_atlas_name else atlas_row[0]
    atlas_description = new_atlas_description if new_atlas_description else atlas_row[1]

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
                    mz, mz_tolerance, adduct, ms1_notes, ms2_notes,
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

    conn_dst.close()
    print(f"Cloned and modified atlas {source_atlas_uid} from {from_db} to new atlas {new_atlas_uid} in {to_db}")
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

        total_files_detected = meta.get('number_of_files', 0)
        ms2_files_with_data = meta.get('ms2_data', {}).get('total_files', 0)
        ms2_best_score = None
        ms2_best_database = None
        ms2_total_matches = None

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

    conn = duckdb.connect(project_db_path)
    insert_sql = '''
        INSERT INTO targeted_analysis (
            analysis_uid, project_name, atlas_uid, compound_uid, inchi_key, compound_name,
            pre_rt_peak, pre_rt_min, pre_rt_max, pre_mz, mz_tolerance, adduct,
            post_rt_peak, post_rt_min, post_rt_max, is_rt_modified,
            best_eic_file, best_eic_rt, best_eic_mz, best_eic_intensity, best_eic_ppm_error, best_eic_rt_error,
            avg_eic_rt, avg_eic_intensity, avg_eic_mz,
            best_ms2_file, best_ms2_database, best_ms2_ref_id, best_ms2_rt_peak, best_ms2_intensity_peak, best_ms2_mz_peak,
            best_ms2_score, best_ms2_num_matches, best_ms2_ref_frags, best_ms2_data_frags, best_ms2_frags_matching,
            avg_ms2_score,
            total_files_detected, ms2_files_with_data, ms2_best_score, ms2_best_database, ms2_total_matches,
            ms1_notes, ms2_notes, analyst, analysis_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    conn.executemany(insert_sql, rows)
    conn.close()
    print(f"Inserted {len(rows)} rows into targeted_analysis table with analysis_uid {analysis_uid}.")

    validated = validate_targeted_analysis_data(project_db_path, analysis_uid)
    if validated is False:
        print(f"    Validation failed for targeted analysis entry {analysis_uid}")
        return None
    elif validated is True:
        return analysis_uid

def verify_single_atlas_for_targeted_analysis(targeted_df: pd.DataFrame):
    """Ensure that only one atlas is used for the targeted analysis."""
    if targeted_df['atlas_uid'].nunique() > 1:
        raise ValueError("Multiple atlases found in targeted analysis results.")
        return None
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
        print(f"No targeted analysis results found for analysis_uid {analysis_uid}")
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

def get_all_db_table_info(db_path = None):
    """
    Print all tables and their columns for a DuckDB database.
    Either config (with 'paths'->'main_database') or db_path must be provided.
    """
    # Determine database path
    if isinstance(db_path, str):
        database = db_path
    elif isinstance(db_path, Dict):
        database = db_path["paths"]["main_database"]
    else:
        raise ValueError("Must provide a valid db_path.")

    con = duckdb.connect(database)
    tables = con.execute("SHOW TABLES").fetchdf()
    for table in tables['name']:
        print(f"Table: {table}")
        columns = con.execute(f"DESCRIBE {table}").fetchdf()
        print(columns[['column_name', 'column_type']])
        print()
    con.close()
    if not targeted_df.empty:
        merged = merged.merge(targeted_df, on=['atlas_uid', 'compound_uid'], how='left', suffixes=('', '_targ'))

    # Optionally, merge atlas metadata
    atlas_meta = conn_proj.execute("SELECT * FROM atlases WHERE atlas_uid = ?", [atlas_uid]).df()
    for col in atlas_meta.columns:
        merged[col] = atlas_meta[col].iloc[0] if not atlas_meta.empty else None

    conn_proj.close()
    conn_main.close()
    return merged

def get_all_db_table_info(db_path = None):
    """
    Print all tables and their columns for a DuckDB database.
    Either config (with 'paths'->'main_database') or db_path must be provided.
    """
    # Determine database path
    if isinstance(db_path, str):
        database = db_path
    elif isinstance(db_path, Dict):
        database = db_path["paths"]["main_database"]
    else:
        raise ValueError("Must provide a valid db_path.")

    con = duckdb.connect(database)
    tables = con.execute("SHOW TABLES").fetchdf()
    for table in tables['name']:
        print(f"Table: {table}")
        columns = con.execute(f"DESCRIBE {table}").fetchdf()
        print(columns[['column_name', 'column_type']])
        print()
    con.close()

