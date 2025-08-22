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

def get_atlas_compounds_from_db(db_path: str, atlas_name: str = None, atlas_uid: str = None, 
                               chromatography: str = None, polarity: str = None) -> Tuple[pd.DataFrame, str, str]:
    """
    Retrieve compounds and their RT/MZ reference data from database for a specific atlas.
    Returns the DataFrame along with the atlas chromatography and polarity.
    """
    conn = duckdb.connect(str(db_path))
    
    if atlas_uid:
        # Get compounds by specific atlas UID
        query = """
        SELECT 
            c.compound_uid,
            c.name as compound_name,
            c.inchi_key,
            mzrt.mz_rt_reference_uid,
            mzrt.rt_peak,
            mzrt.rt_min,
            mzrt.rt_max,
            mzrt.mz,
            mzrt.mz_tolerance,
            mzrt.adduct,
            a.atlas_name,
            a.chromatography,
            a.polarity
        FROM atlases a
        JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
        JOIN compounds c ON aca.compound_uid = c.compound_uid
        JOIN mz_rt_references mzrt ON aca.mz_rt_reference_uid = mzrt.mz_rt_reference_uid
        WHERE a.atlas_uid = ?
        ORDER BY aca.association_order
        """
        df = conn.execute(query, [atlas_uid]).df()
        
    else:
        # Get compounds by atlas name and filters  
        conditions = ["1=1"]
        params = []
        
        if atlas_name:
            conditions.append("a.atlas_name = ?")
            params.append(atlas_name)
        if chromatography:
            # Handle HILIC/HILICZ interchangeability
            if chromatography.upper() in ['HILIC', 'HILICZ']:
                conditions.append("(UPPER(mzrt.chromatography) = 'HILIC' OR UPPER(mzrt.chromatography) = 'HILICZ')")
            else:
                conditions.append("UPPER(mzrt.chromatography) = UPPER(?)")
                params.append(chromatography)
        if polarity:
            conditions.append("mzrt.polarity = ?")
            params.append(polarity)
            
        query = f"""
        SELECT 
            c.compound_uid,
            c.name as compound_name,
            c.inchi_key,
            mzrt.mz_rt_reference_uid,
            mzrt.rt_peak,
            mzrt.rt_min,
            mzrt.rt_max,
            mzrt.mz,
            mzrt.mz_tolerance,
            mzrt.adduct,
            a.atlas_name,
            a.chromatography,
            a.polarity
        FROM atlases a
        JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
        JOIN compounds c ON aca.compound_uid = c.compound_uid
        JOIN mz_rt_references mzrt ON aca.mz_rt_reference_uid = mzrt.mz_rt_reference_uid
        WHERE {' AND '.join(conditions)}
        ORDER BY aca.association_order
        """
        df = conn.execute(query, params).df()
    
    conn.close()
    
    if len(df) == 0:
        print(f"No compounds found for atlas criteria")
        return df, None, None
    else:
        # Get chromatography and polarity from the atlas
        atlas_chromatography = df['chromatography'].iloc[0] if not df.empty else None
        atlas_polarity = df['polarity'].iloc[0] if not df.empty else None
        
        print(f"Retrieved {len(df)} compounds for atlas {atlas_name or atlas_uid}")
        print(f"Atlas chromatography: {atlas_chromatography}, polarity: {atlas_polarity}")
        
        return df, atlas_chromatography, atlas_polarity

def create_project_database(project_db_path: Path) -> None:
    """Create project-specific database with required tables."""
    project_db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(str(project_db_path))
    
    # Create lcmsruns table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lcmsruns (
            file_path VARCHAR PRIMARY KEY,
            filename VARCHAR NOT NULL,
            analysis_type VARCHAR NOT NULL,
            chromatography VARCHAR,
            polarity VARCHAR,
            created_by VARCHAR,
            creation_time TIMESTAMP
        )
    """)
    
    # Create mz_rt_experimental table
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
    
    # Create rt_alignment table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rt_alignment (
            rt_alignment_uid VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            polynomial_degree INTEGER,
            r_squared DOUBLE,
            rmse DOUBLE,
            coefficients TEXT,
            equation TEXT,
            qc_files_count INTEGER,
            compounds_used_count INTEGER,
            chromatography VARCHAR,
            polarity VARCHAR,
            created_by VARCHAR,
            creation_time TIMESTAMP,
            model_metadata TEXT
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lcmsruns_analysis ON lcmsruns(analysis_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mzrt_exp_compound ON mz_rt_experimental(compound_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_alignment_project ON rt_alignment(project_name)")
    
    conn.close()
    print(f"Project database created at {project_db_path}")

def save_lcmsruns_to_db(project_db_path: Path, project_path: Path, project_metadata: dict) -> Dict:
    """
    Save LCMS run files to project database and return file paths grouped by chromatography/polarity/analysis type.
    Chromatography and polarity are inferred from filenames.
    """
    project_name = os.path.basename(project_path)
    files_by_group = lrt.get_project_files(project_path, "lcmsruns")

    conn = duckdb.connect(str(project_db_path))
    conn.execute("DELETE FROM lcmsruns")

    total_files = 0
    for chrom, pol_dict in files_by_group.items():
        for pol, analysis_dict in pol_dict.items():
            for analysis_type, file_list in analysis_dict.items():
                for file_path in file_list:
                    filename = os.path.basename(file_path)
                    conn.execute(
                        "INSERT INTO lcmsruns VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            file_path,
                            filename,
                            analysis_type,
                            chrom,
                            pol,
                            project_metadata["analyst"],
                            project_metadata["timestamp"],
                        ),
                    )
                    total_files += 1

    conn.close()
    print(f"Saved {total_files} LCMS runs to database")
    return files_by_group

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
        COUNT(aca.compound_uid) as compound_count
    FROM atlases a
    LEFT JOIN atlas_compound_associations aca ON a.atlas_uid = aca.atlas_uid
    WHERE {' AND '.join(conditions)}
    GROUP BY a.atlas_uid, a.atlas_name, a.atlas_description, 
             a.chromatography, a.polarity
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
            print(f"\n   Available atlases:")
            for _, row in atlas_info.iterrows():
                print(f"      {row['atlas_name']} (UID: {row['atlas_uid']}) ({row['chromatography']}/{row['polarity']}): {row['compound_count']} compounds")
        else:
            print(f"\n   No atlases found")

        conn.close()
        return
        
    except Exception as e:
        print(f"Error validating database: {e}")

def save_rt_alignment_model_to_db(project_db_path: Path, best_model: dict, qc_files: list, modeling_data: list, project_metadata: dict) -> str:
    """Save RT alignment model to project database."""
    rt_alignment_uid = f"rta-{uuid.uuid4().hex[:32]}"
    
    model_metadata = {
        "qc_files": [os.path.basename(f) for f in qc_files],
        "compounds_used": [d['compound_uid'] for d in modeling_data],
        "correction_timestamp": project_metadata["timestamp"],
        "correction_method": "polynomial_qc_based",
        "analyst": project_metadata["analyst"]
    }
    
    conn = duckdb.connect(str(project_db_path))
    conn.execute("""
        INSERT INTO rt_alignment VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rt_alignment_uid,
        PROJECT,
        "polynomial",
        best_model['degree'],
        best_model['r2'],
        best_model['rmse'],
        json.dumps(best_model['coefficients'].tolist()),
        best_model['equation'],
        len(qc_files),
        len(modeling_data),
        CHROMATOGRAPHY,
        POLARITY,
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
    _create_database_tables(conn)
    
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
            ref_created = _create_mz_rt_reference(conn, row, compound_uid, chromatography, 
                                                polarity, input_file_path, creator_name, timestamp)
            if ref_created:
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

def create_atlas_from_compounds(db_path: Path, atlas_name: str, atlas_description: str,
                               chromatography: str, polarity: str, creator_name: str,
                               compound_uids: list = None, compound_filter: dict = None,
                               target_adducts: list = None):
    """Create an atlas and associate compounds with it. Always creates a new atlas with new UID."""
    conn = duckdb.connect(str(db_path))
    timestamp = datetime.now()
    
    # Always create a new atlas with a new UID (no duplicate checking)
    atlas_uid = f"atlas-{uuid.uuid4().hex[:32]}"
    
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
        placeholders = ','.join(['?' for _ in compound_uids])
        
        # Build the query to filter by chromatography, polarity, and adduct
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
    
    for idx, (compound_uid, mz_rt_ref_uid) in enumerate(compounds_to_associate):
        # Always create new association UID for each atlas
        association_uid = f"assoc-{uuid.uuid4().hex[:32]}"
        
        conn.execute("""
            INSERT INTO atlas_compound_associations VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            association_uid,
            atlas_uid,
            compound_uid,
            mz_rt_ref_uid,
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
    atlas_uid = f"atlas-{uuid.uuid4().hex[:32]}"
    
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

def _create_database_tables(conn):
    """Create all required database tables."""
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
    
    # Create mz_rt_references table with proper column order
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
    
    # Create atlas_compound_associations table with proper column order
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
    
    # Create rt_alignment table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rt_alignment (
            rt_alignment_uid VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            model_type VARCHAR NOT NULL,
            polynomial_degree INTEGER,
            r_squared DOUBLE,
            rmse DOUBLE,
            coefficients TEXT,
            equation TEXT,
            qc_files_count INTEGER,
            compounds_used_count INTEGER,
            chromatography VARCHAR,
            polarity VARCHAR,
            created_by VARCHAR,
            creation_time TIMESTAMP,
            model_metadata TEXT
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_lcmsruns_analysis ON lcmsruns(analysis_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mzrt_exp_compound ON mz_rt_experimental(compound_uid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rt_alignment_project ON rt_alignment(project_name)")
    
def delete_atlas_from_db(db_path: str, atlas_uid: str) -> bool:
    """
    Delete an atlas and all its associated metadata from the database.
    
    Parameters:
        db_path (str): Path to the DuckDB database file
        atlas_uid (str): UID of the atlas to delete
        
    Returns:
        bool: True if atlas was successfully deleted, False otherwise
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
        
        # Get count of associations before deletion for confirmation
        associations_count = conn.execute("""
            SELECT COUNT(*) 
            FROM atlas_compound_associations 
            WHERE atlas_uid = ?
        """, [atlas_uid]).fetchone()[0]

        print(f"Found atlas")
        print(f"Description: {atlas_description}")
        print(f"Atlas has {associations_count} compound associations")
        
        # Delete all atlas-compound associations first (due to foreign key constraints)
        conn.execute("""
            DELETE FROM atlas_compound_associations 
            WHERE atlas_uid = ?
        """, [atlas_uid])
        
        # Delete the atlas record
        conn.execute("""
            DELETE FROM atlases 
            WHERE atlas_uid = ?
        """, [atlas_uid])
        
        print(f"Successfully deleted atlas '{atlas_name}' (UID: {atlas_uid})")
        print(f"Removed {associations_count} compound associations")
        
        return "Deletion successful!"
        
    except Exception as e:
        print(f"Error deleting atlas: {e}")
        return "Deletion failed!"

    finally:
        conn.close()