#!/usr/bin/env python
"""
Generate synthetic test fixtures for metatlas2 system tests.

This script creates:
1. Synthetic parquet files (MS1 and MS2 data with gaussian peaks)
2. Main database (metatlas.duckdb) with compounds and atlases
3. MS2 reference library (ms2_references.tsv)

Run once to generate test data, then commit to repository for reproducible testing.

Usage:
    python tests/fixtures/generate_fixtures.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import yaml
import hashlib

# Add metatlas2 to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import metatlas2.database_interact as dbi
from metatlas2.workflow_objects import Compound, CompoundMZRT, Atlas

# Output directory
FIXTURES_DIR = Path(__file__).parent / "data"
DB_DIR = FIXTURES_DIR / "databases" / "main_db"
PROJECT_DIR = FIXTURES_DIR / "lcmsruns" / "test_owner" / "20260420_JGI_BPK_000000_SYSTEM-TEST_pilot_EXPXXXX_HILICZ_XXXXXXXX"
PARQUET_DIR = PROJECT_DIR / "parquet"
RAW_DIR = PROJECT_DIR / "raw"
MZML_DIR = PROJECT_DIR / "mzML"

# Test compounds - QC atlas (for RT alignment)
QC_COMPOUNDS = [
    {
        "compound_name": "TestQC1_Alanine",
        "inchi_key": "QNAYBMKLOCPYGJ-UHFFFAOYSA-N",  # L-Alanine
        "mz": 90.055, "rt": 2.5, "adduct": "[M+H]+",
        "formula": "C3H7NO2", "smiles": "CC(N)C(=O)O"
    },
    {
        "compound_name": "TestQC2_Valine",
        "inchi_key": "KZSNJWFQEVHDMF-UHFFFAOYSA-N",  # L-Valine
        "mz": 118.086, "rt": 5.0, "adduct": "[M+H]+",
        "formula": "C5H11NO2", "smiles": "CC(C)C(N)C(=O)O"
    },
    {
        "compound_name": "TestQC3_Leucine",
        "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N",  # L-Leucine
        "mz": 132.102, "rt": 7.5, "adduct": "[M+H]+",
        "formula": "C6H13NO2", "smiles": "CC(C)CC(N)C(=O)O"
    },
]

# Test compounds - ISTD atlas (internal standards)
ISTD_COMPOUNDS = [
    {
        "compound_name": "ISTD1_Glucose",
        "inchi_key": "WQZGKKKJIJFFOK-UHFFFAOYSA-N",  # D-Glucose
        "mz": 181.071, "rt": 3.0, "adduct": "[M+H]+",
        "formula": "C6H12O6", "smiles": "OCC1OC(O)C(O)C(O)C1O"
    },
    {
        "compound_name": "ISTD2_Citrate",
        "inchi_key": "KRKNYBCHXYNGOX-UHFFFAOYSA-N",  # Citric acid
        "mz": 193.035, "rt": 4.5, "adduct": "[M+H]+",
        "formula": "C6H8O7", "smiles": "OC(=O)CC(O)(CC(=O)O)C(=O)O"
    },
]

# Test compounds - EMA atlas (experimental metabolite analysis)
EMA_COMPOUNDS = [
    {
        "compound_name": "EMA1_Glutamate",
        "inchi_key": "WHUUTDBJXJRKMK-UHFFFAOYSA-N",  # L-Glutamic acid
        "mz": 148.060, "rt": 2.0, "adduct": "[M+H]+",
        "formula": "C5H9NO4", "smiles": "NC(CCC(=O)O)C(=O)O"
    },
    {
        "compound_name": "EMA2_Aspartate",
        "inchi_key": "CKLJMWTZIZZHCS-UHFFFAOYSA-N",  # L-Aspartic acid
        "mz": 134.045, "rt": 3.5, "adduct": "[M+H]+",
        "formula": "C4H7NO4", "smiles": "NC(CC(=O)O)C(=O)O"
    },
    {
        "compound_name": "EMA3_Proline",
        "inchi_key": "ONIBWKKTOPOVIA-UHFFFAOYSA-N",  # L-Proline
        "mz": 116.071, "rt": 6.0, "adduct": "[M+H]+",
        "formula": "C5H9NO2", "smiles": "OC(=O)C1CCCN1"
    },
    {
        "compound_name": "EMA4_Phenylalanine",
        "inchi_key": "COLNVLDHVKWLRT-UHFFFAOYSA-N",  # L-Phenylalanine
        "mz": 166.086, "rt": 8.0, "adduct": "[M+H]+",
        "formula": "C9H11NO2", "smiles": "NC(Cc1ccccc1)C(=O)O"
    },
    {
        "compound_name": "EMA5_Tryptophan",
        "inchi_key": "QIVBCDIJIAJPQS-UHFFFAOYSA-N",  # L-Tryptophan
        "mz": 205.097, "rt": 9.5, "adduct": "[M+H]+",
        "formula": "C11H12N2O2", "smiles": "NC(Cc1c[nH]c2ccccc12)C(=O)O"
    },
]


def generate_gaussian_peak(rt_center: float, mz_center: float, intensity: float, 
                          rt_width: float = 0.2, n_points: int = 30) -> pd.DataFrame:
    """
    Generate a gaussian-shaped chromatographic peak.
    
    Args:
        rt_center: Retention time at peak apex (minutes)
        mz_center: m/z value
        intensity: Peak height
        rt_width: Peak width (sigma of gaussian, in minutes)
        n_points: Number of data points across the peak
        
    Returns:
        DataFrame with columns [rt, mz, i]
    """
    # Generate RT points around the peak center
    rt_range = rt_width * 6  # Cover ±3 sigma
    rt_points = np.linspace(rt_center - rt_range/2, rt_center + rt_range/2, n_points)
    
    # Gaussian intensity profile
    intensities = intensity * np.exp(-0.5 * ((rt_points - rt_center) / rt_width) ** 2)
    
    # Add slight m/z variation (realistic LCMS jitter)
    mz_points = mz_center + np.random.normal(0, 0.001, n_points)
    
    return pd.DataFrame({
        'rt': rt_points,
        'mz': mz_points,
        'i': intensities
    })


def generate_ms2_spectrum(precursor_mz: float, rt: float, n_fragments: int = 5) -> pd.DataFrame:
    """
    Generate a synthetic MS2 spectrum.
    
    Args:
        precursor_mz: Precursor ion m/z
        rt: Retention time
        n_fragments: Number of fragment ions to generate
        
    Returns:
        DataFrame with columns [rt, mz, i, precursor_MZ, precursor_intensity, collision_energy]
    """
    # Generate fragment m/z values (realistic fragmentation pattern)
    # Fragments should be smaller than precursor
    fragment_mzs = []
    for i in range(n_fragments):
        # Common neutral losses: H2O (18), NH3 (17), CO2 (44), etc.
        losses = [18.01, 17.03, 44.01, 28.01, 46.01, 12.0]  # Added smaller loss
        loss = np.random.choice(losses)
        frag_mz = precursor_mz - loss - (i * 10)  # Reduced ladder spacing
        if frag_mz > 40:  # Lower threshold for small molecules
            fragment_mzs.append(frag_mz)
    
    # If no fragments generated, create at least one simple fragment
    if len(fragment_mzs) == 0:
        fragment_mzs = [precursor_mz - 18.01]  # Simple water loss
    
    # Generate intensities (decreasing for higher fragments)
    intensities = [10000 * (1.0 / (i + 1)) for i in range(len(fragment_mzs))]
    
    return pd.DataFrame({
        'rt': [rt] * len(fragment_mzs),
        'mz': fragment_mzs,
        'i': intensities,
        'precursor_MZ': [precursor_mz] * len(fragment_mzs),
        'precursor_intensity': [50000.0] * len(fragment_mzs),
        'collision_energy': [25.0] * len(fragment_mzs)
    })


def generate_ms1_parquet(filename: str, compounds: list, intensity_scale: float = 1.0):
    """
    Generate a synthetic MS1 parquet file.
    
    Args:
        filename: Output filename (full path)
        compounds: List of compound dicts with mz, rt, compound_name
        intensity_scale: Multiplier for peak intensities
    """
    all_data = []
    
    # Add peaks for each compound
    for comp in compounds:
        peak_data = generate_gaussian_peak(
            rt_center=comp['rt'],
            mz_center=comp['mz'],
            intensity=1e5 * intensity_scale,  # Typical MS1 intensity
            rt_width=0.15,
            n_points=25
        )
        all_data.append(peak_data)
    
    # Add some background noise
    n_noise = 100
    noise_data = pd.DataFrame({
        'rt': np.random.uniform(0.5, 12.0, n_noise),
        'mz': np.random.uniform(80, 400, n_noise),
        'i': np.random.uniform(1000, 5000, n_noise)
    })
    all_data.append(noise_data)
    
    # Combine and sort
    df = pd.concat(all_data, ignore_index=True)
    df = df.sort_values(['mz', 'rt']).reset_index(drop=True)
    
    # Convert to pyarrow table and write
    table = pa.Table.from_pandas(df, schema=pa.schema([
        ('rt', pa.float32()),
        ('mz', pa.float32()),
        ('i', pa.float32())
    ]))
    
    pq.write_table(table, filename, compression='snappy')
    print(f"  Created {filename.name} ({len(df)} data points)")


def generate_ms2_parquet(filename: str, compounds: list):
    """
    Generate a synthetic MS2 parquet file.
    
    Args:
        filename: Output filename (full path)
        compounds: List of compound dicts with mz, rt, compound_name
    """
    all_data = []
    
    # Add MS2 spectra for each compound
    for comp in compounds:
        ms2_data = generate_ms2_spectrum(
            precursor_mz=comp['mz'],
            rt=comp['rt'],
            n_fragments=6
        )
        all_data.append(ms2_data)
    
    # Combine
    df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(
        columns=['rt', 'mz', 'i', 'precursor_MZ', 'precursor_intensity', 'collision_energy']
    )
    
    # Convert to pyarrow table and write
    table = pa.Table.from_pandas(df, schema=pa.schema([
        ('rt', pa.float32()),
        ('mz', pa.float32()),
        ('i', pa.float32()),
        ('precursor_MZ', pa.float32()),
        ('precursor_intensity', pa.float32()),
        ('collision_energy', pa.float32())
    ]))
    
    pq.write_table(table, filename, compression='snappy')
    print(f"  Created {filename.name} ({len(df)} data points)")


def generate_lcms_files():
    """Generate synthetic LCMS parquet files and placeholder raw/mzML files."""
    print("\n=== Generating synthetic LCMS files ===")
    
    # Create all required directories
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MZML_DIR.mkdir(parents=True, exist_ok=True)
    
    all_compounds = QC_COMPOUNDS + ISTD_COMPOUNDS + EMA_COMPOUNDS
    
    # File categories with different file types and intensity scales
    file_configs = [
        ("QC1", "QC", 1.0, QC_COMPOUNDS + ISTD_COMPOUNDS),
        ("QC2", "QC", 0.9, QC_COMPOUNDS + ISTD_COMPOUNDS),
        ("QC3", "QC", 1.1, QC_COMPOUNDS + ISTD_COMPOUNDS),
        ("ISTD1", "ISTD", 0.8, ISTD_COMPOUNDS),
        ("ISTD2", "ISTD", 1.2, ISTD_COMPOUNDS),
        ("Sample1", "EXPERIMENTAL", 0.7, all_compounds),
        ("Sample2", "EXPERIMENTAL", 1.0, all_compounds),
        ("Sample3", "EXPERIMENTAL", 0.9, all_compounds),
        ("Sample4", "EXPERIMENTAL", 1.1, all_compounds),
        ("Sample5", "EXPERIMENTAL", 0.85, all_compounds),
    ]
    
    for basename, file_type, intensity_scale, compounds in file_configs:
        # Generate filename following metatlas2 conventions
        # Pattern: <date>_<project>_<chromatography>_<polarity>_<sample>-<file_type>
        # Note: Use dash before file_type (e.g., Sample1-QC) for proper parsing
        if file_type in ["QC", "ISTD"]:
            sample_name = f"{basename}-{file_type}"
        else:
            sample_name = basename
        
        prefix = f"20260420_SYSTEM_TEST_HILICZ_POS_{sample_name}"
        
        # Generate parquet files
        ms1_file = PARQUET_DIR / f"{prefix}_ms1_pos.parquet"
        ms2_file = PARQUET_DIR / f"{prefix}_ms2_pos.parquet"
        generate_ms1_parquet(ms1_file, compounds, intensity_scale)
        generate_ms2_parquet(ms2_file, compounds)
        
        # Create placeholder .raw file (must include _MS1_ or _MS2_ marker for proper parsing)
        # Using _MS1_POS_ pattern to match expected naming convention
        raw_file = RAW_DIR / f"{prefix}_MS1_POS.raw"
        raw_file.write_text(f"# Placeholder .raw file for {prefix}\n")
        
        # Create placeholder .mzML file (must include _MS1_ or _MS2_ marker for proper parsing)
        mzml_file = MZML_DIR / f"{prefix}_MS1_POS.mzML"
        mzml_content = f"""<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1">
  <cvList count="1">
    <cv id="MS" fullName="Proteomics Standards Initiative Mass Spectrometry Ontology" version="4.1.0"/>
  </cvList>
  <fileDescription>
    <fileContent>
      <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum"/>
      <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum"/>
    </fileContent>
    <sourceFileList count="1">
      <sourceFile id="sf1" name="{prefix}_MS1_POS.raw" location="file:///">
        <cvParam cvRef="MS" accession="MS:1000768" name="Thermo nativeID format"/>
        <cvParam cvRef="MS" accession="MS:1000563" name="Thermo RAW format"/>
      </sourceFile>
    </sourceFileList>
  </fileDescription>
  <run id="run1" defaultInstrumentConfigurationRef="IC1">
    <spectrumList count="0"/>
  </run>
</mzML>
"""
        mzml_file.write_text(mzml_content)
    
    print(f"Generated {len(file_configs) * 2} parquet files in {PARQUET_DIR}")
    print(f"Generated {len(file_configs)} .raw placeholder files in {RAW_DIR}")
    print(f"Generated {len(file_configs)} .mzML placeholder files in {MZML_DIR}")


def create_database_with_compounds():
    """Create main database and populate with test compounds and atlases."""
    print("\n=== Creating main database ===")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(DB_DIR / "metatlas.duckdb")
    
    # Create database with schema
    print(f"Creating database at {db_path}")
    dbi.create_metatlas_database(db_path, overwrite=True)
    
    # Generate deterministic UIDs for test data
    def make_uid(prefix: str, name: str) -> str:
        """Create deterministic UID from prefix and name."""
        hash_val = hashlib.md5(f"{prefix}_{name}".encode()).hexdigest()
        return f"{prefix}-test-{hash_val[:16]}"
    
    # Create Compound and CompoundMZRT objects
    all_compounds_data = QC_COMPOUNDS + ISTD_COMPOUNDS + EMA_COMPOUNDS
    compounds = []
    compound_mzrts = []
    compound_uid_map = {}  # Map inchi_key to compound_uid for later use
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # First, create all compounds and compound_mzrts
    for comp_data in all_compounds_data:
        compound_uid = make_uid("compound", comp_data['compound_name'])
        mz_rt_uid = make_uid("mzrt", f"{comp_data['compound_name']}_{comp_data['adduct']}")
        
        # Create Compound object
        compound = Compound(
            compound_uid=compound_uid,
            compound_name=comp_data['compound_name'],
            inchi_key=comp_data['inchi_key'],
            formula=comp_data.get('formula', ''),
            smiles=comp_data.get('smiles', ''),
            mono_isotopic_molecular_weight=comp_data['mz'] - 1.007,  # Approx neutral mass
            created_by="test_generator",
            created_date=timestamp
        )
        compounds.append(compound)
        
        # Create CompoundMZRT object
        rt_center = comp_data['rt']
        compound_mzrt = CompoundMZRT(
            mz_rt_uid=mz_rt_uid,
            compound_uid=compound_uid,
            compound_name=comp_data['compound_name'],
            inchi_key=comp_data['inchi_key'],
            adduct=comp_data['adduct'],
            rt_peak=rt_center,
            rt_min=rt_center - 0.5,
            rt_max=rt_center + 0.5,
            mz=comp_data['mz'],
            mz_tolerance=5.0,
            chromatography="HILICZ",
            polarity="POS",
            source="test_fixture",
            created_by="test_generator",
            created_date=timestamp
        )
        compound_mzrts.append(compound_mzrt)
        compound_uid_map[comp_data['inchi_key']] = compound_uid
    
    # Save compounds and mzrts to database directly
    # Note: We bypass batch_save_compounds_and_mzrts because it auto-generates UIDs,
    # but we need deterministic UIDs for testing
    print(f"Saving {len(compounds)} compounds and {len(compound_mzrts)} compound_mzrts...")
    with dbi.get_db_connection(db_path) as conn:
        # Insert compounds
        compound_records = []
        for compound in compounds:
            record = (
                compound.compound_uid,
                compound.compound_name,
                compound.inchi_key,
                compound.inchi if hasattr(compound, 'inchi') else None,
                compound.smiles if hasattr(compound, 'smiles') else None,
                compound.formula,
                None,  # compound_classes
                None,  # compound_pathways
                None,  # compound_tags
                compound.mono_isotopic_molecular_weight,
                None,  # iupac_name
                None,  # pubchem_cid
                None,  # cas_number
                None,  # synonyms
                compound.created_by,
                compound.created_date
            )
            compound_records.append(record)
        
        conn.executemany("""
            INSERT INTO compounds VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, compound_records)
        print(f"  Created {len(compound_records)} compounds")
        
        # Insert compound_mzrts
        mzrt_records = []
        for mzrt in compound_mzrts:
            record = (
                mzrt.mz_rt_uid,
                mzrt.compound_uid,
                mzrt.compound_name,
                mzrt.inchi_key,
                mzrt.adduct,
                mzrt.rt_peak,
                mzrt.rt_min,
                mzrt.rt_max,
                mzrt.mz,
                mzrt.mz_tolerance,
                mzrt.chromatography,
                mzrt.polarity,
                None,  # confidence
                mzrt.source,
                None,  # identification_notes
                mzrt.created_by,
                mzrt.created_date
            )
            mzrt_records.append(record)
        
        conn.executemany("""
            INSERT INTO compound_mzrt VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, mzrt_records)
        print(f"  Created {len(mzrt_records)} compound_mzrt entries")
    
    # Now create atlas compound associations using saved compound_mzrts
    # We need to recreate the CompoundMZRT objects with data from the database
    atlas_compounds = {"QC": {}, "ISTD": {}, "EMA": {}}
    
    for atlas_type, compound_list in [("QC", QC_COMPOUNDS), ("ISTD", ISTD_COMPOUNDS), ("EMA", EMA_COMPOUNDS)]:
        for comp_data in compound_list:
            # Find the corresponding compound_mzrt that was saved
            mz_rt_uid = make_uid("mzrt", f"{comp_data['compound_name']}_{comp_data['adduct']}")
            compound_uid = compound_uid_map[comp_data['inchi_key']]
            
            # Create CompoundMZRT object for atlas
            rt_center = comp_data['rt']
            compound_mzrt = CompoundMZRT(
                mz_rt_uid=mz_rt_uid,
                compound_uid=compound_uid,
                compound_name=comp_data['compound_name'],
                inchi_key=comp_data['inchi_key'],
                adduct=comp_data['adduct'],
                rt_peak=rt_center,
                rt_min=rt_center - 0.5,
                rt_max=rt_center + 0.5,
                mz=comp_data['mz'],
                mz_tolerance=5.0,
                chromatography="HILICZ",
                polarity="POS",
                source="test_fixture",
                created_by="test_generator",
                created_date=timestamp
            )
            atlas_compounds[atlas_type][comp_data['inchi_key']] = compound_mzrt
    
    # Create atlases
    atlases_info = [
        ("QC", "QC", "QC Atlas for RT Alignment", "REFERENCE", atlas_compounds["QC"]),
        ("ISTD", "ISTD", "Internal Standards Atlas", "REFERENCE", atlas_compounds["ISTD"]),
        ("EMA", "EMA", "Experimental Metabolite Analysis Atlas", "REFERENCE", atlas_compounds["EMA"]),
    ]
    
    atlas_uids = {}
    for analysis_type, atlas_name, description, atlas_type, compound_mzrt_dict in atlases_info:
        atlas_uid = make_uid("atlas", f"HILICZ_POS_{analysis_type}")
        
        atlas = Atlas(
            atlas_uid=atlas_uid,
            atlas_name=f"Test_{atlas_name}",
            atlas_description=description,
            chromatography="HILICZ",
            polarity="POS",
            analysis_type=analysis_type,
            atlas_type=atlas_type,
            compound_mzrts=compound_mzrt_dict,
            created_by="test_generator",
            created_date=timestamp,
            source="test_fixture"
        )
        
        print(f"Saving atlas: {atlas_name} ({atlas_uid}) with {len(compound_mzrt_dict)} compounds...")
        dbi.save_atlas_to_database(atlas, db_path)
        atlas_uids[analysis_type] = atlas_uid
    
    print(f"\nDatabase created successfully at {db_path}")
    
    return atlas_uids


def create_ms2_reference_library():
    """Create MS2 reference library TSV file in metatlas format."""
    print("\n=== Creating MS2 reference library ===")
    
    # Select a subset of compounds for MS2 reference matching
    ref_compounds = QC_COMPOUNDS[:2] + ISTD_COMPOUNDS + EMA_COMPOUNDS[:3]
    
    ref_data = []
    permanent_index = 1000  # Start with an arbitrary index
    
    for comp in ref_compounds:
        # Generate reference spectrum
        ms2_spec = generate_ms2_spectrum(comp['mz'], comp['rt'], n_fragments=8)
        
        # Extract m/z and intensity arrays
        mz_array = np.array(ms2_spec['mz'].tolist())
        intensity_array = np.array(ms2_spec['i'].tolist())
        
        # Sort by m/z values (required for matchms)
        sort_idx = np.argsort(mz_array)
        mz_array = mz_array[sort_idx].tolist()
        intensity_array = intensity_array[sort_idx].tolist()
        
        # Format spectrum as [[mz_values], [intensity_values]]
        spectrum = f"[{mz_array}, {intensity_array}]"
        
        # Generate a unique ID for this compound
        comp_id = hashlib.md5(f"test_{comp['compound_name']}".encode()).hexdigest()
        
        # Calculate exact mass (approximate from m/z - adduct mass)
        exact_mass = comp['mz'] - 1.007276  # Subtract proton mass for [M+H]+
        
        ref_data.append({
            'permanent_index': permanent_index,
            'database': 'metatlas_test',
            'id': f"test_{comp_id[:32]}",
            'name': comp['compound_name'],
            'spectrum': spectrum,
            'decimal': 4.0,
            'precursor_mz': comp['mz'],
            'polarity': 'positive',
            'adduct': comp['adduct'],
            'fragmentation_method': 'HCD',
            'collision_energy': 'absolute-25',
            'instrument': 'TEST_INSTRUMENT',
            'instrument_type': 'Orbitrap',
            'formula': comp.get('formula', ''),
            'exact_mass': exact_mass,
            'inchi_key': comp['inchi_key'],
            'inchi': '',  # Empty for test data
            'smiles': comp.get('smiles', '')
        })
        permanent_index += 1
    
    df = pd.DataFrame(ref_data)
    output_path = FIXTURES_DIR / "ms2_references.tsv"
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Created MS2 reference library at {output_path} ({len(ref_data)} reference spectra)")


def update_system_test_config(atlas_uids: dict):
    """
    Update configs/system_test_analysis.yaml with generated atlas UIDs.
    
    Args:
        atlas_uids: Dictionary mapping analysis_type to atlas_uid
    """
    print("\n=== Updating system test configuration ===")
    
    # Path to config file (relative to project root)
    config_path = Path(__file__).parent.parent.parent / "configs" / "system_test_analysis.yaml"
    
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Skipping config update. Please manually update the atlas UIDs:")
        for analysis_type, uid in atlas_uids.items():
            print(f"  {analysis_type}: {uid}")
        return
    
    # Read the YAML file preserving formatting
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    # Load as dict to verify structure
    config = yaml.safe_load(config_text)
    
    # Replace atlas UIDs using simple string replacement to preserve formatting
    # This is more reliable than yaml round-tripping which can lose comments/formatting
    replacements = {
        'RT_ALIGNMENT.HILICZ.ATLAS.uid': ('QC', r'RT_ALIGNMENT:\s+HILICZ:\s+ATLAS:\s+uid:\s+[\w-]+'),
        'TARGETED_ANALYSES.HILICZ.POS.ISTD.ATLAS.uid': ('ISTD', r'ISTD:\s+ATLAS:\s+uid:\s+[\w-]+'),
        'TARGETED_ANALYSES.HILICZ.POS.EMA.ATLAS.uid': ('EMA', r'EMA:\s+ATLAS:\s+uid:\s+[\w-]+'),
    }
    
    # Simple line-by-line replacement
    lines = config_text.split('\n')
    updated_lines = []
    
    # Track if we've seen WORKFLOWS and need to inject PATHS
    seen_workflows = False
    injected_paths = False
    
    for line in lines:
        new_line = line
        
        # Check if we're at WORKFLOWS line and haven't injected PATHS yet
        if line.strip() == 'WORKFLOWS:':
            seen_workflows = True
            updated_lines.append(new_line)
            # Check next few lines to see if PATHS already exists
            continue
        
        # If we just saw WORKFLOWS and the next non-empty line is not PATHS, inject it
        if seen_workflows and not injected_paths and line.strip() and not line.strip().startswith('PATHS:'):
            # Inject PATHS section before RT_ALIGNMENT
            updated_lines.append('  PATHS:')
            updated_lines.append('    owner: test_owner')
            updated_lines.append('    msms_refs_path: tests/fixtures/data/ms2_references.tsv')
            injected_paths = True
            # Don't continue, process current line below
        
        if line.strip().startswith('PATHS:'):
            injected_paths = True
        
        # Check for RT_ALIGNMENT atlas UID
        if 'RT_ALIGNMENT:' in config_text and 'uid:' in line and 'atlas-test-' in line:
            # Check if this is within RT_ALIGNMENT section
            # Count preceding lines to determine context
            preceding_text = '\n'.join(updated_lines)
            if 'RT_ALIGNMENT:' in preceding_text and 'TARGETED_ANALYSES:' not in preceding_text:
                indent = len(line) - len(line.lstrip())
                new_line = ' ' * indent + f"uid: {atlas_uids['QC']}"
        
        # Check for ISTD atlas UID
        elif 'ISTD:' in '\n'.join(updated_lines[-5:]) and 'ATLAS:' in '\n'.join(updated_lines[-3:]) and 'uid:' in line and ('atlas-test-' in line or 'PLACEHOLDER' in line):
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + f"uid: {atlas_uids['ISTD']}"
        
        # Check for EMA atlas UID
        elif 'EMA:' in '\n'.join(updated_lines[-5:]) and 'ATLAS:' in '\n'.join(updated_lines[-3:]) and 'uid:' in line and ('atlas-test-' in line or 'PLACEHOLDER' in line):
            indent = len(line) - len(line.lstrip())
            new_line = ' ' * indent + f"uid: {atlas_uids['EMA']}"
        
        updated_lines.append(new_line)
    
    # Write back to file
    with open(config_path, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Updated {config_path} with atlas UIDs:")
    for analysis_type, uid in atlas_uids.items():
        print(f"  {analysis_type}: {uid}")



def main():
    """Generate all test fixtures."""
    print("=" * 60)
    print("Generating metatlas2 system test fixtures")
    print("=" * 60)
    
    # Generate all fixtures
    generate_lcms_files()
    atlas_uids = create_database_with_compounds()
    create_ms2_reference_library()
    
    # Automatically update the system test config file
    update_system_test_config(atlas_uids)
    
    print("\n" + "=" * 60)
    print("✓ Fixture generation complete!")
    print("=" * 60)
    print(f"\nGenerated files in: {FIXTURES_DIR}")
    print("\nNext steps:")
    print("1. Commit these fixtures to the repository")
    print("2. Run system test: nox -s system_test")


if __name__ == "__main__":
    main()
