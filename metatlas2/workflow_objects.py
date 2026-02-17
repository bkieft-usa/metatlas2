from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime
import sys
import json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import targeted_analysis as tga
import load_tools as ldt
import pubchem_retrieval as pcr
import ms2_hit_detection as mhd
import extract_data_from_parquet as pdx
import ms1_ms2_summarizer as mss

logger = lcf.get_logger('workflow_objects')

class WorkflowStage(Enum):
    """Enumeration of workflow stages"""
    PROJECT_SETUP = "project_setup"
    RT_ALIGNMENT = "rt_alignment" 
    AUTO_IDENTIFICATION = "auto_identification"
    MANUAL_CURATION = "manual_curation"
    FINAL_REPORT = "final_report"

# =============================================================================
# CORE DATA CLASSES (Database table representations of unique compounds)
# =============================================================================

@dataclass
class Compound:
    """
    Object-oriented representation of the compounds database table.
    Contains immutable chemical compound metadata.
    """
    
    # Core identifiers (required)
    compound_uid: str
    name: str
    inchi_key: str
    
    # Chemical properties  
    inchi: str = ""
    smiles: str = ""
    formula: str = ""
    
    # Classification and metadata
    compound_classes: str = ""
    compound_pathways: str = ""
    compound_tags: str = ""
    
    # Physical properties
    mono_isotopic_molecular_weight: float = 0.0
    
    # External identifiers
    iupac_name: str = ""
    pubchem_cid: str = ""
    cas_number: str = ""
    synonyms: str = ""
    
    # Database metadata
    created_by: str = ""
    created_date: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'Compound':
        """Create from atlas DataFrame row."""
        return cls(
            compound_uid=row.get('compound_uid', ''),
            name=row.get('compound_name', row.get('label', row.get('name', ''))),
            inchi_key=row.get('inchi_key', ''),
            inchi=row.get('inchi', ''),
            smiles=row.get('smiles', ''),
            formula=row.get('formula', ''),
            compound_classes=row.get('compound_classes', ''),
            compound_pathways=row.get('compound_pathways', ''),
            compound_tags=row.get('compound_tags', ''),
            mono_isotopic_molecular_weight=row.get('mono_isotopic_molecular_weight', 0.0),
            iupac_name=row.get('iupac_name', ''),
            pubchem_cid=row.get('pubchem_cid', ''),
            cas_number=row.get('cas_number', ''),
            synonyms=row.get('synonyms', ''),
            created_by=row.get('created_by', ''),
            created_date=row.get('created_date', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
            'compound_uid': self.compound_uid,
            'name': self.name,
            'inchi_key': self.inchi_key,
            'inchi': self.inchi,
            'smiles': self.smiles,
            'formula': self.formula,
            'compound_classes': self.compound_classes,
            'compound_pathways': self.compound_pathways,
            'compound_tags': self.compound_tags,
            'mono_isotopic_molecular_weight': self.mono_isotopic_molecular_weight,
            'iupac_name': self.iupac_name,
            'pubchem_cid': self.pubchem_cid,
            'cas_number': self.cas_number,
            'synonyms': self.synonyms,
            'created_by': self.created_by,
            'created_date': self.created_date
        }


# =============================================================================
# REFERENCE COMPOUND DATA (Static reference information for a compound)
# =============================================================================

@dataclass
class CompoundReference:
    """
    Object-oriented representation of the mz_rt_references database table.
    Contains RT/MZ reference data linking compounds to analytical methods.
    """
    
    # Database identifiers
    mz_rt_reference_uid: str
    compound_uid: str

    # Link ref to compound for init
    inchi_key: str = ""

    # RT data
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    
    # MZ data
    mz: float = 0.0
    mz_tolerance: float = 5.0
    adduct: str = ""
    
    # Method information
    chromatography: str = ""
    polarity: str = ""
    
    # Metadata
    confidence: str = ""
    source: str = ""
    created_by: str = ""
    created_date: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundReference':
        """Create from atlas DataFrame row."""
        return cls(
            mz_rt_reference_uid=row.get('mz_rt_reference_uid', ''),
            compound_uid=row.get('compound_uid', ''),
            inchi_key=row.get('inchi_key', ''),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz=row.get('mz', 0.0),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            adduct=row.get('adduct', ''),
            chromatography=row.get('chromatography', ''),
            polarity=row.get('polarity', ''),
            confidence=row.get('confidence', ''),
            source=row.get('source', ''),
            created_by=row.get('created_by', ''),
            created_date=row.get('created_date', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
            'mz_rt_reference_uid': self.mz_rt_reference_uid,
            'compound_uid': self.compound_uid,
            'inchi_key': self.inchi_key,
            'rt_peak': self.rt_peak,
            'rt_min': self.rt_min,
            'rt_max': self.rt_max,
            'mz': self.mz,
            'mz_tolerance': self.mz_tolerance,
            'adduct': self.adduct,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'confidence': self.confidence,
            'source': self.source,
            'created_by': self.created_by,
            'created_date': self.created_date
        }

# =============================================================================
# ATLAS (Collection of compounds)
# =============================================================================

@dataclass
class Atlas:
    """
    Object-oriented representation of the atlases database table
    Collection of reference compounds with RT/MZ data.
    Maps to database atlas + atlas_compound_associations tables.
    """
    
    # Core metadata
    atlas_uid: str
    atlas_name: str
    atlas_description: str
    chromatography: str
    polarity: str
    analysis_type: str
    atlas_type: str = "REFERENCE"
    
    # Compound references (immutable reference data)
    compound_references: Dict[str, CompoundReference] = field(default_factory=dict)
    
    # Atlas metadata
    created_by: str = ""
    created_date: str = ""
    source_atlas_uid: Optional[str] = None

    # Utility methods
    def validate(self) -> List[str]:
        """Validate atlas data and return list of issues found."""
        issues = []
        
        # Check basic metadata
        if not self.atlas_uid:
            issues.append("Atlas UID is missing")
        if not self.atlas_name:
            issues.append("Atlas name is missing")
        if not self.chromatography:
            issues.append("Chromatography is missing")
        if not self.polarity:
            issues.append("Polarity is missing")
        
        # Check compounds
        if not self.compound_references:
            issues.append("No compound references in atlas")

        # Check for duplicate compound UIDs
        compound_uids = [c.compound_uid for c in self.compound_references.values()]
        if len(compound_uids) != len(set(compound_uids)):
            issues.append("Duplicate compound UIDs found")
        
        # Check individual compound references
        for inchi_key, compound_ref in self.compound_references.items():
            if not compound_ref.compound_uid:
                issues.append(f"Compound reference {inchi_key} missing compound_uid")
            if compound_ref.mz <= 0:
                issues.append(f"Compound reference {inchi_key} has invalid m/z: {compound_ref.mz}")
            if compound_ref.rt_peak <= 0:
                issues.append(f"Compound reference {inchi_key} has invalid RT peak: {compound_ref.rt_peak}")
            if compound_ref.rt_min >= compound_ref.rt_max:
                issues.append(f"Compound reference {inchi_key} has invalid RT bounds: {compound_ref.rt_min} >= {compound_ref.rt_max}")
        
        return issues

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Atlas to DataFrame format for database operations."""
        rows = []
        for compound_ref in self.compound_references.values():
            compound_dict = compound_ref.to_dict()
            compound_dict.update({
                'atlas_uid': self.atlas_uid,
                'atlas_name': self.atlas_name,
                'atlas_description': self.atlas_description,
                'chromatography': self.chromatography,
                'polarity': self.polarity
            })
            rows.append(compound_dict)
        
        return pd.DataFrame(rows)

    @classmethod 
    def from_dataframe(cls, atlas_df: pd.DataFrame, atlas_uid: str = None, 
                       atlas_name: str = None, atlas_description: str = None,
                       chromatography: str = None, polarity: str = None) -> 'Atlas':
        """Create Atlas object from DataFrame."""
        
        # Get atlas metadata from first row if not provided
        if not atlas_uid and not atlas_df.empty:
            atlas_uid = atlas_df.iloc[0].get('atlas_uid', '')
        if not atlas_name and not atlas_df.empty:
            atlas_name = atlas_df.iloc[0].get('atlas_name', '')
        if not atlas_description and not atlas_df.empty:
            atlas_description = atlas_df.iloc[0].get('atlas_description', '')
        if not chromatography and not atlas_df.empty:
            chromatography = atlas_df.iloc[0].get('chromatography', '')
        if not polarity and not atlas_df.empty:
            polarity = atlas_df.iloc[0].get('polarity', '')
        
        # Convert compounds
        compound_references = {}
        for _, row in atlas_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            # Use compound_uid as key since inchi_key might not be available in CompoundReference
            key = row.get('inchi_key', compound_ref.compound_uid)
            compound_references[key] = compound_ref
        
        return cls(
            atlas_uid=atlas_uid or '',
            atlas_name=atlas_name or '',
            atlas_description=atlas_description or '',
            chromatography=chromatography or '',
            polarity=polarity or '',
            compound_references=compound_references
        )

    
# =============================================================================
# DATABASE MANAGER (primarily for adding compounds to the database)
# =============================================================================

class DatabaseManager:
    """
    Manages main database creation and compound loading operations.
    
    Schema for adding compounds to database from input file:
    DatabaseManager (organizer)
        Compound (attributes read from input file + PubChem queries -> compounds table)
            CompoundReference (attributes read from input file -> mz_rt_references table, 
                             or skipped if RT/MZ columns don't exist)
    """

    config: Dict[str, Any]
    overwrite_db: bool
    use_pubchem_cache: bool
    main_db_path: str
    pubchem_cache_path: str

    def __init__(self, config: Dict[str, Any], overwrite_db: bool = False, use_pubchem_cache: bool = True):
        """
        Initialize DatabaseManager with configuration.

        Args:
            config: Configuration dictionary loaded from YAML
            overwrite_db: Whether to overwrite the main database if it exists
            use_pubchem_cache: Whether to use PubChem cache
        """
        self.config = config
        self.overwrite_db = overwrite_db
        self.use_pubchem_cache = config["PARAMS"].get("use_pubchem_cache", use_pubchem_cache)
        self.main_db_path = config["ENV"]["PATHS"]["main_database"]
        self.pubchem_cache_path = config["ENV"]["PATHS"]["pubchem_cache"]

    def create_main_database(self) -> None:
        """
        Create the main metatlas database.
        """
        db_exists = Path(self.main_db_path).exists()
        if self.overwrite_db or not db_exists:
            if self.overwrite_db and db_exists:
                logger.warning("Overwriting main metatlas database...")
            else:
                logger.warning("Main database not found. Creating new database...")
            dbi.create_metatlas_database(self.main_db_path, self.overwrite_db)
        else:
            logger.info("Main database already exists, not creating a new one.")

    def save_compounds_to_db(self, compound_file_paths: List[str]) -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Load compounds from multiple input files, create Compound and CompoundReference objects,
        and store them in the main database.
        
        Args:
            compound_file_paths: List of paths to compound input files
            
        Returns:
            Tuple of (List of Compound objects, List of CompoundReference objects) that were created
        """
        logger.info(f"Loading compounds from {len(compound_file_paths)} files...")
        
        all_compounds = []
        all_compound_references = []
        
        for file_path in compound_file_paths:
            logger.info(f"Processing compound file: {file_path}")
            
            # Step 1: Load raw compound data from file
            compounds_df = ldt.load_compound_input(file_path)
            
            # Step 2: Retrieve PubChem information if requested
            pcr.retrieve_pubchem_info(compounds_df, self.pubchem_cache_path, self.use_pubchem_cache)

            # Step 3: Create Compound and CompoundReference objects from DataFrame
            compounds, compound_references = self._create_compounds_and_references_from_dataframe(
                compounds_df, source_file=file_path
            )
            
            # Step 4: Save compounds and references to database
            self._save_compounds_and_references_to_database(compounds, compound_references)
            
            all_compounds.extend(compounds)
            all_compound_references.extend(compound_references)
            logger.info(f"Created {len(compounds)} compounds and {len(compound_references)} references from {file_path}")

        logger.info(f"Compound loading complete! Total: {len(all_compounds)} compounds, {len(all_compound_references)} references")
        return all_compounds, all_compound_references

    def _create_compounds_and_references_from_dataframe(self, compounds_df: pd.DataFrame, 
                                                      source_file: str = "") -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Convert DataFrame to Compound and CompoundReference objects.
        
        Args:
            compounds_df: DataFrame with compound data
            source_file: Path to source file for metadata
            
        Returns:
            Tuple of (List of Compound objects, List of CompoundReference objects)
        """
        compounds = []
        compound_references = []
        
        for _, row in compounds_df.iterrows():
            try:
                # Create Compound object from row
                compound = Compound.from_atlas_row(row)
                compounds.append(compound)
                
                # Create CompoundReference object from row (only if RT/MZ data exists)
                if row.get('rt_peak', 0.0) > 0.0 and row.get('mz', 0.0) > 0.0:
                    compound_ref = CompoundReference.from_atlas_row(row)
                    compound_ref.inchi_key = compound.inchi_key
                    if source_file:
                        compound_ref.source = source_file
                    
                    compound_references.append(compound_ref)
                
            except Exception as e:
                logger.warning(f"Failed to create Compound/CompoundReference for row {row.get('name', 'Unknown')}: {e}")
                continue
        
        return compounds, compound_references

    def _save_compounds_and_references_to_database(self, compounds: List[Compound], 
                                                  compound_references: List[CompoundReference]) -> None:
        """
        Save Compound and CompoundReference objects to database using existing database functions.
        
        Args:
            compounds: List of Compound objects to save
            compound_references: List of CompoundReference objects to save
        """
        if not compounds and not compound_references:
            return
            
        # Convert objects to dictionaries for database operations
        compounds_data = [compound.to_dict() for compound in compounds]
        references_data = [compound_ref.to_dict() for compound_ref in compound_references]
        
        # Use batch save function
        compounds_created, references_created = dbi.batch_save_compounds_and_references(
            compounds_data, references_data, self.main_db_path
        )
        
        logger.info(f"Saved to database: {compounds_created} compounds, {references_created} references")

    def load_compounds_from_config(self) -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Load compounds from configuration file paths.
        Automatically iterates through all compound configurations in the config.
            
        Returns:
            Tuple of (List of Compound objects, List of CompoundReference objects) that were created
        """
        compound_configs = self.config.get('COMPOUNDS', {})
        all_file_paths = []
        
        # Extract all file paths from the nested configuration structure
        for _, methods in compound_configs.items():
            if methods:
                for method, method_config in methods.items():
                    paths = method_config.get('PATHS', [])
                    if paths:
                        # Filter out empty paths
                        valid_paths = [path for path in paths if path and path.strip()]
                        all_file_paths.extend(valid_paths)
        
        if not all_file_paths:
            logger.warning("No compound file paths found in configuration")
            return [], []
        
        logger.info(f"Found {len(all_file_paths)} compound files in configuration")
        
        # Load compounds from all discovered files
        return all_file_paths

    def create_compound_db_entries(self) -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Complete database setup: create database and load compounds.
            
        Returns:
            Tuple of (List of Compound objects, List of CompoundReference objects) that were created
        """
        # Create main database
        self.create_main_database()
        
        # Load compounds
        compound_files = self.load_compounds_from_config()
        compounds, compound_references = self.save_compounds_to_db(compound_files)

        # Validate
        dbi.validate_database(self.main_db_path)
        
        return compounds, compound_references

# =============================================================================
# ATLAS MANAGER (primarily for adding atlases to the database)
# =============================================================================

class AtlasManager:
    """
    Manages atlas creation from compound input files.
    
    Schema for adding an atlas to the database from an input file:
    AtlasManager (organizer)
        Atlas (new atlas created with new uid and atlas database table entry)
            Compound (inchi_keys from input file to link to new Atlas - must be present in compounds database table)
                CompoundReference (attributes read from input file - minimum rt_peak, mz must exist. 
                                 If these exactly match an existing mz_rt_references entry for the Compound, 
                                 attach existing to the Compound->Atlas, otherwise create new mz_rt_reference entry)
    """
    
    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing AtlasManager with configuration")
        self.config = config
        self.main_db_path = config["ENV"]["PATHS"]["main_database"]
        self.created_atlases = []
        logger.info("AtlasManager initialized successfully")

    def create_atlas_from_file(self) -> List[Atlas]:
        """
        Create all atlases from the configuration file.
        Automatically iterates through all atlas types, chromatographies, and polarities
        defined in the config and creates atlases for each valid configuration.
        
        Returns:
            List of Atlas objects that were created
        """
        created_atlases = []
        atlas_configs = self.config.get('ATLASES', {})

        logger.info(f"Processing atlas configurations for {len(atlas_configs)} atlas types...")

        for atlas_chrom, atlas_info in atlas_configs.items():
            for atlas_pol, analysis_types in atlas_info.items():
                for analysis_type, atlas_details in analysis_types.items():
                    if atlas_details and atlas_details.get('path') is not None:
                        try:
                            # Extract configuration
                            atlas_file_path = atlas_details['path']
                            atlas_name = atlas_details['name']
                            atlas_description = atlas_details['desc']
                            
                            logger.info(f"Creating atlas from config: {atlas_name} ({analysis_type}/{atlas_chrom}/{atlas_pol})")

                            logger.info(f"Loading atlas data from file: {atlas_file_path}")
                            atlas_compounds_df = ldt.load_atlas_input(atlas_file_path)
                            
                            logger.info(f"Creating Atlas object: {atlas_name}")
                            atlas_obj = self.create_atlas_from_dataframe(atlas_compounds_df, atlas_name, atlas_description, analysis_type, atlas_chrom, atlas_pol)
                            
                            logger.info(f"Saving atlas to database: {atlas_name}")
                            dbi.save_atlas_to_database(atlas_obj, self.main_db_path)

                            # Store in created atlases list
                            logger.info(f"Storing created atlas in workflow state: {atlas_name}")
                            self.created_atlases.append({
                                'uid': atlas_obj.atlas_uid,
                                'name': atlas_obj.atlas_name,
                                'type': analysis_type,
                                'chromatography': atlas_obj.chromatography.lower(),
                                'polarity': atlas_obj.polarity.lower(),
                                'compound_count': len(atlas_obj.compound_references),
                                'atlas_object': atlas_obj
                            })

                            created_atlases.append(atlas_obj)
                            logger.info(f"Successfully created atlas: {atlas_obj.atlas_name}")

                        except Exception as e:
                            raise ValueError(f"Failed to create atlas for {analysis_type}/{atlas_chrom}/{atlas_pol}: {e}")
                    else:
                        logger.debug(f"Skipping {analysis_type}/{atlas_chrom}/{atlas_pol} - no path specified")
        
        logger.info(f"Created {len(created_atlases)} atlases from configuration file input:")
        for atlas in created_atlases:
            logger.info(f"  Atlas: {atlas.atlas_name}")
            logger.info(f"    UID: {atlas.atlas_uid}")
            logger.info(f"    Method: {atlas.chromatography}/{atlas.polarity}")
            logger.info(f"    Compounds: {len(atlas.compound_references)}")

        return created_atlases
    
    def create_atlas_from_dataframe(self, atlas_df: pd.DataFrame, 
                                        atlas_name: str, 
                                        atlas_description: str, 
                                        analysis_type: str,
                                        chromatography: str = None, 
                                        polarity: str = None) -> Atlas:
        """
        Create Atlas with simple 2-step reference logic:
        1. For each inchi_key, get compound_uid from database
        2. Check if exact mz_rt_reference exists, if not create new one
        """
        # Detect chromatography and polarity if not provided
        if not chromatography:
            chromatography = ldt.detect_atlas_input_chromatography(atlas_df)
        if not polarity:
            polarity = ldt.detect_atlas_input_polarity(atlas_df)        

        # Step 1: Get compound_uid for each inchi_key from database
        inchi_keys = atlas_df['inchi_key'].dropna().unique().tolist()
        compound_lookup = dbi.get_compound_uids_by_inchi_keys(self.main_db_path, inchi_keys)

        # Step 2: For each row, find existing reference or create new one
        compound_references = {}
        references_reused = 0
        references_created = 0
        missing_compounds = 0
        for _, row in atlas_df.iterrows():
            inchi_key = row.get('inchi_key', '')
            if not inchi_key or inchi_key not in compound_lookup:
                missing_compounds += 1
                logger.warning(f"Compound with inchi_key {inchi_key} missing from metatlas database, skipping.")
                continue
            compound_uid = compound_lookup[inchi_key]

            # Extract RT/MZ data from input
            rt_peak = row.get('rt_peak', 0.0)
            rt_min = row.get('rt_min', rt_peak - 0.5)
            rt_max = row.get('rt_max', rt_peak + 0.5)
            mz = row.get('mz', 0.0)
            mz_tolerance = row.get('mz_tolerance', 5.0)
            adduct = str(row.get('adduct', ''))

            # Skip if missing essential RT/MZ data
            if rt_peak <= 0 or mz <= 0:
                logger.warning(f"Skipping {inchi_key}: missing RT ({rt_peak}) or MZ ({mz}) data in atlas input file")
                continue

            # Use dbi function to get or create reference UID
            mz_rt_reference_uid, reused = dbi.get_or_create_mz_rt_reference_uid(
                self.main_db_path,
                compound_uid,
                chromatography,
                polarity,
                adduct,
                rt_peak,
                mz,
                mz_tolerance
            )
            if reused:
                references_reused += 1
                #logger.debug(f"Reusing existing reference for {inchi_key}")
            else:
                references_created += 1
                #logger.debug(f"Will create new reference for {inchi_key}")

            # Create CompoundReference object
            compound_ref = CompoundReference(
                mz_rt_reference_uid=mz_rt_reference_uid,
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
                confidence=row.get('confidence', 'Unknown'),
                source='atlas_creation'
            )

            compound_references[inchi_key] = compound_ref

        logger.info(f"Reference processing complete:")
        logger.info(f"  Existing references found: {references_reused}")
        logger.info(f"  New references to create: {references_created}")
        logger.info(f"  Missing compounds: {missing_compounds}")

        # Create Atlas object
        atlas_uid = dbi._generate_uid("ref_atlas", decorator=f"{analysis_type.lower()}-{chromatography.lower()}-{polarity.lower()}")
        atlas_obj = Atlas(
            atlas_uid=atlas_uid,
            atlas_name=atlas_name,
            atlas_description=atlas_description,
            chromatography=chromatography,
            polarity=polarity,
            analysis_type=analysis_type,
            compound_references=compound_references
        )

        # Validate the atlas
        issues = atlas_obj.validate()
        if issues:
            raise ValueError(f"Atlas {atlas_uid} failed validation: {issues}")

        logger.info(f"Created Atlas object with {len(compound_references)} compound references")
        return atlas_obj


# =============================================================================
# INDEPENDENT WORKFLOW FUNCTIONS
# =============================================================================

def run_project_setup(
    project_name: str,
    config: Dict,
    overwrite_existing: bool = False
) -> str:
    """
    Stage 1: Project Setup
    Creates project database and loads LCMS run files.
    
    Returns:
        project_db_path: Path to created project database
    """
    logger.info("STAGE: Project Setup")

    # Setup paths
    workflow_paths = _set_up_paths(config, project_name, "project_setup")
    
    # Create project database
    logger.info(f"Creating project database at {workflow_paths['project_db_path']}...")
    dbi.create_project_database(workflow_paths['project_db_path'], overwrite=overwrite_existing)
    
    # Load LCMS runs
    logger.info(f"Loading LCMS runs for {project_name}...")
    try:
        dbi.save_lcmsruns_to_db(
            workflow_paths['project_db_path'],
            project_name,
            workflow_paths['raw_data_directory'],
            overwrite_existing
        )
        logger.info("LCMS runs loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load LCMS runs: {e}")
    
    return workflow_paths['project_db_path']

def run_rt_alignment(project_name: str,
                     config: Dict,
                     rt_alignment_number: int,
                     chromatography: str
) -> None:
    """Run fresh RT alignment using single RT alignment atlas"""
    
    logger.info("STAGE: RT Alignment")

    logger.info("Parsing configuration file...")
    workflow_paths = _set_up_paths(config, project_name, "rt_alignment")

    logger.info("Setting up RT alignment parameters from configuration...")
    rta_ref_atlas_chromatography = chromatography
    rta_ref_atlas_atlas_uid = config['WORKFLOWS']['RT_ALIGNMENT'][rta_ref_atlas_chromatography].get('ATLAS').get('uid', None)
    rt_alignment_parameters = config['WORKFLOWS']['RT_ALIGNMENT'][rta_ref_atlas_chromatography].get('PARAMS', {})

    logger.info(f"Checking on status of RT Alignment for {rta_ref_atlas_atlas_uid} ({rta_ref_atlas_chromatography}) and RT Alignment number {rt_alignment_number}")
    rta_model = _check_existing_rt_alignment(workflow_paths['project_db_path'],
                                             rta_ref_atlas_atlas_uid, 
                                             rt_alignment_number, 
                                             rt_alignment_parameters.get('use_existing_rt_alignment', False))

    if rta_model is None:
        logger.info(f"Loading QC files for RT alignment from project database...")
        database_files = dbi.get_lcmsruns_from_db(workflow_paths['project_db_path'], 
                                                        file_types=['qc'],
                                                        file_format='parquet',
                                                        chromatography=rta_ref_atlas_chromatography)

        logger.info(f"Extracting QC atlas compound data from QC files...")
        qc_matches = rat.extract_matches_from_qc_files(
            workflow_paths['main_db_path'],
            rta_ref_atlas_atlas_uid,
            database_files,
            rt_alignment_parameters)

        logger.info(f"Evaluating QC compound matching stats for {len(qc_matches)} matches...")
        rat.create_qc_matching_summary(qc_matches)

        logger.info(f"Building RT alignment model using {len(qc_matches)} matches")
        rta_model, modeling_data, _ = rat.build_rt_alignment_model(qc_matches, 
                                                                    rt_alignment_parameters)
        
        logger.info("Saving RT model to database...")
        dbi.save_rt_alignment_model_to_db(rta_ref_atlas_atlas_uid,
                                        workflow_paths['project_db_path'],
                                        rt_alignment_number,
                                        rta_model,
                                        database_files.get('qc', pd.DataFrame()),
                                        modeling_data.to_dict('records'))
        
        logger.info(f"Creating RT alignment plot and saving to RT alignment directory...")
        rat.visualize_RT_model(modeling_data, 
                            rta_model, 
                            str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}"))
    
    logger.info(f"Applying RT alignment to all targeted analysis atlases...")
    aligned_atlases_info = rat.apply_rt_alignment_to_target(workflow_paths['main_db_path'], 
                                                             config['WORKFLOWS']['TARGETED_ANALYSES'], 
                                                             rta_model, 
                                                             rt_alignment_parameters)

    logger.info(f"Saving RT-aligned atlases to database...")
    aligned_atlases_info = dbi.save_rt_aligned_atlas_to_db(workflow_paths['project_db_path'],
                                                                      workflow_paths['main_db_path'],
                                                                      aligned_atlases_info,
                                                                      rta_model,
                                                                      rt_alignment_number)
    
    logger.info("Generating RT alignment summaries...")
    rat.create_rt_alignment_summary(aligned_atlases_info,
                                    rta_model['rt_alignment_uid'])

    return


def run_auto_identification(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1,
    analysis_atlas: str = None
) -> Dict[str, int]:
    """
    Stage 3: Auto Identification
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """
    logger.info("STAGE: Auto Identification")
    if analysis_atlas is None:
        raise ValueError("The analysis_atlas UID must be provided to specify which atlas to run targeted analysis on.")
    
    logger.info("Parsing configuration file...")
    workflow_paths = _set_up_paths(config, project_name, "auto_identification")

    logger.info(f"Getting atlas metadata for input analysis atlas UID {analysis_atlas}")
    rt_aligned_atlas_info = dbi.get_atlas_metadata_from_db(db_path=workflow_paths['project_db_path'],
                                                           atlas_uid=analysis_atlas)

    logger.info("Setting up targeted analysis parameters from configuration...")
    workflow_params = config['WORKFLOWS']['TARGETED_ANALYSES'][rt_aligned_atlas_info['chromatography']][rt_aligned_atlas_info['polarity']][rt_aligned_atlas_info['analysis_type']]['PARAMS']
    msms_refs_path = config["ENV"]["PATHS"]["msms_refs"]
    
    logger.info(f"Running auto identification for {rt_aligned_atlas_info['atlas_name']}...")
    
    logger.info("Loading RT-aligned target atlas...")
    atlas_dataframe = dbi.get_atlas_compounds_table(database_path=workflow_paths['project_db_path'], 
                                                    atlas_uid=rt_aligned_atlas_info['atlas_uid'],
                                                    main_db_path=workflow_paths['main_db_path'])
    logger.info(f"Created Atlas dataframe with {len(atlas_dataframe)} compounds")
    display(atlas_dataframe.head())

    logger.info("Loading experimental files from project database...")
    project_files = dbi.get_lcmsruns_from_db(project_db_path=workflow_paths['project_db_path'], 
                                                     file_types=['experimental', 'istd', 'exctrl'],
                                                     file_format='parquet',
                                                     chromatography=rt_aligned_atlas_info['chromatography'],
                                                     polarity=rt_aligned_atlas_info['polarity'])

    logger.info("Extracting EIC and MS2 data from parquet files...")
    experimental_data_no_hits = pdx.extract_eic_and_ms2_from_parquet(atlas_df=atlas_dataframe,
                                                                     project_files=project_files,
                                                                     ppm_tolerance=workflow_params["default_ppm_error"],
                                                                     extra_time=workflow_params["extra_time"])
    
    logger.info("Finding MS2 reference hits...")
    experimental_data_with_hits = mhd.find_ms2_hits(experimental_data=experimental_data_no_hits, 
                                                    msms_refs_path=msms_refs_path)
    
    logger.info("Calculating MS1 and MS2 summary statistics for each compound and file...")
    experimental_data_with_hits_and_summaries = mss.create_ms_summaries(experimental_data_with_hits)

    logger.info("Saving experimental data to project database...")
    dbi.save_experimental_data_to_db(project_db_path=workflow_paths['project_db_path'],
                                     main_db_path=workflow_paths['main_db_path'],
                                     exp_data=experimental_data_with_hits_and_summaries,
                                     rt_alignment_number=rt_alignment_number,
                                     analysis_number=analysis_number)

    logger.info("Setting up targeted analysis results structure...")
    analysis_results = tga.create_analysis_results_dict(exp_data=experimental_data_with_hits_and_summaries,
                                                      target_atlas_uid=rt_aligned_atlas_info['atlas_uid'],
                                                      project_db_path=workflow_paths['project_db_path'],
                                                      atlas_dataframe=atlas_dataframe)

    logger.info("Calculating summary statistics for analysis results...")
    tga.run_analysis_summary(analysis_results)

def run_targeted_analysis(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1,
    analysis_atlas: str = None):

    logger.info("STAGE: Targeted Analysis")
    if analysis_atlas is None:
        raise ValueError("The analysis_atlas UID must be provided to specify which atlas to run targeted analysis on.")
    
    logger.info("Parsing configuration file...")
    workflow_paths = _set_up_paths(config, project_name, "targeted_analysis")

    dbi.get_experimental_data_from_db(project_db_path=workflow_paths['project_db_path'],
                                      main_db_path=workflow_paths['main_db_path'],
                                      rt_alignment_number=rt_alignment_number,
                                      analysis_number=analysis_number)

def _set_up_paths(config: Dict, project_name: str, stage: str) -> Tuple[str, str, str]:
    """Set up and validate paths for project directory, raw data, and project database"""
    
    workflow_paths = {}

    workflow_paths['raw_data_directory'] = str(Path(config['ENV']['PATHS']['raw_data_dir']) / project_name)
    workflow_paths['project_directory'] = str(Path(config['ENV']['PATHS']['projects_dir']) / project_name)
    workflow_paths['project_db_path'] = str(Path(workflow_paths['project_directory']) / f"{project_name}.duckdb")
    workflow_paths['main_db_path'] = config["ENV"]["PATHS"]["main_database"]
    
    if not Path(workflow_paths['raw_data_directory']).exists():
        raise ValueError(f"Raw data directory not found: {workflow_paths['raw_data_directory']}")
    
    if not Path(workflow_paths['main_db_path']).exists():
        raise ValueError(f"Main database not found: {workflow_paths['main_db_path']}.")

    if stage == "project_setup":
        Path(workflow_paths['project_directory']).mkdir(parents=True, exist_ok=True)

    if stage in ["rt_alignment", "auto_identification"]:
        if not Path(workflow_paths['project_db_path']).exists():
            raise FileNotFoundError(
                f"Project database not found: {workflow_paths['project_db_path']}. "
                "Please run project setup first."
            )
    
    return workflow_paths

def _check_existing_rt_alignment(project_db_path: str,
                                 template_atlas_uid: str, 
                                 rt_alignment_number: int, 
                                 use_existing_rt_alignment: bool) -> Optional[Dict]:

    rta_model = None

    existing_rt_alignment = dbi.get_rt_alignment_model_from_db(project_db_path,
                                                            template_atlas_uid, 
                                                            rt_alignment_number)

    if use_existing_rt_alignment and existing_rt_alignment is not None:
        logger.info(f"Using the existing RT alignment model from database for atlas {template_atlas_uid}")
        rta_model = dbi.get_rt_alignment_model_from_db(project_db_path, 
                                                              template_atlas_uid, 
                                                              rt_alignment_number)
        rta_model = rat.calculate_model_values_from_existing(rta_model)
        if rta_model:
            logger.info(f"Loaded existing RT alignment model with R² = {rta_model['r2']:.4f}, RMSE = {rta_model['rmse']:.4f}")
    elif use_existing_rt_alignment and existing_rt_alignment is None:
        logger.warning(f"Variable 'use_existing_rt_alignment' is True, but no existing RT alignment model found in database for atlas {template_atlas_uid}. Creating new RT Alignment model.")
    elif not use_existing_rt_alignment and existing_rt_alignment is not None:
        raise ValueError(f"Variable 'use_existing_rt_alignment' is False, but RT alignment model already exists in database for atlas {template_atlas_uid}. To avoid overwriting, either set use_existing_rt_alignment to True or choose a different RT alignment number.")
    elif not use_existing_rt_alignment and existing_rt_alignment is None:
        logger.info(f"Variable 'use_existing_rt_alignment' is False and no existing RT alignment model found in database for atlas {template_atlas_uid}. Creating new RT Alignment model.")

    return rta_model
