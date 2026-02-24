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
import load_tools as ldt
import pubchem_retrieval as pcr
import ms2_hit_detection as mhd
import extract_data_from_parquet as pdx
import manual_curation_summarizer as mcs
import extract_data_from_parquet as edp
import lcmsruns_tools as lrt

logger = lcf.get_logger('workflow_objects')

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

@dataclass
class CompoundMZRT:
    """
    Object-oriented representation of the compound_mzrt database table.
    Contains RT/MZ experimental or reference data for compounds.
    """
    
    # Database identifiers
    mz_rt_uid: str
    compound_uid: str

    # Link ref to compound for init
    inchi_key: str = ""
    adduct: str = ""
    name: str = ""

    # RT data
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    
    # MZ data
    mz: float = 0.0
    mz_tolerance: float = 5.0
    
    # Method information
    chromatography: str = ""
    polarity: str = ""
    
    # Metadata
    confidence: str = ""
    identification_notes: str = ""
    source: str = ""
    created_by: str = ""
    created_date: str = ""

    # Optional attributes for experimental data
    rt_alignment_applied: Optional[bool] = False
    manual_curation_applied: Optional[bool] = False
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundMZRT':
        """Create from atlas DataFrame row."""
        return cls(
            mz_rt_uid=row.get('mz_rt_uid', ''),
            compound_uid=row.get('compound_uid', ''),
            inchi_key=row.get('inchi_key', ''),
            name=row.get('compound_name', row.get('label', row.get('name', ''))),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz=row.get('mz', 0.0),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            adduct=row.get('adduct', ''),
            chromatography=row.get('chromatography', ''),
            polarity=row.get('polarity', ''),
            confidence=row.get('confidence', ''),
            identification_notes=row.get('identification_notes', ''),
            source=row.get('source', ''),
            rt_alignment_applied=row.get('rt_alignment_applied', False),
            manual_curation_applied=row.get('manual_curation_applied', False),
            created_by=row.get('created_by', ''),
            created_date=row.get('created_date', '')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
            'mz_rt_uid': self.mz_rt_uid,
            'compound_uid': self.compound_uid,
            'inchi_key': self.inchi_key,
            'name': self.name,
            'rt_peak': self.rt_peak,
            'rt_min': self.rt_min,
            'rt_max': self.rt_max,
            'mz': self.mz,
            'mz_tolerance': self.mz_tolerance,
            'adduct': self.adduct,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'confidence': self.confidence,
            'identification_notes': self.identification_notes,
            'source': self.source,
            'rt_alignment_applied': self.rt_alignment_applied,
            'manual_curation_applied': self.manual_curation_applied,
            'created_by': self.created_by,
            'created_date': self.created_date
        }

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
    compound_mzrts: Dict[str, CompoundMZRT] = field(default_factory=dict)
    
    # Atlas metadata
    rt_alignment_number: Optional[int] = None
    analysis_number: Optional[int] = None
    created_by: str = ""
    created_date: str = ""
    source: str = ""

    # Optional link to source atlas for experimental atlases created from reference atlases
    source_atlas_uid: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> List[str]:
        """Validate atlas data and return list of issues found."""

        logger.info(f"Validating atlas {self.atlas_name} (UID: {self.atlas_uid}) with {len(self.compound_mzrts)} compounds...")

        # Helper for required fields
        def check_required(field, name):
            if not field:
                raise ValueError(f"Atlas {name} is missing")

        check_required(self.atlas_uid, "UID")
        check_required(self.atlas_name, "name")
        check_required(self.chromatography, "chromatography")
        check_required(self.polarity, "polarity")

        # Check compound MZRT data
        if not self.compound_mzrts or len(self.compound_mzrts) == 0:
            raise ValueError("No compound MZRTs in atlas")

        # Check for duplicate compound_uids
        compound_uids = [c.compound_uid for c in self.compound_mzrts.values()]
        if len(compound_uids) != len(set(compound_uids)):
            raise ValueError("Duplicate compound_uids found in compound_mzrts")

        # Validate each CompoundMZRT
        for key, compound in self.compound_mzrts.items():
            inchi_key = compound.inchi_key or "<unknown>"
            if not compound.compound_uid:
                raise ValueError(f"CompoundMZRT {inchi_key} missing compound_uid")
            if compound.mz is None or compound.mz <= 0:
                raise ValueError(f"CompoundMZRT {inchi_key} has invalid m/z: {compound.mz}")
            if compound.rt_peak is None or compound.rt_peak <= 0:
                raise ValueError(f"CompoundMZRT {inchi_key} has invalid RT peak: {compound.rt_peak}")
            if compound.rt_min is None or compound.rt_max is None:
                raise ValueError(f"CompoundMZRT {inchi_key} missing RT min/max")
            elif compound.rt_min >= compound.rt_max:
                raise ValueError(f"CompoundMZRT {inchi_key} has invalid RT bounds: {compound.rt_min} >= {compound.rt_max}")
            if not compound.adduct:
                raise ValueError(f"CompoundMZRT {inchi_key} missing adduct")

        logger.info("Atlas passed validation!")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Atlas to DataFrame format for database operations."""
        rows = []
        for compound_mzrt in self.compound_mzrts.values():
            compound_dict = compound_mzrt.__dict__.copy()
            # Add Atlas-level metadata
            compound_dict.update({
                'atlas_uid': self.atlas_uid,
                'atlas_name': self.atlas_name,
                'atlas_description': self.atlas_description,
                'chromatography': self.chromatography,
                'polarity': self.polarity,
                'analysis_type': self.analysis_type,
                'atlas_type': self.atlas_type,
                'created_by': self.created_by,
                'created_date': self.created_date,
                'source': self.source,
                'source_atlas_uid': self.source_atlas_uid
            })
            rows.append(compound_dict)
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, atlas_df: pd.DataFrame) -> 'Atlas':
        """Create Atlas object from DataFrame, using defaults for missing fields.
        """
        meta = atlas_df.iloc[0] if not atlas_df.empty else {}
        atlas_uid = meta.get('atlas_uid', '')
        atlas_name = meta.get('atlas_name', '')
        atlas_description = meta.get('atlas_description', '')
        chromatography = meta.get('chromatography', '')
        polarity = meta.get('polarity', '')
        analysis_type = meta.get('analysis_type', '')
        atlas_type = meta.get('atlas_type', 'REFERENCE')
        created_by = meta.get('created_by', '')
        created_date = meta.get('created_date', '')
        source = meta.get('source', '')
        source_atlas_uid = meta.get('source_atlas_uid', None)
        compound_mzrts = {}
        for _, row in atlas_df.iterrows():
            key = row.get('inchi_key', row.get('compound_uid', ''))
            compound_mzrt = CompoundMZRT.from_atlas_row(row)
            compound_mzrts[key] = compound_mzrt


        return cls(
            atlas_uid=atlas_uid,
            atlas_name=atlas_name,
            atlas_description=atlas_description,
            chromatography=chromatography,
            polarity=polarity,
            analysis_type=analysis_type,
            atlas_type=atlas_type,
            compound_mzrts=compound_mzrts,
            created_by=created_by,
            created_date=created_date,
            source=source,
            source_atlas_uid=source_atlas_uid
        )

    @classmethod
    def from_database(cls, database_path: str, atlas_uid: str, main_db_path: str = None):
        """
        Load an Atlas object directly from the database using its UID.
        """
        logger.info(f"Loading atlas with UID {atlas_uid} from database...")
        return cls.from_dataframe(dbi.get_atlas_compounds_table(database_path, atlas_uid, main_db_path))


@dataclass
class NewCompound:
    config: Dict[str, Any]
    overwrite_db: bool = False

    def run(self) -> Tuple[List[Compound], List[CompoundMZRT]]:
        main_db_path = self.config["ENV"]["PATHS"]["main_database"]

        compounds = []
        compound_mzrts = []
        dbi.create_metatlas_database(main_db_path, overwrite=self.overwrite_db)

        for chrom, pol_dict in self.config['COMPOUNDS'].items():
            for pol, pol_config in pol_dict.items():
                for file_path in pol_config.get('PATHS', []):
                    if not file_path:
                        logger.debug(f"Skipping empty file path for {chrom}/{pol}")
                        continue
                    logger.info(f"Processing compound file: {file_path}")
                    compounds_df = ldt.load_compound_input(file_path)
                    pcr.retrieve_pubchem_info(
                        compounds=compounds_df, 
                        pubchem_cache_path=self.config["ENV"]["PATHS"]["pubchem_cache"], 
                        use_pubchem_cache=self.config["PARAMS"].get("use_pubchem_cache", True), 
                        update_pubchem_cache=self.config["PARAMS"].get("update_pubchem_cache", False)
                        )
                    for _, row in compounds_df.iterrows():
                        try:
                            compound = Compound.from_atlas_row(row)
                            compounds.append(compound)
                            compound_mzrt = CompoundMZRT.from_atlas_row(row)
                            compound_mzrt.source = file_path
                            compound_mzrts.append(compound_mzrt)
                        except Exception as e:
                            logger.warning(f"Failed to create Compound/CompoundMZRT for row {row.get('name', 'Unknown')}: {e}")

        dbi.batch_save_compounds_and_mzrts(main_db_path, compounds, compound_mzrts)
        dbi.validate_database(main_db_path)

        return

@dataclass
class NewAtlas:
    config: Dict[str, Any]

    def run(self) -> List[Atlas]:
        summary = []
        for chrom, pol_dict in self.config['ATLASES'].items():
            for pol, pol_config in pol_dict.items():
                for analysis_type, atlas_info in pol_config.items():
                    if not atlas_info.get('path'):
                        logger.debug(f"Skipping atlas with no path for {analysis_type}/{chrom}/{pol}")
                        continue
                    try:
                        atlas_compounds_df = ldt.load_atlas_input(atlas_info['path'])
                        atlas_obj = dbi.create_new_atlas_from_dataframe(
                            atlas_df=atlas_compounds_df,
                            atlas_name=atlas_info.get('name', 'Unnamed Atlas'),
                            atlas_description=atlas_info.get('desc', 'No Description'),
                            analysis_type=analysis_type,
                            chromatography=chrom,
                            polarity=pol,
                            atlas_file_path=atlas_info['path'],
                            main_db_path=self.config["ENV"]["PATHS"]["main_database"]
                        )
                        dbi.save_atlas_to_database(atlas_obj, self.config["ENV"]["PATHS"]["main_database"])
                        logger.info(f"Successfully created atlas: {atlas_obj.atlas_name}")
                        atlas_obj.validate()
                        summary.append({
                            'atlas_uid': atlas_obj.atlas_uid,
                            'atlas_name': atlas_obj.atlas_name,
                            'compound_count': len(atlas_obj.compound_mzrts)
                        })
                    except Exception as e:
                        logger.error(f"Failed to create Atlas for {analysis_type}/{chrom}/{pol}: {e}")
        logger.info("Atlas creation summary:")
        for info in summary:
            logger.info(f"Atlas: {info['atlas_name']} (UID: {info['atlas_uid']}) - {info['compound_count']} compounds")

        return

@dataclass
class Project:
    project_name: str = field(default_factory=str)
    config: Dict = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    lcmsruns: List['LCMSRun'] = field(default_factory=list)

    def setup(self, project_name: str, config: dict, overwrite_existing: bool = False):

        self.project_name = project_name
        self.config = config

        logger.info(f"Setting up workflow paths for project {self.project_name}...")
        self.paths = _set_up_paths(
            self.config, 
            self.project_name, 
            "project_setup"
        )

        logger.info(f"Creating project database at {self.paths['project_db_path']}...")
        dbi.create_project_database(
            project_db_path=self.paths['project_db_path'],
            overwrite=overwrite_existing
        )
        
        logger.info(f"Loading LCMS runs...")
        lcmsruns_list = lrt.get_project_lcmsruns_from_disk(
            self.paths['raw_data_directory']
        )

        logger.info("Saving LCMS runs metadata to database...")
        dbi.save_lcmsruns_to_db(
            self.paths['project_db_path'],
            self.project_name,
            lcmsruns_list,
            overwrite_existing
        )

        logger.info(f"Storing LCMS runs in Project object...")
        self.lcmsruns = [LCMSRun(**row) for row in lcmsruns_list]

        return

@dataclass
class LCMSRun:
    file_path: str
    filename: str
    file_format: str
    file_type: str
    chromatography: str
    ms_level: int
    polarity: str
    created_by: str
    created_date: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

@dataclass
class RTAlign:
    # Core metadata
    rt_alignment_uid: str = None
    rt_alignment_number: int = None
    qc_atlas_uid: str = None
    model_type: str = None
    polynomial_degree: int = None
    r_squared: float = None
    rmse: float = None
    coefficients: str = None
    equation: str = None
    num_qc_files: int = None
    num_compounds: int = None
    created_by: str = None
    created_date: str = None
    use_existing_model: bool = False
    chromatography: str = None
    project_name: str = None

    # Attributes added during analysis
    rt_alignment_params: Dict[str, Any] = field(default_factory=dict)
    aligner_lcmsruns: List[LCMSRun] = field(default_factory=list)
    modeling_data: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)

    # Attributes modified by external functions
    rt_shift_stats: Dict[str, Any] = field(default_factory=dict)
    rt_aligned_atlases: Dict[str, Atlas] = field(default_factory=dict)
    rt_alignment_model: Optional[Dict[str, Any]] = None

    # Paths and config
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(self, config: dict, project_name: str, chromatography: str, rt_alignment_number: int):
        """
        Set up RTAlign object using a Project object and RT alignment parameters.
        Populates paths, config, and relevant atlas UID.
        """

        logger.info(f"Setting up RTAlign object for {chromatography} chromatography and RT alignment number {rt_alignment_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.chromatography = chromatography
        self.project_name = project_name
        self.config = config
        self.qc_atlas_uid = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('ATLAS', {}).get('uid', None)
        self.rt_alignment_params = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('PARAMS', {})

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="rt_alignment",
            rt_alignment_number=self.rt_alignment_number
        )

        logger.info(f"Checking for existing RT model in database with UID {self.qc_atlas_uid} ({self.chromatography}) and RT Alignment number {self.rt_alignment_number}")
        self.check_existing_rt_alignment()

        logger.info(f"Retrieving all LCMS runs for project...")
        project_lcmsruns = dbi.get_lcmsruns_from_db(
            project_db_path=self.paths['project_db_path'],
        )

        logger.info(f"Filtering {len(project_lcmsruns)} LCMS runs...")
        self.aligner_lcmsruns = lrt.filter_lcmsruns_list(
            lcmsruns=project_lcmsruns,
            file_type=['qc'],
            chromatography=self.chromatography
        )

    def check_existing_rt_alignment(self) -> Optional[Dict]:
        """
        Check for an existing RT alignment model in the database and handle logic for reuse or creation.
        """

        rta_model = None
        use_existing_rt_alignment = self.rt_alignment_params.get('use_existing_rt_alignment', False)
        existing_rt_aln_model = dbi.get_rt_alignment_model_from_db(self)

        if use_existing_rt_alignment and existing_rt_aln_model is not None:
            logger.info(f"Using the existing RT alignment model from database for atlas {self.qc_atlas_uid} and RT alignment number {self.rt_alignment_number}.")
            rta_model = rat.calculate_model_values_from_existing(existing_rt_aln_model)
            if rta_model:
                logger.info(f"Loaded existing RT alignment model with R² = {rta_model['r2']:.4f}, RMSE = {rta_model['rmse']:.4f}")
        elif use_existing_rt_alignment and existing_rt_aln_model is None:
            logger.warning(f"Variable 'use_existing_rt_alignment' is True, but no existing RT alignment model found in database for atlas {self.qc_atlas_uid}. Creating new RT Alignment model.")
        elif not use_existing_rt_alignment and existing_rt_aln_model is not None:
            raise ValueError(f"Variable 'use_existing_rt_alignment' is False, but RT alignment model already exists in database for atlas {self.qc_atlas_uid}. To avoid overwriting, either set use_existing_rt_alignment to True or choose a different RT alignment number.")
        elif not use_existing_rt_alignment and existing_rt_aln_model is None:
            logger.info(f"Variable 'use_existing_rt_alignment' is False and no existing RT alignment model found in database for atlas {self.qc_atlas_uid}. Creating new RT Alignment model.")

        self.rt_alignment_model = rta_model

class ManualCuration:
    def __init__(self, inchi_key: str, adduct: str, data: pd.DataFrame):
        self.inchi_key = inchi_key
        self.adduct = adduct
        self.data = data

class MS1Data:
    def __init__(self, inchi_key: str, adduct: str, filename: str, data: pd.DataFrame):
        self.inchi_key = inchi_key
        self.adduct = adduct
        self.filename = filename
        self.data = data

class MS2Data:
    def __init__(self, inchi_key: str, adduct: str, filename: str, data: pd.DataFrame):
        self.inchi_key = inchi_key
        self.adduct = adduct
        self.filename = filename
        self.data = data

class MS2Hit:
    def __init__(self, inchi_key: str, adduct: str, filename: str, data: pd.DataFrame):
        self.inchi_key = inchi_key
        self.adduct = adduct
        self.filename = filename
        self.data = data

class ExperimentalData:
    def __init__(self):
        self.manual_curation: List[ManualCuration] = []
        self.ms1_data: List[MS1Data] = []
        self.ms2_data: List[MS2Data] = []
        self.ms2_hits: List[MS2Hit] = []

@dataclass
class AutoIdentification:
    # Core metadata
    auto_id_uid: str = None
    rt_alignment_number: int = None
    analysis_number: int = None
    analysis_atlas_uid: str = None
    chromatography: str = None
    polarity: str = None
    created_by: str = None
    created_date: str = None

    # Attributes added during analysis
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    autoid_lcmsruns: List[LCMSRun] = field(default_factory=list)
    experimental_data: Optional[ExperimentalData] = None
    atlas_obj: Optional[Atlas] = None

    # Paths and config
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(self, config: dict, project_name: str, rt_alignment_number: int, analysis_number: int, analysis_atlas_uid: str):
        """
        Set up AutoIdentification object using a Project object and analysis parameters.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AutoIdentification object for RT alignment number {rt_alignment_number}, analysis number {analysis_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.analysis_atlas_uid = analysis_atlas_uid
        self.config = config
        self.project_name = project_name

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="auto_identification",
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number
        )

        logger.info(f"Checking for existing Auto Identification results within RT Alignment number {self.rt_alignment_number} and analysis number {self.analysis_number}...")
        dbi.check_existing_auto_identification(self)

        logger.info(f"Retrieving all LCMS runs for project...")
        project_lcmsruns = dbi.get_lcmsruns_from_db(
            project_db_path=self.paths['project_db_path'],
        )

        self.atlas_obj = Atlas.from_database(
            database_path=self.paths['project_db_path'],
            atlas_uid=self.analysis_atlas_uid,
            main_db_path=self.paths['main_db_path']
        )

        logger.info(f"Loading workflow parameters for targeted analysis from config...")
        self.workflow_params = self.config['WORKFLOWS']['TARGETED_ANALYSES'][self.atlas_obj.chromatography][self.atlas_obj.polarity][self.atlas_obj.analysis_type]['PARAMS']

        logger.info("Finding LCMSRuns matching criteria for auto-identification...")
        self.autoid_lcmsruns = lrt.filter_lcmsruns_list(
            lcmsruns=project_lcmsruns,
            file_type=['experimental', 'istd', 'exctrl'],
            chromatography=self.atlas_obj.chromatography,
            polarity=self.atlas_obj.polarity
        )

# =============================================================================
# INDEPENDENT WORKFLOW FUNCTIONS
# =============================================================================

def add_compounds_to_db(
    config: Dict[str, Any],
    overwrite_db: bool = False,
) -> None:
    """
    Creates main database (if needed) and loads compounds from config file paths.
    """
    new_compound_obj = NewCompound(
        config=config, 
        overwrite_db=overwrite_db
    )
    
    new_compound_obj.run()

def add_atlases_to_db(
    config: Dict[str, Any]
) -> None:
    """
    Creates atlases from config file paths and saves them to the database.
    """
    new_atlas_obj = NewAtlas(
        config=config
    )
    
    new_atlas_obj.run()

def run_project_setup(
    project_name: str,
    config: Dict,
    overwrite_existing: bool = False
) -> None:
    """
    Creates project database and loads LCMS run files.
    """

    project_obj = Project()

    project_obj.setup(
        project_name=project_name,
        config=config,
        overwrite_existing=overwrite_existing
    )

def run_rt_alignment(
    config: dict,
    project_name: str,
    rt_alignment_number: int,
    chromatography: str
) -> None:
    """Run fresh RT alignment using an RT alignment atlas"""
    
    rt_align_obj = RTAlign()

    rt_align_obj.setup(
        config=config,
        project_name=project_name,
        chromatography=chromatography,
        rt_alignment_number=rt_alignment_number
    )

    if rt_align_obj.rt_alignment_model is None:
        
        template_atlas_obj = Atlas.from_database(
            database_path=rt_align_obj.paths['main_db_path'],
            atlas_uid=rt_align_obj.qc_atlas_uid
        )

        logger.info("Passing Atlas and LCMSRuns to data extractor...")
        experimental_data_obj = edp.extract_eic_and_ms2_from_parquet(
            atlas=template_atlas_obj,
            stage="rt_alignment",
            lcmsruns=rt_align_obj.aligner_lcmsruns,
            workflow_params=rt_align_obj.rt_alignment_params,
            only_ms_level=1
        )

        logger.info("Passing ExperimentalData and Atlas to summarizer...")
        rat.create_qc_matching_summary(
            experimental_data=experimental_data_obj,
            atlas=template_atlas_obj
        )

        logger.info("Passing ExperimentalData, Atlas, and RTAlign to RT alignment model builder...")
        rat.build_rt_alignment_model(
            experimental_data=experimental_data_obj,
            atlas=template_atlas_obj, 
            rt_align=rt_align_obj
        )

        logger.info("Passing RTAlign object to model database table saver...")
        dbi.save_rt_alignment_model_to_db(
            rt_align_obj=rt_align_obj
        )
        
        logger.info("Passing RTAlign object to model visualizer...")
        rat.visualize_RT_model(
            rt_align_obj=rt_align_obj
        )
    
    logger.info("Passing RTAlign object to alignment applicator...")
    rat.apply_rt_alignment_to_target_atlases(
        rt_align_obj=rt_align_obj
    )

    logger.info("Passing aligned Atlases to database saver...")
    for aligned_atlas_uid, aligned_atlas_obj in rt_align_obj.rt_aligned_atlases.items():
        dbi.save_atlas_to_database(
            atlas_obj=aligned_atlas_obj, 
            db_path=rt_align_obj.paths['project_db_path'],
            main_db_path=rt_align_obj.paths['main_db_path']
        )

    logger.info("Passing RTAlign object to RT alignment summary generator...")
    rat.run_rt_alignment_summary(
        rt_align_obj=rt_align_obj
    )

    logger.info(f"RT alignment procedure complete for RT alignment number {rt_align_obj.rt_alignment_number} and chromatography {rt_align_obj.chromatography}!")


def run_auto_identification(
    config: dict,
    project_name: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
    analysis_atlas: str = None
) -> Dict[str, int]:
    """
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """
    
    auto_id_obj = AutoIdentification()

    auto_id_obj.setup(
        config=config, 
        project_name=project_name,
        rt_alignment_number=rt_alignment_number, 
        analysis_number=analysis_number, 
        analysis_atlas_uid=analysis_atlas
    )

    logger.info("Passing Atlas and LCMSRuns to data extractor...")
    auto_id_obj.experimental_data = pdx.extract_eic_and_ms2_from_parquet(
        atlas=auto_id_obj.atlas_obj,
        stage="auto_identification",
        lcmsruns=auto_id_obj.autoid_lcmsruns,
        workflow_params=auto_id_obj.workflow_params
    )

    logger.info("Passing ExperimentalData to MS2 hit finder...")
    mhd.find_ms2_hits(
        auto_id_obj=auto_id_obj,
        msms_refs_path=auto_id_obj.paths['msms_refs_path']
    )

    logger.info("Passing ExperimentalData and Atlas to ManualCuration creator...")
    mcs.create_manual_curation_obj(
        auto_id_obj=auto_id_obj
    )

    logger.info("Passing finalized AutoIdentification object to database saver...")
    dbi.save_auto_identification_results_to_db(
        auto_id_obj=auto_id_obj
    )

def _set_up_paths(
    config: Dict, 
    project_name: str, 
    stage: str, 
    rt_alignment_number: int = None, 
    analysis_number: int = None
) -> Tuple[str, str, str]:
    """Set up and validate paths for project directory, raw data, and project database"""
    
    workflow_paths = {}

    workflow_paths['raw_data_directory'] = str(Path(config['ENV']['PATHS']['raw_data_dir']) / project_name)
    workflow_paths['project_directory'] = str(Path(config['ENV']['PATHS']['projects_dir']) / project_name)
    workflow_paths['project_db_path'] = str(Path(workflow_paths['project_directory']) / f"{project_name}.duckdb")
    workflow_paths['main_db_path'] = config["ENV"]["PATHS"]["main_database"]
    workflow_paths['msms_refs_path'] = config["ENV"]["PATHS"]["msms_refs"]

    if not Path(workflow_paths['raw_data_directory']).exists():
        raise ValueError(f"Raw data directory not found: {workflow_paths['raw_data_directory']}")
    
    if not Path(workflow_paths['main_db_path']).exists():
        raise ValueError(f"Main database not found: {workflow_paths['main_db_path']}.")

    if stage in ["rt_alignment", "auto_identification"]:
        if not Path(workflow_paths['project_db_path']).exists():
            raise FileNotFoundError(
                f"Project database not found: {workflow_paths['project_db_path']}. "
                "Please run project setup first."
            )

    if stage == "project_setup":
        Path(workflow_paths['project_directory']).mkdir(parents=True, exist_ok=True)

    if stage == "rt_alignment":
        if rt_alignment_number is None:
            raise ValueError("RT alignment number must be provided for RT alignment stage.")
        workflow_paths['rt_alignment_output_dir'] = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
        Path(workflow_paths['rt_alignment_output_dir']).mkdir(parents=True, exist_ok=True)
    
    if stage == "auto_identification":
        if rt_alignment_number is None or analysis_number is None:
            raise ValueError("Both RT alignment number and analysis number must be provided for auto identification stage.")
        workflow_paths['auto_id_output_dir'] = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}_TGA{analysis_number}")
        Path(workflow_paths['auto_id_output_dir']).mkdir(parents=True, exist_ok=True)

    return workflow_paths
