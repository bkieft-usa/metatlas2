from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import load_tools as ldt
import pubchem_retrieval as pcr
import lcmsruns_tools as lrt
import analysis_summary as asm

logger = lcf.get_logger('workflow_objects')

# Paths
_BASE_DATA_DIR = Path("/pscratch/sd/b/bkieft/metatlas_lite_data")
_RAW_DATA_DIR  = _BASE_DATA_DIR / "raw_data" / "jgi"
_PROJECTS_DIR  = _BASE_DATA_DIR / "projects"
_MAIN_DB_PATH  = _BASE_DATA_DIR / "databases" / "metatlas.duckdb"
_MSMS_REFS_PATH = _BASE_DATA_DIR / "databases" / "msms_refs" / "msms_refs_no_inosine15n.tab"

def _set_up_paths(
    config: Dict,
    project_name: str,
    stage: str,
    rt_alignment_number: int = None,
    analysis_number: int = None,
) -> Dict[str, str]:
    """Set up and validate paths for project directory, raw data, and project database."""

    project_dir = _PROJECTS_DIR / project_name

    paths = {
        "raw_data_directory": str(_RAW_DATA_DIR / project_name),
        "project_directory":  str(project_dir),
        "project_db_path":    str(project_dir / f"{project_name}.duckdb"),
        "main_db_path":       str(_MAIN_DB_PATH),
        "msms_refs_path":     str(_MSMS_REFS_PATH),
    }

    if not Path(paths["raw_data_directory"]).exists():
        raise ValueError(f"Raw data directory not found: {paths['raw_data_directory']}")
    if not _MAIN_DB_PATH.exists():
        raise ValueError(f"Main database not found: {paths['main_db_path']}")

    if stage == "project_setup":
        project_dir.mkdir(parents=True, exist_ok=True)

    if stage in ["rt_alignment", "auto_identification", "analysis_gui"]:
        if not project_dir.exists():
            raise FileNotFoundError(
                f"Project directory not found: {paths['project_directory']}. "
                "Please run project setup first."
            )
        if not Path(paths["project_db_path"]).exists():
            raise FileNotFoundError(
                f"Project database not found: {paths['project_db_path']}. "
                "Please run project setup first."
            )

    if stage == "auto_identification" and not _MSMS_REFS_PATH.exists():
        raise FileNotFoundError(
            f"MS/MS reference file not found: {paths['msms_refs_path']}. "
            "Please ensure the path is correct in the config file."
        )

    if stage in ["rt_alignment", "auto_identification", "analysis_gui", "analysis_summary"]:
        if rt_alignment_number is None:
            raise ValueError("RT alignment number must be provided for this stage.")
        rta_dir = project_dir / f"RTA{rt_alignment_number}"
        rta_dir.mkdir(parents=True, exist_ok=True)
        paths["rt_alignment_output_dir"] = str(rta_dir)
        paths["aligned_atlases_store_file"] = str(rta_dir / "rt_aligned_atlases.csv")

    if stage in ["auto_identification", "analysis_gui", "analysis_summary"]:
        if analysis_number is None:
            raise ValueError("Both RT alignment number and analysis number must be provided for this stage.")
        analysis_dir = project_dir / f"RTA{rt_alignment_number}" / f"TGA{analysis_number}"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        paths["analysis_output_dir"] = str(analysis_dir)
        paths["auto_ided_atlases_store_file"] = str(analysis_dir / "auto_ided_atlases.csv")
        paths["curated_atlases_store_file"] = str(analysis_dir / "curated_atlases.csv")

    return paths

@dataclass
class Compound:
    """
    Object-oriented representation of the compounds database table.
    Contains immutable chemical compound metadata.
    """
    
    # Core identifiers (required)
    compound_uid: str
    compound_name: str
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
            compound_name=row.get('compound_name', ''),
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
            'compound_name': self.compound_name,
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

    @classmethod
    def create_from_config(cls, config_path: str, overwrite_db: bool = False) -> Tuple[List['Compound'], List['CompoundMZRT']]:
        """Create and save Compounds and CompoundMZRTs from a config file."""
        config = ldt.load_compound_config(config_path)
        main_db_path = str(Path("/pscratch/sd/b/bkieft/metatlas_lite_data/databases/metatlas.duckdb"))

        compounds = []
        compound_mzrts = []
        dbi.create_metatlas_database(main_db_path, overwrite=overwrite_db)

        for chrom, pol_dict in config['COMPOUNDS'].items():
            for pol, pol_config in pol_dict.items():
                for file_path in pol_config.get('PATHS', []):
                    if not file_path:
                        logger.debug(f"Skipping empty file path for {chrom}/{pol}")
                        continue
                    logger.info(f"Processing compound file: {file_path}")
                    compounds_df = ldt.load_compound_input(file_path)
                    pcr.retrieve_pubchem_info(
                        compounds=compounds_df,
                        pubchem_cache_path=str(Path("/pscratch/sd/b/bkieft/metatlas_lite_data/databases/pubchem_cache/pubchem_global_cache.parquet")),
                        use_pubchem_cache=config["PARAMS"].get("use_pubchem_cache", True),
                        update_pubchem_cache=config["PARAMS"].get("update_pubchem_cache", False)
                    )
                    for _, row in compounds_df.iterrows():
                        try:
                            compound = cls.from_atlas_row(row)
                            compounds.append(compound)
                            compound_mzrt = CompoundMZRT.from_atlas_row(row)
                            compound_mzrt.source = file_path
                            compound_mzrts.append(compound_mzrt)
                        except Exception as e:
                            logger.warning(f"Failed to create Compound/CompoundMZRT for row {row.get('compound_name', 'Unknown')}: {e}")

        dbi.batch_save_compounds_and_mzrts(main_db_path, compounds, compound_mzrts)
        return

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
    compound_name: str = ""
    inchi_key: str = ""
    adduct: str = ""

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
    source: str = ""
    ms1_notes: str = ""
    ms2_notes: str = ""
    other_notes: str = ""
    analyst_notes: str = ""
    identification_notes: str = ""
    created_by: str = ""
    created_date: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundMZRT':
        """Create from atlas DataFrame row."""
        return cls(
            mz_rt_uid=row.get('mz_rt_uid', ''),
            compound_uid=row.get('compound_uid', ''),
            compound_name=row.get('compound_name', ''),
            inchi_key=row.get('inchi_key', ''),
            adduct=row.get('adduct', ''),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz=row.get('mz', 0.0),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            chromatography=row.get('chromatography', ''),
            polarity=row.get('polarity', ''),
            confidence=row.get('confidence', ''),
            source=row.get('source', ''),
            ms1_notes=row.get('ms1_notes', ''),
            ms2_notes=row.get('ms2_notes', ''),
            other_notes=row.get('other_notes', ''),
            analyst_notes=row.get('analyst_notes', ''),
            identification_notes=row.get('identification_notes', ''),
            created_by=row.get('created_by', ''),
            created_date=row.get('created_date', '')
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
            'mz_rt_uid': self.mz_rt_uid,
            'compound_uid': self.compound_uid,
            'compound_name': self.compound_name,
            'inchi_key': self.inchi_key,
            'adduct': self.adduct,
            'rt_peak': self.rt_peak,
            'rt_min': self.rt_min,
            'rt_max': self.rt_max,
            'mz': self.mz,
            'mz_tolerance': self.mz_tolerance,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'confidence': self.confidence,
            'source': self.source,
            'ms1_notes': self.ms1_notes,
            'ms2_notes': self.ms2_notes,
            'other_notes': self.other_notes,
            'analyst_notes': self.analyst_notes,
            'identification_notes': self.identification_notes,
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
    atlas_type: str
    
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
        def check_required(field, text):
            if not field:
                raise ValueError(f"Atlas {text} is missing")

        check_required(self.atlas_uid, "UID")
        check_required(self.atlas_name, "atlas_name")
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

    @classmethod
    def create_from_config(cls, config_path: str) -> List['Atlas']:
        """Create and save Atlas objects from a config file."""
        config = ldt.load_atlas_config(config_path)
        main_db_path = str(Path("/pscratch/sd/b/bkieft/metatlas_lite_data/databases/metatlas.duckdb"))
        
        atlases = []
        summary = []
        for chrom, pol_dict in config['ATLASES'].items():
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
                            atlas_file_path=atlas_info.get('path', 'No File Origin Provided'),
                            main_db_path=main_db_path
                        )
                        dbi.save_atlas_to_database(atlas_obj, main_db_path)
                        logger.info(f"Successfully created atlas: {atlas_obj.atlas_name}")
                        atlas_obj.validate()
                        atlases.append(atlas_obj)
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
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
        }

@dataclass
class Project:
    project_name: str = field(default_factory=str)
    config_path: str = field(default_factory=str)
    paths: Dict[str, str] = field(default_factory=dict)
    lcmsruns: List['LCMSRun'] = field(default_factory=list)

    def setup(self, project_name: str, config_path: str, overwrite_existing: bool = False):

        self.project_name = project_name
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(self.config_path)

        logger.info(f"Setting up workflow paths for project {self.project_name}...")
        self.paths = _set_up_paths(
            self.config, 
            self.project_name, 
            "project_setup"
        )

        logger.info(f"Creating project database at {self.paths['project_db_path']}...")
        exists = dbi.create_project_database(
            project_db_path=self.paths['project_db_path'],
            overwrite=overwrite_existing
        )
        if exists:
            return
        
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
    align_atlas_uid: Optional[str] = None
    align_atlas_obj: Optional[Atlas] = None
    rt_alignment_params: Dict[str, Any] = field(default_factory=dict)
    aligner_lcmsruns: List[LCMSRun] = field(default_factory=list)
    modeling_data: Optional[pd.DataFrame] = field(default_factory=pd.DataFrame)

    # Attributes modified by external functions
    rt_shift_stats: Dict[str, Any] = field(default_factory=dict)
    rt_aligned_atlases: Dict[str, Atlas] = field(default_factory=dict)
    rt_alignment_model: Optional[Dict[str, Any]] = None

    # Paths and config
    config_path: str = None
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(self, config_path: str, project_name: str, rt_alignment_number: int):
        """
        Set up RTAlign object using a Project object and RT alignment parameters.
        Populates paths, config, and relevant atlas UID.
        """

        logger.info(f"Setting up RTAlign object with RT alignment number {rt_alignment_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.project_name = project_name
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(self.config_path)
        self.chromatography = next(iter(self.config["WORKFLOWS"]["RT_ALIGNMENT"].keys()))
        self.align_atlas_uid = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('ATLAS', {}).get('uid', None)
        self.rt_alignment_params = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('PARAMS', {})

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="rt_alignment",
            rt_alignment_number=self.rt_alignment_number
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

        return rta_model

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
    analysis_subset: Optional[List[str]] = None
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    autoid_lcmsruns: List[LCMSRun] = field(default_factory=list)
    experimental_data: Optional[ExperimentalData] = None
    pre_autoid_atlas_obj: Optional[Atlas] = None
    post_autoid_atlas_obj: Optional[Atlas] = None

    # Paths and config
    config_path: str = None
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(self, config_path: str, project_name: str, rt_alignment_number: int, analysis_number: int, analysis_subset: Optional[List[str]] = None):
        """
        Set up AutoIdentification object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AutoIdentification object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(self.config_path)
        self.project_name = project_name
        self.analysis_subset = analysis_subset
        self.chromatography = next(iter(self.config["WORKFLOWS"]["TARGETED_ANALYSES"].keys()))

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="auto_identification",
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number
        )

class AnalysisGUI:
    # Core metadata
    analysis_uid: str = None
    rt_alignment_number: int = None
    analysis_number: int = None
    chromatography: str = None
    polarity: str = None
    created_by: str = None
    created_date: str = None

    # Paths and config
    config_path: str = None
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    # Attributes added during analysis
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    pre_curation_atlas_obj: Optional[Atlas] = None
    post_curation_atlas_obj: Optional[Atlas] = None

    # Dfs for in-memory GUI analysis
    manual_curation_df:  Optional[pd.DataFrame] = None
    ms1_df: Optional[pd.DataFrame] = None
    ms2_df: Optional[pd.DataFrame] = None
    ms2_hits_df: Optional[pd.DataFrame] = None
    override_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(self, config_path: str, project_name: str, rt_alignment_number: int, analysis_number: int):
        """
        Set up AnalysisGUI object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AnalysisGUI object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(self.config_path)
        self.project_name = project_name

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="analysis_gui",
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number
        )

class AnalysisSummary:
    # Core metadata
    rt_alignment_number: int = None
    analysis_number: int = None
    chromatography: str = None
    polarity: str = None

    # Attributes added during analysis
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    pre_curation_atlas_obj: Optional[Atlas] = None
    post_curation_atlas_obj: Optional[Atlas] = None
    summary_data: Optional[pd.DataFrame] = None

    # Pre-loaded data tables (populated by load_data())
    manual_curation_df:  Optional[pd.DataFrame] = None
    ms1_all_df:          Optional[pd.DataFrame] = None
    ms2_raw_all_df:      Optional[pd.DataFrame] = None
    ms2_hits_all_df:     Optional[pd.DataFrame] = None
    per_file_metrics_df: Optional[pd.DataFrame] = None

    # Paths and config
    config_path: str = None
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def setup(
        self,
        config_path: str,
        project_name: str,
        rt_alignment_number: int,
        analysis_number: int,
    ):
        """
        Set up AnalysisSummary object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AnalysisSummary object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.chromatography = self.config["WORKFLOWS"].get("RT_ALIGNMENT", {}).keys()[0]
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(self.config_path)
        self.project_name = project_name

        logger.info(f"Setting up workflow paths...")
        self.paths = _set_up_paths(
            config=self.config,
            project_name=self.project_name,
            stage="analysis_summary",
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number
        )

        self.load_data()

    def load_data(self) -> None:
        """Load all analysis data tables from the project database and cache them as attributes.
        """
        if self.manual_curation_df is not None:
            logger.debug("AnalysisSummary.load_data: data already loaded, skipping.")
            return

        project_db_path  = self.paths["project_db_path"]
        rt_alignment_num = self.rt_alignment_number
        analysis_num     = self.analysis_number

        logger.info(
            "AnalysisSummary.load_data: loading RT alignment %d, analysis %d…",
            rt_alignment_num, analysis_num,
        )

        self.manual_curation_df = dbi.get_manual_curation_entries(
            project_db_path, rt_alignment_num, analysis_num
        )
        if self.manual_curation_df.empty:
            logger.error("load_data: no manual curation entries found.")
            return

        self.ms1_all_df = dbi.get_ms1_data_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num
        )
        self.ms2_raw_all_df = dbi.get_ms2_data_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num
        )
        self.ms2_hits_all_df = dbi.get_ms2_hits_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num
        )
        self.per_file_metrics_df = asm.extract_per_file_metrics(self.ms1_all_df)

        logger.info(
            "load_data complete: %d compounds, %d MS1 rows, %d MS2 raw rows, %d MS2 hit rows.",
            len(self.manual_curation_df),
            len(self.ms1_all_df),
            len(self.ms2_raw_all_df),
            len(self.ms2_hits_all_df),
        )