from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import os
from dataclasses import asdict

import metatlas2.database_interact as dbi
import metatlas2.load_tools as ldt
import metatlas2.pubchem_retrieval as pcr
import metatlas2.lcmsruns_tools as lrt
import metatlas2.analysis_summary as asm
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg
logger = lcf.get_logger('workflow_objects')

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
    def create_from_config(
        cls, 
        config_path: str, 
        overwrite_db: bool = False
    ) -> None:
        """Create and save Compounds and CompoundMZRTs from a config file."""
        config = ldt.load_compound_config(config_path)
        paths = rtg.set_up_paths(config=config)
        main_db_path = paths.get("main_db_path", None)
        pubchem_cache_path = paths.get("pubchem_cache_path", None)

        compounds = []
        compound_mzrts = []
        # This will only create the database if it doesn't already exist, unless you want to overwrite
        dbi.create_metatlas_database(main_db_path, overwrite=overwrite_db)

        for chrom, pol_dict in config['COMPOUNDS'].items():
            for pol, pol_config in pol_dict.items():
                for file_path in pol_config.get('PATHS', []):
                    if not file_path:
                        logger.debug(f"Skipping empty file path for {chrom}/{pol}")
                        continue
                    logger.info(f"Processing compound file: {file_path}")
                    compounds_df = ldt.load_compound_input(file_path)
                    compounds_df = pcr.retrieve_pubchem_info(
                        compounds=compounds_df,
                        pubchem_cache_path=pubchem_cache_path,
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
    rt_space: str = "HF"
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
            rt_space=row.get('rt_space', 'HF'),
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
            'rt_space': self.rt_space,
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

    def validate(self) -> None:
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
            compound_dict = asdict(compound_mzrt)
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
    def create_from_config(
        cls, 
        config_path: str
    ) -> None:
        """Create and save Atlas objects from a config file."""
        config = ldt.load_atlas_config(config_path)
        paths = rtg.set_up_paths(config=config)
        main_db_path = paths.get("main_db_path", None)
        
        atlases = []
        summary = []
        for chrom, pol_dict in config['ATLASES'].items():
            for pol, pol_config in pol_dict.items():
                for analysis_type, atlas_info in pol_config.items():
                    atlas_entries = atlas_info if isinstance(atlas_info, list) else [atlas_info]
                    for entry_index, atlas_entry in enumerate(atlas_entries, start=1):
                        if not atlas_entry.get('path'):
                            logger.debug(f"Skipping atlas with no path for {analysis_type}/{chrom}/{pol}")
                            continue
                        try:
                            atlas_compounds_df = ldt.load_atlas_input(atlas_entry['path'])
                            atlas_name = atlas_entry.get('name', 'Unnamed Atlas')
                            if len(atlas_entries) > 1 and not atlas_name:
                                atlas_name = f"{analysis_type} {chrom} {pol} #{entry_index}"
                            atlas_obj = dbi.create_new_atlas_from_dataframe(
                                atlas_df=atlas_compounds_df,
                                atlas_name=atlas_name,
                                atlas_description=atlas_entry.get('desc', 'No Description'),
                                analysis_type=analysis_type,
                                chromatography=chrom,
                                polarity=pol,
                                atlas_file_path=atlas_entry.get('path', 'No File Origin Provided'),
                                main_db_path=main_db_path
                            )
                            dbi.save_atlas_to_database(atlas_obj, main_db_path)
                            logger.info(f"Successfully created atlas: {atlas_obj.atlas_name}")
                            atlases.append(atlas_obj)
                            summary.append({
                                'atlas_uid': atlas_obj.atlas_uid,
                                'atlas_name': atlas_obj.atlas_name,
                                'compound_count': len(atlas_obj.compound_mzrts)
                            })
                        except Exception as e:
                            logger.error(f"Failed to create Atlas for {analysis_type}/{chrom}/{pol} entry {entry_index}: {e}")

        logger.info("Summary of new atlases.")
        logger.info("**Make sure to add these to your analysis config to use as project reference atlases:**")
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
    config: Dict[str, Any] = field(default_factory=dict)
    project_name: str = field(default="")
    paths: Dict[str, str] = field(default_factory=dict)
    lcmsruns: List['LCMSRun'] = field(default_factory=list)

    def setup(self, project_name: str, config: Dict[str, Any], paths: Dict[str, str], overwrite_existing: bool = False):

        self.project_name = project_name
        self.config = config
        self.paths = paths

        logger.info(f"Creating project database at {self.paths['project_db_path']}...")
        exists = dbi.create_project_database(
            project_db_path=self.paths['project_db_path'],
            rt_align_path=self.paths['rt_alignment_output_dir'],
            log_file_path=self.paths['log_path'],
            overwrite=overwrite_existing
        )
        
        # Register project in main database for meta-analysis tracking
        if 'main_db_path' in self.paths:
            logger.info(f"Registering project '{self.project_name}' in main database...")
            dbi.save_project_to_main_db(
                main_db_path=self.paths['main_db_path'],
                project_name=self.project_name,
                project_db_path=self.paths['project_db_path']
            )
        
        if exists:
            return
        
        logger.info(f"Loading LCMS runs...")
        lcmsruns_list = lrt.get_project_lcmsruns_from_disk(
            self.paths['lcmsruns_directory']
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
    run_alignment: bool = True

    # Attributes added during analysis
    align_atlas_uid: Optional[str] = None
    align_atlas_obj: Optional[Atlas] = None
    rt_alignment_params: Dict[str, Any] = field(default_factory=dict)
    aligner_lcmsruns: List[LCMSRun] = field(default_factory=list)
    modeling_data: Optional[pd.DataFrame] = field(default=None)

    # Attributes modified by external functions
    rt_shift_stats: Dict[str, Any] = field(default_factory=dict)
    rt_aligned_atlases: Dict[str, Atlas] = field(default_factory=dict)
    rt_alignment_model: Optional[Dict[str, Any]] = None

    # Paths and config
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def setup(self, project_name: str, rt_alignment_number: int, config: Dict[str, Any], paths: Dict[str, str]):
        """
        Set up RTAlign object using a Project object and RT alignment parameters.
        Populates paths, config, and relevant atlas UID.
        """

        logger.info(f"Setting up RTAlign object with RT alignment number {rt_alignment_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.project_name = project_name
        self.config = config
        self.paths = paths
        self.chromatography = next(iter(self.config["WORKFLOWS"]["RT_ALIGNMENT"].keys()))
        self.align_atlas_uid = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('ATLAS', {}).get('uid', None)
        self.rt_alignment_params = self.config['WORKFLOWS']['RT_ALIGNMENT'][self.chromatography].get('PARAMS', {})

        if self.rt_alignment_params.get('do_alignment', True) is False: # still write the csv for auto id to find the config atlases in text file
            self.run_alignment = False
            logger.info(f"RT alignment is disabled in config. Writing atlases from config to {self.paths['aligned_atlases_store_file']} and exiting...")
            for chrom, pol_dict in self.config['WORKFLOWS']['TARGETED_ANALYSES'].items():
                for pol, analysis_dict in pol_dict.items():
                    for analysis_type, atlas_config in analysis_dict.items():
                        atlas_uid = atlas_config['ATLAS']['uid']
                        atlas_obj = Atlas.from_database(
                            database_path=self.paths['main_db_path'],
                            atlas_uid=atlas_uid
                        )
                        ldt.save_atlas_data_to_csv(
                            atlas_obj=atlas_obj,
                            output_path=self.paths['aligned_atlases_store_file'],
                        )
            return

        logger.info(f"Checking for existing RT aligned atlases table file at {self.paths['aligned_atlases_store_file']}...")
        if os.path.exists(self.paths['aligned_atlases_store_file']) and self.rt_alignment_params.get('use_existing_rt_alignment', False) is True:
            logger.info(f"Aligned atlases have already been generated for RT alignment number {self.rt_alignment_number} at {self.paths['aligned_atlases_store_file']}, not overwriting since use_existing_rt_alignment is set to True in config.")
            self.run_alignment = False
        elif os.path.exists(self.paths['aligned_atlases_store_file']) and self.rt_alignment_params.get('use_existing_rt_alignment', False) is False:
            logger.warning(f"Aligned atlases file already exists at {self.paths['aligned_atlases_store_file']} for RT alignment number {self.rt_alignment_number}, but overwriting existing file since use_existing_rt_alignment is set to False in config.")
            Path(self.paths['aligned_atlases_store_file']).unlink()
        elif not os.path.exists(self.paths['aligned_atlases_store_file']):
            logger.info(f"No existing aligned atlases file found at {self.paths['aligned_atlases_store_file']} for RT alignment number {self.rt_alignment_number}. Aligned atlases will be generated.")

@dataclass
class ManualCuration:
    inchi_key: str
    adduct: str
    data: pd.DataFrame = field(default=None, compare=False)
    # data columns: compound_uid, inchi_key, adduct, rt_alignment_number, analysis_number,
    #   compound_name, auto_ided, polarity, chromatography, mz_tolerance, atlas_mz,
    #   atlas_rt_peak, atlas_rt_min, atlas_rt_max, original_rt_peak, original_rt_min,
    #   original_rt_max, rt_peak, rt_min, rt_max, ms1_notes, ms2_notes, other_notes,
    #   identification_notes, analyst_notes, best_ms1_file, best_ms1_rt, best_ms1_mz,
    #   best_ms1_intensity, best_ms1_ppm_error, best_ms1_rt_error, isomers,
    #   suggested_rt_min, suggested_rt_max, suggested_rt_peak, rt_suggestion_confidence


@dataclass
class _SpecData:
    inchi_key: str
    adduct: str
    filename: str
    data: pd.DataFrame = field(default=None, compare=False)


MS1Data = _SpecData
# data columns: rt, mz, i

MS2Data = _SpecData
# data columns: rt, mz, i, precursor_MZ, precursor_intensity, collision_energy

MS2Hit = _SpecData
# data columns: inchi_key, database, ref_id, ref_name, score, num_matches,
#   mz_theoretical, mz_measured, ppm_error, rt, qry_intensity_peak,
#   ref_frags, data_frags, matched_fragments, aligned_fragment_colors,
#   qry_spectrum, ref_spectrum


@dataclass
class ExperimentalData:
    manual_curation: List[ManualCuration] = field(default_factory=list)
    ms1_data: List[_SpecData] = field(default_factory=list)
    ms2_data: List[_SpecData] = field(default_factory=list)
    ms2_hits: List[_SpecData] = field(default_factory=list)

@dataclass
class AutoIdentification:
    # Core metadata
    project_name: str = None
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
    image_tag: str = "latest"
    paths: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int, config: Dict[str, Any], paths: Dict[str, str], analysis_subset: Optional[List[str]] = None, config_path: str = None, image_tag: str = "latest"):
        """
        Set up AutoIdentification object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AutoIdentification object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.config = config
        self.paths = paths
        self.project_name = project_name
        self.analysis_subset = analysis_subset
        self.config_path = config_path
        self.image_tag = image_tag
        self.chromatography = next(iter(self.config["WORKFLOWS"]["TARGETED_ANALYSES"].keys()))

@dataclass
class AnalysisGUI:
    # Core metadata
    project_name: str = None
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
    post_autoid_atlas_obj: Optional[Atlas] = None
    post_curation_atlas_obj: Optional[Atlas] = None

    # ExperimentalData loaded from the DB (filtered by override thresholds)
    experimental_data: Optional[Any] = None

    # Dfs for in-memory GUI analysis (derived from experimental_data)
    manual_curation_df:  Optional[pd.DataFrame] = None
    ms1_df: Optional[pd.DataFrame] = None
    ms2_df: Optional[pd.DataFrame] = None
    ms2_hits_df: Optional[pd.DataFrame] = None
    override_parameters: Dict[str, Any] = field(default_factory=dict)

    def setup(self, config_path: str, project_name: str, rt_alignment_number: int, analysis_number: int):
        """
        Set up AnalysisGUI object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AnalysisGUI object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.config_path = config_path
        self.project_name = project_name
        self.config = ldt.load_metatlas2_config(config_path)
        self.paths = rtg.set_up_paths(config=self.config, project_name=self.project_name, rt_alignment_number=self.rt_alignment_number, analysis_number=self.analysis_number)

@dataclass
class AnalysisSummary:
    # Core metadata
    rt_alignment_number: int = None
    analysis_number: int = None
    chromatography: str = None
    polarity: str = None

    # Attributes added during analysis
    workflow_params: Dict[str, Any] = field(default_factory=dict)
    override_parameters: Dict[str, Any] = field(default_factory=dict)
    post_autoid_atlas_obj: Optional[Atlas] = None
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
        self.config_path = config_path
        self.config = ldt.load_metatlas2_config(config_path)
        self.chromatography = next(iter(self.config["WORKFLOWS"]["TARGETED_ANALYSES"].keys()))
        self.project_name = project_name
        self.paths = rtg.set_up_paths(config=self.config, project_name=self.project_name, rt_alignment_number=self.rt_alignment_number, analysis_number=self.analysis_number)

    def load_data(self) -> None:
        """Load all analysis data tables from the project database and cache them as attributes.
        """
        if self.manual_curation_df is not None:
            logger.debug("AnalysisSummary.load_data: data already loaded, skipping.")
            return

        project_db_path  = self.paths["project_db_path"]
        rt_alignment_num = self.rt_alignment_number
        analysis_num     = self.analysis_number
        analysis_type    = self.post_curation_atlas_obj.analysis_type

        logger.info(
            "AnalysisSummary.load_data: loading RT alignment %d, analysis %d, analysis_type %s…",
            rt_alignment_num, analysis_num, analysis_type,
        )

        atlas_compounds = None
        if self.post_curation_atlas_obj and self.post_curation_atlas_obj.compound_mzrts:
            import pandas as _pd
            atlas_compounds = _pd.DataFrame([
                {'inchi_key': c.inchi_key, 'adduct': c.adduct}
                for c in self.post_curation_atlas_obj.compound_mzrts.values()
            ])

        self.manual_curation_df = dbi.get_manual_curation_entries(
            project_db_path, rt_alignment_num, analysis_num,
            remove_unidentified_compounds=None,
            atlas_compounds=atlas_compounds,
            analysis_type=analysis_type,
        )
        if self.manual_curation_df.empty:
            logger.error("load_data: no manual curation entries found.")
            return

        self.ms1_all_df = dbi.get_ms1_data_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num,
            atlas_compounds=atlas_compounds, analysis_type=analysis_type,
        )
        self.ms2_raw_all_df = dbi.get_ms2_data_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num,
            atlas_compounds=atlas_compounds, analysis_type=analysis_type,
        )
        self.ms2_hits_all_df = dbi.get_ms2_hits_for_compound(
            project_db_path, None, None, rt_alignment_num, analysis_num,
            atlas_compounds=atlas_compounds, analysis_type=analysis_type,
        )
        self.per_file_metrics_df = asm.extract_per_file_metrics(self.ms1_all_df)

        logger.info(
            "load_data complete: %d compounds, %d MS1 rows, %d MS2 raw rows, %d MS2 hit rows.",
            len(self.manual_curation_df),
            len(self.ms1_all_df),
            len(self.ms2_raw_all_df),
            len(self.ms2_hits_all_df),
        )