from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import os
import yaml
from dataclasses import asdict
from tqdm.auto import tqdm

import metatlas2.database_interact as dbi
import metatlas2.load_tools as ldt
import metatlas2.pubchem_retrieval as pcr
import metatlas2.lcmsruns_tools as lrt
import metatlas2.analysis_summary as asm
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg
import metatlas2.note_options as gno
import metatlas2.file_and_project_format as fpf
logger = lcf.get_logger('workflow_objects')


@dataclass
class TargetedAnalysis:
    """Configuration for a single named targeted analysis entry.

    The unique identity of an analysis is the combination of
    ``chromatography + polarity + analysis_type + name``.
    ``atlas_uid`` is the reference atlas to use, and ``params`` holds
    the validated PARAMS dict for this entry.
    """
    chromatography: str
    polarity: str
    analysis_type: str
    analysis_name: str
    atlas_uid: str
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Human-readable composite key: ``chrom/pol/analysis_type/name``."""
        return f"{self.chromatography}/{self.polarity}/{self.analysis_type}/{self.analysis_name}"


@dataclass
class Metatlas2Config:
    """Parsed representation of a metatlas2 YAML configuration file.

    Attributes
    ----------
    paths_config:
        The ``WORKFLOWS.PATHS`` section as a plain dict (owner, msms_refs_path, etc.).
    rt_alignment_config:
        The ``WORKFLOWS.RT_ALIGNMENT`` section as a plain dict keyed by chromatography.
    targeted_analyses:
        Flat list of :class:`TargetedAnalysis` objects — one per unique
        ``chrom/pol/analysis_type/name`` combination.  Because the YAML
        loader merges duplicate analysis-type keys, **all** named analyses
        are present even when the YAML file repeats the same key (e.g.
        multiple ``EMA:`` blocks under the same polarity).
    """
    paths_config: Dict[str, Any] = field(default_factory=dict)
    rt_alignment_config: Dict[str, Any] = field(default_factory=dict)
    targeted_analyses: List[TargetedAnalysis] = field(default_factory=list)

    def get_targeted_analysis(
        self,
        chromatography: str,
        polarity: str,
        analysis_type: str,
        analysis_name: str,
    ) -> "TargetedAnalysis":
        """Return the :class:`TargetedAnalysis` matching the given key.

        Raises :class:`KeyError` if no match is found.
        """
        for ta in self.targeted_analyses:
            if (
                ta.chromatography == chromatography
                and ta.polarity == polarity
                and ta.analysis_type == analysis_type
                and ta.analysis_name == analysis_name
            ):
                return ta
        raise KeyError(
            f"No targeted analysis found for {chromatography}/{polarity}/{analysis_type}/{analysis_name}. "
            f"Available: {[ta.key for ta in self.targeted_analyses]}"
        )

    @property
    def chromatography(self) -> str:
        """First chromatography key found in targeted analyses (convenience accessor)."""
        if self.targeted_analyses:
            return self.targeted_analyses[0].chromatography
        if self.rt_alignment_config:
            return next(iter(self.rt_alignment_config))
        raise ValueError("No chromatography found in config")

    @property
    def owner(self) -> str:
        return (self.paths_config.get('owner') or 'jgi').lower()

    @property
    def msms_refs_path(self) -> Optional[str]:
        return self.paths_config.get('msms_refs_path', None)

    @property
    def msms_refs_db_filter(self) -> Optional[str]:
        return self.paths_config.get('msms_refs_db_filter', None)

    @property
    def gdrive_subfolder(self) -> Optional[str]:
        return self.paths_config.get('gdrive_subfolder', None)

@dataclass
class AtlasEntry:
    """One atlas input file with its database name and description."""
    path: Optional[str]
    name: Optional[str]
    desc: Optional[str]

    def __post_init__(self):
        self.path = str(self.path) if self.path else None
        self.name = str(self.name) if self.name else None
        self.desc = str(self.desc) if self.desc else None
        if self.path and not Path(self.path).exists():
            raise FileNotFoundError(f"Atlas file not found: {self.path}")

    @property
    def is_empty(self) -> bool:
        return not self.path


@dataclass
class NewAtlasesConfig:
    """Parsed representation of a ``create_atlases.yaml`` config file.

    Structure::

        atlases[chromatography][polarity][analysis_type] -> List[AtlasEntry]

    All keys are preserved exactly as written in the YAML (e.g. ``'HILICZ'``,
    ``'POS'``, ``'QC'``).
    """
    atlases: Dict[str, Dict[str, Dict[str, List[AtlasEntry]]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewAtlasesConfig":
        """Load and validate an atlas config YAML, returning a NewAtlasesConfig."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if 'ATLASES' not in raw:
            raise ValueError(f"Missing required top-level key 'ATLASES' in {path}")

        atlases: Dict[str, Dict[str, Dict[str, List[AtlasEntry]]]] = {}
        for chrom, chrom_cfg in raw['ATLASES'].items():
            if not isinstance(chrom_cfg, dict):
                raise ValueError(f"Expected a dict under ATLASES.{chrom}")
            atlases[chrom] = {}
            for pol, pol_cfg in chrom_cfg.items():
                if not isinstance(pol_cfg, dict):
                    raise ValueError(f"Expected a dict under ATLASES.{chrom}.{pol}")
                atlases[chrom][pol] = {}
                for analysis_type, entries_raw in pol_cfg.items():
                    if isinstance(entries_raw, dict):
                        entries_raw = [entries_raw]
                    elif not isinstance(entries_raw, list):
                        raise ValueError(
                            f"Expected a dict or list under ATLASES.{chrom}.{pol}.{analysis_type}"
                        )
                    entries = []
                    for e in entries_raw:
                        if not isinstance(e, dict):
                            raise ValueError(
                                f"Expected a dict entry under ATLASES.{chrom}.{pol}.{analysis_type}"
                            )
                        for required in ('path', 'name', 'desc'):
                            if required not in e:
                                raise ValueError(
                                    f"Missing required field '{required}' in "
                                    f"ATLASES.{chrom}.{pol}.{analysis_type}"
                                )
                        entries.append(AtlasEntry(path=e['path'], name=e['name'], desc=e['desc']))
                    atlases[chrom][pol][analysis_type] = entries

        logger.info("Loaded atlas configuration from %s", path)
        return cls(atlases=atlases)

    def iter_entries(self):
        """Yield (chromatography, polarity, analysis_type, AtlasEntry) tuples."""
        for chrom, pol_dict in self.atlases.items():
            for pol, type_dict in pol_dict.items():
                for analysis_type, entries in type_dict.items():
                    for entry in entries:
                        yield chrom, pol, analysis_type, entry


@dataclass
class CompoundParams:
    """Parameters section of a ``create_compounds.yaml`` config file."""
    use_pubchem_cache: bool = True
    update_pubchem_cache: bool = False


@dataclass
class NewCompoundsConfig:
    """Parsed representation of a ``create_compounds.yaml`` config file.

    Structure::

        compounds[chromatography][polarity] -> List[str]  (validated file paths)

    All keys are preserved exactly as written in the YAML.
    """
    params: CompoundParams
    compounds: Dict[str, Dict[str, List[str]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewCompoundsConfig":
        """Load and validate a compound config YAML, returning a NewCompoundsConfig."""
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        for required in ('PARAMS', 'COMPOUNDS'):
            if required not in raw:
                raise ValueError(f"Missing required top-level key '{required}' in {path}")

        p = raw['PARAMS']
        params = CompoundParams(
            use_pubchem_cache=bool(p.get('use_pubchem_cache', True)),
            update_pubchem_cache=bool(p.get('update_pubchem_cache', False)),
        )

        compounds: Dict[str, Dict[str, List[str]]] = {}
        for chrom, chrom_cfg in raw['COMPOUNDS'].items():
            if not isinstance(chrom_cfg, dict):
                raise ValueError(f"Expected a dict under COMPOUNDS.{chrom}")
            compounds[chrom] = {}
            for pol, pol_cfg in chrom_cfg.items():
                if not isinstance(pol_cfg, dict):
                    raise ValueError(f"Expected a dict under COMPOUNDS.{chrom}.{pol}")
                if 'PATHS' not in pol_cfg:
                    raise ValueError(f"Missing PATHS under COMPOUNDS.{chrom}.{pol}")
                if not isinstance(pol_cfg['PATHS'], list):
                    raise ValueError(f"PATHS must be a list under COMPOUNDS.{chrom}.{pol}")
                validated: List[str] = []
                for p_raw in pol_cfg['PATHS']:
                    if not p_raw:
                        continue
                    p_str = str(p_raw)
                    if not Path(p_str).exists():
                        raise FileNotFoundError(
                            f"Compound input file not found: {p_str} for {chrom}/{pol}"
                        )
                    validated.append(p_str)
                compounds[chrom][pol] = validated

        logger.info("Loaded compound configuration from %s", path)
        return cls(params=params, compounds=compounds)

    def iter_paths(self):
        """Yield (chromatography, polarity, file_path) tuples for all non-empty paths."""
        for chrom, pol_dict in self.compounds.items():
            for pol, paths in pol_dict.items():
                for file_path in paths:
                    yield chrom, pol, file_path


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
    classes: str = ""
    pathways: str = ""
    tags: str = ""
    
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
            classes=row.get('classes', ''),
            pathways=row.get('pathways', ''),
            tags=row.get('tags', ''),
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
            'classes': self.classes,
            'pathways': self.pathways,
            'tags': self.tags,
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
        config = NewCompoundsConfig.from_yaml(config_path)
        paths = rtg.set_up_paths(config=config)
        main_db_path = paths.get("main_db_path", None)
        pubchem_cache_path = paths.get("pubchem_cache_path", None)

        compounds = []
        # This will only create the database if it doesn't already exist, unless you want to overwrite
        dbi.create_metatlas_database(main_db_path, overwrite=overwrite_db)

        for chrom, pol, file_path in config.iter_paths():
            logger.info(f"Processing compound file: {file_path}")
            compounds_df = ldt.load_compound_input(file_path)
            compounds_df = pcr.retrieve_pubchem_info(
                compounds=compounds_df,
                pubchem_cache_path=pubchem_cache_path,
                use_pubchem_cache=config.params.use_pubchem_cache,
                update_pubchem_cache=config.params.update_pubchem_cache,
            )
            for _, row in compounds_df.iterrows():
                try:
                    compound = cls.from_atlas_row(row)
                    compounds.append(compound)
                    compound_mzrt = CompoundMZRT.from_atlas_row(row)
                    compound_mzrt.source = file_path
                except Exception as e:
                    logger.warning(f"Failed to create Compound/CompoundMZRT for row {row.get('compound_name', 'Unknown')}: {e}")

        dbi.batch_save_compounds(main_db_path, compounds)
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
    rt_space: str = ""
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
            rt_space=row.get('rt_space', 'HF_Aug2019'),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz=row.get('mz', 0.0),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            chromatography=row.get('chromatography', ''),
            polarity=row.get('polarity', ''),
            confidence=row.get('confidence', ''),
            source=row.get('source', ''),
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
    analysis_name: str = "MAIN_ATLAS"

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

        logger.debug(f"Validating atlas {self.atlas_name} (UID: {self.atlas_uid}) with {len(self.compound_mzrts)} compounds...")

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

        logger.debug("Atlas passed validation!")

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
                'source_atlas_uid': self.source_atlas_uid,
                'analysis_name': self.analysis_name,
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
        analysis_name = meta.get('analysis_name', 'MAIN_ATLAS')

        # Use mz_rt_uid as the unique key for each CompoundMZRT
        compound_mzrts = {}
        for _, row in atlas_df.iterrows():
            compound_mzrt = CompoundMZRT.from_atlas_row(row)
            mz_rt_uid = getattr(compound_mzrt, 'mz_rt_uid', None)
            if not mz_rt_uid:
                raise ValueError("Each row must have a valid mz_rt_uid to be used as a unique key in compound_mzrts.")
            compound_mzrts[mz_rt_uid] = compound_mzrt

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
            source_atlas_uid=source_atlas_uid,
            analysis_name=analysis_name,
        )

    @classmethod
    def from_database(cls, database_path: str, atlas_uid: str, main_db_path: str = None):
        """
        Load an Atlas object directly from the database using its UID.
        """
        logger.debug(f"Loading atlas with UID {atlas_uid} from database...")
        return cls.from_dataframe(dbi.get_atlas_compounds_table(database_path, atlas_uid, main_db_path))

    @classmethod
    def create_from_config(
        cls, 
        config_path: str
    ) -> None:
        """
        Create and save Atlas objects from a config file.
        """
        config = NewAtlasesConfig.from_yaml(config_path)
        paths = rtg.set_up_paths(config=config)
        main_db_path = paths.get("main_db_path", None)

        atlases = []
        summary = []

        # Flatten all atlas entries into a list for a single progress bar
        atlas_jobs = [
            (chrom, pol, analysis_type, entry)
            for chrom, pol, analysis_type, entry in config.iter_entries()
        ]

        for chrom, pol, analysis_type, entry in tqdm(atlas_jobs, desc="Creating atlases from config"):
            if entry.is_empty:
                logger.debug(f"Skipping atlas with no path for {analysis_type}/{chrom}/{pol}")
                continue
            try:
                atlas_compounds_df = ldt.load_atlas_input(entry.path)
                atlas_obj = dbi.create_new_atlas_from_dataframe(
                    atlas_df=atlas_compounds_df,
                    atlas_name=entry.name,
                    atlas_description=entry.desc,
                    analysis_type=analysis_type,
                    chromatography=chrom,
                    polarity=pol,
                    atlas_file_path=entry.path,
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
                logger.error(f"Failed to create Atlas for {analysis_type}/{chrom}/{pol}: {e}")

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
            'source_atlas_uid': self.source_atlas_uid,
            'analysis_name': self.analysis_name,
        }

@dataclass
class Project:
    config: Dict[str, Any] = field(default_factory=dict)
    project_name: str = field(default="")
    paths: Dict[str, str] = field(default_factory=dict)
    lcmsruns: List['LCMSRun'] = field(default_factory=list)

    def setup(self, project_name: str, config: Dict[str, Any], paths: Dict[str, str], overwrite_existing: bool = False, config_path: str = None, rt_alignment_number: int = None, analysis_number: int = None):

        self.project_name = fpf.parse_project_name(project_name)
        self.config = config
        self.paths = paths

        logger.info(f"Creating project database at {self.paths['project_db_path']}...")
        exists = dbi.create_project_database(
            project_db_path=self.paths['project_db_path'],
            rt_align_path=self.paths['rt_alignment_output_dir'],
            log_file_path=self.paths['log_path'],
            overwrite=overwrite_existing
        )
        if exists:
            return
        
        logger.info(f"Registering project '{self.project_name}' in main database...")
        dbi.save_project_to_main_db(
            main_db_path=self.paths['main_db_path'],
            project_name=self.project_name,
            project_db_path=self.paths['project_db_path']
        )
        
        logger.info(f"Loading LCMS runs...")
        lcmsruns_list = lrt.get_project_lcmsruns_from_disk(
            self.paths['lcmsruns_directory']
        )

        logger.info(f"Saving {len(lcmsruns_list)} LCMS runs metadata to database...")
        dbi.save_lcmsruns_to_db(
            self.paths['project_db_path'],
            self.project_name,
            lcmsruns_list,
            overwrite_existing
        )

        logger.info(f"Storing LCMS runs in Project object...")
        self.lcmsruns = [LCMSRun(**row) for row in lcmsruns_list]

        logger.info(f"Saving config snapshot to database for RTA{rt_alignment_number}/TGA{analysis_number}...")
        dbi.save_config_to_db(
            project_db_path=self.paths['project_db_path'],
            config_path=config_path,
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
            paths=self.paths,
        )

        return

@dataclass
class LCMSRun:
    file_path: str
    filename: str
    file_format: str
    file_type: str
    chromatography: str
    ms_level: str
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
    config: Optional["Metatlas2Config"] = None

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int):
        """
        Set up RTAlign object by loading config and paths from the project database.

        Config and paths are loaded from the ``project_config`` table written
        during ``run_project_setup``.  No YAML file or ``set_up_paths()`` call
        is required.
        """

        logger.info(f"Setting up RTAlign object with RT alignment number {rt_alignment_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.project_name = project_name

        self.project_db_path = rtg.get_project_db_path(self.project_name) # hack to grab db path before paths is set

        self.paths = dbi.load_paths_from_db(
            project_db_path=self.project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )
        self.config = dbi.load_config_from_db(
            project_db_path=self.project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )

        self.chromatography = next(iter(self.config.rt_alignment_config.keys()))
        self.align_atlas_uid = self.config.rt_alignment_config[self.chromatography].get('ATLAS', {}).get('uid', None)
        self.rt_alignment_params = self.config.rt_alignment_config[self.chromatography].get('PARAMS', {})

        if self.rt_alignment_params.get('do_alignment', True) is False: # short circuit but still save template atlases to DB
            self.run_alignment = False
            self._skip_rt_align_routine()
            return

        self._check_existing_rt_aligned_atlases()

    def _skip_rt_align_routine(self):
        logger.info(
            "RT alignment is disabled in config. "
            "Copying reference atlases into project DB and registering in workflow_runs..."
        )
        for ta in self.config.targeted_analyses:
            atlas_obj = Atlas.from_database(
                database_path=self.paths['main_db_path'],
                atlas_uid=ta.atlas_uid
            )
            atlas_obj.analysis_name = ta.analysis_name
            atlas_obj.rt_alignment_number = self.rt_alignment_number
            atlas_obj.analysis_number = None
            dbi.save_atlas_to_database(
                atlas_obj=atlas_obj,
                db_path=self.paths['project_db_path'],
                main_db_path=self.paths['main_db_path'],
            )
            dbi.register_workflow_run(
                project_db_path=self.paths['project_db_path'],
                rt_alignment_number=self.rt_alignment_number,
                analysis_number=None,
                atlas_obj=atlas_obj,
                stage='RT_ALIGNED',
            )

    def _check_existing_rt_aligned_atlases(self):
        logger.info(f"Checking if RT-aligned atlases already exist for RTA{self.rt_alignment_number}...")
        aligned_count = dbi.count_workflow_runs(
            project_db_path=self.paths['project_db_path'],
            rt_alignment_number=self.rt_alignment_number,
            stage='RT_ALIGNED',
        )
        if aligned_count > 0:
            if self.rt_alignment_params.get('use_existing_rt_alignment', False):
                logger.info(
                    f"RT-aligned atlases already exist for RTA{self.rt_alignment_number} "
                    f"and use_existing_rt_alignment=True. Skipping RT alignment."
                )
                self.run_alignment = False
            else:
                raise ValueError(
                    f"RT-aligned atlases already exist for RTA{self.rt_alignment_number} "
                    f"but use_existing_rt_alignment=False in config. "
                    f"To run a new RT alignment, increment rt_alignment_number. "
                    f"To reuse the existing alignment, set use_existing_rt_alignment=True."
                )
@dataclass
class ExperimentalData:
    def __init__(self):
        self.ms1_df = pd.DataFrame() 
        self.ms2_df = pd.DataFrame() 
        self.curation_df = pd.DataFrame()

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
    image_tag: str = "latest"
    paths: Dict[str, str] = field(default_factory=dict)
    config: Optional["Metatlas2Config"] = None

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int, analysis_subset: Optional[List[str]] = None, image_tag: str = "latest"):
        """
        Set up AutoIdentification object by loading config and paths from the
        project database.

        Config and paths are loaded from the ``project_config`` table written
        during ``run_project_setup``.  No YAML file or ``set_up_paths()`` call
        is required.
        """
        logger.info(f"Setting up AutoIdentification object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.project_name = project_name
        self.analysis_subset = analysis_subset
        self.image_tag = image_tag

        project_db_path = rtg.get_project_db_path(self.project_name) # hack to grab db path before paths is set

        self.paths = dbi.load_paths_from_db(
            project_db_path=project_db_path,
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
        )
        self.config = dbi.load_config_from_db(
            project_db_path=project_db_path,
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
        )

        self.chromatography = self.config.chromatography
        self.msms_refs_db_filter = self.config.msms_refs_db_filter

        self._check_existing_auto_ided_atlases()

    def _check_existing_auto_ided_atlases(self):
        logger.debug(f"Checking if AutoID-aligned atlases already exist for RTA{self.rt_alignment_number} and TGA{self.analysis_number}...")
        autoided_count = dbi.count_workflow_runs(
            project_db_path=self.paths['project_db_path'],
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
            stage='AUTO_IDED',
        )
        if autoided_count > 0:
            raise ValueError(
                f"AutoID-aligned atlases already exist for RTA{self.rt_alignment_number} and TGA{self.analysis_number} "
                f"To run a new AutoID, increment analysis_number. "
            )

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
    config: Optional["Metatlas2Config"] = None
    notes: Dict[str, Any] = field(default_factory=dict)
    owner: str = "jgi"

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

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int, override_parameters: Optional[Dict[str, Any]] = None):
        """
        Set up AnalysisGUI object.
        Populates paths, config, and relevant atlas UID.
        """
        logger.info(f"Setting up AnalysisGUI object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.project_name = project_name

        project_db_path = rtg.get_project_db_path(self.project_name)
        self.config = dbi.load_config_from_db(
            project_db_path=project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )
        if self.config is None:
            raise RuntimeError(
                f"No config found in project database at {project_db_path} "
                f"for RTA={rt_alignment_number}, TGA={analysis_number}. "
                "Ensure run_project_setup was called with a config_path."
            )
        logger.info("Config loaded from project database.")

        self.owner = self.config.owner
        self.paths = rtg.set_up_paths(config=self.config, project_name=self.project_name, rt_alignment_number=self.rt_alignment_number, analysis_number=self.analysis_number)
        if override_parameters is not None:
            self.override_parameters = override_parameters
        dbi.validate_override_parameters(self.override_parameters)
        
    def get_note_options(self):
        """Get note options from config for manual curation."""
        
        ms2_notes_opts, ms1_notes_opts, other_notes_opts = gno.get_notes_opts(owner=self.owner)
        ms1_options, ms1_hotkeys = gno.get_note_options_and_hotkeys(
            self.override_parameters["note_options_overrides"].get("ms1_notes", {}) if self.override_parameters.get("note_options_overrides") else {},
            ms1_notes_opts,
        )
        ms2_options, ms2_hotkeys = gno.get_note_options_and_hotkeys(
            self.override_parameters["note_options_overrides"].get("ms2_notes", {}) if self.override_parameters.get("note_options_overrides") else {},
            ms2_notes_opts,
        )
        other_options, other_hotkeys = gno.get_note_options_and_hotkeys(
            self.override_parameters["note_options_overrides"].get("other_notes", {}) if self.override_parameters.get("note_options_overrides") else {},
            other_notes_opts,
        )

        ms1_key_to_label = {v: k for k, v in ms1_hotkeys.items()}
        ms2_key_to_label = {v: k for k, v in ms2_hotkeys.items()}
        other_key_to_label = {v: k for k, v in other_hotkeys.items()}

        self.notes = {
            "ms1_notes": ms1_options,
            "ms1_hotkeys": ms1_hotkeys,
            "ms2_notes": ms2_options,
            "ms2_hotkeys": ms2_hotkeys,
            "other_notes": other_options,
            "other_hotkeys": other_hotkeys,
            "ms1_key_to_label": ms1_key_to_label,
            "ms2_key_to_label": ms2_key_to_label,
            "other_key_to_label": other_key_to_label
        }

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
    experimental_data: Optional[ExperimentalData] = None
    per_file_metrics_df: Optional[pd.DataFrame] = None

    # Paths and config
    config_path: str = None
    paths: Dict[str, str] = field(default_factory=dict)
    config: Optional["Metatlas2Config"] = None

    def setup(
        self,
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
        self.project_name = project_name

        project_db_path = rtg.get_project_db_path(self.project_name)
        self.config = dbi.load_config_from_db(
            project_db_path=project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )
        if self.config is None:
            raise RuntimeError(
                f"No config found in project database at {project_db_path} "
                f"for RTA={rt_alignment_number}, TGA={analysis_number}. "
                "Ensure run_project_setup was called with a config_path."
            )
        logger.info("Config loaded from project database.")

        self.chromatography = self.config.chromatography
        self.paths = rtg.set_up_paths(config=self.config, project_name=self.project_name, rt_alignment_number=self.rt_alignment_number, analysis_number=self.analysis_number)