from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import Any, TypedDict
from enum import Enum
from pathlib import Path
import json
import math
import pandas as pd
import yaml
from tqdm.auto import tqdm
import pyarrow.dataset as ds

import metatlas2.database_interact as dbi
import metatlas2.load_tools as ldt
import metatlas2.pubchem_retrieval as pcr
import metatlas2.lcmsruns_tools as lrt
import metatlas2.logging_config as lcf
import metatlas2.run_targeted_analysis as rtg
import metatlas2.note_options as gno
import metatlas2.file_and_project_format as fpf
from metatlas2.utils import should_disable_tqdm, as_list
logger = lcf.get_logger('workflow_objects')

class AtlasStage(str, Enum):
    """Defined stages of an atlas in the workflow, used to
    keep track of the progress of an atlas through the workflow."""

    RT_ALIGNED = "RT_ALIGNED"
    AUTO_IDED = "AUTO_IDED"
    MANUALLY_CURATED = "MANUALLY_CURATED"

@dataclass
class AlignedSpectrumPair:
    """A query/reference spectrum pair produced by MS2 library matching.
    Stores the aligned fragment arrays and per-fragment color assignments
    """

    query_mzs: list[float]
    query_ints: list[float]
    ref_mzs: list[float]
    ref_ints: list[float]
    fragment_colors: list[str]

    def __post_init__(self) -> None:
        # Ensure all arrays are plain Python lists (not numpy arrays)
        self.query_mzs = as_list(self.query_mzs)
        self.query_ints = as_list(self.query_ints)
        self.ref_mzs = as_list(self.ref_mzs)
        self.ref_ints = as_list(self.ref_ints)
        self.fragment_colors = as_list(self.fragment_colors)
        # Default colors if missing
        if not self.fragment_colors and self.query_mzs:
            self.fragment_colors = ["tomato"] * len(self.query_mzs)

    @classmethod
    def from_hit_dict(cls, d: dict) -> AlignedSpectrumPair:
        qa = d.get("query_aligned", [[], []])
        ra = d.get("ref_aligned", [[], []])
        return cls(
            query_mzs=as_list(qa[0]) if len(qa) > 0 else [],
            query_ints=as_list(qa[1]) if len(qa) > 1 else [],
            ref_mzs=as_list(ra[0]) if len(ra) > 0 else [],
            ref_ints=as_list(ra[1]) if len(ra) > 1 else [],
            fragment_colors=as_list(d.get("fragment_colors", [])),
        )

    def to_dict(self) -> dict:
        return {
            "query_aligned": [self.query_mzs, self.query_ints],
            "ref_aligned":   [self.ref_mzs,   self.ref_ints],
            "fragment_colors": self.fragment_colors,
        }


@dataclass
class MS2Hit:
    """One MS2 library match result for a single scan against a reference spectrum.
    """

    score: float
    num_matches: int
    ref_frags: int
    mz_measured: float
    matched_fragments: list[float]
    aligned: AlignedSpectrumPair
    filename: str = ""
    scan_rt: float = float("nan")

    def __post_init__(self) -> None:
        self.matched_fragments = as_list(self.matched_fragments)

    @classmethod
    def from_dict(cls, d: dict, filename: str = "", scan_rt: float = float("nan")) -> MS2Hit:
        try:
            mz_measured = float(d.get("mz_measured", float("nan")))
        except (TypeError, ValueError):
            mz_measured = float("nan")
        try:
            score = float(d.get("score", float("nan")))
        except (TypeError, ValueError):
            score = float("nan")
        return cls(
            score=score,
            num_matches=int(d.get("num_matches", 0)),
            ref_frags=int(d.get("ref_frags", 0)),
            mz_measured=mz_measured,
            matched_fragments=as_list(d.get("matched_fragments", [])),
            aligned=AlignedSpectrumPair.from_hit_dict(d),
            filename=filename,
            scan_rt=scan_rt,
        )

    @classmethod
    def list_from_scan_row(cls, scan_row: pd.Series) -> list[MS2Hit]:
        raw = scan_row.get("hits")
        if raw is None:
            return []
        # Handle JSON string (as stored in DuckDB VARCHAR column)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return []
        hits_list = as_list(raw)
        if not hits_list:
            return []
        filename = str(scan_row.get("filename", ""))
        try:
            scan_rt = float(scan_row.get("scan_rt", float("nan")))
        except (TypeError, ValueError):
            scan_rt = float("nan")
        return [
            cls.from_dict(h, filename=filename, scan_rt=scan_rt)
            for h in hits_list
            if isinstance(h, dict)
        ]

    @property
    def has_score(self) -> bool:
        try:
            return not math.isnan(self.score)
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "num_matches": self.num_matches,
            "ref_frags": self.ref_frags,
            "mz_measured": self.mz_measured,
            "matched_fragments": self.matched_fragments,
            **self.aligned.to_dict(),
        }

class PathsConfig(TypedDict, total=False):
    """Dictionary type for the paths on disk for a given run."""

    lcmsruns_directory: str
    project_directory: str
    log_path: str
    project_db_path: str
    main_db_path: str
    msms_refs_path: str | None
    pubchem_cache_path: str
    modelseed_table_path: str
    parquet_output_dir: str
    rt_alignment_output_dir: str
    rt_alignment_results_dir: str
    aligned_atlases_store_file: str
    analysis_output_dir: str
    auto_ided_atlases_store_file: str
    curated_atlases_store_file: str
    analysis_results_output_dir: str

@dataclass
class TargetedAnalysis:

    chromatography: str
    polarity: str
    analysis_type: str
    analysis_name: str
    atlas_uid: str
    params: dict[str, Any] = field(default_factory=dict)

@dataclass
class Metatlas2Config:

    paths_config: dict[str, Any] = field(default_factory=dict)
    rt_alignment_config: dict[str, Any] = field(default_factory=dict)
    targeted_analyses: list[TargetedAnalysis] = field(default_factory=list)

    def get_targeted_analysis(
        self,
        chromatography: str,
        polarity: str,
        analysis_type: str,
        analysis_name: str,
    ) -> "TargetedAnalysis":

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
        )

    @property
    def chromatography(self) -> str:
        if self.targeted_analyses:
            return self.targeted_analyses[0].chromatography
        if self.rt_alignment_config:
            return next(iter(self.rt_alignment_config))
        raise ValueError("No chromatography found in config")

    @property
    def owner(self) -> str:
        return (self.paths_config.get('owner') or 'jgi').lower()

    @property
    def msms_refs_path(self) -> str | None:
        return self.paths_config.get('msms_refs_path', None)

    @property
    def msms_refs_db_filter(self) -> str | None:
        return self.paths_config.get('msms_refs_db_filter', None)

    @property
    def gdrive_subfolder(self) -> str | None:
        return self.paths_config.get('gdrive_subfolder', None)

    def to_snapshot(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any]) -> "Metatlas2Config":
        targeted_analyses_raw = snapshot.get("targeted_analyses", [])
        targeted_analyses = []
        for ta in targeted_analyses_raw:
            targeted_analyses.append(TargetedAnalysis(
                chromatography=ta["chromatography"],
                polarity=ta["polarity"],
                analysis_type=ta["analysis_type"],
                analysis_name=ta["analysis_name"],
                atlas_uid=ta["atlas_uid"],
                params=dict(ta.get("params") or {}),
            ))

        return cls(
            paths_config=dict(snapshot.get("paths_config") or {}),
            rt_alignment_config=dict(snapshot.get("rt_alignment_config") or {}),
            targeted_analyses=targeted_analyses,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_snapshot())

@dataclass
class AtlasEntry:

    path: str | None
    name: str | None
    desc: str | None

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
    """Configuration for the ``metatlas2.sh add-atlases`` command.

    Parsed from a YAML file with an ``ATLASES`` top-level key.
    Call :meth:`execute` to create all atlases in the database.
    """

    atlases: dict[str, dict[str, dict[str, list[AtlasEntry]]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewAtlasesConfig":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if 'ATLASES' not in raw:
            raise ValueError(f"Missing required top-level key 'ATLASES' in {path}")

        atlases: dict[str, dict[str, dict[str, list[AtlasEntry]]]] = {}
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
        for chrom, pol_dict in self.atlases.items():
            for pol, type_dict in pol_dict.items():
                for analysis_type, entries in type_dict.items():
                    for entry in entries:
                        yield chrom, pol, analysis_type, entry

    def execute(self) -> None:
        """Create all atlases defined in this config and save them to the main database.

        This is the entry-point for ``metatlas2.sh add-atlases``.  It replaces
        the old ``Atlas.create_from_config()`` classmethod, keeping pipeline
        orchestration on the config object rather than on the domain object.
        """
        paths = rtg.set_up_paths(config=self)
        main_db_path = paths.get("main_db_path", None)

        atlases = []
        summary = []

        atlas_jobs = [
            (chrom, pol, analysis_type, entry)
            for chrom, pol, analysis_type, entry in self.iter_entries()
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
                    analysis_name="MAIN",
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
            logger.info(f"Created new atlas: {info['atlas_name']} (UID: {info['atlas_uid']}) - {info['compound_count']} compounds")


@dataclass
class CompoundParams:
    use_pubchem_cache: bool = True
    update_pubchem_cache: bool = False


@dataclass
class NewCompoundsConfig:
    """Configuration for the ``metatlas2.sh add-compounds`` command.

    Parsed from a YAML file with ``PARAMS`` and ``COMPOUNDS`` top-level keys.
    Call :meth:`execute` to load all compounds into the main database.
    """

    params: CompoundParams
    compounds: dict[str, dict[str, list[str]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewCompoundsConfig":
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

        compounds: dict[str, dict[str, list[str]]] = {}
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
                validated: list[str] = []
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
        for chrom, pol_dict in self.compounds.items():
            for pol, paths in pol_dict.items():
                for file_path in paths:
                    yield chrom, pol, file_path

    def execute(self, overwrite_db: bool = False) -> None:
        """Load all compounds defined in this config into the main database.

        This is the entry-point for ``metatlas2.sh add-compounds``.  It replaces
        the old ``Compound.create_from_config()`` classmethod, keeping pipeline
        orchestration on the config object rather than on the domain object.
        """
        paths = rtg.set_up_paths(config=self)
        main_db_path = paths.get("main_db_path", None)
        pubchem_cache_path = paths.get("pubchem_cache_path", None)

        compounds: list[Compound] = []
        dbi.create_metatlas_database(main_db_path, overwrite=overwrite_db)

        for chrom, pol, file_path in self.iter_paths():
            logger.info(f"Processing compound file: {file_path}")
            compounds_df = ldt.load_compound_input(file_path)
            compounds_df = pcr.retrieve_pubchem_info(
                compounds=compounds_df,
                pubchem_cache_path=pubchem_cache_path,
                use_pubchem_cache=self.params.use_pubchem_cache,
                update_pubchem_cache=self.params.update_pubchem_cache,
            )
            for _, row in tqdm(compounds_df.iterrows(), total=len(compounds_df), desc="Processing compounds", disable=should_disable_tqdm()):
                try:
                    compound = Compound.from_atlas_row(row)
                    compounds.append(compound)
                except Exception as e:
                    logger.warning(f"Failed to create Compound for row {row.get('compound_name', 'Unknown')}: {e}")

        dbi.batch_save_compounds(main_db_path, compounds)


@dataclass
class Compound:
    
    compound_uid: str
    compound_name: str
    inchi_key: str
    inchi: str = ""
    smiles: str = ""
    formula: str = ""
    classes: str = ""
    pathways: str = ""
    tags: str = ""
    mono_isotopic_molecular_weight: float = 0.0
    iupac_name: str = ""
    pubchem_cid: str = ""
    cas_number: str = ""
    synonyms: str = ""
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
    
    def to_dict(self) -> dict[str, Any]:
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


@dataclass
class CompoundMZRT:
    
    mz_rt_uid: str
    compound_uid: str
    prev_mz_rt_uid: str | None = None
    compound_name: str = ""
    inchi_key: str = ""
    adduct: str = ""
    rt_space: str = ""
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    mz: float = 0.0
    mz_tolerance: float = 5.0
    chromatography: str = ""
    polarity: str = ""
    confidence: str = ""
    source: str = ""
    identification_notes: str = ""
    created_by: str = ""
    created_date: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundMZRT':
        return cls(
            mz_rt_uid=row.get('mz_rt_uid', ''),
            compound_uid=row.get('compound_uid', ''),
            prev_mz_rt_uid=row.get('prev_mz_rt_uid', None),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            'mz_rt_uid': self.mz_rt_uid,
            'compound_uid': self.compound_uid,
            'prev_mz_rt_uid': self.prev_mz_rt_uid,
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
    
    atlas_uid: str
    atlas_name: str
    atlas_description: str
    chromatography: str
    polarity: str
    analysis_type: str
    atlas_type: str = "REFERENCE"
    analysis_name: str = "MAIN"
    compound_mzrts: dict[str, CompoundMZRT] = field(default_factory=dict)
    rt_alignment_number: int | None = None
    analysis_number: int | None = None
    created_by: str = ""
    created_date: str = ""
    source: str = ""
    source_atlas_uid: str | None = None
    
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:

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
        """Reconstruct an :class:`Atlas` from a flat compound-per-row DataFrame.

        This is the inverse of :meth:`to_dataframe`.  All atlas-level metadata
        fields — including ``rt_alignment_number``, ``analysis_number``,
        ``source_atlas_uid``, ``created_by``, ``created_date``, and ``source``
        — are read from the first row so that a round-trip through the database
        preserves every field.
        """
        meta = atlas_df.iloc[0] if not atlas_df.empty else {}
        atlas_uid         = meta.get('atlas_uid', '')
        atlas_name        = meta.get('atlas_name', '')
        atlas_description = meta.get('atlas_description', '')
        chromatography    = meta.get('chromatography', '')
        polarity          = meta.get('polarity', '')
        analysis_type     = meta.get('analysis_type', '')
        atlas_type        = meta.get('atlas_type', 'REFERENCE')
        analysis_name     = meta.get('analysis_name', 'MAIN')
        created_by        = meta.get('created_by', '')
        created_date      = meta.get('created_date', '')
        source            = meta.get('source', '')
        source_atlas_uid  = meta.get('source_atlas_uid', None)
        # Preserve workflow tracking numbers when loading from a project DB atlas
        rt_alignment_number = meta.get('rt_alignment_number', None)
        analysis_number     = meta.get('analysis_number', None)

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
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
        )

    @classmethod
    def from_database(cls, database_path: str, atlas_uid: str, main_db_path: str = None):

        logger.debug(f"Loading atlas with UID {atlas_uid} from database...")
        return cls.from_dataframe(dbi.get_atlas_compounds_table(database_path, atlas_uid, main_db_path))

    def to_metadata_dict(self) -> dict[str, Any]:
        """Return atlas-level metadata fields only (no compound data).

        Used by :func:`metatlas2.load_tools.save_atlas_metadata_to_csv` to
        write one CSV row per atlas without expanding compound_mzrts.
        """
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

    def to_dict(self) -> dict[str, Any]:
        """Return the full atlas as a plain dict — the inverse of :meth:`from_dataframe`.

        The returned dict contains all atlas-level metadata fields plus a
        ``compound_mzrts`` key whose value is a ``{mz_rt_uid: dict}`` mapping
        produced by calling :meth:`CompoundMZRT.to_dict` on each entry.  Round-
        tripping through ``Atlas.from_dataframe(pd.DataFrame(atlas.to_dict()['compound_mzrts'].values()))``
        reconstructs the original object.
        """
        return {
            'atlas_uid': self.atlas_uid,
            'atlas_name': self.atlas_name,
            'atlas_description': self.atlas_description,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'analysis_type': self.analysis_type,
            'atlas_type': self.atlas_type,
            'analysis_name': self.analysis_name,
            'created_by': self.created_by,
            'created_date': self.created_date,
            'source': self.source,
            'source_atlas_uid': self.source_atlas_uid,
            'rt_alignment_number': self.rt_alignment_number,
            'analysis_number': self.analysis_number,
            'compound_mzrts': {
                uid: cmzrt.to_dict()
                for uid, cmzrt in self.compound_mzrts.items()
            },
        }

@dataclass
class Project:

    project_name: str = field(default="")
    paths: PathsConfig = field(default_factory=dict)
    lcmsruns: list[LCMSRun] = field(default_factory=list)

    def setup(self, project_name: str, config: Metatlas2Config, paths: PathsConfig, overwrite_existing: bool = False, rt_alignment_number: int | None = None, analysis_number: int | None = None):

        self.project_name = fpf.parse_project_name(project_name)
        self.paths = paths

        logger.info(f"Creating project database at {self.paths['project_db_path']}...")
        exists = dbi.create_project_database(
            project_db_path=self.paths['project_db_path'],
            rt_align_path=self.paths['rt_alignment_output_dir'],
            log_file_path=self.paths['log_path'],
            overwrite=overwrite_existing
        )

        logger.info(f"Saving config snapshot to database for RTA{rt_alignment_number}/TGA{analysis_number}...")
        dbi.save_config_to_db(
            project_db_path=self.paths['project_db_path'],
            config=config,
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
            paths=self.paths,
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

    rt_alignment_number: int | None = field(default=None)
    analysis_number: int | None = field(default=None)
    project_name: str | None = field(default=None)
    project_db_path: str | None = field(default=None)
    chromatography: str | None = field(default=None)
    owner: str | None = field(default=None)
    run_alignment: bool = field(default=True)
    align_atlas_uid: str | None = field(default=None)
    align_atlas_obj: Atlas | None = field(default=None)
    rt_alignment_params: dict[str, Any] = field(default_factory=dict)
    aligner_lcmsruns: list[LCMSRun] = field(default_factory=list)
    modeling_data: pd.DataFrame | None = field(default=None)
    unaligned_atlas_obj: Atlas | None = field(default=None)
    aligned_atlas_obj: Atlas | None = field(default=None)
    rt_shift_stats: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=dict)
    config: Metatlas2Config | None = field(default=None)

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int):
        """Load config and paths from the project database and configure this object."""
        logger.info(f"Setting up RTAlign object with RT alignment number {rt_alignment_number}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.project_name = project_name
        self.project_db_path = rtg.get_project_db_path(self.project_name)

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

        self.owner = self.config.owner
        self.chromatography = next(iter(self.config.rt_alignment_config.keys()))
        self.align_atlas_uid = self.config.rt_alignment_config[self.chromatography].get('ATLAS', {}).get('uid', None)
        self.rt_alignment_params = self.config.rt_alignment_config[self.chromatography].get('PARAMS', {})

        if self.rt_alignment_params.get('do_alignment', True) is False:
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
            self.aligned_atlas_obj = Atlas.from_database(
                database_path=self.paths['main_db_path'],
                atlas_uid=ta.atlas_uid
            )
            self.aligned_atlas_obj.analysis_name = ta.analysis_name
            self.aligned_atlas_obj.rt_alignment_number = self.rt_alignment_number
            self.aligned_atlas_obj.analysis_number = None

            dbi.save_atlas_to_db_and_disk(
                obj=self,
                atlas_to_update=self.aligned_atlas_obj,
                stage=AtlasStage.RT_ALIGNED,
            )

    def _check_existing_rt_aligned_atlases(self):
        logger.info(f"Checking if RT-aligned atlases already exist for RTA{self.rt_alignment_number}...")
        aligned_count = dbi.count_workflow_runs(
            project_db_path=self.paths['project_db_path'],
            rt_alignment_number=self.rt_alignment_number,
            stage=AtlasStage.RT_ALIGNED,
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
    ms1_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    ms2_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    curation_df: pd.DataFrame = field(default_factory=pd.DataFrame)

@dataclass
class AutoIdentification:

    project_name: str | None = field(default=None)
    project_db_path: str | None = field(default=None)
    rt_alignment_number: int | None = field(default=None)
    analysis_number: int | None = field(default=None)
    owner: str | None = field(default=None)
    chromatography: str | None = field(default=None)
    polarity: str | None = field(default=None)
    msms_refs_db_filter: str | None = field(default=None)
    analysis_subset: list[str] | None = field(default=None)
    image_tag: str = field(default="latest")
    ta: TargetedAnalysis | None = field(default=None)
    aligned_atlas_obj: Atlas | None = field(default=None)
    auto_ided_atlas_obj: Atlas | None = field(default=None)
    autoid_lcmsruns: list[LCMSRun] = field(default_factory=list)
    experimental_data: ExperimentalData | None = field(default=None)
    paths: PathsConfig = field(default_factory=dict)
    config: Metatlas2Config | None = field(default=None)

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int, analysis_subset: list[str] | None = None, image_tag: str = "latest"):

        logger.info(f"Setting up AutoIdentification object for RT alignment number {rt_alignment_number}, analysis number {analysis_number} for project {project_name}...")
        self.rt_alignment_number = rt_alignment_number
        self.analysis_number = analysis_number
        self.project_name = project_name
        self.project_db_path = rtg.get_project_db_path(self.project_name)
        self.analysis_subset = analysis_subset
        self.image_tag = image_tag

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

        self.owner = self.config.owner
        self.chromatography = self.config.chromatography
        self.msms_refs_db_filter = self.config.msms_refs_db_filter

        self._check_existing_auto_ided_atlases()

    def _check_existing_auto_ided_atlases(self):
        logger.debug(f"Checking if AutoID-aligned atlases already exist for RTA{self.rt_alignment_number} and TGA{self.analysis_number}...")
        autoided_count = dbi.count_workflow_runs(
            project_db_path=self.paths['project_db_path'],
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
            stage=AtlasStage.AUTO_IDED,
        )
        if autoided_count > 0:
            raise ValueError(
                f"AutoID-aligned atlases already exist for RTA{self.rt_alignment_number} and TGA{self.analysis_number} "
                f"To run a new AutoID analysis, increment analysis_number in the command line call. "
            )

class CurationStageBase(ABC):
    """Shared logic for AnalysisGUI and AnalysisSummary.
    """

    project_name: str | None
    project_db_path: str | None
    rt_alignment_number: int | None
    analysis_number: int | None
    chromatography: str | None
    polarity: str | None
    analysis_type: str | None
    analysis_name: str | None
    owner: str | None
    ta: TargetedAnalysis | None
    notes: dict[str, Any]
    override_parameters: dict[str, Any]
    paths: PathsConfig
    config: Metatlas2Config | None

    def setup(
        self,
        run_parameters: dict[str, Any],
        override_parameters: dict[str, Any] | None = None,
    ) -> None:
        """Populate all common attributes from *run_parameters*.
        """
        self.rt_alignment_number = run_parameters['rt_alignment_number']
        self.analysis_number = run_parameters['analysis_number']
        self.chromatography = run_parameters['chromatography']
        self.polarity = run_parameters['polarity']
        self.analysis_type = run_parameters['analysis_type']
        self.analysis_name = run_parameters['analysis_name']
        self.project_name = run_parameters['project_name']
        self.project_db_path = rtg.get_project_db_path(self.project_name)

        self.config = dbi.load_config_from_db(
            project_db_path=self.project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )

        self.owner = self.config.owner
        self.paths = rtg.set_up_paths(
            config=self.config,
            project_name=self.project_name,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )

        if override_parameters is not None:
            self.override_parameters = override_parameters
            dbi.validate_override_parameters(self.override_parameters)

        self._populate_note_options()

        self.ta = self.config.get_targeted_analysis(
            self.chromatography, self.polarity, self.analysis_type, self.analysis_name
        )

    def _populate_note_options(self) -> None:
        """Resolve note options and hotkey mappings into ``self.notes``."""
        ms2_notes_opts, ms1_notes_opts, other_notes_opts = gno.get_notes_opts(owner=self.owner)
        note_overrides = self.override_parameters.get("note_options_overrides") or {}

        ms1_options, ms1_hotkeys = gno.get_note_options_and_hotkeys(
            note_overrides.get("ms1_notes", {}), ms1_notes_opts
        )
        ms2_options, ms2_hotkeys = gno.get_note_options_and_hotkeys(
            note_overrides.get("ms2_notes", {}), ms2_notes_opts
        )
        other_options, other_hotkeys = gno.get_note_options_and_hotkeys(
            note_overrides.get("other_notes", {}), other_notes_opts
        )

        self.notes = {
            "ms1_notes":        ms1_options,
            "ms1_hotkeys":      ms1_hotkeys,
            "ms2_notes":        ms2_options,
            "ms2_hotkeys":      ms2_hotkeys,
            "other_notes":      other_options,
            "other_hotkeys":    other_hotkeys,
            "ms1_key_to_label": {v: k for k, v in ms1_hotkeys.items()},
            "ms2_key_to_label": {v: k for k, v in ms2_hotkeys.items()},
            "other_key_to_label": {v: k for k, v in other_hotkeys.items()},
        }


@dataclass
class AnalysisGUI(CurationStageBase):
    """Workflow object for the interactive manual-curation GUI stage."""

    project_name: str | None = field(default=None)
    project_db_path: str | None = field(default=None)
    rt_alignment_number: int | None = field(default=None)
    analysis_number: int | None = field(default=None)
    chromatography: str | None = field(default=None)
    polarity: str | None = field(default=None)
    analysis_type: str | None = field(default=None)
    analysis_name: str | None = field(default=None)
    owner: str = field(default="jgi")
    ta: TargetedAnalysis | None = field(default=None)
    auto_ided_atlas_obj: Atlas | None = field(default=None)
    experimental_data: ExperimentalData | None = field(default=None)
    notes: dict[str, Any] = field(default_factory=dict)
    override_parameters: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=dict)
    config: Metatlas2Config | None = field(default=None)

    def setup(
        self,
        run_parameters: dict[str, Any],
        override_parameters: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            f"Setting up AnalysisGUI object for "
            f"RTA{run_parameters['rt_alignment_number']} / "
            f"TGA{run_parameters['analysis_number']} / "
            f"project {run_parameters['project_name']}..."
        )
        super().setup(run_parameters, override_parameters)


@dataclass
class AnalysisSummary(CurationStageBase):
    """Workflow object for the post-curation analysis summary stage."""

    project_name: str | None = field(default=None)
    project_db_path: str | None = field(default=None)
    rt_alignment_number: int | None = field(default=None)
    analysis_number: int | None = field(default=None)
    chromatography: str | None = field(default=None)
    polarity: str | None = field(default=None)
    analysis_type: str | None = field(default=None)
    analysis_name: str | None = field(default=None)
    owner: str | None = field(default=None)
    ta: TargetedAnalysis | None = field(default=None)
    auto_ided_atlas_obj: Atlas | None = field(default=None)
    manually_curated_atlas_obj: Atlas | None = field(default=None)
    experimental_data: ExperimentalData | None = field(default=None)
    per_file_metrics_df: pd.DataFrame | None = field(default=None)
    best_ms1_metrics_df: pd.DataFrame | None = field(default=None)
    notes: dict[str, Any] = field(default_factory=dict)
    override_parameters: dict[str, Any] = field(default_factory=dict)
    paths: PathsConfig = field(default_factory=dict)
    config: Metatlas2Config | None = field(default=None)

    def setup(
        self,
        run_parameters: dict[str, Any],
        override_parameters: dict[str, Any] | None = None,
    ) -> None:
        logger.info(
            f"Setting up AnalysisSummary object for "
            f"RTA{run_parameters['rt_alignment_number']} / "
            f"TGA{run_parameters['analysis_number']} / "
            f"project {run_parameters['project_name']}..."
        )
        super().setup(run_parameters, override_parameters)

        self.paths['analysis_results_output_dir'] = str(
            Path(self.paths["analysis_output_dir"])
            / f"{self.ta.chromatography}-{self.ta.polarity}-{self.ta.analysis_type}-{self.ta.analysis_name}"
        )
        os.makedirs(self.paths['analysis_results_output_dir'], exist_ok=True)

class ParquetQueryInterpreter:
    def __init__(self, parquet_path: Path):
        parquet_path = Path(parquet_path)
        if parquet_path.is_dir():
            self.dataset = ds.dataset(parquet_path, format="parquet", partitioning=None)
        else:
            self.dataset = ds.dataset([parquet_path], format="parquet", partitioning=None)

    def _translate_to_expression(self, param_name: str, value: Any):
        """
        Translates a YAML key into a PyArrow Dataset Expression.
        """
        # Determine Operator and Column Name
        if param_name.endswith("_min"):
            col, op = param_name[:-4], ">="
        elif param_name.endswith("_max"):
            col, op = param_name[:-4], "<="
        elif param_name.endswith("_abs_gt"):
            col, op = param_name[:-7], "abs_gt"
        elif param_name.endswith("_abs_lt"):
            col, op = param_name[:-7], "abs_lt"
        elif param_name.endswith("_in"):
            col, op = param_name[:-3], "in"
        else:
            col, op = param_name, "=="

        field = ds.field(col)

        # PyArrow Expression
        if op == "==":
            return field == value
        elif op == ">=":
            return field >= value
        elif op == "<=":
            return field <= value
        elif op == "in":
            val_list = value if isinstance(value, list) else [value]
            return field.isin(val_list)
        elif op == "abs_gt":
            return (field > value) | (field < -value)
        elif op == "abs_lt":
            return (field < value) & (field > -value)
        
        return None

    def execute_from_params(self, params_path: Path) -> pd.DataFrame:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f) or {}

        # Build a list of PyArrow expressions
        expressions = []
        for param_name, value in params.items():
            if value is None:
                continue
            
            try:
                expr = self._translate_to_expression(param_name, value)
                if expr is not None:
                    expressions.append(expr)
            except Exception as e:
                print(f"Skipping parameter {param_name} due to error: {e}")

        # Combine all expressions with "AND" logic
        final_filter = None
        for expr in expressions:
            if final_filter is None:
                final_filter = expr
            else:
                final_filter = final_filter & expr

        if final_filter is not None:
            table = self.dataset.to_table(filter=final_filter)
        else:
            table = self.dataset.to_table()

        return table.to_pandas()