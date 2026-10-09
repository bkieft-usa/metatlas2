from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, asdict
from typing import Any, NamedTuple, TypedDict
from enum import Enum
from pathlib import Path
import json
import numpy as np
import csv
import ast
import math
import pandas as pd
import os
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
import metatlas2.parquet_interact as pqi
from metatlas2.utils import should_disable_tqdm, as_list, get_provenance
logger = lcf.get_logger('workflow_objects')

class AtlasStage(str, Enum):
    """Defined stages of an atlas in the workflow, used to
    keep track of the progress of an atlas through the workflow."""
    INPUT = "INPUT"
    RT_ALIGNED = "RT_ALIGNED"
    AUTO_IDED = "AUTO_IDED"
    MANUALLY_CURATED = "MANUALLY_CURATED"

class AtlasCategory(str, Enum):
    """Categorical (semantic) role of an atlas in the workflow.

    QC   — retention-time alignment compounds (quality control).
    ISTD — internal standard compounds.
    REF  — reference / target compound sets to search for in raw data.
    """
    QC   = "QC"
    ISTD = "ISTD"
    REF  = "REF"

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
        self.query_mzs = as_list(self.query_mzs)
        self.query_ints = as_list(self.query_ints)
        self.ref_mzs = as_list(self.ref_mzs)
        self.ref_ints = as_list(self.ref_ints)
        self.fragment_colors = as_list(self.fragment_colors)
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
    """One MS2 library match result for a single scan against a reference spectrum."""
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
    """Dictionary type for the paths on disk for a given run.

    Note: MSMS reference spectra are now stored in the ``msms_refs``
    table of the main DuckDB (``main_db_path``).  The ``msms_refs_path`` field has been
    removed.  To override with a custom ``.jsonl`` file, set ``PATHS.msms_refs_path``
    in the analysis YAML config — it is read at runtime from
    ``auto_id_obj.config.msms_refs_path`` in :func:`~metatlas2.ms2_hit_detection.find_ms2_hits`.
    """

    lcmsruns_directory: str
    project_directory: str
    log_path: str
    project_db_path: str
    main_db_path: str
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
    atlas_category: AtlasCategory
    atlas_name: str
    atlas_uid: str
    params: dict[str, Any] = field(default_factory=dict)
    chromatography_label: str = field(default="")
    polarity_label: str = field(default="")
    atlas_category_label: str = field(default="")

    @property
    def label(self) -> str:
        chrom = self.chromatography
        pol   = self.polarity
        aname = self.atlas_name
        return f"{chrom}-{pol}-{aname}"

@dataclass
class Metatlas2Config:

    general_config: dict[str, Any] = field(default_factory=dict)
    rt_alignment_config: dict[str, Any] = field(default_factory=dict)
    targeted_analyses: list[TargetedAnalysis] = field(default_factory=list)
    gui_config: dict[str, Any] = field(default_factory=dict)

    def get_targeted_analysis(
        self,
        chromatography: str,
        polarity: str,
        atlas_category: "AtlasCategory | str",
        atlas_name: str,
    ) -> "TargetedAnalysis":
        chrom_lc = fpf.normalize_chromatography(chromatography)
        pol_lc   = polarity.lower()
        cat_lc   = (atlas_category.value if isinstance(atlas_category, AtlasCategory) else str(atlas_category)).lower()
        aname_lc = atlas_name.lower()
        for ta in self.targeted_analyses:
            ta_cat = ta.atlas_category.value if isinstance(ta.atlas_category, AtlasCategory) else str(ta.atlas_category)
            if (
                ta.chromatography.lower() == chrom_lc
                and ta.polarity.lower() == pol_lc
                and ta_cat.lower() == cat_lc
                and ta.atlas_name.lower() == aname_lc
            ):
                return ta
        raise KeyError(
            f"No targeted analysis found for {chromatography}/{polarity}/{atlas_category}/{atlas_name}. "
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
        return (self.general_config.get('owner') or 'jgi').lower()

    @property
    def msms_refs_path(self) -> str | None:
        return self.general_config.get('msms_refs_path', None)

    @property
    def msms_refs_db_filter(self) -> str | None:
        return self.general_config.get('msms_refs_db_filter', None)

    @property
    def gdrive_subfolder(self) -> str | None:
        return self.general_config.get('gdrive_subfolder', None)

    @property
    def max_workers(self) -> int | None:
        val = self.general_config.get('max_workers', None)
        return int(val) if val is not None else None

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
                atlas_category=AtlasCategory(ta["atlas_category"]),
                atlas_name=ta["atlas_name"],
                atlas_uid=ta["atlas_uid"],
                params=dict(ta.get("params") or {}),
                chromatography_label=ta.get("chromatography_label", ""),
                polarity_label=ta.get("polarity_label", ""),
                atlas_category_label=ta.get("atlas_category_label", ""),
            ))

        return cls(
            general_config=dict(snapshot.get("general_config") or {}),
            rt_alignment_config=dict(snapshot.get("rt_alignment_config") or {}),
            targeted_analyses=targeted_analyses,
            gui_config=dict(snapshot.get("gui_config") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_snapshot())

@dataclass
class AtlasEntry:

    path: str | None
    desc: str | None
    atlas_category: "AtlasCategory"
    atlas_name: str  # short free-text key from YAML (e.g. "QCv7", "EMA")

    def __post_init__(self):
        self.path = str(self.path) if self.path else None
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
    The YAML structure is four levels deep:
      ATLASES -> chromatography -> polarity -> atlas_category -> atlas_name -> {path, desc}

    ``atlas_category`` must be one of the :class:`AtlasCategory` enum values
    (``QC``, ``ISTD``, or ``REF``).  ``atlas_name`` is a free-text short
    identifier (e.g. ``QCv7``, ``ISTDsEtcV7``, ``EMA``).

    Call :meth:`execute` to create all atlases in the database.
    """

    atlases: dict[str, dict[str, dict[str, dict[str, AtlasEntry]]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewAtlasesConfig":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if 'ATLASES' not in raw:
            raise ValueError(f"Missing required top-level key 'ATLASES' in {path}")

        valid_categories = {c.value for c in AtlasCategory}
        atlases: dict[str, dict[str, dict[str, dict[str, AtlasEntry]]]] = {}
        for chrom, chrom_cfg in raw['ATLASES'].items():
            if not isinstance(chrom_cfg, dict):
                raise ValueError(f"Expected a dict under ATLASES.{chrom}")
            atlases[chrom] = {}
            for pol, pol_cfg in chrom_cfg.items():
                if not isinstance(pol_cfg, dict):
                    raise ValueError(f"Expected a dict under ATLASES.{chrom}.{pol}")
                atlases[chrom][pol] = {}
                for cat_key, cat_cfg in pol_cfg.items():
                    cat_upper = str(cat_key).upper()
                    if cat_upper not in valid_categories:
                        raise ValueError(
                            f"Invalid atlas_category '{cat_key}' under ATLASES.{chrom}.{pol}. "
                            f"Must be one of: {sorted(valid_categories)}"
                        )
                    atlas_category = AtlasCategory(cat_upper)
                    if not isinstance(cat_cfg, dict):
                        raise ValueError(
                            f"Expected a dict of atlas_name -> {{path, desc}} under "
                            f"ATLASES.{chrom}.{pol}.{cat_key}"
                        )
                    atlases[chrom][pol][cat_upper] = {}
                    for atlas_name, entry_raw in cat_cfg.items():
                        location = f"ATLASES.{chrom}.{pol}.{cat_key}.{atlas_name}"
                        if not isinstance(entry_raw, dict):
                            raise ValueError(f"Expected a dict under {location}")
                        if 'path' not in entry_raw:
                            raise ValueError(f"Missing required field 'path' in {location}")
                        atlases[chrom][pol][cat_upper][str(atlas_name)] = AtlasEntry(
                            path=entry_raw.get('path'),
                            desc=entry_raw.get('desc', ''),
                            atlas_category=atlas_category,
                            atlas_name=str(atlas_name),
                        )

        logger.info(f"Loaded atlas configuration from {path}")
        return cls(atlases=atlases)

    def iter_entries(self):
        for chrom, pol_dict in self.atlases.items():
            for pol, cat_dict in pol_dict.items():
                for cat_key, name_dict in cat_dict.items():
                    for atlas_name, entry in name_dict.items():
                        yield chrom, pol, cat_key, atlas_name, entry

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
            (chrom, pol, cat_key, atlas_name, entry)
            for chrom, pol, cat_key, atlas_name, entry in self.iter_entries()
        ]

        for chrom, pol, cat_key, atlas_name, entry in tqdm(atlas_jobs, desc="Creating atlases from config"):
            if entry.is_empty:
                logger.debug(f"Skipping atlas with no path for {cat_key}/{atlas_name}/{chrom}/{pol}")
                continue
            try:
                atlas_compounds_df = ldt.load_atlas_input(entry.path)
                atlas_obj = dbi.create_new_atlas_from_dataframe(
                    atlas_df=atlas_compounds_df,
                    atlas_description=entry.desc,
                    atlas_category=entry.atlas_category,
                    atlas_name=atlas_name,
                    chromatography=chrom,
                    polarity=pol,
                    atlas_file_path=entry.path,
                    main_db_path=main_db_path
                )
                dbi.save_atlas_to_database(atlas_obj, main_db_path)
                logger.info(f"Successfully created atlas: {atlas_obj.atlas_uid}")
                atlases.append(atlas_obj)
                summary.append({
                    'atlas_uid': atlas_obj.atlas_uid,
                    'atlas_name': atlas_obj.atlas_name,
                    'compound_count': len(atlas_obj.compound_mzrts)
                })
            except Exception as e:
                logger.error(f"Failed to create Atlas for {cat_key}/{atlas_name}/{chrom}/{pol}: {e}")

        logger.info("Summary of new atlases.")
        logger.info("**Make sure to add these to your analysis config to use as project input atlases:**")
        for info in summary:
            logger.info(f"Created new atlas: {info['atlas_name']} (UID: {info['atlas_uid']}) - {info['compound_count']} compounds")

@dataclass
class NewCompoundsConfig:
    """Configuration for the ``metatlas2.sh add-compounds`` command.

    Parsed from a YAML file with a ``COMPOUNDS`` top-level key (and an
    optional ``PARAMS`` key whose contents are currently ignored).
    Call :meth:`execute` to load all compounds into the main database.
    """

    compounds: dict[str, dict[str, list[str]]]

    @classmethod
    def from_yaml(cls, path: str) -> "NewCompoundsConfig":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if 'COMPOUNDS' not in raw:
            raise ValueError(f"Missing required top-level key 'COMPOUNDS' in {path}")

        compounds: dict[str, dict[str, list[str]]] = {}
        # PARAMS key is accepted but ignored — all caching is always enabled
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

        logger.info(f"Loaded compound configuration from {path}")
        return cls(compounds=compounds)

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
        import metatlas2.modelseed_retrieval as msr
        import metatlas2.chebi_retrieval as chr

        paths = rtg.set_up_paths(config=self)
        main_db_path = paths.get("main_db_path", None)
        pubchem_cache_path = paths.get("pubchem_cache_path", None)
        modelseed_cache_path = paths.get("modelseed_table_path", None)
        chebi_cache_path = paths.get("chebi_cache_path", None)

        compounds: list[Compound] = []
        dbi.create_metatlas_database(main_db_path, overwrite=overwrite_db)

        for chrom, pol, file_path in self.iter_paths():
            logger.info(f"Processing compound file: {file_path}")
            compounds_df = ldt.load_compound_input(file_path)
            compounds_df = pcr.retrieve_pubchem_info(
                compounds=compounds_df,
                pubchem_cache_path=pubchem_cache_path,
            )
            compounds_df = msr.retrieve_modelseed_info(
                compounds=compounds_df,
                modelseed_cache_path=modelseed_cache_path,
            )
            compounds_df = chr.retrieve_metacyc_info(
                compounds=compounds_df,
                chebi_cache_path=chebi_cache_path,
            )
            for _, row in tqdm(compounds_df.iterrows(), total=len(compounds_df), desc="Processing compounds", disable=should_disable_tqdm()):
                try:
                    compound = Compound.from_atlas_row(row)
                    compounds.append(compound)
                except Exception as e:
                    logger.warning(f"Failed to create Compound for row {row.get('compound_name', 'Unknown')}: {e}")

        dbi.batch_save_compounds(main_db_path, compounds)


@dataclass
class NewMsmsRefsConfig:
    """Configuration for the ``metatlas2.sh add-msms-refs`` command.

    Parsed from a YAML file with an ``MSMS_REFS`` top-level key that lists
    one or more source ``.tab`` files to import.

    Expected TSV/tab format (metatlas ``.tab`` export)::

        database  id  name  spectrum  decimal  precursor_mz  polarity  adduct
        fragmentation_method  collision_energy  instrument  instrument_type
        formula  exact_mass  inchi_key  inchi  smiles

    The ``spectrum`` column must contain a Python list-of-lists literal with
    mz values in the first list and intensities in the second::

        [[59.01, 71.01, 87.01], [1000.0, 2000.0, 3000.0]]

    Column mapping: ``id`` → ``ref_id``,
    ``exact_mass`` → ``mono_isotopic_molecular_weight``.

    Missing / NULL values should be represented as an empty string or ``\\N``.
    """

    entries: list[dict]

    @classmethod
    def from_yaml(cls, path: str) -> "NewMsmsRefsConfig":
        with open(path, 'r') as f:
            raw = yaml.safe_load(f)

        if 'MSMS_REFS' not in raw:
            raise ValueError(f"Missing required top-level key 'MSMS_REFS' in {path}")

        entries_raw = raw['MSMS_REFS']
        if not isinstance(entries_raw, list):
            raise ValueError(f"'MSMS_REFS' must be a list of entries in {path}")

        entries = []
        for i, entry in enumerate(entries_raw):
            if not isinstance(entry, dict):
                raise ValueError(f"MSMS_REFS entry {i} must be a dict with a 'path' key")
            if 'path' not in entry:
                raise ValueError(f"MSMS_REFS entry {i} is missing required 'path' key")
            entries.append({
                'path': str(entry['path']),
                'database_filter': entry.get('database_filter', None),
            })

        logger.info(f"Loaded MSMS refs configuration from {path}: {len(entries)} source file(s)")
        return cls(entries=entries)

    @staticmethod
    def _parse_float_or_none(value: str) -> float | None:
        """Return a float from *value*, or ``None`` for empty / NULL strings."""
        v = value.strip()
        if not v or v in ('\\N', 'NULL', 'None', 'nan', 'NaN'):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    @staticmethod
    def _parse_spectrum_column(value: str) -> tuple[list[float], list[float]]:
        """Parse a ``spectrum`` column containing a Python list-of-lists literal.

        The expected format is ``[[mz1, mz2, ...], [int1, int2, ...]]`` as
        produced by the metatlas ``.tab`` export format.  Returns a
        ``(mz_list, intensity_list)`` tuple, or ``([], [])`` on failure.
        """
        v = value.strip()
        if not v or v in ('\\N', 'NULL', 'None'):
            return [], []
        try:
            parsed = ast.literal_eval(v)
            if (
                isinstance(parsed, list)
                and len(parsed) == 2
                and isinstance(parsed[0], list)
                and isinstance(parsed[1], list)
            ):
                return [float(x) for x in parsed[0]], [float(x) for x in parsed[1]]
        except Exception:
            pass
        return [], []

    @staticmethod
    def _str_or_none(value: str) -> str | None:
        """Return *value* stripped, or ``None`` for empty / NULL strings."""
        v = value.strip()
        if not v or v in ('\\N', 'NULL', 'None'):
            return None
        return v

    def execute(self) -> None:
        """Read each source ``.tab`` file and bulk-insert spectra into the main DB."""

        paths = rtg.set_up_paths(config=self)
        main_db_path = paths.get("main_db_path")

        total_inserted = 0
        total_skipped = 0
        batch_size = 10000  # Control memory usage

        for entry in self.entries:
            source_path = Path(entry['path'])
            db_filter = entry.get('database_filter')

            if not source_path.exists():
                logger.error(f"MSMS refs source file not found: {source_path}. Skipping.")
                continue

            logger.info(
                f"Importing MSMS refs from {source_path}"
                + (f" (database_filter='{db_filter}')" if db_filter else "")
                + "..."
            )

            # 1. Pre-calculate total lines for an accurate tqdm progress bar
            with source_path.open('r') as f:
                total_rows = sum(1 for _ in f) - 1

            records: list[dict] = []
            n_skipped = 0
            file_inserted = 0

            with source_path.open('r', newline='') as fh:
                reader = csv.DictReader(fh, delimiter='\t')
                
                # 2. Wrap the iterator in tqdm for automatic progress tracking
                for row_num, row in tqdm(
                    enumerate(reader, start=2), 
                    total=total_rows, 
                    desc=f"Importing {source_path.name}"
                ):
                    try:
                        # Optional database filter
                        if db_filter and row.get('database') != db_filter:
                            continue

                        mz_list, int_list = self._parse_spectrum_column(
                            row.get('spectrum', '')
                        )

                        if not mz_list or not int_list or len(mz_list) != len(int_list):
                            n_skipped += 1
                            continue

                        mz = np.array(mz_list, dtype=np.float32)
                        intensities = np.array(int_list, dtype=np.float32)

                        # Apply the same precursor_mz + 2.5 truncation used at query time
                        precursor_mz = self._parse_float_or_none(row.get('precursor_mz', ''))
                        if precursor_mz is not None and not np.isnan(precursor_mz):
                            mask = mz < precursor_mz + 2.5
                            mz = mz[mask]
                            intensities = intensities[mask]

                        if len(mz) == 0:
                            n_skipped += 1
                            continue

                        records.append({
                            'database': self._str_or_none(row.get('database', '')),
                            'ref_id': self._str_or_none(row.get('id', '')),
                            'name': self._str_or_none(row.get('name', '')),
                            'inchi_key': row.get('inchi_key', '').strip(),
                            'precursor_mz': precursor_mz,
                            'polarity': self._str_or_none(row.get('polarity', '')),
                            'adduct': self._str_or_none(row.get('adduct', '')),
                            'fragmentation_method': self._str_or_none(row.get('fragmentation_method', '')),
                            'collision_energy': self._parse_float_or_none(row.get('collision_energy', '')),
                            'instrument': self._str_or_none(row.get('instrument', '')),
                            'instrument_type': self._str_or_none(row.get('instrument_type', '')),
                            'formula': self._str_or_none(row.get('formula', '')),
                            'mono_isotopic_molecular_weight': self._parse_float_or_none(
                                row.get('exact_mass', '')
                            ),
                            'inchi': self._str_or_none(row.get('inchi', '')),
                            'smiles': self._str_or_none(row.get('smiles', '')),
                            'mz': mz.tolist(),
                            'intensities': intensities.tolist(),
                        })

                        # 3. Chunking: If buffer reaches batch_size, save to DB and clear
                        if len(records) >= batch_size:
                            inserted = dbi.batch_save_msms_refs(main_db_path, records)
                            file_inserted += inserted
                            total_inserted += inserted
                            records = []

                    except Exception as e:
                        logger.warning(f"  Row {row_num}: parse error — {e}")
                        n_skipped += 1
                        continue

            # 4. Final Flush: Save any remaining records left in the buffer
            if records:
                inserted = dbi.batch_save_msms_refs(main_db_path, records)
                file_inserted += inserted
                total_inserted += inserted

            total_skipped += n_skipped
            logger.info(
                f"  {source_path.name}: inserted {file_inserted} spectra"
                + (f", skipped {n_skipped}" if n_skipped else "")
            )

        logger.info(
            f"add-msms-refs complete: {total_inserted} spectra inserted total"
            + (f", {total_skipped} skipped" if total_skipped else "")
        )

@dataclass
class Compound:
    """A chemical compound record stored in the main metatlas database.
    """

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
    isomeric_smiles: str = ""
    molecular_weight: float | None = None
    xlogp: float | None = None
    tpsa: float | None = None
    complexity: float | None = None
    charge: int | None = None
    heavy_atom_count: int | None = None
    hbd_count: int | None = None
    hba_count: int | None = None
    rotatable_bond_count: int | None = None
    chebi_id: str = ""
    chembl_id: str = ""
    hmdb_id: str = ""
    kegg_id: str = ""
    drugbank_id: str = ""
    metabolights_id: str = ""
    metabolomics_workbench_id: str = ""
    wikipedia_url: str = ""
    description: str = ""
    pubchem_retrieval_date: str = ""
    pubchem_compound_url: str = ""
    modelseed_id: str = ""
    metacyc_id: str = ""

    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'Compound':
        """Create from atlas DataFrame row.
        """

        def _str(key: str, default: str = '') -> str:
            val = row.get(key, default)
            if val is None:
                return default
            try:
                if math.isnan(float(val)):
                    return default
            except (TypeError, ValueError):
                pass
            return str(val) if val != '' else default

        def _float(key: str) -> float | None:
            val = row.get(key, None)
            if val is None:
                return None
            try:
                f = float(val)
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        def _int(key: str) -> int | None:
            val = row.get(key, None)
            if val is None:
                return None
            try:
                f = float(val)
                return None if math.isnan(f) else int(f)
            except (TypeError, ValueError):
                return None

        return cls(
            compound_uid=_str('compound_uid'),
            compound_name=_str('compound_name'),
            inchi_key=_str('inchi_key'),
            inchi=_str('inchi'),
            smiles=_str('smiles'),
            formula=_str('formula'),
            classes=_str('classes'),
            pathways=_str('pathways'),
            tags=_str('tags'),
            mono_isotopic_molecular_weight=_float('mono_isotopic_molecular_weight') or 0.0,
            iupac_name=_str('iupac_name'),
            pubchem_cid=_str('pubchem_cid'),
            cas_number=_str('cas_number'),
            synonyms=_str('synonyms'),
            created_by=_str('created_by'),
            created_date=_str('created_date'),
            isomeric_smiles=_str('isomeric_smiles'),
            molecular_weight=_float('molecular_weight'),
            xlogp=_float('xlogp'),
            tpsa=_float('tpsa'),
            complexity=_float('complexity'),
            charge=_int('charge'),
            heavy_atom_count=_int('heavy_atom_count'),
            hbd_count=_int('hbd_count'),
            hba_count=_int('hba_count'),
            rotatable_bond_count=_int('rotatable_bond_count'),
            chebi_id=_str('chebi_id'),
            chembl_id=_str('chembl_id'),
            hmdb_id=_str('hmdb_id'),
            kegg_id=_str('kegg_id'),
            drugbank_id=_str('drugbank_id'),
            metabolights_id=_str('metabolights_id'),
            metabolomics_workbench_id=_str('metabolomics_workbench_id'),
            wikipedia_url=_str('wikipedia_url'),
            description=_str('description'),
            pubchem_retrieval_date=_str('pubchem_retrieval_date'),
            pubchem_compound_url=_str('pubchem_compound_url'),
            modelseed_id=_str('modelseed_id'),
            metacyc_id=_str('metacyc_id'),
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
            'created_date': self.created_date,
            'isomeric_smiles': self.isomeric_smiles,
            'molecular_weight': self.molecular_weight,
            'xlogp': self.xlogp,
            'tpsa': self.tpsa,
            'complexity': self.complexity,
            'charge': self.charge,
            'heavy_atom_count': self.heavy_atom_count,
            'hbd_count': self.hbd_count,
            'hba_count': self.hba_count,
            'rotatable_bond_count': self.rotatable_bond_count,
            'chebi_id': self.chebi_id,
            'chembl_id': self.chembl_id,
            'hmdb_id': self.hmdb_id,
            'kegg_id': self.kegg_id,
            'drugbank_id': self.drugbank_id,
            'metabolights_id': self.metabolights_id,
            'metabolomics_workbench_id': self.metabolomics_workbench_id,
            'wikipedia_url': self.wikipedia_url,
            'description': self.description,
            'pubchem_retrieval_date': self.pubchem_retrieval_date,
            'pubchem_compound_url': self.pubchem_compound_url,
            'modelseed_id': self.modelseed_id,
            'metacyc_id': self.metacyc_id,
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
    atlas_name: str  # short free-text key from YAML (e.g. "QCv7", "EMA")
    atlas_description: str
    chromatography: str
    polarity: str
    atlas_category: "AtlasCategory"
    atlas_stage: "AtlasStage" = field(default_factory=lambda: AtlasStage.INPUT)
    compound_mzrts: dict[str, "CompoundMZRT"] = field(default_factory=dict)
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

        def check_required(field, text):
            if not field:
                raise ValueError(f"Atlas {text} is missing")

        check_required(self.atlas_uid, "UID")
        check_required(self.atlas_name, "atlas_name")
        check_required(self.chromatography, "chromatography")
        check_required(self.polarity, "polarity")

        if not self.compound_mzrts or len(self.compound_mzrts) == 0:
            raise ValueError("No compound MZRTs in atlas")

        # Validate each CompoundMZRT
        for key, compound in self.compound_mzrts.items():
            inchi_key = compound.inchi_key or "<unknown>"
            if not compound.compound_uid:
                raise ValueError(f"CompoundMZRT {inchi_key} missing compound_uid")
            if compound.mz is None or compound.mz <= 0:
                raise ValueError(f"CompoundMZRT {inchi_key} has invalid m/z: {compound.mz}")
            #if compound.rt_peak is None or compound.rt_peak <= 0:
                #raise ValueError(f"CompoundMZRT {inchi_key} has invalid RT peak: {compound.rt_peak}")
            if compound.rt_min is None or compound.rt_max is None:
                raise ValueError(f"CompoundMZRT {inchi_key} missing RT min/max")
            # elif compound.rt_min >= compound.rt_max:
            #     raise ValueError(f"CompoundMZRT {inchi_key} has invalid RT bounds: {compound.rt_min} >= {compound.rt_max}")
            if not compound.adduct:
                raise ValueError(f"CompoundMZRT {inchi_key} missing adduct")

        logger.debug("Atlas passed validation!")

    def to_dataframe(self) -> pd.DataFrame:

        cat_val = self.atlas_category.value if isinstance(self.atlas_category, AtlasCategory) else str(self.atlas_category)
        stage_val = self.atlas_stage.value if isinstance(self.atlas_stage, AtlasStage) else str(self.atlas_stage)
        rows = []
        for compound_mzrt in self.compound_mzrts.values():
            compound_dict = asdict(compound_mzrt)
            compound_dict.update({
                'atlas_uid': self.atlas_uid,
                'atlas_name': self.atlas_name,
                'atlas_description': self.atlas_description,
                'chromatography': self.chromatography,
                'polarity': self.polarity,
                'atlas_category': cat_val,
                'atlas_stage': stage_val,
                'created_by': self.created_by,
                'created_date': self.created_date,
                'source': self.source,
                'source_atlas_uid': self.source_atlas_uid,
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
        atlas_uid = meta.get('atlas_uid', '')
        atlas_name = meta.get('atlas_name', '')
        atlas_description = meta.get('atlas_description', '')
        chromatography = meta.get('chromatography', '')
        polarity = meta.get('polarity', '')
        atlas_category = AtlasCategory(meta.get('atlas_category', 'REF'))
        atlas_stage = AtlasStage(meta.get('atlas_stage', 'INPUT'))
        created_by = meta.get('created_by', '')
        created_date = meta.get('created_date', '')
        source = meta.get('source', '')
        source_atlas_uid = meta.get('source_atlas_uid', None)
        rt_alignment_number = meta.get('rt_alignment_number', None)
        analysis_number = meta.get('analysis_number', None)

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
            atlas_category=atlas_category,
            atlas_stage=atlas_stage,
            compound_mzrts=compound_mzrts,
            created_by=created_by,
            created_date=created_date,
            source=source,
            source_atlas_uid=source_atlas_uid,
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
        cat_val = self.atlas_category.value if isinstance(self.atlas_category, AtlasCategory) else str(self.atlas_category)
        stage_val = self.atlas_stage.value if isinstance(self.atlas_stage, AtlasStage) else str(self.atlas_stage)
        return {
            'atlas_uid': self.atlas_uid,
            'atlas_name': self.atlas_name,
            'atlas_description': self.atlas_description,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'atlas_category': cat_val,
            'atlas_stage': stage_val,
            'created_by': self.created_by,
            'created_date': self.created_date,
            'source': self.source,
            'source_atlas_uid': self.source_atlas_uid,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full atlas as a plain dict — the inverse of :meth:`from_dataframe`.

        The returned dict contains all atlas-level metadata fields plus a
        ``compound_mzrts`` key whose value is a ``{mz_rt_uid: dict}`` mapping
        produced by calling :meth:`CompoundMZRT.to_dict` on each entry.  Round-
        tripping through ``Atlas.from_dataframe(pd.DataFrame(atlas.to_dict()['compound_mzrts'].values()))``
        reconstructs the original object.
        """
        cat_val = self.atlas_category.value if isinstance(self.atlas_category, AtlasCategory) else str(self.atlas_category)
        stage_val = self.atlas_stage.value if isinstance(self.atlas_stage, AtlasStage) else str(self.atlas_stage)
        return {
            'atlas_uid': self.atlas_uid,
            'atlas_name': self.atlas_name,
            'atlas_description': self.atlas_description,
            'chromatography': self.chromatography,
            'polarity': self.polarity,
            'atlas_category': cat_val,
            'atlas_stage': stage_val,
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

    def setup(self, project_name: str, rt_alignment_number: int, analysis_number: int, skip_alignment: bool = False):
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

        if skip_alignment:
            self.run_alignment = False
            self._skip_rt_align_routine()
            return

        self._check_existing_rt_aligned_atlases()

    def _skip_rt_align_routine(self):
        logger.info(
            "RT alignment skipped via --skip-rt-align flag. "
            "Copying reference atlases into project DB and registering in workflow_runs as RT_ALIGNED..."
        )
        for ta in self.config.targeted_analyses:
            self.aligned_atlas_obj = Atlas.from_database(
                database_path=self.paths['main_db_path'],
                atlas_uid=ta.atlas_uid
            )
            self.aligned_atlas_obj.chromatography = ta.chromatography
            self.aligned_atlas_obj.polarity = ta.polarity
            self.aligned_atlas_obj.atlas_category = ta.atlas_category
            self.aligned_atlas_obj.atlas_name = ta.atlas_name
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
    ms1_df_full_range: pd.DataFrame = field(default_factory=pd.DataFrame)
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
            logger.info(
                f"{autoided_count} AUTO_IDED run(s) already exist for "
                f"RTA{self.rt_alignment_number} and TGA{self.analysis_number}. "
                f"Per-analysis checks will skip already-completed analyses."
            )

class CurationStageBase(ABC):
    """Shared logic for AnalysisGUI and AnalysisSummary."""
    project_name: str | None
    project_db_path: str | None
    rt_alignment_number: int | None
    analysis_number: int | None
    chromatography: str | None
    polarity: str | None
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

        Paths are loaded from the ``project_config`` table (written during
        project setup) so that the same filesystem layout is used regardless
        of which user or working directory calls the GUI or summary stage.
        ``set_up_paths()`` is called as a fallback only when no stored paths
        are found (e.g. projects created before path serialisation was added).
        """
        self.rt_alignment_number = run_parameters['rt_alignment_number']
        self.analysis_number = run_parameters['analysis_number']
        self.chromatography = fpf.canonical_chromatography(run_parameters['chromatography'])
        self.polarity = fpf.canonical_polarity(run_parameters['polarity'])
        self.project_name = run_parameters['project_name']
        self.project_db_path = rtg.get_project_db_path(self.project_name)

        self.config = dbi.load_config_from_db(
            project_db_path=self.project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )

        self.owner = self.config.owner

        stored_paths = dbi.load_paths_from_db(
            project_db_path=self.project_db_path,
            rt_alignment_number=self.rt_alignment_number,
            analysis_number=self.analysis_number,
        )
        if stored_paths:
            self.paths = stored_paths
        else:
            logger.warning(
                "No stored paths found in project_config table for "
                f"RTA{self.rt_alignment_number}/TGA{self.analysis_number}. "
                "Falling back to set_up_paths() — paths may differ from the "
                "original pipeline run if the USER environment variable has changed."
            )
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
            self.chromatography, self.polarity, run_parameters.get('atlas_category'), run_parameters.get('atlas_name')
        )

    def _populate_note_options(self) -> None:
        ms2_notes_opts, ms1_notes_opts, other_notes_opts = gno.get_notes_opts(owner=self.owner)
        _gui_cfg = self.config.gui_config if self.config else {}
        _config_note_overrides = _gui_cfg.get("note_options_overrides") or {}
        _param_note_overrides = self.override_parameters.get("note_options_overrides") or {}
        note_overrides = {**_config_note_overrides, **_param_note_overrides}

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
    owner: str | None = field(default=None)
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
            / self.ta.label
        )
        os.makedirs(self.paths['analysis_results_output_dir'], exist_ok=True)

class ParquetQueryResult(NamedTuple):
    """Return type for :meth:`ParquetQueryInterpreter.execute`.

    Attributes
    ----------
    df:
        :class:`pandas.DataFrame` of rows matching the query filters.
    sql:
        Human-readable SQL SELECT statement equivalent to the query,
        suitable for display, logging, or passing to DuckDB directly.
    """
    df: pd.DataFrame
    sql: str

class ParquetQueryInterpreter:
    def __init__(self, parquet_output_path: Path, grain: str = "compound_lfc"):
        if grain not in pqi._GRAIN_SUBDIR:
            raise ValueError(f"grain must be one of {list(pqi._GRAIN_SUBDIR)}, got {grain!r}")

        dataset_dir = Path(parquet_output_path) / pqi._GRAIN_SUBDIR[grain]
        if not dataset_dir.exists():
            raise FileNotFoundError(f"No dataset found at {dataset_dir}")

        self.grain = grain
        self.dataset = ds.dataset(dataset_dir, format="parquet", partitioning=pqi._PARTITIONING)

    _SUFFIX_ORDER = ("_abs_min", "_abs_max", "_abs_gt", "_abs_lt", "_min", "_max", "_in")

    def _translate_to_expression(self, param_name: str, value: Any):
        col, op = param_name, "=="
        for suffix in self._SUFFIX_ORDER:
            if param_name.endswith(suffix):
                col = param_name[: -len(suffix)]
                op = suffix.lstrip("_")
                break

        field = ds.field(col)

        if op == "==":
            return field == value
        elif op == "min":
            return field >= value
        elif op == "max":
            return field <= value
        elif op == "in":
            val_list = value if isinstance(value, list) else [value]
            return field.isin(val_list)
        elif op == "abs_gt":
            return (field > value) | (field < -value)
        elif op == "abs_lt":
            return (field < value) & (field > -value)
        elif op == "abs_min":
            return (field >= value) | (field <= -value)
        elif op == "abs_max":
            return (field <= value) & (field >= -value)

        return None

    @staticmethod
    def _handle_condition_pair(value: list) -> "ds.Expression":
        """Symmetric match: rows where {condition_1, condition_2} == set(value),
        regardless of which one was stored first. Pair with log2_fold_change
        _abs_min / _abs_gt for magnitude-only comparisons, or use plain
        condition_1/condition_2 keys if you need a specific stored order."""
        c1_val, c2_val = value
        c1, c2 = ds.field("condition_1"), ds.field("condition_2")
        return ((c1 == c1_val) & (c2 == c2_val)) | ((c1 == c2_val) & (c2 == c1_val))

    @staticmethod
    def _handle_lfc_directional_min(value: dict) -> "ds.Expression":
        """Directional threshold: log2_fold_change of `treatment` relative to
        `control` >= threshold, correctly sign-flipping depending on which
        order the pair was stored in during the melt."""
        treatment = value["treatment"]
        control = value["control"]
        threshold = value["threshold"]
        c1, c2 = ds.field("condition_1"), ds.field("condition_2")
        lfc = ds.field("log2_fold_change")
        return (
            ((c1 == treatment) & (c2 == control) & (lfc >= threshold))
            | ((c1 == control) & (c2 == treatment) & (lfc <= -threshold))
        )

    def _build_filter(self, params: dict):
        schema_cols = set(self.dataset.schema.names)
        expressions = []

        for param_name, value in params.items():
            if param_name == "grain" or value is None:
                continue

            try:
                if param_name in pqi._SPECIAL_HANDLERS:
                    required = {"condition_1", "condition_2"} if param_name in (
                        "condition_pair", "lfc_directional_min"
                    ) else set()
                    missing = required - schema_cols
                    if missing:
                        print(f"Skipping '{param_name}': columns {missing} not present in grain "
                              f"'{self.grain}' (this key is compound_lfc-only).")
                        continue
                    handler = getattr(self, pqi._SPECIAL_HANDLERS[param_name])
                    expr = handler(value)
                else:
                    col = self._strip_suffix(param_name)
                    if col not in schema_cols:
                        print(f"Skipping '{param_name}': column '{col}' not present in grain "
                              f"'{self.grain}'. Check which section of the YAML this belongs to.")
                        continue
                    expr = self._translate_to_expression(param_name, value)

                if expr is not None:
                    expressions.append(expr)
            except Exception as e:
                print(f"Skipping parameter {param_name} due to error: {e}")

        final_filter = None
        for expr in expressions:
            final_filter = expr if final_filter is None else final_filter & expr
        return final_filter

    @staticmethod
    def _strip_suffix(param_name: str) -> str:
        for suffix in ParquetQueryInterpreter._SUFFIX_ORDER:
            if param_name.endswith(suffix):
                return param_name[: -len(suffix)]
        return param_name

    def _build_sql_string(self, params: dict) -> str:
        """Return a human-readable SQL SELECT statement equivalent to the query.

        The statement targets the virtual table name matching the grain
        (``compound_file`` or ``compound_lfc``) and renders each active
        filter parameter as a WHERE clause predicate.  Special handlers
        (``condition_pair``, ``lfc_directional_min``) are rendered as
        descriptive SQL comments so the intent is still visible.

        Parameters
        ----------
        params:
            The raw query parameter dict (same dict passed to ``execute``).

        Returns
        -------
        str
            A formatted SQL string suitable for display or logging.
        """
        schema_cols = set(self.dataset.schema.names)
        predicates: list[str] = []

        for param_name, value in params.items():
            if param_name == "grain" or value is None:
                continue

            try:
                if param_name in pqi._SPECIAL_HANDLERS:
                    required = {"condition_1", "condition_2"} if param_name in (
                        "condition_pair", "lfc_directional_min"
                    ) else set()
                    missing = required - schema_cols
                    if missing:
                        continue

                    if param_name == "condition_pair":
                        c1, c2 = value
                        predicates.append(
                            f"((condition_1 = '{c1}' AND condition_2 = '{c2}')"
                            f" OR (condition_1 = '{c2}' AND condition_2 = '{c1}'))"
                        )
                    elif param_name == "lfc_directional_min":
                        treatment = value["treatment"]
                        control = value["control"]
                        threshold = value["threshold"]
                        predicates.append(
                            f"((condition_1 = '{treatment}' AND condition_2 = '{control}'"
                            f" AND log2_fold_change >= {threshold})"
                            f" OR (condition_1 = '{control}' AND condition_2 = '{treatment}'"
                            f" AND log2_fold_change <= {-threshold}))"
                        )
                    else:
                        predicates.append(f"-- special handler: {param_name} = {value!r}")
                else:
                    col = self._strip_suffix(param_name)
                    if col not in schema_cols:
                        continue

                    op = "=="
                    for suffix in self._SUFFIX_ORDER:
                        if param_name.endswith(suffix):
                            op = suffix.lstrip("_")
                            break

                    if op == "==":
                        sql_val = f"'{value}'" if isinstance(value, str) else value
                        predicates.append(f"{col} = {sql_val}")
                    elif op == "min":
                        predicates.append(f"{col} >= {value}")
                    elif op == "max":
                        predicates.append(f"{col} <= {value}")
                    elif op == "in":
                        val_list = value if isinstance(value, list) else [value]
                        quoted = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in val_list)
                        predicates.append(f"{col} IN ({quoted})")
                    elif op == "abs_gt":
                        predicates.append(f"(ABS({col}) > {value})")
                    elif op == "abs_lt":
                        predicates.append(f"(ABS({col}) < {value})")
                    elif op == "abs_min":
                        predicates.append(f"(ABS({col}) >= {value})")
                    elif op == "abs_max":
                        predicates.append(f"(ABS({col}) <= {value})")
            except Exception:
                pass

        table_name = pqi._GRAIN_SUBDIR.get(self.grain, self.grain)
        if predicates:
            where_clause = "\n  AND ".join(predicates)
            return f"SELECT *\nFROM {table_name}\nWHERE {where_clause}"
        return f"SELECT *\nFROM {table_name}"

    def execute(self, params: dict) -> "ParquetQueryResult":
        """Execute the query and return a named tuple with ``df`` and ``sql``.

        Returns
        -------
        ParquetQueryResult
            A named tuple with two fields:

            * ``df``  – :class:`pandas.DataFrame` of matching rows.
            * ``sql`` – ``str`` SQL SELECT statement equivalent to the query.
        """
        final_filter = self._build_filter(params)
        table = (
            self.dataset.to_table(filter=final_filter)
            if final_filter is not None
            else self.dataset.to_table()
        )
        sql = self._build_sql_string(params)
        return ParquetQueryResult(df=table.to_pandas(), sql=sql)

    @classmethod
    def execute_from_params_file(
        cls,
        parquet_output_path: Path,
        params_path: Path,
        query_name: str | None = None,
    ) -> "ParquetQueryResult":
        with open(params_path, "r") as f:
            config = yaml.safe_load(f) or {}

        query_name = query_name
        if not query_name:
            raise ValueError("YAML defines multiple 'queries' but no query_name was passed")
        if query_name not in config["queries"]:
            raise KeyError(f"Query '{query_name}' not found. Available: {list(config['queries'])}")
        params = config["queries"][query_name]

        grain = params.get("grain", "compound_lfc")
        engine = cls(Path(parquet_output_path), grain)
        return engine.execute(params)