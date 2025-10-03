from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from enum import Enum
from datetime import datetime
import sys
import json
import pickle
import numpy as np
import pandas as pd
import shutil

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import ms1_ms2_analysis as msa
import targeted_analysis as tga
import load_tools as ldt
import pubchem_retrieval as pcr
import targeted_gui as tgi
from IPython.display import display, Markdown

logger = lcf.get_logger('workflow_objects')

class WorkflowStage(Enum):
    """Enumeration of workflow stages"""
    PROJECT_SETUP = "project_setup"
    RT_CORRECTION = "rt_correction" 
    PUTATIVE_IDENTIFICATION = "putative_identification"
    MANUAL_CURATION = "manual_curation"
    FINAL_REPORT = "final_report"

# =============================================================================
# Cache Manager (Centralized cache handling for all workflow stages)
# =============================================================================

class CacheManager:
    """
    Centralized cache manager for all workflow stages.
    Handles RT Correction, Putative Identifications, and Manual Curation caching.
    """
    
    def __init__(self, project_directory: str, config: Dict[str, Any] = None):
        """
        Initialize cache manager for a project.
        
        Args:
            project_directory: Path to the project directory
            config: Configuration dictionary with cache settings
        """
        self.project_directory = Path(project_directory)
        self.cache_base_dir = self.project_directory / "cache"
        self.config = config or {}
        
        # Define cache directories for each stage
        self.rt_correction_dir = self.cache_base_dir / "rt_correction"
        self.putative_ids_dir = self.cache_base_dir / "putative_ids"
        self.manual_curation_dir = self.cache_base_dir / "manual_curation"
        
        # Create cache directories
        self._setup_cache_directories()
    
    def _setup_cache_directories(self):
        """Create cache directory structure."""
        for cache_dir in [self.rt_correction_dir, self.putative_ids_dir, self.manual_curation_dir]:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    # =============================================================================
    # RT CORRECTION CACHING
    # =============================================================================
    
    def save_rt_correction(self, rt_models: Dict, corrected_atlases: Dict) -> str:
        """
        Save RT correction results to cache.
        
        Args:
            rt_models: Dictionary of RT correction models by method
            corrected_atlases: Dictionary of corrected atlas UIDs by type/method
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'rt_models': rt_models,
            'corrected_atlases': corrected_atlases,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'rt_correction'
        }
        
        # Save timestamped version
        cache_file = self.rt_correction_dir / f"rt_correction_{timestamp}.pkl"
        metadata_file = self.rt_correction_dir / f"rt_correction_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'rt_correction',
                'cache_version': '2.0',
                'rt_models_count': len(rt_models),
                'corrected_atlases_count': sum(
                    len(chroms) for atlas_chroms in corrected_atlases.values()
                    for chroms in atlas_chroms.values()
                ),
                'methods': list(rt_models.keys())
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks
            self._update_latest_symlinks(self.rt_correction_dir, "rt_correction", timestamp)
            
            logger.info(f"RT correction cache saved: {cache_file}")
            logger.info(f"  Models: {len(rt_models)} methods")
            logger.info(f"  Corrected atlases: {metadata['corrected_atlases_count']}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save RT correction cache: {e}")
            raise
    
    def load_rt_correction(self, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Load RT correction results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            Dictionary with rt_models and corrected_atlases, or None if not found
        """
        
        if timestamp:
            cache_file = self.rt_correction_dir / f"rt_correction_{timestamp}.pkl"
        else:
            cache_file = self.rt_correction_dir / "rt_correction_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No RT correction cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            logger.info(f"Loaded RT correction cache from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Models: {len(cache_data.get('rt_models', {}))}")
            logger.info(f"  Corrected atlases: {cache_data.get('corrected_atlases', {})}")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load RT correction cache: {e}")
            return None
    
    def has_rt_correction_cache(self) -> bool:
        """Check if RT correction cache exists."""
        return (self.rt_correction_dir / "rt_correction_latest.pkl").exists()
    
    # =============================================================================
    # PUTATIVE IDENTIFICATIONS CACHING
    # =============================================================================
    
    def save_putative_identifications(self, putative_ids: Dict, summary_stats: Dict = None) -> str:
        """
        Save putative identification results to cache.
        
        Args:
            putative_ids: Dictionary of putative identifications by atlas type and method
            summary_stats: Optional summary statistics
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'putative_ids': putative_ids,
            'summary_stats': summary_stats,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'putative_identifications'
        }
        
        # Save timestamped version
        cache_file = self.putative_ids_dir / f"putative_ids_{timestamp}.pkl"
        metadata_file = self.putative_ids_dir / f"putative_ids_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate metadata
            total_ids = sum(
                len(method_ids) for atlas_methods in putative_ids.values()
                for method_ids in atlas_methods.values()
            )
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'putative_identifications',
                'cache_version': '2.0',
                'total_putative_ids': total_ids,
                'by_atlas_type': {
                    atlas_type: sum(len(method_ids) for method_ids in methods.values())
                    for atlas_type, methods in putative_ids.items()
                },
                'atlas_methods': {
                    atlas_type: list(methods.keys())
                    for atlas_type, methods in putative_ids.items()
                }
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks
            self._update_latest_symlinks(self.putative_ids_dir, "putative_ids", timestamp)
            
            logger.info(f"Putative identifications cache saved: {cache_file}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save putative identifications cache: {e}")
            raise
    
    def load_putative_identifications(self, timestamp: Optional[str] = None) -> Optional[Dict]:
        """
        Load putative identification results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            Dictionary with putative_ids and summary_stats, or None if not found
        """
        
        if timestamp:
            cache_file = self.putative_ids_dir / f"putative_ids_{timestamp}.pkl"
        else:
            cache_file = self.putative_ids_dir / "putative_ids_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No putative identifications cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            total_ids = sum(
                len(method_ids) for atlas_methods in cache_data.get('putative_ids', {}).values()
                for method_ids in atlas_methods.values()
            )
            
            logger.info(f"Loaded putative identifications cache from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Total identifications: {total_ids}")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load putative identifications cache: {e}")
            return None
    
    def has_putative_identifications_cache(self) -> bool:
        """Check if putative identifications cache exists."""
        return (self.putative_ids_dir / "putative_ids_latest.pkl").exists()
    
    # =============================================================================
    # MANUAL CURATION CACHING
    # =============================================================================
    
    def save_manual_curation(self, putative_ids: Dict, curation_progress: Dict = None, 
                           partial_save: bool = False) -> str:
        """
        Save manual curation results to cache.
        Supports partial saves during curation process.
        
        Args:
            putative_ids: Dictionary of curated putative identifications
            curation_progress: Progress tracking information
            partial_save: Whether this is a partial save during curation
            
        Returns:
            timestamp of saved cache
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cache_data = {
            'putative_ids': putative_ids,
            'curation_progress': curation_progress,
            'timestamp': timestamp,
            'cache_version': '2.0',
            'stage': 'manual_curation',
            'partial_save': partial_save,
            'completed': not partial_save
        }
        
        # Use different naming for partial saves
        if partial_save:
            cache_file = self.manual_curation_dir / f"curation_partial_{timestamp}.pkl"
            metadata_file = self.manual_curation_dir / f"curation_partial_{timestamp}.json"
        else:
            cache_file = self.manual_curation_dir / f"curation_{timestamp}.pkl"
            metadata_file = self.manual_curation_dir / f"curation_{timestamp}.json"
        
        try:
            # Save cache data
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate curation statistics
            all_ids = []
            for atlas_methods in putative_ids.values():
                for method_ids in atlas_methods.values():
                    all_ids.extend(method_ids)
            
            curation_stats = {
                'pending': sum(1 for pid in all_ids if pid.curation_status == 'pending'),
                'reviewed': sum(1 for pid in all_ids if pid.curation_status == 'reviewed'),
                'finalized': sum(1 for pid in all_ids if pid.curation_status == 'finalized'),
                'rt_modified': sum(1 for pid in all_ids if pid.is_rt_modified),
                'annotation_modified': sum(1 for pid in all_ids if pid.is_annotation_modified)
            }
            
            # Save metadata
            metadata = {
                'timestamp': timestamp,
                'stage': 'manual_curation',
                'cache_version': '2.0',
                'partial_save': partial_save,
                'completed': not partial_save,
                'total_identifications': len(all_ids),
                'curation_stats': curation_stats,
                'progress_percent': ((curation_stats['reviewed'] + curation_stats['finalized']) / len(all_ids) * 100) if all_ids else 0
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update latest symlinks (always update, even for partial saves)
            self._update_latest_symlinks(self.manual_curation_dir, "curation", timestamp)
            
            save_type = "partial" if partial_save else "complete"
            logger.info(f"Manual curation cache saved ({save_type}): {cache_file}")
            logger.info(f"  Progress: {metadata['progress_percent']:.1f}%")
            logger.info(f"  Curation stats: {curation_stats}")
            
            return timestamp
            
        except Exception as e:
            logger.error(f"Failed to save manual curation cache: {e}")
            raise
    
    def load_manual_curation(self, timestamp: Optional[str] = None, 
                           prefer_partial: bool = True) -> Optional[Dict]:
        """
        Load manual curation results from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            prefer_partial: If True, prefer partial saves over complete ones when loading latest
            
        Returns:
            Dictionary with putative_ids and curation_progress, or None if not found
        """
        # Check if manual curation caching is enabled
        use_cache = self.config.get("analysis_settings", {}).get("use_manual_curation_cache", False)
        if not use_cache:
            logger.info("Manual curation caching disabled in config - skipping cache load")
            return None
        
        if timestamp:
            # Try both partial and complete versions for specific timestamp
            cache_file = self.manual_curation_dir / f"curation_{timestamp}.pkl"
            if not cache_file.exists():
                cache_file = self.manual_curation_dir / f"curation_partial_{timestamp}.pkl"
        else:
            # Load latest - check for partial saves first if preferred
            if prefer_partial:
                # Find latest partial save
                partial_files = list(self.manual_curation_dir.glob("curation_partial_*.pkl"))
                complete_files = list(self.manual_curation_dir.glob("curation_[0-9]*.pkl"))
                
                all_files = partial_files + complete_files
                if all_files:
                    # Sort by timestamp in filename and get most recent
                    latest_file = max(all_files, key=lambda f: f.stem.split('_')[-1])
                    cache_file = latest_file
                else:
                    cache_file = self.manual_curation_dir / "curation_latest.pkl"
            else:
                cache_file = self.manual_curation_dir / "curation_latest.pkl"
        
        if not cache_file.exists():
            logger.info("No manual curation cache found")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Calculate current progress
            all_ids = []
            for atlas_methods in cache_data.get('putative_ids', {}).values():
                for method_ids in atlas_methods.values():
                    all_ids.extend(method_ids)
            
            reviewed_count = sum(1 for pid in all_ids if pid.curation_status in ['reviewed', 'finalized'])
            progress_percent = (reviewed_count / len(all_ids) * 100) if all_ids else 0
            
            save_type = "partial" if cache_data.get('partial_save', False) else "complete"
            logger.info(f"Loaded manual curation cache ({save_type}) from {cache_data.get('timestamp', 'unknown')}")
            logger.info(f"  Total identifications: {len(all_ids)}")
            logger.info(f"  Progress: {progress_percent:.1f}%")
            
            return cache_data
            
        except Exception as e:
            logger.error(f"Failed to load manual curation cache: {e}")
            return None
    
    def has_manual_curation_cache(self) -> bool:
        """Check if manual curation cache exists (including partial saves)."""
        latest_exists = (self.manual_curation_dir / "curation_latest.pkl").exists()
        partial_exists = len(list(self.manual_curation_dir.glob("curation_partial_*.pkl"))) > 0
        return latest_exists or partial_exists
    
    def auto_save_curation_progress(self, putative_ids: Dict, interval_minutes: int = 5) -> None:
        """
        Auto-save curation progress at regular intervals.
        This would be called periodically during the GUI curation process.
        
        Args:
            putative_ids: Current state of putative identifications
            interval_minutes: How often to auto-save
        """
        # Check if enough time has passed since last auto-save
        last_autosave_file = self.manual_curation_dir / ".last_autosave"
        
        should_save = True
        if last_autosave_file.exists():
            try:
                last_save_time = datetime.fromisoformat(last_autosave_file.read_text().strip())
                time_since_save = (datetime.now() - last_save_time).total_seconds() / 60
                should_save = time_since_save >= interval_minutes
            except Exception:
                should_save = True
        
        if should_save:
            try:
                self.save_manual_curation(putative_ids, partial_save=True)
                last_autosave_file.write_text(datetime.now().isoformat())
                logger.debug("Auto-saved curation progress")
            except Exception as e:
                logger.error(f"Auto-save failed: {e}")
    
    def _update_latest_symlinks(self, cache_dir: Path, prefix: str, timestamp: str) -> None:
        """Update latest symlinks for a cache type."""
        latest_pkl = cache_dir / f"{prefix}_latest.pkl"
        latest_json = cache_dir / f"{prefix}_latest.json"
        
        target_pkl = f"{prefix}_{timestamp}.pkl"
        target_json = f"{prefix}_{timestamp}.json"
        
        # Handle partial saves for manual curation
        if prefix == "curation" and (cache_dir / f"curation_partial_{timestamp}.pkl").exists():
            target_pkl = f"curation_partial_{timestamp}.pkl"
            target_json = f"curation_partial_{timestamp}.json"
        
        # Update symlinks
        for latest_file, target_file in [(latest_pkl, target_pkl), (latest_json, target_json)]:
            if latest_file.exists():
                latest_file.unlink()
            if (cache_dir / target_file).exists():
                latest_file.symlink_to(target_file)

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
# EXPERIMENTAL COMPOUND DATA (Mutable analysis info + experimental results)
# =============================================================================

@dataclass
class CompoundExperimental:
    """
    Mutable compound data for targeted analysis workflow.
    Contains reference data + experimental results + user modifications.
    Maps to database targeted_analysis table.
    """
    
    # Core identifiers (from reference)
    compound_uid: str
    inchi_key: str
    compound_name: str
    
    # Chemical properties (immutable from reference)
    formula: str = ""
    mz: float = 0.0
    adduct: str = ""
    polarity: str = ""
    chromatography: str = ""
    mz_tolerance: float = 5.0
    
    # Original atlas RT data (immutable reference)
    atlas_rt_peak: float = 0.0
    atlas_rt_min: float = 0.0
    atlas_rt_max: float = 0.0

    # Current RT data (modifiable during analysis)
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    
    # Analysis annotations (modifiable)
    ms1_notes: str = "keep"
    ms2_notes: str = "no selection"
    analyst_notes: str = ""
    identification_notes: str = ""
    
    # Best EIC results (populated during analysis)
    best_eic_file: str = ""
    best_eic_rt: float = 0.0
    best_eic_mz: float = 0.0
    best_eic_intensity: float = 0.0
    best_eic_ppm_error: float = 0.0
    best_eic_rt_error: float = 0.0
    
    # Average EIC results
    avg_eic_rt: float = 0.0
    avg_eic_intensity: float = 0.0
    avg_eic_mz: float = 0.0
    
    # Best MS2 results
    best_ms2_file: str = ""
    best_ms2_database: str = ""
    best_ms2_ref_id: str = ""
    best_ms2_rt: float = 0.0
    best_ms2_intensity: float = 0.0
    best_ms2_mz: float = 0.0
    best_ms2_score: float = 0.0
    best_ms2_num_matches: int = 0
    best_ms2_ref_frags: int = 0
    best_ms2_data_frags: int = 0
    best_ms2_matched_fragments: List[float] = field(default_factory=list)
    best_ms2_selection_method: str = "none"
    
    # Average MS2 results
    avg_ms2_score: float = 0.0
    
    # Detection summary
    total_files_detected: int = 0
    ms2_files_with_data: int = 0
    
    # Raw data storage (for GUI - not persisted to database)
    eic_data_files: Dict[str, Dict] = field(default_factory=dict)
    ms2_data_files: Dict[str, Dict] = field(default_factory=dict)
    suggested_rt_bounds: Optional[Dict] = None
    isomers: List[Dict] = field(default_factory=list)
    
    # Workflow state tracking
    is_rt_modified: bool = False
    is_annotation_modified: bool = False
    curation_status: str = "pending"  # pending, reviewed, finalized
    
    def __post_init__(self):
        """Initialize derived values after creation."""
        if self.rt_peak == 0.0 and self.atlas_rt_peak > 0.0:
            self.rt_peak = self.atlas_rt_peak
            self.rt_min = self.atlas_rt_min
            self.rt_max = self.atlas_rt_max
    
    @classmethod
    def from_compound_and_reference(cls, compound: Compound, compound_ref: CompoundReference) -> 'CompoundExperimental':
        """Create from Compound and CompoundReference objects."""
        return cls(
            compound_uid=compound.compound_uid,
            inchi_key=compound.inchi_key,
            compound_name=compound.name,
            formula=compound.formula,
            mz=compound_ref.mz,
            adduct=compound_ref.adduct,
            polarity=compound_ref.polarity,
            chromatography=compound_ref.chromatography,
            mz_tolerance=compound_ref.mz_tolerance,
            atlas_rt_peak=compound_ref.rt_peak,
            atlas_rt_min=compound_ref.rt_min,
            atlas_rt_max=compound_ref.rt_max,
            rt_peak=compound_ref.rt_peak,
            rt_min=compound_ref.rt_min,
            rt_max=compound_ref.rt_max
        )
    
    @classmethod
    def from_atlas_row(cls, atlas_row: pd.Series) -> 'CompoundExperimental':
        """Create from atlas DataFrame row (for compatibility)."""
        return cls(
            compound_uid=atlas_row.get('compound_uid', ''),
            inchi_key=atlas_row.get('inchi_key', ''),
            compound_name=atlas_row.get('compound_name', atlas_row.get('label', '')),
            formula=atlas_row.get('formula', ''),
            mz=atlas_row.get('mz', 0.0),
            adduct=atlas_row.get('adduct', ''),
            polarity=atlas_row.get('polarity', ''),
            chromatography=atlas_row.get('chromatography', ''),
            mz_tolerance=atlas_row.get('mz_tolerance', 5.0),
            atlas_rt_peak=atlas_row.get('rt_peak', 0.0),
            atlas_rt_min=atlas_row.get('rt_min', 0.0),
            atlas_rt_max=atlas_row.get('rt_max', 0.0),
            rt_peak=atlas_row.get('rt_peak', 0.0),
            rt_min=atlas_row.get('rt_min', 0.0),
            rt_max=atlas_row.get('rt_max', 0.0)
        )
    
    # Analysis methods
    def add_eic_data(self, filename: str, eic_dict: Dict):
        """Add EIC data for a file."""
        self.eic_data_files[filename] = eic_dict
        self.total_files_detected = len(self.eic_data_files)
        self._update_best_eic()
    
    def add_ms2_data(self, filename: str, ms2_dict: Dict):
        """Add MS2 data for a file."""
        self.ms2_data_files[filename] = ms2_dict
        if ms2_dict.get('ms2_entries'):
            self.ms2_files_with_data += 1
        self._update_best_ms2()
    
    def update_rt_bounds(self, rt_min: float, rt_max: float, rt_peak: float):
        """Update RT bounds and mark as modified."""
        self.rt_min = rt_min
        self.rt_max = rt_max
        self.rt_peak = rt_peak
        self.is_rt_modified = True
    
    def update_annotations(self, ms1_notes: str = None, ms2_notes: str = None, 
                          analyst_notes: str = None, identification_notes: str = None):
        """Update annotations and mark as modified."""
        if ms1_notes is not None:
            self.ms1_notes = ms1_notes
            self.is_annotation_modified = True
        if ms2_notes is not None:
            self.ms2_notes = ms2_notes
            self.is_annotation_modified = True
        if analyst_notes is not None:
            self.analyst_notes = analyst_notes
            self.is_annotation_modified = True
        if identification_notes is not None:
            self.identification_notes = identification_notes
            self.is_annotation_modified = True
    
    def _update_best_eic(self):
        """Update best EIC statistics from current data."""
        if not self.eic_data_files:
            return
        
        best_intensity = 0.0
        best_file = ""
        
        for filename, eic_data in self.eic_data_files.items():
            intensity = eic_data.get('intensity_peak', 0.0)
            if intensity > best_intensity:
                best_intensity = intensity
                best_file = filename
                self.best_eic_file = filename
                self.best_eic_rt = eic_data.get('rt_peak', 0.0)
                self.best_eic_mz = eic_data.get('mz_peak', 0.0)
                self.best_eic_intensity = intensity
                self.best_eic_ppm_error = eic_data.get('ppm_diff', 0.0)
                self.best_eic_rt_error = eic_data.get('rt_diff', 0.0)
        
        # Calculate averages
        if self.eic_data_files:
            intensities = [d.get('intensity_peak', 0.0) for d in self.eic_data_files.values()]
            rts = [d.get('rt_peak', 0.0) for d in self.eic_data_files.values()]
            mzs = [d.get('mz_peak', 0.0) for d in self.eic_data_files.values()]
            
            self.avg_eic_intensity = np.mean(intensities)
            self.avg_eic_rt = np.mean(rts)
            self.avg_eic_mz = np.mean(mzs)
    
    def _update_best_ms2(self):
        """Update best MS2 statistics from current data."""
        if not self.ms2_data_files:
            return
        
        best_score = 0.0
        best_hit_data = None
        
        for filename, ms2_data in self.ms2_data_files.items():
            if isinstance(ms2_data, dict):
                best_hit = ms2_data.get('best_hit', {})
                if best_hit and best_hit.get('score', 0.0) > best_score:
                    best_score = best_hit.get('score', 0.0)
                    best_hit_data = best_hit
                    self.best_ms2_file = filename
                    self.best_ms2_selection_method = "reference_hit"
        
        if best_hit_data:
            self.best_ms2_database = best_hit_data.get('database', '')
            self.best_ms2_ref_id = best_hit_data.get('ref_id', '')
            self.best_ms2_score = best_hit_data.get('score', 0.0)
            self.best_ms2_num_matches = best_hit_data.get('num_matches', 0)
            self.best_ms2_matched_fragments = best_hit_data.get('matched_fragments', [])
            self.best_ms2_ref_frags = best_hit_data.get('ref_frags', 0)
            self.best_ms2_data_frags = best_hit_data.get('data_frags', 0)
            self.best_ms2_rt = best_hit_data.get('rt_measured', 0.0)
            self.best_ms2_intensity = best_hit_data.get('qry_intensity_peak', 0.0)
            self.best_ms2_mz = best_hit_data.get('mz_measured', 0.0)
        else:
            # No hits, find best by intensity
            best_intensity = 0.0
            for filename, ms2_data in self.ms2_data_files.items():
                if isinstance(ms2_data, dict):
                    best_ms2 = ms2_data.get('best_ms2', {})
                    intensity = best_ms2.get('intensity_peak', 0.0)
                    if intensity > best_intensity:
                        best_intensity = intensity
                        self.best_ms2_file = filename
                        self.best_ms2_rt = best_ms2.get('rt_peak', 0.0)
                        self.best_ms2_intensity = intensity
                        self.best_ms2_mz = best_ms2.get('precursor_mz', 0.0)
                        self.best_ms2_selection_method = "highest_intensity"
        
        # Calculate average score from all files
        all_scores = []
        for ms2_data in self.ms2_data_files.values():
            if isinstance(ms2_data, dict):
                for hit in ms2_data.get('all_hits', []):
                    all_scores.append(hit.get('score', 0.0))
        
        self.avg_ms2_score = np.mean(all_scores) if all_scores else 0.0
    
    def to_plot_data_format(self) -> Dict:
        """Convert to GUI plot_data format for compatibility."""
        return {
            'original_atlas_data': {
                'compound_name': self.compound_name,
                'inchi_key': self.inchi_key,
                'formula': self.formula,
                'mz': self.mz,
                'adduct': self.adduct,
                'polarity': self.polarity,
                'rt_peak': self.atlas_rt_peak,
                'rt_min': self.atlas_rt_min,
                'rt_max': self.atlas_rt_max,
                'mz_tolerance': self.mz_tolerance,
                'isomers': self.isomers,
                'ms1_notes': 'keep',
                'ms2_notes': 'no selection',
                'analyst_notes': '',
                'identification_notes': ''
            },
            'new_atlas_data': {
                'compound_name': self.compound_name,
                'inchi_key': self.inchi_key,
                'formula': self.formula,
                'mz': self.mz,
                'adduct': self.adduct,
                'polarity': self.polarity,
                'rt_peak': self.rt_peak,
                'rt_min': self.rt_min,
                'rt_max': self.rt_max,
                'mz_tolerance': self.mz_tolerance,
                'isomers': self.isomers,
                'ms1_notes': self.ms1_notes,
                'ms2_notes': self.ms2_notes,
                'analyst_notes': self.analyst_notes,
                'identification_notes': self.identification_notes
            },
            'suggested_rt_bounds_data': self.suggested_rt_bounds,
            'eic_data': self.eic_data_files,
            'best_eic': {
                'file_peak': self.best_eic_file,
                'rt_peak': self.best_eic_rt,
                'mz_peak': self.best_eic_mz,
                'intensity_peak': self.best_eic_intensity,
                'ppm_diff': self.best_eic_ppm_error,
                'rt_diff': self.best_eic_rt_error
            },
            'avg_eic': {
                'rt_peak': self.avg_eic_rt,
                'intensity_peak': self.avg_eic_intensity,
                'mz_peak': self.avg_eic_mz
            },
            'best_ms2': {
                'file_peak': self.best_ms2_file,
                'database': self.best_ms2_database,
                'ref_id': self.best_ms2_ref_id,
                'rt_peak': self.best_ms2_rt,
                'intensity_peak': self.best_ms2_intensity,
                'mz_peak': self.best_ms2_mz,
                'score': self.best_ms2_score,
                'num_matches': self.best_ms2_num_matches,
                'ref_frags': self.best_ms2_ref_frags,
                'data_frags': self.best_ms2_data_frags,
                'matched_fragments': self.best_ms2_matched_fragments,
                'selection_method': self.best_ms2_selection_method
            },
            'avg_ms2': {
                'avg_score': self.avg_ms2_score
            },
            'ms2_data': self.ms2_data_files,
            'is_modified': self.is_rt_modified or self.is_annotation_modified
       }

# =============================================================================
# ATLAS (Collection of compounds)
# =============================================================================

@dataclass
class Atlas:
    """
    Collection of reference compounds with RT/MZ data.
    Maps to database atlas + atlas_compound_associations tables.
    """
    
    # Core metadata
    atlas_uid: str
    atlas_name: str
    atlas_description: str
    chromatography: str
    polarity: str
    
    # Compound references (immutable reference data)
    compound_references: Dict[str, CompoundReference] = field(default_factory=dict)
    
    # Atlas metadata
    created_by: str = ""
    last_modified: str = ""
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
# WORKFLOW RESULTS CONTAINER (Central storage for all data as analysis progresses)
# =============================================================================

@dataclass
class WorkflowResults:
    """
    Centralized container for all workflow results.
    Consolidates data that was previously scattered across manager objects.
    """
    # Stage 2: RT Correction Results
    rt_models: Dict[str, Dict] = field(default_factory=dict)  # {chrom_polarity: model_dict}
    corrected_atlases: Dict[str, Dict[str, Dict[str, str]]] = field(default_factory=dict)  # {atlas_type: {chrom: {pol: atlas_uid}}}
    
    # Stage 3: Putative Identification Results
    putative_ids: Dict[str, Dict[str, List[CompoundExperimental]]] = field(default_factory=dict)  # {atlas_type: {chrom_pol: [CompoundExperimental]}}
    summary_stats: Optional[Dict[str, Any]] = field(default=None)  # Summary statistics for quick access
    
    # Stage 4: Manual Curation Cache
    cached_curation_results: Optional[Dict[str, Any]] = field(default=None)
    
    # Metadata
    created_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def update_timestamp(self):
        """Update the last modified timestamp"""
        self.last_modified = datetime.now().isoformat()
    
    # RT Correction Methods
    def has_rt_correction_results(self) -> bool:
        """Check if RT correction has been completed"""
        return len(self.rt_models) > 0 and len(self.corrected_atlases) > 0
    
    def get_rt_correction_summary(self) -> Dict[str, Any]:
        """Get summary of RT correction results"""
        return {
            'rt_models_count': len(self.rt_models),
            'corrected_atlases': {
                atlas_type: {
                    chrom: list(pols.keys()) 
                    for chrom, pols in chroms.items()
                } for atlas_type, chroms in self.corrected_atlases.items()
            },
            'methods_corrected': list(self.rt_models.keys())
        }
    
    def get_corrected_atlas_uid(self, atlas_type: str, chrom: str, pol: str) -> Optional[str]:
        """Get corrected atlas UID for specific type/chrom/pol combination"""
        return self.corrected_atlases.get(atlas_type, {}).get(chrom, {}).get(pol)
    
    def set_corrected_atlas_uid(self, atlas_type: str, chrom: str, pol: str, atlas_uid: str) -> None:
        """Set corrected atlas UID for specific type/chrom/pol combination"""
        if atlas_type not in self.corrected_atlases:
            self.corrected_atlases[atlas_type] = {}
        if chrom not in self.corrected_atlases[atlas_type]:
            self.corrected_atlases[atlas_type][chrom] = {}
        self.corrected_atlases[atlas_type][chrom][pol] = atlas_uid
        self.update_timestamp()
    
    # Putative Identification Methods
    def has_putative_identification_results(self) -> bool:
        """Check if putative identification has been completed"""
        return len(self.putative_ids) > 0 and any(
            len(methods) > 0 for methods in self.putative_ids.values()
        )
    
    def get_all_putative_ids(self) -> List[CompoundExperimental]:
        """Get flat list of all putative IDs across all atlas types and methods"""
        all_ids = []
        for atlas_type, chrom_pol in self.putative_ids.items():
            for putative_list in chrom_pol.values():
                all_ids.extend(putative_list)
        return all_ids

    def get_putative_identification_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary statistics for putative identifications"""
        
        stats = {
            'total_putative_ids': len(self.get_all_putative_ids()),
            'by_atlas_type': {},
            'by_curation_status': {'pending': 0, 'reviewed': 0, 'finalized': 0},
            'with_ms2_data': 0,
            'with_reference_hits': 0,
            'rt_modified': 0,
            'annotation_modified': 0
        }
        
        # Calculate by atlas type
        for atlas_type, chrom_pol in self.putative_ids.items():
            type_count = sum(len(putative_ids) for putative_ids in chrom_pol.values())
            stats['by_atlas_type'][atlas_type] = type_count
        
        # Calculate detailed statistics from individual CompoundExperimental objects
        for pid in self.get_all_putative_ids():
            # CompoundExperimental has curation_status field
            status = getattr(pid, 'curation_status', 'pending')
            stats['by_curation_status'][status] += 1
            if pid.best_ms2_file:
                stats['with_ms2_data'] += 1
            if pid.best_ms2_score > 0:
                stats['with_reference_hits'] += 1
            if pid.is_rt_modified:
                stats['rt_modified'] += 1
            if pid.is_annotation_modified:
                stats['annotation_modified'] += 1
        
        # Cache the stats for quick access
        self.summary_stats = stats
        return stats
    
    def set_putative_ids(self, atlas_type: str, chrom_pol: str, putative_list: List[CompoundExperimental]) -> None:
        """Set putative IDs for specific atlas type and method"""
        if atlas_type not in self.putative_ids:
            self.putative_ids[atlas_type] = {}
        else:
            logger.warning(f"Overwriting existing putative IDs for {atlas_type} - {chrom_pol}")
        self.putative_ids[atlas_type][chrom_pol] = putative_list
        logger.info(f"Set {len(putative_list)} putative IDs for {atlas_type} - {chrom_pol}")
        self.update_timestamp()
    
    # Manual Curation Methods
    def has_curation_data(self) -> bool:
        """Check if manual curation cache exists"""
        return self.cached_curation_results is not None
    
    def get_curation_progress(self) -> Dict:
        """Get curation progress summary"""
        stats = self.get_putative_identification_summary()
        total = stats['total_putative_ids']
        
        if total == 0:
            return {'progress_percent': 0, 'details': stats}
        
        reviewed_and_finalized = stats['by_curation_status']['reviewed'] + stats['by_curation_status']['finalized']
        progress_percent = (reviewed_and_finalized / total) * 100
        
        return {
            'progress_percent': progress_percent,
            'reviewed_count': reviewed_and_finalized,
            'total_count': total,
            'details': stats
        }
    
    def update_curation_status(self, inchi_key: str, new_status: str) -> None:
        """Update curation status for a specific compound"""
        for pid in self.get_all_putative_ids():
            if pid.inchi_key == inchi_key:
                pid.curation_status = new_status
                self.update_timestamp()
                break
    
    # Utility Methods
    def get_workflow_completion_status(self) -> Dict[str, bool]:
        """Get completion status of each workflow stage"""
        return {
            'rt_correction': self.has_rt_correction_results(),
            'putative_identification': self.has_putative_identification_results(),
            'manual_curation': self.has_curation_data()
        }
    
    def clear_stage_results(self, stage: str) -> None:
        """Clear results for a specific workflow stage"""
        if stage == 'rt_correction':
            self.rt_models.clear()
            self.corrected_atlases.clear()
        elif stage == 'putative_identification':
            self.putative_ids.clear()
        elif stage == 'manual_curation':
            self.cached_curation_results = None
        self.update_timestamp()
    
    def clear_all_results(self) -> None:
        """Clear all workflow results"""
        self.rt_models.clear()
        self.corrected_atlases.clear()
        self.putative_ids.clear()
        self.cached_curation_results = None
        self.update_timestamp()


# =============================================================================
# MAIN WORKFLOW ORCHESTRATOR (controls all steps of analysis)
# =============================================================================

@dataclass
class TargetedAnalysisManager:
    """
    Main workflow orchestrator that manages the complete targeted metabolomics workflow.
    
    Schema alignment:
    - TargetedAnalysisManager (organizer)
        - WorkflowResults (central results container)
            - Atlas (atlases from database by UID - minimum one QC + one target)
                - Compound (comes with atlas)
                    - CompoundReference (comes with atlas)
                    - CompoundExperimental (created during analysis -> mz_rt_experimental table)
    """
    config: Dict
    project_directory: str
    project_name: str
    project_lcmsruns_path: str
    rt_alignment_number: int = 1
    analysis_number: int = 1
    current_stage: WorkflowStage = WorkflowStage.PROJECT_SETUP
    results: WorkflowResults = field(default_factory=WorkflowResults)
    cache_manager: Optional[CacheManager] = field(default=None, init=False)
    
    # H5 files by type
    h5_files: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
    # Compute fields
    main_db_path: str = field(init=False)
    atlas_data: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        # Use single project database with iteration tracking
        self.rt_alignment_directory = str(Path(self.project_directory) / f"{self.project_name}_RTA{self.rt_alignment_number}")
        self.analysis_directory = str(Path(self.rt_alignment_directory) / f"{self.project_name}_RTA{self.rt_alignment_number}_ALY{self.analysis_number}")
        
        # Single project database for all iterations
        self.project_db_path = str(Path(self.project_directory) / f"{self.project_name}.duckdb")
        self.main_db_path = self.config["ENV"]["PATHS"]["main_database"]
        
        # Store current iteration info
        self.current_rt_alignment = self.rt_alignment_number
        self.current_analysis = self.analysis_number
        
        # Set up cache manager
        self.cache_manager = CacheManager(self.analysis_directory, self.config)
    
    def _setup_project_database(self, new_lcmsruns: bool = False) -> None:
        """Create project database and load LCMS run files."""
        logger.info(f"Creating project database at {self.project_db_path}...")
        dbi.create_project_database(self.project_db_path, self.rt_alignment_number, self.analysis_number)
        
        logger.info("Parsing analysis configuration file to get atlas and compound info...")
        self.atlas_data = self._parse_config()

        logger.info(f"Loading LCMS runs from {self.project_lcmsruns_path}...")
        try:
            files_by_group = dbi.save_lcmsruns_to_db(
                self.project_db_path,
                self.project_name, 
                self.project_lcmsruns_path,
                new_lcmsruns,
            )
            logger.info("LCMS runs loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load LCMS runs: {e}")
            logger.info("Continuing with workflow - files may need to be loaded manually")
    
    def load_atlases_for_workflow(self) -> Dict[str, Atlas]:
        """
        Load all atlases required for the workflow from database.
        This demonstrates the schema: Atlas -> Compound -> CompoundReference
        
        Returns:
            Dict mapping atlas_uid to Atlas objects
        """
        atlases = {}
        atlas_manager = AtlasManager(self.config)
        
        # Load RT alignment atlas (QC atlas)
        rt_align_template_atlas_data = self.atlas_data.get('rt_align_template_atlas')
        if rt_align_template_atlas_data:
            atlas_uid = rt_align_template_atlas_data['atlas_uid']
            atlas = atlas_manager.load_atlas_from_database(atlas_uid, self.main_db_path)
            atlases[atlas_uid] = atlas
            logger.info(f"Loaded RT alignment atlas: {atlas.atlas_name} with {len(atlas.compound_references)} compound references")
        
        # Load analysis atlases (target atlases)
        for analysis_atlas_data in self.atlas_data.get('analysis_atlases', []):
            atlas_uid = analysis_atlas_data['atlas_uid']
            atlas = atlas_manager.load_atlas_from_database(atlas_uid, self.main_db_path)
            atlases[atlas_uid] = atlas
            logger.info(f"Loaded analysis atlas: {atlas.atlas_name} with {len(atlas.compound_references)} compound references")
        
        return atlases

    def convert_compound_references_to_experimental(self, atlas: Atlas) -> List[CompoundExperimental]:
        """
        Convert CompoundReference objects from an Atlas to CompoundExperimental objects for analysis.
        This demonstrates the schema: CompoundReference -> CompoundExperimental
        
        Args:
            atlas: Atlas object containing CompoundReference objects
            
        Returns:
            List of CompoundExperimental objects ready for targeted analysis
        """
        experimental_compounds = []
        
        # We need to get Compound objects to create CompoundExperimental objects
        # This requires a database lookup to get the compound metadata
        compound_uids = [ref.compound_uid for ref in atlas.compound_references.values()]
        
        # Get compound metadata from database
        compounds_dict = {}
        with dbi.get_db_connection(self.main_db_path) as conn:
            if compound_uids:
                placeholders = ','.join(['?'] * len(compound_uids))
                compound_data = conn.execute(f"""
                    SELECT * FROM compounds WHERE compound_uid IN ({placeholders})
                """, compound_uids).fetchall()
                
                columns = [desc[0] for desc in conn.description]
                for row_data in compound_data:
                    row_dict = dict(zip(columns, row_data))
                    compound = Compound(
                        compound_uid=row_dict['compound_uid'],
                        name=row_dict['name'],
                        inchi_key=row_dict['inchi_key'],
                        inchi=row_dict['inchi'] or '',
                        smiles=row_dict['smiles'] or '',
                        formula=row_dict['formula'] or '',
                        compound_classes=row_dict['compound_classes'] or '',
                        compound_pathways=row_dict['compound_pathways'] or '',
                        compound_tags=row_dict['compound_tags'] or '',
                        mono_isotopic_molecular_weight=row_dict['mono_isotopic_molecular_weight'] or 0.0,
                        iupac_name=row_dict['iupac_name'] or '',
                        pubchem_cid=row_dict['pubchem_cid'] or '',
                        cas_number=row_dict['cas_number'] or '',
                        synonyms=row_dict['synonyms'] or '',
                        created_by=row_dict['created_by'] or '',
                        created_date=row_dict['created_date'] or ''
                    )
                    compounds_dict[compound.compound_uid] = compound
        
        # Create CompoundExperimental objects
        for inchi_key, compound_ref in atlas.compound_references.items():
            compound = compounds_dict.get(compound_ref.compound_uid)
            if compound:
                experimental_compound = CompoundExperimental.from_compound_and_reference(
                    compound, compound_ref
                )
                experimental_compounds.append(experimental_compound)
            else:
                logger.warning(f"Compound {compound_ref.compound_uid} not found in database")
        
        logger.info(f"Converted {len(experimental_compounds)} CompoundReference objects to CompoundExperimental objects")
        return experimental_compounds

    def load_h5_files(self) -> None:
        """Load and categorize H5 file dataframes of info from project database"""
        self.h5_files = {
            'qc': dbi.get_files_by_type_from_db(self.project_db_path, 'qc'),
            'experimental': dbi.get_files_by_type_from_db(self.project_db_path, 'experimental'),
            'istd': dbi.get_files_by_type_from_db(self.project_db_path, 'istd'),
            'exctrl': dbi.get_files_by_type_from_db(self.project_db_path, 'exctrl')
        }

    def _parse_config(self) -> Dict[str, Any]:
        """
        Parse atlas configuration for all workflows.
        
        Returns:
            Dict with rt_align atlas and analysis atlases info for all workflows
        """
        all_workflows = self.config.get('WORKFLOWS', {})
        
        atlas_data = {
            'rt_align_template_atlas': None,
            'analysis_atlases': []
        }
        
        # Process each workflow
        for workflow_name, workflow_config in all_workflows.items():
            #logger.info(f"Processing workflow: {workflow_name}")
            
            # Get RT alignment atlas (typically one per workflow)
            rt_align_config = workflow_config.get('RT_ALIGN', {})
            rt_align_template_atlas_uid = rt_align_config.get('ATLAS', {}).get('uid')
            rt_align_params = rt_align_config.get('PARAMS', {})
            
            if rt_align_template_atlas_uid and not atlas_data['rt_align_template_atlas']:
                # Get RT alignment atlas info (use first one found)
                try:
                    if Path(self.project_db_path).exists():
                        logger.info(f"Looking for RT alignment template atlas {rt_align_template_atlas_uid} in project database")
                        atlas_df = dbi.get_atlas_metadata_from_db(self.project_db_path, rt_align_template_atlas_uid, validation=True)
                    else:
                        atlas_df = pd.DataFrame()
                    if atlas_df.empty:
                        logger.info(f"Looking for RT alignment template atlas {rt_align_template_atlas_uid} in main database")
                        atlas_df = dbi.get_atlas_metadata_from_db(self.main_db_path, rt_align_template_atlas_uid, validation=True)
                    
                    if not atlas_df.empty:
                        atlas_info = atlas_df.iloc[0]
                        atlas_data['rt_align_template_atlas'] = {
                            'atlas_uid': rt_align_template_atlas_uid,
                            'atlas_name': atlas_info.get('atlas_name', ''),
                            'atlas_description': atlas_info.get('atlas_description', ''),
                            'chromatography': atlas_info.get('chromatography', '').lower(),
                            'polarity': atlas_info.get('polarity', '').lower(),
                            'workflow': workflow_name.lower(),
                            'rt_align_params': rt_align_params
                        }
                        #logger.info(f"RT alignment atlas: {rt_align_template_atlas_uid} (workflow: {workflow_name})")
                    else:
                        logger.error(f"RT alignment atlas {rt_align_template_atlas_uid} not found in databases")
                        
                except Exception as e:
                    logger.error(f"Error loading RT alignment atlas {rt_align_template_atlas_uid}: {e}")
            
            # Get analysis atlases from ANALYSES section
            analyses_config = workflow_config.get('ANALYSES', {})
            for atlas_type, methods in analyses_config.items():
                for polarity, config_data in methods.items():
                    analysis_atlas_uid = config_data.get('ATLAS', {}).get('uid')
                    if analysis_atlas_uid:
                        try:
                            if Path(self.project_db_path).exists():
                                logger.info(f"Looking for analysis atlas {analysis_atlas_uid} in project database")
                                atlas_df = dbi.get_atlas_metadata_from_db(self.project_db_path, analysis_atlas_uid, validation=True)
                            else:
                                atlas_df = pd.DataFrame()
                            if atlas_df.empty:
                                logger.info(f"Looking for analysis atlas {analysis_atlas_uid} in main database")
                                atlas_df = dbi.get_atlas_metadata_from_db(self.main_db_path, analysis_atlas_uid, validation=True)
                            
                            if not atlas_df.empty:
                                atlas_info = atlas_df.iloc[0]
                                analysis_atlas_data = {
                                    'atlas_uid': analysis_atlas_uid,
                                    'atlas_name': atlas_info.get('atlas_name', ''),
                                    'atlas_description': atlas_info.get('atlas_description', ''),
                                    'chromatography': atlas_info.get('chromatography', '').lower(),
                                    'polarity': polarity.lower(),
                                    'atlas_type': atlas_type.lower(),
                                    'workflow': workflow_name.lower(),
                                    'analysis_params': config_data.get('PARAMS', {})
                                }
                                atlas_data['analysis_atlases'].append(analysis_atlas_data)
                                #logger.info(f"Analysis atlas: {workflow_name}/{atlas_type}/{polarity} -> {analysis_atlas_uid}")
                            else:
                                logger.error(f"Analysis atlas {analysis_atlas_uid} not found in databases")
                                
                        except Exception as e:
                            logger.error(f"Error loading analysis atlas {analysis_atlas_uid}: {e}")
                    #else:
                        #logger.info(f"No atlas UID provided for {workflow_name}/{atlas_type}/{polarity} - skipping")
        
        return atlas_data

    def run_complete_workflow(self,
                            new_lcmsruns: bool = False, 
                            stop_at_stage: WorkflowStage = WorkflowStage.FINAL_REPORT,
                            analysis_subset: List[Tuple[str, str, str]] = None,
                            create_analysis_notebooks: bool = False) -> None:
        """
        Run the complete workflow up to the specified stage with optional caching
        
        Args:
            stop_at_stage: Which stage to stop at
            analysis_subset: List of (atlas_type, polarity) tuples to limit analysis
            create_analysis_notebooks: Whether to create individual notebooks per analysis
        """
        
        # Stage 1: Project Setup
        if self.current_stage == WorkflowStage.PROJECT_SETUP:
            logger.info("\n========== Stage 1: Project Setup ==========\n")

            # Initialize project setup directly in workflow
            self._setup_project_database(new_lcmsruns)
            self.load_h5_files()
            
            if stop_at_stage == self.current_stage.value:
                logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
                return

            self.current_stage = WorkflowStage.RT_CORRECTION
        
        # Stage 2: RT Correction
        if self.current_stage == WorkflowStage.RT_CORRECTION:
            logger.info("\n========== Stage 2: RT Correction ==========\n")

            logger.info("Starting RT correction...")
            self._run_rt_correction_workflow()

            if stop_at_stage == self.current_stage.value:
                logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
                return

            self.current_stage = WorkflowStage.PUTATIVE_IDENTIFICATION
        
        # Stage 3: Putative Identification
        if self.current_stage == WorkflowStage.PUTATIVE_IDENTIFICATION:
            logger.info("\n========== Stage 3: Putative Identification ==========\n")

            logger.info(f"Starting putative identifications...")
            compound_id_stats = self._run_putative_identification_workflow(self.config, analysis_subset)

            # Print putative identification stats in a readable notebook format
            stats_md = "### Putative Identification Summary\n"
            stats_md += f"- **Total putative IDs:** {compound_id_stats.get('total_putative_ids', 0)}\n"
            stats_md += "- **By atlas type:**\n"
            for k, v in compound_id_stats.get('by_atlas_type', {}).items():
                stats_md += f"    - {k}: {v}\n"
            stats_md += "- **By curation status:**\n"
            for k, v in compound_id_stats.get('by_curation_status', {}).items():
                stats_md += f"    - {k}: {v}\n"
            stats_md += f"- **With MS2 data:** {compound_id_stats.get('with_ms2_data', 0)}\n"
            stats_md += f"- **With reference hits:** {compound_id_stats.get('with_reference_hits', 0)}\n"
            stats_md += f"- **RT modified:** {compound_id_stats.get('rt_modified', 0)}\n"
            stats_md += f"- **Annotation modified:** {compound_id_stats.get('annotation_modified', 0)}\n"
            display(Markdown(stats_md))
            #logger.info("Putative identification stats:\n" + json.dumps(self.results.summary_stats, indent=2))

            if stop_at_stage == self.current_stage.value:
                logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
                if create_analysis_notebooks:
                    self._create_individual_curation_notebooks()
                return
            
            self.current_stage = WorkflowStage.MANUAL_CURATION

        # Stage 4: Manual Curation (returns GUI for interactive work) 
        if self.current_stage == WorkflowStage.MANUAL_CURATION:
            logger.info("\n========== Stage 4: Manual Curation ==========\n")

            if stop_at_stage == self.current_stage.value:
                logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
                if create_analysis_notebooks:
                    return self._create_individual_curation_notebooks()

            self.current_stage = WorkflowStage.FINAL_REPORT

        # Stage 5: Final Report
        if self.current_stage == WorkflowStage.FINAL_REPORT or stop_at_stage == WorkflowStage.FINAL_REPORT:
            logger.info("\n========== Stage 5: Final Report Generation ==========\n")

            # Create final report manager
            final_report = FinalReportManager(self.results.putative_ids)
            
            output_path = Path(self.analysis_directory) / "final_targeted_analysis_report"
            report_df = final_report.generate_comprehensive_report(self.config, str(output_path))
            
            logger.info(f"Workflow complete! Final report with {len(report_df)} identifications")
            #return report_df
    
    def get_workflow_status(self) -> Dict:
        """Get current workflow status and progress"""
        status = {
            'current_stage': self.current_stage.value,
            'stages_completed': [],
            'next_stage': None
        }
        
        # Determine completed stages based on data existence
        if self.h5_files:
            status['stages_completed'].append('project_setup')
        if self.results.has_rt_correction_results():
            status['stages_completed'].append('rt_correction')
        if self.results.has_putative_identification_results():
            status['stages_completed'].append('putative_identification')
        
        # Determine next stage
        stage_order = [
            WorkflowStage.PROJECT_SETUP,
            WorkflowStage.RT_CORRECTION,
            WorkflowStage.PUTATIVE_IDENTIFICATION,
            WorkflowStage.MANUAL_CURATION,
            WorkflowStage.FINAL_REPORT
        ]
        
        current_idx = stage_order.index(self.current_stage)
        if current_idx < len(stage_order) - 1:
            status['next_stage'] = stage_order[current_idx + 1].value
        
        return status

    def _create_individual_curation_notebooks(self) -> List[str]:
        """Create individual notebooks for each analysis type/polarity combination"""
        created_notebooks = []
        logger.info("Creating individual curation notebooks for each analysis...")
        # Get all analysis combinations that have putative identifications

        for analysis_key, chrom_pol in self.results.putative_ids.items():
            for chrom_pol, putative_list in chrom_pol.items():
                if putative_list:
                    logger.info(f"Creating curation notebook for {analysis_key} in {chrom_pol} mode ({len(putative_list)} compounds)")
                    notebook_path = self._create_analysis_specific_notebook(analysis_key, chrom_pol, putative_list)
                    created_notebooks.append(notebook_path)
        
        logger.info(f"Created {len(created_notebooks)} individual curation notebooks")
        return created_notebooks
    
    def _create_analysis_specific_notebook(self, atlas_type: str, chrom_pol: str, putative_list: List) -> str:
        """Create a notebook for a specific analysis type and chrom_pol"""
        
        # Copy the existing config file to the analysis directory for notebook reproducibility
        analysis_config_path = Path(self.analysis_directory) / "metatlas2_config.yaml"
        existing_config_path = self.config['ENV']['PATHS']['config_path']
        if not analysis_config_path.exists():
            shutil.copy(existing_config_path, analysis_config_path)
        
        notebook_filename = f"curation_{atlas_type}_{chrom_pol}.ipynb"
        notebook_path = Path(self.analysis_directory) / notebook_filename
        
        # Create notebook content specific to this analysis
        notebook_content = self._generate_analysis_notebook_content(atlas_type, chrom_pol, putative_list)
        
        # Write notebook file
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        logger.info(f"Created analysis-specific notebook: {notebook_path}")
        return str(notebook_path)
    
    def _generate_analysis_notebook_content(self, atlas_type: str, chrom_pol: str, putative_list: List) -> Dict:
        """Generate notebook content for a specific analysis"""
        
        cells = [
            # Title cell
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Manual Curation: {atlas_type.upper()} {chrom_pol.upper()}\n",
                    f"## Project: {self.project_name}\n\n",
                    f"This notebook contains manual curation for **{atlas_type.upper()} {chrom_pol.upper()}** analysis.\n\n",
                    f"- Total compounds: **{len(putative_list)}**\n",
                    f"- Generated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**\n\n",
                    "**Note:** This notebook uses cached workflow data and does not re-run the analysis pipeline."
                ]
            },
            
            # Setup and load workflow with forced cache usage
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import sys\n",
                    "sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')\n",
                    "import workflow_objects as wfo\n",
                    "import load_tools as ldt\n",
                    "import targeted_gui as tgi\n",
                    "\n",
                    f"# Load configuration\n",
                    f"config_path = r'{Path(self.analysis_directory) / 'metatlas2_config.yaml'}'\n",
                    "config = ldt.load_metatlas2_config(config_path)\n",
                    "\n",
                    "# Create workflow manager\n",
                    "workflow = wfo.TargetedAnalysisManager(\n",
                    "    config=config,\n",
                    f"    project_directory=r'{self.project_directory}',\n",
                    f"    project_name=r'{self.project_name}',\n",
                    f"    project_lcmsruns_path=r'{self.project_lcmsruns_path}',\n",
                    f"    rt_alignment_number={self.rt_alignment_number},\n",
                    f"    analysis_number={self.analysis_number}\n",
                    ")\n",
                    "\n",
                    "# Load cached workflow results directly (skip pipeline execution)\n",
                    "print('Loading cached workflow results...')\n",
                    "rt_cache = workflow.cache_manager.load_rt_correction()\n",
                    "putative_cache = workflow.cache_manager.load_putative_identifications()\n",
                    "curation_cache = workflow.cache_manager.load_manual_curation(prefer_partial=True)\n",
                    "\n",
                    "if rt_cache:\n",
                    "    workflow.results.rt_models = rt_cache['rt_models']\n",
                    "    workflow.results.corrected_atlases = rt_cache.get('corrected_atlases', {})\n",
                    "    print('✓ RT correction cache loaded')\n",
                    "\n",
                    "if putative_cache:\n",
                    "    workflow.results.putative_ids = putative_cache['putative_ids']\n",
                    "    workflow.results.summary_stats = putative_cache.get('summary_stats')\n",
                    "    workflow.current_stage = wfo.WorkflowStage.MANUAL_CURATION\n",
                    "    print('✓ Putative identification cache loaded')\n",
                    "\n",
                    "if curation_cache:\n",
                    "    # Update with any existing curation progress\n",
                    "    workflow.results.putative_ids = curation_cache['putative_ids']\n",
                    "    print('✓ Curation progress cache loaded')\n",
                    "\n",
                    f"print(f'Cache loading complete. Current stage: {{workflow.current_stage.value}}')"
                ]
            },
            
            # Launch curation GUI
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Launch manual curation GUI\n",
                    f"analysis_key = '{atlas_type}'\n",
                    f"chrom_pol = '{chrom_pol}'\n",
                    "\n",
                    "# Get putative IDs from cached results\n",
                    "if analysis_key in workflow.results.putative_ids and chrom_pol in workflow.results.putative_ids[analysis_key]:\n",
                    "    putative_ids = workflow.results.putative_ids[analysis_key][chrom_pol]\n",
                    "    print(f'Found {len(putative_ids)} compounds for curation')\n",
                    "    \n",
                    "    # Auto-save callback\n",
                    "    def save_progress(updated_ids):\n",
                    "        workflow.results.putative_ids[analysis_key][chrom_pol] = updated_ids\n",
                    "        timestamp = workflow.save_curation_progress(partial=True)\n",
                    "        print(f'Progress saved at {timestamp} for {len(updated_ids)} compounds')\n",
                    "    \n",
                    "    # Launch GUI directly with cached CompoundExperimental objects\n",
                    "    gui_container, updated_putative_ids = tgi.create_gui_with_compounds(\n",
                    "        putative_ids, \n",
                    "        config, \n",
                    f"        analysis_dir=r'{self.analysis_directory}',\n",
                    "        auto_save_callback=save_progress\n",
                    "    )\n",
                    "    \n",
                    "    # Display the GUI\n",
                    "    if gui_container:\n",
                    "        display(gui_container)\n",
                    "        print('\\n=== Manual Curation GUI ===')\n",
                    "        print('Use the GUI above to curate your compounds.')\n",
                    "        print('Changes are auto-saved every 30 seconds.')\n",
                    "        print('Close this notebook when curation is complete.')\n",
                    "    else:\n",
                    "        print('Failed to create GUI - check console for errors')\n",
                    "        \n",
                    "else:\n",
                    f"    print('ERROR: No putative identifications found for {atlas_type.upper()} {chrom_pol.upper()}')\n",
                    "    print('Available data:')\n",
                    "    for at, methods in workflow.results.putative_ids.items():\n",
                    "        for method, pids in methods.items():\n",
                    "            print(f'  {at} - {method}: {len(pids)} compounds')"
                ]
            }
        ]
        
        # Create notebook structure
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
        
    def _save_workflow_state(self) -> None:
        """Save workflow state to cache for notebook reconstruction"""
        
        cache_dir = Path(self.analysis_directory) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_state = {
            'config': self.config,
            'project_db_path': self.project_db_path,
            'project_directory': self.analysis_directory,
            'atlas_data': self.atlas_data,
            'current_stage': self.current_stage.value,
            'timestamp': datetime.now().isoformat()
        }
        
        state_file = cache_dir / "workflow_state.pkl"
        
        try:
            with open(state_file, 'wb') as f:
                pickle.dump(workflow_state, f)
            logger.info(f"Saved workflow state: {state_file}")
        except Exception as e:
            logger.error(f"Failed to save workflow state: {e}")
    
    # =============================================================================
    # RT CORRECTION METHODS (Stage 2)
    # =============================================================================

    def _run_rt_correction_workflow(self) -> None:
        """Run single RT correction using the RT alignment atlas"""
        
        # Check if we should use cached RT correction
        use_rt_cache = self.atlas_data['rt_align_template_atlas']['rt_align_params'].get('use_rt_correction_cache', False)

        if use_rt_cache:
            # Try to load cached RT correction data
            cached_data = self.cache_manager.load_rt_correction()
            if cached_data:
                logger.info("Using cached RT correction data")
                self.results.rt_models = cached_data['rt_models']
                self.results.corrected_atlases = cached_data.get('corrected_atlases', {})
                return
        
        logger.info("Running RT correction workflow")
        self._do_rt_correction()

        # Always save to cache when running fresh analysis
        self.cache_manager.save_rt_correction(self.results.rt_models, self.results.corrected_atlases)

    def _do_rt_correction(self) -> None:
        """Run fresh RT correction using single RT alignment atlas"""
        
        rt_align_template_atlas = self.atlas_data.get('rt_align_template_atlas')
        if not rt_align_template_atlas:
            raise ValueError("No RT alignment atlas specified in configuration")
        
        atlas_uid = rt_align_template_atlas['atlas_uid']
        chromatography = rt_align_template_atlas['chromatography']
        polarity = rt_align_template_atlas['polarity']
        
        logger.info(f"Running RT correction using atlas {atlas_uid} ({chromatography}_{polarity})")

        # Get QC files matching the RT alignment atlas method
        chrom_pol = f"{chromatography}_{polarity}"
        logger.info(f"Loading QC files for {chrom_pol}...")
        
        qc_files_df = self._filter_files_by_method(
            self.h5_files['qc'], 
            chrom_pol
        )
        
        if qc_files_df.empty:
            raise ValueError(f"No QC files found for {chrom_pol}")
        
        logger.info(f"Found {len(qc_files_df)} QC files for {chrom_pol}")

        # Extract QC matches for RT model building
        logger.info(f"Extracting QC atlas compound data from {len(qc_files_df)} QC files...")
        qc_matches = rat.extract_matches_from_qc_files(
            self.project_db_path,
            atlas_uid,
            qc_files_df,
            self.config
        )
        
        matching_stats = rat.evaluate_qc_matching_stats(qc_matches)
        # Print QC compound matching stats in a readable notebook format
        matching_md = "### QC Compound Matching Summary\n"
        for key, value in matching_stats.items():
            if isinstance(value, dict):
                matching_md += f"- **{key}:**\n"
                for subkey, subval in value.items():
                    matching_md += f"    - {subkey}: {subval}\n"
            else:
                matching_md += f"- **{key}:** {value}\n"
        display(Markdown(matching_md))
        #logger.info("QC compound matching completed:\n" + json.dumps(matching_stats, indent=2))

        # Build single RT correction model
        logger.info(f"Building RT alignment model using {len(qc_matches)} matches")
        best_model, modeling_data, compound_stats = rat.build_rt_alignment_model(qc_matches, self.config)

        logger.info(f"RT model created with R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        
        # Store the single RT model
        self.results.rt_models[chrom_pol] = best_model
        
        # Save RT model to database
        rt_alignment_uid = dbi.save_rt_alignment_model_to_db(
            atlas_uid,
            self.project_db_path,
            self.rt_alignment_number,
            best_model,
            qc_files_df,
            modeling_data.to_dict('records')
        )

        # Create RT model plot
        rat.visualize_RT_model(modeling_data, best_model, self.analysis_directory, rt_alignment_uid)

        # Apply RT correction to all analysis atlases
        self._apply_rt_correction_to_analysis_atlases(best_model, rt_alignment_uid)

    def _filter_files_by_method(self, h5_file_info: pd.DataFrame, chrom_pol: str) -> pd.DataFrame:
        """Filter files by chromatography and polarity based on filename conventions."""
        chrom_map = {
            'hilicz': ['HILICZ', 'HILIC'],
            'hilic': ['HILICZ', 'HILIC'],
            'c18': ['C18'],
        }
        pol_map = {
            'positive': ['POS', 'FPS'],
            'negative': ['NEG', 'FPS'],
        }
        chrom, pol = chrom_pol.split('_')
        chrom_tag = chrom_map.get(chrom.lower(), None)
        pol_tags = pol_map.get(pol.lower(), None)
        if not chrom_tag or not pol_tags:
            logger.warning(f"Unknown chromatography or polarity in method: {chrom_pol}")
            return pd.DataFrame()

        def matches(filename):
            parts = filename.split('_')
            if len(parts) < 10:
                logger.warning(f"Filename {filename} does not conform to expected format. Has {len(parts)} parts instead of 10.")
                return False
            chrom_match = parts[7] in chrom_tag
            pol_match = parts[9] in pol_tags
            return chrom_match and pol_match

        mask = h5_file_info['file_path'].apply(lambda f: matches(Path(f).name))
        return h5_file_info[mask]

    def _apply_rt_correction_to_analysis_atlases(self, model: Dict, rt_alignment_uid: str) -> None:
        """Apply RT correction to all analysis atlases using the single RT model"""
        
        logger.info("Applying RT correction to analysis atlases...")
        
        for analysis_atlas in self.atlas_data['analysis_atlases']:
            atlas_type = analysis_atlas['atlas_type']
            polarity = analysis_atlas['polarity']
            chromatography = analysis_atlas['chromatography']
            atlas_uid = analysis_atlas['atlas_uid']
            
            try:
                logger.info(f"Applying RT correction to {atlas_type} {chromatography} {polarity} atlas {atlas_uid}")
                
                # Apply RT correction to this atlas
                corr_atlas_df, stats = rat.apply_rt_correction_to_target(analysis_atlas, model, self.config)
                
                # Save corrected atlas to database
                corr_atlas_uid, corr_atlas_name = dbi.save_rt_corrected_atlas_to_db(
                    self.project_db_path,
                    analysis_atlas,
                    model,
                    corr_atlas_df,
                    self.analysis_number,
                    self.rt_alignment_number,
                )
                
                # Create alignment summary
                rt_align_summary = rat.create_rt_alignment_summary(
                    corr_atlas_uid, rt_alignment_uid, corr_atlas_name, stats
                )
                
                # Print RT correction summary in a readable notebook format
                rt_align_md = f"### RT Correction Summary for {atlas_type} {chromatography} {polarity}\n"
                for key, value in rt_align_summary.items():
                    if isinstance(value, dict):
                        rt_align_md += f"- **{key}:**\n"
                        for subkey, subval in value.items():
                            rt_align_md += f"    - {subkey}: {subval}\n"
                    else:
                        rt_align_md += f"- **{key}:** {value}\n"
                display(Markdown(rt_align_md))
                #logger.info("RT correction summary:\n" + json.dumps(rt_align_summary, indent=2))
                
                # Store corrected atlas UID
                self.results.set_corrected_atlas_uid(atlas_type, 'hilic', polarity, corr_atlas_uid)
                logger.info(f"Created RT-corrected {atlas_type} {polarity} atlas: {corr_atlas_uid}")
                
            except Exception as e:
                logger.error(f"Failed to apply RT correction to {atlas_type} {polarity} atlas {atlas_uid}: {e}")
                continue

    # =============================================================================
    # PUTATIVE IDENTIFICATION METHODS (Stage 3)
    # =============================================================================
    
    def _run_putative_identification_workflow(self, config: Dict, analysis_subset: List[Tuple[str, str, str]] = None) -> None:
        """Run putative identification for specified analysis atlases with optional caching"""

        logger.info(f"Running putative identifications workflow...")
        
        self.results.putative_ids = {}
        for analysis_atlas in self.atlas_data['analysis_atlases']:
            atlas_type = analysis_atlas['atlas_type']
            polarity = analysis_atlas['polarity']
            chromatography = analysis_atlas['chromatography']
            workflow_name = analysis_atlas['workflow']
            chrom_pol = f"{chromatography}_{polarity}"
            analysis_params = analysis_atlas.get('analysis_params', {})

            # Check analysis subset to process
            if analysis_subset:
                current_tuple = (workflow_name.lower(), atlas_type.lower(), polarity.lower())
                user_tuple = list((w.lower(), a.lower(), p.lower()) for (w, a, p) in analysis_subset)
                if current_tuple not in user_tuple:
                    logger.info(f"Skipping analysis for {atlas_type} with attributes {current_tuple} as not in supplied analysis subset {user_tuple}")
                    continue
            
            # Get corrected atlas UID or fallback to original
            corrected_atlas_uid = self.results.get_corrected_atlas_uid(atlas_type, chromatography, polarity)
            target_atlas_uid = corrected_atlas_uid or analysis_atlas['atlas_uid']
            
            # Get analysis parameters from config
            use_cached_data = analysis_params.get("use_id_search_cache", False)

            if use_cached_data:
                # Try to load cached putative identifications
                cached_data = self.cache_manager.load_putative_identifications()
                if cached_data:
                    logger.info("Using cached putative identification data")
                    self.results.putative_ids = cached_data['putative_ids']
                    self.results.summary_stats = cached_data.get('summary_stats')
                    return

            logger.info(f"Running putative identification for {atlas_type} {polarity} using atlas {target_atlas_uid}")
            
            try:                        
                # Run targeted analysis workflow with specific parameters
                atlas_dataframe, analysis_results = tga.run_targeted_analysis_workflow(
                    project_db_path=self.project_db_path,
                    target_atlas_uid=target_atlas_uid,
                    config=config,
                    analysis_params=analysis_params  # Pass specific analysis parameters
                )
                
                # Convert raw analysis results directly to putative identifications
                putative_list = self._convert_raw_results_to_putative_ids(
                    analysis_results, atlas_type, chrom_pol
                )
                
                # Store results using atlas_type_polarity as key
                self.results.set_putative_ids(atlas_type, chrom_pol, putative_list)
                
            except Exception as e:
                logger.error(f"Failed putative identification for {atlas_type} {polarity}: {e}")
                self.results.set_putative_ids(atlas_type, chrom_pol, [])

        # Calculate summary statistics for workflow results
        self.results.get_putative_identification_summary()
        logger.info(f"Putative identification complete. Total: {len(self.results.get_all_putative_ids())} identifications")
        
        # Always save to cache when running new analysis
        self.cache_manager.save_putative_identifications(
            self.results.putative_ids, 
            self.results.summary_stats
        )

        return self.results.summary_stats

    # =============================================================================
    # HELPER METHODS FOR PUTATIVE IDENTIFICATION CONVERSION
    # =============================================================================
    
    def _convert_raw_results_to_putative_ids(self, analysis_results: Dict, 
                                           atlas_type: str, chrom_pol: str) -> List[CompoundExperimental]:
        """Convert raw analysis results directly to list of CompoundExperimental objects"""
        compound_list = []
        
        # Extract compounds from analysis results
        compounds_data = analysis_results.get('compounds', {})
        
        for inchi_key, compound_data in compounds_data.items():
            # Extract chromatography and polarity from chrom_pol
            chrom_parts = chrom_pol.split('_')
            chromatography = chrom_parts[0] if len(chrom_parts) > 0 else ''
            polarity = chrom_parts[1] if len(chrom_parts) > 1 else ''
            
            compound_experimental = CompoundExperimental(
                compound_uid=compound_data.get('compound_uid', ''),
                inchi_key=inchi_key,
                compound_name=compound_data.get('compound_name', ''),
                formula=compound_data.get('formula', ''),
                mz=compound_data.get('mz', 0.0),
                adduct=compound_data.get('adduct', ''),
                polarity=polarity,
                chromatography=chromatography,
                mz_tolerance=compound_data.get('mz_tolerance', 5.0),
                
                # Original atlas RT data (immutable reference)
                atlas_rt_peak=compound_data.get('original_rt_peak', 0.0),
                atlas_rt_min=compound_data.get('original_rt_min', 0.0),
                atlas_rt_max=compound_data.get('original_rt_max', 0.0),
                
                # Current RT data (modifiable during analysis)
                rt_peak=compound_data.get('rt_peak', 0.0),
                rt_min=compound_data.get('rt_min', 0.0),
                rt_max=compound_data.get('rt_max', 0.0),
                
                # Analysis annotations
                ms1_notes=compound_data.get('ms1_notes', 'keep'),
                ms2_notes=compound_data.get('ms2_notes', 'no selection'),
                analyst_notes=compound_data.get('analyst_notes', ''),
                identification_notes=compound_data.get('identification_notes', ''),
                
                # Best EIC results
                best_eic_file=compound_data.get('best_eic_file', ''),
                best_eic_rt=compound_data.get('best_eic_rt', 0.0),
                best_eic_mz=compound_data.get('best_eic_mz', 0.0),
                best_eic_intensity=compound_data.get('best_eic_intensity', 0.0),
                best_eic_ppm_error=compound_data.get('best_eic_ppm_error', 0.0),
                best_eic_rt_error=compound_data.get('best_eic_rt_error', 0.0),
                
                # Average EIC results
                avg_eic_rt=compound_data.get('avg_eic_rt', 0.0),
                avg_eic_intensity=compound_data.get('avg_eic_intensity', 0.0),
                avg_eic_mz=compound_data.get('avg_eic_mz', 0.0),
                
                # Best MS2 results
                best_ms2_file=compound_data.get('best_ms2_file', ''),
                best_ms2_database=compound_data.get('best_ms2_database', ''),
                best_ms2_ref_id=compound_data.get('best_ms2_ref_id', ''),
                best_ms2_rt=compound_data.get('best_ms2_rt', 0.0),
                best_ms2_intensity=compound_data.get('best_ms2_intensity', 0.0),
                best_ms2_mz=compound_data.get('best_ms2_mz', 0.0),
                best_ms2_score=compound_data.get('best_ms2_score', 0.0),
                best_ms2_num_matches=compound_data.get('best_ms2_num_matches', 0),
                best_ms2_ref_frags=compound_data.get('best_ms2_ref_frags', 0),
                best_ms2_data_frags=compound_data.get('best_ms2_data_frags', 0),
                best_ms2_matched_fragments=compound_data.get('best_ms2_matched_fragments', []),
                best_ms2_selection_method=compound_data.get('best_ms2_selection_method', 'none'),
                
                # Average MS2 results
                avg_ms2_score=compound_data.get('avg_ms2_score', 0.0),
                
                # Detection summary
                total_files_detected=compound_data.get('total_files_detected', 0),
                ms2_files_with_data=compound_data.get('ms2_files_with_data', 0),
                
                # Raw data storage (for GUI)
                eic_data_files=compound_data.get('eic_data_files', {}),
                ms2_data_files=compound_data.get('ms2_data_files', {}),
                suggested_rt_bounds=compound_data.get('suggested_rt_bounds'),
                isomers=compound_data.get('isomers', []),
                
                # Workflow state tracking
                is_rt_modified=compound_data.get('is_rt_modified', False),
                is_annotation_modified=compound_data.get('is_annotation_modified', False)
            )
            compound_list.append(compound_experimental)
        
        return compound_list

# =============================================================================
# MANUAL CURATION METHODS (Stage 4)
# =============================================================================
    
    def save_curation_progress(self, putative_ids: List = None, partial: bool = True) -> str:
        """
        Public method to save curation progress during manual curation.
        Can be called from GUI or external scripts.
        
        Args:
            putative_ids: Optional list of updated putative IDs. If None, saves current state
            partial: Whether this is a partial save during curation
            
        Returns:
            timestamp of saved cache
        """
        if putative_ids:
            # Update results from provided putative IDs
            for pid in putative_ids:
                # Find and update the corresponding putative ID in results
                for atlas_type, methods in self.results.putative_ids.items():
                    for method, method_pids in methods.items():
                        for i, existing_pid in enumerate(method_pids):
                            if existing_pid.inchi_key == pid.inchi_key:
                                method_pids[i] = pid
                                break
        
        # Save to cache
        timestamp = self.cache_manager.save_manual_curation(
            self.results.putative_ids,
            partial_save=partial
        )
        
        return timestamp
    
    def load_curation_progress(self, timestamp: str = None) -> bool:
        """
        Load curation progress from cache.
        
        Args:
            timestamp: Specific timestamp to load, or None for latest
            
        Returns:
            True if successfully loaded, False otherwise
        """
        cached_data = self.cache_manager.load_manual_curation(timestamp=timestamp)
        
        if cached_data:
            self.results.putative_ids = cached_data['putative_ids']
            logger.info("Loaded curation progress from cache")
            return True
        
        return False
    
    def auto_save_curation(self, putative_ids: List = None):
        """
        Convenience method for auto-saving curation progress.
        This can be called from the GUI at regular intervals.
        
        Args:
            putative_ids: Current putative identifications state
        """
        if putative_ids:
            self.cache_manager.auto_save_curation_progress(putative_ids)
        else:
            # Use current workflow state
            self.cache_manager.auto_save_curation_progress(self.results.putative_ids)


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
            logger.info("Main database already exists.")

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
        for atlas_type, methods in compound_configs.items():
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

    def create_compound_db_entries(self, compound_file_paths: List[str] = None) -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Complete database setup: create database and load compounds.
        
        Args:
            compound_file_paths: Optional list of paths to compound input files
                                 If None, uses paths from config
            
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
        self.config = config
        self.main_db_path = config["ENV"]["PATHS"]["main_database"]
        self.created_atlases = []

    def load_atlas_from_database(self, atlas_uid: str, database_path: str = None) -> Atlas:
        """
        Load an existing atlas from the database and create an Atlas object.
        
        Args:
            atlas_uid: UID of the atlas to load
            database_path: Path to database (defaults to main database)
            
        Returns:
            Atlas object
        """
        if database_path is None:
            database_path = self.main_db_path
            
        logger.info(f"Loading atlas {atlas_uid} from database")
        
        # Get atlas metadata
        atlas_metadata = dbi.get_atlas_metadata_from_db(database_path, atlas_uid, validation=True)
        if atlas_metadata.empty:
            raise ValueError(f"Atlas {atlas_uid} not found in database")
        atlas_info = atlas_metadata.iloc[0]
        
        # Get atlas compounds
        atlas_compounds_df = dbi.get_atlas_compounds_from_db(database_path, atlas_uid)
        
        # Convert compounds to CompoundReference objects
        compounds_dict = {}
        for _, row in atlas_compounds_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            compounds_dict[row.get('inchi_key', compound_ref.compound_uid)] = compound_ref
        
        # Create Atlas object
        atlas_obj = Atlas(
            atlas_uid=atlas_uid,
            atlas_name=atlas_info.get('atlas_name', ''),
            atlas_description=atlas_info.get('atlas_description', ''),
            chromatography=atlas_info.get('chromatography', ''),
            polarity=atlas_info.get('polarity', ''),
            compound_references=compounds_dict,
            created_by=atlas_info.get('created_by', ''),
            last_modified=atlas_info.get('last_modified', ''),
            source_atlas_uid=atlas_info.get('source_atlas_uid')
        )
        
        # Validate the loaded atlas
        issues = atlas_obj.validate()
        if issues:
            #logger.error(f"Atlas {atlas_uid} validation issues: {issues}")
            raise ValueError(f"Atlas {atlas_uid} failed validation: {issues}")
        else:
            logger.info(f"Atlas {atlas_uid} loaded and validated successfully")
        
        return atlas_obj

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

        for atlas_type, atlas_info in atlas_configs.items():
            if atlas_info:
                for chrom, pols in atlas_info.items():
                    for pol, atlas_details in pols.items():
                        if atlas_details and atlas_details.get('path') is not None:
                            try:
                                # Extract configuration
                                atlas_file_path = atlas_details['path']
                                atlas_name = atlas_details['name']
                                atlas_description = atlas_details['desc']
                                
                                logger.info(f"Creating atlas from config: {atlas_name} ({atlas_type}/{chrom}/{pol})")

                                logger.info(f"Loading atlas data from file: {atlas_file_path}")
                                atlas_compounds_df = ldt.load_atlas_input(atlas_file_path)
                                
                                logger.info(f"Creating Atlas object: {atlas_name}")
                                atlas_obj = self.create_atlas_from_dataframe(atlas_compounds_df, atlas_name, atlas_description, atlas_type)
                                
                                logger.info(f"Saving atlas to database: {atlas_name}")
                                dbi.save_atlas_to_database(atlas_obj, self.main_db_path)

                                # Store in created atlases list
                                self.created_atlases.append({
                                    'uid': atlas_obj.atlas_uid,
                                    'name': atlas_obj.atlas_name,
                                    'type': atlas_type,
                                    'chromatography': atlas_obj.chromatography.lower(),
                                    'polarity': atlas_obj.polarity.lower(),
                                    'compound_count': len(atlas_obj.compound_references),
                                    'atlas_object': atlas_obj
                                })

                                created_atlases.append(atlas_obj)
                                logger.info(f"Successfully created atlas: {atlas_obj.atlas_name}")

                            except Exception as e:
                                logger.error(f"Failed to create atlas for {atlas_type}/{chrom}/{pol}: {e}")
                        else:
                            logger.info(f"Skipping {atlas_type}/{chrom}/{pol} - no path specified")
        
        logger.info(f"Created {len(created_atlases)} atlases from configuration file input:")
        for atlas in created_atlases:
            logger.info(f"  Atlas: {atlas.atlas_name}")
            logger.info(f"    UID: {atlas.atlas_uid}")
            logger.info(f"    Method: {atlas.chromatography}/{atlas.polarity}")
            logger.info(f"    Compounds: {len(atlas.compound_references)}")

        return created_atlases
    
    def create_atlas_from_dataframe(self, atlas_df: pd.DataFrame, 
                                        atlas_name: str, atlas_description: str, 
                                        atlas_type: str = "ref", chromatography: str = None, 
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
                logger.debug(f"Reusing existing reference for {inchi_key}")
            else:
                references_created += 1
                logger.debug(f"Will create new reference for {inchi_key}")

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
        atlas_uid = dbi._generate_uid("atlas", decorator=atlas_type.lower())
        atlas_obj = Atlas(
            atlas_uid=atlas_uid,
            atlas_name=atlas_name,
            atlas_description=atlas_description,
            chromatography=chromatography,
            polarity=polarity,
            compound_references=compound_references
        )

        # Validate the atlas
        issues = atlas_obj.validate()
        if issues:
            raise ValueError(f"Atlas {atlas_uid} failed validation: {issues}")

        logger.info(f"Created Atlas object with {len(compound_references)} compound references")
        return atlas_obj

# =============================================================================
# FINAL REPORT GENERATION (after analysis and optional manual curation)
# =============================================================================

@dataclass
class FinalReportManager:
    """
    Manages final report generation from curated CompoundExperimental objects.
    """
    putative_ids: Dict[str, Dict[str, List[CompoundExperimental]]]
    
    def generate_comprehensive_report(self, config: Dict, output_path: str = None) -> pd.DataFrame:
        """Generate final comprehensive report from all curated identifications"""
        all_putative_ids = []
        for atlas_type in self.putative_ids.values():
            for method_ids in atlas_type.values():
                all_putative_ids.extend(method_ids)
        
        if not all_putative_ids:
            logger.warning("No putative identifications found for report generation")
            return pd.DataFrame()
        
        # Build comprehensive report
        report_rows = []
        
        for idx, pid in enumerate(all_putative_ids):
            # Calculate quality scores
            msms_quality = self._calculate_msms_quality(pid)
            mz_quality = self._calculate_mz_quality(pid)
            rt_quality = self._calculate_rt_quality(pid)
            total_score = msms_quality + mz_quality + rt_quality
            msi_level = self._determine_msi_level(msms_quality, mz_quality, rt_quality)
            
            # Get curation status with fallback
            curation_status = getattr(pid, 'curation_status', 'pending')
            
            report_row = {
                'index': idx,
                'atlas_type': getattr(pid, 'atlas_type', 'unknown'),
                'chromatography_polarity': f"{pid.chromatography}_{pid.polarity}",
                'compound_name': pid.compound_name,
                'inchi_key': pid.inchi_key,
                'formula': pid.formula,
                'adduct': pid.adduct,
                'curation_status': curation_status,
                'msms_quality': msms_quality,
                'mz_quality': mz_quality,
                'rt_quality': rt_quality,
                'total_score': total_score,
                'msi_level': msi_level,
                'ms1_notes': pid.ms1_notes,
                'ms2_notes': pid.ms2_notes,
                'analyst_notes': pid.analyst_notes,
                'identification_notes': pid.identification_notes,
                'atlas_rt_peak': pid.atlas_rt_peak,
                'current_rt_peak': pid.rt_peak,
                'rt_shift': pid.rt_peak - pid.atlas_rt_peak,
                'rt_modified': pid.is_rt_modified,
                'best_eic_file': pid.best_eic_file,
                'best_eic_intensity': pid.best_eic_intensity,
                'best_eic_ppm_error': pid.best_eic_ppm_error,
                'best_eic_rt_error': pid.best_eic_rt_error,
                'best_ms2_file': pid.best_ms2_file,
                'best_ms2_database': pid.best_ms2_database,
                'best_ms2_score': pid.best_ms2_score,
                'best_ms2_num_matches': pid.best_ms2_num_matches
            }
            
            report_rows.append(report_row)
        
        # Create DataFrame
        report_df = pd.DataFrame(report_rows)
        
        # Sort by atlas type, then by chromatography/polarity, then by RT
        report_df = report_df.sort_values([
            'atlas_type', 
            'chromatography_polarity', 
            'atlas_rt_peak'
        ]).reset_index(drop=True)
        
        # Update index after sorting
        report_df['index'] = range(len(report_df))
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as both Excel and CSV
            report_df.to_excel(output_path.with_suffix('.xlsx'), index=False)
            report_df.to_csv(output_path.with_suffix('.csv'), index=False)
            
            logger.info(f"Final report saved to {output_path}")
        
        return report_df
    
    def _calculate_msms_quality(self, pid: CompoundExperimental) -> float:
        """Calculate MS/MS quality score"""
        ms2_notes = pid.ms2_notes.lower()
        if '1.0' in ms2_notes:
            return 3.0
        elif '0.5' in ms2_notes:
            return 1.5
        elif '0.0' in ms2_notes or 'no selection' in ms2_notes:
            return 0.0
        else:
            return min(3.0, pid.best_ms2_score * 3.0) if pid.best_ms2_score > 0 else 0.0
    
    def _calculate_mz_quality(self, pid: CompoundExperimental) -> float:
        """Calculate m/z quality score"""
        ppm_error = abs(pid.best_eic_ppm_error) if pid.best_eic_ppm_error else float('inf')
        
        if ppm_error <= 2.0:
            return 2.0
        elif ppm_error <= 5.0:
            return 1.5
        elif ppm_error <= 10.0:
            return 1.0
        elif ppm_error <= 20.0:
            return 0.5
        else:
            return 0.0
    
    def _calculate_rt_quality(self, pid: CompoundExperimental) -> float:
        """Calculate RT quality score"""
        rt_error = abs(pid.best_eic_rt_error) if pid.best_eic_rt_error else float('inf')
        
        if rt_error <= 0.1:
            return 2.0
        elif rt_error <= 0.2:
            return 1.5
        elif rt_error <= 0.5:
            return 1.0
        elif rt_error <= 1.0:
            return 0.5
        else:
            return 0.0
    
    def _determine_msi_level(self, msms_quality: float, mz_quality: float, rt_quality: float) -> str:
        """Determine MSI identification level"""
        total_score = msms_quality + mz_quality + rt_quality
        
        if total_score >= 6.0 and msms_quality >= 2.0:
            return "MSI Level 2"
        elif total_score >= 4.0 and mz_quality >= 1.0:
            return "MSI Level 3"
        elif total_score >= 2.0:
            return "MSI Level 4"
        else:
            return "MSI Level 5"
