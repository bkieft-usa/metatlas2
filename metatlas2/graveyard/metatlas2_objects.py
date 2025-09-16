from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf

logger = lcf.get_logger('metatlas2_objects')

# =============================================================================
# COMPOUND REFERENCE (Immutable atlas reference data)
# =============================================================================

@dataclass
class CompoundReference:
    """
    Immutable reference data for a compound in an atlas.
    This represents the "ground truth" from the database/atlas.
    Maps directly to database compound + mz_rt_reference tables.
    """
    
    # Core identifiers (required)
    compound_uid: str
    inchi_key: str
    compound_name: str
    
    # Chemical properties
    formula: str = ""
    mz: float = 0.0
    adduct: str = ""
    polarity: str = ""
    chromatography: str = ""
    mz_tolerance: float = 5.0
    
    # RT reference data
    rt_peak: float = 0.0
    rt_min: float = 0.0
    rt_max: float = 0.0
    
    # Database references
    mz_rt_reference_uid: str = ""
    
    # Optional metadata
    confidence: str = ""
    source: str = ""
    
    @classmethod
    def from_atlas_row(cls, row: pd.Series) -> 'CompoundReference':
        """Create from atlas DataFrame row."""
        return cls(
            compound_uid=row.get('compound_uid', ''),
            inchi_key=row.get('inchi_key', ''),
            compound_name=row.get('compound_name', row.get('label', '')),
            formula=row.get('formula', ''),
            mz=row.get('mz', 0.0),
            adduct=row.get('adduct', ''),
            polarity=row.get('polarity', ''),
            chromatography=row.get('chromatography', ''),
            mz_tolerance=row.get('mz_tolerance', 5.0),
            rt_peak=row.get('rt_peak', 0.0),
            rt_min=row.get('rt_min', 0.0),
            rt_max=row.get('rt_max', 0.0),
            mz_rt_reference_uid=row.get('mz_rt_reference_uid', ''),
            confidence=row.get('confidence', ''),
            source=row.get('source', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database serialization."""
        return {
            'compound_uid': self.compound_uid,
            'inchi_key': self.inchi_key,
            'compound_name': self.compound_name,
            'formula': self.formula,
            'mz': self.mz,
            'adduct': self.adduct,
            'polarity': self.polarity,
            'chromatography': self.chromatography,
            'mz_tolerance': self.mz_tolerance,
            'rt_peak': self.rt_peak,
            'rt_min': self.rt_min,
            'rt_max': self.rt_max,
            'mz_rt_reference_uid': self.mz_rt_reference_uid,
            'confidence': self.confidence,
            'source': self.source
        }

# =============================================================================
# COMPOUND DATA (Mutable analysis data + experimental results)
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
    original_rt_peak: float = 0.0
    original_rt_min: float = 0.0
    original_rt_max: float = 0.0
    
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
    
    def __post_init__(self):
        """Initialize derived values after creation."""
        if self.rt_peak == 0.0 and self.original_rt_peak > 0.0:
            self.rt_peak = self.original_rt_peak
            self.rt_min = self.original_rt_min
            self.rt_max = self.original_rt_max
    
    @classmethod
    def from_compound_reference(cls, compound_ref: CompoundReference) -> 'CompoundExperimental':
        """Create from CompoundReference object."""
        return cls(
            compound_uid=compound_ref.compound_uid,
            inchi_key=compound_ref.inchi_key,
            compound_name=compound_ref.compound_name,
            formula=compound_ref.formula,
            mz=compound_ref.mz,
            adduct=compound_ref.adduct,
            polarity=compound_ref.polarity,
            chromatography=compound_ref.chromatography,
            mz_tolerance=compound_ref.mz_tolerance,
            original_rt_peak=compound_ref.rt_peak,
            original_rt_min=compound_ref.rt_min,
            original_rt_max=compound_ref.rt_max,
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
            original_rt_peak=atlas_row.get('rt_peak', 0.0),
            original_rt_min=atlas_row.get('rt_min', 0.0),
            original_rt_max=atlas_row.get('rt_max', 0.0),
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
                'rt_peak': self.original_rt_peak,
                'rt_min': self.original_rt_min,
                'rt_max': self.original_rt_max,
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
    
    def to_database_row(self, analysis_uid: str, project_name: str, atlas_uid: str) -> Tuple:
        """Convert to targeted_analysis table row format."""
        import load_tools as ldt
        prov = ldt.get_provenance()
        
        return (
            analysis_uid, project_name, atlas_uid, self.compound_uid, self.inchi_key, self.compound_name,
            self.original_rt_peak, self.original_rt_min, self.original_rt_max, self.mz, self.mz_tolerance, self.adduct,
            json.dumps(self.isomers) if self.isomers else None,
            self.rt_peak, self.rt_min, self.rt_max, self.is_rt_modified,
            self.best_eic_file, self.best_eic_rt, self.best_eic_mz, self.best_eic_intensity, self.best_eic_ppm_error, self.best_eic_rt_error,
            self.avg_eic_rt, self.avg_eic_intensity, self.avg_eic_mz,
            self.best_ms2_file, self.best_ms2_database, self.best_ms2_ref_id, self.best_ms2_rt, self.best_ms2_intensity, self.best_ms2_mz,
            self.best_ms2_score, self.best_ms2_num_matches, self.best_ms2_ref_frags, self.best_ms2_data_frags,
            json.dumps(self.best_ms2_matched_fragments) if self.best_ms2_matched_fragments else None,
            self.avg_ms2_score, self.total_files_detected, self.ms2_files_with_data, self.best_ms2_score, self.best_ms2_database, self.best_ms2_num_matches or 0,
            self.ms1_notes, self.ms2_notes, prov["analyst"], prov["timestamp"]
        )

# =============================================================================
# ATLAS (Collection of compound references)
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
    compounds: Dict[str, CompoundReference] = field(default_factory=dict)
    
    # Atlas metadata
    created_by: str = ""
    last_modified: str = ""
    is_rt_corrected: bool = False
    source_atlas_uid: Optional[str] = None
    
    @classmethod
    def from_database(cls, project_db_path: str, atlas_uid: str, 
                     main_db_path: str = None) -> 'Atlas':
        """Load atlas from database using existing database functions."""
        logger.info(f"Loading atlas {atlas_uid} from database...")
        
        # Get atlas metadata
        atlas_metadata_df = dbi.get_atlas_from_db(project_db_path, atlas_uid)
        if atlas_metadata_df.empty:
            raise ValueError(f"Atlas {atlas_uid} not found in database")
        
        atlas_row = atlas_metadata_df.iloc[0]
        
        # Get compounds with metadata
        atlas_compounds_df = dbi.get_atlas_compounds_with_metadata(
            project_db_path=project_db_path,
            main_db_path=main_db_path,
            atlas_uid=atlas_uid
        )
        
        if atlas_compounds_df.empty:
            logger.warning(f"No compounds found for atlas {atlas_uid}")
        
        # Create atlas object
        atlas = cls(
            atlas_uid=atlas_uid,
            atlas_name=atlas_row.get('atlas_name', ''),
            atlas_description=atlas_row.get('atlas_description', ''),
            chromatography=atlas_row.get('chromatography', ''),
            polarity=atlas_row.get('polarity', ''),
            created_by=atlas_row.get('created_by', ''),
            last_modified=atlas_row.get('last_modified', ''),
            is_rt_corrected=atlas_compounds_df.get('rt_correction_applied', False).any() if not atlas_compounds_df.empty else False
        )
        
        # Load compounds
        for _, row in atlas_compounds_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            atlas.compounds[compound_ref.inchi_key] = compound_ref
        
        logger.info(f"Loaded atlas '{atlas.atlas_name}' with {len(atlas.compounds)} compounds")
        return atlas
    
    @classmethod
    def from_dataframe(cls, atlas_df: pd.DataFrame, atlas_uid: str = None, 
                      atlas_name: str = None) -> 'Atlas':
        """Create atlas from DataFrame (for compatibility with existing code)."""
        if atlas_df.empty:
            raise ValueError("Cannot create atlas from empty DataFrame")
        
        # Extract metadata from first row
        first_row = atlas_df.iloc[0]
        
        atlas = cls(
            atlas_uid=atlas_uid or first_row.get('atlas_uid', 'unknown'),
            atlas_name=atlas_name or first_row.get('atlas_name', 'Unknown Atlas'),
            atlas_description=first_row.get('atlas_description', ''),
            chromatography=first_row.get('chromatography', ''),
            polarity=first_row.get('polarity', ''),
            created_by=first_row.get('created_by', ''),
            last_modified=first_row.get('last_modified', ''),
            is_rt_corrected=atlas_df.get('rt_correction_applied', False).any()
        )
        
        # Load compounds
        for _, row in atlas_df.iterrows():
            compound_ref = CompoundReference.from_atlas_row(row)
            atlas.compounds[compound_ref.inchi_key] = compound_ref
        
        return atlas
    
    # Access methods
    def get_compound_by_inchi_key(self, inchi_key: str) -> Optional[CompoundReference]:
        """Get compound reference by InChI key."""
        return self.compounds.get(inchi_key)
    
    def get_compound_by_uid(self, compound_uid: str) -> Optional[CompoundReference]:
        """Get compound reference by compound UID."""
        for compound in self.compounds.values():
            if compound.compound_uid == compound_uid:
                return compound
        return None
    
    # Filtering methods
    def filter_by_chromatography(self, chromatography: str) -> 'Atlas':
        """Create filtered atlas copy with only compounds matching chromatography."""
        filtered_compounds = {
            inchi_key: compound for inchi_key, compound in self.compounds.items()
            if compound.chromatography == chromatography
        }
        
        return Atlas(
            atlas_uid=f"{self.atlas_uid}_filtered_{chromatography}",
            atlas_name=f"{self.atlas_name} ({chromatography})",
            atlas_description=f"Filtered version of {self.atlas_name} for {chromatography}",
            chromatography=chromatography,
            polarity=self.polarity,
            compounds=filtered_compounds,
            created_by=self.created_by,
            last_modified=self.last_modified,
            is_rt_corrected=self.is_rt_corrected,
            source_atlas_uid=self.atlas_uid
        )
    
    def filter_by_polarity(self, polarity: str) -> 'Atlas':
        """Create filtered atlas copy with only compounds matching polarity."""
        filtered_compounds = {
            inchi_key: compound for inchi_key, compound in self.compounds.items()
            if compound.polarity == polarity
        }
        
        return Atlas(
            atlas_uid=f"{self.atlas_uid}_filtered_{polarity}",
            atlas_name=f"{self.atlas_name} ({polarity})",
            atlas_description=f"Filtered version of {self.atlas_name} for {polarity}",
            chromatography=self.chromatography,
            polarity=polarity,
            compounds=filtered_compounds,
            created_by=self.created_by,
            last_modified=self.last_modified,
            is_rt_corrected=self.is_rt_corrected,
            source_atlas_uid=self.atlas_uid
        )
    
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
        if not self.compounds:
            issues.append("No compounds in atlas")
        
        # Check for duplicate compound UIDs
        compound_uids = [c.compound_uid for c in self.compounds.values()]
        if len(compound_uids) != len(set(compound_uids)):
            issues.append("Duplicate compound UIDs found")
        
        # Check individual compounds
        for inchi_key, compound in self.compounds.items():
            if not compound.compound_uid:
                issues.append(f"Compound {inchi_key} missing compound_uid")
            if not compound.compound_name:
                issues.append(f"Compound {inchi_key} missing name")
            if compound.mz <= 0:
                issues.append(f"Compound {inchi_key} has invalid m/z: {compound.mz}")
            if compound.rt_peak <= 0:
                issues.append(f"Compound {inchi_key} has invalid RT peak: {compound.rt_peak}")
            if compound.rt_min >= compound.rt_max:
                issues.append(f"Compound {inchi_key} has invalid RT bounds: {compound.rt_min} >= {compound.rt_max}")
        
        return issues
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert atlas back to DataFrame format for compatibility."""
        if not self.compounds:
            return pd.DataFrame()
        
        rows = []
        for compound in self.compounds.values():
            row = compound.to_dict()
            # Add atlas metadata to each row
            row.update({
                'atlas_uid': self.atlas_uid,
                'atlas_name': self.atlas_name,
                'atlas_description': self.atlas_description,
                'label': compound.compound_name,  # For compatibility
                'rt_correction_applied': self.is_rt_corrected
            })
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this atlas."""
        chromatographies = set(c.chromatography for c in self.compounds.values())
        polarities = set(c.polarity for c in self.compounds.values())
        adducts = set(c.adduct for c in self.compounds.values())
        
        return {
            'atlas_uid': self.atlas_uid,
            'atlas_name': self.atlas_name,
            'total_compounds': len(self.compounds),
            'chromatographies': list(chromatographies),
            'polarities': list(polarities),
            'adducts': list(adducts),
            'is_rt_corrected': self.is_rt_corrected,
            'source_atlas_uid': self.source_atlas_uid
        }
    
    def __len__(self) -> int:
        """Return number of compounds in atlas."""
        return len(self.compounds)
    
    def __iter__(self):
        """Iterate over compounds."""
        return iter(self.compounds.values())

# =============================================================================
# ANALYSIS PROJECT (Collection of compound analysis data + atlas reference)
# =============================================================================

@dataclass
class AnalysisProject:
    """
    Analysis project managing compounds with experimental data.
    Maps to database targeted_analysis table.
    """
    
    project_db_path: str
    atlas: Atlas
    compounds: Dict[str, CompoundExperimental] = field(default_factory=dict)
    
    # Caching metadata
    _cache_metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize metadata for caching."""
        self._cache_metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'last_modified': pd.Timestamp.now().isoformat(),
            'cache_version': '2.0'
        }
    
    @property
    def atlas_uid(self) -> str:
        """Get atlas UID from contained atlas."""
        return self.atlas.atlas_uid
    
    @classmethod
    def from_atlas_dataframe(cls, project_db_path: str, atlas_df: pd.DataFrame, 
                           atlas_uid: str = None) -> 'AnalysisProject':
        """Create AnalysisProject from atlas DataFrame (for backward compatibility)."""
        atlas = Atlas.from_dataframe(atlas_df, atlas_uid)
        project = cls(project_db_path=project_db_path, atlas=atlas)
        project.load_from_atlas()
        return project
    
    @classmethod  
    def from_database(cls, project_db_path: str, atlas_uid: str, 
                     main_db_path: str = None) -> 'AnalysisProject':
        """Create AnalysisProject by loading atlas from database."""
        atlas = Atlas.from_database(project_db_path, atlas_uid, main_db_path)
        project = cls(project_db_path=project_db_path, atlas=atlas)
        project.load_from_atlas()
        return project
    
    def update_cache_metadata(self, operation: str):
        """Update cache metadata when analysis is modified."""
        self._cache_metadata['last_modified'] = pd.Timestamp.now().isoformat()
        self._cache_metadata['last_operation'] = operation
    
    def load_from_atlas(self, atlas_df: pd.DataFrame = None):
        """Load compounds from atlas. If atlas_df provided, use it for compatibility."""
        if atlas_df is not None:
            # Backward compatibility: update atlas from dataframe
            self.atlas = Atlas.from_dataframe(atlas_df, self.atlas.atlas_uid)
        
        # Load compounds from atlas
        for inchi_key, compound_ref in self.atlas.compounds.items():
            compound = CompoundExperimental.from_compound_reference(compound_ref)
            self.compounds[compound.inchi_key] = compound
        self.update_cache_metadata('loaded_from_atlas')
    
    def add_experimental_data_simple(self, experimental_data: Dict[str, Dict]):
        """Add experimental data from simplified extraction format."""
        for inchi_key, compound_experimental_data in experimental_data.items():
            if inchi_key in self.compounds:
                compound = self.compounds[inchi_key]
                
                # Add EIC data directly from simplified format
                eic_files = compound_experimental_data.get('eic_files', {})
                for filename, eic_data in eic_files.items():
                    compound.add_eic_data(filename, eic_data)
                
                # Add MS2 data using consistent per-file structure
                ms2_files = compound_experimental_data.get('ms2_files', {})
                for filename, file_data in ms2_files.items():
                    compound.add_ms2_data(filename, file_data)
        
        self.update_cache_metadata('added_experimental_data')

    def generate_plot_data(self) -> Dict:
        """Generate plot_data format for GUI compatibility."""
        return {inchi_key: compound.to_plot_data_format() 
                for inchi_key, compound in self.compounds.items()}
    
    def save_to_database(self, project_name: str, atlas_uid: str) -> str:
        """Save all results to database and return analysis_uid."""
        import load_tools as ldt
        
        analysis_uid = dbi._generate_uid('analysis')
        rows = []
        
        for compound in self.compounds.values():
            row = compound.to_database_row(analysis_uid, project_name, atlas_uid)
            rows.append(row)
        
        if rows:
            with dbi.get_db_connection(self.project_db_path) as conn:
                insert_sql = '''
                    INSERT INTO targeted_analysis (
                        analysis_uid, project_name, atlas_uid, compound_uid, inchi_key, compound_name,
                        pre_rt_peak, pre_rt_min, pre_rt_max, pre_mz, mz_tolerance, adduct, isomers,
                        post_rt_peak, post_rt_min, post_rt_max, is_rt_modified,
                        best_eic_file, best_eic_rt, best_eic_mz, best_eic_intensity, best_eic_ppm_error, best_eic_rt_error,
                        avg_eic_rt, avg_eic_intensity, avg_eic_mz,
                        best_ms2_file, best_ms2_database, best_ms2_ref_id, best_ms2_rt_peak, best_ms2_intensity_peak, best_ms2_mz_peak,
                        best_ms2_score, best_ms2_num_matches, best_ms2_ref_frags, best_ms2_data_frags, best_ms2_matched_fragments,
                        avg_ms2_score,
                        total_files_detected, ms2_files_with_data, ms2_best_score, ms2_best_database, ms2_total_matches,
                        ms1_notes, ms2_notes, analyst, analysis_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                '''
                conn.executemany(insert_sql, rows)
        
        self.update_cache_metadata('saved_to_database')
        return analysis_uid
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary statistics for this analysis."""
        compounds_with_eic = sum(1 for c in self.compounds.values() if c.eic_data_files)
        compounds_with_ms2 = sum(1 for c in self.compounds.values() 
                               if c.ms2_data_files)
        modified_compounds = sum(1 for c in self.compounds.values() 
                               if c.is_rt_modified or c.is_annotation_modified)
        
        return {
            'total_compounds': len(self.compounds),
            'compounds_with_eic': compounds_with_eic,
            'compounds_with_ms2': compounds_with_ms2,
            'modified_compounds': modified_compounds,
            'atlas_uid': self.atlas_uid,
            'project_db_path': self.project_db_path,
            'cache_metadata': self._cache_metadata.copy()
        }