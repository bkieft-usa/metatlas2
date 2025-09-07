from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('targeted_gui')

@dataclass
class CompoundData:
    """Flat class representing all compound data throughout targeted analysis workflow."""
    
    # Core identifiers
    compound_uid: str
    inchi_key: str
    compound_name: str
    
    # Chemical properties (immutable from database)
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
    
    # Raw data storage (for GUI compatibility)
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
    def from_atlas_row(cls, atlas_row: pd.Series) -> 'CompoundData':
        """Create from atlas DataFrame row."""
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
        
        # Use consistent per-file structure: {filename: {ms2_entries: [], all_hits: [], best_hit: {}, best_ms2: {}}}
        best_score = 0.0
        best_hit_data = None
        
        for filename, ms2_data in self.ms2_data_files.items():
            if isinstance(ms2_data, dict):
                # Check for best hit in this file
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
            # Extract additional fields from improved hit data
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
            analysis_uid,
            project_name, 
            atlas_uid,
            self.compound_uid,
            self.inchi_key,
            self.compound_name,
            self.original_rt_peak,
            self.original_rt_min,
            self.original_rt_max,
            self.mz,
            self.mz_tolerance,
            self.adduct,
            json.dumps(self.isomers) if self.isomers else None,
            self.rt_peak,
            self.rt_min,
            self.rt_max,
            self.is_rt_modified,
            self.best_eic_file,
            self.best_eic_rt,
            self.best_eic_mz,
            self.best_eic_intensity,
            self.best_eic_ppm_error,
            self.best_eic_rt_error,
            self.avg_eic_rt,
            self.avg_eic_intensity,
            self.avg_eic_mz,
            self.best_ms2_file,
            self.best_ms2_database,
            self.best_ms2_ref_id,
            self.best_ms2_rt,
            self.best_ms2_intensity,
            self.best_ms2_mz,
            self.best_ms2_score,
            self.best_ms2_num_matches,
            self.best_ms2_ref_frags,
            self.best_ms2_data_frags,
            json.dumps(self.best_ms2_matched_fragments) if self.best_ms2_matched_fragments else None,
            self.avg_ms2_score,
            self.total_files_detected,
            self.ms2_files_with_data,
            self.best_ms2_score,
            self.best_ms2_database,
            self.best_ms2_num_matches or 0,
            self.ms1_notes,
            self.ms2_notes,
            prov["analyst"],
            prov["timestamp"]
        )

@dataclass
class ProjectAnalysis:
    """Flat class managing entire targeted analysis project."""
    
    project_db_path: str
    atlas_uid: str
    compounds: Dict[str, CompoundData] = field(default_factory=dict)
    
    # Add caching metadata
    _cache_metadata: Dict[str, Any] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Initialize metadata for caching."""
        self._cache_metadata = {
            'created_at': pd.Timestamp.now().isoformat(),
            'last_modified': pd.Timestamp.now().isoformat(),
            'cache_version': '2.0'
        }
    
    def update_cache_metadata(self, operation: str):
        """Update cache metadata when analysis is modified."""
        self._cache_metadata['last_modified'] = pd.Timestamp.now().isoformat()
        self._cache_metadata['last_operation'] = operation
    
    def load_from_atlas(self, atlas_df: pd.DataFrame):
        """Load compounds from atlas DataFrame."""
        for _, row in atlas_df.iterrows():
            compound = CompoundData.from_atlas_row(row)
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
        import database_interact as dbi
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