from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path

@dataclass
class EICData:
    """Represents extracted ion chromatogram data for a single compound in a file."""
    inchi_key: str
    compound_uid: str
    label: str
    adduct: str
    filename: str
    file_path: str
    
    # Chromatographic data
    rt_values: np.ndarray
    mz_values: np.ndarray
    intensity_values: np.ndarray
    
    # Peak information
    rt_peak: float
    mz_peak: float
    intensity_peak: float
    
    # Atlas reference data
    atlas_rt_min: float = 0.0
    atlas_rt_max: float = 0.0
    atlas_rt_peak: float = 0.0
    atlas_mz: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties."""
        self.rt_values = np.asarray(self.rt_values)
        self.mz_values = np.asarray(self.mz_values)
        self.intensity_values = np.asarray(self.intensity_values)
    
    @property
    def ppm_error(self) -> float:
        """Calculate PPM error from atlas m/z."""
        if self.atlas_mz > 0:
            return abs(self.mz_peak - self.atlas_mz) / self.atlas_mz * 1e6
        return 0.0
    
    @property
    def rt_error(self) -> float:
        """Calculate RT error from atlas RT."""
        return self.rt_peak - self.atlas_rt_peak
    
    def is_within_rt_window(self, rt_min: float, rt_max: float) -> bool:
        """Check if peak RT is within specified window."""
        return rt_min <= self.rt_peak <= rt_max
    
    def get_intensity_in_rt_range(self, rt_min: float, rt_max: float) -> float:
        """Get maximum intensity within RT range."""
        mask = (self.rt_values >= rt_min) & (self.rt_values <= rt_max)
        if np.any(mask):
            return np.max(self.intensity_values[mask])
        return 0.0

@dataclass
class MS2Hit:
    """Represents a reference database match for an MS2 spectrum with standardized data."""
    database: str
    ref_id: str
    score: float
    num_matches: int
    
    # Reference spectrum data
    ref_name: str
    ref_precursor_mz: float
    ref_mz_values: np.ndarray
    ref_intensity_values: np.ndarray
    
    # Aligned spectra for comparison (consistent length)
    query_mz_aligned: np.ndarray
    query_intensity_aligned: np.ndarray
    ref_mz_aligned: np.ndarray
    ref_intensity_aligned: np.ndarray
    
    # Fragment matching info (standardized)
    matched_fragments: List[float] = field(default_factory=list)
    fragment_colors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert arrays to numpy and validate consistency."""
        self.ref_mz_values = np.asarray(self.ref_mz_values)
        self.ref_intensity_values = np.asarray(self.ref_intensity_values)
        self.query_mz_aligned = np.asarray(self.query_mz_aligned)
        self.query_intensity_aligned = np.asarray(self.query_intensity_aligned)
        self.ref_mz_aligned = np.asarray(self.ref_mz_aligned)
        self.ref_intensity_aligned = np.asarray(self.ref_intensity_aligned)
        
        # Validate aligned arrays have consistent lengths
        aligned_lengths = [
            len(self.query_mz_aligned),
            len(self.query_intensity_aligned), 
            len(self.ref_mz_aligned),
            len(self.ref_intensity_aligned)
        ]
        if len(set(aligned_lengths)) > 1:
            raise ValueError("All aligned arrays must have the same length")
        
        # Validate fragment colors match aligned array length
        if len(self.fragment_colors) > 0 and len(self.fragment_colors) != aligned_lengths[0]:
            raise ValueError("Fragment colors length must match aligned array length")
    
    @property
    def is_valid_hit(self) -> bool:
        """Check if this is a valid hit with meaningful data."""
        return (self.score > 0 and 
                self.num_matches > 0 and 
                len(self.ref_mz_values) > 0 and
                len(self.query_mz_aligned) > 0)
    
    @property
    def total_ref_fragments(self) -> int:
        """Total number of fragments in reference spectrum."""
        return len(self.ref_mz_values)
    
    @property
    def total_query_fragments(self) -> int:
        """Total number of fragments in query spectrum (from aligned data)."""
        return np.sum(self.query_intensity_aligned > 0)
    
    @property
    def match_ratio(self) -> float:
        """Ratio of matched fragments to total query fragments."""
        total_query = self.total_query_fragments
        return self.num_matches / max(total_query, 1)

@dataclass
class MS2Spectrum:
    """Represents a single MS2 spectrum with associated metadata and hits."""
    inchi_key: str
    compound_uid: str
    label: str
    adduct: str
    filename: str
    file_path: str
    
    # Spectrum data
    precursor_mz: float
    precursor_intensity: float
    rt: float
    mz_values: np.ndarray
    intensity_values: np.ndarray
    
    # Reference hits
    hits: List[MS2Hit] = field(default_factory=list)
    
    # Atlas reference data
    atlas_rt_min: float = 0.0
    atlas_rt_max: float = 0.0
    atlas_rt_peak: float = 0.0
    atlas_mz: float = 0.0
    
    def __post_init__(self):
        """Convert spectrum arrays to numpy."""
        self.mz_values = np.asarray(self.mz_values)
        self.intensity_values = np.asarray(self.intensity_values)
    
    @property
    def has_hits(self) -> bool:
        """Check if spectrum has any reference hits."""
        return len(self.hits) > 0
    
    @property
    def best_hit(self) -> Optional[MS2Hit]:
        """Get the hit with highest score."""
        if not self.hits:
            return None
        return max(self.hits, key=lambda h: h.score)
    
    @property
    def num_fragments(self) -> int:
        """Number of fragments in spectrum."""
        return len(self.mz_values)
    
    @property
    def max_intensity(self) -> float:
        """Maximum fragment intensity."""
        return np.max(self.intensity_values) if len(self.intensity_values) > 0 else 0.0
    
    def get_hits_by_database(self, database: str) -> List[MS2Hit]:
        """Get all hits from a specific database."""
        return [hit for hit in self.hits if hit.database == database]
    
    def get_hits_above_score(self, min_score: float) -> List[MS2Hit]:
        """Get hits above a minimum score threshold."""
        return [hit for hit in self.hits if hit.score >= min_score]

class CompoundDataCollection:
    """Collection of EIC and MS2 data for a single compound across all files."""
    
    def __init__(self, inchi_key: str):
        self.inchi_key = inchi_key
        self.eic_data: List[EICData] = []
        self.ms2_spectra: List[MS2Spectrum] = []
    
    def add_eic(self, eic: EICData):
        """Add EIC data for this compound."""
        if eic.inchi_key == self.inchi_key:
            self.eic_data.append(eic)
    
    def add_ms2_spectrum(self, spectrum: MS2Spectrum):
        """Add MS2 spectrum for this compound."""
        if spectrum.inchi_key == self.inchi_key:
            self.ms2_spectra.append(spectrum)
    
    @property
    def best_eic_by_intensity(self) -> Optional[EICData]:
        """Get EIC with highest intensity."""
        if not self.eic_data:
            return None
        return max(self.eic_data, key=lambda e: e.intensity_peak)
    
    @property
    def best_ms2_by_score(self) -> Optional[MS2Spectrum]:
        """Get MS2 spectrum with best hit score."""
        spectra_with_hits = [s for s in self.ms2_spectra if s.has_hits]
        if not spectra_with_hits:
            return None
        return max(spectra_with_hits, key=lambda s: s.best_hit.score)
    
    @property
    def best_ms2_by_intensity(self) -> Optional[MS2Spectrum]:
        """Get MS2 spectrum with highest intensity (fallback when no hits)."""
        if not self.ms2_spectra:
            return None
        return max(self.ms2_spectra, key=lambda s: s.precursor_intensity)
    
    @property
    def files_with_eic_data(self) -> List[str]:
        """Get list of files with EIC data."""
        return list(set(eic.filename for eic in self.eic_data))
    
    @property
    def files_with_ms2_data(self) -> List[str]:
        """Get list of files with MS2 data."""
        return list(set(spec.filename for spec in self.ms2_spectra))
    
    def get_eic_by_file(self, filename: str) -> List[EICData]:
        """Get EIC data for a specific file."""
        return [eic for eic in self.eic_data if eic.filename == filename]
    
    def get_ms2_by_file(self, filename: str) -> List[MS2Spectrum]:
        """Get MS2 spectra for a specific file."""
        return [spec for spec in self.ms2_spectra if spec.filename == filename]
    
    def get_spectra_with_hits(self) -> List[MS2Spectrum]:
        """Get all spectra that have reference hits."""
        return [spec for spec in self.ms2_spectra if spec.has_hits]
    
    def get_ms2_in_rt_window(self, rt_min: float, rt_max: float) -> List[MS2Spectrum]:
        """Get MS2 spectra within RT window, sorted by score/intensity."""
        in_window = [spec for spec in self.ms2_spectra if rt_min <= spec.rt <= rt_max]
        # Sort by score if hits available, otherwise by intensity
        return sorted(in_window, key=lambda s: s.best_hit.score if s.has_hits else s.precursor_intensity, reverse=True)
    
    def get_average_hit_score(self) -> float:
        """Calculate average score across all hits."""
        all_scores = []
        for spectrum in self.ms2_spectra:
            for hit in spectrum.hits:
                if hit.is_valid_hit:
                    all_scores.append(hit.score)
        return np.mean(all_scores) if all_scores else 0.0

class ProjectDataCollection:
    """Collection of all compound data for an experiment."""
    
    def __init__(self):
        self.compounds: Dict[str, CompoundDataCollection] = {}
    
    def add_eic_data(self, eic: EICData):
        """Add EIC data, creating compound collection if needed."""
        if eic.inchi_key not in self.compounds:
            self.compounds[eic.inchi_key] = CompoundDataCollection(eic.inchi_key)
        self.compounds[eic.inchi_key].add_eic(eic)
    
    def add_ms2_spectrum(self, spectrum: MS2Spectrum):
        """Add MS2 spectrum, creating compound collection if needed."""
        if spectrum.inchi_key not in self.compounds:
            self.compounds[spectrum.inchi_key] = CompoundDataCollection(spectrum.inchi_key)
        self.compounds[spectrum.inchi_key].add_ms2_spectrum(spectrum)
    
    def get_compound(self, inchi_key: str) -> Optional[CompoundDataCollection]:
        """Get compound data collection by inchi_key."""
        return self.compounds.get(inchi_key)
    
    def get_compounds_with_eic_data(self) -> List[CompoundDataCollection]:
        """Get all compounds that have EIC data."""
        return [comp for comp in self.compounds.values() if comp.eic_data]
    
    def get_compounds_with_ms2_data(self) -> List[CompoundDataCollection]:
        """Get all compounds that have MS2 data."""
        return [comp for comp in self.compounds.values() if comp.ms2_spectra]
    
    def get_compounds_with_hits(self) -> List[CompoundDataCollection]:
        """Get all compounds that have MS2 hits."""
        return [comp for comp in self.compounds.values() if comp.get_spectra_with_hits()]
    
    @property
    def total_eic_count(self) -> int:
        """Total number of EIC traces across all compounds."""
        return sum(len(comp.eic_data) for comp in self.compounds.values())
    
    @property
    def total_ms2_count(self) -> int:
        """Total number of MS2 spectra across all compounds."""
        return sum(len(comp.ms2_spectra) for comp in self.compounds.values())
    
    @property
    def compounds_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_compounds': len(self.compounds),
            'compounds_with_eic': len(self.get_compounds_with_eic_data()),
            'compounds_with_ms2': len(self.get_compounds_with_ms2_data()),
            'compounds_with_hits': len(self.get_compounds_with_hits()),
            'total_eic_traces': self.total_eic_count,
            'total_ms2_spectra': self.total_ms2_count
        }

@dataclass
class AnalystModifications:
    """Track all user modifications to compounds during GUI interaction."""
    
    # RT modifications per compound
    rt_modifications: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Annotation modifications per compound
    annotation_modifications: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # Track which compounds have been modified
    modified_compounds: set = field(default_factory=set)
    
    def update_rt_bounds(self, inchi_key: str, rt_min: float, rt_max: float, rt_peak: float):
        """Update RT bounds for a compound and mark as modified."""
        self.rt_modifications[inchi_key] = {
            'rt_min': rt_min,
            'rt_max': rt_max,
            'rt_peak': rt_peak
        }
        self.modified_compounds.add(inchi_key)
    
    def update_annotations(self, inchi_key: str, ms1_notes: str = None, ms2_notes: str = None, 
                          analyst_notes: str = None, identification_notes: str = None):
        """Update annotations for a compound and mark as modified."""
        if inchi_key not in self.annotation_modifications:
            self.annotation_modifications[inchi_key] = {}
        
        if ms1_notes is not None:
            self.annotation_modifications[inchi_key]['ms1_notes'] = ms1_notes
            self.modified_compounds.add(inchi_key)
        
        if ms2_notes is not None:
            self.annotation_modifications[inchi_key]['ms2_notes'] = ms2_notes
            self.modified_compounds.add(inchi_key)
        
        if analyst_notes is not None:
            self.annotation_modifications[inchi_key]['analyst_notes'] = analyst_notes
            self.modified_compounds.add(inchi_key)
        
        if identification_notes is not None:
            self.annotation_modifications[inchi_key]['identification_notes'] = identification_notes
            self.modified_compounds.add(inchi_key)
    
    def get_rt_bounds(self, inchi_key: str) -> Optional[Dict[str, float]]:
        """Get RT bounds for a compound, or None if not modified."""
        return self.rt_modifications.get(inchi_key)
    
    def get_annotations(self, inchi_key: str) -> Dict[str, str]:
        """Get annotations for a compound."""
        return self.annotation_modifications.get(inchi_key, {})
    
    def is_modified(self, inchi_key: str) -> bool:
        """Check if a compound has been modified."""
        return inchi_key in self.modified_compounds
    
    def reset_compound(self, inchi_key: str):
        """Reset all modifications for a compound."""
        if inchi_key in self.rt_modifications:
            del self.rt_modifications[inchi_key]
        if inchi_key in self.annotation_modifications:
            del self.annotation_modifications[inchi_key]
        self.modified_compounds.discard(inchi_key)
    
    def get_modified_compounds(self) -> List[str]:
        """Get list of all modified compound InChI keys."""
        return list(self.modified_compounds)
    
    def to_plot_data_format(self, original_metadata: Dict) -> Dict:
        """Convert to the format expected by existing functions."""
        plot_data = {}
        
        for inchi_key, compound_meta in original_metadata.items():
            # Start with original data
            plot_data[inchi_key] = {
                'original_atlas_data': compound_meta['original_atlas_data'].copy(),
                'new_atlas_data': compound_meta['original_atlas_data'].copy(),  # Start with original
                'suggested_rt_bounds_data': compound_meta.get('suggested_rt_bounds_data'),
                'eic_data': compound_meta.get('eic_data', {}),
                'best_eic': compound_meta.get('best_eic', {}),
                'avg_eic': compound_meta.get('avg_eic', {}),
                'best_ms2': compound_meta.get('best_ms2', {}),
                'avg_ms2': compound_meta.get('avg_ms2', {}),
                'ms2_data': compound_meta.get('ms2_data', {}),
                'is_modified': self.is_modified(inchi_key)
            }
            
            # Apply RT modifications
            rt_mods = self.get_rt_bounds(inchi_key)
            if rt_mods:
                plot_data[inchi_key]['new_atlas_data'].update(rt_mods)
            
            # Apply annotation modifications
            annotation_mods = self.get_annotations(inchi_key)
            if annotation_mods:
                plot_data[inchi_key]['new_atlas_data'].update(annotation_mods)
        
        return plot_data

@dataclass
class SpectrumMatch:
    """Results of comparing query and reference spectra."""
    # Similarity metrics
    similarity_score: float
    num_matched_fragments: int
    
    # Fragment counts
    total_query_fragments: int
    total_ref_fragments: int
    
    # Matched fragment information
    matched_mz_values: List[float]
    matched_colors: List[str]  # 'green' for matches, 'red' for non-matches
    
    # Original spectra (unaligned)
    query_mz: np.ndarray
    query_intensity: np.ndarray
    ref_mz: np.ndarray
    ref_intensity: np.ndarray
    
    # Aligned spectra (same length for plotting)
    aligned_mz: np.ndarray
    aligned_query_intensity: np.ndarray
    aligned_ref_intensity: np.ndarray
    
    @property
    def match_ratio(self) -> float:
        """Ratio of matched fragments to total query fragments."""
        return self.num_matched_fragments / max(self.total_query_fragments, 1)
    
    @property
    def coverage_ratio(self) -> float:
        """Ratio of matched fragments to total reference fragments."""
        return self.num_matched_fragments / max(self.total_ref_fragments, 1)