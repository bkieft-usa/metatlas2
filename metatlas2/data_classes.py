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
    """Represents a reference database match for an MS2 spectrum."""
    database: str
    ref_id: str
    score: float
    num_matches: int
    
    # Reference spectrum data
    ref_name: str
    ref_precursor_mz: float
    ref_mz_values: np.ndarray
    ref_intensity_values: np.ndarray
    
    # Aligned spectra for comparison
    query_mz_aligned: np.ndarray
    query_intensity_aligned: np.ndarray
    ref_mz_aligned: np.ndarray
    ref_intensity_aligned: np.ndarray
    
    # Fragment matching info
    matched_fragments: List[float] = field(default_factory=list)
    fragment_colors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Convert arrays to numpy."""
        self.ref_mz_values = np.asarray(self.ref_mz_values)
        self.ref_intensity_values = np.asarray(self.ref_intensity_values)
        self.query_mz_aligned = np.asarray(self.query_mz_aligned)
        self.query_intensity_aligned = np.asarray(self.query_intensity_aligned)
        self.ref_mz_aligned = np.asarray(self.ref_mz_aligned)
        self.ref_intensity_aligned = np.asarray(self.ref_intensity_aligned)
    
    @property
    def is_valid_hit(self) -> bool:
        """Check if this is a valid hit (has score and matches)."""
        return self.score > 0 and self.num_matches > 0

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

class ExperimentDataCollection:
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

# Convenience functions for filtering and analysis
def filter_compounds_by_rt_error(experiment_data: ExperimentDataCollection, max_rt_error: float) -> List[CompoundDataCollection]:
    """Filter compounds where best EIC has RT error within threshold."""
    filtered = []
    for compound in experiment_data.compounds.values():
        best_eic = compound.best_eic_by_intensity
        if best_eic and abs(best_eic.rt_error) <= max_rt_error:
            filtered.append(compound)
    return filtered

def filter_compounds_by_score(experiment_data: ExperimentDataCollection, min_score: float) -> List[CompoundDataCollection]:
    """Filter compounds where best MS2 hit has score above threshold."""
    filtered = []
    for compound in experiment_data.compounds.values():
        best_ms2 = compound.best_ms2_by_score
        if best_ms2 and best_ms2.best_hit and best_ms2.best_hit.score >= min_score:
            filtered.append(compound)
    return filtered

def get_summary_dataframe(experiment_data: ExperimentDataCollection) -> pd.DataFrame:
    """Create a summary DataFrame for all compounds."""
    rows = []
    for compound in experiment_data.compounds.values():
        best_eic = compound.best_eic_by_intensity
        best_ms2 = compound.best_ms2_by_score
        best_hit = best_ms2.best_hit if best_ms2 else None
        
        row = {
            'inchi_key': compound.inchi_key,
            'eic_files': len(compound.files_with_eic_data),
            'ms2_files': len(compound.files_with_ms2_data),
            'total_spectra': len(compound.ms2_spectra),
            'spectra_with_hits': len(compound.get_spectra_with_hits()),
            'best_eic_intensity': best_eic.intensity_peak if best_eic else 0,
            'best_eic_rt_error': best_eic.rt_error if best_eic else np.nan,
            'best_eic_ppm_error': best_eic.ppm_error if best_eic else np.nan,
            'best_hit_score': best_hit.score if best_hit else 0,
            'best_hit_database': best_hit.database if best_hit else '',
            'average_hit_score': compound.get_average_hit_score()
        }
        rows.append(row)
    
    return pd.DataFrame(rows)
