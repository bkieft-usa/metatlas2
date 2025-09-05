"""
Spectrum handling utilities for consistent spectrum operations across metatlas2.

This module provides standardized functions for:
- Spectrum alignment and matching
- Fragment counting and classification
- Similarity scoring
- Data conversion between formats

Variable naming conventions:
- query_*: Always refers to experimental/measured spectrum
- ref_*: Always refers to reference/library spectrum
- aligned_*: Arrays of same length for plotting/comparison
- matched_*: Fragments that meet matching criteria
- total_*: Complete counts before filtering
"""

import numpy as np
import sys
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
from matchms import Spectrum
from matchms.similarity import CosineHungarian

sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import logging_config as lcf

logger = lcf.get_logger('ms1_ms2_analysis')

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

def align_spectra_for_comparison(query_mz: np.ndarray, query_intensity: np.ndarray,
                                ref_mz: np.ndarray, ref_intensity: np.ndarray,
                                mz_tolerance: float = 0.005, 
                                intensity_threshold: float = 100) -> SpectrumMatch:
    """
    Align query and reference spectra for comparison and scoring.
    
    Args:
        query_mz: Query spectrum m/z values
        query_intensity: Query spectrum intensities
        ref_mz: Reference spectrum m/z values  
        ref_intensity: Reference spectrum intensities
        mz_tolerance: m/z tolerance for matching peaks
        intensity_threshold: Minimum intensity for counting matches
    
    Returns:
        SpectrumMatch object with all comparison results
    """
    # Convert to numpy arrays and validate
    query_mz = np.asarray(query_mz, dtype=np.float64)
    query_intensity = np.asarray(query_intensity, dtype=np.float64)
    ref_mz = np.asarray(ref_mz, dtype=np.float64)
    ref_intensity = np.asarray(ref_intensity, dtype=np.float64)
    
    if len(query_mz) != len(query_intensity):
        raise ValueError("query_mz and query_intensity must have same length")
    if len(ref_mz) != len(ref_intensity):
        raise ValueError("ref_mz and ref_intensity must have same length")
    
    # Get all unique m/z values for alignment grid
    all_mz = np.concatenate([query_mz, ref_mz])
    unique_mz = np.unique(np.round(all_mz, 10))  # Round to avoid floating point issues
    
    # Initialize aligned arrays
    aligned_query_intensity = []
    aligned_ref_intensity = []
    matched_mz_values = []
    matched_colors = []
    num_matched_fragments = 0
    
    for mz in unique_mz:
        # Find matching peaks within tolerance
        query_matches = np.where(np.abs(query_mz - mz) <= mz_tolerance)[0]
        ref_matches = np.where(np.abs(ref_mz - mz) <= mz_tolerance)[0]
        
        # Take max intensity if multiple matches within tolerance
        query_intensity_at_mz = np.max(query_intensity[query_matches]) if len(query_matches) > 0 else 0.0
        ref_intensity_at_mz = np.max(ref_intensity[ref_matches]) if len(ref_matches) > 0 else 0.0
        
        aligned_query_intensity.append(query_intensity_at_mz)
        aligned_ref_intensity.append(ref_intensity_at_mz)
        
        # Determine if this is a match and assign color
        if query_intensity_at_mz > intensity_threshold and ref_intensity_at_mz > intensity_threshold:
            matched_colors.append('green')
            matched_mz_values.append(mz)
            num_matched_fragments += 1
        else:
            matched_colors.append('red')
    
    # Calculate similarity score using cosine similarity
    similarity_score = calculate_cosine_similarity(query_mz, query_intensity, ref_mz, ref_intensity)
    
    return SpectrumMatch(
        similarity_score=similarity_score,
        num_matched_fragments=num_matched_fragments,
        total_query_fragments=len(query_mz),
        total_ref_fragments=len(ref_mz),
        matched_mz_values=matched_mz_values,
        matched_colors=matched_colors,
        query_mz=query_mz,
        query_intensity=query_intensity,
        ref_mz=ref_mz,
        ref_intensity=ref_intensity,
        aligned_mz=unique_mz,
        aligned_query_intensity=np.array(aligned_query_intensity),
        aligned_ref_intensity=np.array(aligned_ref_intensity)
    )

def calculate_cosine_similarity(query_mz: np.ndarray, query_intensity: np.ndarray,
                               ref_mz: np.ndarray, ref_intensity: np.ndarray,
                               tolerance: float = 0.005) -> float:
    """
    Calculate cosine similarity between two spectra using matchms.
    
    Args:
        query_mz: Query spectrum m/z values
        query_intensity: Query spectrum intensities  
        ref_mz: Reference spectrum m/z values
        ref_intensity: Reference spectrum intensities
        tolerance: m/z tolerance for matching
    
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        # Filter out zero intensities and ensure arrays are positive
        query_mask = query_intensity > 0
        ref_mask = ref_intensity > 0
        
        if not np.any(query_mask) or not np.any(ref_mask):
            return 0.0
        
        query_mz_filtered = query_mz[query_mask]
        query_intensity_filtered = query_intensity[query_mask]
        ref_mz_filtered = ref_mz[ref_mask]
        ref_intensity_filtered = ref_intensity[ref_mask]
        
        # Create matchms Spectrum objects with proper metadata
        query_spectrum = Spectrum(
            mz=query_mz_filtered, 
            intensities=query_intensity_filtered,
            metadata={'precursor_mz': float(np.median(query_mz_filtered))}
        )
        ref_spectrum = Spectrum(
            mz=ref_mz_filtered, 
            intensities=ref_intensity_filtered,
            metadata={'precursor_mz': float(np.median(ref_mz_filtered))}
        )
        
        # Calculate cosine similarity - correct calling syntax
        cosine_hungarian = CosineHungarian(tolerance=tolerance)
        score = cosine_hungarian(query_spectrum, ref_spectrum)
        logger.info(f"Cosine similarity score between query and reference spectra: {score}")

        return float(score)
    
    except Exception as e:
        # If similarity calculation fails, return 0.0
        return 0.0

def convert_spectrum_match_to_ms2hit_format(match: SpectrumMatch, database: str, ref_id: str, 
                                           ref_name: str, ref_precursor_mz: float) -> Dict[str, Any]:
    """
    Convert SpectrumMatch to MS2Hit-compatible format for legacy compatibility.
    
    Args:
        match: SpectrumMatch object
        database: Name of reference database
        ref_id: Reference spectrum ID
        ref_name: Reference spectrum name
        ref_precursor_mz: Reference precursor m/z
    
    Returns:
        Dictionary compatible with MS2Hit dataclass
    """
    return {
        'database': database,
        'ref_id': ref_id,
        'score': match.similarity_score,
        'num_matches': match.num_matched_fragments,
        'ref_name': ref_name,
        'ref_precursor_mz': ref_precursor_mz,
        'ref_mz_values': match.ref_mz,
        'ref_intensity_values': match.ref_intensity,
        'query_mz_aligned': match.aligned_mz,
        'query_intensity_aligned': match.aligned_query_intensity,
        'ref_mz_aligned': match.aligned_mz,
        'ref_intensity_aligned': match.aligned_ref_intensity,
        'matched_fragments': match.matched_mz_values,
        'fragment_colors': match.matched_colors
    }

def convert_legacy_spectrum_data(spectrum_data: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert various spectrum data formats to standardized numpy arrays.
    
    Args:
        spectrum_data: Spectrum data in various formats (list, tuple, numpy array)
    
    Returns:
        Tuple of (mz_array, intensity_array)
    """
    if spectrum_data is None:
        return np.array([]), np.array([])
    
    try:
        if isinstance(spectrum_data, np.ndarray) and spectrum_data.shape[0] == 2:
            mz_vals = np.array(spectrum_data[0], dtype=np.float64)
            intensity_vals = np.array(spectrum_data[1], dtype=np.float64)
        elif isinstance(spectrum_data, (list, tuple)) and len(spectrum_data) == 2:
            mz_vals = np.array(spectrum_data[0], dtype=np.float64)
            intensity_vals = np.array(spectrum_data[1], dtype=np.float64)
        else:
            return np.array([]), np.array([])
        
        # Validate arrays have same length
        if len(mz_vals) != len(intensity_vals):
            return np.array([]), np.array([])
        
        return mz_vals, intensity_vals
    
    except (ValueError, TypeError, IndexError):
        return np.array([]), np.array([])

def find_spectrum_matches(query_mz: np.ndarray, query_intensity: np.ndarray,
                         reference_spectra: List[Dict], inchi_key: str,
                         mz_tolerance: float = 0.005,
                         intensity_threshold: float = 100) -> List[SpectrumMatch]:
    """
    Find all matching reference spectra for a query spectrum.
    
    Args:
        query_mz: Query spectrum m/z values
        query_intensity: Query spectrum intensities
        reference_spectra: List of reference spectrum dictionaries
        inchi_key: InChI key to match against
        mz_tolerance: m/z tolerance for matching
        intensity_threshold: Minimum intensity threshold
    
    Returns:
        List of SpectrumMatch objects sorted by similarity score
    """
    matches = []
    
    for ref_spec in reference_spectra:
        # Check if InChI key matches
        if ref_spec.get('inchi_key') != inchi_key:
            continue
        
        # Convert reference spectrum data
        ref_mz, ref_intensity = convert_legacy_spectrum_data(ref_spec.get('spectrum'))
        if len(ref_mz) == 0:
            continue
        
        # Align and compare spectra
        match = align_spectra_for_comparison(
            query_mz, query_intensity, ref_mz, ref_intensity,
            mz_tolerance, intensity_threshold
        )
        
        # Add reference metadata
        match.database = ref_spec.get('database', 'unknown')
        match.ref_id = str(ref_spec.get('id', ''))
        match.ref_name = ref_spec.get('name', 'Unknown')
        match.ref_precursor_mz = ref_spec.get('precursor_mz', 0.0)
        
        matches.append(match)
    
    # Sort by similarity score (highest first)
    matches.sort(key=lambda x: x.similarity_score, reverse=True)
    
    return matches

def create_empty_spectrum_match() -> SpectrumMatch:
    """Create an empty SpectrumMatch for cases with no reference data."""
    return SpectrumMatch(
        similarity_score=0.0,
        num_matched_fragments=0,
        total_query_fragments=0,
        total_ref_fragments=0,
        matched_mz_values=[],
        matched_colors=[],
        query_mz=np.array([]),
        query_intensity=np.array([]),
        ref_mz=np.array([]),
        ref_intensity=np.array([]),
        aligned_mz=np.array([]),
        aligned_query_intensity=np.array([]),
        aligned_ref_intensity=np.array([])
    )
