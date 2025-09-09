"""
Workflow objects for metatlas2 analysis pipeline.
Classes to organize and simplify common workflow steps.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path
from enum import Enum

import sys
sys.path.append('/Users/BKieft/Metabolomics/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import targeted_analysis as tga
import load_tools as ldt
import targeted_gui as tgui
import pubchem_retrieval as pcr

logger = lcf.get_logger('workflow_objects')

class WorkflowStage(Enum):
    """Enumeration of workflow stages"""
    PROJECT_SETUP = "project_setup"
    RT_CORRECTION = "rt_correction" 
    PUTATIVE_IDENTIFICATION = "putative_identification"
    MANUAL_CURATION = "manual_curation"
    FINAL_REPORT = "final_report"

# =============================================================================
# STAGE 1: PROJECT SETUP
# =============================================================================

@dataclass
class ProjectSetup:
    """
    Manages project initialization and atlas loading.
    Uses pre-existing atlas UIDs provided by upstream workflow.
    """
    project_db_path: str
    project_directory: str
    main_db_path: str
    atlas_uids: Dict[str, Dict[str, str]]  # {atlas_type: {chrom_polarity: atlas_uid}}
    
    # Atlas collections organized by type and method (populated from atlas_uids)
    qc_atlases: Dict[str, str] = field(default_factory=dict)  # {chrom_polarity: atlas_uid}
    istd_atlases: Dict[str, str] = field(default_factory=dict)  # {chrom_polarity: atlas_uid}
    ema_atlases: Dict[str, str] = field(default_factory=dict)  # {chrom_polarity: atlas_uid}
    
    # H5 files organized by type
    h5_files: Dict[str, List[str]] = field(default_factory=dict)  # {file_type: [file_paths]}
    
    def __post_init__(self):
        """Initialize atlas collections from provided atlas_uids."""
        self.qc_atlases = self.atlas_uids.get('qc', {})
        self.istd_atlases = self.atlas_uids.get('istd', {})
        self.ema_atlases = self.atlas_uids.get('ema', {})
    
    def validate_atlases_exist(self) -> Dict[str, bool]:
        """Validate that all provided atlas UIDs exist in the database."""
        validation_results = {}
        
        for atlas_type, atlases in [
            ('qc', self.qc_atlases),
            ('istd', self.istd_atlases), 
            ('ema', self.ema_atlases)
        ]:
            for chrom_pol, atlas_uid in atlases.items():
                try:
                    atlas_df = dbi.get_atlas_from_db(self.project_db_path, atlas_uid)
                    if atlas_df.empty:
                        atlas_df = dbi.get_atlas_from_db(self.main_db_path, atlas_uid)
                    
                    exists = not atlas_df.empty
                    validation_results[f"{atlas_type}_{chrom_pol}"] = exists
                    
                    if not exists:
                        logger.warning(f"Atlas {atlas_uid} for {atlas_type}_{chrom_pol} not found")
                    else:
                        logger.info(f"Validated atlas {atlas_uid} for {atlas_type}_{chrom_pol}")
                        
                except Exception as e:
                    logger.error(f"Error validating atlas {atlas_uid}: {e}")
                    validation_results[f"{atlas_type}_{chrom_pol}"] = False
        
        return validation_results
    
    def load_h5_files(self) -> None:
        """Load and categorize H5 files from project database"""
        self.h5_files = {
            'qc': dbi.get_files_by_type_from_db(self.project_db_path, 'qc')['file_path'].tolist(),
            'experimental': dbi.get_files_by_type_from_db(self.project_db_path, 'experimental')['file_path'].tolist(),
            'istd': dbi.get_files_by_type_from_db(self.project_db_path, 'istd')['file_path'].tolist(),
            'exctrl': dbi.get_files_by_type_from_db(self.project_db_path, 'exctrl')['file_path'].tolist()
        }
    
    def get_methods_coverage(self) -> Dict[str, Dict]:
        """Get summary of atlas coverage by chromatography/polarity"""
        all_methods = set()
        all_methods.update(self.qc_atlases.keys())
        all_methods.update(self.istd_atlases.keys())
        all_methods.update(self.ema_atlases.keys())
        
        coverage = {}
        for method in all_methods:
            coverage[method] = {
                'qc_atlas': self.qc_atlases.get(method),
                'istd_atlas': self.istd_atlases.get(method),
                'ema_atlas': self.ema_atlases.get(method),
                'complete': all([
                    self.qc_atlases.get(method),
                    self.istd_atlases.get(method),
                    self.ema_atlases.get(method)
                ])
            }
        
        return coverage

# =============================================================================
# STAGE 2: RT CORRECTION  
# =============================================================================

@dataclass
class RTCorrectionManager:
    """
    Manages RT correction workflow for all atlases.
    Creates RT-corrected versions of QC, ISTD, and EMA atlases.
    """
    project_setup: ProjectSetup
    
    # RT correction models per method
    rt_models: Dict[str, Dict] = field(default_factory=dict)  # {chrom_polarity: model_dict}
    
    # RT-corrected atlas UIDs
    corrected_qc_atlases: Dict[str, str] = field(default_factory=dict)
    corrected_istd_atlases: Dict[str, str] = field(default_factory=dict) 
    corrected_ema_atlases: Dict[str, str] = field(default_factory=dict)
    
    def run_rt_correction_workflow(self, config: Dict) -> None:
        """Run complete RT correction for all methods"""
        
        for chrom_pol, qc_atlas_uid in self.project_setup.qc_atlases.items():
            logger.info(f"Running RT correction for {chrom_pol}")
            
            # Load QC atlas
            qc_atlas_df = dbi.get_atlas_compounds_table(
                self.project_setup.main_db_path, 
                qc_atlas_uid
            )
            
            # Get QC files for this method
            qc_files = self._filter_files_by_method(
                self.project_setup.h5_files['qc'], 
                chrom_pol
            )
            
            if not qc_files:
                logger.warning(f"No QC files found for {chrom_pol}")
                continue
            
            # Build RT correction model
            best_model, modeling_data = rat.build_rt_correction_model(
                qc_atlas_df, qc_files, config
            )
            
            self.rt_models[chrom_pol] = best_model
            
            # Apply RT correction to all atlas types for this method
            self._apply_rt_correction_to_atlases(chrom_pol, best_model, config)
            
            # Save RT model to database
            dbi.save_rt_alignment_model_to_db(
                self.corrected_qc_atlases[chrom_pol],
                self.project_setup.project_db_path,
                best_model,
                qc_files,
                modeling_data
            )
    
    def _filter_files_by_method(self, file_paths: List[str], chrom_pol: str) -> List[str]:
        """Filter files by chromatography and polarity"""
        chrom, pol = chrom_pol.split('_')
        # Implementation depends on your file naming conventions
        # This is a placeholder - you'll need to implement based on your file structure
        return [f for f in file_paths if chrom.lower() in f.lower() and pol.lower() in f.lower()]
    
    def _apply_rt_correction_to_atlases(self, chrom_pol: str, model: Dict, config: Dict) -> None:
        """Apply RT correction to QC, ISTD, and EMA atlases for this method"""
        atlas_types = [
            ('qc', self.project_setup.qc_atlases, self.corrected_qc_atlases),
            ('istd', self.project_setup.istd_atlases, self.corrected_istd_atlases),
            ('ema', self.project_setup.ema_atlases, self.corrected_ema_atlases)
        ]
        
        for atlas_type, source_atlases, corrected_atlases in atlas_types:
            if chrom_pol in source_atlases:
                source_uid = source_atlases[chrom_pol]
                
                # Load atlas and apply correction
                atlas_df = dbi.get_atlas_compounds_table(
                    self.project_setup.main_db_path,
                    source_uid
                )
                
                atlas_info = dbi.get_atlas_from_db(
                    self.project_setup.main_db_path,
                    source_uid
                ).iloc[0]
                
                corrected_uid, stats = dbi.create_rt_corrected_atlas(
                    self.project_setup.project_db_path,
                    source_uid,
                    atlas_info,
                    model,
                    atlas_df,
                    self.project_setup.main_db_path
                )
                
                corrected_atlases[chrom_pol] = corrected_uid
                logger.info(f"Created RT-corrected {atlas_type} atlas: {corrected_uid}")

# =============================================================================
# STAGE 3: PUTATIVE IDENTIFICATION
# =============================================================================

@dataclass 
class PutativeIdentification:
    """
    Single putative identification result.
    Maps directly to targeted_analysis table.
    """
    # Core identification
    compound_uid: str
    inchi_key: str
    compound_name: str
    atlas_type: str  # 'qc', 'istd', 'ema'
    chromatography_polarity: str  # e.g., 'hilic_positive'
    
    # Reference data (from atlas)
    reference_rt_peak: float
    reference_rt_min: float  
    reference_rt_max: float
    reference_mz: float
    reference_adduct: str
    
    # Working RT bounds (modifiable)
    current_rt_peak: float
    current_rt_min: float
    current_rt_max: float
    
    # Experimental results
    best_eic_file: str = ""
    best_eic_rt: float = 0.0
    best_eic_mz: float = 0.0
    best_eic_intensity: float = 0.0
    best_eic_ppm_error: float = 0.0
    best_eic_rt_error: float = 0.0
    
    # MS2 results
    best_ms2_file: str = ""
    best_ms2_database: str = ""
    best_ms2_score: float = 0.0
    best_ms2_num_matches: int = 0
    best_ms2_matched_fragments: str = ""
    
    # Analyst annotations
    ms1_notes: str = "keep"
    ms2_notes: str = "no selection"
    analyst_notes: str = ""
    identification_notes: str = ""
    
    # Workflow tracking
    is_rt_modified: bool = False
    is_annotation_modified: bool = False
    curation_status: str = "pending"  # pending, reviewed, finalized
    
    def update_rt_bounds(self, rt_min: float, rt_max: float, rt_peak: float = None):
        """Update RT bounds with optional peak recalculation"""
        self.current_rt_min = rt_min
        self.current_rt_max = rt_max
        if rt_peak is not None:
            self.current_rt_peak = rt_peak
        else:
            self.current_rt_peak = rt_min + (rt_max - rt_min) / 2
        self.is_rt_modified = True
    
    def update_annotations(self, **kwargs):
        """Update analyst annotations"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.is_annotation_modified = True

@dataclass
class PutativeIdentificationManager:
    """
    Manages collection of putative identifications for the entire project.
    Organizes by atlas type and chromatography/polarity.
    """
    rt_correction_manager: RTCorrectionManager
    
    # Putative IDs organized by atlas type and method
    putative_ids: Dict[str, Dict[str, List[PutativeIdentification]]] = field(default_factory=dict)
    
    def run_putative_identification_workflow(self, config: Dict) -> None:
        """Run putative identification for all RT-corrected atlases"""
        
        # Initialize storage structure
        self.putative_ids = {
            'qc': {},
            'istd': {}, 
            'ema': {}
        }
        
        atlas_collections = [
            ('qc', self.rt_correction_manager.corrected_qc_atlases),
            ('istd', self.rt_correction_manager.corrected_istd_atlases),
            ('ema', self.rt_correction_manager.corrected_ema_atlases)
        ]
        
        for atlas_type, atlas_dict in atlas_collections:
            for chrom_pol, atlas_uid in atlas_dict.items():
                logger.info(f"Running putative identification for {atlas_type} - {chrom_pol}")
                
                # Run targeted analysis workflow
                atlas_df, analysis_project = tga.run_targeted_analysis_workflow(
                    self.rt_correction_manager.project_setup.project_db_path,
                    atlas_uid,
                    config
                )
                
                # Convert to PutativeIdentification objects
                putative_list = []
                for inchi_key, compound in analysis_project.compounds.items():
                    putative_id = PutativeIdentification(
                        compound_uid=compound.compound_uid,
                        inchi_key=compound.inchi_key,
                        compound_name=compound.compound_name,
                        atlas_type=atlas_type,
                        chromatography_polarity=chrom_pol,
                        reference_rt_peak=compound.original_rt_peak,
                        reference_rt_min=compound.original_rt_min,
                        reference_rt_max=compound.original_rt_max,
                        reference_mz=compound.mz,
                        reference_adduct=compound.adduct,
                        current_rt_peak=compound.rt_peak,
                        current_rt_min=compound.rt_min,
                        current_rt_max=compound.rt_max,
                        best_eic_file=compound.best_eic_file,
                        best_eic_rt=compound.best_eic_rt,
                        best_eic_mz=compound.best_eic_mz,
                        best_eic_intensity=compound.best_eic_intensity,
                        best_eic_ppm_error=compound.best_eic_ppm_error,
                        best_eic_rt_error=compound.best_eic_rt_error,
                        best_ms2_file=compound.best_ms2_file,
                        best_ms2_database=compound.best_ms2_database,
                        best_ms2_score=compound.best_ms2_score,
                        best_ms2_num_matches=compound.best_ms2_num_matches,
                        best_ms2_matched_fragments=json.dumps(compound.best_ms2_matched_fragments),
                        ms1_notes=compound.ms1_notes,
                        ms2_notes=compound.ms2_notes,
                        analyst_notes=compound.analyst_notes,
                        identification_notes=compound.identification_notes
                    )
                    putative_list.append(putative_id)
                
                self.putative_ids[atlas_type][chrom_pol] = putative_list
                
                # Save to database with clear atlas type and method tags
                analysis_uid = self._save_putative_ids_to_db(atlas_type, chrom_pol, putative_list)
                logger.info(f"Saved {len(putative_list)} putative IDs with analysis_uid: {analysis_uid}")
    
    def _save_putative_ids_to_db(self, atlas_type: str, chrom_pol: str, putative_list: List[PutativeIdentification]) -> str:
        """Save putative identifications to database with clear organization"""
        
        analysis_uid = dbi._generate_uid('analysis')
        project_name = f"{Path(self.rt_correction_manager.project_setup.project_db_path).stem}_{atlas_type}_{chrom_pol}"
        
        rows = []
        for putative_id in putative_list:
            prov = ldt.get_provenance()
            row = (
                analysis_uid, project_name, putative_id.atlas_type,
                putative_id.compound_uid, putative_id.inchi_key, putative_id.compound_name,
                putative_id.reference_rt_peak, putative_id.reference_rt_min, putative_id.reference_rt_max,
                putative_id.reference_mz, 5.0, putative_id.reference_adduct, None,  # mz_tolerance, isomers
                putative_id.current_rt_peak, putative_id.current_rt_min, putative_id.current_rt_max,
                putative_id.is_rt_modified,
                putative_id.best_eic_file, putative_id.best_eic_rt, putative_id.best_eic_mz,
                putative_id.best_eic_intensity, putative_id.best_eic_ppm_error, putative_id.best_eic_rt_error,
                0.0, 0.0, 0.0,  # avg_eic values
                putative_id.best_ms2_file, putative_id.best_ms2_database, "",  # best_ms2_ref_id
                0.0, 0.0, 0.0,  # best_ms2 rt/intensity/mz
                putative_id.best_ms2_score, putative_id.best_ms2_num_matches, 0, 0,  # ms2 fragments
                putative_id.best_ms2_matched_fragments, 0.0,  # avg_ms2_score
                1, 1 if putative_id.best_ms2_file else 0,  # file counts
                putative_id.best_ms2_score, putative_id.best_ms2_database, putative_id.best_ms2_num_matches,
                putative_id.ms1_notes, putative_id.ms2_notes, prov["analyst"], prov["timestamp"]
            )
            rows.append(row)
        
        # Save to database
        with dbi.get_db_connection(self.rt_correction_manager.project_setup.project_db_path) as conn:
            insert_sql = '''
                INSERT INTO targeted_analysis (
                    analysis_uid, project_name, atlas_uid, compound_uid, inchi_key, compound_name,
                    pre_rt_peak, pre_rt_min, pre_rt_max, pre_mz, mz_tolerance, adduct, isomers,
                    post_rt_peak, post_rt_min, post_rt_max, is_rt_modified,
                    best_eic_file, best_eic_rt, best_eic_mz, best_eic_intensity, best_eic_ppm_error, best_eic_rt_error,
                    avg_eic_rt, avg_eic_intensity, avg_eic_mz,
                    best_ms2_file, best_ms2_database, best_ms2_ref_id, best_ms2_rt_peak, best_ms2_intensity_peak, best_ms2_mz_peak,
                    best_ms2_score, best_ms2_num_matches, best_ms2_ref_frags, best_ms2_data_frags, best_ms2_matched_fragments,
                    avg_ms2_score, total_files_detected, ms2_files_with_data, ms2_best_score, ms2_best_database, ms2_total_matches,
                    ms1_notes, ms2_notes, analyst, analysis_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            conn.executemany(insert_sql, rows)
        
        return analysis_uid
    
    def get_all_putative_ids(self) -> List[PutativeIdentification]:
        """Get flat list of all putative IDs across all atlas types and methods"""
        all_ids = []
        for atlas_type in self.putative_ids.values():
            for method_ids in atlas_type.values():
                all_ids.extend(method_ids)
        return all_ids
    
    def get_putative_ids_by_type(self, atlas_type: str) -> List[PutativeIdentification]:
        """Get putative IDs for specific atlas type (qc, istd, ema)"""
        ids = []
        if atlas_type in self.putative_ids:
            for method_ids in self.putative_ids[atlas_type].values():
                ids.extend(method_ids)
        return ids
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics for putative identifications"""
        stats = {
            'total_putative_ids': len(self.get_all_putative_ids()),
            'by_atlas_type': {},
            'by_curation_status': {'pending': 0, 'reviewed': 0, 'finalized': 0},
            'with_ms2_hits': 0,
            'rt_modified': 0
        }
        
        for atlas_type, methods in self.putative_ids.items():
            type_count = sum(len(method_ids) for method_ids in methods.values())
            stats['by_atlas_type'][atlas_type] = type_count
        
        all_ids = self.get_all_putative_ids()
        for pid in all_ids:
            stats['by_curation_status'][pid.curation_status] += 1
            if pid.best_ms2_score > 0:
                stats['with_ms2_hits'] += 1
            if pid.is_rt_modified:
                stats['rt_modified'] += 1
        
        return stats

# =============================================================================
# STAGE 4: MANUAL CURATION MANAGER
# =============================================================================

# =============================================================================
# COMPOUND DATA (Mutable analysis data + experimental results)
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
            json.dumps(self.best.ms2_matched_fragments) if self.best.ms2_matched_fragments else None,
            self.avg_ms2_score, self.total_files_detected, self.ms2_files_with_data, self.best_ms2_score, self.best_ms2_database, self.best.ms2_num_matches or 0,
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

@dataclass
class ManualCurationManager:
    """
    Manages the manual curation workflow.
    Provides structured access to putative IDs for GUI and tracks curation progress.
    """
    putative_manager: PutativeIdentificationManager
    
    def create_curation_gui(self, config: Dict, atlas_type: str = None, chrom_pol: str = None):
        """Create GUI for manual curation with filtering options"""
        
        # Filter putative IDs based on parameters
        if atlas_type and chrom_pol:
            putative_list = self.putative_manager.putative_ids.get(atlas_type, {}).get(chrom_pol, [])
        elif atlas_type:
            putative_list = self.putative_manager.get_putative_ids_by_type(atlas_type)
        else:
            putative_list = self.putative_manager.get_all_putative_ids()
        
        if not putative_list:
            logger.warning("No putative identifications found for curation")
            return None
        
        # Convert to format expected by existing GUI
        # This bridges the gap between new workflow classes and existing GUI
        analysis_project = self._convert_to_analysis_project(putative_list)
        
        return tgui.create_gui(
            analysis_project, 
            config, 
            self.putative_manager.rt_correction_manager.project_setup.project_directory
        )
    
    def _convert_to_analysis_project(self, putative_list: List[PutativeIdentification]):
        """Convert putative IDs back to AnalysisProject format for GUI compatibility"""
        
        # Create mock atlas for GUI
        atlas_data = []
        compounds = {}
        
        for pid in putative_list:
            # Create CompoundExperimental object
            compound = CompoundExperimental(
                compound_uid=pid.compound_uid,
                inchi_key=pid.inchi_key,
                compound_name=pid.compound_name,
                mz=pid.reference_mz,
                adduct=pid.reference_adduct,
                original_rt_peak=pid.reference_rt_peak,
                original_rt_min=pid.reference_rt_min,
                original_rt_max=pid.reference_rt_max,
                rt_peak=pid.current_rt_peak,
                rt_min=pid.current_rt_min,
                rt_max=pid.current_rt_max,
                ms1_notes=pid.ms1_notes,
                ms2_notes=pid.ms2_notes,
                analyst_notes=pid.analyst_notes,
                identification_notes=pid.identification_notes,
                best_eic_file=pid.best_eic_file,
                best_eic_rt=pid.best_eic_rt,
                best_eic_mz=pid.best_eic_mz,
                best_eic_intensity=pid.best_eic_intensity,
                best_eic_ppm_error=pid.best_eic_ppm_error,
                best_eic_rt_error=pid.best_eic_rt_error,
                best_ms2_file=pid.best_ms2_file,
                best_ms2_database=pid.best_ms2_database,
                best_ms2_score=pid.best_ms2_score,
                best_ms2_num_matches=pid.best_ms2_num_matches,
                is_rt_modified=pid.is_rt_modified,
                is_annotation_modified=pid.is_annotation_modified
            )
            compounds[pid.inchi_key] = compound
            
            # Create atlas row
            atlas_data.append({
                'compound_uid': pid.compound_uid,
                'inchi_key': pid.inchi_key,
                'compound_name': pid.compound_name,
                'atlas_uid': f"{pid.atlas_type}_{pid.chromatography_polarity}",
                'atlas_name': f"{pid.atlas_type.upper()} Atlas - {pid.chromatography_polarity}",
                'mz': pid.reference_mz,
                'adduct': pid.reference_adduct,
                'rt_peak': pid.reference_rt_peak,
                'rt_min': pid.reference_rt_min,
                'rt_max': pid.reference_rt_max,
                'chromatography': pid.chromatography_polarity.split('_')[0],
                'polarity': pid.chromatography_polarity.split('_')[1]
            })
        
        # Create mock atlas
        atlas_df = pd.DataFrame(atlas_data)
        atlas = Atlas.from_dataframe(atlas_df, atlas_df['atlas_uid'].iloc[0] if not atlas_df.empty else 'combined')
        
        # Create AnalysisProject
        analysis_project = AnalysisProject(
            project_db_path=self.putative_manager.rt_correction_manager.project_setup.project_db_path,
            atlas=atlas,
            compounds=compounds
        )
        
        return analysis_project
    
    def update_curation_status(self, inchi_key: str, new_status: str) -> None:
        """Update curation status for a specific compound"""
        all_ids = self.putative_manager.get_all_putative_ids()
        for pid in all_ids:
            if pid.inchi_key == inchi_key:
                pid.curation_status = new_status
                break
    
    def get_curation_progress(self) -> Dict:
        """Get curation progress summary"""
        stats = self.putative_manager.get_summary_stats()
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

# =============================================================================
# STAGE 5: FINAL REPORT GENERATION
# =============================================================================

@dataclass
class FinalReportManager:
    """
    Manages final report generation from curated putative identifications.
    """
    curation_manager: ManualCurationManager
    
    def generate_comprehensive_report(self, config: Dict, output_path: str = None) -> pd.DataFrame:
        """Generate final comprehensive report from all curated identifications"""
        all_putative_ids = self.curation_manager.putative_manager.get_all_putative_ids()
        
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
            
            report_row = {
                'index': idx,
                'atlas_type': pid.atlas_type,
                'chromatography_polarity': pid.chromatography_polarity,
                'compound_name': pid.compound_name,
                'inchi_key': pid.inchi_key,
                'formula': '',  # Could be populated from compounds table
                'adduct': pid.reference_adduct,
                'curation_status': pid.curation_status,
                'msms_quality': msms_quality,
                'mz_quality': mz_quality,
                'rt_quality': rt_quality,
                'total_score': total_score,
                'msi_level': msi_level,
                'ms1_notes': pid.ms1_notes,
                'ms2_notes': pid.ms2_notes,
                'analyst_notes': pid.analyst_notes,
                'identification_notes': pid.identification_notes,
                'reference_rt_peak': pid.reference_rt_peak,
                'current_rt_peak': pid.current_rt_peak,
                'rt_shift': pid.current_rt_peak - pid.reference_rt_peak,
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
            'reference_rt_peak'
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
    
    def _calculate_msms_quality(self, pid: PutativeIdentification) -> float:
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
    
    def _calculate_mz_quality(self, pid: PutativeIdentification) -> float:
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
    
    def _calculate_rt_quality(self, pid: PutativeIdentification) -> float:
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

# =============================================================================
# MAIN WORKFLOW ORCHESTRATOR
# =============================================================================

@dataclass
class TargetedMetabolomicsWorkflow:
    """
    Main workflow orchestrator that manages the complete targeted metabolomics workflow.
    """
    config: Dict
    project_db_path: str
    project_directory: str
    main_db_path: str
    atlas_uids: Dict[str, Dict[str, str]]  # {atlas_type: {chrom_polarity: atlas_uid}}
    
    # Workflow stage managers
    project_setup: Optional[ProjectSetup] = None
    rt_correction: Optional[RTCorrectionManager] = None
    putative_identification: Optional[PutativeIdentificationManager] = None
    manual_curation: Optional[ManualCurationManager] = None
    final_report: Optional[FinalReportManager] = None
    
    current_stage: WorkflowStage = WorkflowStage.PROJECT_SETUP
    
    def run_complete_workflow(self, stop_at_stage: WorkflowStage = WorkflowStage.FINAL_REPORT) -> None:
        """Run the complete workflow up to the specified stage"""
        
        # Stage 1: Project Setup
        if self.current_stage == WorkflowStage.PROJECT_SETUP:
            logger.info("=== Stage 1: Project Setup ===")
            self.project_setup = ProjectSetup(
                self.project_db_path, 
                self.project_directory, 
                self.main_db_path,
                self.atlas_uids
            )
            
            # Validate that all provided atlas UIDs exist
            validation_results = self.project_setup.validate_atlases_exist()
            failed_validations = [k for k, v in validation_results.items() if not v]
            if failed_validations:
                raise ValueError(f"Atlas validation failed for: {failed_validations}")
            
            self.project_setup.load_h5_files()
            
            coverage = self.project_setup.get_methods_coverage()
            logger.info(f"Atlas coverage: {coverage}")
            
            self.current_stage = WorkflowStage.RT_CORRECTION
            
            if stop_at_stage == WorkflowStage.PROJECT_SETUP:
                return
        
        # Stage 2: RT Correction
        if self.current_stage == WorkflowStage.RT_CORRECTION:
            logger.info("=== Stage 2: RT Correction ===")
            self.rt_correction = RTCorrectionManager(self.project_setup)
            self.rt_correction.run_rt_correction_workflow(self.config)
            
            self.current_stage = WorkflowStage.PUTATIVE_IDENTIFICATION
            
            if stop_at_stage == WorkflowStage.RT_CORRECTION:
                return
        
        # Stage 3: Putative Identification
        if self.current_stage == WorkflowStage.PUTATIVE_IDENTIFICATION:
            logger.info("=== Stage 3: Putative Identification ===")
            self.putative_identification = PutativeIdentificationManager(self.rt_correction)
            self.putative_identification.run_putative_identification_workflow(self.config)
            
            stats = self.putative_identification.get_summary_stats()
            logger.info(f"Putative identification stats: {stats}")
            
            self.current_stage = WorkflowStage.MANUAL_CURATION
            
            if stop_at_stage == WorkflowStage.PUTATIVE_IDENTIFICATION:
                return
        
        # Stage 4: Manual Curation (returns GUI for interactive work)
        if self.current_stage == WorkflowStage.MANUAL_CURATION:
            logger.info("=== Stage 4: Manual Curation ===")
            self.manual_curation = ManualCurationManager(self.putative_identification)
            
            # This stage requires manual interaction - return GUI
            if stop_at_stage == WorkflowStage.MANUAL_CURATION:
                return self.manual_curation.create_curation_gui(self.config)
        
        # Stage 5: Final Report
        if self.current_stage == WorkflowStage.FINAL_REPORT or stop_at_stage == WorkflowStage.FINAL_REPORT:
            logger.info("=== Stage 5: Final Report Generation ===")
            if not self.manual_curation:
                self.manual_curation = ManualCurationManager(self.putative_identification)
            
            self.final_report = FinalReportManager(self.manual_curation)
            
            output_path = Path(self.project_directory) / "final_targeted_analysis_report"
            report_df = self.final_report.generate_comprehensive_report(self.config, str(output_path))
            
            logger.info(f"Workflow complete! Final report with {len(report_df)} identifications")
            return report_df
    
    def continue_to_final_report(self) -> pd.DataFrame:
        """Continue workflow to final report generation after manual curation"""
        self.current_stage = WorkflowStage.FINAL_REPORT
        return self.run_complete_workflow(WorkflowStage.FINAL_REPORT)
    
    def get_workflow_status(self) -> Dict:
        """Get current workflow status and progress"""
        status = {
            'current_stage': self.current_stage.value,
            'stages_completed': [],
            'next_stage': None
        }
        
        # Determine completed stages
        if self.project_setup:
            status['stages_completed'].append('project_setup')
        if self.rt_correction:
            status['stages_completed'].append('rt_correction')
        if self.putative_identification:
            status['stages_completed'].append('putative_identification')
        if self.manual_curation:
            status['stages_completed'].append('manual_curation')
        if self.final_report:
            status['stages_completed'].append('final_report')
        
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

# =============================================================================
# DATABASE AND ATLAS MANAGERS
# =============================================================================

class DatabaseManager:
    """
    Manages main database creation and compound loading operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DatabaseManager with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.main_db_path = config["paths"]["main_database"]
        
    def create_main_database(self, overwrite: bool = False) -> None:
        """
        Create the main metatlas database.
        
        Args:
            overwrite: Whether to overwrite existing database
        """
        if overwrite:
            self.config["database_options"]["overwrite_existing_main_db"] = True
        
        logger.info("Creating main metatlas database...")
        dbi.create_metatlas_database(self.config)
        
    def load_compounds_from_files(self, compound_file_paths: List[str], 
                                 retrieve_pubchem: bool = True) -> None:
        """
        Load compounds from multiple input files into the main database.
        
        Args:
            compound_file_paths: List of paths to compound input files
            retrieve_pubchem: Whether to retrieve PubChem information
        """
        logger.info(f"Loading compounds from {len(compound_file_paths)} files...")
        
        for file_path in compound_file_paths:
            logger.info(f"Processing file: {Path(file_path).name}")
            
            # Load compound data
            compounds_df = ldt.load_compound_input(file_path)
            
            # Retrieve PubChem information if requested
            if retrieve_pubchem:
                pcr.retrieve_pubchem_info(compounds_df, self.config)
            
            # Add compounds to database
            dbi.add_compounds_to_db(compounds_df, self.config, file_path)
        
        logger.info("Compound loading complete!")
        
    def validate_database(self) -> None:
        """Validate the main database structure and contents."""
        dbi.validate_database(self.config)
        
    def setup_database_with_compounds(self, compound_file_paths: List[str], 
                                    overwrite: bool = False,
                                    retrieve_pubchem: bool = True) -> None:
        """
        Complete database setup: create database and load compounds.
        
        Args:
            compound_file_paths: List of paths to compound input files
            overwrite: Whether to overwrite existing database
            retrieve_pubchem: Whether to retrieve PubChem information
        """
        # Create main database
        self.create_main_database(overwrite=overwrite)
        
        # Load compounds
        self.load_compounds_from_files(compound_file_paths, retrieve_pubchem)
        
        # Validate
        self.validate_database()

class AtlasManager:
    """
    Manages atlas creation from compound input files.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AtlasManager with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.created_atlases = []
        
    def create_atlas_from_file(self, atlas_file_path: str, atlas_name: str, 
                              atlas_description: str, atlas_type: str) -> tuple:
        """
        Create an atlas from a compound input file.
        
        
        Args:
            atlas_file_path: Path to atlas input file
            atlas_name: Name for the atlas
            atlas_description: Description for the atlas
            atlas_type: Type of atlas (e.g., 'QC', 'ISTD')
            
        Returns:
            Tuple of (atlas_uid, atlas_name)
        """
        logger.info(f"Creating atlas: {atlas_name}")
        
        # Load atlas compounds
        atlas_compounds_df = ldt.load_atlas_input(atlas_file_path)
        
        # Detect chromatography and polarity
        chromatography = ldt.detect_atlas_input_chromatography(atlas_compounds_df)
        polarity = ldt.detect_atlas_input_polarity(atlas_compounds_df)
        
        # Create atlas
        atlas_uid, created_name = dbi.create_atlas_from_compounds(
            atlas_compounds_df, atlas_name, atlas_description, atlas_type,
            chromatography, polarity, self.config
        )
        
        # Store created atlas info
        atlas_info = {
            'uid': atlas_uid,
            'name': created_name,
            'type': atlas_type,
            'chromatography': chromatography.lower(),
            'polarity': polarity.lower(),
            'file_path': atlas_file_path
        }
        self.created_atlases.append(atlas_info)
        
        return atlas_info
        
    def create_multiple_atlases(self) -> List[tuple]:
        """
        Create multiple atlases from configuration list.
        
        Args:
            config: Configuration dictionary with atlas file info (self)
                          
        Returns:
            List of (atlas_uid, atlas_name) tuples
        """
        created_atlases = {}
        atlas_configs = self.config.get('atlases', {})

        for atlas_type, atlas_info in atlas_configs.items():
            new_atlas_data = self.create_atlas_from_file(
                atlas_info['atlas_table_path'],
                atlas_info['atlas_name'],
                atlas_info['atlas_description'],
                atlas_type
            )

            created_atlases[new_atlas_data['uid']] = new_atlas_data
            
        return created_atlases
        
