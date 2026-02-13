from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from enum import Enum
from datetime import datetime
import sys
import json
import numpy as np
import pandas as pd
import shutil

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sys.path.append('/global/homes/b/bkieft/metatlas2/metatlas2')
import database_interact as dbi
import logging_config as lcf
import rt_align_tools as rat
import targeted_analysis as tga
import load_tools as ldt
import pubchem_retrieval as pcr
from IPython.display import display, Markdown

logger = lcf.get_logger('workflow_objects')

class WorkflowStage(Enum):
    """Enumeration of workflow stages"""
    PROJECT_SETUP = "project_setup"
    RT_ALIGNMENT = "rt_alignment" 
    AUTO_IDENTIFICATION = "auto_identification"
    MANUAL_CURATION = "manual_curation"
    FINAL_REPORT = "final_report"

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
# ATLAS (Collection of compounds)
# =============================================================================

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
    
    # Compound references (immutable reference data)
    compound_references: Dict[str, CompoundReference] = field(default_factory=dict)
    
    # Atlas metadata
    created_by: str = ""
    created_date: str = ""
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
# MAIN WORKFLOW ORCHESTRATOR (controls all steps of analysis)
# =============================================================================

# @dataclass
# class TargetedAnalysisManager:
#     """
#     Main workflow orchestrator that manages the complete targeted metabolomics workflow.
#     """
#     config: Dict
#     project_name: str
#     rt_alignment_number: int = 1
#     analysis_number: int = 1
#     current_stage: WorkflowStage = WorkflowStage.PROJECT_SETUP
#     results: dict = field(default_factory=dict)
#     rt_models: Dict[str, Dict] = field(default_factory=dict)

#     # Parquet files by type
#     parquet_files: Dict[str, pd.DataFrame] = field(default_factory=dict)
    
#     # Compute fields
#     main_db_path: str = field(init=False)
#     atlas_data: Dict[str, Any] = field(init=False)

#     def __post_init__(self):

#         logger.info("Setting up Targeted Analysis Manager to run workflow and store results")
#         # Use single project database with iteration tracking
#         self.raw_data_directory = str(Path(self.config['ENV']['PATHS']['raw_data_dir']) / str(self.project_name))
#         self.project_directory = str(Path(self.config['ENV']['PATHS']['projects_dir']) / str(self.project_name))
#         self.rt_alignment_directory = str(Path(self.project_directory) / f"{self.project_name}_RTA{self.rt_alignment_number}")
#         self.analysis_directory = str(Path(self.rt_alignment_directory) / f"{self.project_name}_RTA{self.rt_alignment_number}_ALY{self.analysis_number}")
#         self.project_db_path = str(Path(self.project_directory) / f"{self.project_name}.duckdb")
#         self.main_db_path = self.config["ENV"]["PATHS"]["main_database"]
#         logger.info("Completed Targeted Analysis Manager setup")
    
#     def _setup_project_database(self, new_lcmsruns: bool = False) -> None:
#         """Create project database and load LCMS run files."""
#         logger.info(f"Creating project database at {self.project_db_path}...")
#         dbi.create_project_database(self.project_db_path, self.rt_alignment_number, self.analysis_number)
        
#         logger.info("Parsing analysis configuration file to get atlas and compound info...")
#         self.atlas_data = self._parse_config()

#         logger.info(f"Loading LCMS runs for {self.project_name}...")
#         try:
#             _ = dbi.save_lcmsruns_to_db(
#                 self.project_db_path,
#                 self.project_name, 
#                 self.raw_data_directory,
#                 new_lcmsruns,
#             )
#             logger.info("LCMS runs loaded successfully")
#         except Exception as e:
#             logger.warning(f"Failed to load LCMS runs: {e}")
#             logger.info("Continuing with workflow - files may need to be loaded manually")
    
#     def load_atlases_for_workflow(self) -> Dict[str, Atlas]:
#         """
#         Load all atlases required for the workflow from database.
#         This demonstrates the schema: Atlas -> Compound -> CompoundReference
        
#         Returns:
#             Dict mapping atlas_uid to Atlas objects
#         """
#         atlases = {}
#         atlas_manager = AtlasManager(self.config)
        
#         # Load RT alignment atlas (QC atlas)
#         rt_align_template_atlas_data = self.atlas_data.get('rt_align_template_atlas')
#         if rt_align_template_atlas_data:
#             atlas_uid = rt_align_template_atlas_data['atlas_uid']
#             atlas = atlas_manager.load_atlas_from_database(atlas_uid, self.main_db_path)
#             atlases[atlas_uid] = atlas
#             logger.info(f"Loaded RT alignment atlas: {atlas.atlas_name} with {len(atlas.compound_references)} compound references")
        
#         # Load analysis atlases (target atlases)
#         for analysis_atlas_data in self.atlas_data.get('analysis_atlases', []):
#             atlas_uid = analysis_atlas_data['atlas_uid']
#             atlas = atlas_manager.load_atlas_from_database(atlas_uid, self.main_db_path)
#             atlases[atlas_uid] = atlas
#             logger.info(f"Loaded analysis atlas: {atlas.atlas_name} with {len(atlas.compound_references)} compound references")
        
#         return atlases

#     def load_parquet_files(self) -> None:
#         """Load and categorize parquet file dataframes of info from project database"""
#         self.parquet_files = {
#             'qc': dbi.get_files_by_type_from_db(self.project_db_path, ['qc'], 'parquet'),
#             'experimental': dbi.get_files_by_type_from_db(self.project_db_path, ['experimental'], 'parquet'),
#             'istd': dbi.get_files_by_type_from_db(self.project_db_path, ['istd'], 'parquet'),
#             'exctrl': dbi.get_files_by_type_from_db(self.project_db_path, ['exctrl'], 'parquet')
#         }
#         logger.debug(f"Loaded parquet files: { {k: len(df) for k, df in self.parquet_files.items()} }")

#     def _parse_config(self) -> Dict[str, Any]:
#         """
#         Parse atlas configuration for all workflows.
        
#         Returns:
#             Dict with rt_align atlas and analysis atlases info for all workflows
#         """
#         all_workflows = self.config.get('WORKFLOWS', {})
        
#         atlas_data = {
#             'rt_align_template_atlas': None,
#             'analysis_atlases': []
#         }
        
#         # Process each workflow
#         for workflow_name, workflow_config in all_workflows.items():
#             #logger.info(f"Processing workflow: {workflow_name}")
            
#             # Get RT alignment atlas (typically one per workflow)
#             rt_align_config = workflow_config.get('RT_ALIGN', {})
#             rt_align_template_atlas_uid = rt_align_config.get('ATLAS', {}).get('uid')
#             rt_align_params = rt_align_config.get('PARAMS', {})
            
#             if rt_align_template_atlas_uid and not atlas_data['rt_align_template_atlas']:
#                 # Get RT alignment atlas info (use first one found)
#                 try:
#                     if Path(self.project_db_path).exists():
#                         logger.info(f"Looking for RT alignment template atlas {rt_align_template_atlas_uid} in project database")
#                         atlas_df = dbi.get_atlas_metadata_from_db(self.project_db_path, rt_align_template_atlas_uid, validation=True)
#                     else:
#                         atlas_df = pd.DataFrame()
#                     if atlas_df.empty:
#                         logger.info(f"Looking for RT alignment template atlas {rt_align_template_atlas_uid} in main database")
#                         atlas_df = dbi.get_atlas_metadata_from_db(self.main_db_path, rt_align_template_atlas_uid, validation=True)
                    
#                     if not atlas_df.empty:
#                         atlas_info = atlas_df.iloc[0]
#                         atlas_data['rt_align_template_atlas'] = {
#                             'atlas_uid': rt_align_template_atlas_uid,
#                             'atlas_name': atlas_info.get('atlas_name', ''),
#                             'atlas_description': atlas_info.get('atlas_description', ''),
#                             'chromatography': atlas_info.get('chromatography', '').lower(),
#                             'polarity': atlas_info.get('polarity', '').lower(),
#                             'workflow': workflow_name.lower(),
#                             'rt_align_params': rt_align_params
#                         }
#                         #logger.info(f"RT alignment atlas: {rt_align_template_atlas_uid} (workflow: {workflow_name})")
#                     else:
#                         logger.error(f"RT alignment atlas {rt_align_template_atlas_uid} not found in databases")
                        
#                 except Exception as e:
#                     logger.error(f"Error loading RT alignment atlas {rt_align_template_atlas_uid}: {e}")
            
#             # Get analysis atlases from ANALYSES section
#             analyses_config = workflow_config.get('ANALYSES', {})
#             for atlas_type, methods in analyses_config.items():
#                 for polarity, config_data in methods.items():
#                     analysis_atlas_uid = config_data.get('ATLAS', {}).get('uid')
#                     if analysis_atlas_uid:
#                         try:
#                             if Path(self.project_db_path).exists():
#                                 logger.info(f"Looking for analysis atlas {analysis_atlas_uid} in project database")
#                                 atlas_df = dbi.get_atlas_metadata_from_db(self.project_db_path, analysis_atlas_uid, validation=True)
#                             else:
#                                 atlas_df = pd.DataFrame()
#                             if atlas_df.empty:
#                                 logger.info(f"Looking for analysis atlas {analysis_atlas_uid} in main database")
#                                 atlas_df = dbi.get_atlas_metadata_from_db(self.main_db_path, analysis_atlas_uid, validation=True)
                            
#                             if not atlas_df.empty:
#                                 atlas_info = atlas_df.iloc[0]
#                                 analysis_atlas_data = {
#                                     'atlas_uid': analysis_atlas_uid,
#                                     'atlas_name': atlas_info.get('atlas_name', ''),
#                                     'atlas_description': atlas_info.get('atlas_description', ''),
#                                     'chromatography': atlas_info.get('chromatography', '').lower(),
#                                     'polarity': polarity.lower(),
#                                     'atlas_type': atlas_type.lower(),
#                                     'workflow': workflow_name.lower(),
#                                     'analysis_params': config_data.get('PARAMS', {})
#                                 }
#                                 atlas_data['analysis_atlases'].append(analysis_atlas_data)
#                             else:
#                                 logger.error(f"Analysis atlas {analysis_atlas_uid} not found in databases")
                                
#                         except Exception as e:
#                             logger.error(f"Error loading analysis atlas {analysis_atlas_uid}: {e}")
        
#         return atlas_data

#     def run_complete_workflow(self,
#                             new_lcmsruns: bool = False, 
#                             stop_at_stage: WorkflowStage = WorkflowStage.FINAL_REPORT,
#                             analysis_subset: List[Tuple[str, str, str]] = None,
#                             create_analysis_notebooks: bool = False) -> None:
#         """
#         Run the complete workflow up to the specified stage with optional caching
        
#         Args:
#             stop_at_stage: Which stage to stop at
#             analysis_subset: List of (atlas_type, polarity) tuples to limit analysis
#             create_analysis_notebooks: Whether to create individual notebooks per analysis
#         """
        
#         # Stage 1: Project Setup
#         if self.current_stage == WorkflowStage.PROJECT_SETUP:
#             logger.info("\n========== Stage 1: Project Setup ==========\n")

#             # Initialize project setup directly in workflow
#             self._setup_project_database(new_lcmsruns)
#             self.load_parquet_files()
            
#             if stop_at_stage == self.current_stage.value:
#                 logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
#                 return

#             self.current_stage = WorkflowStage.RT_CORRECTION
        
#         # Stage 2: RT Correction
#         if self.current_stage == WorkflowStage.RT_CORRECTION:
#             logger.info("\n========== Stage 2: RT Correction ==========\n")

#             logger.info("Starting RT correction...")
#             self._run_rt_correction_workflow(analysis_subset)

#             if stop_at_stage == self.current_stage.value:
#                 logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
#                 return

#             self.current_stage = WorkflowStage.AUTO_IDENTIFICATION
        
#         # Stage 3: Auto Identification
#         if self.current_stage == WorkflowStage.AUTO_IDENTIFICATION:
#             logger.info("\n========== Stage 3: Auto Identification ==========\n")

#             logger.info(f"Starting auto identifications...")
#             self._run_auto_identification_workflow(self.config, analysis_subset)

#             if stop_at_stage == self.current_stage.value:
#                 logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
#                 if create_analysis_notebooks:
#                     self._create_individual_curation_notebooks()
#                 return
            
#             self.current_stage = WorkflowStage.MANUAL_CURATION

#         # Stage 4: Manual Curation (returns GUI for interactive work) 
#         if self.current_stage == WorkflowStage.MANUAL_CURATION:
#             logger.info("\n========== Stage 4: Manual Curation ==========\n")

#             if stop_at_stage == self.current_stage.value:
#                 logger.info(f"Stopping workflow at the end of the {stop_at_stage} stage.")
#                 if create_analysis_notebooks:
#                     return self._create_individual_curation_notebooks()

#             self.current_stage = WorkflowStage.FINAL_REPORT

#         # Stage 5: Final Report
#         if self.current_stage == WorkflowStage.FINAL_REPORT or stop_at_stage == WorkflowStage.FINAL_REPORT:
#             logger.info("\n========== Stage 5: Final Report Generation ==========\n")

#             # Create final report manager
#             final_report = FinalReportManager()
            
#             output_path = Path(self.analysis_directory) / "final_targeted_analysis_report"
#             report_df = final_report.generate_comprehensive_report(self.config, str(output_path))
            
#             logger.info(f"Workflow complete! Final report with {len(report_df)} identifications")
#             #return report_df
    
#     def run_auto_identification_db_verification(self) -> None:
#         """Run database verification for auto identification stage to ensure all required data is present before analysis"""
#         results = dbi.verify_project_db_holdings(self.project_db_path, self.rt_alignment_number, self.analysis_number)

#         for table, table_contents in results.items():
#             if not table_contents.empty:
#                 logger.info(f"{table} table contents:")
#                 display(table_contents)
#         return

#     def _create_individual_curation_notebooks(self) -> List[str]:
#         """Create individual notebooks for each analysis type/polarity combination"""
#         created_notebooks = []
#         logger.info("Creating individual curation notebooks for each analysis...")
#         # Get all analysis combinations that have auto identifications

#         for analysis_key, chrom_pol in self.results.auto_ids.items():
#             for chrom_pol, auto_list in chrom_pol.items():
#                 if auto_list:
#                     logger.info(f"Creating curation notebook for {analysis_key} in {chrom_pol} mode ({len(auto_list)} compounds)")
#                     notebook_path = self._create_analysis_specific_notebook(analysis_key, chrom_pol, auto_list)
#                     created_notebooks.append(notebook_path)
        
#         logger.info(f"Created {len(created_notebooks)} individual curation notebooks")
#         return created_notebooks
    
#     def _create_analysis_specific_notebook(self, atlas_type: str, chrom_pol: str, auto_list: List) -> str:
#         """Create a notebook for a specific analysis type and chrom_pol"""
        
#         # Copy the existing config file to the analysis directory for notebook reproducibility
#         analysis_config_path = Path(self.analysis_directory) / "metatlas2_config.yaml"
#         existing_config_path = self.config['ENV']['PATHS']['config_path']
#         if not analysis_config_path.exists():
#             shutil.copy(existing_config_path, analysis_config_path)
        
#         notebook_filename = f"curation_{atlas_type}_{chrom_pol}.ipynb"
#         notebook_path = Path(self.analysis_directory) / notebook_filename
        
#         # Create notebook content specific to this analysis
#         notebook_content = self._generate_analysis_notebook_content(atlas_type, chrom_pol, auto_list)
        
#         # Write notebook file
#         with open(notebook_path, 'w') as f:
#             json.dump(notebook_content, f, indent=2)
        
#         logger.info(f"Created analysis-specific notebook: {notebook_path}")
#         return str(notebook_path)
    
#     # =============================================================================
#     # RT CORRECTION METHODS (Stage 2)
#     # =============================================================================

#     def _run_rt_correction_workflow(self, analysis_subset: List[Tuple[str, str, str]] = None) -> None:
#         """Run single RT correction using the RT alignment atlas"""

#         if self.atlas_data['rt_align_template_atlas']['rt_align_params'].get('use_existing_rt_correction', False):
#             logger.info("Parameter 'use_existing_rt_correction' is True - checking for existing RT alignment model in database...")
#             rt_model = self._get_existing_rt_alignment_from_db()
#             if rt_model is None:
#                 logger.warning(f"No existing RT alignment model found in database for alignment number {self.rt_alignment_number} and QC atlas UID {self.atlas_data['rt_align_template_atlas']['atlas_uid']}. Running fresh RT correction workflow.")
#         logger.info("Running RT correction workflow")
#         rt_model = self._do_rt_correction()

#         logger.info("Applying RT correction to all analysis atlases...")
#         self._apply_rt_correction_to_analysis_atlases(rt_model, analysis_subset)

#     def _get_existing_rt_alignment_from_db(self) -> Optional[Dict]:
#         """
#         Check if RT alignment already exists in database for this project.
#         Returns the full RT model dict needed for applying corrections.
#         """
#         qc_atlas_uid = self.atlas_data.get('rt_align_template_atlas', {}).get('atlas_uid', None)
#         if not qc_atlas_uid:
#             logger.debug(f"Could not find QC atlas UID: {qc_atlas_uid}")
#             return None
        
#         with dbi.get_db_connection(self.project_db_path) as conn:
#             result = conn.execute("""
#             SELECT 
#                 rt_alignment_uid,
#                 project_name,
#                 rt_alignment_number,
#                 qc_atlas_uid,
#                 model_type,
#                 polynomial_degree,
#                 r_squared,
#                 rmse,
#                 coefficients,
#                 equation,
#                 num_qc_files,
#                 num_compounds,
#                 created_by,
#                 created_date,
#                 metadata
#             FROM rt_alignment
#             WHERE project_name = ? 
#                 AND rt_alignment_number = ?
#                 AND qc_atlas_uid = ?
#             ORDER BY created_date DESC
#             LIMIT 1
#             """, [self.project_name, self.rt_alignment_number, qc_atlas_uid]).fetchone()
            
#             if result:
#                 rt_model = {
#                     'rt_alignment_uid': result[0],
#                     'project_name': result[1],
#                     'rt_alignment_number': result[2],
#                     'qc_atlas_uid': result[3],
#                     'model_type': result[4],
#                     'polynomial_degree': result[5],
#                     'r_squared': result[6],
#                     'rmse': result[7],
#                     'coefficients': json.loads(result[8]) if result[8] else [],
#                     'equation': result[9],
#                     'num_qc_files': result[10],
#                     'num_compounds': result[11],
#                     'created_by': result[12],
#                     'created_date': result[13],
#                     'metadata': result[14]
#                 }
                
#                 logger.info(f"Found existing RT alignment in database: {result[0]}")
        
#                 if rt_model:
#                     logger.info(f"Using existing RT correction with UID {rt_model['rt_alignment_uid']} from database created on {rt_model['created_date']} by {rt_model['created_by']}")
#                     logger.info(f"  R² = {rt_model['r2']:.4f}, RMSE = {rt_model['rmse']:.4f}")
                    
#                     # Store the model and apply to target atlases
#                     rt_align_template_atlas = self.atlas_data.get('rt_align_template_atlas')
#                     chrom_pol = f"{rt_align_template_atlas['chromatography']}_{rt_align_template_atlas['polarity']}"
#                     self.rt_models[chrom_pol] = rt_model
#                 else:
#                     logger.error(f"Tried to find RT model for alignment number {self.rt_alignment_number} and UID {self.atlas_data['rt_align_template_atlas']['atlas_uid']} but none found in database")
#                     raise ValueError("No existing RT alignment model found in database - cannot proceed with RT correction. Please run without 'use_existing_rt_correction' or ensure the correct RT model exists in the database.")
        
#         return None

#     def _do_rt_correction(self) -> None:
#         """Run fresh RT correction using single RT alignment atlas"""
        
#         rt_align_template_atlas = self.atlas_data.get('rt_align_template_atlas', {})
#         if not rt_align_template_atlas:
#             raise ValueError("No RT alignment atlas specified in configuration")
        
#         atlas_uid = rt_align_template_atlas['atlas_uid']
#         chromatography = rt_align_template_atlas['chromatography']
#         polarity = rt_align_template_atlas['polarity']
        
#         logger.info(f"Running RT correction using atlas {atlas_uid} ({chromatography}_{polarity})")

#         # Get QC files matching the RT alignment atlas method
#         chrom_pol = f"{chromatography}_{polarity}"
#         logger.info(f"Loading QC files for {chrom_pol}...")
#         qc_files_df = self._filter_files_by_method(
#             self.parquet_files['qc'], 
#             chrom_pol
#         )
#         if qc_files_df.empty:
#             raise ValueError(f"No QC files found for {chrom_pol}")
#         logger.info(f"Found {len(qc_files_df)} QC files for {chrom_pol}")

#         # Extract QC matches for RT model building
#         logger.info(f"Extracting QC atlas compound data from {len(qc_files_df)} QC files...")
#         qc_matches = rat.extract_matches_from_qc_files(
#             self.main_db_path,
#             atlas_uid,
#             qc_files_df,
#             self.config["WORKFLOWS"][chromatography.upper()]["RT_ALIGN"]["PARAMS"]
#         )
        
#         matching_stats = rat.evaluate_qc_matching_stats(qc_matches)
#         # Print QC compound matching stats in a readable notebook format
#         matching_md = "### QC Compound Matching Summary\n"
#         for key, value in matching_stats.items():
#             if isinstance(value, dict):
#                 matching_md += f"- **{key}:**\n"
#                 for subkey, subval in value.items():
#                     matching_md += f"    - {subkey}: {subval}\n"
#             else:
#                 matching_md += f"- **{key}:** {value}\n"
#         display(Markdown(matching_md))
#         #logger.info("QC compound matching completed:\n" + json.dumps(matching_stats, indent=2))

#         logger.info(f"Building RT alignment model using {len(qc_matches)} matches")
#         best_model, modeling_data, _ = rat.build_rt_alignment_model(qc_matches, self.config["WORKFLOWS"][chromatography.upper()]["RT_ALIGN"]["PARAMS"])
#         logger.info(f"RT model created with R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        
#         logger.info("Saving RT model to database...")
#         rt_alignment_uid = dbi.save_rt_alignment_model_to_db(
#             atlas_uid,
#             self.project_db_path,
#             self.rt_alignment_number,
#             best_model,
#             qc_files_df,
#             modeling_data.to_dict('records')
#         )

#         # Store the single RT model
#         best_model['rt_alignment_uid'] = rt_alignment_uid
#         self.rt_models[chrom_pol] = best_model
        
#         # Create RT model plot
#         rat.visualize_RT_model(modeling_data, best_model, self.analysis_directory, rt_alignment_uid)

#         return best_model

#     def _filter_files_by_method(self, parquet_file_info: pd.DataFrame, chrom_pol: str) -> pd.DataFrame:
#         """Filter files by chromatography and polarity using DataFrame columns."""
#         logger.info(f"Filtering {len(parquet_file_info)} parquet files for method {chrom_pol}...")
        
#         # Parse requested chromatography and polarity
#         chrom, pol = chrom_pol.split('_')
#         chrom_map = {
#             'hilicz': ['hilic', 'hilicz'],
#             'hilic': ['hilic', 'hilicz'],
#             'c18': ['c18'],
#         }
#         chrom_values = chrom_map.get(chrom.lower(), [chrom.lower()])
#         logger.debug(f"Filtering for chromatography values: {chrom_values} and polarity: {pol}")
        
#         # Filter using DataFrame columns (case-insensitive)
#         filtered_parquet_file_info = parquet_file_info[
#             (parquet_file_info['chromatography'].str.lower().isin(chrom_values)) &
#             (parquet_file_info['polarity'].str.lower() == pol.lower())
#         ]        
#         logger.info(f"Filtered to {len(filtered_parquet_file_info)} files")
        
#         return filtered_parquet_file_info

#     def _apply_rt_correction_to_analysis_atlases(self, model: Dict, analysis_subset: List[Tuple[str, str, str]] = None) -> None:
#         """Apply RT correction to all analysis atlases using the single RT model"""
        
#         for analysis_atlas in self.atlas_data['analysis_atlases']:
#             atlas_type = analysis_atlas['atlas_type']
#             polarity = analysis_atlas['polarity']
#             chromatography = analysis_atlas['chromatography']
#             atlas_uid = analysis_atlas['atlas_uid']
#             workflow_name = analysis_atlas['workflow']

#             # Check analysis subset to run RT alignment on)
#             if analysis_subset:
#                 current_tuple = (workflow_name.lower(), atlas_type.lower(), polarity.lower())
#                 user_tuple = list((w.lower(), a.lower(), p.lower()) for (w, a, p) in analysis_subset)
#                 if current_tuple not in user_tuple:
#                     logger.info(f"Skipping RT alignment for {atlas_type} with attributes {current_tuple} as not in supplied analysis subset {user_tuple}")
#                     continue
            
#             try:
#                 logger.info(f"Applying RT correction to {atlas_type} {chromatography} {polarity} atlas {atlas_uid}")
                
#                 # Apply RT correction to this atlas
#                 corr_atlas_df, stats = rat.apply_rt_correction_to_target(self.main_db_path, analysis_atlas, model, self.config["WORKFLOWS"][chromatography.upper()]["RT_ALIGN"]["PARAMS"])
#                 if corr_atlas_df is None or (isinstance(corr_atlas_df, pd.DataFrame) and corr_atlas_df.empty):
#                     logger.error(f"RT correction returned no data for {atlas_type} {polarity} atlas {atlas_uid}")
#                     continue

#                 # Save corrected atlas to database
#                 corr_atlas_uid, corr_atlas_name = dbi.save_rt_corrected_atlas_to_db(
#                     self.project_db_path,
#                     analysis_atlas,
#                     model,
#                     corr_atlas_df,
#                     self.rt_alignment_number,
#                     self.analysis_number,
#                 )
                
#                 # Create alignment summary
#                 rt_align_summary = rat.create_rt_alignment_summary(corr_atlas_uid,
#                                                                    model['rt_alignment_uid'], 
#                                                                    corr_atlas_name, 
#                                                                    stats
#                                                                 )
                
#                 # Print RT correction summary in a readable notebook format
#                 rt_align_md = f"### RT Correction Summary for {atlas_type} {chromatography} {polarity}\n"
#                 for key, value in rt_align_summary.items():
#                     if isinstance(value, dict):
#                         rt_align_md += f"- **{key}:**\n"
#                         for subkey, subval in value.items():
#                             rt_align_md += f"    - {subkey}: {subval}\n"
#                     else:
#                         rt_align_md += f"- **{key}:** {value}\n"
#                 display(Markdown(rt_align_md))
                
#             except Exception as e:
#                 raise ValueError(f"Failed to apply RT correction to {atlas_type} {polarity} atlas {atlas_uid}: {e}")

#     # =============================================================================
#     # AUTO IDENTIFICATION METHODS (Stage 3)
#     # =============================================================================
    
#     def _run_auto_identification_workflow(self, config: Dict, analysis_subset: List[Tuple[str, str, str]] = None) -> None:
#         """Run auto identification for specified analysis atlases with optional caching"""

#         logger.info(f"Running auto identifications workflow...")
        
#         for analysis_atlas in self.atlas_data['analysis_atlases']:
#             atlas_type = analysis_atlas['atlas_type']
#             polarity = analysis_atlas['polarity']
#             chromatography = analysis_atlas['chromatography']
#             workflow_name = analysis_atlas['workflow']
#             chrom_pol = f"{chromatography}_{polarity}"
#             analysis_params = analysis_atlas.get('analysis_params', {})

#             # Check analysis subset to process
#             if analysis_subset:
#                 current_tuple = (workflow_name.lower(), atlas_type.lower(), polarity.lower())
#                 user_tuple = list((w.lower(), a.lower(), p.lower()) for (w, a, p) in analysis_subset)
#                 if current_tuple not in user_tuple:
#                     logger.info(f"Skipping analysis for {atlas_type} with attributes {current_tuple} as not in supplied analysis subset {user_tuple}")
#                     continue
            
#             # Get analysis parameters from config
#             use_existing_ids_in_db = analysis_params.get("use_existing_hits", False)

#             if use_existing_ids_in_db:
#                 pass

#             logger.info(f"Running auto identification for {atlas_type} {polarity} using atlas {target_atlas_uid}")
            
#             # Run targeted analysis workflow with specific parameters
#             analysis_results = tga.run_targeted_analysis_workflow(
#                 project_db_path=self.project_db_path,
#                 target_atlas_uid=target_atlas_uid,
#                 rt_alignment_number=self.rt_alignment_number,
#                 analysis_number=self.analysis_number,
#                 config=config,
#                 analysis_params=analysis_params
#             )
    
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
            logger.info("Main database already exists, not creating a new one.")

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

    def create_compound_db_entries(self) -> Tuple[List[Compound], List[CompoundReference]]:
        """
        Complete database setup: create database and load compounds.
            
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
        logger.info("Initializing AtlasManager with configuration")
        self.config = config
        self.main_db_path = config["ENV"]["PATHS"]["main_database"]
        self.created_atlases = []
        logger.info("AtlasManager initialized successfully")

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
        atlas_compounds_df = dbi.get_atlas_compounds_table(database_path, atlas_uid)
        
        # Convert compounds to CompoundReference objects
        logger.info(f"Converting {len(atlas_compounds_df)} atlas compounds to CompoundReference objects")
        compounds_dict = {}
        for _, row in atlas_compounds_df.iterrows():
            try:
                compound_ref = CompoundReference.from_atlas_row(row)
                compounds_dict[row.get('inchi_key', compound_ref.compound_uid)] = compound_ref
            except Exception as e:
                raise ValueError(f"Failed to convert atlas compound row to CompoundReference: {e}")
        
        # Create Atlas object
        logger.info(f"Creating Atlas object for {atlas_uid} with {len(compounds_dict)} compounds")
        atlas_obj = Atlas(
            atlas_uid=atlas_uid,
            atlas_name=atlas_info.get('atlas_name', ''),
            atlas_description=atlas_info.get('atlas_description', ''),
            chromatography=atlas_info.get('chromatography', ''),
            polarity=atlas_info.get('polarity', ''),
            compound_references=compounds_dict,
            created_by=atlas_info.get('created_by', ''),
            created_date=atlas_info.get('created_date', ''),
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

        for atlas_chrom, atlas_info in atlas_configs.items():
            for atlas_pol, atlas_types in atlas_info.items():
                for atlas_type, atlas_details in atlas_types.items():
                    if atlas_details and atlas_details.get('path') is not None:
                        try:
                            # Extract configuration
                            atlas_file_path = atlas_details['path']
                            atlas_name = atlas_details['name']
                            atlas_description = atlas_details['desc']
                            
                            logger.info(f"Creating atlas from config: {atlas_name} ({atlas_type}/{atlas_chrom}/{atlas_pol})")

                            logger.info(f"Loading atlas data from file: {atlas_file_path}")
                            atlas_compounds_df = ldt.load_atlas_input(atlas_file_path)
                            
                            logger.info(f"Creating Atlas object: {atlas_name}")
                            atlas_obj = self.create_atlas_from_dataframe(atlas_compounds_df, atlas_name, atlas_description, atlas_type, atlas_chrom, atlas_pol)
                            
                            logger.info(f"Saving atlas to database: {atlas_name}")
                            dbi.save_atlas_to_database(atlas_obj, self.main_db_path)

                            # Store in created atlases list
                            logger.info(f"Storing created atlas in workflow state: {atlas_name}")
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
                            raise ValueError(f"Failed to create atlas for {atlas_type}/{atlas_chrom}/{atlas_pol}: {e}")
                    else:
                        logger.debug(f"Skipping {atlas_type}/{atlas_chrom}/{atlas_pol} - no path specified")
        
        logger.info(f"Created {len(created_atlases)} atlases from configuration file input:")
        for atlas in created_atlases:
            logger.info(f"  Atlas: {atlas.atlas_name}")
            logger.info(f"    UID: {atlas.atlas_uid}")
            logger.info(f"    Method: {atlas.chromatography}/{atlas.polarity}")
            logger.info(f"    Compounds: {len(atlas.compound_references)}")

        return created_atlases
    
    def create_atlas_from_dataframe(self, atlas_df: pd.DataFrame, 
                                        atlas_name: str, 
                                        atlas_description: str, 
                                        atlas_type: str = "ref", 
                                        chromatography: str = None, 
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
                #logger.debug(f"Reusing existing reference for {inchi_key}")
            else:
                references_created += 1
                #logger.debug(f"Will create new reference for {inchi_key}")

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
        atlas_uid = dbi._generate_uid("ref_atlas", decorator=f"{atlas_type.lower()}-{chromatography.lower()}-{polarity.lower()}")
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
    Manages final report generation from analysis objects.
    """
    auto_ids: Dict[str, Dict[str, Dict]]
    
    def generate_comprehensive_report(self, config: Dict, output_path: str = None) -> pd.DataFrame:
        """Generate final comprehensive report from all curated identifications"""
        all_auto_ids = []
        for atlas_type in self.auto_ids.values():
            for method_ids in atlas_type.values():
                all_auto_ids.extend(method_ids)
        
        if not all_auto_ids:
            logger.warning("No auto identifications found for report generation")
            return pd.DataFrame()
        
        # Build comprehensive report
        report_rows = []
        
        for idx, pid in enumerate(all_auto_ids):
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
    
    def _calculate_msms_quality(self, pid) -> float:
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
    
    def _calculate_mz_quality(self, pid) -> float:
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
    
    def _calculate_rt_quality(self, pid) -> float:
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
# INDEPENDENT WORKFLOW FUNCTIONS
# =============================================================================

def _set_up_paths(config: Dict, project_name: str) -> Tuple[str, str, str]:
    """Set up and validate paths for project directory, raw data, and project database"""
    
    workflow_paths = {}

    workflow_paths['raw_data_directory'] = str(Path(config['ENV']['PATHS']['raw_data_dir']) / project_name)
    workflow_paths['project_directory'] = str(Path(config['ENV']['PATHS']['projects_dir']) / project_name)
    workflow_paths['project_db_path'] = str(Path(workflow_paths['project_directory']) / f"{project_name}.duckdb")
    workflow_paths['main_db_path'] = config["ENV"]["PATHS"]["main_database"]
    
    # Validate raw data directory exists
    if not Path(workflow_paths['raw_data_directory']).exists():
        raise ValueError(f"Raw data directory not found: {workflow_paths['raw_data_directory']}")
    
    # Create project directory if it doesn't exist
    Path(workflow_paths['project_directory']).mkdir(parents=True, exist_ok=True)
    
    return workflow_paths

def run_project_setup(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1,
    new_lcmsruns: bool = False
) -> str:
    """
    Stage 1: Project Setup
    Creates project database and loads LCMS run files.
    
    Returns:
        project_db_path: Path to created project database
    """
    logger.info("\n========== Stage 1: Project Setup ==========\n")
    
    # Setup paths
    workflow_paths = _set_up_paths(config, project_name)
    
    # Create project database
    logger.info(f"Creating project database at {workflow_paths['project_db_path']}...")
    dbi.create_project_database(workflow_paths['project_db_path'], rt_alignment_number, analysis_number)
    
    # Load LCMS runs
    logger.info(f"Loading LCMS runs for {project_name}...")
    try:
        dbi.save_lcmsruns_to_db(
            workflow_paths['project_db_path'],
            project_name,
            workflow_paths['raw_data_directory'],
            new_lcmsruns
        )
        logger.info("LCMS runs loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load LCMS runs: {e}")
    
    return workflow_paths['project_db_path']


def run_rt_alignment(project_name: str,
                     config: Dict,
                     rt_alignment_number: int,
                     workflow: List
) -> None:
    """Run fresh RT alignment using single RT alignment atlas"""
    
    logger.info("Parsing configuration file...")
    workflow_paths = _set_up_paths(config, project_name)
    template_chromatography = workflow[0]
    template_polarity = "POS"
    template_atlas_uid = config['WORKFLOWS']['RT_ALIGNMENT'][template_chromatography][template_polarity].get('ATLAS').get('uid', None)
    target_atlas_uid = config['WORKFLOWS']['TARGETED_ANALYSIS'][workflow[0]][workflow[1]][workflow[2]].get('ATLAS').get('uid', None)
    rt_alignment_parameters = config['WORKFLOWS']['RT_ALIGNMENT'][template_chromatography][template_polarity].get('PARAMS', {})
    rt_alignment_directory = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
    main_db_path = workflow_paths['main_db_path']
    use_existing_rt_alignment = rt_alignment_parameters.get('use_existing_rt_alignment', False)
    existing_rt_alignment = dbi.get_rt_alignment_model_from_db(workflow_paths['project_db_path'],
                                                               template_atlas_uid, 
                                                               rt_alignment_number)

    best_model = None
    if use_existing_rt_alignment and existing_rt_alignment is not None:
        logger.info(f"Using the existing RT alignment model from database for atlas {template_atlas_uid} ({template_chromatography} {template_polarity}) and RT Alignment number {rt_alignment_number}")
        best_model = dbi.get_rt_alignment_model_from_db(workflow_paths['project_db_path'], 
                                                              template_atlas_uid, 
                                                              rt_alignment_number)
        best_model = rat.calculate_model_values_from_existing(best_model)
        if best_model:
            logger.info(f"Loaded existing RT alignment model with R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
    elif use_existing_rt_alignment and existing_rt_alignment is None:
        logger.warning(f"Variable 'use_existing_rt_alignment' is True, but no existing RT alignment model found in database for atlas {template_atlas_uid} ({template_chromatography} {template_polarity}) and RT alignment number {rt_alignment_number}. Running RT alignment from scratch.")
    elif not use_existing_rt_alignment and existing_rt_alignment is not None:
        raise ValueError(f"Variable 'use_existing_rt_alignment' is False, but RT alignment model already exists in database for atlas {template_atlas_uid} ({template_chromatography} {template_polarity}) and RT alignment number {rt_alignment_number}. To avoid overwriting, either set use_existing_rt_alignment to True or choose a different RT alignment number.")
    elif not use_existing_rt_alignment and existing_rt_alignment is None:
        logger.info(f"Variable 'use_existing_rt_alignment' is False and no existing RT alignment model found in database for atlas {template_atlas_uid} ({template_chromatography} {template_polarity}) and RT alignment number {rt_alignment_number}. Running RT alignment from scratch.")

    if best_model is None:
        logger.info(f"Loading QC files for RT alignment from project database...")
        database_files = dbi.get_files_by_type_from_db(workflow_paths['project_db_path'], 
                                                        file_types=['qc'],
                                                        file_format='parquet',
                                                        chromatography=template_chromatography,
                                                        polarity=template_polarity)
        qc_files_df = database_files.get('qc', pd.DataFrame())
        if qc_files_df.empty:
            raise ValueError(f"No QC files found for input method {template_chromatography} {template_polarity} in project database.")
        logger.info(f"Found {len(qc_files_df)} QC files for {template_chromatography} {template_polarity}")

        logger.info(f"Extracting QC atlas compound data from {len(qc_files_df)} QC files...")
        qc_matches = rat.extract_matches_from_qc_files(
            main_db_path,
            template_atlas_uid,
            qc_files_df,
            rt_alignment_parameters
        )

        logger.info(f"Evaluating QC compound matching stats for {len(qc_matches)} matches...")
        matching_stats = rat.evaluate_qc_matching_stats(qc_matches)
        matching_md = "### QC Compound Matching Summary\n"
        for key, value in matching_stats.items():
            if isinstance(value, dict):
                matching_md += f"- **{key}:**\n"
                for subkey, subval in value.items():
                    matching_md += f"    - {subkey}: {subval}\n"
            else:
                matching_md += f"- **{key}:** {value}\n"
        display(Markdown(matching_md))

        logger.info(f"Building RT alignment model using {len(qc_matches)} matches")
        best_model, modeling_data, _ = rat.build_rt_alignment_model(qc_matches, 
                                                                    rt_alignment_parameters)
        logger.info(f"RT model created with R² = {best_model['r2']:.4f}, RMSE = {best_model['rmse']:.4f}")
        
        logger.info("Saving RT model to database...")
        dbi.save_rt_alignment_model_to_db(template_atlas_uid,
                                        workflow_paths['project_db_path'],
                                        rt_alignment_number,
                                        best_model,
                                        qc_files_df,
                                        modeling_data.to_dict('records')
                                        )
        
        logger.info(f"Creating RT alignment plot and saving to {rt_alignment_directory}...")
        rat.visualize_RT_model(modeling_data, 
                            best_model, 
                            rt_alignment_directory
                            )
    
    logger.info(f"Applying RT alignment to {workflow[0]} {workflow[1]} {workflow[2]} atlas {target_atlas_uid}...")
    align_atlas_df, stats = rat.apply_rt_alignment_to_target(main_db_path, 
                                                             target_atlas_uid, 
                                                             best_model, 
                                                             rt_alignment_parameters
                                                             )

    logger.info(f"Saving RT-aligned atlas to database...")
    align_atlas_uid, align_atlas_name = dbi.save_rt_aligned_atlas_to_db(workflow_paths['project_db_path'],
                                                                      workflow_paths['main_db_path'],
                                                                      target_atlas_uid,
                                                                      best_model,
                                                                      align_atlas_df,
                                                                      rt_alignment_number
                                                                      )
    
    logger.info("Generating RT alignment summary...")
    rt_align_summary = rat.create_rt_alignment_summary(align_atlas_uid,
                                                        best_model['rt_alignment_uid'], 
                                                        align_atlas_name, 
                                                        stats
                                                    )
    
    rt_align_md = f"### RT Alignment Summary for {workflow[0]} {workflow[1]} {workflow[2]}\n"
    for key, value in rt_align_summary.items():
        if isinstance(value, dict):
            rt_align_md += f"- **{key}:**\n"
            for subkey, subval in value.items():
                rt_align_md += f"    - {subkey}: {subval}\n"
        else:
            rt_align_md += f"- **{key}:** {value}\n"
    display(Markdown(rt_align_md))

    return best_model


def run_auto_identification(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1,
    workflow: Optional[List[Tuple[str, str, str]]] = None
) -> Dict[str, int]:
    """
    Stage 3: Auto Identification
    Runs targeted analysis using RT-aligned atlases from database.
    Can be run independently if RT alignment has been completed.
    
    Returns:
        Dict with analysis statistics: {atlas_uid: num_identifications}
    """
    logger.info("\n========== Stage 3: Auto Identification ==========\n")
    
    # Get paths
    workflow_paths = _set_up_paths(config, project_name)
    print(workflow)
    print(config['WORKFLOWS']['TARGETED_ANALYSIS'][workflow[0]][workflow[1]])
    workflow_params = config['WORKFLOWS']['TARGETED_ANALYSIS'][workflow[0]][workflow[1]][workflow[2]]['PARAMS']
    workflow_template_atlas_uid = config['WORKFLOWS']['RT_ALIGNMENT'][workflow[0]][workflow[1]][workflow[2]]['ATLAS']['uid']

    # Verify project database exists
    if not Path(workflow_paths['project_db_path']).exists():
        raise FileNotFoundError(
            f"Project database not found: {workflow_paths['project_db_path']}. "
            "Please run project setup first."
        )
    
    # Get RT-aligned atlases from database
    rt_aligned_atlases = dbi.get_rt_aligned_atlases_from_db(project_db_path=workflow_paths['project_db_path'],
                                                            rt_alignment_number=rt_alignment_number,
                                                            )
    print(rt_aligned_atlases)
    
    logger.info(f"Found {len(rt_aligned_atlases)} RT-aligned atlases to analyze")
    
    # Run analysis for each atlas
    results = {}
    for atlas_info in rt_aligned_atlases:
        
        # # Check if analysis already exists
        # use_existing = config['ANALYSIS'].get('use_existing_identifications', False)
        # if use_existing:
        #     existing_count = dbi.count_existing_identifications(
        #         project_db_path=workflow_paths['project_db_path'],
        #         atlas_uid=atlas_info['atlas_uid'],
        #         rt_alignment_number=rt_alignment_number,
        #         analysis_number=analysis_number
        #     )
        #     if existing_count > 0:
        #         logger.info(
        #             f"Using existing {existing_count} identifications for "
        #             f"{atlas_info['atlas_name']}"
        #         )
        #         results[atlas_info['atlas_uid']] = existing_count
        #         continue
        
        # Run targeted analysis
        logger.info(f"Running auto identification for {atlas_info['atlas_name']}...")
        atlas_df, analysis_results = tga.run_targeted_analysis_workflow(
            project_db_path=workflow_paths['project_db_path'],
            target_atlas_uid=atlas_info['atlas_uid'],
            rt_alignment_number=rt_alignment_number,
            analysis_number=analysis_number,
            config=config,
            analysis_params=workflow_params
        )
        
        num_identifications = len(analysis_results.get('compounds', {}))
        results[atlas_info['atlas_uid']] = num_identifications
        logger.info(f"Completed: {num_identifications} identifications")
    
    total_identifications = sum(results.values())
    logger.info(f"Auto identification complete: {total_identifications} total identifications")
    return results


def run_manual_curation(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1,
    create_notebooks: bool = True
) -> List[str]:
    """
    Stage 4: Manual Curation
    Creates individual curation notebooks for each analysis.
    
    Returns:
        List of created notebook paths
    """
    logger.info("\n========== Stage 4: Manual Curation ==========\n")
    
    if not create_notebooks:
        logger.info("Notebook creation skipped")
        return []
    
    # Get paths
    workflow_paths = _set_up_paths(config, project_name)
    rt_alignment_directory = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
    analysis_directory = str(Path(rt_alignment_directory) / f"{project_name}_RTA{rt_alignment_number}_ALY{analysis_number}")
    
    # Get all analyses from database
    analyses = dbi.get_analyses_summary(
        project_db_path=workflow_paths['project_db_path'],
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number
    )
    
    # Create notebook for each analysis
    created_notebooks = []
    for analysis in analyses:
        if analysis['compound_count'] > 0:
            notebook_path = _create_curation_notebook(
                analysis_directory=analysis_directory,
                analysis_info=analysis,
                config=config
            )
            created_notebooks.append(notebook_path)
            logger.info(f"Created notebook: {notebook_path}")
    
    logger.info(f"Created {len(created_notebooks)} curation notebooks")
    return created_notebooks


def run_final_report(
    project_name: str,
    config: Dict,
    rt_alignment_number: int = 1,
    analysis_number: int = 1
) -> pd.DataFrame:
    """
    Stage 5: Final Report Generation
    Generates comprehensive report from all curated identifications.
    
    Returns:
        DataFrame with final report
    """
    logger.info("\n========== Stage 5: Final Report Generation ==========\n")
    
    # Get paths
    workflow_paths = _set_up_paths(config, project_name)
    rt_alignment_directory = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
    analysis_directory = str(Path(rt_alignment_directory) / f"{project_name}_RTA{rt_alignment_number}_ALY{analysis_number}")
    output_path = Path(analysis_directory) / "final_targeted_analysis_report"
    
    # Generate report (this would use your FinalReportManager logic)
    report_df = _generate_final_report(
        project_name=project_name,
        config=config,
        rt_alignment_number=rt_alignment_number,
        analysis_number=analysis_number,
        output_path=str(output_path)
    )
    
    logger.info(f"Final report generated: {len(report_df)} identifications")
    return report_df

