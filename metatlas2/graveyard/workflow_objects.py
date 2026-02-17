
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
# FINAL REPORT GENERATION (after analysis and optional manual curation)
# =============================================================================

# @dataclass
# class FinalReportManager:
#     """
#     Manages final report generation from analysis objects.
#     """
#     auto_ids: Dict[str, Dict[str, Dict]]
    
#     def generate_comprehensive_report(self, config: Dict, output_path: str = None) -> pd.DataFrame:
#         """Generate final comprehensive report from all curated identifications"""
#         all_auto_ids = []
#         for atlas_type in self.auto_ids.values():
#             for method_ids in atlas_type.values():
#                 all_auto_ids.extend(method_ids)
        
#         if not all_auto_ids:
#             logger.warning("No auto identifications found for report generation")
#             return pd.DataFrame()
        
#         # Build comprehensive report
#         report_rows = []
        
#         for idx, pid in enumerate(all_auto_ids):
#             # Calculate quality scores
#             msms_quality = self._calculate_msms_quality(pid)
#             mz_quality = self._calculate_mz_quality(pid)
#             rt_quality = self._calculate_rt_quality(pid)
#             total_score = msms_quality + mz_quality + rt_quality
#             msi_level = self._determine_msi_level(msms_quality, mz_quality, rt_quality)
            
#             # Get curation status with fallback
#             curation_status = getattr(pid, 'curation_status', 'pending')
            
#             report_row = {
#                 'index': idx,
#                 'atlas_type': getattr(pid, 'atlas_type', 'unknown'),
#                 'chromatography_polarity': f"{pid.chromatography}_{pid.polarity}",
#                 'compound_name': pid.compound_name,
#                 'inchi_key': pid.inchi_key,
#                 'formula': pid.formula,
#                 'adduct': pid.adduct,
#                 'curation_status': curation_status,
#                 'msms_quality': msms_quality,
#                 'mz_quality': mz_quality,
#                 'rt_quality': rt_quality,
#                 'total_score': total_score,
#                 'msi_level': msi_level,
#                 'ms1_notes': pid.ms1_notes,
#                 'ms2_notes': pid.ms2_notes,
#                 'analyst_notes': pid.analyst_notes,
#                 'identification_notes': pid.identification_notes,
#                 'atlas_rt_peak': pid.atlas_rt_peak,
#                 'current_rt_peak': pid.rt_peak,
#                 'rt_shift': pid.rt_peak - pid.atlas_rt_peak,
#                 'rt_modified': pid.is_rt_modified,
#                 'best_eic_file': pid.best_eic_file,
#                 'best_eic_intensity': pid.best_eic_intensity,
#                 'best_eic_ppm_error': pid.best_eic_ppm_error,
#                 'best_eic_rt_error': pid.best_eic_rt_error,
#                 'best_ms2_file': pid.best_ms2_file,
#                 'best_ms2_database': pid.best_ms2_database,
#                 'best_ms2_score': pid.best_ms2_score,
#                 'best_ms2_num_matches': pid.best_ms2_num_matches
#             }
            
#             report_rows.append(report_row)
        
#         # Create DataFrame
#         report_df = pd.DataFrame(report_rows)
        
#         # Sort by atlas type, then by chromatography/polarity, then by RT
#         report_df = report_df.sort_values([
#             'atlas_type', 
#             'chromatography_polarity', 
#             'atlas_rt_peak'
#         ]).reset_index(drop=True)
        
#         # Update index after sorting
#         report_df['index'] = range(len(report_df))
        
#         # Save to file if path provided
#         if output_path:
#             output_path = Path(output_path)
#             output_path.parent.mkdir(parents=True, exist_ok=True)
            
#             # Save as both Excel and CSV
#             report_df.to_excel(output_path.with_suffix('.xlsx'), index=False)
#             report_df.to_csv(output_path.with_suffix('.csv'), index=False)
            
#             logger.info(f"Final report saved to {output_path}")
        
#         return report_df
    
#     def _calculate_msms_quality(self, pid) -> float:
#         """Calculate MS/MS quality score"""
#         ms2_notes = pid.ms2_notes.lower()
#         if '1.0' in ms2_notes:
#             return 3.0
#         elif '0.5' in ms2_notes:
#             return 1.5
#         elif '0.0' in ms2_notes or 'no selection' in ms2_notes:
#             return 0.0
#         else:
#             return min(3.0, pid.best_ms2_score * 3.0) if pid.best_ms2_score > 0 else 0.0
    
#     def _calculate_mz_quality(self, pid) -> float:
#         """Calculate m/z quality score"""
#         ppm_error = abs(pid.best_eic_ppm_error) if pid.best_eic_ppm_error else float('inf')
        
#         if ppm_error <= 2.0:
#             return 2.0
#         elif ppm_error <= 5.0:
#             return 1.5
#         elif ppm_error <= 10.0:
#             return 1.0
#         elif ppm_error <= 20.0:
#             return 0.5
#         else:
#             return 0.0
    
#     def _calculate_rt_quality(self, pid) -> float:
#         """Calculate RT quality score"""
#         rt_error = abs(pid.best_eic_rt_error) if pid.best_eic_rt_error else float('inf')
        
#         if rt_error <= 0.1:
#             return 2.0
#         elif rt_error <= 0.2:
#             return 1.5
#         elif rt_error <= 0.5:
#             return 1.0
#         elif rt_error <= 1.0:
#             return 0.5
#         else:
#             return 0.0
    
#     def _determine_msi_level(self, msms_quality: float, mz_quality: float, rt_quality: float) -> str:
#         """Determine MSI identification level"""
#         total_score = msms_quality + mz_quality + rt_quality
        
#         if total_score >= 6.0 and msms_quality >= 2.0:
#             return "MSI Level 2"
#         elif total_score >= 4.0 and mz_quality >= 1.0:
#             return "MSI Level 3"
#         elif total_score >= 2.0:
#             return "MSI Level 4"
#         else:
#             return "MSI Level 5"



# def run_manual_curation(
#     project_name: str,
#     config: Dict,
#     rt_alignment_number: int = 1,
#     analysis_number: int = 1,
#     create_notebooks: bool = True
# ) -> List[str]:
#     """
#     Stage 4: Manual Curation
#     Creates individual curation notebooks for each analysis.
    
#     Returns:
#         List of created notebook paths
#     """
#     logger.info("\n========== Stage 4: Manual Curation ==========\n")
    
#     if not create_notebooks:
#         logger.info("Notebook creation skipped")
#         return []
    
#     # Get paths
#     workflow_paths = _set_up_paths(config, project_name)
#     rt_alignment_directory = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
#     analysis_directory = str(Path(rt_alignment_directory) / f"{project_name}_RTA{rt_alignment_number}_ALY{analysis_number}")
    
#     # Get all analyses from database
#     analyses = dbi.get_analyses_summary(
#         project_db_path=workflow_paths['project_db_path'],
#         rt_alignment_number=rt_alignment_number,
#         analysis_number=analysis_number
#     )
    
#     # Create notebook for each analysis
#     created_notebooks = []
#     for analysis in analyses:
#         if analysis['compound_count'] > 0:
#             notebook_path = _create_curation_notebook(
#                 analysis_directory=analysis_directory,
#                 analysis_info=analysis,
#                 config=config
#             )
#             created_notebooks.append(notebook_path)
#             logger.info(f"Created notebook: {notebook_path}")
    
#     logger.info(f"Created {len(created_notebooks)} curation notebooks")
#     return created_notebooks


# def run_final_report(
#     project_name: str,
#     config: Dict,
#     rt_alignment_number: int = 1,
#     analysis_number: int = 1
# ) -> pd.DataFrame:
#     """
#     Stage 5: Final Report Generation
#     Generates comprehensive report from all curated identifications.
    
#     Returns:
#         DataFrame with final report
#     """
#     logger.info("\n========== Stage 5: Final Report Generation ==========\n")
    
#     # Get paths
#     workflow_paths = _set_up_paths(config, project_name)
#     rt_alignment_directory = str(Path(workflow_paths['project_directory']) / f"{project_name}_RTA{rt_alignment_number}")
#     analysis_directory = str(Path(rt_alignment_directory) / f"{project_name}_RTA{rt_alignment_number}_ALY{analysis_number}")
#     output_path = Path(analysis_directory) / "final_targeted_analysis_report"
    
#     # Generate report (this would use your FinalReportManager logic)
#     report_df = _generate_final_report(
#         project_name=project_name,
#         config=config,
#         rt_alignment_number=rt_alignment_number,
#         analysis_number=analysis_number,
#         output_path=str(output_path)
#     )
    
#     logger.info(f"Final report generated: {len(report_df)} identifications")
#     return report_df

