# Targeted Metabolomics Workflow Organization Recommendations

## Key Recommendations

### 1. **Workflow-Stage Organization**
- Organize classes by analytical workflow stages rather than data types
- Each stage has clear inputs, outputs, and responsibilities
- Linear progression with clear handoff points between stages

### 2. **Flat Database Storage**
- Enhanced `targeted_analysis` table with `atlas_type` and `chromatography_polarity` columns
- Atlas metadata includes `atlas_type` for categorization (QC, ISTD, EMA)
- Avoid complex nested data structures in database

### 3. **Multi-Atlas Management**
- Explicit tracking of QC, ISTD, and EMA atlases per chromatography/polarity
- Organized workflow that handles RT correction across all atlas types
- Clear separation of concerns between atlas types

## New File Structure

### `workflow_objects.py` - Core workflow classes organized by stage:

1. **ProjectSetup** - Stage 1: Project initialization
   - Manages multiple atlases (QC, ISTD, EMA) per chromatography/polarity
   - Loads and categorizes H5 files by type
   - Validates atlas coverage across methods

2. **RTCorrectionManager** - Stage 2: RT correction
   - Builds RT correction models using QC atlases
   - Applies corrections to all atlas types (QC, ISTD, EMA)
   - Creates RT-corrected versions in project database

3. **PutativeIdentificationManager** - Stage 3: Putative ID generation
   - Runs targeted analysis on all RT-corrected atlases
   - Generates PutativeIdentification objects (flat structure)
   - Organizes results by atlas type and method

4. **ManualCurationManager** - Stage 4: Manual curation
   - Provides filtered access to putative IDs for GUI
   - Tracks curation progress and status
   - Bridges between workflow classes and existing GUI

5. **FinalReportManager** - Stage 5: Report generation
   - Generates comprehensive reports from curated data
   - Calculates quality scores and MSI levels
   - Supports filtering by atlas type and method

6. **TargetedMetabolomicsWorkflow** - Main orchestrator
   - Manages progression through all workflow stages
   - Provides checkpoint/resume functionality
   - Handles workflow state tracking

## Database Schema

### `targeted_analysis` table:
```sql
CREATE TABLE targeted_analysis (
    analysis_uid TEXT,
    project_name TEXT,
    atlas_uid TEXT,
    atlas_type TEXT,
    chromatography_polarity TEXT,
    compound_uid TEXT,
    curation_status TEXT,
    PRIMARY KEY (analysis_uid, compound_uid)
);
```

### Enhanced `atlases` table:
```sql
CREATE TABLE atlases (
    atlas_uid TEXT PRIMARY KEY,
    atlas_name TEXT,
    atlas_description TEXT,
    chromatography TEXT,
    polarity TEXT,
    atlas_type TEXT,
    source_atlas_uid TEXT,
    created_by TEXT,
    last_modified TEXT
);
```

## Benefits of New Organization

### 1. **Clear Workflow Progression**
- Each stage has well-defined inputs and outputs
- Easy to understand current progress and next steps
- Natural checkpoints for saving/resuming work

### 2. **Multi-Atlas Support**
- Explicit handling of QC, ISTD, and EMA atlases
- RT correction applied consistently across all atlas types
- Clear organization by chromatography/polarity methods

### 3. **Simplified Data Access**
- Flat database structure avoids complex nested queries
- Easy filtering by atlas type, method, curation status
- Direct mapping between workflow objects and database tables

### 4. **Better Separation of Concerns**
- Each manager class has a single, clear responsibility
- Workflow orchestrator handles stage coordination
- Existing GUI code can be preserved with minimal changes

### 5. **Scalability**
- Easy to add new atlas types or workflow stages
- Supports parallel processing of different methods
- Clear extension points for new features

## Migration Strategy

### Phase 1: Database Schema Updates
- Add new columns to existing tables
- Populate atlas_type and chromatography_polarity fields
- Test with existing data

### Phase 2: Workflow Class Implementation
- Implement new workflow classes alongside existing code
- Create adapter layers for backward compatibility
- Test with simplified workflows

### Phase 3: GUI Integration
- Update GUI to work with new PutativeIdentification objects
- Add filtering by atlas type and method
- Preserve existing manual curation functionality

### Phase 4: Full Migration
- Replace existing workflow scripts with new organization
- Update all analysis pipelines
- Deprecate old class structure

## Usage Examples

### Complete Workflow:
```python
workflow = TargetedMetabolomicsWorkflow(config, project_db_path, project_dir, main_db_path)
gui = workflow.run_complete_workflow(stop_at_stage=WorkflowStage.MANUAL_CURATION)
# Manual curation in GUI
final_report = workflow.continue_to_final_report()
```

### Atlas-Type Specific Analysis:
```python
# Process only QC compounds
curation_manager = ManualCurationManager(putative_manager)
qc_gui = curation_manager.create_curation_gui(config, atlas_type='qc')
```

### Method-Specific Analysis:
```python
# Process only HILIC positive compounds
hilic_pos_gui = curation_manager.create_curation_gui(
    config, 
    atlas_type='ema', 
    chrom_pol='hilic_positive'
)
```

This organization provides a much cleaner, more maintainable structure that directly reflects your analytical workflow while preserving the functionality of your existing codebase.