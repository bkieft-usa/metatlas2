# Standards annotation: analyst workflow

This guide covers preparing reference standards, reviewing their peaks,
aligning retention times (RTs), appending atlas rows to Google Sheets, and
importing compounds and MS/MS references into the main database.

## 0. Quickstart guide

*Note: skip to Step 1 for detailed setup and workflow instructions*

### Create inputs

Log in to the NERSC Perlmutter system and copy the example input CSV file at `/global/cfs/cdirs/metatlas/projects/reference_standards_annotation/reference_standards_inputs/example_standards_input.csv` and fill it in, one row per compound + project. Then, copy the example configuration YAML file `/global/cfs/cdirs/metatlas/projects/reference_standards_annotation/reference_standards_inputs/example_standards_annotation.yaml` and fill it in; ensure you enter the correct input CSV (which you just created) and a unique `standards_run_name` (which will become the output folder for your annotations). And note the full path of the config file, which you'll be using in the command below.

### Run input table expansion and automated peak finding

On a NERSC Perlmutter login node or compute node, run the following command
(from anywhere in the filesystem).

```bash
metatlas2.sh standards-annotation --config_path /absolute/path/to/standards_config.yaml
```

### Annotate and curate compound peaks in EICs

An annotation notebook will be created and its location printed to the command
line standard output. Open the notebook to begin annotation.

Run the notebook through Step 1 and click the "Open Standards Annotation GUI" link.
A Dash app will open in another tab in your browser. Complete the annotation by using
checkboxes to select high-quality peaks (or manually create new peaks) that will go 
into the centralized metatlas atlas table. After finishing peak selection (either
a completed or partial analysis), click "Save and Exit" in the top right corner of
the GUI to ensure all changes are saved to the database. Note: all analyst input
(checkboxes, notes, manual peaks) is saved every Next/Previous navigation, but the final
"Save and Exit" does database checks and shuts down the app port.

### Create peak selection summaries

Once the message "Analyst clicked Save & Exit -- shutting down GUI server" appears
in the notebook cell output, run the cells in Step 2. This summarizes and freezes the
current **annotation run** (numbered 1, 2, 3, ...): a summary table and figures are
written to `outputs/annotation_run_N/` and (optionally) uploaded to a shared Google
Drive folder. The peak selections are now ready for review.

After review, if changes are necessary, rerun through Step 1 again to re-open the
GUI. Because annotation run N is frozen, the GUI automatically starts annotation run
N+1 as a copy of run N; make the requested updates there. **Important:** after updating
peaks in the GUI, ALWAYS re-run Step 2 to summarize and freeze the new annotation run.
Every annotation run stays in the project database: set `ANNOTATION_RUN_ID` in Step 1
to view any earlier run read-only in the GUI.

### Align peak retention times to historical space

Once peak selections are approved, run the Step 3 cells. The history cell lists every
annotation run with its dates; set `ANNOTATION_RUN_ID` to the approved run. Step 3
corrects the measured RT peaks back to historical RT space (based on the QC atlas
in the config for each chromatography) and creates **RT alignment run** M in
`annotation_run_N/rt_alignment_M/`. Review the RT alignment model and summary figure
and the output table to ensure Aligned RT Peaks are as expected. If the model needs
changes (for example, removing an InChIKey in `exclude_inchikeys` or removing a QC
file), rerun Step 3; each rerun creates a new numbered alignment run.

### Deposit MSMS refs to the main metatlas database and upload peaks to the centralized atlas

Finally, run the Step 4 cells with `ANNOTATION_RUN_ID` and `ALIGNMENT_RUN_ID` set to
the approved annotation run and its approved alignment run. Run first with `DRY_RUN = True` to write editable previews of the atlas table
and MS/MS references (generated from the project database) to
`rt_alignment_M/deposit_K/`. If needed, edit those files by hand. Then run with
`DRY_RUN = False`: the edited files are uploaded/deposited exactly as they are, and
the edits are recorded in the project database.

## 1. Before you start

Complete [initial setup](initial_setup.md) of the metatlas environment on NERSC, 
including `METATLAS_DATA_DIR`, the script PATH, and a metatlas2 Jupyter kernel. 
Use the same software version, data directory, and analyst account throughout the 
workflow. Project paths depend on the current analyst, not the `owner` in the config 
(which is used only for finding raw data).

## 2. Prepare the input CSV

Use the [example input CSV](/global/cfs/cdirs/metatlas/projects/reference_standards_annotation/reference_standards_inputs/example_standards_input.csv)
as a template. Supply one row per compound and standards sample + experiment;
use separate rows for different chromatographies or RT windows. Use consistent
compound names and structures across rows for the same compound. **Important**: the `group_name` 
column must exactly match the 15th underscore-separated target filename field.

| Column | What to enter |
|---|---|
| `label` | The name used in review and exported as `Compound Label` in the output tables.
| `compound_name` | An optional secondary name that is stored and exported as `Compound Name` in the output tables but not used during annotation. |
| `smiles`, `inchi`, `inchi_key` | Supply a valid SMILES (preferred) or InChI where possible. If only an InChIKey is available, the workflow attempts a PubChem lookup. Missing InChI/InChIKey can be derived from the structural information in the SMILES. |
| `project_name` | The experiment folder name on disk for locating raw files, not a full path. The workflow searches `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/*.h5`. |
| `group_name` | The sample name parsed from the HDF5 filename, matched case-insensitively. It is not a substring search - the group in the input CSV must match the filename field exactly. |
| `rt_min`, `rt_max` | Optional: The expected extraction window in minutes. Blank or nonnumeric values default to min or max TIC values from the files, respectively. |
| `annotation_notes` | Optional: Existing notes for this input row for reference, carried through to the GUI, summaries, and atlases. |

### Rules and checks for expanding the input CSV

Quote CSV fields containing commas. Check structures and names carefully;
the workflow requires a resolvable InChI and InChIKey for each input row.
It desalts the structure by keeping its largest fragment and neutralizes
charges using RDKit and custom cleaning rules. If the resulting InChIKey after
cleaning differs from the supplied or derived key an error will be raised. If so, 
provide a matching neutral, desalted structure and key. If both SMILES and
InChI are supplied, both are checked against the key. An InChIKey alone
requires a successful PubChem structure lookup for this check.
Rows whose exact mass cannot be determined are skipped with a warning.
Rows with no matching HDF5 files cause an error. File names determine
chromatography, polarity, run, and collision-energy (CE) labels; use the
standard filename conventions. For each compound/chromatography/polarity,
one matching file is allowed with a warning, two are expected, and more than
two cause an error. Summary export expects CE labels `CE102040` and
`CE205060` to appear in the appropriate filename field.

## 3. Prepare the YAML config

Copy and edit the [example standards config](/global/cfs/cdirs/metatlas/projects/reference_standards_annotation/reference_standards_inputs/example_standards_annotation.yaml).
At minimum, `PROJECT.standards_run_name` and `PROJECT.standards_input_file`
are required. **Important**: make sure to use the absolute path to the correect input CSV file 
you created in Step 2.

```yaml
PROJECT:
  standards_run_name: my_standards_run
  standards_input_file: /absolute/path/to/standards_input.csv
  owner: jgi # the raw_data subdirectory for locating .h5 files
  chem_source: "Cambridge" # Vendor or mix name, or conventionally 'individual standard'
  include_polarities: [POS, NEG]
  include_adducts: ["[M+H]+", "[M+Na]+", "[M-H]-"] # use matchms-compatible adducts

EXTRACTION:
  ms1_mz_tolerance_ppm: 2.0
  ms1_min_num_points: 1
  ms1_min_peak_intensity: 0
  ms2_min_num_scans: 0
  ms2_min_precursor_intensity: 0
  max_workers: 4
  peak_finding:
    gaussian_sigma: 2.5
    min_height_fraction: 0.15
    min_prominence_fraction: 0.10
    min_distance: 5
    min_width: 0
    prominence_snr: 5.0

TAGS:
  atlas: [EMA, my_specific_tag] # tag that will be applied to all compounds in centralized atlas (editable before upload) 

GDRIVE:
  root_folder_id: 0B-ZDcHbPi-aqZzE5V3hOZFc0dms # do not change

RT_ALIGNMENT:
  C18:
    ATLAS:
      tags: [QCv7] # determines the compounds in centralized atlas used for reverse RT alignment
      uid: # alternatively, use a custom-generated uid in the metatlas database (created with add-atlases command)
      polarity: POS # do not change
    model_type: polynomial
    poly_degree: 2
    rt_peak_source: median
    min_compounds_for_modeling: 3
    min_observations_per_compound: 1
    exclude_inchikeys: []
    mz_tolerance_ppm: 2.0
    rt_window_padding: 1.5
```

Replace the example values with those appropriate to the experiment. Add an
`RT_ALIGNMENT` section for **every** chromatography in the input CSV, for example
`HILIC` as well as `C18`. For each `ATLAS`, specify exactly one of `tags`
(a nonempty list of tags or a single tag) or `uid` (an existing QC atlas UID
in the main database), and set its `polarity` (for JGI, always "POS").
Tags select from the centralized atlas sheet for that chromatography and polarity
(one tag must match). Tag selection requires sheet access and creates or reuses
a main-database QC atlas. UID selection does not require sheet access and
validates the atlas chromatography, polarity, and category. Chromatography
keys are case-insensitive.

`PROJECT.chem_source` is an optional string written to every deposited atlas
row that records where the reference standard molecules were obtained. Typically, 
this would be the vendor or PI. If omitted, it is blank. The deposit command reads 
its value from the config at deposit time, but they can be modified manually in the 
atlas preview before uploading.

`TAGS.atlas` is the list of tags written to every deposited atlas row. Note: 
it is separate from the QC atlas selector `RT_ALIGNMENT.<chrom>.ATLAS.tags`.
If `TAGS` is omitted, the default is `[EMA]`. Tags must be nonempty strings and
**cannot** contain semicolons. The deposit command reads the tags from the config 
at deposit time, but they can be modified manually in the atlas preview before uploading.

The `GDRIVE.root_folder_id` parameter is pre-set to the ID of the shared
`JGI_Metabolomics_Projects` folder. The summary, alignment, and deposit stages each 
upload their corresponding outputs to this Google Drive folder with the following 
hierarchy:

```text
Reference_Standards_Annotation/
  <standards_run_name>/
    annotation_run_<N>/
      StandardsAnnotation_Summary_<standards_run_name>.csv
      summary_figures/
      rt_alignment_<M>/
        StandardsAnnotation_Summary_Aligned_<standards_run_name>.csv
        alignment_config.json
        model_figures/
        deposit_<K>/
          Atlas_Deposit_<standards_run_name>.csv
          MsmsRefs_Deposit_<standards_run_name>.tab
          Atlas_Deposit_<standards_run_name>.final.csv
          MsmsRefs_Deposit_<standards_run_name>.final.tab
          deposit_changes.csv
```

Other config parameter details:

* `owner` selects the raw data directory and defaults to `jgi`.
* `include_polarities` defaults to both `POS` and `NEG`.
* `include_adducts` defaults to the built-in list if omitted; unknown or 
  polarity-incompatible adducts are not used.
* The extraction m/z tolerance applies to MS1 and MS2 **precursor** matching.
  The minimum MS1 point/intensity and MS2 scan/precursor-intensity settings
  filter extraction targets. With both MS2 minimums at zero, all matching scans
  are retained; active MS2 filters apply in-feature criteria. `max_workers`
  controls extraction parallelism.
* `peak_finding` controls smoothing and candidate height, prominence,
  scan-point distance/width, and signal-to-noise thresholds. These automated peaks 
  are the candidates that will be reviewed in the GUI, not final annotations.
* For each chromatography, `model_type` is `polynomial`, `linear`, or
  `median_offset`; `poly_degree` applies to polynomial fits. `rt_peak_source`
  is `min`, `median`, or `max` across QC files, or an exact QC filename.
  `min_compounds_for_modeling` sets the minimum QC targets; for composite
  sources, `min_observations_per_compound` sets the minimum file observations
  per target. `exclude_inchikeys` removes QC compounds from fitting.
  `mz_tolerance_ppm` and `rt_window_padding` control QC extraction.

## 4. Start extraction

```bash
metatlas2.sh standards-annotation --config_path /absolute/path/to/standards_config.yaml
```

The command creates a new project directory, project DuckDB, log, copies of the
config YAML and input CSV, expanded target table, and review notebook. It refuses to overwrite an existing
directory, including a failed setup or one in progress. **Important**: To resume 
an existing annotation, reopen the standards run notebook that was created, **do not** 
rerun this command. To start over, choose a new `standards_run_name`, or deliberately 
remove the old project only after preserving any work you need.

Here is what the outputs written to disk after running the command should look like:

```text
$METATLAS_DATA_DIR/projects/reference_standards_annotation/<analyst>/<standards_run_name>/
  <standards_run_name>.duckdb
  <standards_config>.yaml            # copy of the config passed to --config_path
  <standards_input_file>.csv         # copy of project.standards_input_file
  <standards_input_file>_expanded.csv
  StandardsAnnotation_<standards_run_name>.ipynb
  logs/standards_annotation.log
  outputs/
```

Check the log and `<standards_input_file>_expanded.csv` for skipped compounds, 
unmatched files, adducts, extraction errors, and expected coverage before reviewing.
Extraction errors for individual polarity/chromatography groups are logged
and processing can continue for other groups; a generated notebook alone does
not prove all groups succeeded.

The workflow expands each input row across matched files and valid adducts,
then extracts EICs and MS2 scans. It detects candidate peaks on smoothed EICs
and reports raw peak values. RT min and max bounds are peak +/- 0.5 min. 
For each compound/chromatography/polarity, the file (i.e., the CE) with
the highest detected MS1 peak is used for **all** adducts. If no peak is
detected, the file with the highest raw intensity is used in the GUI.
Extracted MS1 rows for both CE files remain in the project database.

For each peak and adduct, the closest MS2 scan (by RT) is associated separately
from each available file. A scan may fall outside the peak bounds or be
associated with multiple peaks. Missing scans are warned about. The displayed
TIC is the sum of extracted target windows, not the instrument's full TIC;
the closest injection blank by run number provides background context and is
not used for peak detection.

## 5. Review and save annotations

Open the generated notebook in JupyterHub with its `metatlas2 (latest)` kernel and run
the cells up through the GUI cell. The notebook reloads the project database;
its generated `RUN_PARAMS` should not need editing. The Google Drive upload is
automatically turned on (`True`) for the summary, alignment, or deposit steps; make 
sure to manually turn to `False` to skip that stage's Drive upload.

### Annotation runs

Each annotation iteration is a numbered **annotation run** stored in the project
database. A run is a *draft* while you edit it in the GUI and becomes *frozen*
when it is summarized (Step 2); frozen runs never change (at most one draft
exists at a time). The history cell above the GUI lists every run with its
status, dates, number of selected peaks, and output directory.

* `ANNOTATION_RUN_ID = None` (default): the GUI edits the draft run. If there is
  no draft, it starts one: run 1 starts empty and later runs start as a copy of
  the latest frozen run. **Important**: you will probably always want to just leave
  this as `None` - see explanation below.
* If you want to view the peaks selected for an existing frozen run, set 
  `ANNOTATION_RUN_ID = N`. The GUI shows that run exactly as it
  was summarized and it is **read-only** (checkboxes, notes, and Add are disabled and
  nothing is saved). As stated above, to update an existing annotation run that you've
  already summarized (i.e., finished annotating and sent for review), simply leave the
  `ANNOTATION_RUN_ID` as `None` and a full copy of the frozen run will be created for you
  to edit. This ensures that the subsequent summary that is created will make a new
  directory and fresh database entries.
* If you need to continue from an earlier run instead of the latest one, you'll need to run
  `start_new_annotation_run(RUN_PARAMS['project_db_path'], from_annotation_run_id=N)`
  in a cell, then open the GUI with `ANNOTATION_RUN_ID = None`. If an
  unsummarized draft exists, add `discard_draft=True` to discard it (and the
  manual peaks added in it). `discard_draft_annotation_run(...)` discards a
  draft without starting a new one.

### Using the GUI

1. For each compound, examine each chromatography and polarity, which have EICs and
   peak markers, the target-compound TIC, nearest blank, and MS2 Scans
   (when available). Use the linear/log control as needed. Checklist labels
   include the chosen MS1 run and CE. MS2 spectra use two CE columns; a
   missing scan appears as a placeholder. EIC MS2 markers use the scan RT and
   the nearest EIC point's intensity, not precursor intensity (for readability).
2. Select all acceptable peaks in the positive and negative checklists. For
   each compound/chromatography/polarity, select at least one peak **or**
   select **No good peaks**; the latter excludes all peak selections for that
   group across adducts. A no-good decision is not exported as a peak. This stage
   is what determines the values (compound/chromatography/polarity/adduct(s)/peak(s))
   that will be deposited to the centralized atlas.
3. If a suitable peak was missed, choose its **Adduct** and an EIC **RT peak**, then
   click **Add**. The manual peak uses that MS1 file, initial
   bounds of +/- 0.1 min (lower bound no less than zero), and the same
   closest-MS2 rule. Review its selection and spectra. Manual peaks belong to
   the draft run they were added in and remain available to later runs.
4. Enter analyst notes, which are recorded into the reviewable summaries and appended
   to the identification notes (in the input csv) when the atlas rows are deposited. 
5. Use **Previous**, **Next**, or the compound dropdown to move between compounds.
   A navigation action saves decisions and notes to the database. Click
   **Save & Exit** when finished with the GUI. Reopen the notebook and GUI to resume the
   draft from its saved checkpoint (remember, it's still a draft and editable until
   the summary cell has been run).

## 6. Generate and approve the summary

After **Save & Exit**, run the notebook's summary cell. It summarizes the
**draft** annotation run N (it fails if there is no draft). Every group
must have a decision and at least one actual peak must be selected. The cell
creates:

```text
outputs/annotation_run_<N>/
  StandardsAnnotation_Summary_<standards_run_name>.csv
  summary_figures/
```

Outputs are staged and published only when complete, at which point the run 
is then frozen and its output directory recorded in the database. If summary 
generation fails, nothing is published and the run stays a draft. In that case,
review the notebook output or logs to see what caused the failure.

By default, the summary uploads the annotation run directory to Google Drive and
records the link. Set `UPLOAD_SUMMARY_TO_GDRIVE = False` in the summary
parameters cell to skip this upload. **Impoortant:** Look at the generated 
summary CSV and PDFs and share the outputs with the reviewer before proceeding to 
the next notebook cell and Step 7. If changes are requested, rerun the noteook until
the GUI cell (keep `ANNOTATION_RUN_ID` as `None` so a new draft run N+1 is 
started from the latest summarized run) and do the annotation and summary again.

### Summary outputs

The CSV contains one row per selected peak, notes, MS1 source, and separate
CE102040/CE205060 MS2 fields. Spectra are `[[m/z values], [intensities]]`;
missing scans leave empty fields. If multiple peaks were selected per adduct,
they are renumbered from what was shown in GUI (i.e., the results from the
automated peak finding algorithm) starting with index 1 and ascending by RT
within each chromatography/polarity/compound/source/adduct group. `GUI Peak Num` 
preserves the original GUI peak number for reference. Summary generation does
not renumber database peaks, it happens on the fly each time.

Before creating outputs, the summary compares selected adducts for candidate
isomers within each chromatography and polarity. Compounds are isomers
when their InChIKeys share the first block or any of their theoretical target
m/z values (including unselected adducts) are within 5 ppm.

The summary also warns when RTs of selected adduct differ by more than 5%
within a compound/chromatography/polarity, or when the median RT of all selected
adducts differs by more than 5% between polarities for a compound/chromatography.
If multiple peaks are selected for one or more adducts for a compound, the detection
is a little trickier: adduct peaks are matched one-to-one to nearest-RT buckets
(starting with the adduct with the most peaks); polarity bucket medians are
likewise matched one-to-one. Unmatched peaks remain separate and are not
compared to other buckets.

## 7. Reverse RT alignment and tags

Reverse alignment models **measured QC RT to historical QC atlas RT** (one
per chromatography), then applies it to selected standards in both
polarities. The QC files must be in the same experiment directory as the
input files. The model runs on the RT peak, min, and max to shift the entire
window.

Run the history cell, set `ANNOTATION_RUN_ID` in the alignment parameters cell
to the **approved frozen** annotation run, and run the alignment cell.
Alignment always uses that run's saved (i.e., reviewed) selections, even if a newer draft
exists (a warning is logged). It writes **RT alignment run** M, numbered across
the whole project:

* `annotation_run_<N>/rt_alignment_<M>/StandardsAnnotation_Summary_Aligned_<standards_run_name>.csv`
  has aligned peak/min/max columns, signed `RT diff` (aligned minus measured
  peak), and the config's `TAGS.atlas`.
* `model_figures/` contains per-chromatography model PDFs and
  `alignment_config.json` is a snapshot of the loaded settings.
* The project database records the run with each chromatography's model
  (type, coefficients/offset, R², RMSE, equation, QC atlas UID and selector,
  settings, and the QC filenames used), its QC points (measured vs. atlas RT,
  prediction, residual), and every aligned peak.

Inspect the QC points, fit, residuals, R², RMSE, aligned RTs, and bounds.
Missing QC data, insufficient observations, or an unusable model cause
alignment to stop and nothing is published or recorded. Adjust the config (for
example `exclude_inchikeys`) or QC files and rerun to create a new alignment
run (earlier runs are kept for reference). If intentionally skipping alignment is
appropriate, set `SKIP_RT_ALIGNMENT = True`; aligned RTs then equal measured
RTs and no model is fitted. The alignment directory is uploaded into the same
Drive annotation folder by default; set `UPLOAD_ALIGNMENT_TO_GDRIVE = False`
to skip this upload.

## 8. Preview, edit, and deposit the atlas and references

Run the history cell and set both `ANNOTATION_RUN_ID` and `ALIGNMENT_RUN_ID`
in the deposit parameters cell to the **approved** annotation run and the
**approved** alignment run. Alignment run numbers are unique across the project;
the deposit stops if the alignment run did not align that annotation run, which
guards against choosing the wrong run.

**Important**: The `DRY_RUN` variable should be set to `True` for the first run
of the deposit cell. This will create the atlas and MSMS refs outputs on disk to
review and/or edit, and a subsequent run of the same cell with `DRY_RUN` set to
`False` will pick these files up from disk and that is what will be uploaded to
the centralized atlas and deposited to the MSMS refs database table (see details
below).

### Dry run: previews generated from the database

`DRY_RUN = True` creates deposit K in
`annotation_run_<N>/rt_alignment_<M>/deposit_<K>/` and, by default, uploads it
to Drive for review. **Nothing is deposited**. All content comes from the project
database (aligned peaks, structures, the annotation run's notes, input annotation
notes, the alignment run's tags, and the selected peaks' MS2 scans). After a dry run,
two output files will be available for review:

* `Atlas_Deposit_<standards_run_name>.csv`: the centralized atlas columns
  (aligned RTs, theoretical m/z, 5 ppm m/z tolerance, tags, preparation dates,
  combined analyst/input notes, `PROJECT.chem_source` from the config, and
  formula derived from the structure) followed by `sa_entry_id` and
  `sa_peak_uid` (the sa_* columns can be ignored, they are used to updated
  the database after an upload in case a row(s) in the table(s) are manually
  modified)/
* `MsmsRefs_Deposit_<standards_run_name>.tab`: one formatted row per
  available selected peak/CE spectrum, followed by `sa_entry_id`,
  `sa_peak_uid`, `sa_ce_label`, and `sa_measured_collision_energy`. Missing or
  empty spectra for selected peaks are warned about and omitted. The `database` 
  column is always set to `metatlas`, the `id` column is a unique 32-character
  ID (stable across later previews of the same alignment run), and `name` is the
  atlas compound label. The MS2 scan filename supplies the instrument and 
  normalized CE ramp based on filename fields (e.g., `CE102040norm` becomes
  `ramp-102040norm`). A malformed or unsupported filename stops preparation.

### Editing the previews

Inspect structures, formula, exact mass, metadata, spectra, and counts. You
may edit any value, delete rows, or add rows (leave `sa_entry_id` blank for new
rows; a blank MS/MS `id` gets a new ID). Do not rename or remove columns. The
`sa_*` columns link rows to the database and are removed before upload and
import; `sa_measured_collision_energy` is stored in the main database's numeric
`collision_energy` column. Edits to the preview files on disk will automatically
be applied to the reference standards database for this project and to the data
uploaded to the centralized atlas and MSMS refs database table.

### Deposit

Set `DRY_RUN = False` and rerun the parameters and deposit cells. The latest
preview of the alignment run is used (pass `deposit_id=K` to the deposit call
to choose a different stored preview from the latest one). The cell:

1. Reads the preview files back and validates them (columns, required values,
   polarity, numeric RTs with `rt_min <= rt_peak <= rt_max`, spectra, unique
   IDs and `sa_entry_id`s). Invalid files stop the deposit with nothing
   changed.
2. Compares them to the generated preview and records every edited value,
   added row, and removed row in the project database and in
   `deposit_changes.csv`.
3. Writes the edits back to the alignment run's deposit content
   (`sa_atlas_entries` / `sa_msms_entries`: edited rows updated, added rows
   stored as `manual`, deleted rows marked excluded), snapshots the exact rows,
   and writes `*.final.csv` / `*.final.tab` (without `sa_*` columns).
4. Imports compounds from the final atlas file, appends the final atlas rows to
   the centralized atlas Google Sheet (then sorts the whole sheet by ascending
   RT peak), and imports the final MS/MS file, recording for each row
   whether it was appended or skipped as a duplicate, and whether each
   spectrum was inserted (with its main-database `ref_uid`), skipped as an
   existing InChIKey/adduct pair, or filtered by the importer. See below for
   variables that can be used when writing final entries.

`SKIP_ATLAS_UPLOAD = True` or `SKIP_MSMS_REFS_DEPOSIT = True` skips a stage;
the skip is recorded and rerunning with the flag cleared performs that stage
for the same deposit. If a stage fails, rerun with `DRY_RUN = False`: completed
stages are not repeated. Once a deposit has started, its preview files must
not change (the deposit refuses changed files).

The atlas deposit checks that the `complete_atlas` worksheet header matches
the expected schema. Rows with duplicate `(label, inchi_key, adduct, chromatography, polarity)`
identities are warned about and skipped, including duplicates in the input, or
set `FORCE_ADD = True` and rerun the parameters cell to append them anyway. The entire
data portion is then sorted by `rt_peak`. Append and sort are separate requests. If
sorting fails or an append response is lost, inspect the sheet before retrying
to avoid duplicate rows. Confirm the new rows and values in the sheet.

### Project database tables

The project database (`<standards_run_name>.duckdb`) can be queried directly
with DuckDB. See [database schema](database_schema.md#standards-annotation-project-database)
for the full table list. Useful tables: `sa_annotation_runs` and
`sa_run_selections` (review history), `sa_alignment_runs`,
`sa_alignment_models`, `sa_alignment_qc_points`, `sa_aligned_peaks`
(alignment history), `sa_atlas_entries` / `sa_msms_entries` (current deposit
content per alignment run), and `sa_deposits`, `sa_deposit_atlas_rows`,
`sa_deposit_msms_rows`, `sa_deposit_changes` (what was previewed, what was
deposited, and how it was edited).

## Troubleshooting and recordkeeping

* No files: check `owner`, experiment folder, exact parsed sample name,
  polarity, and filename conventions. More than two matching files per group
  must be resolved before peak selection.
* Missing compounds: check `standards_annotation.log` for invalid structures,
  PubChem lookup failures, skipped masses, extraction errors, and invalid
  adducts. Check the expanded table against the CSV.
* Incomplete review: reopen the GUI, resolve every group, save, and rerun the
  summary. A summary requires at least one selected peak overall.
* "No draft annotation run to summarize": the latest run is already frozen.
  Open the GUI to start a new run, or use the existing run's outputs.
* Failed alignment: check the chromatography's QC atlas UID, matching QC
  files/polarity, extraction window and tolerance, and minimum observations.
  Review diagnostic PDFs rather than assuming a fitted model is acceptable.
* Failed Drive upload: fix rclone configuration or access. Local outputs
  remain.