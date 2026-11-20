# Standards annotation: analyst workflow

This guide covers preparing reference standards, reviewing their peaks,
aligning retention times (RTs), appending atlas rows to Google Sheets, and
importing compounds and MS/MS references into the main database. RTs are in **minutes**.

## 0. Quickstart guide

On a NERSC Perlmutter login node or compute node, run the following command
(from anywhere in the filesystem).

```bash
metatlas2.sh standards-annotation --config_path /absolute/path/to/standards_config.yaml
```

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

Once peak selections are approved, run the Step 3 cells. The history cell lists every
annotation run with its dates; set `ANNOTATION_RUN_ID` to the approved run. Step 3
corrects the measured RT peaks back to historical RT space (based on the QC atlas
in the config for each chromatography) and creates **RT alignment run** M in
`annotation_run_N/rt_alignment_M/`. Review the RT alignment model and summary figure
and the output table to ensure Aligned RT Peaks are as expected. If the model needs
changes (for example, removing an InChIKey in `exclude_inchikeys` or removing a QC
file), rerun Step 3; each rerun creates a new numbered alignment run.

Finally, run the Step 4 cells with `ANNOTATION_RUN_ID` and `ALIGNMENT_RUN_ID` set to
the approved annotation run and its approved alignment run. Run first with `DRY_RUN = True` to write editable previews of the atlas table
and MS/MS references (generated from the project database) to
`rt_alignment_M/deposit_K/`. If needed, edit those files by hand. Then run with
`DRY_RUN = False`: the edited files are uploaded/deposited exactly as they are, and
the edits are recorded in the project database.

## 1. Before you start

Complete [initial setup](initial_setup.md), including `METATLAS_DATA_DIR`, the
script PATH, and a metatlas2 Jupyter kernel. Use the same software version,
data directory, and analyst account throughout the workflow. Project paths
depend on the current analyst, not the `owner` in the config.

You need:

* Accessible standards and QC `.h5` files with metatlas-compatible filenames.
* A main database at `$METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb`,
  including the appropriate historical QC atlases for RT alignment. The
  separate project database created below stores extraction and review.
* Write access to the project output location and, for MS/MS import, to the
  main database. For sheet deposition, the configured Google service account
  must have edit access to the production atlas sheet.
* An rclone `drive` remote rooted at the shared `JGI_Metabolomics_Projects`
  folder, with write access to `Reference_Standards_Annotation`. The analyst
  and supervisor also need access to the uploaded folder. Configure rclone
  for the account and environment used by the notebook and commands;
  see [rclone setup](run_anywhere.md#google-drive-upload).
* A CSV and YAML config prepared as described below. Keep both accessible
  throughout the workflow: the notebook and later steps reload the config.

On Docker, keep the config and referenced input files under
`$METATLAS_DATA_DIR` so the container can access them. The Docker wrapper may
download the main database from Zenodo when its version stamp is missing or
changed; coordinate with the database administrator before modifying a
production database or relying on a local copy to persist.

## 2. Prepare the input CSV

Use the [example input CSV](../configs/example_configs/standards_annotation/example_standards_input.csv)
as a template. Supply one row per compound and standards sample/experiment;
use separate rows for different chromatographies or RT windows. Use consistent
compound names and structures across rows for the same compound.

| Column | What to enter |
|---|---|
| `label` | The name used in review and exported as `Compound Label` in the output tables.
| `compound_name` | An optional secondary name that is stored and exported as `Compound Name` in the output tables but not used during annotation. |
| `smiles`, `inchi`, `inchi_key` | Supply a valid SMILES (preferred) or InChI where possible. If only an InChIKey is available, the workflow attempts a PubChem lookup. Missing InChI/InChIKey can be derived from a structure. |
| `project_name` | The experiment folder name for locating raw files, not a full path. The workflow searches `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/*.h5`. |
| `group_name` | The sample name parsed from the HDF5 filename, matched case-insensitively. It is not a substring search - the group in the input CSV must match the filename field exactly. |
| `rt_min`, `rt_max` | Optional: The expected extraction window in minutes. Blank or nonnumeric values default to min or max TIC values, respectively. |
| `annotation_notes` | Optional: Existing notes for this input row, carried through to the GUI, summaries, and atlases. |

### Rules and checks for expanding the input CSV

Quote CSV fields containing commas. Check structures and names carefully:
the workflow requires a resolvable InChI and InChIKey for each input row.
It desalts the structure by keeping its largest fragment and neutralizes
charges using RDKit and custom cleaning rules. If the resulting InChIKey after
cleaning differs from the supplied or derived key and error will be raised;
provide a matching neutral, desalted structure and key. If both SMILES and
InChI are supplied, both are checked against the key. An InChIKey alone
requires a successful PubChem structure lookup for this check.
Rows whose exact mass cannot be determined are skipped with a warning.
Rows with no matching HDF5 files cause an error. File names determine
chromatography, polarity, run, and collision-energy (CE) labels; use the
standard filename conventions. For each compound/chromatography/polarity,
one matching file is allowed with a warning, two are expected, and more than
two cause an error. Summary export expects CE labels `CE102040` and
`CE205060`.

## 3. Prepare the YAML config

Copy and edit the [example standards config](../configs/example_configs/standards_annotation/example_standards_annotation.yaml).
At minimum, `PROJECT.standards_run_name` and `PROJECT.standards_input_file`
are required; use an accessible absolute CSV path.

```yaml
PROJECT:
  standards_run_name: my_standards_run
  standards_input_file: /absolute/path/to/standards_input.csv
  owner: jgi
  chem_source: "Vendor or mix name, or conventionally 'individual standard'"
  include_polarities: [POS, NEG]
  include_adducts: ["[M+H]+", "[M+Na]+", "[M-H]-"]

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
  atlas: [EMA, my_specific_tag]

GDRIVE:
  root_folder_id: 0B-ZDcHbPi-aqZzE5V3hOZFc0dms

RT_ALIGNMENT:
  C18:
    ATLAS:
      tags: [QCv7]
      uid: # Alternatively, use uid: <existing_qc_atlas_uid>
      polarity: POS
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
`RT_ALIGNMENT` entry for **every** selected chromatography, for example
`HILIC` as well as `C18`. For each `ATLAS`, specify exactly one of `tags`
(a nonempty list of tags or a single tag) or `uid` (an existing QC atlas UID
in the main database), and set its `polarity` (for JGI, always "POS").
Tags select from the centralized atlas sheet for that chromatography and polarity
(one tag must match). Tag selection requires sheet access and creates or reuses
a main-database QC atlas. UID selection does not require sheet access and
validates the atlas chromatography, polarity, and category. Chromatography
keys are case-insensitive. The alignment config snapshot records the resolved
UID as `resolved_qc_atlas_uid` alongside the original selector.
Replace the former `qc_atlas_uid` setting with `ATLAS.uid` and `ATLAS.polarity`
when updating an older config. These settings are only needed when alignment
is not skipped.

`PROJECT.chem_source` is an optional string written to every deposited atlas
row. If omitted, it is blank. The deposit command reads its value from the
config at deposit time; check the atlas preview before uploading.

`TAGS.atlas` is the list of tags written to every deposited atlas row.
It is separate from the QC atlas selector `RT_ALIGNMENT.<chrom>.ATLAS.tags`.
If `TAGS` is omitted, the default is `[EMA]`. Entries must be nonempty and
cannot contain semicolons. Change the tags in the config before alignment;
they are saved in the aligned summary and its config snapshot.

Set `GDRIVE.root_folder_id` to the ID in the URL of the shared
`JGI_Metabolomics_Projects` folder. The folder hierarchy under that root is:

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

The summary, alignment, and deposit stages each upload their corresponding
outputs. Upload errors are reported and stop the stage; local files remain.

* `owner` selects the raw data directory and defaults to `jgi`.
  `include_polarities` defaults to both `POS` and `NEG`. `include_adducts`
  defaults to the built-in list if omitted; unknown or polarity-incompatible
  adducts are not used.
* The extraction m/z tolerance applies to MS1 and MS2 **precursor** matching.
  The minimum MS1 point/intensity and MS2 scan/precursor-intensity settings
  filter extraction targets. With both MS2 minimums at zero, matching scans
  are retained; active MS2 filters apply in-feature criteria. `max_workers`
  controls extraction parallelism.
* `peak_finding` controls smoothing and candidate height, prominence,
  scan-point distance/width, and signal-to-noise thresholds. Review its
  choices; automated peaks are candidates, not final annotations.
* For each chromatography, `model_type` is `polynomial`, `linear`, or
  `median_offset`; `poly_degree` applies to polynomial fits. `rt_peak_source`
  is `min`, `median`, or `max` across QC files, or an exact QC filename.
  `min_compounds_for_modeling` sets the minimum QC targets; for composite
  sources, `min_observations_per_compound` sets the minimum file observations
  per target. `exclude_inchikeys` removes QC compounds from fitting.
  `mz_tolerance_ppm` and `rt_window_padding` control QC extraction.

The example also contains `GUI` dimensions and `ms2_top_n_per_peak`; the
current standards workflow does not use these settings to size its GUI or
limit the MS2 scans associated with peaks.

## 4. Start extraction

```bash
metatlas2.sh standards-annotation --config_path /absolute/path/to/standards_config.yaml
```

The command creates a new project directory, project DuckDB, log, expanded
target table, and review notebook. It refuses to overwrite an existing
directory, including one left by a failed setup. To resume an existing review,
reopen its notebook; do not rerun this command. To start over, choose a new
`standards_run_name`, or deliberately remove the old project only after
preserving any work you need.

```text
$METATLAS_DATA_DIR/projects/reference_standards_annotation/<analyst>/<standards_run_name>/
  <standards_run_name>.duckdb
  full_search_table.csv
  StandardsAnnotation_<standards_run_name>.ipynb
  logs/standards_annotation.log
  outputs/
```

Check the log and `full_search_table.csv` for skipped compounds, unmatched
files, adducts, extraction errors, and expected coverage before reviewing.
Extraction errors for individual polarity/chromatography groups are logged
and processing can continue for other groups; a generated notebook alone does
not prove all groups succeeded.

The workflow expands each input row across matched files and valid adducts,
then extracts EICs and MS2 scans. It detects candidate peaks on smoothed EICs
and reports raw apex values; automatic bounds are apex +/- 0.5 min, clamped
to the EIC extent. For each compound/chromatography/polarity, the file with
the highest detected MS1 peak is used for **all** adducts. If no peak is
detected, the file with the highest raw intensity is used for manual review.
Extracted MS1 rows for both CE files remain in the project database.

For each peak and adduct, the closest MS2 scan in RT is associated separately
from each available file. A scan may fall outside the peak bounds or be
associated with multiple peaks. Missing scans are warned about. The displayed
TIC is the sum of extracted target windows, not the instrument's full TIC;
the closest injection blank by run number provides background context and is
not used for peak detection.

## 5. Review and save annotations

Open the generated notebook in JupyterHub with its `metatlas2 (latest)` kernel and run
the cells up through the GUI cell. The notebook reloads the project database;
its generated `RUN_PARAMS` should not need editing. Pass `upload_to_gdrive=False`
to a summary, alignment, or deposit call to skip that stage's Drive upload.

### Annotation runs

Each review iteration is a numbered **annotation run** stored in the project
database. A run is a *draft* while you edit it in the GUI and becomes *frozen*
when it is summarized (Step 2); frozen runs never change. At most one draft
exists at a time. The history cell above the GUI lists every run with its
status, dates, number of selected peaks, and output directory.

* `ANNOTATION_RUN_ID = None` (default): the GUI edits the draft run. If there is
  no draft, it starts one: run 1 starts empty and later runs start as a copy of
  the latest frozen run.
* `ANNOTATION_RUN_ID = N` for a frozen run: the GUI shows that run exactly as it
  was summarized, **read-only** (checkboxes, notes, and Add are disabled and
  nothing is saved). Manual peaks added in later runs are hidden.
* To continue from an earlier run instead of the latest one, run
  `start_new_annotation_run(RUN_PARAMS['project_db_path'], from_annotation_run_id=N)`
  in a cell, then open the GUI with `ANNOTATION_RUN_ID = None`. If an
  unsummarized draft exists, add `discard_draft=True` to discard it (and the
  manual peaks added in it). `discard_draft_annotation_run(...)` discards a
  draft without starting a new one.

### Using the GUI

1. For each compound, examine each chromatography and polarity: EICs and
   candidate markers, the target-compound TIC, blank, and **Show MS2 Scans**
   where available. Use the linear/log control as needed. Checklist labels
   include the chosen MS1 run and CE. MS2 spectra use two CE columns; a
   missing scan appears as a placeholder. EIC MS2 markers use the scan RT and
   the nearest EIC point's intensity, not precursor intensity.
2. Select all acceptable peaks in the positive and negative checklists. For
   each compound/chromatography/polarity, select at least one peak **or**
   select **No good peaks**; the latter excludes all peak selections for that
   group across adducts. A no-good decision is not exported as a peak.
3. If a suitable peak was missed, choose its **Adduct** and an EIC **RT
   peak**, then click **Add**. The manual peak uses that MS1 file, initial
   bounds of +/- 0.1 min (lower bound no less than zero), and the same
   closest-MS2 rule. Review its selection and spectra. Manual peaks belong to
   the draft run they were added in and remain available to later runs.
4. Enter analyst notes. Use **Previous**, **Next**, or the compound dropdown
   to navigate; navigation saves decisions and notes to the draft run. Click
   **Save & Exit** when finished. Reopen the notebook and GUI to resume the
   draft from its saved checkpoint.

## 6. Generate and approve the summary

After **Save & Exit**, run the notebook's summary cell. It summarizes the
**draft** annotation run N (it fails if there is no draft). Every target group
must have a decision and at least one actual peak must be selected. The cell
creates:

```text
outputs/annotation_run_<N>/
  StandardsAnnotation_Summary_<standards_run_name>.csv
  summary_figures/
```

Outputs are staged and published only when complete; the run is then frozen
and its output directory recorded in the database. If summary generation
fails, nothing is published and the run stays a draft.

By default, the cell uploads the annotation run directory to Google Drive and
records the link. Set `UPLOAD_SUMMARY_TO_GDRIVE = False` in the summary
parameters cell to skip this upload. Share the outputs with the supervisor and
review the CSV and PDFs before proceeding. If changes are requested, reopen the
GUI (a new draft run N+1 is started from run N) and summarize again.

The CSV contains one row per selected peak, notes, MS1 source, and separate
CE102040/CE205060 MS2 fields. Spectra are `[[m/z values], [intensities]]`;
missing scans leave empty fields. Selected peaks are numbered by measured RT
within each chromatography/polarity/compound/source/adduct group; `GUI Peak
Num` preserves the original GUI number. Summary generation does not renumber
database peaks.

Before creating outputs, the summary compares selected adducts for candidate
isomers within each chromatography and polarity. Compounds are candidates
when their InChIKeys share the first block or any of their theoretical target
m/z values (including unselected adducts) are within 5 ppm. If they have no
selected adduct in common, summary generation stops; extra adducts on either
compound produce warnings.

The summary also warns when selected adduct RTs differ by more than 5% for a
compound/chromatography/polarity, or when the median selected-adduct RT
differs by more than 5% between polarities for a compound/chromatography.
The percentage uses the mean of the two compared RTs as its denominator.
For multiple peaks, adduct peaks are matched one-to-one to nearest-RT buckets
(starting with the adduct with the most peaks); polarity bucket medians are
likewise matched one-to-one. Unmatched peaks remain separate and are not
compared to other buckets.

## 7. Reverse RT alignment and tags

Reverse alignment maps **measured QC RT to historical QC atlas RT**, one
model per chromatography, then applies it to selected standards in both
polarities. The QC files must be in the relevant experiment directories and
match the atlas polarity. The aligned peak's RT difference is also added to
its min and max bounds.

Run the history cell, set `ANNOTATION_RUN_ID` in the alignment parameters cell
to the **approved frozen** annotation run, and run the alignment cell.
Alignment always uses that run's saved selections, even if a newer draft
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

Inspect the QC points, fit, residuals, R², RMSE, aligned RTs, bounds, and
tags. Missing QC data, insufficient observations, or an unusable model stop
alignment and nothing is published or recorded. Adjust the config (for
example `exclude_inchikeys`) or QC files and rerun to create a new alignment
run; earlier runs are kept. If intentionally skipping alignment is
appropriate, set `SKIP_RT_ALIGNMENT = True`; aligned RTs then equal measured
RTs and no model is fitted. The alignment directory is uploaded into the same
Drive annotation folder by default; set `UPLOAD_ALIGNMENT_TO_GDRIVE = False`
to skip this upload. Alignment does not itself write an atlas to the main
database.

## 8. Preview, edit, and deposit the atlas and references

Run the history cell and set both `ANNOTATION_RUN_ID` and `ALIGNMENT_RUN_ID`
in the deposit parameters cell to the **approved** annotation run and its
approved alignment run. Alignment run numbers are unique across the project;
the deposit stops if the alignment run did not align that annotation run, which
guards against choosing the wrong run. `DRY_RUN` defaults to `True`.

### Dry run: previews generated from the database

`DRY_RUN = True` creates deposit K in
`annotation_run_<N>/rt_alignment_<M>/deposit_<K>/` and, by default, uploads it
to Drive. Nothing is deposited. All content comes from the project database
(aligned peaks, structures, the annotation run's notes, input annotation notes,
the alignment run's tags, and the selected peaks' MS2 scans); the aligned
summary CSV is a review artifact only.

* `Atlas_Deposit_<standards_run_name>.csv`: the centralized-atlas columns
  (aligned RTs, theoretical m/z, 5 ppm m/z tolerance, tags, preparation dates,
  combined analyst/input notes, `PROJECT.chem_source` from the config, and
  formula derived from the structure) followed by `sa_entry_id` and
  `sa_peak_uid`.
* `MsmsRefs_Deposit_<standards_run_name>.tab`: one importer-format row per
  available selected peak/CE spectrum, followed by `sa_entry_id`,
  `sa_peak_uid`, `sa_ce_label`, and `sa_measured_collision_energy`. Missing or
  empty spectra are warned about and omitted. `database` is `metatlas`, `id`
  is a unique 32-character ID (stable across later previews of the same
  alignment run), and `name` is the atlas compound label. The MS2 scan filename
  supplies the instrument and normalized CE ramp (`CE102040norm` becomes
  `ramp-102040norm`); a malformed or unsupported filename stops preparation.

The first preview of an alignment run stores its rows as that run's deposit
content. Later previews of the same alignment run reuse that content,
including edits written back by an earlier deposit. Each dry run creates a new
deposit and supersedes older, undeposited previews of that alignment run.

### Editing the previews

Inspect structures, formula, exact mass, metadata, spectra, and counts. You
may edit any value, delete rows, or add rows (leave `sa_entry_id` blank for new
rows; a blank MS/MS `id` gets a new ID). Do not rename or remove columns. The
`sa_*` columns link rows to the database and are removed before upload and
import; `sa_measured_collision_energy` is stored in the main database's numeric
`collision_energy` column.

### Deposit

Set `DRY_RUN = False` and rerun the parameters and deposit cells. The latest
preview of the alignment run is used (pass `deposit_id=K` to the deposit call
to choose another unsuperseded preview). The cell:

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
   Google Sheets, and imports the final MS/MS file, recording for each row
   whether it was appended or skipped as a duplicate, and whether each
   spectrum was inserted (with its main-database `ref_uid`), skipped as an
   existing InChIKey/adduct pair, or filtered by the importer.

The project database therefore records exactly what reached the sheet and
`msms_refs`.

`SKIP_ATLAS_UPLOAD = True` or `SKIP_MSMS_REFS_DEPOSIT = True` skips a stage;
the skip is recorded and rerunning with the flag cleared performs that stage
for the same deposit. If a stage fails, rerun with `DRY_RUN = False`: completed
stages are not repeated. Once a deposit has started, its preview files must
not change (the deposit refuses changed files); restore them, or start over
with a new dry run (not possible while a deposit is partially complete).
A deposit that is complete cannot be deposited again; a new dry run is
required, and a warning notes that a project deposit already exists.

For a real deposit, the service account key must be accessible at
`/global/cfs/cdirs/metatlas/.jgi_production_atlas_google_api_key_1d798f188868.json`
and have edit access to the production `lbl_metabolomics_atlas` spreadsheet
unless atlas upload is skipped.
The append operation checks that the `complete_atlas` worksheet header matches
the expected schema, including `chem_source`. Rows
with duplicate `(label, inchi_key, adduct, chromatography, polarity)`
identities are warned about and skipped, including duplicates in the input;
set `FORCE_ADD = True` and rerun the parameters cell to append them anyway. The entire
data portion is then sorted by `rt_peak`. Append and sort are separate requests. If
sorting fails or an append response is lost, inspect the sheet before retrying
to avoid duplicate rows. Confirm the new rows and values in the sheet.

## 9. Verify the database imports

The deposit cell uses the existing compound enrichment workflow and skips compounds
already present by InChIKey. It imports spectra into `msms_refs` in
`$METATLAS_DATA_DIR/databases/main_db/metatlas.duckdb`, not the project DB.
Spectra whose `(inchi_key, adduct)` already exists in `msms_refs` are warned
about and skipped, regardless of source database or collision energy. Multiple
spectra for a new pair in the same deposit are retained. Set `FORCE_ADD = True`
to insert existing pairs as well. The importer skips invalid/empty spectra and
removes fragment peaks at or above precursor m/z + 2.5 when a precursor is
present; spectra emptied by that filter are skipped. Each row's outcome is
recorded in the project database.

Check stdout and the deposit log in the project's `logs` directory for
inserted and skipped counts and warnings. To compare with the main database,
export the table and match `ref_id` against the `id` column of the final
MS/MS file:

```bash
metatlas2.sh get-db-table --table-name msms_refs --output_path /absolute/path/to/msms_refs_export.csv
```

**Imports are not idempotent:** each import creates new rows. Compound import,
sheet append/sort, and MS/MS import are separate operations, in that order. If
any step fails or its outcome is uncertain, inspect both destinations before
retrying; coordinate any cleanup with the database administrator.

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
  remain. To retry without rerunning review, alignment, or deposit, run the
  following in the notebook after replacing the path with the affected
  `annotation_run_<N>` directory. It uploads that directory and prints its link:

  ```python
  from pathlib import Path
  from metatlas2.gdrive_upload import upload_standards_outputs
  from metatlas2.load_tools import load_standards_config

  config = load_standards_config(RUN_PARAMS["config_path"])
  annotation_dir = Path("/absolute/path/to/outputs/annotation_run_<N>")
  print(upload_standards_outputs(config, annotation_dir, [annotation_dir]))
  ```

  If the deposit preview upload failed, the preview is still recorded and no
  sheet or database deposit was attempted; you can continue with
  `DRY_RUN = False`. For other deposit failures, inspect the destinations
  and the deposit's stages in the history cell before retrying.
* Keep the input CSV, config, project database, summary, alignment outputs,
  deposit files, import log, and confirmation of sheet/database results
  together according to your team's recordkeeping practice. The project
  database records which annotation run, alignment run, and deposit were used. Project
  schemas are not migrated from older versions; start a new project when
  required.
