# Standards annotation: analyst workflow

This guide covers preparing reference standards, reviewing their peaks,
aligning retention times (RTs), appending atlas rows to Google Sheets, and
importing compounds and MS/MS references into the main database. RTs are in **minutes**.

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
| `compound_name` | The name used in review and exported as `Compound Name`; the separate `Compound Label` field becomes the canonical atlas `label`. |
| `smiles`, `inchi`, `inchi_key` | Supply a valid SMILES or InChI where possible. If only an InChIKey is available, the workflow attempts a PubChem lookup. Missing InChI/InChIKey can be derived from a structure. |
| `project_name` | The experiment folder name, not a full path. The workflow searches `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/*.h5`. |
| `group_name` | The sample name parsed from the HDF5 filename, matched case-insensitively. It is not a substring search. |
| `rt_min`, `rt_max` | The expected extraction window in minutes. Blank or nonnumeric values default to 0 and 20, respectively; enter explicit numeric bounds to avoid an unintended window. |
| `annotation_notes` | Optional existing notes for this input row, carried through to the summary and atlas. |

Quote CSV fields containing commas. Check structures and names carefully:
the workflow requires a resolvable InChI and InChIKey for each input row.
It desalts the structure by keeping its largest fragment and neutralizes
charges using the historical standards cleaning rules. If the resulting
InChIKey differs from the supplied or derived key, input expansion stops;
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
  chem_source: "Chemical source"
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
    annotation_run_<annotation_timestamp>/
      StandardsAnnotation_Summary_<standards_run_name>.csv
      summary_figures/
      rt_align_<alignment_timestamp>/
        StandardsAnnotation_Summary_Aligned_<standards_run_name>.csv
        rt_alignment/
        Atlas_Deposit_Preview_<standards_run_name>.csv
        MsmsRefs_Deposit_Preview_<standards_run_name>.tab
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
its generated `RUN_PARAMS` should not need editing.

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
   closest-MS2 rule. Review its selection and spectra.
4. Enter analyst notes. Use **Previous**, **Next**, or the compound dropdown
   to navigate; navigation saves decisions and notes. Click **Save & Exit**
   when finished. Reopen the notebook and GUI to resume from saved review.

## 6. Generate and approve the summary

After **Save & Exit**, run the notebook's summary cell. Every target group
must have a decision and at least one actual peak must be selected. The cell
prints a UTC timestamp of the form `YYYYMMDDTHHMMSS_ffffffZ` and creates:

```text
outputs/annotation_run_<annotation_timestamp>/
  StandardsAnnotation_Summary_<standards_run_name>.csv
  summary_figures/
  .summary_complete
```

The cell uploads the entire annotation directory to Google Drive and prints
the link to that `annotation_run_<annotation_timestamp>` folder. Send this
link to the supervisor and review the CSV and PDFs before proceeding.
The CSV contains one row per selected peak, notes, MS1 source, and separate
CE102040/CE205060 MS2 fields. Spectra are `[[m/z values], [intensities]]`;
missing scans leave empty fields. Selected peaks are numbered by measured RT
within each chromatography/polarity/compound/source/adduct group; `GUI Peak
Num` preserves the original GUI number. Summary generation does not renumber
database peaks. Each invocation creates a new directory.

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

In the parameters cell immediately before reverse alignment, set
`ANNOTATION_TIMESTAMP` to the approved summary timestamp (without the
directory prefix), run that cell, then run the reverse alignment cell.

The summary timestamp must name a completed summary. Alignment nevertheless
reads the **current saved project database selections**, not a snapshot of
that summary. After any review changes, generate and approve a new summary
and use its timestamp. The software does not compare the two. Do not edit the
config or QC inputs while alignment is running.

The cell prints an alignment timestamp and writes a new
`rt_align_<alignment_timestamp>/` inside the chosen summary directory:

* `StandardsAnnotation_Summary_Aligned_<standards_run_name>.csv` has aligned
  peak/min/max columns, signed `RT diff` (aligned minus measured peak), and
  the config's `TAGS.atlas`.
* `rt_alignment/` contains per-chromatography model PDFs and
  `alignment_config.json`, a snapshot of the loaded settings.

Inspect the QC points, fit, residuals, R², RMSE, aligned RTs, bounds, and
tags. Missing QC data, insufficient observations, or an unusable model stop
alignment. Adjust the config and rerun to compare separate timestamped
outputs; approve the particular alignment to deposit. If intentionally
skipping alignment is appropriate, set `SKIP_RT_ALIGNMENT = True` in the
alignment parameters cell and rerun it; aligned RTs then equal measured RTs
and no model plots are produced. The
alignment directory is uploaded into the same Drive annotation folder.
Alignment does not itself write an atlas to the main database.

## 8. Preview and deposit the atlas and references

Set `ALIGNMENT_TIMESTAMP` in the parameters cell immediately before deposit to
the **approved** alignment timestamp, without the directory prefix. Keep
`ANNOTATION_TIMESTAMP` set to the corresponding approved summary timestamp.
Run the deposit parameters cell and then the deposit cell. `DRY_RUN` defaults
to `True`.

The dry run writes the following next to the chosen aligned summary and
uploads them to the same Drive alignment folder, without depositing to the
sheet or changing the main database:

* `Atlas_Deposit_Preview_<standards_run_name>.csv`: aligned RTs,
  theoretical m/z, 5 ppm m/z tolerance, tags from the aligned summary,
  preparation dates, and combined analyst/input notes. Structures are read
  from the **current** project database and formula is derived from them.
* `MsmsRefs_Deposit_Preview_<standards_run_name>.tab`: one importer-format
  row per available selected peak/CE spectrum. Missing or empty spectra are
  warned about and omitted; missing precursor m/z or CE is also warned about.
  `database` is `metatlas`, `id` is a unique 32-character ID, and `name`
  is the atlas compound label. The MS2 scan filename supplies the instrument
  and normalized CE ramp (`CE102040norm` becomes `ramp-102040norm`, and
  `CE205060norm` becomes `ramp-205060norm`); instrument type is `Orbitrap`.
  A missing, malformed, or unsupported scan filename stops preparation.
  Inspect structures, formula, exact mass, metadata, spectra, and counts.

The analyst must select the reviewed timestamps and ensure project structures
still match the review; the tool does not verify subsequent changes. Both
previews are created exclusively and cannot be overwritten. After reviewing
them, move or remove **both** preview files, set `DRY_RUN = False`, rerun the
deposit parameters cell, and rerun the deposit cell. This recreates both files,
imports compounds, appends the atlas, and imports MS/MS references. The full
previews are used for both database imports, including rows skipped from the
sheet.

If one deposit step has already completed, move or remove both previews before
retrying with `SKIP_ATLAS_UPLOAD = True` to skip the Google Sheets upload or
`SKIP_MSMS_REFS_DEPOSIT = True` to skip the MS/MS reference import. Rerun the
parameters cell after changing either option. Both flags
default to false and can be combined; previews are still created. Compound
import and InChIKey validation still run when either deposit step is enabled.
Check the sheet and main database after a failure before deciding which step
to skip, since a failed operation may have partially completed.

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
to insert existing pairs as well. The preview contains all prepared spectra,
including any that will be skipped.
Generated rows use `database=metatlas`. The main database's numeric
`collision_energy` column stores the measured scan energy from the aligned
summary, not the ramp string in the `.tab`. Importing the `.tab` separately
with `add-msms-refs` cannot recover that measured value and warns that the
numeric field will be blank. If the MS/MS preview is
missing at import time, the cell fails after the sheet deposit; inspect the
sheet before retrying.

Check stdout and the deposit log in the project's `logs` directory for
inserted and skipped counts and warnings. The importer skips invalid/empty spectra and removes fragment peaks
at or above precursor m/z + 2.5 when a precursor is present; spectra emptied
by that filter are skipped. Compare the resulting records and counts with the
reviewed preview, accounting for these filters. For example, with read access
to the main database, export the table for inspection:

```bash
metatlas2.sh get-db-table --table-name msms_refs --output_path /absolute/path/to/msms_refs_export.csv
```

Match `database=metatlas` and `ref_id` against the IDs in the deposited preview
(the IDs are regenerated on each preview). **Imports are not idempotent:** each import
creates new rows, even for the same file or InChIKey. Compound import, sheet
append/sort, and MS/MS import are separate operations, in that order. If any
step fails or its outcome is uncertain, inspect both destinations before
retrying; coordinate any cleanup with the database administrator.

## Troubleshooting and recordkeeping

* No files: check `owner`, experiment folder, exact parsed sample name,
  polarity, and filename conventions. More than two matching files per group
  must be resolved before peak selection.
* Missing compounds: check `standards_annotation.log` for invalid structures,
  PubChem lookup failures, skipped masses, extraction errors, and invalid
  adducts. Check the expanded table against the CSV.
* Incomplete review: reopen the GUI, resolve every group, save, and rerun the
  summary. A summary requires at least one selected peak overall.
* Failed alignment: check the chromatography's QC atlas UID, matching QC
  files/polarity, extraction window and tolerance, and minimum observations.
  Review diagnostic PDFs rather than assuming a fitted model is acceptable.
* Failed Drive upload: fix rclone configuration or access. Local outputs
  remain. To retry without rerunning review, alignment, or deposit, run the
  following in the notebook after replacing the path with the affected
  annotation directory. It uploads that directory and prints its link:

  ```python
  from pathlib import Path
  from metatlas2.gdrive_upload import upload_standards_outputs
  from metatlas2.load_tools import load_standards_config

  config = load_standards_config(RUN_PARAMS["config_path"])
  annotation_dir = Path("/absolute/path/to/outputs/annotation_run_<annotation_timestamp>")
  print(upload_standards_outputs(config, annotation_dir, [annotation_dir]))
  ```

  If the deposit preview upload failed, no database or sheet deposit was
  attempted. After resolving the upload, move or remove both local previews
  before retrying the deposit cell. For other deposit failures, inspect
  the destinations before retrying.
* Keep the input CSV, config, project database, approved timestamps, summary,
  alignment outputs, previews, import log, and confirmation of sheet/database
  results together according to your team's recordkeeping practice. Project
  schemas are not migrated from older versions; start a new project when
  required.
