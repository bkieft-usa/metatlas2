# Running a Targeted Analysis Workflow

`run_targeted_analysis.py` is the main entry point for the metatlas2 pre-curation workflow. It orchestrates three sequential stages:

1. **Project Setup** — creates a project output directory, project database, and retrieves raw LCMS run files on disk.
2. **RT Alignment** — fits a polynomial retention-time correction model using QC compounds and QC LCMS run files and applies it to all target atlases supplied in the analysis configuration file (unless an `--analysis-subset` flag is provided).
3. **Auto Identification** — extracts MS1/MS2 data, scores any MS2 hits against a reference library, applies RT-aligned peak windows, and generates Jupyter notebooks for manual curation via a GUI.

The script can be run directly (`run` subcommand) or submitted as a Slurm batch job (`submit` subcommand). An optional `--skip-curation` flag bypasses the manual GUI step and applies automatic curation defaults, then runs the analysis summary directly.

---

## Prerequisites

Complete the one-time environment setup described in [initial_setup.md](initial_setup.md) before running this workflow. If you are running **outside NERSC** (local laptop or non-NERSC HPC), see [run_anywhere.md](run_anywhere.md) for the full setup guide. Additionally:

- The main metatlas database contains the compounds and atlases for your project. Run `metatlas2.sh add-compounds` and `metatlas2.sh add-atlases` first (see [add_compounds_to_db.md](add_compounds_to_db.md) and [add_atlases_to_db.md](add_atlases_to_db.md)).
- LCMS run data files in `.h5` format are present at the expected path under `$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/`. See [Raw data file placement](#raw-data-file-placement) below.
- Atlas UIDs referenced in the input analysis configuration file (e.g., `analysis.yaml`) exist in the main database.

---

## Raw data file placement

Raw LCMS data must be in `.h5` (HDF5) format. Place all files for a project directly in:

```
$METATLAS_DATA_DIR/raw_data/<owner>/<project_name>/
```

- `<owner>` must match the `GENERAL.owner` field in your analysis config YAML (e.g. `jgi`, `egsb`).
- `<project_name>` must match the `--project` argument passed to `metatlas2.sh run` **exactly**.

### Project name format

The project name must contain **at least 5 underscore-delimited fields**. The 5th field (index 4) is extracted as the short name used in log filenames and output directory labels:

```
20260101_JGI_XX_000000_MYPROJECT_HILICZ_TESTXXXX
│         │   │  │      └── field[4]: short name (used in log/output filenames)
│         │   │  └── field[3]
│         │   └── field[2]
│         └── field[1]
└── field[0]: date
```

A minimal valid project name: `20260101_JGI_XX_000000_MYPROJECT`

> **Note:** This naming convention matches the NERSC LCMS project directory convention. If your project name does not follow this format, `metatlas2.sh run` will fail with a `ValueError` during path setup.

---

## Command-line usage

### Run directly

```bash
metatlas2.sh run \
    --config        /path/to/analysis.yaml \
    --project       MyProject \
    [--rt-align-num  0] \
    [--analysis-num  0] \
    [--analysis-subset POS-ISTD-default,POS-EMA-default] \
    [--overwrite] \
    [--skip-rt-align] \
    [--skip-curation] \
    [--log-to-stdout]
```

### Submit as a Slurm job

```bash
metatlas2.sh submit \
    --config        /path/to/analysis.yaml \
    --project       MyProject \
    [--rt-align-num  0] \
    [--analysis-num  0] \
    [--analysis-subset POS-ISTD-default,POS-EMA-default] \
    [--overwrite] \
    [--skip-rt-align] \
    [--skip-curation] \
    [--account    m2650] \
    [--qos        regular] \
    [--constraint cpu] \
    [--cpus       8] \
    [--mem        128G] \
    [--time       00:30:00] \
    [--image      latest] \
    [--output     /path/to/custom_script.sh]
```

The `submit` subcommand writes a Slurm batch script to the analysis output directory and immediately calls `sbatch` on it. The script path and job ID are printed to stdout.

---

## Arguments

### Shared arguments (`run` and `submit`)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Path to the analysis YAML config file (e.g. `configs/analysis.yaml`). |
| `--project` | Yes | — | Project name. Must match the name of the raw data subdirectory under `$METATLAS_DATA_DIR/raw_data/<owner>/`. |
| `--rt-align-num` | No | `0` | RT alignment iteration number. Increment this to run a new alignment attempt while preserving previous results. |
| `--analysis-num` | No | `0` | Analysis iteration number. Increment to run multiple analysis passes under the same RT alignment. |
| `--analysis-subset` | No | None | Comma-separated list of `POLARITY-ANALYSIS_TYPE-ANALYSIS_NAME` triples to process (e.g. `POS-ISTD-default,POS-EMA-default`). When omitted, all targeted analyses in the config are processed. |
| `--overwrite` | No | `False` | Overwrite the project database if it already exists during setup. |
| `--skip-rt-align` | No | `False` | Skip the RT Alignment stage. Reference atlases are copied into the project DB as RT-aligned without applying a model. |
| `--skip-curation` | No | `False` | Skip the manual curation GUI step after auto-identification. Applies automatic curation defaults and runs the analysis summary directly. |
| `--log-to-stdout` | No | `False` | Write log output to stdout in addition to the project log file. |

### Additional `submit`-only arguments

| Argument | Default | Description |
|---|---|---|
| `--account` | `m2650` | NERSC account/project to charge (e.g., `m1234`). |
| `--qos` | `regular` | Slurm QOS partition. |
| `--constraint` | `cpu` | Node constraint (e.g., `cpu` for Perlmutter CPU nodes). |
| `--cpus` | `8` | Number of CPUs to request. |
| `--mem` | `128G` | Memory to request (e.g. `64G`, `128G`). |
| `--time` | `00:30:00` | Wall-clock time limit (`HH:MM:SS`). |
| `--image` | `latest` | Container image tag to use for the job (can also be set via `METATLAS2_IMAGE_TAG` env var). |
| `--output` | auto | Override the output path for the generated `.sh` batch script. Defaults to `<project_directory>/<project_short>.sh`. |

---

## Output directory structure

All outputs are written under a versioned directory derived from the project name and iteration numbers:

```
$METATLAS_DATA_DIR/projects/targeted_outputs/<owner>/<user>/<project_name>/
├── <project_name>.duckdb                 # Project database
├── <project_short>_RTA<N>_TGA<M>.log    # Run log
└── RTA<rt_align_num>/
    ├── rt_aligned_atlases.csv            # Atlas UIDs and metadata after RT alignment
    ├── rt_alignment_results/             # RT alignment model plots and diagnostics
    └── TGA<analysis_num>/
        ├── auto_ided_atlases.csv         # Atlas UIDs after auto identification
        ├── pre_curation_<jobid>.log      # Slurm stdout (submit mode only)
        ├── pre_curation_<jobid>.err      # Slurm stderr (submit mode only)
        └── <CHROM>-<POL>-<TYPE>-<NAME>/ # One subdirectory per targeted analysis
            ├── curated_atlases.csv       # (written after manual curation)
            └── <notebooks>/              # Generated Jupyter curation notebooks
```

Incrementing `--rt-align-num` creates a new `RTA<N>/` branch; incrementing `--analysis-num` creates a new `TGA<N>/` subdirectory within that `RTA` branch.

---

## Config file: `analysis.yaml`

The analysis config drives all three workflow stages. It has four top-level sections: `GENERAL`, `GUI`, `RT_ALIGNMENT`, and `TARGETED_ANALYSES`.

```yaml
GENERAL:
  owner: <owner_name>           # e.g. jgi or egsb — determines output directory path
  msms_refs_path:               # Optional: path to a custom .jsonl MS2 reference file
  msms_refs_db_filter:          # Optional: filter MS2 refs by database name (e.g. metatlas)
  gdrive_subfolder:             # Optional: Google Drive folder ID for uploads
  max_workers:                  # Optional: max parallel workers for data extraction

GUI:
  gui_require_all_evaluated: false
  gui_top_n_hits: 10
  gui_lcmsruns_colors:
    ISTD: blue
    QC: orange
    EXCTRL: red
  note_options_overrides:       # Optional: override default MS1/MS2/other note options
    ms1_notes:
    ms2_notes:
    other_notes:

RT_ALIGNMENT:
  <CHROMATOGRAPHY>:
    ATLAS:
      uid: <atlas_uid_for_rt_alignment>
    PARAMS:
      ...

TARGETED_ANALYSES:
  <CHROMATOGRAPHY>:
    <POLARITY>:
      <ANALYSIS_TYPE>:
        <ANALYSIS_NAME>:
          ATLAS:
            uid: <atlas_uid_for_targeted_analysis>
          PARAMS:
            ...
```

> **Note:** The `TARGETED_ANALYSES` section uses a **four-level hierarchy**: chromatography → polarity → analysis type → analysis name. The analysis name (e.g. `DEFAULT`) allows multiple named variants of the same analysis type to coexist (e.g. different filtering parameters for the same atlas). The `--analysis-subset` flag uses the format `POLARITY-ANALYSIS_TYPE-ANALYSIS_NAME` (e.g. `POS-ISTD-DEFAULT`).

---

### `GENERAL` section

| Key | Type | Default | Description |
|---|---|---|---|
| `owner` | string | `jgi` | Owner label used to determine the output directory path (`targeted_outputs/<owner>/<user>/<project>/`). |
| `msms_refs_path` | string | `null` | Path to a custom `.jsonl` MS2 reference file. When set, overrides the main database reference spectra. |
| `msms_refs_db_filter` | string | `null` | Filter MS2 reference spectra by database name (e.g. `metatlas`). Applied to both file and database sources. |
| `gdrive_subfolder` | string | `null` | Google Drive folder ID for optional output uploads. |
| `max_workers` | int | `null` | Maximum number of parallel worker processes for data extraction. `null` uses the system default. |

---

### `GUI` section

| Key | Type | Default | Description |
|---|---|---|---|
| `gui_require_all_evaluated` | bool | `true` | Require all compounds to have an MS2 note selection before the summary cell can run. |
| `gui_top_n_hits` | int | `10` | Number of top MS2 hits to display per compound in the GUI. |
| `gui_lcmsruns_colors` | dict | `{}` | Map from LCMS run category to color (e.g. `ISTD: blue`). Used to color-code run traces in the GUI. |
| `gui_width` | float | `null` | Optional GUI width override in pixels. |
| `gui_height` | float | `null` | Optional GUI height override in pixels. |
| `note_options_overrides` | dict | `{}` | Override default note options for `ms1_notes`, `ms2_notes`, or `other_notes`. |

---

### `RT_ALIGNMENT` section

One entry per chromatographic method. The atlas UID must be a QC atlas already in the main database (e.g. created by `add_atlases_to_db.py`).

```yaml
RT_ALIGNMENT:
  HILICZ:
    ATLAS:
      uid: atl-ref-qc-hilicz-pos-cdf8c6709c6e4953b75917e72e851130 # This comes from running add_atlases_to_db.py
    PARAMS:
      include_lcmsruns:
        - QC
      exclude_lcmsruns:
        - NEG
      use_existing_rt_alignment: false
      model_type: polynomial
      polynomial_degree: 2
      apply_model_to_min_max: true
      atlas_extra_time: 2.0
      ms1_mz_tolerance_ppm: 5.0
      ms1_min_peak_intensity: 0
      ms1_min_num_points: 0
      min_observations_per_compound: 1
      min_compounds_for_modeling: 2
      r2_threshold: 0.5
      exclude_inchikeys:
        - OVRNDRQMDRJTHS-ZEUBEQSHSA-N
      upload_to_gdrive: false
```

#### `RT_ALIGNMENT` PARAMS reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_lcmsruns` | list of strings | `["QC"]` | LCMS run categories to be used for alignment. See category method [below](#lcms-file-categorization). |
| `exclude_lcmsruns` | list of strings | `[]` | LCMS run categories to be excluded from alignment. See category method [below](#lcms-file-categorization). |
| `use_existing_rt_alignment` | bool | `false` | Set true to reuse atlases created from a previous alignment (matching RTA number), or false to create a new model and atlases from scratch. |
| `model_type` | string | `polynomial` | Type of RT correction model. Options: `polynomial`, `linear`, `median_offset`. |
| `polynomial_degree` | int | `2` | Degree of the polynomial used to model RT drift (only used when `model_type` is `polynomial`). |
| `apply_model_to_min_max` | bool | `true` | Apply the correction to `rt_min` and `rt_max` in addition to `rt_peak`. |
| `atlas_extra_time` | float | `2.0` | Time (min) added to atlas `rt_min`/`rt_max` for initial peak detection during alignment. |
| `ms1_mz_tolerance_ppm` | float | `5.0` | m/z tolerance for EIC extraction (ppm). |
| `ms1_min_peak_intensity` | float | `0.0` | Minimum peak intensity required for a compound to be included in model fitting. |
| `ms1_min_num_points` | int | `0` | Minimum number of MS1 data points required across the peak window. |
| `min_observations_per_compound` | int | `1` | Minimum number of LCMS runs in which a compound must be detected to be included in model fitting. |
| `min_compounds_for_modeling` | int | `2` | Minimum number of compounds with observations needed before a model is fitted. |
| `r2_threshold` | float | `0.5` | Minimum R² of the fitted model; runs that fall below this are rejected. |
| `exclude_inchikeys` | list of strings | `[]` | InChIKeys of compounds to exclude from model fitting (e.g. compounds with erratic RT behaviour). |
| `upload_to_gdrive` | bool | `false` | Upload RT alignment results to Google Drive after completion. |
| `only_keep_data_in_feature` | bool | `true` | Only retain data points within the RT feature window (reduces memory usage). |
| `remove_unided_compounds` | bool | `false` | Remove compounds with no detected signal from the alignment atlas. |

---

### `TARGETED_ANALYSES` section

One entry per `CHROMATOGRAPHY / POLARITY / ANALYSIS_TYPE / ANALYSIS_NAME` combination. Atlas UIDs should match atlases already present in the main database.

```yaml
TARGETED_ANALYSES:
  HILICZ:
    POS:
      ISTD:
        DEFAULT:
          ATLAS:
            uid: atl-ref-istd-hilicz-pos-8b5ff31b79704f728c046a40623ace2b # This comes from running add_atlases_to_db.py
          PARAMS:
            include_lcmsruns:
            exclude_lcmsruns:
              data_extraction:
                - QC
                - NEG
              gui:
              id_sheet:
              chromatograms:
              id_plots:
              data_sheets:
            apply_alignment: true
            remove_unided_compounds: false
            remove_flagged_compounds: false
            only_keep_data_in_feature: false
            apply_istd_curation_to_ema: true
            apply_cross_polarity_curation: false
            suggested_min_conf: 0.75
            atlas_extra_time: 0.5
            extract_extra_time:
            ms1_min_peak_intensity: 0
            ms1_min_num_points: 0
            ms1_mz_tolerance_ppm: 5.0
            ms2_min_num_scans: 0
            ms2_min_precursor_intensity: 0
            ms2_min_score: 0
            ms2_min_matching_frags: 0
            ms2_mz_tolerance_ppm: 20.0
            ms2_frag_mz_tolerance: 0.05
            keep_top_scan_per_compound_file: true
            create_curation_notebooks: true
            upload_to_gdrive: false
            skip_outputs:
```

#### `TARGETED_ANALYSES` PARAMS reference

**Workflow flags**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `apply_alignment` | bool | `true` | Apply the RT alignment correction to the atlas before data extraction. When `false`, the reference atlas is registered as RT-aligned without applying the model. |
| `create_curation_notebooks` | bool | `true` | Generate Jupyter notebooks for manual compound curation. |
| `remove_unided_compounds` | bool | `true` | Remove compounds with no MS1 signal during auto identification. |
| `remove_flagged_compounds` | bool | `true` | Remove compounds that were flagged as "remove" during manual GUI analysis. |
| `apply_istd_curation_to_ema` | bool | `true` | Apply ISTD curation decisions to EMA compounds (propagate ISTD keep/remove flags). |
| `apply_cross_polarity_curation` | bool | `true` | Apply curation decisions from one polarity to the same compound in the opposite polarity. |
| `upload_to_gdrive` | bool | `false` | Upload summary outputs to Google Drive after completion. |
| `skip_outputs` | list or null | `null` | List of output types to skip during summary generation. |

**Run filtering**

`include_lcmsruns` and `exclude_lcmsruns` filter which LCMS runs are used at each step.

| Parameter | Type | Description |
|---|---|---|
| `include_lcmsruns` | list of strings | LCMS run categories to be included for data extraction. Applied globally across all steps. When `null`, defaults to `EXPERIMENTAL`, `ISTD`, `EXCTRL`, `REFSTD`, `INJBLK`. Categories are determined by substrings in filenames using the logic described in the section [below](#lcms-file-categorization). |
| `exclude_lcmsruns` | dict of lists | Per-step exclusion filters. Keys are `data_extraction`, `gui`, `id_sheet`, `chromatograms`, `id_plots`, `data_sheets`. Each value is a list of LCMS run categories to exclude at that step. |

The `exclude_lcmsruns` step keys are:

| Step key | Affected output |
|---|---|
| `data_extraction` | Which runs have EIC/MS2 data extracted. |
| `gui` | Which runs appear in the interactive curation GUI. |
| `id_sheet` | Which runs appear in the identification summary sheet. |
| `chromatograms` | Which runs are plotted in chromatogram summaries. |
| `id_plots` | Which runs appear in identification plots. |
| `data_sheets` | Which runs appear in exported data sheets. |

**MS1 parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ms1_mz_tolerance_ppm` | float | `5.0` | m/z tolerance for EIC extraction (ppm). |
| `atlas_extra_time` | float | `0.5` | Extra time (min) added to atlas `rt_min`/`rt_max` for the `in_feature` tag boundary. |
| `extract_extra_time` | float or null | `null` | Optional wider RT padding (min) used only for the HDF5 pre-filter step. When `null`, uses `atlas_extra_time`. |
| `ms1_min_peak_intensity` | float | `1e5` | Minimum peak intensity required for a compound to be retained. |
| `ms1_min_num_points` | int | `5` | Minimum number of MS1 data points required across the peak window. |
| `only_keep_data_in_feature` | bool | `false` | Only retain data points within the RT feature window. |
| `keep_top_scan_per_compound_file` | bool | `true` | Keep only the highest-intensity MS2 scan per compound per file. |
| `suggested_min_conf` | float | `0.75` | Minimum confidence threshold for auto-suggested RT bounds. |

**MS2 / identification parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ms2_min_score` | float | `0.25` | Minimum cosine similarity score for an MS2 hit to be accepted. |
| `ms2_min_matching_frags` | int | `1` | Minimum number of matching fragment ions required. |
| `ms2_frag_mz_tolerance` | float | `0.05` | Fragment ion m/z tolerance (Da) for MS2 matching. |
| `ms2_mz_tolerance_ppm` | float | `20.0` | Precursor m/z tolerance (ppm) for MS2 scan matching. |
| `ms2_min_num_scans` | int | `1` | Minimum number of MS2 scans required for a compound to be retained. |
| `ms2_min_precursor_intensity` | float | `0.0` | Minimum precursor intensity for an MS2 scan to be used. |

**GUI parameters**

> **Note:** GUI parameters (`gui_require_all_evaluated`, `gui_top_n_hits`, `gui_lcmsruns_colors`) are set globally in the top-level `GUI` section of the config, not per-analysis. They can be overridden at runtime via the `OVERRIDE_PARAMS` cell in the curation notebook.

---

## LCMS File Categorization

LCMS run categories are inferred from filename substrings in the following priority order:

| Category | Filename must contain (case-insensitive) | Example use |
|---|---|---|
| `qc` | `-QC` (but not `-QC-C18QU`) | Quality-control runs used for RT alignment |
| `istd` | `-ISTD` | Internal standard runs |
| `exctrl` | `EXCTRL-` or `TXCTRL-` | Extraction or treatment controls |
| `injbl` | `-INJBL` or `BLANK` | Injection blank or solvent blank runs |
| `refstd` | `-REFSTD` or `-STANDARD` | Reference standard runs |
| `experimental` | *(none of the above)* | All other sample runs |

The category names used in `include_lcmsruns` and `exclude_lcmsruns` are case-insensitive and match these category labels.

## Typical workflows

### First run for a new project

```bash
# 1. Add the compounds you're trying to identify to the database, if necessary
metatlas2.sh add-compounds --config_path /path/to/create_compounds.yaml

# 2. Add reference atlases to the database, if necessary
metatlas2.sh add-atlases --config_path /path/to/create_atlases.yaml
#    → note the atlas UIDs printed in the log and add them to analysis.yaml

# 3. Run the full pre-curation workflow
metatlas2.sh run \
    --config  /path/to/analysis.yaml \
    --project MyProject \
    --rt-align-num 0 \
    --analysis-num 0 \
    [--other-flags]

```
### Re-running auto identification only, e.g., with new filtering parameters in the config YAML (skip alignment)

```bash
metatlas2.sh run \
    --config       /path/to/analysis.yaml \
    --project      MyProject \
    --rt-align-num 0 \
    --analysis-num 1 \
    --skip-rt-align
```

### Running only a subset of atlases

```bash
metatlas2.sh run \
    --config          /path/to/analysis.yaml \
    --project         MyProject \
    --analysis-subset POS-ISTD-DEFAULT,POS-EMA-DEFAULT
```

### Skipping manual curation (automated pipeline)

```bash
metatlas2.sh run \
    --config       /path/to/analysis.yaml \
    --project      MyProject \
    --skip-curation
```

### Submitting to Slurm

```bash
metatlas2.sh submit \
    --config      /path/to/analysis.yaml \
    --project     MyProject \
    --account     m2650 \
    --qos         regular \
    --cpus        16 \
    --mem         128G \
    --time        00:30:00
```

---

## Notes

- Increment `--rt-align-num` whenever you want to redo the RT alignment from scratch while keeping previous results intact.
- Increment `--analysis-num` to run another auto-identification pass under the same RT alignment (e.g., with different MS2 thresholds).
- The `--analysis-subset` flag accepts polarity, analysis-type, and analysis-name triples separated by hyphens and commas (e.g. `POS-ISTD-DEFAULT,NEG-EMA-DEFAULT`). These must match the `POLARITY`, analysis-type, and analysis-name keys in `TARGETED_ANALYSES` exactly (case-insensitive).
- If `use_existing_rt_alignment: true` is set in the config, the RT alignment stage reads the previously generated RT-aligned atlases, even if `--skip-rt-align` is not passed.
- JupyterLab notebooks are generated in the `TGA<N>/<atlas_label>/` directory for manual curation analysis at the end of this workflow. Open them to manually curate identifications and export final results.
- The `--skip-curation` flag is useful for automated pipelines where manual review is not required. It applies default curation decisions (keeping all auto-identified compounds) and runs the analysis summary immediately after auto-identification.
