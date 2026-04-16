# Running a Targeted Analysis Workflow

`run_targeted_analysis.py` is the main entry point for the metatlas2 pre-curation workflow. It orchestrates three sequential stages:

1. **Project Setup** — creates a project output directory, project database, and retrieves raw LCMS run files on disk.
2. **RT Alignment** — fits a polynomial retention-time correction model using QC compounds and QC LCMS run files and applies it to all target atlases supplied in the analysis configuration file (unless an --analysis-subset flag is provided).
3. **Auto Identification** — extracts MS1/MS2 data, scores any MS2 hits against a reference library, applies RT-aligned peak windows, and generates Jupyter notebooks for manual curation via a GUI.

The script can be run directly (`run` subcommand) or submitted as a Slurm batch job (`submit` subcommand).

---

## Prerequisites

- The metatlas2 conda environment is active.
- The main metatlas database contains the compounds and atlases for your project. Run `add_compounds_to_db.py` and `add_atlases_to_db.py` first (see the corresponding docs).
- LCMS run data files (.parquet format, converted from .raw and .mzML) are present at the expected path.
- Atlas UIDs referenced in the input analysis configuration file (e.g., `analysis.yaml`) exist in the main database.

---

## Command-line usage

### Run directly

```bash
python run_targeted_analysis.py run \
    --config        /path/to/analysis.yaml \
    --project       MyProject \
    [--rt-align-num  0] \
    [--analysis-num  0] \
    [--analysis-subset POS-ISTD,POS-EMA] \
    [--overwrite] \
    [--skip-setup] \
    [--skip-rt-align] \
    [--skip-auto-id]
```

### Submit as a Slurm job

```bash
python run_targeted_analysis.py submit \
    --config        /path/to/analysis.yaml \
    --project       MyProject \
    [--rt-align-num  0] \
    [--analysis-num  0] \
    [--analysis-subset POS-ISTD,POS-EMA] \
    [--overwrite] \
    [--skip-setup] \
    [--skip-rt-align] \
    [--skip-auto-id] \
    [--qos       regular] \
    [--cpus      8] \
    [--mem       64G] \
    [--time      03:00:00] \
    [--conda-env metatlas2] \
    [--output    /path/to/custom_script.sh]
```

The `submit` subcommand writes a Slurm batch script to the analysis output directory and immediately calls `sbatch` on it. The script path and job ID are printed to stdout.

---

## Arguments

### Shared arguments (`run` and `submit`)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--config` | Yes | — | Path to the analysis YAML config file (e.g. `configs/analysis.yaml`). |
| `--project` | Yes | — | Project name. Must match the name of the raw data subdirectory under `lcmsruns`. |
| `--rt-align-num` | No | `0` | RT alignment iteration number. Increment this to run a new alignment attempt while preserving previous results. |
| `--analysis-num` | No | `0` | Analysis iteration number. Increment to run multiple analysis passes under the same RT alignment. |
| `--analysis-subset` | No | None | Comma-separated list of `POLARITY-ANALYSIS_TYPE` pairs to process (e.g. `POS-ISTD,POS-EMA`). When omitted, all atlases in the config are processed. |
| `--overwrite` | No | `False` | Overwrite the project database if it already exists during setup. |
| `--skip-setup` | No | `False` | Skip the Project Setup stage (use if the project database already exists). |
| `--skip-rt-align` | No | `False` | Skip the RT Alignment stage (use if aligned atlases already exist). |
| `--skip-auto-id` | No | `False` | Skip the Auto Identification stage. |

### Additional `submit`-only arguments

| Argument | Default | Description |
|---|---|---|
| `--qos` | `regular` | Slurm QOS partition. |
| `--cpus` | `8` | Number of CPUs to request. |
| `--mem` | `64G` | Memory to request (e.g. `64G`, `128G`). |
| `--time` | `03:00:00` | Wall-clock time limit (`HH:MM:SS`). |
| `--conda-env` | `metatlas2` | Name of the conda environment to activate in the job script. |
| `--output` | auto | Override the output path for the generated `.sh` batch script. Defaults to `<analysis_output_dir>/<project>_pre_curation.sh`. |

---

## Output directory structure

All outputs are written under a versioned directory derived from the project name and iteration numbers:

```
<projects_dir>/<project_name>/
└── RTA<rt_align_num>/
    ├── rt_aligned_atlases.csv        # Atlas UIDs and metadata after RT alignment
    ├── TGA<analysis_num>/
    │   ├── auto_ided_atlases.csv     # Atlas UIDs after auto identification
    │   ├── curated_atlases.csv       # (written after manual curation)
    │   ├── pre_curation_<jobid>.log  # Slurm stdout (submit mode only)
    │   ├── pre_curation_<jobid>.err  # Slurm stderr (submit mode only)
    │   └── <notebooks>/              # Generated Jupyter curation notebooks
```

Incrementing `--rt-align-num` creates a new `RTA<N>/` branch; incrementing `--analysis-num` creates a new `TGA<N>/` subdirectory within that `RTA` branch.

---

## Config file: `analysis.yaml`

The analysis config drives all three workflow stages.

```yaml
WORKFLOWS:
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
          ATLAS:
            uid: <atlas_uid_for_targeted_analysis>
          PARAMS:
            ...
```

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
      ppm_error: 20.0
      extra_time: 1.0
      polynomial_degree: 2 
      min_observations_per_compound: 1
      min_compounds_for_modeling: 2
      r2_threshold: 0.5
      apply_model_to_min_max: true
      exclude_inchikeys:
        - OVRNDRQMDRJTHS-ZEUBEQSHSA-N
```

#### `RT_ALIGNMENT` PARAMS reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `include_lcmsruns` | list of strings | `["QC"]` | LCMS run categories to be used for alignment. See category method [below](#lcms-file-categorization). |
| `exclude_lcmsruns` | list of strings | `["NEG"]` | LCMS run categories to be excluded from alignment. See category method [below](#lcms-file-categorization).|
| `use_existing_rt_alignment` | bool | `false` | Set true to reuse atlases created from a previous alignment (matching RTA number), or false to create a new model and atlases from scratch (will overwrite an existing alignment for the RTA number). |
| `ppm_error` | float | `20.0` | m/z tolerance for EIC extraction (ppm). |
| `extra_time` | float | `1.0` | Time (min) added to atlas rt_min/rt_max for initial peak detection. |
| `polynomial_degree` | int | `2` | Degree of the polynomial used to model RT drift. |
| `min_observations_per_compound` | int | `1` | Minimum number of LCMS runs in which a compound must be detected to be included in model fitting. |
| `min_compounds_for_modeling` | int | `2` | Minimum number of compounds with observations needed before a model is fitted. |
| `r2_threshold` | float | `0.5` | Minimum R² of the fitted model; runs that fall below this are rejected. |
| `apply_model_to_min_max` | bool | `true` | Apply the correction to `rt_min` and `rt_max` in addition to `rt_peak`, otherwise use existing atlas windows for each compound. |
| `exclude_inchikeys` | list of strings | `[]` | InChIKeys of compounds to exclude from model fitting (e.g. compounds with erratic RT behaviour). |

---

### `TARGETED_ANALYSES` section

One entry per `CHROMATOGRAPHY / POLARITY / ANALYSIS_TYPE` combination. Atlas UIDs should match atlases already present in the main database.

```yaml
TARGETED_ANALYSES:
  HILICZ:
    POS:
      ISTD:
        ATLAS:
          uid: atl-ref-istd-hilicz-pos-8b5ff31b79704f728c046a40623ace2b # This comes from running add_atlases_to_db.py
        PARAMS:
          include_lcmsruns:
            - EXPERIMENTAL
            - ISTD
          exclude_lcmsruns:
            data_extraction:
              - QC
              - NEG
            gui:
              - INJBL
              - BLANK
            id_sheet:
              - INJBL
              - BLANK
              - REFSTD
            chromatograms:
              - INJBL
              - BLANK
            id_plots:
              - INJBL
              - BLANK
              - REFSTD
            data_sheets:
              - INJBL
              - BLANK
          do_alignment: true
          create_curation_notebooks: true
          remove_unided_compounds: true
          remove_flagged_compounds: true
          ms1_min_peak_intensity: 1.0e5
          ms1_min_num_points: 5
          ppm_error: 5.0
          extra_time: 1.0
          ms2_min_score: 0.1
          ms2_min_matching_frags: 1
          ms2_frag_mz_tolerance: 0.05
          gui_require_all_evaluated: false
          gui_top_n_hits: 20
          gui_lcmsruns_colors:
            ISTD: blue
            QC: blue
            EXCTRL: red
            TXCTRL: red
            REFSTD: black
```

#### `TARGETED_ANALYSES` PARAMS reference

**Workflow flags**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `do_alignment` | bool | `true` | Apply the RT alignment correction to the atlas before data extraction. |
| `create_curation_notebooks` | bool | `true` | Generate Jupyter notebooks for manual compound curation. |
| `remove_unided_compounds` | bool | `true` | Remove compounds with no MS1 or MS2 data during the auto identification. |
| `remove_flagged_compounds` | bool | `true` | Remove compounds that were flagged as "remove" during manual GUI analysis. |

**Run filtering**

`include_lcmsruns` and `exclude_lcmsruns` filter which LCMS runs are used at each step.

| Parameter | Type | Description |
|---|---|---|
| `include_lcmsruns` | list of strings | LCMS run categories to be included for data extraction. Applied globally across all steps. Categories are determined by substrings in filenames using the logic described in the section [below](#lcms-file-categorization). |
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
| `ppm_error` | float | `5.0` | m/z tolerance for EIC extraction (ppm). |
| `extra_time` | float | `0.0` | Extra time (min) added to rt_min/rt_max for data extraction. |
| `ms1_min_peak_intensity` | float | `1e5` | Minimum peak intensity required for a compound to be retained. |
| `ms1_min_num_points` | int | `5` | Minimum number of MS1 data points required across the peak window. |

**MS2 / identification parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ms2_min_score` | float | `0.1` | Minimum cosine similarity score for an MS2 hit to be accepted. |
| `ms2_min_matching_frags` | int | `1` | Minimum number of matching fragment ions required. |
| `ms2_frag_mz_tolerance` | float | `0.05` | Fragment ion m/z tolerance (Da) for MS2 matching. |

**GUI parameters**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `gui_require_all_evaluated` | bool | `true` | Require all compounds to be evaluated (i.e., to get an MS2 note selection that is not the default 'no selection') in the curation GUI before saving. |
| `gui_top_n_hits` | int | `20` | Number of top MS2 hits to display per compound in the GUI. |
| `gui_lcmsruns_colors` | dict | `{}` | Map from LCMS run category to color (e.g. `ISTD: blue`). Used to color-code run traces in the GUI. |

---

## LCMS File Categorization

LCMS run categories are inferred from filename substrings in the following priority order:

| Category | Filename must contain (case-insensitive) | Example use |
|---|---|---|
| `QC` | `-QC` | Quality-control runs used for RT alignment |
| `ISTD` | `-ISTD` | Internal standard runs |
| `EXCTRL` | `EXCTRL-` or `TXCTRL-` | Extraction or treatment controls |
| `INJBL` | `-INJBL` or `BLANK` | Injection blank or solvent blank runs |
| `REFSTD` | `-REFSTD` or `-STANDARD` | Reference standard runs |
| `EXPERIMENTAL` | *(none of the above)* | All other sample runs |

## Typical workflows

### First run for a new project

```bash
# 1. Add the compounds you're trying to identify to the database, if necessary
python add_compounds_to_db.py --config_path /path/to/create_compounds.yaml

# 2. Add reference atlases to the database, if necessary
python add_atlases_to_db.py --config_path /path/to/create_atlases.yaml
#    → note the atlas UIDs printed in the log and add them to analysis.yaml

# 3. Run the full pre-curation workflow
python run_targeted_analysis.py run \
    --config  /path/to/analysis.yaml \
    --project MyProject \
    --rt-align-num 0 \
    --analysis-num 0 \
    [--other-flags]

```
### Re-running auto identification only, e.g., with new filtering parameters in the config YAML (skip setup and alignment)

```bash
python run_targeted_analysis.py run \
    --config       /path/to/analysis.yaml \
    --project      MyProject \
    --rt-align-num 0 \
    --analysis-num 1 \
    --skip-setup \
    --skip-rt-align
```

### Running only a subset of atlases

```bash
python run_targeted_analysis.py run \
    --config          /path/to/analysis.yaml \
    --project         MyProject \
    --analysis-subset POS-ISTD,POS-EMA
```

### Submitting to Slurm

```bash
python run_targeted_analysis.py submit \
    --config      /path/to/analysis.yaml \
    --project     MyProject \
    --qos         regular \
    --cpus        16 \
    --mem         128G \
    --time        06:00:00
```

---

## Notes

- Increment `--rt-align-num` whenever you want to redo the RT alignment from scratch while keeping previous results intact.
- Increment `--analysis-num` to run another auto-identification pass under the same RT alignment (e.g., with different MS2 thresholds).
- The `--analysis-subset` flag accepts polarity and analysis-type pairs separated by a hyphen and a comma (e.g. `POS-ISTD,NEG-EMA` or `NEG-ISTD`). These must match the `POLARITY` and analysis-type keys in `TARGETED_ANALYSES` exactly (case-sensitive).
- If `use_existing_rt_alignment: true` is set in the config, the RT alignment stage reads the previously generated RT-aligned atlases, even if `--skip-rt-align` is not passed.
- JupyterLab notebooks are generated in the `TGA<N>/` directory for manual curation analysis at the end of this workflow. Open them to manually curate identifications and export final results.
