# metatlas2 Codebase Overview

A programmer-oriented reference for understanding how a typical targeted metabolomics workflow unfolds: how to get started with metatlas2, which objects are created at each stage, which functions are called in order, and what each one does.

---

## Table of Contents

- [Module Map for Targeted Analysis](#module-map-for-targeted-analysis)
- [Other Scripts and Tools](#other-scripts-and-tools)
- [Adding to the central metatlas knowledge store](#adding-to-the-central-metatlas-knowledge-store)
  - [1. Create new Compounds](#1-create-new-compounds-in-the-main-database--compoundcreate_from_configconfig_path)
  - [2. Create new Atlases](#2-create-new-atlases-in-the-main-database--atlascreate_from_configconfig_path)
- [Per-Project Workflow](#per-project-workflow)
  - [Entry Point: run_targeted_analysis.main()](#entry-point-run_targeted_analysismain)
  - [Phase 1 — Project Setup](#phase-1--project-setup-wfsrun_project_setup)
  - [Phase 2 — RT Alignment](#phase-2--rt-alignment-wfsrun_rt_alignment)
  - [Phase 3 — Auto Identification](#phase-3--auto-identification-wfsrun_auto_identification)
  - [Phase 4 — Analysis GUI](#phase-4--analysis-gui-wfsrun_analysis_gui)
  - [Phase 5 — Analysis Summary](#phase-5--analysis-summary-wfsrun_analysis_summary)
- [Key Data Objects](#key-data-objects-classes-found-in-workflow_objects)
- [First-Time Setup](#first-time-setup)
- [Container-Based Deployment](#container-based-deployment)
  - [Architecture at a Glance](#architecture-at-a-glance)
  - [Execution Modes](#execution-modes)
  - [Container File Map](#container-file-map)
  - [Image Registry and Tagging](#image-registry-and-tagging)
  - [Host Wrapper Script (metatlas2.sh)](#host-wrapper-script-metatlas2sh)
  - [Jupyter Kernel Specs](#jupyter-kernel-specs)
  - [Development Workflow](#development-workflow)
  - [Keeping the Local Cache Current](#keeping-the-local-cache-current)
- [Output Directory Layout](#output-directory-layout)

---

## Module Map for Targeted Analysis

| Module | Role |
|---|---|
| `run_targeted_analysis.py` | CLI entry point — parses args, builds `paths` dict, dispatches to `workflows.py` |
| `workflows.py` | Orchestration layer — one `run_*` function per workflow phase |
| `workflow_objects.py` | Dataclasses that carry state between workflow functions |
| `database_interact.py` (`dbi`) | All DuckDB reads and writes |
| `load_tools.py` (`ldt`) | Config/file loading and CSV serialization |
| `lcmsruns_tools.py` (`lrt`) | Parquet file discovery and LCMS run filtering |
| `extract_data_from_parquet.py` (`edp`) | Reads raw parquet files to build `ExperimentalData` |
| `rt_align_tools.py` (`rat`) | RT alignment model fitting, application, and visualization |
| `ms2_hit_detection.py` (`mhd`) | MS2 spectral matching against reference library |
| `manual_curation_summarizer.py` (`mcs`) | Builds `ManualCuration` objects from identification results |
| `analysis_gui.py` (`agu`) | Builds the interactive Dash curation app |
| `analysis_summary.py` (`asm`) | Generates final summary files and QC figures |
| `notebook_generator.py` (`nbg`) | Generates Jupyter notebooks for analyst curation |
| `pubchem_retrieval.py` (`pcr`) | Fetches and caches compound metadata from PubChem |
| `logging_config.py` | Sets up logging to point to scripts where each step takes place for transparency |

## Other Scripts and Tools

| Module | Role |
|---|---|
| `add_atlases_to_db.py` | Adds new atlas (set of compounds) to main database via config |
| `add_compounds_to_db.py` | Adds new compounds (minimal information) to main database so they are findable |
| `convert_raw_files.py` | An automated background script that converts .raw LCMS run files to .mzML, .h5, and .parquet |

---

## Adding to the central metatlas knowledge store

### 1. Create new Compounds in the main database — `Compound.create_from_config(config_path)`

| Call | What it does |
|---|---|
| `ldt.load_compound_config(config_path)` | Parses the compounds YAML config |
| `rta.set_up_paths(config)` | Builds the `paths` dict pointing at the main DB and PubChem cache |
| `dbi.create_metatlas_database(main_db_path)` | Creates the DuckDB schema if it does not already exist |
| `ldt.load_compound_input(file_path)` | Reads a compound CSV/TSV into a DataFrame |
| `pcr.retrieve_pubchem_info(compounds_df, ...)` | Enriches the DataFrame with PubChem identifiers, using a local parquet cache to avoid redundant API calls |
| `Compound.from_atlas_row(row)` | Converts each DataFrame row into a `Compound` dataclass |
| `CompoundMZRT.from_atlas_row(row)` | Converts each row into a `CompoundMZRT` dataclass holding RT/MZ reference values |
| `dbi.batch_save_compounds_and_mzrts(main_db_path, compounds, compound_mzrts)` | Bulk-inserts both lists into the main database |

---

### 2. Create new Atlases in the main database — `Atlas.create_from_config(config_path)`

| Call | What it does |
|---|---|
| `ldt.load_atlas_config(config_path)` | Parses the atlases YAML config |
| `ldt.load_atlas_input(atlas_info['path'])` | Reads the atlas CSV into a DataFrame |
| `dbi.create_new_atlas_from_dataframe(atlas_df, ...)` | Constructs an `Atlas` object with `CompoundMZRT` entries and assigns a UID |
| `dbi.save_atlas_to_database(atlas_obj, main_db_path)` | Writes the atlas and its compound-association rows to the main database |

---

## Per-Project Workflow

Run for every new experimental project via `python -m metatlas2.run_targeted_analysis run --config ...` (or `submit` to wrap in a Slurm job).

### Entry Point: `run_targeted_analysis.main()`

| Call | What it does |
|---|---|
| `parse_args()` | Parses CLI arguments including `--config`, `--project`, `--rt-align-num`, `--analysis-num`, and skip/overwrite flags |
| `ldt.load_metatlas2_config(args.config)` | Loads the project-level `analysis.yaml` |
| `set_up_paths(config, project_name, ...)` | Builds all directory paths, creates output directories, and validates that the raw data and main DB exist |
| `lcf.setup_logging(...)` | Configures rotating-file or stdout logging for the run |

The three main phases are then called in sequence unless individually skipped (see below).

---

### Phase 1 — Project Setup: `wfs.run_project_setup(...)`

Sets up the project directory structure and load LCMS run .parquet files into the database.

| Call | What it does |
|---|---|
| `Project()` | Instantiates an empty project container dataclass |
| `Project.setup(project_name, config, paths, overwrite_existing)` | Orchestrates all setup sub-steps below |
| `dbi.create_project_database(project_db_path, rt_align_path, overwrite)` | Creates the project-scoped DuckDB file; returns early if it already exists and overwrite is False |
| `lrt.get_project_lcmsruns_from_disk(lcmsruns_directory)` | Walks the raw data directory to discover parquet files, inferring file type, chromatography, polarity, and MS level from filenames |
| `dbi.save_lcmsruns_to_db(project_db_path, project_name, lcmsruns_list, overwrite)` | Writes the run metadata list to the `lcmsruns` table in the project database |
| `LCMSRun(**row)` | Wraps each run's metadata dict into a typed `LCMSRun` dataclass stored in `Project.lcmsruns` |

**State after this phase:** Project database exists with a populated `lcmsruns` table.

---

### Phase 2 — RT Alignment: `wfs.run_rt_alignment(...)`

Creates an RT alignment model from QC files based on an alignment template atlas, then applies the model to all atlases in the config file.

| Call | What it does |
|---|---|
| `RTAlign()` | Instantiates the RT alignment class object |
| `RTAlign.setup(project_name, rt_alignment_number, config, paths)` | Reads chromatography and QC atlas UID from config; checks for an existing aligned-atlases CSV and sets `run_alignment = False` if `use_existing_rt_alignment` is True or if alignment is globally disabled |
| `dbi.get_lcmsruns_from_db(project_db_path)` | Fetches all `LCMSRun` rows from the project database |
| `lrt.filter_lcmsruns_list(lcmsruns, include_file_type, exclude_file_type, chromatography, ms_level=1)` | Filters runs to those used for alignment (typically QC files); result stored in `RTAlign.aligner_lcmsruns` |
| `Atlas.from_database(main_db_path, align_atlas_uid)` | Loads the reference QC atlas from the main database into `RTAlign.align_atlas_obj` |
| `edp.extract_eic_and_ms2_from_parquet(obj=rt_align_obj, stage="rt_alignment")` | Reads parquet files in parallel; returns an `ExperimentalData` object holding `MS1Data` entries (EICs) — MS2 extraction is skipped at this stage |
| `rat.create_file_matching_summary(experimental_data, atlas)` | Logs and writes a summary of how many QC files contained detectable signal for each atlas compound |
| `rat.build_rt_alignment_model(experimental_data, atlas, rt_align)` | Fits a polynomial regression between observed and reference RTs; stores the model and residual stats in `RTAlign.rt_alignment_model` and `RTAlign.rt_shift_stats` |
| `dbi.save_rt_alignment_model_to_db(rt_align_obj)` | Persists model coefficients, R², and RMSE to the project database |
| `rat.visualize_rt_alignment_model(rt_align_obj)` | Writes a model-fit diagnostic plot to the RT alignment output directory |
| `rat.apply_rt_alignment_to_target_atlases(rt_align_obj)` | Applies the polynomial correction to RT windows for every analysis-type atlas in the config; new `Atlas` objects are stored in `RTAlign.rt_aligned_atlases` |
| `dbi.save_atlas_to_database(aligned_atlas_obj, project_db_path, main_db_path)` | Saves each RT-corrected atlas to the project database |
| `ldt.save_atlas_data_to_csv(atlas_obj, aligned_atlases_store_file)` | Appends each aligned atlas row to the CSV that the next phase reads |
| `rat.display_rt_alignment_summary(rt_align_obj)` | Prints final alignment statistics (RMSE, R², compound count) |

**State after this phase:** Project DB contains one RT alignment record and one aligned atlas per analysis type; `RTA<N>/rt_aligned_atlases.csv` exists.

---

### Phase 3 — Auto Identification: `wfs.run_auto_identification(...)`

Loops over every aligned atlas (each chromatography × polarity × analysis_type entry in the CSV). For each:

| Call | What it does |
|---|---|
| `AutoIdentification()` | Instantiates the auto-ID class object |
| `AutoIdentification.setup(project_name, rt_alignment_number, analysis_number, config, paths, analysis_subset)` | Populates metadata; `analysis_subset` allows restricting processing to a polarity–analysis_type subset |
| `dbi.check_existing_auto_identification(auto_id_obj)` | Guards against re-running if results already exist in the database for this run number |
| `dbi.get_lcmsruns_from_db(project_db_path)` | Fetches all `LCMSRun` rows |
| `ldt.load_atlas_data_from_csv(aligned_atlases_store_file)` | Reads the CSV written by Phase 2 to get the list of aligned atlas UIDs to process |
| `Atlas.from_database(project_db_path, atlas_uid, main_db_path)` | Loads the aligned atlas into `AutoIdentification.pre_autoid_atlas_obj` |
| `lrt.filter_lcmsruns_list(lcmsruns, ..., chromatography, polarity)` | Filters to sample files for the atlas polarity/chromatography; stored in `AutoIdentification.autoid_lcmsruns` |
| `edp.extract_eic_and_ms2_from_parquet(obj=auto_id_obj, stage="auto_identification")` | Extracts both MS1 (EICs) and MS2 spectra from parquet files in parallel; returned `ExperimentalData` stored in `AutoIdentification.experimental_data` |
| `mhd.find_ms2_hits(auto_id_obj)` | Compares extracted MS2 spectra against the MS/MS reference library using spectral similarity scoring; hit results are stored in `ExperimentalData.ms2_hits` |
| `mcs.create_manual_curation_obj(auto_id_obj)` | Constructs `ManualCuration` objects with per-compound identification summaries (best MS1 file, RT error, isomers, suggested RT bounds, auto-ID flag), stored in `ExperimentalData.manual_curation` |
| `dbi.save_auto_identification_results_to_db(auto_id_obj)` | Bulk-inserts all MS1 data, MS2 raw spectra, MS2 hits, and `ManualCuration` records into the project database |
| `dbi.display_auto_id_summary(auto_id_obj)` | Prints a table summary of identification counts and confidence levels |
| `dbi.create_new_atlas_after_auto_id(auto_id_obj)` | Builds a new `Atlas` from the auto-ID results, copying over compounds that passed the auto-ID filters; stored in `AutoIdentification.post_autoid_atlas_obj` |
| `ldt.save_atlas_data_to_csv(post_autoid_atlas_obj, auto_ided_atlases_store_file)` | Saves the post-auto-ID atlas to `TGA<N>/auto_ided_atlases.csv` |
| `nbg.generate_gui_notebooks(auto_id_obj)` | Generates Jupyter notebooks (one per atlas) pre-configured for analyst curation in the analysis output directory |

**State after this phase:** Project DB contains MS1/MS2/hit/curation tables; curation notebooks are ready for the analyst.

---

### Phase 4 — Analysis GUI: `wfs.run_analysis_gui(...)`

Launched by the analyst from the curation notebook (auto-created during Phase 3). Runs an interactive Dash app for manual review of all compounds that were detected.

| Call | What it does |
|---|---|
| `AnalysisGUI()` | Instantiates the GUI class object |
| `AnalysisGUI.setup(project_name, rt_alignment_number, analysis_number, config, paths)` | Populates metadata and paths |
| `Atlas.from_database(project_db_path, pre_curation_atlas_uid, main_db_path)` | Loads the pre-curation (post-auto-ID) atlas into `AnalysisGUI.pre_curation_atlas_obj` |
| `dbi.load_and_filter_gui_inputs(analysis_gui_obj, override_parameters)` | Queries the project DB for all MS1, MS2, hits, and manual-curation data; applies any analyst-supplied filter overrides; stores DataFrames in `AnalysisGUI.ms1_df`, `.ms2_df`, `.ms2_hits_df`, `.manual_curation_df` |
| `agu.build_dash_app(analysis_gui_obj, port, shutdown_holder)` | Constructs the Dash application object with all callbacks and layouts bound to the GUI state |
| `make_server(...)` + `threading.Thread(target=server.serve_forever)` | Starts a Werkzeug WSGI server in a background thread serving the Dash app |

**State after this phase:** Analyst uses the GUI to accept/reject identifications and adjust RT windows; updated curation rows are written back to the project DB during the session.

---

### Phase 5 — Analysis Summary: `wfs.run_analysis_summary(...)`

Launched by the analyst from the curation notebook. This creates summary files, tables, and figures for the entire analysis.

| Call | What it does |
|---|---|
| `AnalysisSummary()` | Instantiates the summary class object |
| `AnalysisSummary.setup(project_name, rt_alignment_number, analysis_number, config, paths)` | Populates metadata then immediately calls `load_data()` |
| `AnalysisSummary.load_data()` | Pre-loads all four data tables from the project DB (`manual_curation_df`, `ms1_all_df`, `ms2_raw_all_df`, `ms2_hits_all_df`) plus `per_file_metrics_df` derived from MS1 data so summary functions don't re-query |
| `Atlas.from_database(project_db_path, pre_curation_atlas_uid, main_db_path)` | Loads the pre-curation atlas into `AnalysisSummary.pre_curation_atlas_obj` |
| `dbi.create_new_atlas_after_manual_curation(summary_obj)` | Builds a final curated `Atlas` from the analyst's accepted identifications; stored in `AnalysisSummary.post_curation_atlas_obj` |
| `ldt.save_atlas_data_to_csv(post_curation_atlas_obj, curated_atlases_store_file)` | Saves the curated atlas to `TGA<N>/curated_atlases.csv` |
| `asm.run_all_summaries(summary_obj, overwrite)` | Generates all output files: per-compound plots, QC tables, identification-confidence summaries, and export CSVs |

**State after this phase:** All summary files are written to the `RTA<N>/TGA<N>/` directory; the curated atlas CSV is ready for downstream use.

---

## Key Data Objects (classes found in `workflow_objects`)

| Object | Purpose |
|---|---|
| `Compound` |  Immutable chemical identity record (name, InChI key, formula, PubChem CID, etc.) |
| `CompoundMZRT` | Reference RT/MZ window for a compound under a specific chromatography/polarity/adduct |
| `Atlas` | Named collection of `CompoundMZRT` entries for one chromatography × polarity × analysis type |
| `LCMSRun` | Metadata record for a single raw parquet file (path, type, polarity, MS level) |
| `Project` | Container holding project config, paths, and `LCMSRun` list during setup |
| `RTAlign` | Carries RT alignment state: QC atlas, filtered runs, model coefficients, and aligned atlases |
| `ExperimentalData` | Holds extracted `MS1Data`, `MS2Data`, `MS2Hit`, and `ManualCuration` lists for one atlas × LCMS run set |
| `ManualCuration` | Per-compound identification summary: best MS1 file, RT error, auto-ID flag, suggested RT bounds |
| `MS1Data` / `MS2Data` / `MS2Hit` | Thin wrappers (`_SpecData`) around a per-file DataFrame of spectral data or hit scores |
| `AutoIdentification` | Carries auto-ID state: pre/post atlases, filtered runs, `ExperimentalData`, and results |
| `AnalysisGUI` | Holds in-memory DataFrames and atlas objects for the interactive curation Dash app |
| `AnalysisSummary` | Pre-loads all analysis tables and holds pre/post-curation atlases for summary generation |

---

## First-Time Setup

See [initial_setup.md](initial_setup.md) for the step-by-step instructions to configure your environment before running any metatlas2 workflow.

---

## Container-Based Deployment

metatlas2 is distributed as a **container image** hosted on the GitHub Container Registry (GHCR).  All Python dependencies are frozen inside the image, so analysts never need to clone the repository or manage a virtual environment.  Both the automated targeted analysis (running via Shifter on the compute cluster) and the interactive curation notebook (running via Shifter as a JupyterLab kernel) use the same image.

### Architecture at a Glance

Four distinct layers interact every time the entrypoint script is called.

| Layer | What lives here | Role |
|---|---|---|
| **GitHub** (`bkieft-usa/metatlas2`) | Source code, `Dockerfile`, CI workflow (`.github/workflows/docker.yml`) | Every push to `main` triggers a CI action, which builds and pushes a new container image to GHCR |
| **GHCR** (`ghcr.io/bkieft-usa/metatlas2`) | Frozen container images (e.g., `:latest`, `:sha-fb90592`) | The versioned Python runtime; pulled to the login node by a cronjob (every 5 min) or manually |
| **NERSC host filesystem** | `~/metatlas2/scripts/metatlas2.sh` (bash wrapper), `~/.jupyter/kernels/` (kernel specs), `$METATLAS_DATA_DIR/` (raw data + databases on shared CFS), `~/<owner>_metabolomics_data/` (project outputs) | Input data, user config, project outputs, and the thin shell scripts that glue everything together; **no Python runs here** |
| **Container** (Shifter process) | `/app/metatlas2/` (Python package + all deps, frozen at build time) | All Python execution; NERSC GPFS filesystems are auto-mounted at identical absolute paths — no path translation |

---

### Execution Modes

`metatlas2.sh` routes to one of four execution modes based on the subcommand.  The table below shows where Python actually runs and what distinguishes each mode:

| Subcommand | Where Python runs | Key distinction |
|---|---|---|
| `run` | Shifter container on the **login node** | Uses the host network by default; exposes the Dash curation server and kernel ZMQ ports so JupyterLab can reach them |
| `submit` | Shifter on login node writes the SLURM script, then **Shifter on a NERSC compute node** runs the workflow |
| `add-compounds` / `add-atlases` | Shifter container on the **login node** | GPFS is auto-mounted read-write so the container can write to `metatlas.duckdb` on the shared CFS |
| **Jupyter notebook** (Phases 4–5) | Shifter container launched by the **Jupyter kernel spec** on the JupyterHub spawner | JupyterHub connects to `ipykernel` inside the container over ZMQ; host network is used by default |

---

### Container File Map

```
metatlas2/
├── Dockerfile                        # Image definition; uv-based install; exposes IMAGE_TAG
├── .github/
│   └── workflows/
│       └── docker.yml                # CI: build + push on main push and version tags
└── scripts/
    ├── metatlas2.sh                     # Host wrapper script (run/submit, --image, --dev)
    ├── install_kernels.sh            # Registers metatlas2/metatlas2-dev/metatlas2-{tag} kernels
    └── pull_latest.sh                # Cronjob helper: shifterimg pull latest
```

---

### Image Registry and Tagging

Images are hosted at `ghcr.io/bkieft-usa/metatlas2`.  Every push to `main` builds a new image and pushes **two tags simultaneously**:

| Tag | Meaning |
|---|---|
| `sha-<7chars>` | 7-character short SHA of the triggering commit — unique, immutable, directly traceable to a GitHub commit |
| `latest` | Floating pointer to the most recent build; convenient for day-to-day use but not pinnable |

---

### Keeping the Local Cache Current

A cronjob pulls the latest image automatically in the background every 5 minutes to ensure the version on NERSC is up-to-date with the newest version in the container repository:

```bash
*/5 * * * * ~/metatlas2/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1
```

---

### Host Wrapper Script (`metatlas2.sh`)

The analyst never invokes `shifter` directly.  The wrapper script handles environment variables, optional volume mounts (dev mode), and the submit/sbatch split:

```bash
# Run the automated pre-curation workflow directly (e.g. from a login node)
metatlas2.sh run --config analysis.yaml --project MY_PROJECT_0000_0000_00 --rt-align-num 0 --analysis-num 0

# Generate a Shifter SLURM script and submit it immediately
metatlas2.sh submit --config analysis.yaml --project MY_PROJECT_0000_0000_00 --rt-align-num 0 --analysis-num 0

# Pin to a specific image tag instead of latest
metatlas2.sh --image sha-a1b2c3d run --config analysis.yaml --project MY_PROJECT_0000_0000_00 --rt-align-num 0 --analysis-num 0

# Use local working-tree edits instead of the installed image (dev mode)
metatlas2.sh --dev run --config analysis.yaml --project MY_PROJECT_0000_0000_00 --rt-align-num 0 --analysis-num 0
```

Shifter automatically mounts all NERSC GPFS filesystems (home, CFS, scratch) inside the container at the same absolute paths, so no explicit `-v` flags are needed for data access. The host network is used by default, so the Dash curation server and Jupyter kernel ZMQ ports are directly reachable by JupyterHub.

#### `submit` mode — container vs. host responsibility

`sbatch` is a host-side SLURM command not available inside the container.  The split works as follows:

1. The wrapper pre-generates a temp path with `mktemp /tmp/metatlas2_XXXXXX.sh` and calls `shifter … --entrypoint submit --script-only --output /tmp/metatlas2_XXXX.sh` — Python writes the SLURM `.sh` script to that path and exits. `/tmp` is accessible in the container because GPFS is auto-mounted and `/tmp` is on local storage which shifter also makes available.
2. The wrapper calls `sbatch` on the pre-known temp path directly.

The SLURM script uses `shifter --image=docker:ghcr.io/bkieft-usa/metatlas2:{tag}` so the batch job runs inside the same container image.  The image tag is embedded in the script at generation time; pass `--image v1.2.3` to the wrapper to pin a specific release for a batch job.

---

### Jupyter Kernel Specs

Three kernel specs are available:

| Kernel name | Image used | Source code | How it is registered |
|---|---|---|---|
| `metatlas2` | `latest` | Installed inside the image | Once, manually via `scripts/install_kernels.sh` during setup |
| `metatlas2-dev` | `latest` | Local repo's `metatlas2/` package mounted at `/app/metatlas2` | Once, manually via `scripts/install_kernels.sh` during setup |
| `metatlas2-<tag>` | `<tag>` | Installed inside the pinned image | **Automatically**, the first time `metatlas2.sh --image <tag> …` is called |

Generated curation notebooks embed the kernel name in their `kernelspec` metadata so the notebook re-opens against the same image that was used to generate it.  Analysts can switch kernels at any time in JupyterLab.

---

### Development Workflow

To test local changes without waiting for CI to build and push an image:

1. **Interactive / notebook**: Switch to the `metatlas2-dev` kernel.  The local repo's `metatlas2/` package directory is mounted at `/app/metatlas2` inside the container, directly overlaying the installed copy, so edits take effect on the next cell execution.
2. **CLI `run`, `add-compounds`, `add-atlases`**: Add `--dev` to the wrapper script — the local `metatlas2/` package directory is bind-mounted over the installed copy inside the container.
3. **SLURM batch**: Dev mode is not currently supported for SLURM batch jobs.

---

## Output Directory Layout

```
~/<owner>_metabolomics_data/<project_name>/
├── <project_name>.duckdb            # Project database
├── <project_short>.log              # Run log
├── RTA<N>/                          # One directory per rt_alignment_number
│   ├── rt_aligned_atlases.csv       # Atlas UIDs + metadata passed to auto-ID
│   └── (alignment plots)
│   └── TGA<M>/                      # One directory per analysis_number
│       ├── auto_ided_atlases.csv    # Post-auto-ID atlas snapshot
│       ├── curated_atlases.csv      # Post-curation atlas snapshot
│       └── (notebooks, summary files, figures)
```
