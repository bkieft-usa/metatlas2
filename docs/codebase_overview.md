# metatlas2 Codebase Overview

A programmer-oriented reference for understanding how a typical targeted metabolomics workflow unfolds: which objects are created at each stage, which functions are called in order, and what each one does.

---

## Module Map

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

---

## Adding to the central metatlas knowledge store

These steps run once per instrument method / compound list update. They populate the **main shared database** (`metatlas.duckdb`).

### 1. Create new Compounds in the main database — `Compound.create_from_config(config_path)`

**Objects created:** `Compound`, `CompoundMZRT` (one per compound row in the input file)

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

**Objects created:** `Atlas` (one per chromatography × polarity × analysis_type combination)

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

The three main phases are then called in sequence unless individually skipped.

---

### Phase 1 — Project Setup: `wfs.run_project_setup(...)`

**Objects created:** `Project`, `LCMSRun` (one per parquet file found on disk)

| Call | What it does |
|---|---|
| `Project()` | Instantiates an empty project container dataclass |
| `Project.setup(project_name, config, paths, overwrite_existing)` | Orchestrates all setup sub-steps below |
| `dbi.create_project_database(project_db_path, rt_align_path, overwrite)` | Creates the project-scoped DuckDB file; returns early if it already exists and overwrite is False |
| `lrt.get_project_lcmsruns_from_disk(raw_data_directory)` | Walks the raw data directory to discover parquet files, inferring file type, chromatography, polarity, and MS level from filenames |
| `dbi.save_lcmsruns_to_db(project_db_path, project_name, lcmsruns_list, overwrite)` | Writes the run metadata list to the `lcmsruns` table in the project database |
| `LCMSRun(**row)` | Wraps each run's metadata dict into a typed `LCMSRun` dataclass stored in `Project.lcmsruns` |

**State after this phase:** Project database exists with a populated `lcmsruns` table.

---

### Phase 2 — RT Alignment: `wfs.run_rt_alignment(...)`

**Objects created:** `RTAlign`, `Atlas` (template + one aligned atlas per analysis type), `ExperimentalData`

| Call | What it does |
|---|---|
| `RTAlign()` | Instantiates the RT alignment state container |
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

**Objects created (per atlas loop):** `AutoIdentification`, `ExperimentalData`, `ManualCuration` entries, `Atlas` (post-autoid)

| Call | What it does |
|---|---|
| `AutoIdentification()` | Instantiates the auto-ID state container |
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

Launched from a generated curation notebook. Runs an interactive Dash app for manual review.

**Objects created:** `AnalysisGUI`, `Atlas`

| Call | What it does |
|---|---|
| `AnalysisGUI()` | Instantiates the GUI state container |
| `AnalysisGUI.setup(project_name, rt_alignment_number, analysis_number, config, paths)` | Populates metadata and paths |
| `Atlas.from_database(project_db_path, pre_curation_atlas_uid, main_db_path)` | Loads the pre-curation (post-auto-ID) atlas into `AnalysisGUI.pre_curation_atlas_obj` |
| `dbi.load_and_filter_gui_inputs(analysis_gui_obj, override_parameters)` | Queries the project DB for all MS1, MS2, hits, and manual-curation data; applies any analyst-supplied filter overrides; stores DataFrames in `AnalysisGUI.ms1_df`, `.ms2_df`, `.ms2_hits_df`, `.manual_curation_df` |
| `agu.build_dash_app(analysis_gui_obj, port, shutdown_holder)` | Constructs the Dash application object with all callbacks and layouts bound to the GUI state |
| `make_server(...)` + `threading.Thread(target=server.serve_forever)` | Starts a Werkzeug WSGI server in a background thread serving the Dash app |

**State after this phase:** Analyst uses the GUI to accept/reject identifications and adjust RT windows; updated curation rows are written back to the project DB during the session.

---

### Phase 5 — Analysis Summary: `wfs.run_analysis_summary(...)`

Launched from a generated notebook after curation is complete.

**Objects created:** `AnalysisSummary`, `Atlas` (post-curation)

| Call | What it does |
|---|---|
| `AnalysisSummary()` | Instantiates the summary state container |
| `AnalysisSummary.setup(project_name, rt_alignment_number, analysis_number, config, paths)` | Populates metadata then immediately calls `load_data()` |
| `AnalysisSummary.load_data()` | Pre-loads all four data tables from the project DB (`manual_curation_df`, `ms1_all_df`, `ms2_raw_all_df`, `ms2_hits_all_df`) plus `per_file_metrics_df` derived from MS1 data so summary functions don't re-query |
| `Atlas.from_database(project_db_path, pre_curation_atlas_uid, main_db_path)` | Loads the pre-curation atlas into `AnalysisSummary.pre_curation_atlas_obj` |
| `dbi.create_new_atlas_after_manual_curation(summary_obj)` | Builds a final curated `Atlas` from the analyst's accepted identifications; stored in `AnalysisSummary.post_curation_atlas_obj` |
| `ldt.save_atlas_data_to_csv(post_curation_atlas_obj, curated_atlases_store_file)` | Saves the curated atlas to `TGA<N>/curated_atlases.csv` |
| `asm.run_all_summaries(summary_obj, overwrite)` | Generates all output files: per-compound plots, QC tables, identification-confidence summaries, and export CSVs |

**State after this phase:** All summary files are written to `TGA<N>/`; the curated atlas CSV is ready for downstream use.

---

## Key Data Objects

| Object | Module | Purpose |
|---|---|---|
| `Compound` | `workflow_objects` | Immutable chemical identity record (name, InChI key, formula, PubChem CID, etc.) |
| `CompoundMZRT` | `workflow_objects` | Reference RT/MZ window for a compound under a specific chromatography/polarity/adduct |
| `Atlas` | `workflow_objects` | Named collection of `CompoundMZRT` entries for one chromatography × polarity × analysis type |
| `LCMSRun` | `workflow_objects` | Metadata record for a single raw parquet file (path, type, polarity, MS level) |
| `Project` | `workflow_objects` | Container holding project config, paths, and `LCMSRun` list during setup |
| `RTAlign` | `workflow_objects` | Carries RT alignment state: QC atlas, filtered runs, model coefficients, and aligned atlases |
| `ExperimentalData` | `workflow_objects` | Holds extracted `MS1Data`, `MS2Data`, `MS2Hit`, and `ManualCuration` lists for one atlas × LCMS run set |
| `ManualCuration` | `workflow_objects` | Per-compound identification summary: best MS1 file, RT error, auto-ID flag, suggested RT bounds |
| `MS1Data` / `MS2Data` / `MS2Hit` | `workflow_objects` | Thin wrappers (`_SpecData`) around a per-file DataFrame of spectral data or hit scores |
| `AutoIdentification` | `workflow_objects` | Carries auto-ID state: pre/post atlases, filtered runs, `ExperimentalData`, and results |
| `AnalysisGUI` | `workflow_objects` | Holds in-memory DataFrames and atlas objects for the interactive curation Dash app |
| `AnalysisSummary` | `workflow_objects` | Pre-loads all analysis tables and holds pre/post-curation atlases for summary generation |

---

## Container-Based Deployment

metatlas2 is distributed as a **Podman/Docker container** hosted on the GitHub Container Registry (GHCR).  All Python dependencies are frozen inside the image, so analysts never need to clone the repository or manage a virtual environment.  The workflow components that run on the compute cluster (batch jobs via Shifter) and those that run interactively in JupyterLab (the curation notebook) both use the same image.

---

### Image Registry and Tagging

Images are hosted at `ghcr.io/bkieft-usa/metatlas2` and follow a two-tag convention:

| Tag | When it is built | Meaning |
|---|---|---|
| `latest` | Every push to `main` | The current HEAD of the main branch |
| `v<MAJOR>.<MINOR>.<PATCH>` | Every semver git tag (e.g. `git tag v1.2.3 && git push --tags`) | A pinned, reproducible release |

The GitHub Actions workflow (`.github/workflows/docker.yml`) builds and pushes the image automatically.  The `IMAGE_TAG` build-arg is injected at build time and exposed as the `METATLAS2_IMAGE_TAG` environment variable inside the container, so any code running inside knows exactly which version it is.

The `AutoIdentification` dataclass records `image_tag` and `config_path` as fields set during `setup()`.  `notebook_generator.py` reads `auto_id_obj.image_tag` when writing the curation notebook so the image version used at analysis time is permanently embedded in the notebook metadata and variables cell.

---

### Keeping the Local Cache Current

A cronjob pulls the latest image automatically in the background:

```bash
*/5 * * * * /path/to/metatlas2/scripts/pull_latest.sh >> ~/pull_metatlas2.log 2>&1
```

`scripts/pull_latest.sh` runs `podman pull ghcr.io/bkieft-usa/metatlas2:latest` and logs the result with a timestamp.  The first manual pull after initial setup is also done with this script.

---

### Host Wrapper Script (`scripts/metatlas2`)

The analyst never invokes `podman run` directly.  The wrapper script handles volume mounts, environment variables, and the submit/sbatch split:

```bash
# Run the automated pre-curation workflow directly (e.g. from a login node)
scripts/metatlas2 run --config analysis.yaml --project MY_PROJECT_0000_0000_00

# Generate a Shifter SLURM script and submit it immediately
scripts/metatlas2 submit --config analysis.yaml --project MY_PROJECT_0000_0000_00 --qos regular

# Pin to a specific image tag instead of latest
scripts/metatlas2 --image v1.2.3 run --config analysis.yaml --project MY_PROJECT_0000_0000_00

# Use local working-tree edits instead of the installed image (dev mode)
scripts/metatlas2 --dev run --config analysis.yaml --project MY_PROJECT_0000_0000_00
```

The wrapper always mounts:
- `/pscratch/sd/b/bkieft/metatlas_lite_data` (read-only) — raw data, main DB, PubChem cache
- `$HOME` — project output directories, config files, notebooks

`--network=host` is passed for `run` mode so the Dash curation server and Jupyter kernel ZMQ ports bind directly on the host network stack, allowing JupyterHub to reach them.

#### `submit` mode — container vs. host responsibility

`sbatch` is a host-side SLURM command not available inside the container.  The split works as follows:

1. The wrapper calls `podman run … submit --script-only --output /tmp/metatlas2_XXXX.sh` — Python generates and writes the SLURM `.sh` script to a temp path on `/tmp` (visible to both container and host via the bind mount), then exits.
2. The wrapper reads the temp path from stdout and calls `sbatch /tmp/metatlas2_XXXX.sh` on the host.

The SLURM script uses `shifter --image=docker:ghcr.io/bkieft-usa/metatlas2:{tag}` so the batch job runs inside the same container image.  The image tag is embedded in the script at generation time; pass `--image v1.2.3` to the wrapper to pin a specific release for a batch job.

---

### Jupyter Kernel Specs

The curation notebooks require Python packages from inside the container.  Rather than installing packages on the host, a Jupyter kernel spec is registered that launches an `ipykernel` process inside a Podman container.  JupyterLab connects to it over ZMQ using `--network=host`.

Run once (or after a new version release) to register the kernel specs:

```bash
# Install 'metatlas2' (latest) and 'metatlas2-dev' kernels
scripts/install_kernels.sh

# Also install a pinned kernel for a specific release tag
scripts/install_kernels.sh --tag v1.2.3
```

Three kernel specs are available:

| Kernel name | Image used | Source code |
|---|---|---|
| `metatlas2` | `latest` | Installed inside the image |
| `metatlas2-dev` | `latest` | Local repo mounted at `/dev_repo`; `PYTHONPATH=/dev_repo` shadows installed code |
| `metatlas2-{tag}` | `{tag}` | Installed inside the pinned image |

Generated curation notebooks embed the kernel name in their `kernelspec` metadata.  When the analysis was run with a specific tag (e.g. `v1.2.3`), the notebook targets `metatlas2-v1.2.3`; when run with `latest`, it targets `metatlas2`.  Analysts can switch kernels at any time via **Kernel → Change Kernel…** in JupyterLab and update the `IMAGE_TAG` variable in the variables cell to match.

---

### Development Workflow

To test local changes without waiting for CI to build and push an image:

1. **Interactive / notebook**: Switch to the `metatlas2-dev` kernel.  The local repo at `/global/homes/b/bkieft/metatlas2` is mounted into the container and placed first on `PYTHONPATH`, so edits take effect on the next cell execution.
2. **CLI `run` mode**: Add `--dev` to the wrapper:
   ```bash
   scripts/metatlas2 --dev run --config analysis.yaml --project ...
   ```
3. **SLURM batch**: Add `--dev` to the wrapper's `submit` call.  The generated SLURM script passes `--volume` and `PYTHONPATH` arguments to Shifter.

When changes are ready, push to `main` (or tag a release) and the CI pipeline automatically builds and pushes the updated image to GHCR.  The cronjob then pulls it within 5 minutes.

---

### Container File Map

```
metatlas2/
├── Dockerfile                        # Image definition; uv-based install; exposes IMAGE_TAG
├── .github/
│   └── workflows/
│       └── docker.yml                # CI: build + push on main push and version tags
└── scripts/
    ├── metatlas2                     # Host wrapper (run/submit, --image, --dev)
    ├── install_kernels.sh            # Registers metatlas2/metatlas2-dev/metatlas2-{tag} kernels
    └── pull_latest.sh                # Cronjob helper: podman pull latest
```

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
