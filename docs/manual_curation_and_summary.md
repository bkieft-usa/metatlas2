# Manual Curation and Analysis Summary

After running `run_targeted_analysis.py`, a Jupyter curation notebook is generated for each targeted analysis. Opening this notebook and running the cells sequentially launches an interactive curation GUI and — once curation is complete — runs all summary exports of the results.

---

## Step 1: Open the generated notebook

Curation notebooks are written to the `TGA<M>/` directory inside the project output directory:

```
$METATLAS_DATA_DIR/projects/targeted_outputs/<owner>/<user>/<project_name>/RTA<N>/TGA<M>/
    CurationNotebook__HILIC-POS-ISTDsEtc__<project_name>.ipynb
    CurationNotebook__HILIC-NEG-ISTDsEtc__<project_name>.ipynb
    QueryNotebook__<project_name>.ipynb
    ...
```

The notebook filename follows the pattern `CurationNotebook__{CHROM}-{POL}-{ATLAS_NAME}__{project_name}.ipynb`, where `ATLAS_NAME` is the 4th-level key from the `TARGETED_ANALYSES` config (e.g., `EMA`). One curation notebook is generated per targeted analysis entry. A single `QueryNotebook__<project_name>.ipynb` is also generated for Parquet-based downstream queries (see [parquet_query.md](parquet_query.md)).

Open the curation notebook in NERSC JupyterHub. The kernel is pre-set to **metatlas2 (latest)** in the notebook metadata. If prompted to select a kernel, choose **metatlas2 (latest)**. If it is not listed, run `install_kernels.sh` from a NERSC terminal and refresh the JupyterHub page.
> **Note:** The metatlas2 kernels run inside a Shifter container, not on the NERSC host system. This means all Python packages and system libraries come from the container image, ensuring a consistent environment across all users. The container automatically mounts NERSC filesystems at the same paths, so your data files are directly accessible. For technical details about the containerized architecture, see [codebase_overview.md](codebase_overview.md#container-based-deployment). If you need to use the local repository to define the JupyterLab kernel (e.g., for development or to bypass a bug), select `metatlas2 (local repo)` from the kernel dropdown.


The notebook has five cells:

| Cell | Type | Purpose |
|---|---|---|
| 1 | Markdown | Header — project name |
| 2 | Code | Imports — loads required libraries and sets up logging |
| 3 | Code | `RUN_PARAMS` and `OVERRIDE_PARAMS` — fixed run identifiers and optional parameter overrides |
| 4 | Code | **GUI cell** — launches the interactive curation Dash app |
| 5 | Code | **Summary cell** — exports all summary files after curation |

---

## Step 2: (Optional) Override parameters

Cell 3 contains both a `RUN_PARAMS` dictionary (pre-filled at notebook generation time — do not edit) and an `OVERRIDE_PARAMS` dictionary. Any parameter in `OVERRIDE_PARAMS` set to a non-`None` value will override the corresponding value from the analysis configuration file (e.g., `analysis.yaml`) for this GUI session only. Leave a parameter as `None` to use the config value.

The purpose of this cell is to further filter the results or update the GUI interface (e.g., increase `ms2_min_score` to be more strict, change `gui_lcmsruns_colors` to change LCMS run file color displays in MS1 plots) without having to rerun a new targeted analysis.

```python
OVERRIDE_PARAMS = {
    # Analysis parameters
    'ms1_min_peak_intensity':       None,  # current value: 100000.0
    'ms1_min_num_points':           None,  # current value: 5
    'ms2_min_score':                None,  # current value: 0.25
    'ms2_min_matching_frags':       None,  # current value: 1
    'remove_unided_compounds':      None,  # current value: True
    'remove_flagged_compounds':     None,  # current value: True
    'apply_istd_curation_to_ref':   None,  # current value: True
    'apply_cross_polarity_curation': None, # current value: True
    'upload_to_gdrive':             None,  # current value: False
    # GUI parameters
    'gui_width':                    None,
    'gui_height':                   None,
    'gui_require_all_evaluated':    None,  # current value: False
    'gui_top_n_hits':               None,  # current value: 10
    'gui_lcmsruns_colors':          None,  # current value: {'ISTD': 'blue', ...}
    'note_options_overrides':       None,
}
```

> **Note:** The key `apply_istd_curation_to_ref` controls whether ISTD curation decisions are propagated to reference (EMA) compounds. This corresponds to the `apply_istd_curation_to_ema` concept in the config but uses the internal parameter name.

---

## Step 3: Run the GUI cell

Run cell 4. The Dash app launches in a background thread and a clickable link is displayed in the cell output:

```
▶ Open Dash App ↗
```

- **On NERSC JupyterHub:** The link is a JupyterHub proxy URL (e.g., `/user/<username>/proxy/8050/`). Click it to open the curation GUI in a new browser tab within your JupyterHub session.
- **On a local machine (Docker mode):** The link is `http://localhost:8050/`. Click it to open the GUI in your local browser.

The GUI cell blocks until you click **Save and Exit** in the GUI. Do not run the Summary cell (cell 5) until the GUI cell has finished executing.

---

## GUI layout

### Left panel — controls

| Element | Description |
|---|---|
| Dropdown | Select any compound by name or index. |
| **◀ Prev ID / Next ID ▶** | Navigate between compounds (hotkeys `j` or `k`). |
| **Accept Suggestions** | Apply the auto-identification's suggested RT bounds and MS1/MS2 notes (hotkey `n`). |
| **Snap to Isomer** | Cycle RT bounds to the next overlapping isomer's window (hotkey `m`). |
| **Analyst notes** | Free-text note saved with the compound (not used in scoring). |
| **Identification notes** | Free-text identification note from the atlas, displayed for analyst reference. |
| **MS1 quality** | Radio — set the MS1 evaluation outcome (see table [below](#ms1-quality-options), various hotkeys). |
| **◀ Prev MS2 / Next MS2 ▶** | Cycle through available MS2 scans for this compound (hotkeys `l` or `;`). |
| **MS2 quality** | Radio — score the MS2 match (see table [below](#ms2-quality-options), various hotkeys). |
| **Other notes** | Radio — flag additional observations (see table [below](#other-notes-options)). |
| **Save and Exit** | Flush all outstanding/in-progress changes to the database and shut down the Dash app. |

### Right panel — plots

| Plot | Description |
|---|---|
| **MS1 EIC** | Extracted ion chromatogram for the compound across all included LCMS runs. Vertical lines show atlas RT peak (solid black), current RT min/max bounds (dashed purple), and suggested RT bounds (dashed orange, if available). Drag the RT min/max boundary lines or use the hotkeys `a`, `s`, `d`, and `f` to adjust the peak window. |
| **MS2 Spectra** | Mirror plot of the current MS2 scan (query, top) against the best-matching reference spectrum (bottom). If no reference match exists, the raw MS2 spectrum is shown. If no MS2 data exist, the panel shows "No MS2 Data". |

---

## MS1 quality options

| Option | Hotkey | Meaning |
|---|---|---|
| `keep` | `1` | Compound peak is present and acceptable. |
| `remove` | `2` | No peak or peak is not usable — compound will be excluded from the post-curation atlas by default (can be kept with configuration parameter `remove_flagged_compounds` set to `False`). |
| `unresolvable isomers` | `3` | Peak is present but cannot be separated from a co-eluting isomer, still kept. |
| `poor peak shape` | `4` | Peak is detected but has poor chromatographic shape, still kept. |

---

## MS2 quality options

| Option | Hotkey | Meaning |
|---|---|---|
| `no selection` | `q` | Not yet evaluated (cannot be selected if configuration parameter `gui_require_all_evaluated` is `True`). |
| `-1.0, poor match, should remove` | `w` | MS2 hit is present but clearly incorrect — compound should be removed. |
| `0.0, no match or no MSMS collected` | `e` | No MS2 data or no library match. |
| `0.5, partial or putative match of fragments` | `r` | Some fragment ions match but confidence is limited. |
| `1.0, good match` | `t` | Strong library match. |
| `0.5, co-isolated precursor, partial match` | `y` | Co-isolated precursor resulting in a partial fragmentation match. |
| `1.0, co-isolated precursor, good match` | `u` | Co-isolated precursor resulting in a good fragmentation match. |
| `0.5, single ion match, no evidence` | `i` | Only one fragment matched (could be incorrect or precursor), but no additional evidence. |
| `1.0, single ion match, ISTD/ref evidence` | `o` | Only one fragment matched but supported by ISTD or reference standard evidence. |

---

## Other notes options

| Option | Hotkey | Meaning |
|---|---|---|
| `no selection` | `5` | No additional note (default). |
| `potential rt shifting` | `6` | RT may be drifting across experiment. |
| `high ppm diff` | `7` | m/z error is higher than expected for this compound. |
| `noisy or high background` | `8` | MS1 EIC plot is noisy or background/blank is high. |
| `needs review` | `9` | Flag for further general review, usually accompanies a description in the analyst notes box. |

---

## Keyboard shortcuts

All hotkeys are active when focus is on the GUI page (not in a text input field).

| Key | Action |
|---|---|
| `j` | Previous compound |
| `k` | Next compound |
| `n` | Accept suggestions |
| `m` | Snap to isomer |
| `l` | Previous MS2 scan |
| `;` | Next MS2 scan |
| `1`–`4` | MS1 quality options |
| `q`–`o` | MS2 quality options (see table above) |
| `5`–`9` | Other notes options |

---

## Adjusting RT bounds

Drag the dashed purple vertical RT-min and RT-max lines in the MS1 EIC plot to adjust the peak integration window, or use the hotkeys `a`, `s`, `d`, and `f` to move the rt_min left or right and rt_max left or right, respectively, by a default of 0.05 minutes per click. You can also use the `m` or `n` hotkeys to snap the rt_min and rt_max to the closest isomer and the suggested rt bounds (dashed orange vertical lines), respectively. Changes are saved to the database automatically when you navigate to the next compound or click **Save and Exit**.

---

## Step 4: Save and Exit

Click **Save and Exit** in the bottom-right corner of the GUI when curation is complete. This flushes all unsaved changes to the database, shuts down the Dash app server, and unblocks the GUI cell so it finishes executing. The button is disabled after clicking and a confirmation message appears. You can then safely close the browser tab and return to the notebook.

> **Important:** Always click **Save and Exit** before running the Summary cell. The GUI cell does not finish executing until **Save and Exit** is clicked — the Summary cell (cell 5) cannot be run while the GUI cell is still blocking. If any compound curation data could not be saved to the database (e.g., due to a database lock), a warning is printed in the cell output and the `DB_ERR` variable will be non-empty. Do **not** run the Summary cell if `DB_ERR` is non-empty — resolve the error first and re-curate the affected compounds. If `gui_require_all_evaluated: true` is set in the config, the GUI will warn you if any compounds still have `ms2_notes = "no selection"`.

---

## Step 5: Run the Summary cell

Run cell 5 after the GUI cell has finished executing. The Summary cell calls `wfs.run_analysis_summary()`, which:

1. Checks `DB_ERR` — if any compound curation data failed to save during the GUI session, the summary is aborted immediately with an error message listing the affected compounds.
2. Creates a post-curation (`MANUALLY_CURATED`) atlas from the analyst's accepted identifications.
3. Runs all summary export functions (described below).
4. Optionally uploads outputs to Google Drive (if `upload_to_gdrive: true` in the config).
5. Changes group ownership of the project output folder to the shared `metatlas` group.

All summary files are written to the analysis results output directory: `TGA<M>/<CHROM>-<POL>-<ATLAS_NAME>/`.

---

## Summary outputs

All files are written to the analysis results output directory (`TGA<N>/<CHROM>-<POL>-<ATLAS_NAME>/`) unless noted otherwise.

### Identification figures

**One PDF per compound**, named by compound, adduct, and index.

Each figure is a 3-row × 4-column layout:

```
┌─────────────────┬────────────────┬────────────────┬───────────┐
│ Best MS2 mirror │ 2nd best MS2   │ 3rd best MS2   │ Structure │  Row 1
├─────────────────┴────────────────┼────────────────┴───────────┤
│    EIC (linear scale)            │    EIC (log₁₀ scale)       │  Row 2
├──────────────────────────────────┴────────────────────────────┤
│         MS2 hit summary table (best match metrics)            │  Row 3
└───────────────────────────────────────────────────────────────┘
```

- **Row 1:** Mirror plots for the top 3 MS2 hits (query spectrum above, reference below). Panels with no match show "No MS2 Data".
- **Row 1, column 4:** Molecular structure drawn from SMILES (RDKit); falls back to InChI, then InChIKey label.
- **Row 2:** EIC traces colour-coded by run type. Atlas RT bounds (purple lines) and peak RT (black line) are overlaid.
- **Row 3:** Best match summary table — theoretical vs. measured m/z and RT, ppm error, RT error, cosine score, and number of matched fragment ions.

### EIC thumbnails

**Two PDF files** — one with a shared y-axis scale across all compounds, one with independent y-axis scaling per compound.

Each page is a 5×5 grid of EIC thumbnails (25 compounds per page). Thumbnails show the peak window with atlas RT bounds and are colour-coded by a consistent compound index.

### Identification summary table

**One Excel file** named `<project>_RTA<N>_TGA<M>_Final_Identifications.xlsx`.

One row per compound with the following column groups:

| Column group | Contents |
|---|---|
| Compound metadata | Name, formula, adduct, polarity, chromatography, InChIKey |
| Atlas reference values | Atlas m/z, atlas RT peak, atlas RT min/max |
| Best MS1 measurements | Best measured m/z, RT, ppm error, RT error |
| MS2 identification | Best cosine score, number of matching fragments, reference compound name |
| Quality scores | MS1 quality (0/0.5/1), MS2 quality (0/0.5/1), RT quality (0/0.5/1), total score (0–3) |
| MSI level | Metabolomics Standards Initiative confidence level (MSI-1 through MSI-4) |
| Curation notes | MS1 note, MS2 note, other note, analyst notes, identification notes |
| Overlapping compounds | Co-eluting / isobaric compounds detected in the same atlas |

### Boxplots

**Six PDF files** — three metrics × two y-axis scales (linear and log₁₀):

| File | Metric |
|---|---|
| `peak_height_linear.pdf` / `peak_height_log.pdf` | MS1 peak height per compound per run |
| `rt_error_linear.pdf` / `rt_error_log.pdf` | RT error (measured − atlas RT) per compound per run |
| `ppm_error_linear.pdf` / `ppm_error_log.pdf` | m/z ppm error per compound per run |

Each box represents one compound; points are coloured by run type (ISTD, QC, EXCTRL, etc.).

### Quantitative data sheets

**Multiple CSV files** — filtered peak height data and log fold change tables:

| File | Contents |
|---|---|
| `peak_height_filtered.csv` | Filtered peak heights per compound per run (after applying `remove_flagged_compounds` and `remove_unided_compounds`) |
| `group_mean_log_fold_changes.csv` | Log₂ fold changes computed from filtered peak heights |

### Best MS2 hit fragment ions CSV

**`best_ms2_hit_fragment_ions.csv`** — one row per compound (compounds with no MS2 hits are omitted). Contains:

| Column | Description |
|---|---|
| `compound_index` | 1-based integer matching the compound's position in the curation table |
| `compound_name` | Compound name |
| `adduct` | Ion adduct |
| `file_name` | Basename of the file containing the best-scoring MS2 hit |
| `rt_peak` | Retention time of the best-scoring MS2 scan (min) |
| `mz_peak` | Measured precursor m/z |
| `spectrum` | Query fragment spectrum as JSON `[[mz0, mz1, …], [int0, int1, …]]` (fragments below 1×10⁴ intensity are filtered out) |

### Metabomap

**`metabomap.csv`** — a merged positive/negative mode peak height and log fold change table, combining results across polarities for the same project. Only generated when both polarities have been analyzed.

### Analysis Parquet files

**Parquet output files** written to `$METATLAS_DATA_DIR/projects/parquet_outputs/` — unified compound-level and compound-file-level data in Parquet format for downstream programmatic analysis. These can be queried using the `QueryNotebook__<project_name>.ipynb` notebook generated in the `TGA<M>/` directory (see [parquet_query.md](parquet_query.md)).

---

## MSI confidence levels

The identification summary table assigns an MSI (Metabolomics Standards Initiative) confidence level based on the combination of MS2, m/z, and RT quality scores:

| Level | Criteria |
|---|---|
| **MSI-1** | MS2 score = 1.0 **and** at least one of m/z or RT score = 1.0 |
| **MSI-2** | Total score ≥ 1.5 (but not MSI-1) |
| **MSI-3** | At least one score component is non-zero |
| **MSI-4** | No quality evidence (all scores zero or missing) |

Individual quality score thresholds:

| Metric | Score 1.0 | Score 0.5 | Score 0.0 |
|---|---|---|---|
| **m/z** | ppm ≤ 5 or Δm/z ≤ 0.0015 Da | ppm ≤ 10 | ppm > 10 |
| **RT (HILICZ)** | RT error ≤ 0.5 min | RT error ≤ 1.0 min | RT error > 1.0 min |
| **RT (C18)** | RT error ≤ 0.3 min | RT error ≤ 0.5 min | RT error > 0.5 min |
| **MS2** | Cosine score from MS2 quality radio selection (0, 0.5, or 1.0) | — | — |

---

## Skipping manual curation (automated pipeline)

If manual curation is not required, pass `--skip-curation` to `metatlas2.sh run`. This applies automatic curation defaults to all compounds and runs the analysis summary immediately after auto-identification:

- `ms1_notes` → `"keep"` (for all compounds that passed auto-ID)
- `ms2_notes` → `"0.5, curation skipped, putative match"`
- `analyst_notes` → appends `"(manual curation skipped)"`

The summary outputs are identical to the manual curation path, but no GUI notebook is opened.
