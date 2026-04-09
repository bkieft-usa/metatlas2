# Manual Curation and Analysis Summary

After running `run_targeted_analysis.py`, a Jupyter notebook is generated for each atlas (e.g. `HILICZ_ISTD_POS_RTA0_TGA0.ipynb`). Opening this notebook and sequentially running the cells launches an interactive curation GUI and — once curation is complete — runs all summary exports of the results.

---

## Step 1: Open the generated notebook

Notebooks are written to the analysis output directory:

```
<projects_dir>/<project_name>/RTA<N>/TGA<M>/
    HILICZ_ISTD_POS_RTA0_TGA0.ipynb
    HILICZ_EMA_POS_RTA0_TGA0.ipynb
    ...
```

Open the notebook in JupyterLab at NERSC on a login or dedicated node. Make sure the kernel is set to **metatlas2**.

The notebook has six cells:

| Cell | Type | Purpose |
|---|---|---|
| 1 | Markdown | Header — project name, chromatography, polarity, analysis type, and iteration numbers |
| 2 | Code | Imports — loads required libraries and sets up logging |
| 3 | Code | Variables — project name, config path, atlas UID, iteration numbers |
| 4 | Code | `OVERRIDE_PARAMS` — optional parameter overrides for the GUI session |
| 5 | Code | **GUI cell** — launches the interactive curation app |
| 6 | Code | **Summary cell** — exports all summary files after curation |

---

## Step 2: (Optional) Override parameters

Cell 4 contains an `OVERRIDE_PARAMS` dictionary. Any parameter set to a non-`None` value will override the corresponding value from the analysis configuration file (e.g., `analysis.yaml`) for this GUI session only. Leave a parameter as `None` to use the config value.

The purpose of this cell is to further filter the results or update the GUI interface (e.g., increase ms2_min_score to be more strict, change gui_lcmsruns_colors to change LCMS run file color displays in MS1 plots) without having to rerun a new targeted analysis.

```python
OVERRIDE_PARAMS = {
    'ms1_min_peak_intensity': None,     # current value: 100000.0
    'ms1_min_num_points':     None,     # current value: 5
    'ms2_min_score':          None,     # current value: 0.1
    'ms2_min_matching_frags': None,     # current value: 1
    'gui_lcmsruns_colors':    None,     # current value: {'ISTD': 'blue', ...}
}
```

---

## Step 3: Run the GUI cell

Run cell 5. The Dash app launches and link (localhost URL) is printed in the output. Click or copy the URL to open the curation GUI in a new browser tab.

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
| **Identification notes** | Free-text identification note saved with the compound, usually comes from the atlas to notify analyst. |
| **MS1 quality** | Radio — set the MS1 evaluation outcome (see table [below](#ms1-quality-options), various hotkeys). |
| **◀ Prev MS2 / Next MS2 ▶** | Cycle through available MS2 scans for this compound (hotkeys `l` or `;`). |
| **MS2 quality** | Radio — score the MS2 match (see table [below](#ms2-quality-options), various hotkeys). |
| **Other notes** | Radio — flag additional observations (see table [below](#other-notes-options)). |
| **Save and Exit** | Flush all outstanding/in-progress changes to the database (for saving progress or ending an analysis). |

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

Drag the dashed purple vertical RT-min and RT-max lines in the MS1 EIC plot to adjust the peak integration window, or use the hotkeys `a`, `s`, `d`, and `f` to move the rt_min left or right and rt_max left or right, respectively, by a default of 0.05 minutes per click. You can also use the `m` or `n` hotkeys to snap the rt_min and rt_max to the closest isomer and the suggested rt bounds (dashed organe vertical lines), respectively. Changes are saved to the database automatically when you navigate to the next compound or click **Save and Exit**.

---

## Step 4: Save and Exit

Click **Save and Exit** in the bottom-right corner of the GUI when curation is complete. This flushes all unsaved changes to the database and closes the GUI app. The button is disabled after clicking and a confirmation message appears, and you can safely close the browser tab and return to the notebook.

> **Important:** It is recommended but not required to click **Save and Exit** before running the Summary cell. If you do not, make sure there are no unsaved changes by navigating to a new compound (to initiate a databse flush) before closing the GUI browser tab. Changes are flushed to the analysis database continuously as you navigate, but the final flush is only guaranteed after clicking this button. If `gui_require_all_evaluated: true` is set in the config, the GUI will warn you if any compounds still have `ms2_notes = "no selection"`.

---

## Step 5: Run the Summary cell

Run cell 6 after closing the GUI. This calls the analysis summarizer, which:

1. Creates a post-curation atlas.
2. Saves the curated atlas UID to `curated_atlases.csv` in the analysis output directory.
3. Runs all summary export functions (described below).
4. Saves a copy of `analysis_config.yaml` to the analysis output directory for later recall.

---

## Summary outputs

All files are written to the analysis output directory (`TGA<N>/`) unless noted otherwise.

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

### Manual curation CSV

**`manually_curated_compound_data.csv`** — a flat CSV export of the full `manual_curation` database table. Contains one row per compound with all fields from the curation table including RT bounds, notes, auto-ID suggestions, and measurement statistics.

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

### Post-curation atlas CSV

**`<atlas_uid>.csv`** — the full post-curation atlas as a flat CSV, exported for downstream use.

### Config snapshot

**`analysis_config.yaml`** — a copy of the `analysis.yaml` used for this run, saved alongside the outputs for reproducibility.

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
