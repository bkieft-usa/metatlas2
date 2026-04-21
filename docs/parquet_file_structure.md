# Parquet File Structure

This document describes the structure and organization of `.parquet` files generated from mzML data during the .raw conversion process.

---

## Table of Contents

- [Overview](#overview)
- [File Organization](#file-organization)
- [File Naming Convention](#file-naming-convention)
- [Data Schema](#data-schema)
  - [MS1 Schema](#ms1-schema)
  - [MS2 Schema](#ms2-schema)
- [Data Types and Units](#data-types-and-units)
- [File Properties](#file-properties)
- [Data Processing](#data-processing)
- [Example Usage](#example-usage)

---

## Overview

Parquet files are created as part of the raw file conversion pipeline (`scripts/convert_raw_files.py`). Each mzML file is processed to extract relevant MSMS data (see [below](#data-schema)) and split it into separate parquet files based on:
- **MS Level**: MS1 (full scan) or MS2 (fragmentation/tandem MS)
- **Polarity**: Positive or Negative ion mode

This separation allows for efficient querying and analysis of specific data subsets without loading unnecessary data during targeted analysis.

---

## File Organization

Parquet files are organized within project directories:

```
project_directory/
├── raw/              # Original .raw files
├── mzML/             # Converted .mzML files
├── h5/               # HDF5 format files (legacy)
└── parquet/          # Parquet format files
    ├── {filename}_ms1_pos.parquet
    ├── {filename}_ms1_neg.parquet
    ├── {filename}_ms2_pos.parquet
    └── {filename}_ms2_neg.parquet
```

**Note**: Not all four parquet files are created for every mzML file. The files created depend on the acquisition method:
- **MS1-only runs** produce only `ms1_pos` and/or `ms1_neg` files
- **MS2 runs** produce both MS1 and MS2 files (MS1 data is collected during MS2 runs)
- **Single polarity runs** produce only positive or negative files
- **FPS (Fast Polarity Switching) runs** produce both positive and negative files (polarities split)

---

## File Naming Convention

The filename determines which parquet files should exist. The convention is:

```
{project_info}_{polarity}_{ms_level}_{additional_info}
```

Where:
- **Position 9** (0-indexed): Polarity indicator
  - `FPS`: Fast Polarity Switching (both pos and neg)
  - `POS`: Positive ion mode only
  - `NEG`: Negative ion mode only
- **Position 10**: MS level
  - `MS1`: MS1 data only
  - `MS2`: Both MS1 and MS2 data

**Examples**:
- `20201015_sample_FPS_MS2_...` → Creates: `ms1_pos`, `ms1_neg`, `ms2_pos`, `ms2_neg`
- `20201015_sample_POS_MS1_...` → Creates: `ms1_pos` only
- `20201015_sample_NEG_MS2_...` → Creates: `ms1_neg`, `ms2_neg`

---

## Data Schema

### MS1 Schema

MS1 (full scan) parquet files contain the following columns:

| Column     | Data Type | Description                                    |
|------------|-----------|------------------------------------------------|
| `mz`       | float32   | Mass-to-charge ratio (m/z) of the detected ion |
| `i`        | float32   | Intensity (abundance) of the ion               |
| `rt`       | float32   | Retention time in minutes                      |
| `polarity` | int16     | Polarity mode: 0 = negative, 1 = positive      |

**Example row**:
```
mz: 202.0814, i: 45823.5, rt: 5.23, polarity: 1
```

### MS2 Schema

MS2 (fragmentation/tandem MS) parquet files contain additional precursor information:

| Column                 | Data Type | Description                                    |
|------------------------|-----------|------------------------------------------------|
| `mz`                   | float32   | Mass-to-charge ratio of the fragment ion       |
| `i`                    | float32   | Intensity of the fragment ion                  |
| `rt`                   | float32   | Retention time in minutes (scan)               |
| `polarity`             | int16     | Polarity mode: 0 = negative, 1 = positive      |
| `precursor_MZ`         | float32   | m/z of the precursor ion that was fragmented   |
| `precursor_intensity`  | float32   | Intensity of the precursor ion                 |
| `collision_energy`     | float32   | Collision energy used for fragmentation        |

**Example row**:
```
mz: 119.0342, i: 8234.2, rt: 5.23, polarity: 1,
precursor_MZ: 202.0814, precursor_intensity: 45823.5, collision_energy: 20.0
```

---

## Data Types and Units

| Field                  | Unit / Range           | Notes                                           |
|------------------------|------------------------|-------------------------------------------------|
| `mz`                   | Daltons (Da)           | Typically 50-2000 Da for metabolomics           |
| `i`                    | Arbitrary units        | Detector response intensity                     |
| `rt`                   | Minutes                | Scan time from start of LC run                  |
| `polarity`             | 0 or 1                 | Binary encoding: 0 = negative, 1 = positive     |
| `precursor_MZ`         | Daltons (Da)           | m/z of isolated ion before fragmentation        |
| `precursor_intensity`  | Arbitrary units        | Intensity of precursor in MS1 scan              |
| `collision_energy`     | eV or %                | Energy applied for fragmentation                |

Note: The .parquet/.mzML file CE value of **23.3333339691162** corresponds to the human-readable value **CE102040** and **43.3333320617676** corresponds to **CE205060**.

---

## File Properties

Parquet files are written with the following characteristics:

- **Compression**: Snappy compression algorithm
- **Dictionary Encoding**: Enabled for efficient storage of repeated values
- **Row Group Size**: 100,000 rows per row group
- **Statistics**: Column statistics enabled for query optimization
- **Data Page Size**: 1 MB
- **Sorting**: All data is sorted by `mz` (ascending) for efficient range queries

---

## Data Processing

### Quality Control

1. **Easy-IC Lock Mass Filtering** (optional, default enabled):
   - Removes lock mass calibration signals at m/z 202.07770 (positive) and 202.07880 (negative)
   - Tolerance: ±0.001 Da
   - This removes an internal standard used by ThermoFisher

2. **Polarity Separation**:
   - Each spectrum is classified as positive (1) or negative (0) polarity
   - Data is routed to separate files based on polarity

3. **MS Level Separation**:
   - MS1 spectra contain full scan data
   - MS2 spectra contain fragmentation data with precursor information

### Empty Files

If a file type is expected based on the filename convention but contains no data (e.g., no positive ions detected in a positive mode run), an empty parquet file is still created with the correct schema but zero rows. This is logged as a warning.

---

## Example Usage

### Reading and Filtering Parquet Files in Python

```python
import pyarrow.parquet as pq

# Find all compounds eluting within the RT window of 2.15 and 2.25
df = pq.read_table('sample_ms1_pos.parquet').to_pandas()
rt_compounds = df[
  (df['rt'] > 2.15) &
  (df['rt'] < 2.25)
]

# Find all fragments from precursor m/z 202.08 ± 0.01 Da
df = pq.read_table('sample_ms2_pos.parquet').to_pandas()
fragments = df[
    (df['precursor_MZ'] >= 202.07) & 
    (df['precursor_MZ'] <= 202.09)
]
```

---
