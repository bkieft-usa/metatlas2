import pandas as pd

def load_compound_input(file_path: str) -> pd.DataFrame:
    """Load compound data from a CSV or TSV file."""
    print(f"Loading input data from: {file_path}")
    delimiter = '\t' if file_path.endswith(('.tsv', '.tab', '.txt')) else ','
    df = pd.read_csv(file_path, sep=delimiter)
    required_columns = ['inchi_key', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"Loaded {len(df)} rows from input table")
    return df

def load_atlas_input(file_path: str) -> pd.DataFrame:
    """Load compound data from a CSV or TSV file."""
    print(f"Loading input data from: {file_path}")
    delimiter = '\t' if file_path.endswith(('.tsv', '.tab', '.txt')) else ','
    df = pd.read_csv(file_path, sep=delimiter)
    required_columns = ['inchi_key', 'chromatography', 'polarity', 'rt_peak', 'mz', 'adduct']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if 'label' not in df.columns:
        df['label'] = df['inchi_key']

    chrom = df['chromatography'].dropna()
    pol = df['polarity'].dropna()
    if len(chrom.unique()) > 1:
        raise ValueError(f"Multiple chromatography methods found in input table: {chrom.unique()}")
    if len(pol.unique()) > 1:
        raise ValueError(f"Multiple polarities found in input table: {pol.unique()}")

    detected_chrom = chrom.iloc[0]
    detected_pol = pol.iloc[0]

    print(f"    Detected chromatography: {detected_chrom}")
    print(f"    Detected polarity: {detected_pol}")

    print(f"    Loaded {len(df)} rows from input table")
    return df, detected_chrom, detected_pol