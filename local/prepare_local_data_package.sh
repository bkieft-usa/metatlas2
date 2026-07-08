#!/bin/bash
# Prepare standalone development environment data package
#
# This script must be run at NERSC with access to production data.
# It extracts a minimal set of files and creates the dev package structure.
#
# Usage:
#   cd /global/homes/b/bkieft/metatlas2
#   ./local/prepare_dev_package.sh [output_dir] [--keep-existing-archive] [--no-h5-subset] [--skip-zenodo-upload]
#
# Options:
#   --keep-existing-archive  Skip package creation and only upload existing tarball to Zenodo
#   --no-h5-subset  Skip filtering copied h5 files (filtering is on by default)
#   --skip-zenodo-upload  Skip upload to Zenodo (useful for testing)
# Output:
#   Creates metatlas2-dev-data.tar.gz ready for Zenodo upload

# If .venv is not available, run uv sync to build it
if [[ ! -d "/global/cfs/cdirs/metatlas/tools/metatlas2/.venv" ]]; then
    echo "Virtual environment not found. Running uv sync to create it..."
    uv sync
    exit 1
fi

# activate env with 'source /global/cfs/cdirs/metatlas/tools/metatlas2/.venv/bin/activate'
source /global/cfs/cdirs/metatlas/tools/metatlas2/.venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse arguments
KEEP_EXISTING=false
SUBSET_H5=true
SKIP_ZENODO_UPLOAD=false
OUTPUT_DIR=""

for arg in "$@"; do
    if [[ "$arg" == "--keep-existing-archive" ]]; then
        KEEP_EXISTING=true
    elif [[ "$arg" == "--no-h5-subset" ]]; then
        SUBSET_H5=false
    elif [[ "$arg" == "--skip-zenodo-upload" ]]; then
        SKIP_ZENODO_UPLOAD=true
    elif [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$arg"
    fi
done

# Set default output directory if not provided
OUTPUT_DIR="${OUTPUT_DIR:-${METATLAS_DATA_DIR}/databases/standalone_dev_data}"
PACKAGE_NAME="metatlas2-dev-data"

# Source project with representative ISTD data
SOURCE_PROJECT="20260311_JGI_AE_511825_SorghAnth_final_EXP120B_HILICZ_USHXG03401"
SOURCE_OWNER="jgi"
SOURCE_BASE="${METATLAS_DATA_DIR}/raw_data/${SOURCE_OWNER}/${SOURCE_PROJECT}"

echo "========================================"
echo "Metatlas2 Dev Package Preparation"
echo "========================================"
echo "Source project: ${SOURCE_PROJECT}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Validate METATLAS_DATA_DIR is set
if [[ -z "${METATLAS_DATA_DIR:-}" ]]; then
    echo "Error: METATLAS_DATA_DIR is not set." >&2
    exit 1
fi

# Check if we should skip package creation
if [[ "$KEEP_EXISTING" == true ]]; then
    echo "Using existing package (--keep-existing-archive flag set)"
    echo ""
    
    # Validate tarball exists
    if [[ ! -f "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" ]]; then
        echo "Error: Tarball not found: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" >&2
        echo "Remove --keep-existing-archive flag to create a new package." >&2
        exit 1
    fi
    
    TARBALL_SIZE=$(du -h "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" | cut -f1)
    echo "Found existing tarball: ${PACKAGE_NAME}.tar.gz (${TARBALL_SIZE})"
    echo "Skipping to Zenodo upload..."
else
    # Validate source directory exists
    if [[ ! -d "${SOURCE_BASE}" ]]; then
        echo "Error: Source raw_data directory not found: ${SOURCE_BASE}" >&2
        echo "Please verify the project name and path." >&2
        exit 1
    fi

# Remove existing output if it exists
if [[ -d "${OUTPUT_DIR}/${PACKAGE_NAME}" ]]; then
    echo "Warning: Output directory already exists. Removing: ${OUTPUT_DIR}/${PACKAGE_NAME}"
    rm -rf "${OUTPUT_DIR:?}/${PACKAGE_NAME}"
fi

# Create output structure matching expected paths:
# - raw_data/dev/{project_name}/   (h5 files live directly here)
# - databases/main_db/
# - databases/msms_refs/
# - configs/
# - atlases/
# - projects/targeted_outputs/ (will be created by workflow)
STANDALONE_PROJECT="20260101_JGI_XX_000000_STANDALONE-DEV_test_EXP000_HILICZ_TESTXXXX"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/raw_data/dev/${STANDALONE_PROJECT}"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/databases/main_db"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/databases/msms_refs"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/configs"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/atlases"
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}/projects/targeted_outputs"
cd "${OUTPUT_DIR}/${PACKAGE_NAME}"

echo "Collecting h5 files..."
echo "------------------------------------"

# Copy all h5 files for specific run types
QC_RUNS=(
    "Run196" 
    "Run150" 
    "Run58"
)
EXPERIMENTAL_RUNS=(
    "Run159" # NEG_MS2_56_T1-256534-8-Tr-RE_1
    "Run162" # NEG_MS2_57_T1-256534-8-Tr-RE_2
    "Run183" # NEG_MS2_58_T1-256534-8-Tr-RE_3
    "Run158" # POS_MS2_56_T1-256534-8-Tr-RE_1
    "Run161" # POS_MS2_57_T1-256534-8-Tr-RE_2
    "Run182" # POS_MS2_58_T1-256534-8-Tr-RE_3
    "Run153" # NEG_MS2_51_T1-256534-8-WT-RE_1
    "Run174" # NEG_MS2_52_T1-256534-8-WT-RE_2
    "Run186" # NEG_MS2_53_T1-256534-8-WT-RE_3
    "Run152" # POS_MS2_51_T1-256534-8-WT-RE_1
    "Run173" # POS_MS2_52_T1-256534-8-WT-RE_2
    "Run185" # POS_MS2_53_T1-256534-8-WT-RE_3
)

ALL_RUNS=("${QC_RUNS[@]}" "${EXPERIMENTAL_RUNS[@]}")

COPIED=0
MISSING=0

echo "Copying h5 files for ${#ALL_RUNS[@]} runs (each run may have multiple h5 files)..."
echo ""

for run in "${ALL_RUNS[@]}"; do
    # Find all h5 files for this run
    shopt -s nullglob
    run_files=("${SOURCE_BASE}/"*"_${run}".h5)
    shopt -u nullglob
    
    if [[ ${#run_files[@]} -gt 0 ]]; then
        echo -n "  [$((COPIED+MISSING+1))/${#ALL_RUNS[@]}] ${run}... "
        
        local_count=0
        for hfile in "${run_files[@]}"; do
            cp "$hfile" "raw_data/dev/${STANDALONE_PROJECT}/" && local_count=$((local_count + 1))
        done
        
        echo "OK (${local_count} files)"
        COPIED=$((COPIED + 1))
    else
        echo "  [$((COPIED+MISSING+1))/${#ALL_RUNS[@]}] ${run}... MISSING"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "Summary: ${COPIED} runs copied successfully, ${MISSING} missing"

if [[ ${MISSING} -gt 0 ]]; then
    echo ""
    echo "Warning: Some runs were not found. Continuing with available files..."
    echo "You may need to adjust the run list in this script."
fi

# Count total h5 files copied
TOTAL_H5=$(find "raw_data/dev/${STANDALONE_PROJECT}/" -name "*.h5" 2>/dev/null | wc -l)
echo ""
echo "Total h5 files copied: ${TOTAL_H5}"

echo ""
echo "Creating compound definitions..."
echo "-----------------------------------------"

# Create positive mode compounds

cat > atlases/qc_compounds_pos.tsv << 'EOF'
compound_name	adduct	mz	rt_peak	rt_min	rt_max	inchi_key	mz_tolerance	polarity
ABMBA (unlabeled)	[M+H]+	229.9811	1.09380632	0.79380632	1.39380632	LCMZECCEEOQWLQ-UHFFFAOYSA-N	20	positive
thymine (U - 13C, 15N)	[M+H]+	134.06104	1.25523106	0.95523106	1.55523106	RWQNBRDOKXIBIV-BNUYUSEDSA-N	20	positive
hypoxanthine (U - 15N)	[M+H]+	141.03392	3.10296734	2.80296734	3.40296734	FDGQSTZJBFJUBT-NNZQUYKOSA-N	20	positive
guanine (U - 15N)	[M+H]+	157.04186	6.26535999	5.96535999	6.56535999	UYTPUPDQBNUYGX-CIKZIQIKSA-N	20	positive
methionine (U - 13C, 15N)	[M+H]+	156.07213	10.4409554	10.1409554	10.7409554	FFEARJCKVFRZRR-XAFSXMPTSA-N	20	positive
asparagine (U - 13C, 15N)	[M+H]+	139.06825	14.3680889	14.0680889	14.6680889	DCXYFEDJOCDNAF-FLEDYEEXSA-N	20	positive
cystine (U - 13C, 15N)	[M+H]+	249.04532	16.9043083	16.6043083	17.2043083	LEVWYRKDKASIDU-OGYFDXEDSA-N	20	positive
EOF

echo "   Created atlases/qc_compounds_pos.tsv"

cat > atlases/ema_compounds_pos.tsv << 'EOF'
label	compound_name	inchi_key	adduct	mz	rt_peak	rt_min	rt_max	mz_tolerance	polarity
adenine	adenine	GFFGJBXGBJISGV-UHFFFAOYSA-N	[M+H]+	136.06177	2.675523943	1.925523943	3.425523943	20	positive
riboflavin	riboflavin	AUNGANRZJHBGPY-SCRDCRAPSA-N	[M+H]+	377.14556	4.559205568	3.809205568	5.309205568	20	positive
guanine	guanine	UYTPUPDQBNUYGX-UHFFFAOYSA-N	[M+H]+	152.05669	6.255773885	5.505773885	7.005773885	20	positive
leucine	leucine	ROHFNLRQFUQHCH-UHFFFAOYSA-N	[M+H]+	132.10191	9.326296805	8.576296805	10.07629681	20	positive
norleucine	norleucine	LRQKBLKVPFOOQJ-YFKPBYRVSA-N	[M+H]+	132.10191	9.341396963	8.591396963	10.09139696	20	positive
isoleucine	isoleucine	AGPKZVBTJJNPAG-WHFBIAKZSA-N	[M+H]+	132.10191	9.716726546	8.966726546	10.46672655	20	positive
serine	serine	MTCFGRXMJLQNBG-UHFFFAOYSA-N	[M+H]+	106.0498691	14.11753228	13.61753228	14.61753228	5	positive
lysine	lysine	KDXKERNSBIXSRK-YFKPBYRVSA-N	[M+H]+	147.1128	17.01131041	16.26131041	17.76131041	20	positive
EOF

echo "   Created atlases/ema_compounds_pos.tsv"

cat > atlases/ema_compounds_neg.tsv << 'EOF'
label	compound_name	inchi_key	adduct	mz	rt_peak	rt_min	rt_max	mz_tolerance	polarity
adenine	adenine	GFFGJBXGBJISGV-UHFFFAOYSA-N	[M-H]-	134.04722	2.677601998	1.927601998	3.427601998	5	negative
riboflavin	riboflavin	AUNGANRZJHBGPY-SCRDCRAPSA-N	[M-H]-	375.13101	4.556357991	3.806357991	5.306357991	5	negative
guanine	guanine	UYTPUPDQBNUYGX-UHFFFAOYSA-N	[M-H]-	150.04213	6.265359988	5.515359988	7.015359988	5	negative
leucine	leucine	ROHFNLRQFUQHCH-UHFFFAOYSA-N	[M-H]-	130.08735	9.319656306	8.569656306	10.06965631	5	negative
norleucine	norleucine	LRQKBLKVPFOOQJ-YFKPBYRVSA-N	[M-H]-	130.08735	9.336960779	8.586960779	10.08696078	5	negative
isoleucine	isoleucine	AGPKZVBTJJNPAG-WHFBIAKZSA-N	[M-H]-	130.08735	9.70543744	8.95543744	10.45543744	5	negative
histidine	histidine	HNDVDQJCIGZPNO-YFKPBYRVSA-N	[M-H]-	154.0622	14.87779878	14.12779878	15.62779878	5	negative
lysine	lysine	KDXKERNSBIXSRK-YFKPBYRVSA-N	[M-H]-	145.09825	17.01238259	16.26238259	17.76238259	5	negative
EOF

echo "   Created atlases/ema_compounds_neg.tsv"

echo ""
echo "Creating MS2 references..."
echo "----------------------------------"

# Create MS2 references file
cat > databases/msms_refs/ms2_references.json << 'EOF'
{"ix": 40, "database": "metatlas", "id": "36aacde4bdf647aabd2f6012491e1b37", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 377.146, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [51.5585, 57.0344, 57.2594, 59.0501, 61.0291, 68.1039, 69.0343, 70.3038, 71.0134, 71.0498, 71.3187, 73.029, 74.7888, 74.7934, 75.0447, 79.9236, 81.034, 94.4088, 94.4139, 99.0447, 117.055, 120.772, 130.617, 140.291, 167.512, 169.607, 172.087, 178.786, 200.081, 216.077, 243.088, 244.091, 334.14, 359.135, 360.139, 377.146, 378.149], "intensities": [41067.0, 416448.0, 51929.0, 45272.0, 61642.0, 46252.0, 687842.0, 52021.0, 186327.0, 213927.0, 60040.0, 105045.0, 248913.0, 105136.0, 224415.0, 40740.0, 269204.0, 80258.0, 65152.0, 685077.0, 209501.0, 45974.0, 47854.0, 51782.0, 48939.0, 52416.0, 101369.0, 54696.0, 57464.0, 59277.0, 10433600.0, 780488.0, 53680.0, 727446.0, 65661.0, 32985600.0, 3942320.0]}
{"ix": 61, "database": "metatlas", "id": "4d60ac06c206441e88226f9d8f1f124e", "name": "isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [51.0053, 51.6623, 54.0121, 54.328, 55.2777, 58.0977, 58.9816, 69.071, 86.0974, 87.1008, 121.602, 132.103], "intensities": [2034.0, 2228.0, 2164.0, 2066.0, 2245.0, 2105.0, 2254.0, 41822.0, 433756.0, 22292.0, 2518.0, 5459.0]}
{"ix": 63, "database": "metatlas", "id": "73b8891fc94445539a7db8fd25d0e6be", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 152.057, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [50.6763, 51.2455, 51.2477, 52.1463, 58.0662, 59.0045, 59.0742, 77.0243, 89.0606, 93.1761, 102.352, 109.052, 110.036, 128.046, 133.686, 134.048, 134.06, 135.031, 151.079, 152.057, 153.042, 153.061, 154.045], "intensities": [4022.0, 4617.0, 3711.0, 3699.0, 17317.0, 3618.0, 32252.0, 3928.0, 5215.0, 4889.0, 4226.0, 20149.0, 56572.0, 66276.0, 4118.0, 5812.0, 4222.0, 44883.0, 5033.0, 3535550.0, 255008.0, 109830.0, 7859.0]}
{"ix": 76, "database": "metatlas", "id": "2a6aa9f6f7694650b91fcee0d4f02642", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 150.042, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [55.0174, 55.2387, 58.0046, 59.0125, 66.0085, 69.033, 71.0125, 72.9919, 73.0282, 78.0082, 82.0397, 83.0126, 85.028, 87.0073, 89.0231, 91.039, 97.0281, 101.023, 105.507, 105.954, 106.027, 107.035, 108.019, 113.023, 119.033, 126.03, 131.034, 133.015, 133.761, 134.019, 136.016, 148.195, 149.034, 150.001, 150.041, 151.026, 151.048, 151.06], "intensities": [1327.0, 1558.0, 8481.0, 71636.0, 9735.0, 3978.0, 151767.0, 12354.0, 36152.0, 1793.0, 2572.0, 4056.0, 17128.0, 11353.0, 134356.0, 4403.0, 3390.0, 79289.0, 1820.0, 1859.0, 1932.0, 23014.0, 22746.0, 2279.0, 4559.0, 37877.0, 2030.0, 220088.0, 1996.0, 6471.0, 3148.0, 1828.0, 2378.0, 12145.0, 100767.0, 4220.0, 4118.0, 110230.0]}
{"ix": 86, "database": "metatlas", "id": "d068cc4073d84ebba5ebd2ccc918eec0", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [61.2616, 75.5892, 88.0395, 90.5145, 92.1074, 105.846, 112.039, 118.516, 125.437, 125.91, 126.396, 130.086, 131.09], "intensities": [3094.0, 2817.0, 3334.0, 2930.0, 5223.0, 2855.0, 2656.0, 3153.0, 2903.0, 2986.0, 2996.0, 836909.0, 25554.0]}
{"ix": 96, "database": "metatlas", "id": "dd0233a19fe741c9aa3dfdfba356b88d", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [53.1997, 58.0659, 58.3278, 67.9673, 69.0707, 69.987, 74.0882, 80.9538, 86.0971, 87.1004, 90.6052, 94.5782, 103.055, 114.878, 120.844, 123.931, 125.585, 132.102], "intensities": [1841.0, 1867.0, 1749.0, 1978.0, 53737.0, 2161.0, 1717.0, 1926.0, 580540.0, 21792.0, 1947.0, 2134.0, 3875.0, 2338.0, 2095.0, 2045.0, 2350.0, 15201.0]}
{"ix": 121, "database": "metatlas", "id": "b13873a30039423187600cfa404802cb", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.047, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [51.2764, 54.9831, 58.2846, 65.3541, 68.0239, 83.6965, 84.0449, 92.024, 107.028, 107.035, 134.046, 135.051], "intensities": [58059.0, 47168.0, 41870.0, 53235.0, 49391.0, 49639.0, 45542.0, 748094.0, 111920.0, 4344710.0, 26113100.0, 754261.0]}
{"ix": 153, "database": "metatlas", "id": "7e8c0e4274be41d19bf617f45e4b2829", "name": "isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [50.3485, 54.2354, 69.2599, 73.3472, 84.4848, 130.086, 131.09], "intensities": [2665.0, 2462.0, 2777.0, 2789.0, 2488.0, 479696.0, 24090.0]}
{"ix": 256, "database": "metatlas", "id": "9c63f35dbf5649b08463b5d3903ae813", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 375.131, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [52.8588, 53.0988, 54.0123, 59.0123, 66.9271, 71.0126, 84.0859, 89.0232, 101.023, 101.283, 176.062, 212.083, 213.085, 225.72, 241.073, 242.076, 255.089, 256.093, 257.106, 263.858, 375.133, 376.137], "intensities": [7879.0, 7625.0, 8500.0, 44664.0, 7118.0, 11149.0, 9917.0, 15866.0, 47262.0, 8904.0, 9670.0, 264453.0, 12063.0, 12478.0, 204309.0, 11168.0, 2536140.0, 239964.0, 17438.0, 10581.0, 121838.0, 13690.0]}
{"ix": 282, "database": "metatlas", "id": "f3d2e9f1d51a4847a3229ab895b84194", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": null, "mz": [50.4856, 53.7233, 54.2718, 60.3272, 65.9816, 78.986, 86.0973, 86.7415, 87.1006, 99.011, 132.102], "intensities": [7535.0, 8780.0, 8535.0, 8351.0, 8893.0, 9538.0, 3758980.0, 9183.0, 113776.0, 7681.0, 103603.0]}
{"ix": 286, "database": "metatlas", "id": "04917e7f29604db7934c92e91a932302", "name": "L-lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 145.098, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [50.0076, 50.6047, 67.0938, 68.0233, 68.9016, 102.034, 130.72, 145.097, 146.024, 146.101], "intensities": [1669.0, 1637.0, 1804.0, 1479.0, 1501.0, 1965.0, 1529.0, 205873.0, 3703.0, 4250.0]}
{"ix": 304, "database": "metatlas", "id": "b2746b7690774cb7b44aa5b223925476", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 377.146, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [51.4752, 55.8477, 57.0347, 60.7073, 61.0295, 62.2312, 69.0346, 71.0135, 71.0502, 73.0292, 74.789, 74.7937, 75.0137, 75.0449, 81.0342, 99.045, 112.109, 117.055, 172.087, 200.083, 216.078, 243.089, 244.093, 257.104, 332.288, 334.142, 341.126, 359.136, 360.12, 361.124, 376.134, 377.147, 378.15, 378.2], "intensities": [26128.0, 25465.0, 210339.0, 26855.0, 45700.0, 30141.0, 395947.0, 117799.0, 217086.0, 141833.0, 173823.0, 97380.0, 26002.0, 112902.0, 167122.0, 514085.0, 33740.0, 180208.0, 122942.0, 67387.0, 129680.0, 7104580.0, 634341.0, 279227.0, 28789.0, 58779.0, 28770.0, 594140.0, 657875.0, 42903.0, 56827.0, 22923400.0, 2480940.0, 63543.0]}
{"ix": 323, "database": "metatlas", "id": "5520d9dcc50342428759bee39c9b3fd1", "name": "lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.113, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [56.0506, 58.777, 62.1319, 67.0553, 74.6282, 84.0817, 85.0658, 85.0796, 85.085, 101.108, 102.092, 102.922, 112.076, 119.703, 123.965, 129.103, 130.087, 131.09, 147.113], "intensities": [6538.0, 13338.0, 2424.0, 6719.0, 2563.0, 2206050.0, 4425.0, 5430.0, 64026.0, 5734.0, 10481.0, 4721.0, 4527.0, 2495.0, 3366.0, 39548.0, 1155270.0, 51045.0, 159650.0]}
{"ix": 331, "database": "metatlas", "id": "a95ea83e819246da8cfdbea92a641f94", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": null, "mz": [52.9132, 53.9553, 59.2875, 59.702, 69.0708, 86.0972, 87.0951, 87.1004, 108.5, 112.235, 129.305, 132.102], "intensities": [3900.0, 3582.0, 3552.0, 3946.0, 151035.0, 4236450.0, 8567.0, 147192.0, 4662.0, 4573.0, 4070.0, 75282.0]}
{"ix": 333, "database": "metatlas", "id": "4fdbc101a43d405692b530edc9db6816", "name": "lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 145.098, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [51.1882, 53.2209, 54.1691, 84.747, 89.6175, 95.0599, 99.0914, 104.439, 124.561, 145.097, 146.023, 146.101], "intensities": [1442.0, 1416.0, 1475.0, 1643.0, 1621.0, 1924.0, 1794.0, 2017.0, 1782.0, 430994.0, 1859.0, 17780.0]}
{"ix": 363, "database": "metatlas", "id": "8ba70c0f245247eeb6ba90011026763a", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.062, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [59.3596, 62.4513, 63.2027, 76.4601, 86.8208, 94.0407, 115.912, 115.975, 119.036, 123.375, 136.062, 137.046, 137.067], "intensities": [55769.0, 43616.0, 118692.0, 54358.0, 48393.0, 121260.0, 45996.0, 55157.0, 306316.0, 61623.0, 41864400.0, 370525.0, 1357390.0]}
{"ix": 392, "database": "metatlas", "id": "50334867a31f4cab973459a59d5731c4", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.062, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [52.1001, 53.5537, 54.6096, 57.8238, 63.3067, 64.108, 82.7587, 93.0862, 94.0407, 94.6115, 111.471, 113.584, 115.21, 119.036, 136.062, 137.046, 137.067, 137.476], "intensities": [491091.0, 614205.0, 486992.0, 569335.0, 2513570.0, 554436.0, 577010.0, 580100.0, 2437690.0, 930338.0, 567270.0, 515519.0, 616418.0, 7680000.0, 514804000.0, 4940020.0, 17234000.0, 693366.0]}
{"ix": 419, "database": "metatlas", "id": "5a1add6611ca40aeb051bb79831e961a", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 150.042, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [56.0599, 66.0083, 78.0084, 82.0396, 90.0084, 94.5778, 107.029, 107.035, 108.019, 108.844, 109.022, 113.497, 118.728, 120.151, 120.363, 121.014, 126.03, 132.03, 133.014, 134.019, 150.041, 150.232, 151.025, 151.044], "intensities": [8218.0, 203144.0, 43620.0, 95518.0, 50795.0, 8874.0, 7480.0, 446200.0, 745290.0, 8331.0, 9028.0, 8961.0, 14257.0, 9916.0, 9085.0, 19076.0, 555678.0, 12347.0, 4581310.0, 86499.0, 1525320.0, 8682.0, 55961.0, 19851.0]}
{"ix": 436, "database": "metatlas", "id": "befb9115882746a39eaecedd788a6dc6", "name": "L-lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.113, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [54.2354, 56.0505, 58.7769, 67.0551, 69.8754, 78.0773, 84.0817, 85.0656, 85.085, 101.108, 102.092, 105.316, 112.076, 129.103, 130.087, 131.09, 147.113, 147.124], "intensities": [21589.0, 44001.0, 176579.0, 53866.0, 25439.0, 23818.0, 16481300.0, 45566.0, 392937.0, 44108.0, 58562.0, 25030.0, 42084.0, 243610.0, 8180870.0, 186398.0, 1083200.0, 48551.0]}
{"ix": 449, "database": "metatlas", "id": "4d70969f0520473588e7a3bf50c59948", "name": "DL-Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 104.035, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.0425931, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": null, "mz": [55.9539, 57.9886, 60.8063, 62.906, 63.1275, 72.0034, 72.0077, 74.0233, 75.0265, 86.8951, 95.7638, 103.003, 104.034, 105.037], "intensities": [1608.0, 1390.0, 1652.0, 1326.0, 1391.0, 2495.0, 254100.0, 308556.0, 2657.0, 1637.0, 1555.0, 1734.0, 124056.0, 1883.0]}
{"ix": 489, "database": "metatlas", "id": "03080a1300b1483183fe8573f9731373", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 375.131, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [54.9478, 58.1346, 59.0126, 66.0726, 70.3782, 73.9624, 84.9927, 89.0234, 100.32, 101.023, 133.094, 156.137, 165.628, 186.067, 189.266, 212.083, 241.073, 243.09, 255.089, 256.093, 375.132], "intensities": [6573.0, 5411.0, 12605.0, 6985.0, 6490.0, 6021.0, 6540.0, 7772.0, 7171.0, 11471.0, 7536.0, 7168.0, 7451.0, 12746.0, 8711.0, 112786.0, 70266.0, 96014.0, 1236720.0, 71002.0, 63392.0]}
{"ix": 546, "database": "metatlas", "id": "da41022f22ac4f90813f4c63a05bc25a", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": null, "mz": [58.2117, 64.748, 74.1138, 74.9539, 77.7553, 84.0511, 94.5075, 95.479, 98.6033, 100.354, 102.876, 113.059, 116.457, 128.241, 130.086, 131.09], "intensities": [7891.0, 7868.0, 7212.0, 10191.0, 7501.0, 9490.0, 9520.0, 9125.0, 7973.0, 8562.0, 8569.0, 13747.0, 7768.0, 8021.0, 2159850.0, 65069.0]}
{"ix": 583, "database": "metatlas", "id": "59f8740011d645e9ba36f3bce715a03c", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 152.057, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [51.0919, 51.094, 51.0973, 53.7215, 59.7425, 60.7568, 72.815, 91.81, 93.3212, 94.4459, 96.5063, 100.255, 101.604, 109.051, 110.035, 122.208, 128.046, 134.046, 135.03, 136.014, 136.035, 152.057, 153.041, 153.061, 154.025, 154.045], "intensities": [45172.0, 63508.0, 42955.0, 29054.0, 26059.0, 24528.0, 26613.0, 25225.0, 25656.0, 25736.0, 23862.0, 27462.0, 27210.0, 228686.0, 864863.0, 29941.0, 333990.0, 118385.0, 1605340.0, 57626.0, 45912.0, 30525800.0, 2288710.0, 751035.0, 116924.0, 32540.0]}
{"ix": 587, "database": "metatlas", "id": "9fdf6af132e74af491be9b938195a68e", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": null, "mz": [53.2199, 54.4991, 62.6112, 63.1982, 65.6667, 76.5021, 78.4424, 81.8963, 84.0804, 92.108, 119.573, 130.086, 131.09], "intensities": [16321.0, 13859.0, 16134.0, 14706.0, 17193.0, 18282.0, 18143.0, 18066.0, 25348.0, 25329.0, 21561.0, 5338920.0, 170620.0]}
{"ix": 598, "database": "metatlas", "id": "c8efb837c4a44ec2873694fa00d2b210", "name": "DL-Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 106.05, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.0425931, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": null, "mz": [57.4739, 59.5975, 60.0453, 61.0486, 66.5664, 67.6551, 70.0296, 70.0661, 88.0399, 88.0764, 98.6441, 106.05, 106.087], "intensities": [1249.0, 1322.0, 348839.0, 2444.0, 1431.0, 1480.0, 9167.0, 2476.0, 52802.0, 3058.0, 1669.0, 19206.0, 1983.0]}
{"ix": 624, "database": "metatlas", "id": "9477b8c9428345f69d4408e2a5108bfb", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.047, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [59.5429, 65.0133, 75.164, 92.0242, 107.035, 114.47, 134.046, 135.051], "intensities": [184691.0, 238694.0, 190293.0, 3303670.0, 19467000.0, 189916.0, 110157000.0, 2466710.0]}
{"ix": 192032, "database": "metatlas", "id": "99d9dee377304155b946cc560f9c6c02", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 150.042, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [65.9985, 66.0094, 78.0097, 80.0253, 82.041, 90.0097, 95.1699, 97.0731, 105.021, 106.005, 107.036, 108.021, 121.016, 126.031, 132.032, 132.613, 132.622, 133.016, 133.419, 149.56, 150.042, 151.026, 151.039, 152.01], "intensities": [45074.0, 1366820.0, 516984.0, 39261.0, 1281920.0, 384496.0, 26391.0, 28309.0, 45602.0, 192331.0, 6603530.0, 10719200.0, 304343.0, 8924520.0, 263006.0, 42141.0, 27456.0, 76227700.0, 33658.0, 29618.0, 40922000.0, 974198.0, 55355.0, 62405.0]}
{"ix": 192046, "database": "metatlas", "id": "1ab9e54c6ab844999073eaf41fcb5080", "name": "DL-Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 104.035, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.0425931, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": null, "mz": [56.8651, 59.0135, 60.0171, 61.5819, 66.7489, 72.0089, 74.0246, 82.4046, 102.419, 102.809, 104.036], "intensities": [2381.0, 3908.0, 2235.0, 2509.0, 2465.0, 942307.0, 1095530.0, 2733.0, 2222.0, 2130.0, 407556.0]}
{"ix": 192056, "database": "metatlas", "id": "5938f71b59834597acdc23a8e6ef01a0", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.047, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [65.0141, 68.025, 75.1848, 80.0253, 92.0254, 92.3953, 93.0831, 107.036, 119.692, 128.994, 132.032, 133.64, 134.047, 134.456], "intensities": [542179.0, 403180.0, 46141.0, 380654.0, 7633880.0, 62138.0, 54993.0, 36504300.0, 51458.0, 51375.0, 64773.0, 79471.0, 163076000.0, 66448.0]}
{"ix": 192057, "database": "metatlas", "id": "a24f437eb0564354b2b6e093cfc5cbd5", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 170.024, "polarity": "negative", "adduct": "[M+Cl]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [60.5481, 63.7868, 82.0061, 96.2304, 118.673, 119.166, 126.092, 130.351, 134.047, 169.85], "intensities": [2157.0, 2194.0, 2008.0, 1844.0, 1991.0, 1988.0, 3540.0, 2077.0, 71311.0, 2812.0]}
{"ix": 192142, "database": "metatlas", "id": "2c777781039348b780b8fdc86ee7a779", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 150.042, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [50.4548, 59.9769, 60.5793, 63.1815, 66.0093, 66.7881, 78.0094, 82.0408, 107.036, 108.012, 108.02, 122.061, 126.031, 133.016, 150.042, 151.026], "intensities": [9195.0, 7894.0, 8892.0, 7786.0, 64439.0, 9132.0, 13148.0, 48802.0, 326011.0, 16076.0, 414266.0, 20232.0, 465851.0, 3888630.0, 2050780.0, 70780.0]}
{"ix": 192186, "database": "metatlas", "id": "9f361d6e203b45d79d0f1ffea7ed3a83", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": null, "mz": [52.5524, 67.4952, 84.0817, 100.307, 104.315, 118.691, 130.087], "intensities": [40055.0, 36661.0, 71349.0, 51603.0, 54967.0, 57366.0, 10176600.0]}
{"ix": 192285, "database": "metatlas", "id": "ed5ccf13572748a4a7d598afa50019ba", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": null, "mz": [57.7576, 66.7674, 67.7339, 130.087], "intensities": [42809.0, 41882.0, 38103.0, 7335690.0]}
{"ix": 192304, "database": "metatlas", "id": "6d130f86669c4c37a04882e6ee5d3c40", "name": "L-lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 145.098, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [55.1889, 67.5929, 74.0041, 79.0358, 82.9102, 92.4417, 97.0771, 99.0928, 104.358, 128.072, 132.434, 136.305, 145.099], "intensities": [2705.0, 3199.0, 3174.0, 2544.0, 3091.0, 4588.0, 6358.0, 6597.0, 3966.0, 5424.0, 2622.0, 3576.0, 1460770.0]}
{"ix": 192377, "database": "metatlas", "id": "83030936a70549e6bf27c823de748258", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 170.024, "polarity": "negative", "adduct": "[M+Cl]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [56.7233, 71.0496, 82.416, 86.3337, 89.7334, 126.093, 134.047, 151.989, 155.143], "intensities": [1832.0, 1893.0, 2496.0, 2054.0, 2165.0, 3419.0, 152590.0, 1961.0, 2167.0]}
{"ix": 192378, "database": "metatlas", "id": "dc73b8eb9db143e48df1cb0516cdf6fe", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.047, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [58.1063, 63.0105, 74.6545, 80.9597, 80.965, 83.5183, 92.0252, 92.3999, 107.036, 116.921, 133.992, 134.047], "intensities": [1699.0, 1703.0, 1925.0, 3568.0, 102617.0, 1873.0, 11653.0, 1955.0, 60928.0, 29803.0, 10795.0, 326077.0]}
{"ix": 192457, "database": "metatlas", "id": "b657afc6f5f04143a89fbe1ea1a87272", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 375.131, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [63.0889, 64.6587, 73.9865, 83.1369, 104.33, 118.705, 191.876, 196.16, 212.083, 241.073, 243.089, 249.566, 255.089, 257.104, 375.131], "intensities": [143643.0, 139509.0, 165668.0, 139509.0, 263168.0, 160265.0, 170238.0, 167248.0, 3724340.0, 1778690.0, 188508.0, 210801.0, 28035800.0, 230860.0, 7484060.0]}
{"ix": 192458, "database": "metatlas", "id": "48500d4a8b3e421eb1b93651cbe6117b", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 751.269, "polarity": "negative", "adduct": "[2M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [52.6003, 57.1364, 80.7035, 136.271, 210.918, 212.083, 241.073, 243.089, 255.089, 375.131, 447.717, 554.848], "intensities": [11665.0, 15275.0, 16238.0, 14703.0, 33173.0, 115587.0, 126092.0, 16536.0, 1203020.0, 1055000.0, 17585.0, 15036.0]}
{"ix": 192459, "database": "metatlas", "id": "ca6a1203b085487cac52a85dc42c613c", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 411.108, "polarity": "negative", "adduct": "[M+Cl]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [56.2438, 68.8887, 71.7213, 88.9238, 102.693, 104.33, 182.148, 208.644, 212.083, 241.072, 255.089, 275.778, 302.22, 308.489, 375.131, 411.107], "intensities": [7544.0, 8319.0, 8209.0, 8924.0, 9863.0, 11209.0, 8985.0, 46685.0, 64407.0, 19355.0, 663172.0, 10421.0, 9952.0, 11591.0, 606562.0, 146552.0]}
{"ix": 192460, "database": "metatlas", "id": "65d4225269b94c7e8396c2b01a523145", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 435.152, "polarity": "negative", "adduct": "[M+acetate]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [67.4127, 73.2558, 74.1953, 82.4457, 86.5861, 102.475, 118.72, 149.332, 167.141, 255.089, 369.197, 375.132], "intensities": [6607.0, 6160.0, 7154.0, 6607.0, 8117.0, 5822.0, 7264.0, 6739.0, 6992.0, 226549.0, 7028.0, 9823.0]}
{"ix": 192464, "database": "metatlas", "id": "2f403a2efea14662ba0d2d216582c10c", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.087, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [60.5539, 66.7623, 67.1701, 70.1654, 92.4033, 104.317, 118.935, 130.087], "intensities": [74010.0, 53702.0, 49928.0, 41712.0, 57608.0, 69863.0, 49563.0, 9336190.0]}
{"ix": 192486, "database": "metatlas", "id": "f53c57b107b1414ab4b2de360f6ebf8f", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 152.057, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [55.0301, 58.2215, 62.5575, 65.5462, 72.7401, 82.8672, 88.6356, 95.9689, 106.81, 109.051, 110.035, 128.034, 128.046, 134.047, 135.03, 136.014, 136.235, 151.571, 152.057, 152.545, 153.041, 154.025], "intensities": [60051.0, 63629.0, 66252.0, 58352.0, 49004.0, 60753.0, 53071.0, 54490.0, 60860.0, 2223760.0, 9414720.0, 89173.0, 5168990.0, 974016.0, 18213100.0, 602708.0, 82845.0, 280594.0, 281599000.0, 246403.0, 28284300.0, 617221.0]}
{"ix": 192487, "database": "metatlas", "id": "b919ae532da94dc69faa271d9e9f5216", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 174.039, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [50.2687, 51.9001, 54.534, 55.2187, 55.622, 56.0522, 57.5154, 60.0563, 60.8913, 79.0386, 84.2441, 91.2755, 100.076, 111.985, 129.995, 131.974, 132.08, 132.959, 136.218, 142.098, 149.984, 150.969, 155.974, 174.039, 174.127], "intensities": [2421.0, 2479.0, 2459.0, 2686.0, 2940.0, 3091.0, 2697.0, 4139.0, 3177.0, 2992.0, 2724.0, 2805.0, 15605.0, 5848.0, 6199.0, 49826.0, 3198.0, 35211.0, 3118.0, 5890.0, 20697.0, 46371.0, 27382.0, 3086850.0, 47800.0]}
{"ix": 192501, "database": "metatlas", "id": "2889529d7f574e06a50712dba25098e9", "name": "DL-Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 106.05, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.0425931, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": null, "mz": [60.0452, 70.0297, 82.3946, 88.0401, 95.4463, 104.282, 106.05], "intensities": [188719.0, 7009.0, 2980.0, 29901.0, 1903.0, 3472.0, 2762.0]}
{"ix": 192512, "database": "metatlas", "id": "65355cf82d9340629184a0ee83d01d0e", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 158.044, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [66.7532, 92.3928, 100.077, 102.106, 104.301, 116.972, 118.668, 131.995, 134.32, 134.959, 139.988, 148.961, 152.969, 153.443, 158.044, 158.097, 158.113, 158.153, 158.192], "intensities": [2205.0, 2429.0, 8968.0, 2153.0, 1971.0, 33332.0, 1980.0, 7701.0, 1987.0, 3058.0, 2366.0, 2954.0, 14326.0, 2061.0, 48712.0, 84335.0, 2786.0, 14683.0, 2252.0]}
{"ix": 192513, "database": "metatlas", "id": "9d53a44c42004e16a468e92e2b0a7009", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.062, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [55.0301, 56.3854, 66.7513, 67.0298, 81.1529, 82.4076, 92.0251, 92.3892, 94.0405, 104.302, 109.051, 112.051, 119.035, 135.054, 135.653, 136.062, 136.227, 136.474, 137.046], "intensities": [246689.0, 186484.0, 198526.0, 974057.0, 232546.0, 306008.0, 388476.0, 265393.0, 7436560.0, 246201.0, 1625240.0, 1318880.0, 23732500.0, 345780.0, 925801.0, 884493000.0, 254046.0, 715569.0, 23845700.0]}
{"ix": 192609, "database": "metatlas", "id": "b855911576974f348058d4b61e693848", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 174.039, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [66.7625, 70.9585, 71.8982, 84.9021, 85.3093, 111.985, 122.853, 129.994, 131.974, 132.081, 132.958, 149.985, 150.942, 150.969, 155.974, 173.984, 174.039, 174.128, 174.148], "intensities": [4190.0, 6729.0, 4238.0, 3763.0, 3733.0, 21400.0, 3940.0, 4950.0, 49280.0, 4663.0, 51449.0, 19029.0, 7956.0, 46343.0, 37117.0, 26805.0, 273763.0, 42600.0, 8268.0]}
{"ix": 192610, "database": "metatlas", "id": "e229d07ba12f4f4cb1e948920c9cd0d4", "name": "guanine", "decimal": 4.0, "inchi_key": "UYTPUPDQBNUYGX-UHFFFAOYSA-N", "precursor_mz": 152.057, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5O", "mono_isotopic_molecular_weight": 151.0494098, "inchi": "InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)", "smiles": null, "mz": [52.0734, 59.2609, 104.536, 106.065, 109.051, 110.035, 124.076, 128.046, 134.048, 134.06, 135.03, 152.057, 153.04, 153.055], "intensities": [26160.0, 22977.0, 27725.0, 26043.0, 131702.0, 600352.0, 50733.0, 294291.0, 53449.0, 583997.0, 1163050.0, 19940900.0, 1918830.0, 34515.0]}
{"ix": 192652, "database": "metatlas", "id": "78c1085e6c7540738feeff806e1d8c2a", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": null, "mz": [61.7252, 85.2999, 86.097, 92.3951, 100.06, 123.926, 132.102], "intensities": [70289.0, 52984.0, 14252800.0, 63353.0, 55436.0, 59237.0, 273707.0]}
{"ix": 192774, "database": "metatlas", "id": "4094de90b9c945b6bba907cc54aa0e5c", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": null, "mz": [63.8225, 66.7584, 69.0704, 80.7418, 82.4203, 86.0969, 102.535, 104.311, 132.103], "intensities": [55015.0, 96019.0, 826416.0, 57160.0, 72730.0, 15275600.0, 59061.0, 73775.0, 99620.0]}
{"ix": 192799, "database": "metatlas", "id": "563acc9493fe4e1cbcd2d2c5dc660e8a", "name": "L-lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 129.102, "polarity": "positive", "adduct": "[M+H-H2O]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [56.0502, 84.0813, 85.7055, 101.108, 104.32, 104.327, 111.092, 117.118, 118.701, 129.102], "intensities": [4987.0, 552478.0, 2121.0, 8172.0, 2458.0, 2615.0, 8063.0, 2037.0, 2407.0, 102203.0]}
{"ix": 192800, "database": "metatlas", "id": "27a3beb34e2e44a08cf659ee592e0265", "name": "L-lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.113, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.1055277, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": null, "mz": [56.0502, 61.4161, 65.6724, 66.7634, 67.0549, 74.0244, 75.553, 84.0814, 85.0653, 95.5892, 96.7936, 102.092, 104.317, 112.076, 123.964, 128.641, 129.102, 130.086, 136.255, 140.307, 147.113], "intensities": [18457.0, 4410.0, 4213.0, 7582.0, 19047.0, 5956.0, 4609.0, 3072060.0, 19113.0, 4845.0, 4536.0, 7503.0, 5274.0, 10749.0, 7886.0, 5814.0, 24451.0, 741597.0, 4716.0, 4515.0, 50216.0]}
{"ix": 192875, "database": "metatlas", "id": "29247268c3cf4acfb649ebce7b0c9e0c", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.062, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [51.3947, 91.0548, 94.0404, 119.035, 136.022, 136.062, 136.112, 137.046], "intensities": [1870.0, 3051.0, 13543.0, 28284.0, 55585.0, 1607820.0, 17469.0, 43758.0]}
{"ix": 192876, "database": "metatlas", "id": "5c7f01e6faf349219bea4784ae36361b", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 158.044, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": null, "mz": [72.0809, 82.9952, 96.5922, 100.076, 106.008, 114.091, 114.111, 116.972, 118.673, 129.954, 152.969, 158.044, 158.096, 158.117, 158.154], "intensities": [2117.0, 2583.0, 2397.0, 3699.0, 2283.0, 2570.0, 3546.0, 13307.0, 3199.0, 2239.0, 8472.0, 64225.0, 104106.0, 12763.0, 14920.0]}
{"ix": 192960, "database": "metatlas", "id": "c5478aefe6b64a25b573decf76899c9f", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 377.146, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [53.0391, 57.0341, 61.029, 69.034, 71.0132, 71.0495, 73.029, 75.0444, 76.002, 81.0339, 82.4216, 84.4191, 94.5098, 95.727, 99.0444, 113.213, 117.055, 136.249, 140.845, 165.319, 172.087, 200.081, 211.396, 216.076, 243.087, 244.071, 272.103, 359.133, 377.145], "intensities": [1120210.0, 6109040.0, 3064790.0, 14310100.0, 2847140.0, 4312480.0, 1293220.0, 3259390.0, 737842.0, 4120670.0, 914105.0, 799334.0, 739293.0, 738551.0, 7510420.0, 702375.0, 1392900.0, 952613.0, 721616.0, 696115.0, 11332600.0, 7154020.0, 1234450.0, 7331640.0, 124559000.0, 975223.0, 788363.0, 7611330.0, 460029000.0]}
{"ix": 192961, "database": "metatlas", "id": "8ce64b2772044aa1bc4cac3f7088903e", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 376.138, "polarity": "positive", "adduct": "[M-e]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [69.0339, 73.0288, 85.029, 97.0287, 112.506, 146.756, 158.011, 190.074, 201.089, 213.088, 228.101, 242.079, 243.087, 244.092, 253.963, 256.095, 257.103, 259.107, 269.104, 285.098, 286.101, 287.856, 288.084, 323.114, 327.11, 340.115, 341.125, 343.104, 348.143, 358.123, 376.136], "intensities": [2929.0, 4742.0, 2314.0, 11078.0, 2706.0, 2695.0, 3124.0, 9107.0, 44309.0, 4207.0, 3706.0, 42604.0, 117698.0, 62542.0, 2818.0, 291731.0, 52442.0, 3082.0, 3444.0, 14861.0, 5232.0, 2726.0, 77853.0, 4146.0, 3039.0, 5094.0, 11581.0, 4935.0, 11063.0, 53533.0, 1290270.0]}
{"ix": 192962, "database": "metatlas", "id": "adb42b794e43482ba2ae681f795a869f", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 399.127, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [52.4582, 68.8144, 69.0267, 82.4174, 104.311, 143.031, 144.033, 167.006, 199.073, 265.067, 279.084, 287.893, 299.163, 356.121, 375.949, 382.099, 393.956, 399.127], "intensities": [15978.0, 19266.0, 17771.0, 20803.0, 19417.0, 28852.0, 17442.0, 18940.0, 19785.0, 23752.0, 258136.0, 32656.0, 21687.0, 137936.0, 19803.0, 38074.0, 94427.0, 14390200.0]}
{"ix": 192963, "database": "metatlas", "id": "cdc7fe97ad8a471d973b9c0b19ca8371", "name": "riboflavin", "decimal": 4.0, "inchi_key": "AUNGANRZJHBGPY-SCRDCRAPSA-N", "precursor_mz": 415.101, "polarity": "positive", "adduct": "[M+K]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C17H20N4O6", "mono_isotopic_molecular_weight": 376.1382844, "inchi": "InChI=1S/C17H20N4O6/c1-7-3-9-10(4-8(7)2)21(5-11(23)14(25)12(24)6-22)15-13(18-9)16(26)20-17(27)19-15/h3-4,11-12,14,22-25H,5-6H2,1-2H3,(H,20,26,27)/t11-,12+,14-/m0/s1", "smiles": null, "mz": [67.3443, 75.5479, 78.304, 82.425, 104.318, 104.503, 118.694, 119.086, 119.155, 124.332, 194.704, 220.707, 286.894, 297.279, 374.946, 387.827, 391.969, 392.958, 415.1], "intensities": [2233.0, 2328.0, 2773.0, 2985.0, 2739.0, 2941.0, 2757.0, 26528.0, 2510.0, 2483.0, 2424.0, 2697.0, 19362.0, 2727.0, 12483.0, 2644.0, 9421.0, 29471.0, 678685.0]}
{"ix": 192970, "database": "metatlas", "id": "ab260ea055ac414583962a2377d69dcb", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.102, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": null, "mz": [66.1637, 66.7035, 69.0666, 69.0705, 69.1037, 86.097, 92.2665, 92.3872, 104.3, 128.836], "intensities": [52997.0, 56375.0, 59366.0, 1548240.0, 54088.0, 12820900.0, 53819.0, 76253.0, 62657.0, 58356.0]}
{"ix": 199068, "database": "metatlas", "id": "c18atlas20200609-M01B11", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [51.422, 55.0949, 65.5137, 69.0706, 86.097, 90.9038, 121.9783, 122.0193, 132.1025], "intensities": [28177.0, 31096.0, 29584.0, 724728.0, 6450192.0, 72942.0, 31711.0, 127920.0, 71390.0]}
{"ix": 199069, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c34", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [56.131, 69.0706, 86.097, 104.2874, 132.1018], "intensities": [64596.0, 842339.0, 15446866.0, 69940.0, 272128.0]}
{"ix": 199070, "database": "metatlas", "id": "c18atlas20200609-C0110", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [56.3595, 57.4206, 69.0703, 71.6652, 73.9687, 83.1543, 83.1611, 86.0967, 90.9035, 92.9014, 94.8324, 99.0034, 114.0014, 122.0191, 132.1018], "intensities": [22079.0, 19139.0, 532667.0, 21627.0, 19899.0, 18535.0, 22212.0, 5219059.0, 40753.0, 23259.0, 18763.0, 25265.0, 19071.0, 28953.0, 42251.0]}
{"ix": 199071, "database": "metatlas", "id": "c18atlas20200609-NPL800-ST079276", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [53.1153, 58.0651, 61.011, 63.998, 64.2079, 69.0702, 79.0215, 82.3823, 86.0966, 99.0029, 122.0193, 132.1017], "intensities": [2032.0, 2271.0, 4652.0, 2653.0, 2237.0, 124817.0, 2585.0, 2273.0, 1071424.0, 8208.0, 12055.0, 19236.0]}
{"ix": 199072, "database": "metatlas", "id": "c18atlas20200609-M01B11", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [57.7572, 61.9879, 64.1204, 66.7459, 73.9568, 82.4021, 92.3815, 104.2903, 125.6534, 130.0873], "intensities": [2826.0, 10795.0, 2393.0, 3298.0, 2943.0, 3196.0, 2673.0, 3585.0, 3143.0, 546846.0]}
{"ix": 199073, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c34", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [55.1631, 61.9881, 67.3493, 75.0875, 82.3995, 82.8603, 92.3828, 102.5202, 104.2945, 128.0353, 130.051, 130.0875], "intensities": [1779.0, 2302.0, 1918.0, 2389.0, 2360.0, 1807.0, 2086.0, 2064.0, 2112.0, 3851.0, 11295.0, 325187.0]}
{"ix": 199261, "database": "metatlas", "id": "c18atlas20200609-M02A11", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [75.0388, 76.1753, 86.0971, 90.9035, 97.7902, 99.0037, 122.0195, 132.1022], "intensities": [27374.0, 30130.0, 4041860.0, 65652.0, 32394.0, 41196.0, 122496.0, 104620.0]}
{"ix": 199262, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c33", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [56.131, 69.0706, 86.097, 104.2874, 132.1018], "intensities": [64596.0, 842339.0, 15446866.0, 69940.0, 272128.0]}
{"ix": 199263, "database": "metatlas", "id": "c18atlas20200609-C0907", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [55.0548, 56.05, 58.7477, 71.8796, 73.0652, 77.0974, 83.0494, 86.0967, 99.0034, 100.0755, 103.974, 104.009, 115.0754, 122.0194, 132.1026], "intensities": [1582703.0, 133853.0, 54310.0, 56558.0, 1418855.0, 58502.0, 132525.0, 7593863.0, 97609.0, 106307.0, 68256.0, 65436.0, 3058514.0, 389672.0, 441308.0]}
{"ix": 199264, "database": "metatlas", "id": "c18atlas20200609-NPL800-ST074999", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [61.0109, 62.783, 63.9981, 65.4186, 86.0967, 95.6523, 99.0031, 104.5159, 122.0192, 132.1016], "intensities": [3126.0, 1974.0, 3385.0, 1867.0, 266544.0, 2162.0, 9078.0, 1899.0, 16370.0, 4234.0]}
{"ix": 199265, "database": "metatlas", "id": "c18atlas20200609-M02A11", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [61.9879, 62.1536, 65.5548, 103.2034, 103.4419, 123.4833, 130.0873], "intensities": [25446.0, 2062.0, 2116.0, 1890.0, 2249.0, 2065.0, 299387.0]}
{"ix": 199266, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c33", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [55.1631, 61.9881, 67.3493, 75.0875, 82.3995, 82.8603, 92.3828, 102.5202, 104.2945, 128.0353, 130.051, 130.0875], "intensities": [1779.0, 2302.0, 1918.0, 2389.0, 2360.0, 1807.0, 2086.0, 2064.0, 2112.0, 3851.0, 11295.0, 325187.0]}
{"ix": 199514, "database": "metatlas", "id": "c18atlas20200609-M04B07", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": "CCCCC(N)C(=O)O", "mz": [55.3055, 63.3566, 66.7354, 69.0706, 86.0971, 122.0199, 131.8762, 132.1018], "intensities": [78457.0, 95160.0, 95874.0, 917521.0, 22982224.0, 163750.0, 86791.0, 196190.0]}
{"ix": 199515, "database": "metatlas", "id": "c18atlas20200609-M04B07", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": "CCCCC(N)C(=O)O", "mz": [61.9878, 66.7403, 92.3734, 97.3714, 100.7281, 118.6478, 130.0874], "intensities": [3506.0, 1894.0, 2250.0, 2030.0, 2138.0, 1965.0, 305048.0]}
{"ix": 203824, "database": "metatlas", "id": "c18atlas20200609-M01D04", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.0617952, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": "Nc1nc[nH]c2ncnc1-2", "mz": [77.7348, 82.3928, 92.686, 94.0405, 102.9324, 104.2803, 109.0514, 118.6449, 119.0354, 136.0226, 136.0619, 136.4739, 137.0459], "intensities": [7051.0, 7280.0, 6756.0, 59274.0, 6114.0, 9323.0, 7116.0, 7198.0, 104579.0, 142689.0, 5806716.0, 9381.0, 100963.0]}
{"ix": 203825, "database": "metatlas", "id": "c18atlas20200609-M01D04", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.0471952, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": "Nc1nc[nH]c2ncnc1-2", "mz": [65.0138, 67.2175, 71.0433, 82.407, 92.0253, 92.3877, 104.5468, 107.0363, 110.8499, 134.0472, 134.8656], "intensities": [4771.0, 3488.0, 3308.0, 3592.0, 60734.0, 4636.0, 3128.0, 269661.0, 3141.0, 1575772.0, 4785.0]}
{"ix": 210408, "database": "metatlas", "id": "c18atlas20200609-M01B11", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [52.1094, 53.4097, 58.0658, 65.1271, 69.0705, 69.0745, 73.9413, 86.097, 93.6313, 104.2682], "intensities": [120218.0, 87661.0, 103935.0, 96614.0, 3726554.0, 212196.0, 100866.0, 19318904.0, 108218.0, 132908.0]}
{"ix": 210409, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c34", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [67.3371, 69.0705, 81.339, 86.097, 89.7397, 92.3681, 106.9589, 118.6424, 132.102], "intensities": [66941.0, 1245820.0, 54796.0, 9036559.0, 61942.0, 85057.0, 61648.0, 85437.0, 78653.0]}
{"ix": 210410, "database": "metatlas", "id": "c18atlas20200609-C0110", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [69.0703, 69.0745, 82.4167, 86.0967, 86.6622, 92.3949, 118.6755, 132.103], "intensities": [556788.0, 27202.0, 18854.0, 3029129.0, 18756.0, 20572.0, 21343.0, 18767.0]}
{"ix": 210411, "database": "metatlas", "id": "c18atlas20200609-NPL800-ST079276", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [57.0577, 58.0655, 61.0111, 63.9981, 69.0703, 86.0966, 92.353, 93.6827, 95.5619, 99.0031, 104.2569, 118.6201, 122.0191, 132.1017], "intensities": [4141.0, 3873.0, 4410.0, 9001.0, 183408.0, 845947.0, 2349.0, 2371.0, 2163.0, 10366.0, 2282.0, 2347.0, 9021.0, 3866.0]}
{"ix": 210412, "database": "metatlas", "id": "c18atlas20200609-M01B11", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [57.4532, 61.9879, 62.9938, 65.8016, 66.7422, 71.6649, 73.953, 82.3988, 86.0606, 118.6539, 130.0874], "intensities": [2149.0, 11975.0, 1727.0, 1956.0, 2035.0, 2022.0, 2535.0, 2148.0, 2636.0, 2416.0, 294822.0]}
{"ix": 210413, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c34", "name": "l-isoleucine", "decimal": 4.0, "inchi_key": "AGPKZVBTJJNPAG-WHFBIAKZSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-3-4(2)5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)/t4-,5-/m0/s1", "smiles": "CCC(C)C(N)C(=O)O", "mz": [57.5022, 61.9879, 77.8427, 82.3928, 92.373, 118.5665, 121.2038, 128.0355, 130.0508, 130.0873], "intensities": [1775.0, 3359.0, 2062.0, 2369.0, 2503.0, 2436.0, 2218.0, 2826.0, 4871.0, 159616.0]}
{"ix": 210601, "database": "metatlas", "id": "c18atlas20200609-M02A11", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [54.4385, 56.7161, 68.4565, 71.6736, 73.9517, 86.097, 105.2156, 121.3192, 132.1037], "intensities": [67101.0, 73425.0, 81279.0, 79273.0, 88820.0, 9937913.0, 75624.0, 86752.0, 82323.0]}
{"ix": 210602, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c33", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [67.3371, 69.0705, 81.339, 86.097, 89.7397, 92.3681, 106.9589, 118.6424, 132.102], "intensities": [66941.0, 1245820.0, 54796.0, 9036559.0, 61942.0, 85057.0, 61648.0, 85437.0, 78653.0]}
{"ix": 210603, "database": "metatlas", "id": "c18atlas20200609-C0907", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [55.0548, 56.0501, 60.5338, 73.0653, 83.0496, 86.0968, 92.3761, 99.0033, 100.0761, 104.279, 115.0756, 118.6514, 122.0196, 132.1018], "intensities": [1835086.0, 371857.0, 60321.0, 2014203.0, 229665.0, 5251014.0, 61329.0, 225482.0, 113241.0, 54918.0, 1810396.0, 53570.0, 265919.0, 82804.0]}
{"ix": 210604, "database": "metatlas", "id": "c18atlas20200609-NPL800-ST074999", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [50.8915, 54.157, 61.011, 63.9982, 70.6869, 78.7178, 86.0966, 99.0032, 122.0194, 132.1018], "intensities": [1716.0, 1693.0, 7472.0, 4631.0, 1795.0, 2227.0, 252904.0, 9647.0, 14879.0, 4797.0]}
{"ix": 210605, "database": "metatlas", "id": "c18atlas20200609-M02A11", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [61.988, 63.744, 66.7409, 73.9515, 92.3769, 117.1087, 130.0874], "intensities": [12840.0, 2160.0, 2276.0, 1927.0, 2123.0, 1871.0, 204320.0]}
{"ix": 210606, "database": "metatlas", "id": "c18atlas20200609-QCsopV3c33", "name": "DL-Leucine", "decimal": 4.0, "inchi_key": "ROHFNLRQFUQHCH-UHFFFAOYSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-4(2)3-5(7)6(8)9/h4-5H,3,7H2,1-2H3,(H,8,9)", "smiles": "CC(C)CC(N)C(=O)O", "mz": [57.5022, 61.9879, 77.8427, 82.3928, 92.373, 118.5665, 121.2038, 128.0355, 130.0508, 130.0873], "intensities": [1775.0, 3359.0, 2062.0, 2369.0, 2503.0, 2436.0, 2218.0, 2826.0, 4871.0, 159616.0]}
{"ix": 210853, "database": "metatlas", "id": "c18atlas20200609-M04B07", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 132.1019287, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": "CCCCC(N)C(=O)O", "mz": [69.0706, 82.3811, 85.9283, 86.0971, 104.2648, 106.2866, 122.0196, 130.1295], "intensities": [1609755.0, 111145.0, 107489.0, 19635828.0, 136636.0, 97884.0, 109386.0, 94864.0]}
{"ix": 210854, "database": "metatlas", "id": "c18atlas20200609-M04B07", "name": "L-Norleucine", "decimal": 4.0, "inchi_key": "LRQKBLKVPFOOQJ-YFKPBYRVSA-N", "precursor_mz": 130.0873287, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C6H13NO2", "mono_isotopic_molecular_weight": 131.0946287, "inchi": "InChI=1S/C6H13NO2/c1-2-3-4-5(7)6(8)9/h5H,2-4,7H2,1H3,(H,8,9)/t5-/m0/s1", "smiles": "CCCCC(N)C(=O)O", "mz": [50.7763, 53.5008, 61.9879, 81.4991, 92.3849, 92.5489, 130.0874], "intensities": [1967.0, 1634.0, 4409.0, 1831.0, 2289.0, 1849.0, 188624.0]}
{"ix": 215149, "database": "metatlas", "id": "c18atlas20200609-M01D04", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 136.0617952, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": "Nc1nc[nH]c2ncnc1-2", "mz": [62.7196, 67.0297, 94.0404, 109.0113, 109.0515, 109.4987, 112.0509, 119.0354, 121.0874, 122.5546, 136.0223, 136.0618, 136.1122, 136.9603, 137.0459], "intensities": [5376.0, 6828.0, 153320.0, 8061.0, 15000.0, 7303.0, 24453.0, 332638.0, 10013.0, 6178.0, 141328.0, 4739170.0, 40310.0, 5672.0, 246659.0]}
{"ix": 215150, "database": "metatlas", "id": "c18atlas20200609-M01D04", "name": "adenine", "decimal": 4.0, "inchi_key": "GFFGJBXGBJISGV-UHFFFAOYSA-N", "precursor_mz": 134.0471952, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "QEHF", "instrument_type": "Orbitrap", "formula": "C5H5N5", "mono_isotopic_molecular_weight": 135.0544952, "inchi": "InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)", "smiles": "Nc1nc[nH]c2ncnc1-2", "mz": [65.014, 68.025, 92.0254, 92.396, 107.0364, 108.6023, 134.0473, 134.8656], "intensities": [8960.0, 3680.0, 89313.0, 2834.0, 343912.0, 3136.0, 937489.0, 4180.0]}
{"ix": 216450, "database": "metatlas", "id": "hatzenpichlerrefstdsb31ba3e6cd2e436aa6247f219d0cf321", "name": "Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 104.0353, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.042593084, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": "NC(CO)C(=O)O", "mz": [41.9987, 44.0144, 56.3493, 72.0092, 72.0263, 73.0128, 74.0249, 83.4545, 99.2627, 103.9206, 104.0128, 104.0356, 105.0391, 107.4165, 111.1083, 112.8336, 114.9396, 119.9618], "intensities": [13902.45703, 6854.51172, 6078.37402, 713562.9375, 6965.69092, 8284.52344, 456419.15625, 5874.604, 5015.76709, 26999.33008, 23901.42578, 1132199.25, 18938.63672, 5288.97363, 5432.41748, 5204.95166, 5204.49414, 5206.78271]}
{"ix": 216451, "database": "metatlas", "id": "hatzenpichlerrefstds0d7660a5ad124a7087e73bfcb08bb0ee", "name": "Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 104.0353, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.042593084, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": "NC(CO)C(=O)O", "mz": [41.2875, 41.9986, 44.4387, 72.0092, 74.0249, 88.0144, 103.9202, 104.0129, 104.0264, 104.0355, 104.0446, 105.0386], "intensities": [6861.27881, 6384.99463, 6222.27734, 502513.875, 484507.1875, 6161.24414, 11256.24121, 9586.97852, 13632.94141, 493330.65625, 12300.21777, 9245.11035]}
{"ix": 216462, "database": "metatlas", "id": "hatzenpichlerrefstdsd53e45350f80421093f5b56c205caf86", "name": "Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 106.0499, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.042593084, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": "NC(CO)C(=O)O", "mz": [40.6597, 41.4845, 60.0444, 63.6108, 65.2123, 86.5899, 88.0395, 106.0501, 117.4417], "intensities": [8503.69043, 7719.24072, 98116.9375, 8025.75342, 7472.55225, 7813.15234, 23225.34375, 32792.78906, 7402.21143]}
{"ix": 216463, "database": "metatlas", "id": "hatzenpichlerrefstdsc907d3c1cc9b4dac87d0a68693b52274", "name": "Serine", "decimal": 4.0, "inchi_key": "MTCFGRXMJLQNBG-UHFFFAOYSA-N", "precursor_mz": 106.0499, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C3H7NO3", "mono_isotopic_molecular_weight": 105.042593084, "inchi": "InChI=1S/C3H7NO3/c4-2(1-5)3(6)7/h2,5H,1,4H2,(H,6,7)", "smiles": "NC(CO)C(=O)O", "mz": [57.8234, 60.0445, 71.3652, 88.0393, 106.0502], "intensities": [8321.2627, 96463.38281, 7444.92676, 14036.89062, 13122.83398]}
{"ix": 219969, "database": "metatlas", "id": "wrighton20251527274614f44d44a502c310953ddd85", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 145.0983, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [41.8665, 55.4039, 61.7204, 66.5635, 76.6881, 77.7813, 90.2236, 98.257, 101.0607, 109.7066, 126.9646, 144.0018, 145.0272, 145.0982, 145.963, 146.0246, 146.1012], "intensities": [12107.33594, 11283.52734, 10325.0625, 12090.03027, 11822.49609, 11206.37891, 12640.35645, 10673.1377, 17194.73438, 11765.62109, 2411333.0, 10774.59961, 20405.57617, 2508478.5, 686017.625, 129738.6875, 41207.76953]}
{"ix": 219972, "database": "metatlas", "id": "wrighton202526c4df4245f346f3a3129df5abcb3bdf", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 145.0983, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [41.9985, 42.9096, 48.3173, 78.7877, 83.7243, 126.9647, 145.0983, 145.9632, 146.0249, 146.1015], "intensities": [13407.53906, 11192.98047, 11567.91895, 12118.8457, 10563.63184, 2256333.5, 1656562.375, 171767.34375, 87499.07031, 26372.14453]}
{"ix": 220089, "database": "metatlas", "id": "wrighton2025d641a41601e64c6fa372ca7296db174f", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.1128, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [51.5793, 53.7647, 54.9666, 55.0542, 69.7057, 79.8068, 84.0808, 85.0781, 85.0841, 85.091, 101.0598, 101.4237, 112.1734, 130.0864, 130.1055, 131.0899, 132.0904, 147.1133, 148.0789, 148.0955, 148.1148, 149.0604], "intensities": [8447.03711, 8214.43262, 8261.24902, 8096.81689, 7738.30322, 9082.60645, 305653.625, 42837.27344, 757122.5, 15887.59863, 23339.16016, 7180.12842, 8115.47217, 114705.60938, 18614.00977, 823184.1875, 15918.75488, 27030.55859, 39661.69141, 18331.33789, 384786.875, 13493.63965]}
{"ix": 220092, "database": "metatlas", "id": "wrighton2025e87645da118545eea3df65a7641e8441", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 129.1022, "polarity": "positive", "adduct": "[M-H2O+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [47.3661, 56.0498, 70.5358, 72.9846, 78.7375, 84.0808, 84.086, 84.0938, 84.1019, 84.494, 92.3226, 101.107, 111.5742, 112.0755, 129.1022, 129.1125, 130.0532, 130.0606, 130.0863, 130.0957, 130.1058, 130.1267], "intensities": [9869.00586, 16319.74902, 8911.5127, 8882.11426, 8253.83594, 3336591.75, 47510.42188, 23042.54492, 14940.69434, 8958.83203, 9058.67383, 15779.83301, 8360.52344, 14654.43652, 578053.5625, 8929.17383, 13525.20117, 23820.06836, 4331682.0, 81100.21875, 47168.34766, 13173.06934]}
{"ix": 220095, "database": "metatlas", "id": "wrighton20256ccf7eb31a334aa2a88479addd6ff02d", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.1128, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [41.3783, 44.2352, 47.8841, 55.0544, 56.0496, 67.0543, 74.0238, 84.0635, 84.0677, 84.0756, 84.0808, 84.1022, 85.0649, 85.0841, 95.0491, 99.786, 101.1075, 102.0915, 112.076, 123.9642, 129.1024, 130.0617, 130.0774, 130.0863, 131.0895, 147.1007, 147.1129, 148.0794, 157.9502], "intensities": [9587.50195, 9004.89355, 9015.22852, 14187.28711, 103782.33594, 108900.88281, 31511.44336, 35708.12891, 48959.89453, 94473.63281, 7329778.0, 23789.94531, 45070.24609, 144757.48438, 17955.76367, 13608.03125, 16420.57617, 32389.12305, 26489.78125, 19243.09961, 86071.14844, 13558.16797, 44198.01172, 2098344.25, 49638.37891, 9996.24414, 599837.8125, 15979.84473, 9925.94434]}
{"ix": 220098, "database": "metatlas", "id": "wrighton2025832f787a310b4fed921130f0c85cc331", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 129.1022, "polarity": "positive", "adduct": "[M-H2O+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "EXP120B", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [52.4942, 56.0495, 67.0544, 84.0635, 84.0808, 84.0938, 84.102, 85.0642, 85.0839, 104.9986, 112.076, 129.1022, 130.0607, 130.0863, 130.0955], "intensities": [8525.44727, 78553.11719, 17160.85742, 25810.39258, 5084836.0, 30806.41992, 24948.37109, 10677.76562, 16111.59961, 9248.97559, 17118.92383, 207859.5625, 11665.85645, 1944512.625, 39124.03125]}
{"ix": 220235, "database": "metatlas", "id": "wrighton202529b01de9197b41be94148ea876a16653", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.1128, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "IQX", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [48.0327, 71.9286, 78.742, 84.0801, 84.9591, 86.0867, 89.9392, 90.7034, 105.0327, 107.9498, 109.5999, 112.9552, 130.9656, 131.0849, 132.0905, 141.1429, 148.9765, 149.0227, 149.0588, 149.1169, 154.7339], "intensities": [8041.65039, 24599.57227, 8056.05078, 58489.5625, 18185.12891, 9806.22461, 74594.35156, 9064.65527, 18668.10156, 35389.32031, 9237.59473, 10757.42773, 230658.0, 34680.76562, 76123.45312, 8858.50879, 39127.57812, 27230.05469, 42604.58203, 56308.875, 25030.9668]}
{"ix": 220237, "database": "metatlas", "id": "wrighton2025a75e906cb85047288094bcf4e5a96950", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 169.0947, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "IQX", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [44.3751, 48.0552, 53.5084, 64.4342, 73.026, 76.0508, 95.6269, 114.3363, 127.2133, 152.0676, 169.0774, 169.0937], "intensities": [7690.65869, 8042.48828, 8587.92285, 8174.86621, 7863.35107, 7500.36816, 8563.24805, 9192.29395, 8705.5918, 15504.72168, 11841.35254, 255020.32812]}
{"ix": 220239, "database": "metatlas", "id": "wrighton202558560ad6b7fa408b8bde0503717e291f", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 147.1128, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "IQX", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [48.0702, 49.5046, 54.2267, 71.9287, 72.9365, 77.0381, 78.495, 84.0802, 84.959, 85.0833, 86.0806, 86.0869, 89.9391, 91.0536, 93.4169, 95.0488, 105.0327, 107.2852, 107.9497, 127.2081, 130.9655, 131.0846, 132.0907, 148.9764, 149.023, 149.0591, 154.7347], "intensities": [8311.64746, 8649.20898, 9826.52441, 50458.53516, 35713.71484, 10637.55469, 9229.34863, 90274.85156, 22473.30469, 18860.13672, 7917.52197, 28496.25391, 41265.01562, 11966.45508, 9540.31543, 13232.65039, 15846.23926, 7667.8623, 37618.56641, 9763.72363, 58086.57812, 10926.30664, 39132.42578, 10867.82031, 10189.20605, 13564.5625, 11115.29004]}
{"ix": 220241, "database": "metatlas", "id": "wrighton20252bf2b28d94fc4cffbef9fd64cfb88d5e", "name": "L-Lysine", "decimal": 4.0, "inchi_key": "KDXKERNSBIXSRK-YFKPBYRVSA-N", "precursor_mz": 169.0947, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "HCD", "collision_energy": null, "instrument": "IQX", "instrument_type": "Orbitrap", "formula": "C6H14N2O2", "mono_isotopic_molecular_weight": 146.105527688, "inchi": "InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1", "smiles": "NCCCC[C@H](N)C(=O)O", "mz": [44.1747, 44.3193, 45.4645, 49.8115, 53.5054, 56.6444, 65.3456, 73.8035, 74.9498, 119.0252, 131.2935, 154.7304, 169.0938], "intensities": [9613.03711, 7808.07178, 9064.68066, 8719.9209, 9094.3916, 8747.55664, 9029.90039, 8645.44922, 8512.74023, 9015.37891, 9828.0127, 10663.53516, 39297.23828]}
{"ix": 71, "database": "metatlas", "id": "93fd7dad83114081a76e51df3276aa2f", "name": "L-histidine", "decimal": 4.0, "inchi_key": "HNDVDQJCIGZPNO-YFKPBYRVSA-N", "precursor_mz": 154.062, "polarity": "negative", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H9N3O2", "mono_isotopic_molecular_weight": 155.0694765, "inchi": "InChI=1S/C6H9N3O2/c7-5(6(10)11)1-4-2-8-3-9-4/h2-3,5H,1,7H2,(H,8,9)(H,10,11)/t5-/m0/s1", "smiles": null, "mz": [51.2329, 53.7193, 64.2214, 67.0289, 72.0078, 76.969, 80.0368, 81.0445, 93.0446, 108.056, 108.766, 109.04, 110.071, 110.295, 118.04, 136.051, 137.035, 138.039, 153.65, 154.062, 154.947, 155.064], "intensities": [8512.0, 7962.0, 10088.0, 20299.0, 62265.0, 16813.0, 17499.0, 70593.0, 418347.0, 65605.0, 10060.0, 46795.0, 163568.0, 11845.0, 15494.0, 48996.0, 385968.0, 12249.0, 10378.0, 792539.0, 2909880.0, 16362.0]}
{"ix": 575, "database": "metatlas", "id": "ee3b3881b5af49cb854876f7ed7985d8", "name": "L-histidine", "decimal": 4.0, "inchi_key": "HNDVDQJCIGZPNO-YFKPBYRVSA-N", "precursor_mz": 156.077, "polarity": "positive", "adduct": null, "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H9N3O2", "mono_isotopic_molecular_weight": 155.0694765, "inchi": "InChI=1S/C6H9N3O2/c7-5(6(10)11)1-4-2-8-3-9-4/h2-3,5H,1,7H2,(H,8,9)(H,10,11)/t5-/m0/s1", "smiles": null, "mz": [53.2108, 55.4719, 56.026, 61.5148, 63.13, 63.1329, 79.3942, 81.0455, 81.7705, 82.0535, 83.0612, 90.9248, 92.1052, 93.0455, 95.0611, 110.072, 111.056, 111.069, 111.075, 114.267, 118.642, 138.067, 139.05, 149.858, 156.077, 157.081], "intensities": [31119.0, 23098.0, 20590.0, 28868.0, 25782.0, 42642.0, 36585.0, 41624.0, 55310.0, 171059.0, 812981.0, 34349.0, 90539.0, 332274.0, 1471530.0, 24634300.0, 47384.0, 122586.0, 616321.0, 31549.0, 25540.0, 53794.0, 30484.0, 45034.0, 6815280.0, 240211.0]}
{"ix": 192348, "database": "metatlas", "id": "453948d7ce12467ba0937ce2bc9b08f8", "name": "L-histidine", "decimal": 4.0, "inchi_key": "HNDVDQJCIGZPNO-YFKPBYRVSA-N", "precursor_mz": 154.062, "polarity": "negative", "adduct": "[M-H]-", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H9N3O2", "mono_isotopic_molecular_weight": 155.0694765, "inchi": "InChI=1S/C6H9N3O2/c7-5(6(10)11)1-4-2-8-3-9-4/h2-3,5H,1,7H2,(H,8,9)(H,10,11)/t5-/m0/s1", "smiles": null, "mz": [58.0692, 66.4914, 66.7615, 67.0296, 72.0087, 74.0244, 80.0377, 81.0454, 82.0171, 88.9203, 93.0456, 93.3744, 98.4899, 102.628, 104.313, 108.057, 109.041, 110.072, 110.08, 118.041, 136.052, 136.333, 137.036, 151.038, 154.062], "intensities": [60784.0, 73731.0, 81192.0, 770493.0, 1734580.0, 91115.0, 891250.0, 1169480.0, 79984.0, 68573.0, 8843560.0, 70118.0, 84114.0, 73861.0, 106592.0, 1392070.0, 1102750.0, 3361150.0, 113487.0, 500639.0, 1349670.0, 74656.0, 9202850.0, 116314.0, 22704200.0]}
{"ix": 192859, "database": "metatlas", "id": "560b7bf00fd5495a9e6cdc45536376d5", "name": "L-histidine", "decimal": 4.0, "inchi_key": "HNDVDQJCIGZPNO-YFKPBYRVSA-N", "precursor_mz": 178.059, "polarity": "positive", "adduct": "[M+Na]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H9N3O2", "mono_isotopic_molecular_weight": 155.0694765, "inchi": "InChI=1S/C6H9N3O2/c7-5(6(10)11)1-4-2-8-3-9-4/h2-3,5H,1,7H2,(H,8,9)(H,10,11)/t5-/m0/s1", "smiles": null, "mz": [64.8262, 82.4128, 88.0217, 92.3933, 98.9844, 133.037, 135.091, 137.121, 159.964, 175.529, 177.975, 178.059], "intensities": [1962.0, 2005.0, 16668.0, 2765.0, 2616.0, 36006.0, 1929.0, 2024.0, 3776.0, 1951.0, 9071.0, 188311.0]}
{"ix": 192860, "database": "metatlas", "id": "d4fbde171db04d11b624299c5c6b19ba", "name": "L-histidine", "decimal": 4.0, "inchi_key": "HNDVDQJCIGZPNO-YFKPBYRVSA-N", "precursor_mz": 156.077, "polarity": "positive", "adduct": "[M+H]+", "fragmentation_method": "cid", "collision_energy": null, "instrument": null, "instrument_type": null, "formula": "C6H9N3O2", "mono_isotopic_molecular_weight": 155.0694765, "inchi": "InChI=1S/C6H9N3O2/c7-5(6(10)11)1-4-2-8-3-9-4/h2-3,5H,1,7H2,(H,8,9)(H,10,11)/t5-/m0/s1", "smiles": null, "mz": [54.1546, 66.7538, 73.9653, 81.9797, 82.0531, 83.0608, 93.0451, 95.0608, 110.071, 111.056, 138.066, 150.321, 156.077], "intensities": [23994.0, 26332.0, 28819.0, 23714.0, 128483.0, 492566.0, 297150.0, 1135080.0, 16821600.0, 54367.0, 23499.0, 20691.0, 2915520.0]}
EOF

echo "   Created ms2_references.json"

if [[ "$SUBSET_H5" == true ]]; then
    echo ""
    echo "Subsetting copied h5 files..."
    echo "----------------------------------"

    # Use conservative defaults so rows needed by both RT alignment and targeted analysis are retained.
    SUBSET_PPM_MS1=5.0
    SUBSET_PPM_MS2=20.0
    SUBSET_EXTRA_TIME=0.5

    H5_DIR="raw_data/dev/${STANDALONE_PROJECT}"

    python3 - <<PY
import csv
import sys
import tables
import numpy as np
from pathlib import Path


def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_windows(tsv_path):
    windows = []
    if not tsv_path.exists():
        return windows
    with tsv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mz = parse_float(row.get("mz"))
            rt_min = parse_float(row.get("rt_min"))
            rt_max = parse_float(row.get("rt_max"))
            if mz is None or rt_min is None or rt_max is None:
                continue
            windows.append((mz, rt_min, rt_max))
    return windows


def ppm_bounds(mz, ppm):
    delta = mz * ppm / 1e6
    return mz - delta, mz + delta


def filter_h5_table(h5file, node_path, mz_col, rt_col, windows, ppm, extra_time):
    """Read a PyTables node and return a filtered structured array.

    Mirrors the logic in extract_data_from_h5._load_h5_table + _join_ms1/ms2_to_atlas:
    - For MS1 nodes, the production code reads the *_mz sorted variant and uses
      searchsorted for a fast mz-range pre-filter, then does a full mz+rt interval
      join per atlas window.
    - We replicate this by keeping any row whose (mz, rt) falls within at least one
      atlas window (mz ± ppm, rt ± extra_time).  The result is a superset of what
      the pipeline would actually use, ensuring no data is lost.
    """
    try:
        node = h5file.get_node(node_path)
    except tables.NoSuchNodeError:
        return None

    try:
        data = node.read()
    except Exception as exc:
        print(f"    Warning: could not read {node_path}: {exc}")
        return None

    if len(data) == 0:
        return data

    mz_vals = data[mz_col].astype(np.float64)
    rt_vals = data[rt_col].astype(np.float64)

    keep = np.zeros(len(data), dtype=bool)
    for mz, rt_min, rt_max in windows:
        mz_lo, mz_hi = ppm_bounds(mz, ppm)
        rt_lo = rt_min - extra_time
        rt_hi = rt_max + extra_time
        mask = (mz_vals >= mz_lo) & (mz_vals <= mz_hi) & (rt_vals >= rt_lo) & (rt_vals <= rt_hi)
        keep |= mask

    return data[keep]


def rewrite_h5_file(h5_path, windows_ms1, windows_ms2, ppm_ms1, ppm_ms2, extra_time):
    """Filter an h5 file in-place, keeping only rows within atlas windows.

    The production pipeline (extract_data_from_h5._load_h5_table) reads MS1 data
    from the *_mz sorted variant (e.g. ms1_pos_mz) using searchsorted for a fast
    mz pre-filter, then does a full interval join.  We must filter BOTH the base
    node (ms1_pos) AND its sorted variant (ms1_pos_mz) identically so the file
    remains consistent and the pipeline can read either form.

    MS2 nodes have no _mz variant — the pipeline reads ms2_pos directly.
    """
    import shutil

    # Base MS1/MS2 node paths
    ms1_base_keys = {"/ms1_pos", "/ms1_neg"}
    ms2_base_keys = {"/ms2_pos", "/ms2_neg"}
    # Sorted-by-mz MS1 variants read by the production pipeline
    ms1_mz_keys = {"/ms1_pos_mz", "/ms1_neg_mz"}

    tmp_path = h5_path.with_suffix(".tmp.h5")
    changed = False

    try:
        with tables.open_file(str(h5_path), mode="r") as src, \
             tables.open_file(str(tmp_path), mode="w") as dst:

            for node in src.walk_nodes("/", classname="Table"):
                node_path = node._v_pathname
                is_ms1_base = node_path in ms1_base_keys
                is_ms1_mz   = node_path in ms1_mz_keys
                is_ms2      = node_path in ms2_base_keys

                if (is_ms1_base or is_ms1_mz) and windows_ms1:
                    # Both the base and the _mz sorted variant are filtered the same way.
                    # The _mz variant is sorted by mz, so searchsorted works on it; our
                    # mask approach is equivalent and preserves the sort order because we
                    # keep a contiguous boolean mask (not a shuffle).
                    filtered = filter_h5_table(src, node_path, "mz", "rt", windows_ms1, ppm_ms1, extra_time)
                elif is_ms2 and windows_ms2:
                    filtered = filter_h5_table(src, node_path, "precursor_MZ", "rt", windows_ms2, ppm_ms2, extra_time)
                else:
                    filtered = None

                parent_path = node._v_parent._v_pathname
                node_name = node._v_name

                if filtered is not None:
                    before = len(node.read())
                    after = len(filtered)
                    if after != before:
                        changed = True
                    # Create parent groups if needed
                    if parent_path != "/":
                        dst.create_group("/", parent_path.lstrip("/"), createparents=True)
                    dst.create_table(parent_path, node_name, obj=filtered,
                                     filters=tables.Filters(complevel=1, complib="blosc"))
                else:
                    # Copy node as-is (metadata tables, etc.)
                    node.copy(dst.get_node(parent_path) if parent_path != "/" else dst.root,
                              newname=node_name)

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        print(f"    Warning: failed to filter {h5_path.name}: {exc}")
        return False, 0, 0

    if changed:
        shutil.move(str(tmp_path), str(h5_path))
    else:
        tmp_path.unlink(missing_ok=True)

    return changed, 0, 0


root = Path(".")
h5_dir = root / Path("${H5_DIR}")
ppm_ms1 = float("${SUBSET_PPM_MS1}")
ppm_ms2 = float("${SUBSET_PPM_MS2}")
extra_time = float("${SUBSET_EXTRA_TIME}")

windows = []
windows.extend(parse_windows(root / "atlases/qc_compounds_pos.tsv"))
windows.extend(parse_windows(root / "atlases/ema_compounds_pos.tsv"))
windows.extend(parse_windows(root / "atlases/ema_compounds_neg.tsv"))

if not windows:
    print("No compound windows found; skipping h5 subsetting.")
    sys.exit(0)

files = sorted(h5_dir.glob("*.h5"))
if not files:
    print("No h5 files found; skipping h5 subsetting.")
    sys.exit(0)

print(f"Filtering {len(files)} h5 files to atlas windows (ppm_ms1={ppm_ms1}, ppm_ms2={ppm_ms2}, extra_time={extra_time} min)...")

updated = 0
for hfile in files:
    changed, _, _ = rewrite_h5_file(hfile, windows, windows, ppm_ms1, ppm_ms2, extra_time)
    if changed:
        updated += 1
    print(f"  {hfile.name}: {'filtered' if changed else 'unchanged'}")

print(f"H5 subset summary: {len(files)} files processed, {updated} rewritten")
PY

    echo "   H5 subsetting complete"
fi

echo ""
echo "Creating configuration files..."
echo "----------------------------------------"

# Create compounds configuration
cat > configs/compounds_config.yaml << 'EOF'
PARAMS:
  use_pubchem_cache: true
  update_pubchem_cache: false

COMPOUNDS:
    HILICZ:
      POS:
        PATHS:
          - atlases/qc_compounds_pos.tsv
          - atlases/ema_compounds_pos.tsv
          - atlases/ema_compounds_neg.tsv
EOF

# Create atlases configuration
cat > configs/atlases_config.yaml << 'EOF'
ATLASES:
  HILICZ:
    POS:
      QC:
        path: atlases/qc_compounds_pos.tsv
        name: Dev HILICZ QC Atlas Positive
        desc: Dev quality control compounds for RT alignment
      EMA:
        path: atlases/ema_compounds_pos.tsv
        name: Dev HILICZ EMA Atlas Positive
        desc: Dev metabolite atlas for positive mode
    NEG:
      EMA:
        path: atlases/ema_compounds_neg.tsv
        name: Dev HILICZ EMA Atlas Negative
        desc: Dev metabolite atlas for negative mode
EOF

# Create analysis configuration
cat > configs/analysis_config.yaml << 'EOF'
WORKFLOWS:
  PATHS:
    owner: dev
    msms_refs_path: databases/msms_refs/ms2_references.json
    gdrive_subfolder: 
  RT_ALIGNMENT:
    HILICZ:
      ATLAS:
        uid: dev-qc-hilicz-pos
      PARAMS:
          upload_to_gdrive: true
          include_lcmsruns: # 'QC'
          exclude_lcmsruns:
            - NEG
          use_existing_rt_alignment: true
          remove_unided_compounds: false
          only_keep_data_in_feature: true
          atlas_extra_time: 0.0
          ms1_min_peak_intensity: 0
          ms1_min_num_points: 0
          ms1_mz_tolerance_ppm: 5.0
          apply_model_to_min_max: true
          polynomial_degree: 2
          min_observations_per_compound: 1
          min_compounds_for_modeling: 2
          r2_threshold: 0.5
  TARGETED_ANALYSES:
    HILICZ:
      POS:
        EMA:
          DEFAULT:
            ATLAS:
              uid: dev-ema-hilicz-pos
            PARAMS:
              include_lcmsruns: # 'EXPERIMENTAL', 'ISTD', 'EXCTRL', 'REFSTD', 'INJBLK'
              exclude_lcmsruns:
                data_extraction:
                  - QC
                  - NEG
                gui: # INJBL, BLANK
                id_sheet: # INJBL, BLANK, REFSTD
                chromatograms: # INJBL, BLANK
                id_plots: # INJBL, BLANK, REFSTD
                data_sheets: # INJBL, BLANK
              do_alignment: true
              remove_unided_compounds: true
              remove_flagged_compounds: true
              only_keep_data_in_feature: false
              apply_cross_polarity_curation: true
              suggested_min_conf: 0.75
              atlas_extra_time: 0
              ms1_min_peak_intensity: 2e5
              ms1_min_num_points: 2
              ms1_mz_tolerance_ppm: 5.0
              ms2_min_num_scans: 1
              ms2_min_precursor_intensity: 0
              ms2_min_score: 0.0
              ms2_min_matching_frags: 1
              ms2_mz_tolerance_ppm: 20.0
              ms2_frag_mz_tolerance: 0.05
              gui_require_all_evaluated: false
              gui_top_n_hits: 10
              gui_lcmsruns_colors:
                ISTD: blue
                QC: orange
                EXCTRL: red
                TXCTRL: green
                REFSTD: black
              note_options_overrides:
                ms1_notes:
                ms2_notes:
                other_notes:
              create_curation_notebooks: true
              upload_to_gdrive: true
              skip_outputs:
      NEG:
        EMA:
          DEFAULT:
            ATLAS:
              uid: dev-ema-hilicz-neg
            PARAMS:
              include_lcmsruns: # 'EXPERIMENTAL', 'ISTD', 'EXCTRL', 'REFSTD', 'INJBLK'
              exclude_lcmsruns:
                data_extraction:
                  - QC
                  - POS
                gui: # INJBL, BLANK
                id_sheet: # INJBL, BLANK, REFSTD
                chromatograms: # INJBL, BLANK
                id_plots: # INJBL, BLANK, REFSTD
                data_sheets: # INJBL, BLANK
              do_alignment: true
              remove_unided_compounds: true
              remove_flagged_compounds: true
              only_keep_data_in_feature: false
              apply_cross_polarity_curation: true
              suggested_min_conf: 0.75
              atlas_extra_time: 0
              ms1_min_peak_intensity: 2e5
              ms1_min_num_points: 2
              ms1_mz_tolerance_ppm: 5.0
              ms2_min_num_scans: 1
              ms2_min_precursor_intensity: 0
              ms2_min_score: 0.0
              ms2_min_matching_frags: 1
              ms2_mz_tolerance_ppm: 20.0
              ms2_frag_mz_tolerance: 0.05
              gui_require_all_evaluated: false
              gui_top_n_hits: 10
              gui_lcmsruns_colors:
                ISTD: blue
                QC: orange
                EXCTRL: red
                TXCTRL: green
                REFSTD: black
              note_options_overrides:
                ms1_notes:
                ms2_notes:
                other_notes:
              create_curation_notebooks: true
              upload_to_gdrive: true
              skip_outputs:
EOF

echo "   Created configs/compounds_config.yaml"
echo "   Created configs/atlases_config.yaml"
echo "   Created configs/analysis_config.yaml"

echo ""
echo "Creating tarball..."
echo "---------------------------"

cd "${OUTPUT_DIR}"
tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}/"

TARBALL_SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)
EXTRACTED_SIZE=$(du -sh "${PACKAGE_NAME}" | cut -f1)

echo "   Created ${PACKAGE_NAME}.tar.gz"
echo "    Compressed size: ${TARBALL_SIZE}"
echo "    Extracted size: ${EXTRACTED_SIZE}"

fi  # End of package creation (skip if --keep-existing-archive)

# Ensure we have the tarball size for upload step
if [[ -z "${TARBALL_SIZE:-}" ]]; then
    TARBALL_SIZE=$(du -h "${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz" | cut -f1)
fi

echo ""
if [[ "$SKIP_ZENODO_UPLOAD" == true ]]; then
    echo "Skipping Zenodo upload as requested."
    echo "Package location: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz (size: ${TARBALL_SIZE})"
    exit 0
fi

echo ""
if [[ "$KEEP_EXISTING" == true ]]; then
    echo "Uploading to Zenodo..."
else
    echo "Uploading to Zenodo..."
fi
echo "-------------------------------"

# Check for Zenodo access token
if [[ -z "${ZENODO_ACCESS_TOKEN:-}" ]]; then
    echo "Warning: ZENODO_ACCESS_TOKEN not set. Skipping Zenodo upload."
    echo "Package location: ${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"
    exit 0
fi

ZENODO_RECORD_ID="20172505"
VERSION_TAG="v$(date +%Y%m%d)"
TARBALL_PATH="${OUTPUT_DIR}/${PACKAGE_NAME}.tar.gz"

echo "   Retrieving latest version of record ${ZENODO_RECORD_ID}..."

# First, get the published record to find the latest version's deposition ID
RECORD_RESPONSE=$(curl -s "https://zenodo.org/api/records/${ZENODO_RECORD_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# Extract the record_id (this is the deposition ID we need)
LATEST_DEPOSITION_ID=$(echo "$RECORD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('id', ''))" 2>/dev/null)

if [[ -z "$LATEST_DEPOSITION_ID" ]]; then
    echo "Error: Could not retrieve record. Response:"
    echo "$RECORD_RESPONSE"
    exit 1
fi

echo "   Latest deposition ID: ${LATEST_DEPOSITION_ID}"

# Check for any existing unpublished drafts by querying the deposition directly
echo "   Checking for existing drafts..."
DEPOSITION_CHECK=$(curl -s "https://zenodo.org/api/deposit/depositions/${LATEST_DEPOSITION_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# If the deposition has a 'newversion' link, there's no draft. If not, there might be a draft already.
HAS_NEWVERSION=$(echo "$DEPOSITION_CHECK" | python3 -c "import sys, json; d=json.load(sys.stdin); print('yes' if 'newversion' in d.get('links', {}) else 'no')" 2>/dev/null)

if [[ "$HAS_NEWVERSION" == "no" ]]; then
    # No newversion link means a draft already exists - need to find and delete it
    echo "   Found existing draft version, searching for it..."
    
    # Get all versions of this record
    CONCEPT_RECORD=$(curl -s "https://zenodo.org/api/records/${ZENODO_RECORD_ID}/versions" \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")
    
    # Find the draft deposition ID
    DRAFT_DEPOSITION_ID=$(echo "$CONCEPT_RECORD" | python3 -c "
import sys, json
hits = json.load(sys.stdin).get('hits', {}).get('hits', [])
for hit in hits:
    if hit.get('links', {}).get('self', '').startswith('https://zenodo.org/api/deposit/'):
        print(hit.get('id', ''))
        break
" 2>/dev/null)
    
    if [[ -n "$DRAFT_DEPOSITION_ID" ]] && [[ "$DRAFT_DEPOSITION_ID" != "$LATEST_DEPOSITION_ID" ]]; then
        echo "   Deleting draft deposition ${DRAFT_DEPOSITION_ID}..."
        curl -s -X DELETE "https://zenodo.org/api/deposit/depositions/${DRAFT_DEPOSITION_ID}" \
            -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" > /dev/null
        sleep 2
    fi
fi

echo "   Creating new version..."

# Create new version using the correct deposition ID
NEW_VERSION_RESPONSE=$(curl -s -X POST \
    "https://zenodo.org/api/deposit/depositions/${LATEST_DEPOSITION_ID}/actions/newversion" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

# Extract the latest_draft URL
LATEST_DRAFT_URL=$(echo "$NEW_VERSION_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['links']['latest_draft'])" 2>/dev/null)

if [[ -z "$LATEST_DRAFT_URL" ]]; then
    echo "Error: Failed to create new version. Response:"
    echo "$NEW_VERSION_RESPONSE"
    exit 1
fi

echo "   New draft created: $LATEST_DRAFT_URL"

# Get the new deposition details
NEW_DEPOSITION=$(curl -s "$LATEST_DRAFT_URL" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

NEW_DEPOSITION_ID=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
BUCKET_URL=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; print(json.load(sys.stdin)['links']['bucket'])")

echo "   New deposition ID: ${NEW_DEPOSITION_ID}"

# Delete old files from the new version
echo "   Removing old files from draft..."
OLD_FILES=$(echo "$NEW_DEPOSITION" | python3 -c "import sys, json; [print(f['id']) for f in json.load(sys.stdin)['files']]")

for file_id in $OLD_FILES; do
    curl -s -X DELETE \
        "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}/files/${file_id}" \
        -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" > /dev/null
done

# Upload new tarball
echo "   Uploading ${PACKAGE_NAME}.tar.gz (${TARBALL_SIZE})..."
UPLOAD_RESPONSE=$(curl -s \
    -X PUT "${BUCKET_URL}/${PACKAGE_NAME}.tar.gz" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    -H "Content-Type: application/octet-stream" \
    --data-binary "@${TARBALL_PATH}")

UPLOAD_KEY=$(echo "$UPLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('key', 'ERROR'))" 2>/dev/null)

if [[ "$UPLOAD_KEY" == "ERROR" ]] || [[ -z "$UPLOAD_KEY" ]]; then
    echo "Error: Upload failed. Response:"
    echo "$UPLOAD_RESPONSE"
    exit 1
fi

echo "   Upload successful: ${UPLOAD_KEY}"

# Update metadata with all required fields
echo "   Updating metadata with version ${VERSION_TAG}..."
METADATA_UPDATE=$(cat <<EOF
{
    "metadata": {
        "title": "Metatlas2 Development Environment Data",
        "upload_type": "dataset",
        "description": "This dataset contains all required files for testing the local standalone instance of the metatlas2 software.",
        "creators": [
            {
                "name": "Kieft, Brandon",
                "orcid": "0000-0003-2458-9844"
            }
        ],
        "version": "${VERSION_TAG}",
        "publication_date": "$(date +%Y-%m-%d)",
        "related_identifiers": [
            {
                "identifier": "https://github.com/bkieft-usa/metatlas2",
                "relation": "isSupplementTo",
                "resource_type": "software"
            }
        ]
    }
}
EOF
)

curl -s -X PUT \
    "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$METADATA_UPDATE" > /dev/null

# Publish the new version
echo "   Publishing new version..."
PUBLISH_RESPONSE=$(curl -s -X POST \
    "https://zenodo.org/api/deposit/depositions/${NEW_DEPOSITION_ID}/actions/publish" \
    -H "Authorization: Bearer ${ZENODO_ACCESS_TOKEN}")

NEW_DOI=$(echo "$PUBLISH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('doi', 'ERROR'))" 2>/dev/null)

if [[ "$NEW_DOI" == "ERROR" ]] || [[ -z "$NEW_DOI" ]]; then
    echo "Error: Publish failed. Response:"
    echo "$PUBLISH_RESPONSE"
    exit 1
fi

# Update scripts/metatlas2.sh to use the newly published DOI for standalone downloads.
METATLAS2_SCRIPT="${REPO_DIR}/scripts/metatlas2.sh"
if [[ -f "${METATLAS2_SCRIPT}" ]]; then
    DOI_URL="${NEW_DOI}"
    if [[ "${DOI_URL}" != https://doi.org/* ]]; then
        DOI_URL="https://doi.org/${DOI_URL}"
    fi

    DOI_PATTERN='^[[:space:]]*ZENODO_DOI="[^"]*"'
    if grep -qE "${DOI_PATTERN}" "${METATLAS2_SCRIPT}"; then
        sed -i -E "s|${DOI_PATTERN}|    ZENODO_DOI=\"${DOI_URL}\"|" "${METATLAS2_SCRIPT}"
        echo "Updated scripts/metatlas2.sh with DOI: ${DOI_URL}"
    else
        echo "Warning: Could not find ZENODO_DOI line in scripts/metatlas2.sh" >&2
    fi
else
    echo "Warning: scripts/metatlas2.sh not found at ${METATLAS2_SCRIPT}" >&2
fi

echo ""
echo "========================================"
echo "SUCCESS!"
echo "========================================"
echo "New version published: ${VERSION_TAG}"
echo "DOI: ${NEW_DOI}"
echo "Local package: ${TARBALL_PATH}"
echo ""
