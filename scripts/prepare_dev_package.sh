#!/bin/bash
# Prepare standalone development environment data package
#
# This script must be run at NERSC with access to production data.
# It extracts a minimal set of files and creates the dev package structure.
#
# Usage:
#   cd /global/homes/b/bkieft/metatlas2
#   ./scripts/prepare_dev_package.sh [output_dir]
#
# Output:
#   Creates metatlas2-dev-data.tar.gz ready for Zenodo upload

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${METATLAS_DATA_DIR}/databases/standalone_dev_data}"
PACKAGE_NAME="metatlas2-dev-data"

# Source project with representative ISTD data
SOURCE_PROJECT="20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066"
SOURCE_OWNER="jgi"
SOURCE_BASE="${METATLAS_DATA_DIR}/lcmsruns/${SOURCE_OWNER}/${SOURCE_PROJECT}"

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

# Validate source directory exists
if [[ ! -d "${SOURCE_BASE}/parquet" ]]; then
    echo "Error: Source parquet directory not found: ${SOURCE_BASE}/parquet" >&2
    echo "Please verify the project name and path." >&2
    exit 1
fi

# Create output structure
mkdir -p "${OUTPUT_DIR}/${PACKAGE_NAME}"/{parquet,configs}
cd "${OUTPUT_DIR}/${PACKAGE_NAME}"

# Remove raw/ directory if it exists from previous runs
rm -rf raw/

echo "Step 1: Collecting parquet files..."
echo "------------------------------------"

# Copy all parquet files for specific run types
# Parquet files are named: <basename>_ms1_pos.parquet, <basename>_ms1_neg.parquet, <basename>_ms2_pos.parquet, etc.

ISTD_RUNS=("Run58" "Run150" "Run21")
QC_RUNS=("Run57" "Run149" "Run195")
EXPERIMENTAL_RUNS=("Run29" "Run47" "Run55")
INJBL_RUNS=("Run36" "Run18" "Run183")
EXCTRL_RUNS=("Run15" "Run168" "Run122")

ALL_RUNS=("${ISTD_RUNS[@]}" "${QC_RUNS[@]}" "${EXPERIMENTAL_RUNS[@]}" "${INJBL_RUNS[@]}" "${EXCTRL_RUNS[@]}")

COPIED=0
MISSING=0

echo "Copying parquet files for ${#ALL_RUNS[@]} runs (each run may have multiple parquet files)..."
echo ""

for run in "${ALL_RUNS[@]}"; do
    # Find all parquet files for this run
    shopt -s nullglob
    run_files=("${SOURCE_BASE}/parquet/"*"${run}"*.parquet)
    shopt -u nullglob
    
    if [[ ${#run_files[@]} -gt 0 ]]; then
        echo -n "  [$((COPIED+MISSING+1))/${#ALL_RUNS[@]}] ${run}... "
        
        local_count=0
        for pfile in "${run_files[@]}"; do
            cp "$pfile" parquet/ && local_count=$((local_count + 1))
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

# Count total parquet files copied
TOTAL_PARQUET=$(find parquet/ -name "*.parquet" 2>/dev/null | wc -l)
echo ""
echo "Total parquet files copied: ${TOTAL_PARQUET}"

echo ""
echo "Step 2: Creating compound definitions..."
echo "-----------------------------------------"

# Create positive mode compounds
cat > compounds_pos.tsv << 'EOF'
inchi_key	compound_name	rt_max	rt_min	rt_peak	rt_units	detected_polarity	mz	mz_tolerance	mz_tolerance_units	mono_isotopic_molecular_weight	pubchem_compound_id	adduct	identification_notes
LCMZECCEEOQWLQ-UHFFFAOYSA-N	ABMBA (unlabeled)	1.393806323	0.793806323	1.093806323	min	positive	229.9811	5	ppm	228.9738406	2774400	[M+H]+	nan
GFFGJBXGBJISGV-UHFFFAOYSA-N	adenine (unlabeled)	2.977601998	2.377601998	2.677601998	min	positive	136.06177	5	ppm	135.0544952	190	[M+H]+	nan
UYTPUPDQBNUYGX-UHFFFAOYSA-N	guanine (unlabeled)	6.565359988	5.965359988	6.265359988	min	positive	152.05668	5	ppm	151.0494098	764	[M+H]+	nan
QNAYBMKLOCPYGJ-REOHCLBHSA-N	alanine (unlabeled)	13.70509074	13.10509074	13.40509074	min	positive	90.05495	5	ppm	89.04767846	5950	[M+H]+	nan
HDTRYLNUVZCQOY-LIZSDCNHSA-N	trehalose (unlabeled)	14.74384736	14.14384736	14.44384736	min	positive	365.1054242	5	ppm	342.1162115	7427	[M+Na]+	nan
KDXKERNSBIXSRK-YFKPBYRVSA-N	lysine (unlabeled)	17.31131041	16.71131041	17.01131041	min	positive	147.1128	5	ppm	146.1055277	5962	[M+H]+	nan
EOF

echo "   Created compounds_pos.tsv (6 compounds)"

# Create negative mode compounds
cat > compounds_neg.tsv << 'EOF'
inchi_key	compound_name	rt_max	rt_min	rt_peak	rt_units	detected_polarity	mz	mz_tolerance	mz_tolerance_units	mono_isotopic_molecular_weight	pubchem_compound_id	adduct	identification_notes
LCMZECCEEOQWLQ-UHFFFAOYSA-N	ABMBA (unlabeled)	1.421446478	0.821446478	1.121446478	min	negative	227.96656	5	ppm	228.9738406	2774400	[M-H]-	nan
GFFGJBXGBJISGV-UHFFFAOYSA-N	adenine (unlabeled)	2.977601998	2.377601998	2.677601998	min	negative	134.04721	5	ppm	135.0544952	190	[M-H]-	nan
UYTPUPDQBNUYGX-UHFFFAOYSA-N	guanine (unlabeled)	6.565359988	5.965359988	6.265359988	min	negative	150.04213	5	ppm	151.0494098	764	[M-H]-	nan
QNAYBMKLOCPYGJ-REOHCLBHSA-N	alanine (unlabeled)	13.70509074	13.10509074	13.40509074	min	negative	88.0404	5	ppm	89.04767846	5950	[M-H]-	nan
HDTRYLNUVZCQOY-LIZSDCNHSA-N	trehalose (unlabeled)	14.74384736	14.14384736	14.44384736	min	negative	341.10893	5	ppm	342.1162115	7427	[M-H]-	nan
KDXKERNSBIXSRK-YFKPBYRVSA-N	lysine (unlabeled)	17.31131041	16.71131041	17.01131041	min	negative	145.09825	5	ppm	146.1055277	5962	[M-H]-	nan
EOF

echo "   Created compounds_neg.tsv (6 compounds)"

echo ""
echo "Step 3: Creating MS2 references..."
echo "----------------------------------"

# Create MS2 references file
cat > ms2_references.tsv << 'EOF'
	database	id	name	spectrum	decimal	precursor_mz	polarity	adduct	fragmentation_method	collision_energy	instrument	instrument_type	formula	exact_mass	inchi_key	inchi	smiles
192032	metatlas	99d9dee377304155b946cc560f9c6c02	guanine	[[65.99850, 66.00940, 78.00970, 80.02530, 82.04100, 90.00970, 95.16990, 97.07310, 105.02100, 106.00500, 107.03600, 108.02100, 121.01600, 126.03100, 132.03200, 132.61300, 132.62200, 133.01600, 133.41900, 149.56000, 150.04200, 151.02600, 151.03900, 152.01000], [45074.00000, 1366820.00000, 516984.00000, 39261.00000, 1281920.00000, 384496.00000, 26391.00000, 28309.00000, 45602.00000, 192331.00000, 6603530.00000, 10719200.00000, 304343.00000, 8924520.00000, 263006.00000, 42141.00000, 27456.00000, 76227700.00000, 33658.00000, 29618.00000, 40922000.00000, 974198.00000, 55355.00000, 62405.00000]]	4.0	150.042	negative	[M-H]-	cid			C5H5N5O	151.0494098	UYTPUPDQBNUYGX-UHFFFAOYSA-N	InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)	
192038	metatlas	08bfcc1e58e94826ae3c5199ec3c3958	L-alanine	[[51.09330, 67.57920, 73.96510, 82.41020, 84.75570, 88.04040], [1854.00000, 1946.00000, 2102.00000, 1946.00000, 1829.00000, 186684.00000]]	4.0	88.0404	negative	[M-H]-	cid			C3H7NO2	89.04767846	QNAYBMKLOCPYGJ-REOHCLBHSA-N	InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1	
192056	metatlas	5938f71b59834597acdc23a8e6ef01a0	adenine	[[65.01410, 68.02500, 75.18480, 80.02530, 92.02540, 92.39530, 93.08310, 107.03600, 119.69200, 128.99400, 132.03200, 133.64000, 134.04700, 134.45600], [542179.00000, 403180.00000, 46141.00000, 380654.00000, 7633880.00000, 62138.00000, 54993.00000, 36504300.00000, 51458.00000, 51375.00000, 64773.00000, 79471.00000, 163076000.00000, 66448.00000]]	4.0	134.047	negative	[M-H]-	cid			C5H5N5	135.0544952	GFFGJBXGBJISGV-UHFFFAOYSA-N	InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)	
192142	metatlas	2c777781039348b780b8fdc86ee7a779	guanine	[[50.45480, 59.97690, 60.57930, 63.18150, 66.00930, 66.78810, 78.00940, 82.04080, 107.03600, 108.01200, 108.02000, 122.06100, 126.03100, 133.01600, 150.04200, 151.02600], [9195.00000, 7894.00000, 8892.00000, 7786.00000, 64439.00000, 9132.00000, 13148.00000, 48802.00000, 326011.00000, 16076.00000, 414266.00000, 20232.00000, 465851.00000, 3888630.00000, 2050780.00000, 70780.00000]]	4.0	150.042	negative	[M-H]-	cid			C5H5N5O	151.0494098	UYTPUPDQBNUYGX-UHFFFAOYSA-N	InChI=1S/C5H5N5O/c6-5-9-3-2(4(11)10-5)7-1-8-3/h1H,(H4,6,7,8,9,10,11)	
192200	metatlas	14cf99a83c144abbb46cc7f8c3f19874	L-alanine	[[54.32030, 73.95410, 82.39930, 88.04040], [23232.00000, 24775.00000, 31713.00000, 3382250.00000]]	4.0	88.0404	negative	[M-H]-	cid			C3H7NO2	89.04767846	QNAYBMKLOCPYGJ-REOHCLBHSA-N	InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1	
192304	metatlas	6d130f86669c4c37a04882e6ee5d3c40	L-lysine	[[55.18890, 67.59290, 74.00410, 79.03580, 82.91020, 92.44170, 97.07710, 99.09280, 104.35800, 128.07200, 132.43400, 136.30500, 145.09900], [2705.00000, 3199.00000, 3174.00000, 2544.00000, 3091.00000, 4588.00000, 6358.00000, 6597.00000, 3966.00000, 5424.00000, 2622.00000, 3576.00000, 1460770.00000]]	4.0	145.098	negative	[M-H]-	cid			C6H14N2O2	146.1055277	KDXKERNSBIXSRK-YFKPBYRVSA-N	InChI=1S/C6H14N2O2/c7-4-2-1-3-5(8)6(9)10/h5H,1-4,7-8H2,(H,9,10)/t5-/m0/s1	
192359	metatlas	4d3300ef478a4643b18fa0f3eb01a716	trehalose	[[58.00560, 59.01340, 71.01360, 72.99290, 73.02920, 75.34170, 83.01380, 85.02930, 85.25440, 86.01220, 87.00850, 89.02430, 101.02500, 104.31800, 105.96500, 113.02500, 119.03500, 131.03500, 143.03500, 149.04600, 161.04600, 179.04200, 179.05600, 341.10900], [18535.00000, 345908.00000, 302659.00000, 31307.00000, 11334.00000, 6898.00000, 14028.00000, 29277.00000, 6646.00000, 7436.00000, 38416.00000, 827253.00000, 302382.00000, 11966.00000, 7313.00000, 214777.00000, 454534.00000, 53780.00000, 95014.00000, 45845.00000, 124180.00000, 17428.00000, 602928.00000, 1117100.00000]]	4.0	341.109	negative	[M-H]-	cid			C12H22O11	342.1162115	HDTRYLNUVZCQOY-LIZSDCNHSA-N	InChI=1S/C12H22O11/c13-1-3-5(15)7(17)9(19)11(21-3)23-12-10(20)8(18)6(16)4(2-14)22-12/h3-20H,1-2H2/t3-,4-,5-,6-,7+,8+,9-,10-,11-,12-/m1/s1	
192378	metatlas	dc73b8eb9db143e48df1cb0516cdf6fe	adenine	[[58.10630, 63.01050, 74.65450, 80.95970, 80.96500, 83.51830, 92.02520, 92.39990, 107.03600, 116.92100, 133.99200, 134.04700], [1699.00000, 1703.00000, 1925.00000, 3568.00000, 102617.00000, 1873.00000, 11653.00000, 1955.00000, 60928.00000, 29803.00000, 10795.00000, 326077.00000]]	4.0	134.047	negative	[M-H]-	cid			C5H5N5	135.0544952	GFFGJBXGBJISGV-UHFFFAOYSA-N	InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)	
192494	metatlas	b40a2bd4c67845ae82f38598ccf4e354	L-alanine	[[53.38210, 59.74880, 72.08150, 85.58390, 104.29300, 134.01900, 134.06300, 135.44200], [1784.00000, 1960.00000, 10801.00000, 1899.00000, 2018.00000, 45110.00000, 13156.00000, 1916.00000]]	4.0	134.019	positive	[M-H+2Na]+	cid			C3H7NO2	89.04767846	QNAYBMKLOCPYGJ-REOHCLBHSA-N	InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1	
192672	metatlas	60ae976e212b483aa48011888e948a05	L-alanine	[[55.15990, 58.48300, 60.31330, 72.08140, 73.95580, 82.76740, 92.37870, 92.69710, 109.62600, 134.01900, 134.06300], [2130.00000, 1665.00000, 1784.00000, 14547.00000, 2873.00000, 2030.00000, 2260.00000, 2057.00000, 1778.00000, 32798.00000, 29302.00000]]	4.0	134.019	positive	[M-H+2Na]+	cid			C3H7NO2	89.04767846	QNAYBMKLOCPYGJ-REOHCLBHSA-N	InChI=1S/C3H7NO2/c1-2(4)3(5)6/h2H,4H2,1H3,(H,5,6)/t2-/m0/s1	
203825	metatlas	c18atlas20200609-M01D04	adenine	[[65.01376, 67.21752, 71.04333, 82.40697, 92.02531, 92.38770, 104.54678, 107.03629, 110.84988, 134.04716, 134.86562], [4771.00000, 3488.00000, 3308.00000, 3592.00000, 60734.00000, 4636.00000, 3128.00000, 269661.00000, 3141.00000, 1575772.00000, 4785.00000]]	4.0	134.0471952	negative	[M-H]-	HCD	ramp-102040	QEHF	Orbitrap	C5H5N5	135.0544952	GFFGJBXGBJISGV-UHFFFAOYSA-N	InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)	Nc1nc[nH]c2ncnc1-2
215150	metatlas	c18atlas20200609-M01D04	adenine	[[65.01403, 68.02502, 92.02540, 92.39604, 107.03639, 108.60234, 134.04729, 134.86560], [8960.00000, 3680.00000, 89313.00000, 2834.00000, 343912.00000, 3136.00000, 937489.00000, 4180.00000]]	4.0	134.0471952	negative	[M-H]-	HCD	ramp-205060	QEHF	Orbitrap	C5H5N5	135.0544952	GFFGJBXGBJISGV-UHFFFAOYSA-N	InChI=1S/C5H5N5/c6-4-3-5(9-1-7-3)10-2-8-4/h1-2H,(H3,6,7,8,9,10)	Nc1nc[nH]c2ncnc1-2
216196	metatlas	istdv7d4b0e45606c941b688c50663710dce0e	ABMBA	[[50.93693, 56.33490, 77.64671, 78.90083, 78.91284, 78.91888, 78.93839, 101.63193, 103.15343, 110.02148, 117.02408, 123.06790, 174.22089, 183.97688, 202.03038, 203.08301, 224.77602, 227.96664], [68452.20312, 66212.17969, 73473.10156, 76644.28906, 318706.40625, 46597924.00000, 82010.42969, 68197.00781, 63107.66797, 79525.91406, 65461.33594, 62304.22656, 72661.72656, 954855.43750, 64707.18750, 177475.26562, 70292.71875, 6487878.50000]]	4.0	227.9666	negative	[M-H]-	HCD	absolute-102040	EXP120B	Orbitrap	C8H8BrNO2	228.973840596	LCMZECCEEOQWLQ-UHFFFAOYSA-N	InChI=1S/C8H8BrNO2/c1-4-2-5(8(11)12)7(10)6(9)3-4/h2-3H,10H2,1H3,(H,11,12)	Cc1cc(Br)c(N)c(C(=O)O)c1
216197	metatlas	istdv7d9d5f77f797444b8acd119eb70d24386	ABMBA	[[66.37357, 68.24240, 70.51434, 73.23862, 76.11681, 78.91288, 78.91891, 78.92505, 78.92654, 78.93877, 82.23162, 97.29940, 138.35910, 153.18608, 172.06372, 203.08241], [35192.81250, 36060.87500, 33001.53125, 38123.24609, 42858.50000, 194144.78125, 30976786.00000, 209199.07812, 78080.07031, 48755.22266, 36910.28516, 38436.95312, 38685.62891, 34273.64453, 50365.60938, 110199.18750]]	4.0	227.9666	negative	[M-H]-	HCD	absolute-205060	EXP120B	Orbitrap	C8H8BrNO2	228.973840596	LCMZECCEEOQWLQ-UHFFFAOYSA-N	InChI=1S/C8H8BrNO2/c1-4-2-5(8(11)12)7(10)6(9)3-4/h2-3H,10H2,1H3,(H,11,12)	Cc1cc(Br)c(N)c(C(=O)O)c1
216198	metatlas	istdv777836b9550ea439994264b0c04a4bfb9	ABMBA	[[54.44872, 58.20813, 59.65763, 61.46074, 66.40463, 78.90137, 78.91295, 78.91891, 78.92501, 78.93779, 98.70604, 183.96640, 183.97675, 187.99323, 203.08255, 204.51361, 221.51900, 227.93750, 227.96674, 228.16083, 228.96996], [57161.75000, 47797.47656, 49287.78906, 55910.88281, 47730.05469, 64754.58984, 211653.82812, 28062886.00000, 224305.68750, 62970.62891, 49103.51562, 111746.67969, 1898768.50000, 56094.41797, 114074.57031, 50946.97266, 58986.53906, 159418.87500, 22239324.00000, 138251.98438, 398011.31250]]	4.0	227.9666	negative	[M-H]-	HCD	normalized-102040	EXP120B	Orbitrap	C8H8BrNO2	228.973840596	LCMZECCEEOQWLQ-UHFFFAOYSA-N	InChI=1S/C8H8BrNO2/c1-4-2-5(8(11)12)7(10)6(9)3-4/h2-3H,10H2,1H3,(H,11,12)	Cc1cc(Br)c(N)c(C(=O)O)c1
216199	metatlas	istdv767a296caef494737868f046e04b87681	ABMBA	[[65.69606, 66.74689, 78.91290, 78.91885, 78.92355, 78.92490, 80.65199, 144.81189, 179.08038, 183.97670, 203.08226, 207.11198, 227.96651], [128305.38281, 123413.87500, 490593.12500, 68953208.00000, 297210.25000, 417448.59375, 141062.17188, 165649.42188, 147877.03125, 1592699.50000, 494700.90625, 129060.27344, 10726662.00000]]	4.0	227.9666	negative	[M-H]-	HCD	normalized-205060	EXP120B	Orbitrap	C8H8BrNO2	228.973840596	LCMZECCEEOQWLQ-UHFFFAOYSA-N	InChI=1S/C8H8BrNO2/c1-4-2-5(8(11)12)7(10)6(9)3-4/h2-3H,10H2,1H3,(H,11,12)	Cc1cc(Br)c(N)c(C(=O)O)c1
EOF

echo "   Created ms2_references.tsv (17 spectra)"

echo ""
echo "Step 4: Creating configuration files..."
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
          - /data/compounds_pos.tsv
      NEG:
        PATHS:
          - /data/compounds_neg.tsv
EOF

# Create atlases configuration
cat > configs/atlases_config.yaml << 'EOF'
ATLASES:
  HILICZ:
    POS:
      ISTD:
        path: /data/compounds_pos.tsv
        name: Dev HILICZ ISTD Atlas Positive
        desc: Development internal standard compounds for positive mode
    NEG:
      ISTD:
        path: /data/compounds_neg.tsv
        name: Dev HILICZ ISTD Atlas Negative
        desc: Development internal standard compounds for negative mode
EOF

# Create analysis configuration
cat > configs/analysis_config.yaml << 'EOF'
WORKFLOWS:
  PATHS:
    owner: dev
    msms_refs_path: /data/ms2_references.tsv
    gdrive_subfolder: 
  RT_ALIGNMENT:
    HILICZ:
      ATLAS:
        uid: dev-istd-hilicz-pos
      PARAMS:
          include_lcmsruns:
            - ISTD
          exclude_lcmsruns:
            - NEG
          use_existing_rt_alignment: false
          ppm_error: 20.0
          extra_time: 2.0
          apply_model_to_min_max: true
          polynomial_degree: 2
          min_observations_per_compound: 1
          min_compounds_for_modeling: 2
          r2_threshold: 0.5
  TARGETED_ANALYSES:
    HILICZ:
      POS:
        ISTD:
          ATLAS:
            uid: dev-istd-hilicz-pos
          PARAMS:
            include_lcmsruns:
              - EXPERIMENTAL
              - ISTD
              - EXCTRL
              - INJBL
            exclude_lcmsruns:
              data_extraction:
                - QC
                - NEG
              gui:
                - INJBL
              id_sheet:
                - INJBL
              chromatograms:
                - INJBL
              id_plots:
                - INJBL
              data_sheets:
                - INJBL
            do_alignment: true
            create_curation_notebooks: true
            upload_to_gdrive: false
            skip_outputs: []
            remove_unided_compounds: false
            remove_flagged_compounds: false
            apply_istd_to_ema: false
            ms1_min_peak_intensity: 1000
            ms1_min_num_points: 1
            ppm_error: 20.0
            extra_time: 1.0
            ms2_min_score: 0.1
            ms2_min_matching_frags: 1
            ms2_frag_mz_tolerance: 0.05
            gui_require_all_evaluated: false
            gui_top_n_hits: 10
            gui_lcmsruns_colors:
              ISTD: blue
              EXCTRL: red
            note_options_overrides:
              ms1_notes: {}
              ms2_notes: {}
              other_notes: {}
      NEG:
        ISTD:
          ATLAS:
            uid: dev-istd-hilicz-neg
          PARAMS:
            include_lcmsruns:
              - EXPERIMENTAL
              - ISTD
              - EXCTRL
              - INJBL
            exclude_lcmsruns:
              data_extraction:
                - QC
                - POS
              gui:
                - INJBL
              id_sheet:
                - INJBL
              chromatograms:
                - INJBL
              id_plots:
                - INJBL
              data_sheets:
                - INJBL
            do_alignment: true
            create_curation_notebooks: true
            upload_to_gdrive: false
            skip_outputs: []
            remove_unided_compounds: false
            remove_flagged_compounds: false
            apply_istd_to_ema: false
            ms1_min_peak_intensity: 1000
            ms1_min_num_points: 1
            ppm_error: 20.0
            extra_time: 1.0
            ms2_min_score: 0.1
            ms2_min_matching_frags: 1
            ms2_frag_mz_tolerance: 0.05
            gui_require_all_evaluated: false
            gui_top_n_hits: 10
            gui_lcmsruns_colors:
              ISTD: blue
              EXCTRL: red
            note_options_overrides:
              ms1_notes: {}
              ms2_notes: {}
              other_notes: {}
EOF

echo "   Created configs/compounds_config.yaml"
echo "   Created configs/atlases_config.yaml"
echo "   Created configs/analysis_config.yaml"

echo ""
echo "Step 5: Creating README..."
echo "--------------------------"

cat > README.md << 'EOF'
# Metatlas2 Standalone Development Environment Data

This package contains a minimal dataset for running metatlas2 in standalone mode without NERSC access.

## Contents

- `parquet/` - 22 parquet MS data files (HILIC positive/negative mode)
  - 5 ISTD files
  - 3 QC files
  - 14 Experimental samples (7 POS + 7 NEG MS2)
  - 3 Injection blank files
  - 3 Extraction control files
  - **Note:** Raw→mzML→parquet conversion already completed

- `compounds_pos.tsv` - 6 positive mode compound definitions
- `compounds_neg.tsv` - 6 negative mode compound definitions
- `ms2_references.tsv` - Minimal MS2 reference spectra
- `configs/` - Configuration files
  - `compounds_config.yaml` - Compound paths configuration
  - `atlases_config.yaml` - Atlas definitions
  - `analysis_config.yaml` - Workflow and analysis parameters
- `dev_environment.yaml` - Environment metadata

## Setup

Extract this archive to `~/.metatlas2-dev/` (or custom location):

```bash
mkdir -p ~/.metatlas2-dev
tar -xzf metatlas2-dev-data.tar.gz -C ~/.metatlas2-dev --strip-components=1
```

Then run:

```bash
metatlas2 --standalone
```

This will launch a Jupyter notebook with all workflow stages ready to execute.

## Size

- Compressed: ~XXX MB
- Uncompressed: ~XXX MB

## Source

Extracted from production project: `20230223_JGI_MC_508469_AlgaHeatChlUWO241_final_EXP120B_HILICZ_USHXG02066`
EOF

echo "Created README.md"

echo ""
echo "Step 6: Creating tarball..."
echo "---------------------------"

cd "${OUTPUT_DIR}"
tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}/"

TARBALL_SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)
EXTRACTED_SIZE=$(du -sh "${PACKAGE_NAME}" | cut -f1)

echo "   Created ${PACKAGE_NAME}.tar.gz"
echo "    Compressed size: ${TARBALL_SIZE}"
echo "    Extracted size: ${EXTRACTED_SIZE}"
