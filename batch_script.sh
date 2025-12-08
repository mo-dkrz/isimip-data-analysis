#!/bin/bash
#SBATCH --job-name=drought_indices
#SBATCH --qos=short
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=drought_%j.log
#SBATCH --error=drought_-%j.err

# =============================================================================
# ISIMIP Drought Indices - HPC Production Script
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - Edit these lists for your setup
# =============================================================================

# Models to process (lowercase:UPPERCASE pairs)
MODELS=(
    "gfdl-esm4:GFDL-ESM4"
    "ukesm1-0-ll:UKESM1-0-LL"
    "mpi-esm1-2-hr:MPI-ESM1-2-HR"
    "ipsl-cm6a-lr:IPSL-CM6A-LR"
    "mri-esm2-0:MRI-ESM2-0"
)

# Scenarios to process
SCENARIOS=(
    "ssp126"
    "ssp370"
    "ssp585"
)

# Calibration period
CAL_START=1901
CAL_END=2014

# Scales
SPI_SCALES="2 3 6"
SPEI_SCALES="2 3 6"
MCWD_SCALES="2 6 12"
MCWD_RESET_MONTH=10

# Paths
VENV_DIR="${HOME}/venv"
# DATA_BASE="/p/projects/isimip/isimip/ISIMIP3b/InputData/climate/atmosphere/bias-adjusted/global/daily"
DATA_BASE="/p/projects/ou/labs/ai/mariafe/data"
OUTPUT_BASE="${HOME}/drought_outputs"

# =============================================================================
# END OF CONFIGURATION
# =============================================================================

CALIBRATION="${CAL_START}-${CAL_END}"

# Environment setup
module load anaconda 2>/dev/null || true
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
fi

if ! command -v isimip-drought &> /dev/null; then
    echo "ERROR: isimip-drought not found"
    exit 1
fi

# Count total combinations
TOTAL=$((${#MODELS[@]} * ${#SCENARIOS[@]}))
CURRENT=0

echo "=============================================="
echo "Processing ${#MODELS[@]} models x ${#SCENARIOS[@]} scenarios = ${TOTAL} combinations"
echo "Calibration: ${CALIBRATION}"
echo "=============================================="

# -----------------------------------------------------------------------------
# Main loop over models and scenarios
# -----------------------------------------------------------------------------

for MODEL_PAIR in "${MODELS[@]}"; do
    # Split "lowercase:UPPERCASE"
    MODEL="${MODEL_PAIR%%:*}"
    MODEL_UPPER="${MODEL_PAIR##*:}"

    for SCENARIO in "${SCENARIOS[@]}"; do
        CURRENT=$((CURRENT + 1))
        
        echo ""
        echo "=============================================="
        echo "[${CURRENT}/${TOTAL}] ${MODEL_UPPER} / ${SCENARIO}"
        echo "=============================================="

        # Paths
        HIST_DIR="${DATA_BASE}/historical/${MODEL_UPPER}"
        FUT_DIR="${DATA_BASE}/${SCENARIO}/${MODEL_UPPER}"
        OUTPUT_DIR="${OUTPUT_BASE}/${MODEL}_${SCENARIO}"
        mkdir -p "${OUTPUT_DIR}"

        # File patterns
        PR="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_pr*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_pr*.nc"
        TAS="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_tas_*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_tas_*.nc"
        TASMIN="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_tasmin*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_tasmin*.nc"
        TASMAX="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_tasmax*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_tasmax*.nc"
        HURS="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_hurs*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_hurs*.nc"
        RSDS="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_rsds*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_rsds*.nc"
        SFCWIND="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_sfcwind*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_sfcwind*.nc"
        PS="${HIST_DIR}/${MODEL}_r1i1p1f1_w5e5_historical_ps*.nc,${FUT_DIR}/${MODEL}_r1i1p1f1_w5e5_${SCENARIO}_ps*.nc"

        # Output files
        PET_FILE="${OUTPUT_DIR}/pet_penman_${MODEL}_${SCENARIO}.nc"
        SPI_FILE="${OUTPUT_DIR}/spi_${MODEL}_${SCENARIO}.nc"
        SPEI_FILE="${OUTPUT_DIR}/spei_${MODEL}_${SCENARIO}.nc"
        MCWD_FILE="${OUTPUT_DIR}/mcwd_${MODEL}_${SCENARIO}.nc"

        # Step 1: PET
        echo "  PET..."
        if [ -f "${PET_FILE}" ]; then
            echo "    + exists, skipping"
        else
            isimip-drought pet \
                --method penman \
                --tas "${TAS}" \
                --tasmin "${TASMIN}" \
                --tasmax "${TASMAX}" \
                --hurs "${HURS}" \
                --rsds "${RSDS}" \
                --sfcwind "${SFCWIND}" \
                --ps "${PS}" \
                --out "${PET_FILE}"
        fi

        # Step 2: SPI
        echo "  SPI..."
        if [ -f "${SPI_FILE}" ]; then
            echo "    + exists, skipping"
        else
            SCALE_ARGS=""
            for s in ${SPI_SCALES}; do SCALE_ARGS="${SCALE_ARGS} -s ${s}"; done
            isimip-drought spi \
                --precip "${PR}" \
                ${SCALE_ARGS} \
                --calibration "${CALIBRATION}" \
                --out "${SPI_FILE}"
        fi

        # Step 3: SPEI
        echo "  SPEI..."
        if [ -f "${SPEI_FILE}" ]; then
            echo "    + exists, skipping"
        else
            SCALE_ARGS=""
            for s in ${SPEI_SCALES}; do SCALE_ARGS="${SCALE_ARGS} -s ${s}"; done
            isimip-drought spei \
                --precip "${PR}" \
                --pet "${PET_FILE}" \
                ${SCALE_ARGS} \
                --calibration "${CALIBRATION}" \
                --out "${SPEI_FILE}"
        fi

        # Step 4: MCWD
        echo "  MCWD..."
        if [ -f "${MCWD_FILE}" ]; then
            echo "    + exists, skipping"
        else
            SCALE_ARGS=""
            for s in ${MCWD_SCALES}; do SCALE_ARGS="${SCALE_ARGS} -s ${s}"; done
            isimip-drought mcwd \
                --precip "${PR}" \
                --pet "${PET_FILE}" \
                ${SCALE_ARGS} \
                --reset-month ${MCWD_RESET_MONTH} \
                --out "${MCWD_FILE}"
        fi

        echo "  + Done: ${MODEL}_${SCENARIO}"

    done  # scenarios
done  # models

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "All complete! Output structure:"
echo "=============================================="
ls -la ${OUTPUT_BASE}/*/