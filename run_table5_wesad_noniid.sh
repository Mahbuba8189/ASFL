#!/bin/bash
# run_table5_wesad_noniid.sh
# Runs Table 5: Ablation Study on WESAD Non-IID with ASFL
# Runs 3 ablations: no_dual, no_privacy, no_evasion
# Using ASFL N5 configuration from tuning sheet

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

# Experiment parameters (WESAD Non-IID ASFL N5)
EPOCHS=50
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=1e-5
CUT_LAYER=3
SEED=99

# ASFL loss weights (N5)
LAMBDA_1=0.5
LAMBDA_2=0.1
LAMBDA_3=0.005

# PGD attack parameters 
PGD_EPSILON=0.3
PGD_STEPS=40
PGD_ALPHA=0.01

# Training phases
WARMUP_EPOCHS=10

# Non-IID Dirichlet concentration
ALPHA=0.5

cd $PROJECT_DIR


mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

echo "========================================"
echo "Table 5: Ablation Study - WESAD Non-IID"
echo "WITH CLASS WEIGHTS (oversampling: BIDMC only)"
echo "Using ASFL N5 configuration"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Build common arguments
COMMON_ARGS="--run_ablation --run_ablation_dataset wesad"
COMMON_ARGS="$COMMON_ARGS --alpha $ALPHA"
COMMON_ARGS="$COMMON_ARGS --epochs $EPOCHS --batch_size $BATCH_SIZE"
COMMON_ARGS="$COMMON_ARGS --learning_rate $LR --weight_decay $WEIGHT_DECAY"
COMMON_ARGS="$COMMON_ARGS --cut_layer $CUT_LAYER"
COMMON_ARGS="$COMMON_ARGS --lambda_1 $LAMBDA_1 --lambda_2 $LAMBDA_2 --lambda_3 $LAMBDA_3"
COMMON_ARGS="$COMMON_ARGS --pgd_epsilon $PGD_EPSILON --pgd_steps $PGD_STEPS --pgd_alpha $PGD_ALPHA"
COMMON_ARGS="$COMMON_ARGS --warmup_epochs $WARMUP_EPOCHS"
COMMON_ARGS="$COMMON_ARGS --use_class_weights"
COMMON_ARGS="$COMMON_ARGS --output_dir ${RESULTS_DIR}/ablation/wesad_noniid --device cuda --seed $SEED"

echo "Running all 3 ablations for WESAD Non-IID..."
echo "  Common args: $COMMON_ARGS"
echo "  Logging to: ${LOGS_DIR}/ablation_wesad_noniid_all.log"

python -u main.py $COMMON_ARGS > "${LOGS_DIR}/ablation_wesad_noniid_all.log" 2>&1

if [ $? -eq 0 ]; then
    echo "  All ablations completed successfully"
else
    echo "  Ablations failed"
    tail -30 "${LOGS_DIR}/ablation_wesad_noniid_all.log"
    exit 1
fi

echo ""
echo "========================================"
echo "WESAD Non-IID Ablation Study completed at: $(date)"
echo "========================================"
echo ""
echo "Logs saved to:"
echo "  - ${LOGS_DIR}/ablation_wesad_noniid_all.log"
echo ""
echo "Output directories:"
echo "  - ${RESULTS_DIR}/ablation/wesad_noniid/no_dual/"
echo "  - ${RESULTS_DIR}/ablation/wesad_noniid/no_privacy/"
echo "  - ${RESULTS_DIR}/ablation/wesad_noniid/no_evasion/"