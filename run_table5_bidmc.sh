#!/bin/bash
# run_table5_bidmc.sh
# Table 5: Ablation Study on BIDMC with ASFL
# Runs IID (N5) then Non-IID (alpha=0.5, N6) sequentially
# Each runs 3 ablations: no_dual, no_privacy, no_evasion
# WITH OVERSAMPLING (target=800) + CLASS WEIGHTS

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

cd $PROJECT_DIR


mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

echo "Table 5: Ablation Study - BIDMC"
echo "Running IID (N5) and Non-IID alpha=0.5 (N6) sequentially"
echo "Start time: $(date)"
echo ""


# BIDMC IID - Matched to N5


EPOCHS=50
BATCH_SIZE=32
LR=1e-3
WEIGHT_DECAY=1e-5
CUT_LAYER=3
SEED=99

LAMBDA_1=0.5
LAMBDA_2=0.1
LAMBDA_3=0.005

PGD_EPSILON=0.3
PGD_STEPS=40
PGD_ALPHA=0.01

WARMUP_EPOCHS=10
WINDOW_SIZE=8
OVERSAMPLE_TARGET=800

echo "Running BIDMC IID ablation (N5)..."
echo "  lambda_1=0.5, lambda_2=0.1, lambda_3=0.005"
echo "  BS=32, LR=1e-3, WD=1e-5, Cut=3"
echo "  PGD: eps=0.3, alpha=0.01, K=40"
echo ""

for ABLATION in no_dual no_privacy no_evasion; do
    echo "--- IID ablation: $ABLATION ---"

    IID_ARGS="--run_ablation --run_ablation_dataset bidmc"
    IID_ARGS="$IID_ARGS --ablation $ABLATION"
    IID_ARGS="$IID_ARGS --epochs $EPOCHS --batch_size $BATCH_SIZE"
    IID_ARGS="$IID_ARGS --learning_rate $LR --weight_decay $WEIGHT_DECAY"
    IID_ARGS="$IID_ARGS --cut_layer $CUT_LAYER"
    IID_ARGS="$IID_ARGS --lambda_1 $LAMBDA_1 --lambda_2 $LAMBDA_2 --lambda_3 $LAMBDA_3"
    IID_ARGS="$IID_ARGS --pgd_epsilon $PGD_EPSILON --pgd_steps $PGD_STEPS --pgd_alpha $PGD_ALPHA"
    IID_ARGS="$IID_ARGS --warmup_epochs $WARMUP_EPOCHS"
    IID_ARGS="$IID_ARGS --oversample_target $OVERSAMPLE_TARGET"
    IID_ARGS="$IID_ARGS --use_class_weights"
    IID_ARGS="$IID_ARGS --window_size $WINDOW_SIZE"
    IID_ARGS="$IID_ARGS --output_dir ${RESULTS_DIR}/ablation/bidmc_iid --device cuda --seed $SEED"

    python -u main.py $IID_ARGS > "${LOGS_DIR}/ablation_bidmc_iid_${ABLATION}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "  BIDMC IID [$ABLATION] completed successfully"
    else
        echo "  BIDMC IID [$ABLATION] failed"
        tail -30 "${LOGS_DIR}/ablation_bidmc_iid_${ABLATION}.log"
        exit 1
    fi
done

echo ""

# BIDMC Non-IID (alpha=0.5) - N6


EPOCHS=50
BATCH_SIZE=32
LR=5e-4
WEIGHT_DECAY=1e-5
CUT_LAYER=4
SEED=99

LAMBDA_1=0.5
LAMBDA_2=0.1
LAMBDA_3=0.01

PGD_EPSILON=0.3
PGD_STEPS=5
PGD_ALPHA=0.01

WARMUP_EPOCHS=10
WINDOW_SIZE=8
OVERSAMPLE_TARGET=800
ALPHA=0.5

echo "Running BIDMC Non-IID alpha=0.5 ablation (N6)..."
echo "  lambda_1=0.5, lambda_2=0.1, lambda_3=0.01"
echo "  BS=32, LR=5e-4, WD=1e-5, Cut=4"
echo "  PGD: eps=0.3, alpha=0.01, K=5"
echo ""

for ABLATION in no_dual no_privacy no_evasion; do
    echo "--- Non-IID ablation: $ABLATION ---"

    NONIID_ARGS="--run_ablation --run_ablation_dataset bidmc"
    NONIID_ARGS="$NONIID_ARGS --ablation $ABLATION"
    NONIID_ARGS="$NONIID_ARGS --alpha $ALPHA"
    NONIID_ARGS="$NONIID_ARGS --epochs $EPOCHS --batch_size $BATCH_SIZE"
    NONIID_ARGS="$NONIID_ARGS --learning_rate $LR --weight_decay $WEIGHT_DECAY"
    NONIID_ARGS="$NONIID_ARGS --cut_layer $CUT_LAYER"
    NONIID_ARGS="$NONIID_ARGS --lambda_1 $LAMBDA_1 --lambda_2 $LAMBDA_2 --lambda_3 $LAMBDA_3"
    NONIID_ARGS="$NONIID_ARGS --pgd_epsilon $PGD_EPSILON --pgd_steps $PGD_STEPS --pgd_alpha $PGD_ALPHA"
    NONIID_ARGS="$NONIID_ARGS --warmup_epochs $WARMUP_EPOCHS"
    NONIID_ARGS="$NONIID_ARGS --oversample_target $OVERSAMPLE_TARGET"
    NONIID_ARGS="$NONIID_ARGS --use_class_weights"
    NONIID_ARGS="$NONIID_ARGS --window_size $WINDOW_SIZE"
    NONIID_ARGS="$NONIID_ARGS --output_dir ${RESULTS_DIR}/ablation/bidmc_noniid --device cuda --seed $SEED"

    python -u main.py $NONIID_ARGS > "${LOGS_DIR}/ablation_bidmc_noniid_${ABLATION}.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "  BIDMC Non-IID [$ABLATION] completed successfully"
    else
        echo "  BIDMC Non-IID [$ABLATION] failed"
        tail -30 "${LOGS_DIR}/ablation_bidmc_noniid_${ABLATION}.log"
        exit 1
    fi
done

echo ""
echo "BIDMC Ablation Study completed at: $(date)"
echo ""
echo "Logs saved to:"
echo "  - ${LOGS_DIR}/ablation_bidmc_iid_no_dual.log"
echo "  - ${LOGS_DIR}/ablation_bidmc_iid_no_privacy.log"
echo "  - ${LOGS_DIR}/ablation_bidmc_iid_no_evasion.log"
echo "  - ${LOGS_DIR}/ablation_bidmc_noniid_no_dual.log"
echo "  - ${LOGS_DIR}/ablation_bidmc_noniid_no_privacy.log"
echo "  - ${LOGS_DIR}/ablation_bidmc_noniid_no_evasion.log"
echo ""
echo "Output directories:"
echo "  - ${RESULTS_DIR}/ablation/bidmc_iid/ablation/no_dual/"
echo "  - ${RESULTS_DIR}/ablation/bidmc_iid/ablation/no_privacy/"
echo "  - ${RESULTS_DIR}/ablation/bidmc_iid/ablation/no_evasion/"
echo "  - ${RESULTS_DIR}/ablation/bidmc_noniid/ablation/no_dual/"
echo "  - ${RESULTS_DIR}/ablation/bidmc_noniid/ablation/no_privacy/"
echo "  - ${RESULTS_DIR}/ablation/bidmc_noniid/ablation/no_evasion/"