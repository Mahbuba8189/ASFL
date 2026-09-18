#!/bin/bash
# run_table5_mhealth.sh
# Table 5: Ablation Study on MHEALTH with ASFL
# Runs IID and Non-IID (Dirichlet alpha=0.5) sequentially
# Matched to best MHEALTH ASFL configuration (A1)

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"

EPOCHS=50
BATCH_SIZE=16
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
WINDOW_SIZE=15
ALPHA=0.5

cd $PROJECT_DIR


mkdir -p $RESULTS_DIR
mkdir -p $LOGS_DIR

echo "Table 5: Ablation Study - MHEALTH"
echo "Running IID and Non-IID alpha=0.5 sequentially"
echo "Matched to best MHEALTH ASFL (A1)"
echo "  lambda_1=0.5, lambda_2=0.1, lambda_3=0.005"
echo "  BS=16, LR=1e-3, WD=1e-5, Cut=3"
echo "  PGD: eps=0.3, alpha=0.01, K=40"
echo "  Window=15, Warmup=10, Epochs=50"
echo "Start time: $(date)"
echo ""

# IID
echo "Running MHEALTH IID ablation..."
COMMON_ARGS="--run_ablation --run_ablation_dataset mhealth"
COMMON_ARGS="$COMMON_ARGS --epochs $EPOCHS --batch_size $BATCH_SIZE"
COMMON_ARGS="$COMMON_ARGS --learning_rate $LR --weight_decay $WEIGHT_DECAY"
COMMON_ARGS="$COMMON_ARGS --cut_layer $CUT_LAYER"
COMMON_ARGS="$COMMON_ARGS --lambda_1 $LAMBDA_1 --lambda_2 $LAMBDA_2 --lambda_3 $LAMBDA_3"
COMMON_ARGS="$COMMON_ARGS --pgd_epsilon $PGD_EPSILON --pgd_steps $PGD_STEPS --pgd_alpha $PGD_ALPHA"
COMMON_ARGS="$COMMON_ARGS --warmup_epochs $WARMUP_EPOCHS"
COMMON_ARGS="$COMMON_ARGS --use_class_weights"
COMMON_ARGS="$COMMON_ARGS --window_size $WINDOW_SIZE"
COMMON_ARGS="$COMMON_ARGS --output_dir ${RESULTS_DIR}/ablation/mhealth --device cuda --seed $SEED"

python -u main.py $COMMON_ARGS > "${LOGS_DIR}/ablation_mhealth_all.log" 2>&1

if [ $? -eq 0 ]; then
    echo "  MHEALTH IID ablation completed successfully"
else
    echo "  MHEALTH IID ablation failed"
    tail -30 "${LOGS_DIR}/ablation_mhealth_all.log"
    exit 1
fi

echo ""

# Non-IID
echo "Running MHEALTH Non-IID alpha=0.5 ablation..."
COMMON_ARGS="--run_ablation --run_ablation_dataset mhealth"
COMMON_ARGS="$COMMON_ARGS --alpha $ALPHA"
COMMON_ARGS="$COMMON_ARGS --epochs $EPOCHS --batch_size $BATCH_SIZE"
COMMON_ARGS="$COMMON_ARGS --learning_rate $LR --weight_decay $WEIGHT_DECAY"
COMMON_ARGS="$COMMON_ARGS --cut_layer $CUT_LAYER"
COMMON_ARGS="$COMMON_ARGS --lambda_1 $LAMBDA_1 --lambda_2 $LAMBDA_2 --lambda_3 $LAMBDA_3"
COMMON_ARGS="$COMMON_ARGS --pgd_epsilon $PGD_EPSILON --pgd_steps $PGD_STEPS --pgd_alpha $PGD_ALPHA"
COMMON_ARGS="$COMMON_ARGS --warmup_epochs $WARMUP_EPOCHS"
COMMON_ARGS="$COMMON_ARGS --use_class_weights"
COMMON_ARGS="$COMMON_ARGS --window_size $WINDOW_SIZE"
COMMON_ARGS="$COMMON_ARGS --output_dir ${RESULTS_DIR}/ablation/mhealth_noniid --device cuda --seed $SEED"

python -u main.py $COMMON_ARGS > "${LOGS_DIR}/ablation_mhealth_noniid_all.log" 2>&1

if [ $? -eq 0 ]; then
    echo "  MHEALTH Non-IID ablation completed successfully"
else
    echo "  MHEALTH Non-IID ablation failed"
    tail -30 "${LOGS_DIR}/ablation_mhealth_noniid_all.log"
    exit 1
fi

echo ""
echo "MHEALTH Ablation Study completed at: $(date)"
echo ""
echo "Logs saved to:"
echo "  - ${LOGS_DIR}/ablation_mhealth_all.log"
echo "  - ${LOGS_DIR}/ablation_mhealth_noniid_all.log"
echo ""
echo "Output directories:"
echo "  - ${RESULTS_DIR}/ablation/mhealth/no_dual/"
echo "  - ${RESULTS_DIR}/ablation/mhealth/no_privacy/"
echo "  - ${RESULTS_DIR}/ablation/mhealth/no_evasion/"
echo "  - ${RESULTS_DIR}/ablation/mhealth_noniid/no_dual/"
echo "  - ${RESULTS_DIR}/ablation/mhealth_noniid/no_privacy/"
echo "  - ${RESULTS_DIR}/ablation/mhealth_noniid/no_evasion/"