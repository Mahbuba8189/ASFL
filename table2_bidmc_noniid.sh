#!/bin/bash

# TABLE 2: BIDMC Non-IID Hyperparameter Search (α=0.5)
# 30 Configurations (10 Plain SFL + 10 ResSFL + 10 ASFL)




# Fixed Parameters
EPOCHS=50
SEED=99
DEVICE="cuda"
USE_CLASS_WEIGHTS="--use_class_weights"
OVERSAMPLE_TARGET="--oversample_target 800"
PGD_EPS=0.3
PGD_ALPHA=0.01
PGD_STEPS=40
WARMUP_EPOCHS=10
WINDOW_SIZE=8
DATASET="bidmc"
DISTRIBUTION="noniid"
ALPHA=0.5

# ResSFL Fixed Parameters
RESSFL_TEACHER_LAMBDA=0.3
RESSFL_USE_PRETRAIN="--ressfl_use_pretrain"
RESSFL_PRETRAIN_EPOCHS=50
RESSFL_WARMUP_EPOCHS=10
RESSFL_FINETUNE_LR=0.005
RESSFL_UPDATE_FREQUENCY=5

# Directories
RESULTS_DIR="results/table2_bidmc_noniid"
LOGS_DIR="logs/table2_bidmc_noniid"
mkdir -p ${RESULTS_DIR}
mkdir -p ${LOGS_DIR}

echo "TABLE 2: BIDMC Non-IID Hyperparameter Search (α=0.5)"
echo "Start Time: $(date)"

# Method 1: Plain SFL (10 Configurations)

echo ""
echo "BIDMC-Non_IID Plain SFL (10 configs)"

# Batch Size 16
echo "Running P1..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P1 > ${LOGS_DIR}/P1.log 2>&1

echo "Running P2..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P2 > ${LOGS_DIR}/P2.log 2>&1

echo "Running P3..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P3 > ${LOGS_DIR}/P3.log 2>&1

echo "Running P4..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 2 \
    --output_dir ${RESULTS_DIR}/P4 > ${LOGS_DIR}/P4.log 2>&1

# Batch Size 32
echo "Running P5..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P5 > ${LOGS_DIR}/P5.log 2>&1

echo "Running P6..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 4 \
    --output_dir ${RESULTS_DIR}/P6 > ${LOGS_DIR}/P6.log 2>&1

echo "Running P7..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P7 > ${LOGS_DIR}/P7.log 2>&1

echo "Running P8..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P8 > ${LOGS_DIR}/P8.log 2>&1

# Batch Size 64
echo "Running P9..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --output_dir ${RESULTS_DIR}/P9 > ${LOGS_DIR}/P9.log 2>&1

echo "Running P10..."
python -u main.py --dataset ${DATASET} --method plain --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 2 \
    --output_dir ${RESULTS_DIR}/P10 > ${LOGS_DIR}/P10.log 2>&1

echo "Plain SFL Completed: $(date)"

# Method 2: ResSFL (10 Configurations)

echo ""
echo "BIDMC Non-IID ResSFL (10 configs)"

# Batch Size 16
echo "Running R1..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R1 > ${LOGS_DIR}/R1.log 2>&1

echo "Running R2..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R2 > ${LOGS_DIR}/R2.log 2>&1

echo "Running R3..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.01 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R3 > ${LOGS_DIR}/R3.log 2>&1

echo "Running R4..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 2 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R4 > ${LOGS_DIR}/R4.log 2>&1

# Batch Size 32
echo "Running R5..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R5 > ${LOGS_DIR}/R5.log 2>&1

echo "Running R6..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 4 \
    --ressfl_lambda_rec 0.01 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R6 > ${LOGS_DIR}/R6.log 2>&1

echo "Running R7..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R7 > ${LOGS_DIR}/R7.log 2>&1

echo "Running R8..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R8 > ${LOGS_DIR}/R8.log 2>&1

# Batch Size 64
echo "Running R9..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --ressfl_lambda_rec 0.001 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R9 > ${LOGS_DIR}/R9.log 2>&1

echo "Running R10..."
python -u main.py --dataset ${DATASET} --method ressfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 2 \
    --ressfl_lambda_rec 0.01 --ressfl_teacher_lambda ${RESSFL_TEACHER_LAMBDA} \
    ${RESSFL_USE_PRETRAIN} --ressfl_pretrain_epochs ${RESSFL_PRETRAIN_EPOCHS} \
    --ressfl_warmup_epochs ${RESSFL_WARMUP_EPOCHS} \
    --ressfl_finetune_lr ${RESSFL_FINETUNE_LR} \
    --ressfl_update_frequency ${RESSFL_UPDATE_FREQUENCY} \
    --output_dir ${RESULTS_DIR}/R10 > ${LOGS_DIR}/R10.log 2>&1

echo "ResSFL Completed: $(date)"

# Method 3: ASFL (10 Configurations)

echo ""
echo "BIDMC Non-IID ASFL (10 configs)"

# Batch Size 16
echo "Running N1..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N1 > ${LOGS_DIR}/N1.log 2>&1

echo "Running N2..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N2 > ${LOGS_DIR}/N2.log 2>&1

echo "Running N3..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 1.0 --lambda_2 0.5 --lambda_3 0.01 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N3 > ${LOGS_DIR}/N3.log 2>&1

echo "Running N4..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 16 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 2 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N4 > ${LOGS_DIR}/N4.log 2>&1

# Batch Size 32
echo "Running N5..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N5 > ${LOGS_DIR}/N5.log 2>&1

echo "Running N6..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 4 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.01 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N6 > ${LOGS_DIR}/N6.log 2>&1

echo "Running N7..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 20 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-4 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N7 > ${LOGS_DIR}/N7.log 2>&1

echo "Running N8..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 32 --learning_rate 1e-3 --weight_decay 1e-4 --cut_layer 3 \
    --lambda_1 1.0 --lambda_2 0.5 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N8 > ${LOGS_DIR}/N8.log 2>&1

# Batch Size 64
echo "Running N9..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 40 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 1e-3 --weight_decay 1e-5 --cut_layer 3 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.005 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N9 > ${LOGS_DIR}/N9.log 2>&1

echo "Running N10..."
python -u main.py --dataset ${DATASET} --method asfl --distribution ${DISTRIBUTION} --alpha ${ALPHA} \
    --window_size ${WINDOW_SIZE} --epochs ${EPOCHS} --seed ${SEED} --device ${DEVICE} \
    ${USE_CLASS_WEIGHTS} ${OVERSAMPLE_TARGET} \
    --pgd_epsilon ${PGD_EPS} --pgd_steps 5 --pgd_alpha ${PGD_ALPHA} \
    --batch_size 64 --learning_rate 5e-4 --weight_decay 1e-5 --cut_layer 2 \
    --lambda_1 0.5 --lambda_2 0.1 --lambda_3 0.01 --warmup_epochs ${WARMUP_EPOCHS} \
    --output_dir ${RESULTS_DIR}/N10 > ${LOGS_DIR}/N10.log 2>&1

echo "ASFL Completed: $(date)"

# Summary

echo ""
echo "TABLE 2: BIDMC Non-IID HYPERPARAMETER SEARCH COMPLETE!"
echo "End Time: $(date)"
echo "Results saved to: ${RESULTS_DIR}"
echo "Logs saved to: ${LOGS_DIR}"
echo ""
echo "30 Configurations Completed:"
echo "  - Plain SFL: 10 (P1-P10)"
echo "  - ResSFL: 10 (R1-R10)"
echo "  - ASFL: 10 (N1-N10)"