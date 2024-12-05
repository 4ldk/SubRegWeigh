#!/bin/sh

set -eux

cd AnnotationCorrection
pip install --user -r requirements.txt

export NUM_EPOCH=5
export NUM_N=500
export NUM_K=10
export METHOD="k-means"
export TRAIN_FILE=data/mrpc_train.txt
export VALID_FILE=data/mrpc_valid.txt
export TEST_FILE=data/mrpc_test.txt
export FIXED_FILE=outputs/mrpc_fixed_${NUM_K}_${METHOD}.txt
export PREDICT_PATH=outputs/predict_mrpc_${NUM_K}_${METHOD}.txt
export MODEL_PATH=model/mrpc_model.pth
export MODEL_NAME=roberta-base

python ./classification/roberta_c_train.py --train_path ${TRAIN_FILE} \
                                    --output_path ${MODEL_PATH}\
                                    --num_epoch ${NUM_EPOCH}\
                                    --model_name ${MODEL_NAME}

python ./classification/roberta_c_pred.py --test_path ${TRAIN_FILE} \
                                    --model_path ${MODEL_PATH}\
                                    --output_path ${PREDICT_PATH}\
                                    --num_n ${NUM_N}\
                                    --num_k ${NUM_K}\
                                    --alpha 0.05\
                                    --method ${METHOD}\
                                    --model_name ${MODEL_NAME}


python ./classification/correct.py --predict_path ${PREDICT_PATH}\
                        --output_path ${FIXED_FILE}

## Eval
export EVAL_EPOCH=20
export MID_EPOCH=5
export EVAL_MODEL_PATH=model/mrpc_model_${NUM_K}_${METHOD}_fixed.pth
export EVAL_VALID_PATH=outputs/eval_valid_mrpc_${NUM_K}_${METHOD}.txt
export EVAL_PREDICT_PATH=outputs/eval_mrpc_${NUM_K}_${METHOD}.txt

python ./classification/roberta_c_train.py --train_path ${FIXED_FILE} \
                                    --valid_path ${VALID_FILE}\
                                    --test_path ${TEST_FILE}\
                                    --output_path ${EVAL_MODEL_PATH}\
                                    --num_epoch ${EVAL_EPOCH}\
                                    --model_name ${MODEL_NAME}

python ./classification/roberta_c_pred.py --test_path ${VALID_FILE} \
                                    --model_path ${EVAL_MODEL_PATH}\
                                    --output_path ${EVAL_VALID_PATH}\
                                    --num_n 1\
                                    --num_k 1\
                                    --alpha 0\
                                    --method random\
                                    --model_name ${MODEL_NAME}

python ./classification/get_score.py --eval_path ${EVAL_VALID_PATH}