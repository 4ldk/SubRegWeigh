#!/bin/sh

set -eux

cd SubRegWeigh
pip install --user -r requirements.txt

export NUM_EPOCH=5
export NUM_N=500
export NUM_K=10
export METHOD=random
export TRAIN_FILE=data/MedTxt_train.txt
export FIXED_FILE=outputs/medtxt_fixed_${NUM_K}_${METHOD}.txt
export PREDICT_PATH=outputs/predict_medtxt_${NUM_K}_${METHOD}.txt
export MODEL_PATH=model/model_medtxt_${NUM_K}_${METHOD}.pth
export MODEL_NAME=roberta-large

python ./roberta/roberta_train.py --train_path ${TRAIN_FILE} \
                                    --output_path ${MODEL_PATH} \
                                    --num_epoch ${NUM_EPOCH} \
                                    --model_name ${MODEL_NAME}\
                                    --post_sentence_padding \
                                    --add_sep_between_sentences

python ./roberta/roberta_pred.py --test_path ${TRAIN_FILE} \
                                    --model_path ${MODEL_PATH}\
                                    --output_path ${PREDICT_PATH}\
                                    --num_n ${NUM_N}\
                                    --num_k ${NUM_K}\
                                    --method ${METHOD}\
                                    --model_name ${MODEL_NAME}

python ./correct.py --predict_path ${PREDICT_PATH}\
                        --train_path ${TRAIN_FILE} \
                        --output_path ${FIXED_FILE}

export TEST_NUM_EPOCH=20
export TEST_FILE=data/MedTxt_test.txt
export TEST_PREDICT_PATH=outputs/predict_medtxt_test_${NUM_K}_${METHOD}.txt
export TEST_MODEL_PATH=model/model_medtxt_test_${NUM_K}_${METHOD}.pth
python ./roberta/roberta_train.py --train_path ${FIXED_FILE} \
                                    --output_path ${TEST_MODEL_PATH} \
                                    --num_epoch ${TEST_NUM_EPOCH} \
                                    --model_name ${MODEL_NAME}\
                                    --post_sentence_padding \
                                    --add_sep_between_sentences

python ./roberta/roberta_pred.py --test_path ${TEST_FILE} \
                                    --model_path ${TEST_MODEL_PATH}\
                                    --output_path ${TEST_PREDICT_PATH}\
                                    --num_n 1\
                                    --num_k 1\
                                    --method "random"\
                                    --model_name ${MODEL_NAME}