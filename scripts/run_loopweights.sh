#!/bin/sh

set -eux

cd SubRegWeigh
pip install --user -r requirements.txt

export LOOP=3
export NUM_EPOCH=5
export NUM_N=500
export NUM_K=10
export METHOD=random
export CONLL03_TRAIN_FILE=data/eng.train
export MODEL_NAME=roberta-large

CURRENT_TRAIN_FILE=${CONLL03_TRAIN_FILE}
for NUM_LOOP in $(seq 1 1 ${loop}); do
    export PREDICT_PATH=outputs/predict_${NUM_K}_${METHOD}_loop${NUM_LOOP}.txt
    export MODEL_PATH=model/model_${NUM_K}_${METHOD}_loop${NUM_LOOP}.pth
    export FIXED_FILE=outputs/conll_fixed_${NUM_K}_${METHOD}_loop${NUM_LOOP}.txt

    python ./roberta/roberta_train.py --train_path ${CURRENT_TRAIN_FILE} \
                                        --output_path ${MODEL_PATH} \
                                        --num_epoch ${NUM_EPOCH} \
                                        --model_name ${MODEL_NAME}\
                                        --post_sentence_padding \
                                        --add_sep_between_sentences

    python ./roberta/roberta_pred.py --test_path ${CURRENT_TRAIN_FILE} \
                                        --model_path ${MODEL_PATH}\
                                        --output_path ${PREDICT_PATH}\
                                        --num_n ${NUM_N}\
                                        --num_k ${NUM_K}\
                                        --method ${METHOD}\
                                        --model_name ${MODEL_NAME}

    python ./correct.py --predict_path ${PREDICT_PATH}\
                            --train_path ${CONLL03_TRAIN_FILE} \
                            --output_path ${FIXED_FILE}

    CURRENT_TRAIN_FILE=${FIXED_FILE}

done
