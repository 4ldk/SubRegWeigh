#!/bin/sh

set -eux

cd SubRegWeigh
pip install --user -r requirements.txt

export NUM_EPOCH=5
export NUM_N=500
export NUM_K=10
export METHOD=random
export CONLL03_TRAIN_FILE=data/eng.train
export FIXED_FILE=outputs/conll_fixed_${NUM_K}_${METHOD}.txt
export PREDICT_PATH=outputs/predict_${NUM_K}_${METHOD}.txt
export MODEL_PATH=model/model_${NUM_K}_${METHOD}.pth
export MODEL_NAME=roberta-large

python ./roberta/roberta_train.py --train_path ${CONLL03_TRAIN_FILE} \
                                    --output_path ${MODEL_PATH} \
                                    --num_epoch ${NUM_EPOCH} \
                                    --model_name ${MODEL_NAME}\
                                    --post_sentence_padding \
                                    --add_sep_between_sentences

python ./roberta/roberta_pred.py --test_path ${CONLL03_TRAIN_FILE} \
                                    --model_path ${MODEL_PATH}\
                                    --output_path ${PREDICT_PATH}\
                                    --num_n ${NUM_N}\
                                    --num_k ${NUM_K}\
                                    --method ${METHOD}\
                                    --model_name ${MODEL_NAME}

python ./correct.py --predict_path ${PREDICT_PATH}\
                        --train_path ${CONLL03_TRAIN_FILE} \
                        --output_path ${FIXED_FILE}
