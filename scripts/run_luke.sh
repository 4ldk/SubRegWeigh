#!/bin/sh

set -eux

cd SubRegWeigh
pip install --user -r requirements.txt

export NUM_EPOCH=5
export NUM_N=500
export NUM_K=10
export METHOD=random
export CONLL03_TRAIN_FILE=data/eng.train # put conll original train data to data/
export CONLL03_VALID_FILE=data/eng.testa # put conll original valid data to data/
export FIXED_FILE=outputs/conll_fixed_${NUM_K}_${METHOD}_luke.txt
export PREDICT_PATH=outputs/predict_${NUM_K}_${METHOD}_luke.txt
export MODEL_PATH=model/luke-large
export MODEL_NAME=studio-ousia/luke-large-finetuned-conll-2003

git clone https://github.com/studio-ousia/luke.git
cd luke

python examples/ner/convert_io_to_bio_format.py ../${CONLL03_TRAIN_FILE}
python examples/ner/convert_io_to_bio_format.py ../${CONLL03_VALID_FILE}

export TRAIN_DATA_PATH=../${CONLL03_TRAIN_FILE}.bio;\
export VALIDATION_DATA_PATH=../${CONLL03_VALID_FILE}.bio;\
export TRANSFORMERS_MODEL_NAME="studio-ousia/luke-large";\
allennlp train examples/ner/configs/transformers_luke_with_entity_aware_attention.jsonnet -s results/ner/luke-large --include-package examples -o '{"trainer.cuda_device": 0, "trainer.use_amp": true}'

python examples/ner/convert_allennlp_to_huggingface_model.py results/ner/luke-large ../${MODEL_PATH}

cd ../

python ./luke/luke_pred.py --test_path ${CONLL03_TRAIN_FILE} \
                                    --model_path ${MODEL_PATH}\
                                    --output_path ${PREDICT_PATH}\
                                    --num_n ${NUM_N}\
                                    --num_k ${NUM_K}\
                                    --method ${METHOD}\
                                    --model_name ${MODEL_NAME}


python ./correct.py --predict_path ${PREDICT_PATH}\
                        --train_path ${CONLL03_TRAIN_FILE} \
                        --output_path ${FIXED_FILE}
