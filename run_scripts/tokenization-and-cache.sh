

TRAIN=$1
VAL=$2
MODEL_TYPE=$3

cd ..
BASE_DIR=${PWD}
source activate ./env
export PYTHONPATH=${BASE_DIR}/jiant:$PYTHONPATH

cd jiant

DATA_DIR=${BASE_DIR}/data
MODELS_DIR=${BASE_DIR}/models
CACHE_DIR=${BASE_DIR}/cache

python jiant/proj/simple/tokenize_and_cache.py \
    --task_config_path ${DATA_DIR}/preprocessed/configs/${TRAIN}-${VAL}.json \
    --model_type ${MODEL_TYPE} \
    --model_tokenizer_path ${MODELS_DIR}/${MODEL_TYPE}/tokenizer \
    --phases train,val,test \
    --max_seq_length 256 \
    --do_iter \
    --force_overwrite \
    --smart_truncate \
    --output_dir ${CACHE_DIR}/${TRAIN}-${VAL}

ls ${CACHE_DIR}/${TRAIN}-${VAL}