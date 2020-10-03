

TRAIN=$1
VAL=$2

cd ..
BASE_DIR=${PWD}
source activate ./env
export PYTHONPATH=${BASE_DIR}/jiant:$PYTHONPATH

cd jiant

DATA_DIR=${BASE_DIR}/data/preprocessed

python jiant/scripts/preproc/counterfactual/make_data_config.py \
    --data_base_path ${DATA_DIR}/data \
    --output_base_path ${DATA_DIR}/ \
    --train ${TRAIN} \
    --val ${VAL}

echo Created ${DATA_DIR}/configs/${TRAIN}-${VAL}.json