MODEL_TYPE=$1

cd ..
BASE_DIR=${PWD}
source activate ./env
export PYTHONPATH=${BASE_DIR}/jiant:$PYTHONPATH

cd jiant

python jiant/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${BASE_DIR}/models/${MODEL_TYPE}

echo Downloaded ${MODEL_TYPE} to ${BASE_DIR}/models/${MODEL_TYPE}