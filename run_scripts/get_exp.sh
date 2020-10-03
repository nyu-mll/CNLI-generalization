TRAIN=$1
VAL=$2
MODEL_TYPE=$3
SBATCH=$4

cd ..
BASE_DIR=${PWD}
source activate ./env
export PYTHONPATH=${BASE_DIR}/jiant:$PYTHONPATH

cd jiant

MODELS_DIR=${BASE_DIR}/models

# Used in arguments
JIANT_DIR=${BASE_DIR}/jiant
COMMAND_DIR=${BASE_DIR}/exp_scripts
EPOCHS=20
N_TRIALS=20
CHECK_STEPS=100
EARLY_INT=3

CACHE_DIR=${BASE_DIR}/cache/${TRAIN}-${VAL}
DATA_DIR=${BASE_DIR}/data

TASK_CONFIG=${DATA_DIR}/preprocessed/configs/${TRAIN}-${VAL}.json
RUN_CONFIG_DIR=${BASE_DIR}/run_configs/${TRAIN}-${VAL}
OUTPUT_DIR=${BASE_DIR}/output_dir/${TRAIN}-${VAL}

MODEL_CONFIG=${MODELS_DIR}/${MODEL_TYPE}/config.json

python jiant/scripts/preproc/counterfactual/generate_exp.py \
    --task_config_path ${TASK_CONFIG} \
	--task_cache_base_path ${CACHE_DIR} \
	--run_config_path ${RUN_CONFIG_DIR} \
	--output_path ${OUTPUT_DIR} \
	--jiant_path ${JIANT_DIR} \
	--model_config ${MODEL_CONFIG} \
	--exp_command_path ${COMMAND_DIR} \
	--train ${TRAIN} \
	--val ${VAL} \
	--epochs ${EPOCHS} \
	--n_trials ${N_TRIALS} \
	--sbatch_name ${SBATCH} \
	--train_batch_tolerance ${TRAIN_BATCH_TOL} \
	--fp16 \
	--no_improvements_for_n_evals ${CHECK_STEPS} \
	--eval_every_steps ${CHECK_STEPS} \
	--extract_exp_name_valpreds