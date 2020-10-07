MODEL_TYPE=$1
SBATCH=$2

TRAINS=(snlisub0 snlisub1 snlisub2 snlisub3 snlisub4 cnli cnli_seed)
VALS=(glue_diagnostic stress mnli)

RUN_SCRIPTS=${PWD}

# Export model
cd ..
BASE_DIR=${PWD}
source activate ./env
export PYTHONPATH=${BASE_DIR}/jiant:$PYTHONPATH

cd jiant

python jiant/scripts/preproc/export_model.py \
    --model_type ${MODEL_TYPE} \
    --output_base_path ${BASE_DIR}/models/${MODEL_TYPE}

echo Downloaded ${MODEL_TYPE} to ${BASE_DIR}/models/${MODEL_TYPE}

cd $RUN_SCRIPTS

# Get data configurations for jiant
for train in "${TRAINS[@]}"
do
	echo $train

	for val in "${VALS[@]}"
	do
		echo $val
		sh make_data_config.sh ${train} ${val}
	done
done

# Tokenize and cache data for jiant
for train in "${TRAINS[@]}"
do
	echo $train

	for val in "${VALS[@]}"
	do
		echo $val
		sh tokenization-and-cache.sh ${train} ${val} ${MODEL_TYPE}
	done
done

# Get commands for experiments
for train in "${TRAINS[@]}"
do
	echo $train

	for val in "${VALS[@]}"
	do
		echo $val
		sh get_exp.sh ${train} ${val} ${MODEL_TYPE} ${SBATCH}
	done
done