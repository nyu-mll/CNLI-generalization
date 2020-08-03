import os

import torch
import numpy as np
from numpy.random import rand, randint
from datetime import datetime as dt

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf
import jiant.scripts.preproc.pairing_nli.make_config as make_config

EVAL_STEPS = 5000
NO_IMPROV_INT = 30
SAVE_STEPS = 10000

@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    # Directories
    task_config_path = zconf.attr(type=str, required=True)
    task_cache_base_path = zconf.attr(type=str, required=True)
    run_config_path = zconf.attr(type=str, required=True)
    output_path = zconf.attr(type=str, required=True)
    jiant_path = zconf.attr(type=str, required=True)
    model_config = zconf.attr(type=str, required=True)
    exp_command_path = zconf.attr(type=str, required=True)

    # Others
    task_name = zconf.attr(type=str, required=True)
    epochs = zconf.attr(type=int, required=True)
    n_trials = zconf.attr(type=int, required=True)
    matchlist_pickle_path = zconf.attr(type=str, required=True)
    sbatch_name = zconf.attr(type=str, required=True)

    # === Optional parameters === #
    batch_clustering = zconf.attr(action='store_true')
    train_batch_tolerance = zconf.attr(type=int, default=0)
    extract_exp_name_valpreds = zconf.attr(action="store_true")
    fp16 = zconf.attr(action="store_true")
    no_improvements_for_n_evals = zconf.attr(type=int, default=0)
    eval_every_steps = zconf.attr(type=int, default=0)
    boolq = zconf.attr(action="store_true")

def main(args: RunConfiguration):
    os.makedirs(args.run_config_path, exist_ok=True)
    os.makedirs(args.exp_command_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)

    now = dt.now().strftime("%Y%m%d%H%M")
    commands = []
    cluster = "_cluster" if args.batch_clustering else ""
    for idx in range(args.n_trials):
        sample_seed, sample_lr, sample_bs = sample_hyper_parameters(boolq=args.boolq)
        exp_name = f"{args.task_name}{cluster}-bs_{sample_bs}-lr_{sample_lr}-seed_{sample_seed}-epochs_{args.epochs}"
        task_container_config = os.path.join(args.run_config_path, f"{exp_name}.json")

        py_io.write_json(
            single_task_config(
                task_config_path=args.task_config_path,
                task_cache_base_path=args.task_cache_base_path,
                train_batch_size=sample_bs,
                epochs=args.epochs,
                batch_clustering=args.batch_clustering,
                matchlist_pickle_path=args.matchlist_pickle_path,
                train_batch_tolerance=args.train_batch_tolerance,
            ),
            path=task_container_config,
        )
        commands.append(
            single_task_command(
                args=args,
                task_container_config=task_container_config,
                lr=sample_lr,
                seed=sample_seed,
                exp_name = exp_name,
            )
        )

    with open(os.path.join(args.exp_command_path,f'submit_exp_{args.task_name}{cluster}_{now}.sh'), "a") as f:
        for command in commands:
            f.write(command)


def sample_hyper_parameters(boolq=False):
    seed_max = 1e6
    if boolq:
        lr_cands = [1e-5]
        bs_cands = [16, 32]
    else:
        lr_cands = [1e-5, 2e-5, 3e-5]
        bs_cands = [32, 64]

    sample_seed = int(rand()*seed_max)
    sample_lr = lr_cands[randint(len(lr_cands))]
    sample_bs = bs_cands[randint(len(bs_cands))]

    return sample_seed, sample_lr, sample_bs

def get_num_examples_from_cache(cache_path):
    cache_metadata_path = os.path.join(cache_path, "data_args.p")
    return torch.load(cache_metadata_path)["length"]

def single_task_command(
        args: RunConfiguration,
        task_container_config: str,
        lr: float,
        seed: int,
        exp_name: str,
        phases=("train", "val"),
):
    do_train = "--do_train " if "train" in phases else ""
    do_val = "--do_val " if "val" in phases else ""

    command = [
        f"{os.path.join(args.jiant_path, 'jiant', 'proj', 'main', 'runscript.py')} ",
        f"run ",
        f"--ZZsrc {args.model_config} ",
        f"--jiant_task_container_config_path {task_container_config} "
        f"--model_load_mode from_transformers ",
        f"--learning_rate {lr} ",
        f"--force_overwrite ",
        f"{do_train}{do_val}",
        f"--do_save ",
        f"--eval_every_steps {EVAL_STEPS} ",
        f"--no_improvements_for_n_evals {NO_IMPROV_INT} ",
        f"--save_checkpoint_every_steps {SAVE_STEPS} ",
        f"--seed {seed} ",
        f"--output_dir {args.output_path} ",
        f"--val_jsonl --args_jsonl ",
        f"--custom_best_name {f'best_{exp_name}'} ",
        f"--custom_checkpoint_name {f'checkpoint_{exp_name}'} ",
        f"--custom_logger_post _{exp_name} ",
        f"--no_improvements_for_n_evals {args.no_improvements_for_n_evals} ",
        f"--eval_every_steps {args.eval_every_steps} ",
    ]

    if args.extract_exp_name_valpreds:
        command.append(f"--write_val_preds ")
        command.append(f"--extract_exp_name_valpreds ")

    if args.fp16:
        command.append(f"--fp16 ")

    assert os.path.basename(args.sbatch_name).split('.')[1] == 'sbatch', f"arg.sbatch_name is not an sbatch file: {args.sbatch_name}"
    return f'COMMAND="{"".join(command)}" sbatch {args.sbatch_name}\n'

def single_task_config(
        task_config_path,
        train_batch_size=None,
        task_cache_base_path=None,
        task_cache_train_path=None,
        task_cache_val_path=None,
        task_cache_val_labels_path=None,
        epochs=None, max_steps=None,
        eval_batch_multiplier=2,
        eval_batch_size=None,
        gradient_accumulation_steps=1,
        eval_subset_num=500,
        warmup_steps_proportion=0.1,
        phases=("train", "val"),
        batch_clustering=False,
        matchlist_pickle_path="",
        train_batch_tolerance=0,
):
    task_config = py_io.read_json(os.path.expandvars(task_config_path))
    task_name = task_config["name"]

    do_train = "train" in phases
    do_val = "val" in phases

    cache_path_dict = {}
    # task_cache_base_path = f"{task_cache_base_path}-clustered" if batch_clustering else task_cache_base_path
    if do_train:
        if task_cache_train_path is None:
            task_cache_train_path = os.path.join(task_cache_base_path, "train")
        cache_path_dict["train"] = os.path.expandvars(task_cache_train_path)

    if do_val:
        if task_cache_val_path is None:
            task_cache_val_path = os.path.join(task_cache_base_path, "val")
        if task_cache_val_labels_path is None:
            task_cache_val_labels_path = os.path.join(task_cache_base_path, "val_labels")
        cache_path_dict["val"] = os.path.expandvars(task_cache_val_path)
        cache_path_dict["val_labels"] = os.path.expandvars(task_cache_val_labels_path)

    if do_train:
        assert (epochs is None) != (max_steps is None)
        assert train_batch_size is not None
        effective_batch_size = train_batch_size * gradient_accumulation_steps
        num_training_examples = get_num_examples_from_cache(
            cache_path=os.path.expandvars(task_cache_train_path),
        )
        max_steps = num_training_examples * epochs // effective_batch_size
    else:
        max_steps = 0
        train_batch_size = 0

    if do_val:
        if eval_batch_size is None:
            assert train_batch_size is not None
            eval_batch_size = train_batch_size * eval_batch_multiplier

    batch_method = 'clustered' if batch_clustering else 'default'
    min_batch_size = max(0,train_batch_size - train_batch_tolerance)
    total_batches = max_steps*gradient_accumulation_steps

    config_dict = {
        "task_config_path_dict": {
            task_name: os.path.expandvars(task_config_path),
        },
        "task_cache_config_dict": {
            task_name: cache_path_dict,
        },
        "sampler_config": {
            "sampler_type": "UniformMultiTaskSampler",
        },
        "global_train_config": {
            "max_steps": max_steps,
            "warmup_steps": int(max_steps * warmup_steps_proportion),
        },
        "task_specific_configs_dict": {
            task_name: {
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "eval_subset_num": eval_subset_num,
                "batch_method" : batch_method,
                "min_batch_size" : min_batch_size,
                "total_batches" : total_batches,
                "matchlist_pickle_path" : matchlist_pickle_path,
            },
        },
        "taskmodels_config": {
            "task_to_taskmodel_map": {
                task_name: task_name,
            },
            "taskmodel_config_map": {
                task_name: None,
            }
        },
        "task_run_config": {
            "train_task_list": [task_name] if do_train else [],
            "train_val_task_list": [task_name] if do_train else [],
            "val_task_list": [task_name] if do_val else [],
            "test_task_list": [],
        },
        "metric_aggregator_config": {
            "metric_aggregator_type": "EqualMetricAggregator",
        },
    }
    return config_dict


if __name__ == "__main__":
    main(RunConfiguration.default_run_cli())
