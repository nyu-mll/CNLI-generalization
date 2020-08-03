import os

import torch

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    # === Required parameters === #
    task_config_path = zconf.attr(type=str, required=True)
    task_cache_base_path = zconf.attr(type=str, required=True)
    train_batch_size = zconf.attr(type=int, required=True)
    epochs = zconf.attr(type=int, required=True)
    output_path = zconf.attr(type=str, required=True)


def main(args: RunConfiguration):
    os.makedirs(os.path.split(args.output_path)[0], exist_ok=True)
    py_io.write_json(
        single_task_config(
            task_config_path=args.task_config_path,
            task_cache_base_path=args.task_cache_base_path,
            train_batch_size=args.train_batch_size,
            epochs=args.epochs,
        ),
        path=args.output_path,
    )


def get_num_examples_from_cache(cache_path):
    cache_metadata_path = os.path.join(cache_path, "data_args.p")
    return torch.load(cache_metadata_path)["length"]


def single_task_config(task_config_path,
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
                       phases=("train", "val")):
    task_config = py_io.read_json(os.path.expandvars(task_config_path))
    task_name = task_config["name"]

    do_train = "train" in phases
    do_val = "val" in phases

    cache_path_dict = {}
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
