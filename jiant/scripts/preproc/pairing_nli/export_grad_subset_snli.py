import csv
import os
import tqdm
import pickle

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


DATA_CONVERSION = {
    "snli": {
        "data": {
            "train": {
                "cols": {"sentence1": 7, "sentence2": 8, "gold_label": 10},
                "meta": {"skiprows": 1},
            },
            "val": {
                "cols": {"sentence1": 7, "sentence2": 8, "gold_label": 14},
                "meta": {"filename": "dev", "skiprows": 1},
            },
            "test": {
                "cols": {"sentence1": 7, "sentence2": 8},
                "meta": {"filename": "test", "skiprows": 1},
            },
            "val_mnli": {
                "cols": {"sentence1": 8, "sentence2": 9, "gold_label": 15},
                "meta": {"filename": "dev_matched", "skiprows": 1},
            },
            "train_hypothesis": {
                "cols": {"sentence1": 7, "sentence2": 8, "gold_label": 10},
                "meta": {"skiprows": 1},
            },
            "val_hypothesis": {
                "cols": {"sentence1": 7, "sentence2": 8, "gold_label": 14},
                "meta": {"filename": "dev", "skiprows": 1},
            },
        },
        "dir_name": "SNLI",
        "file_format": "tsv",
    },
}


def read_tsv(input_file, quotechar=None, skiprows=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        result = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    if skiprows:
        result = result[skiprows:]
    return result


def get_full_examples(
        task_name,
        input_base_path,
        keep_list_path,
        sub_idx,
):
    task_metadata = DATA_CONVERSION[task_name]
    file_format = task_metadata["file_format"]
    all_examples = {}
    keep_list = pickle.load(open(f"{keep_list_path}{sub_idx}.pkl", 'rb'))
    for phase, phase_config in task_metadata["data"].items():
        meta_dict = phase_config.get("meta", {})
        filename = meta_dict.get("filename", phase)

        file_path = os.path.join(input_base_path, task_metadata["dir_name"], f"{filename}.{file_format}")
        rows = None
        if file_format == "tsv":
            rows = read_tsv(
                file_path,
                skiprows=meta_dict.get("skiprows"),
            )
        elif file_format == "jsonl":
            rows = py_io.read_jsonl(
                file_path,
            )

        assert not rows is None, f"File format for task {task_name} not supported: {file_format}"

        examples = []
        idx = 0
        for row in rows:
            if not idx in keep_list:
                idx += 1
                continue

            try:
                example = {}
                for col, i in phase_config["cols"].items():
                    if "hypothesis" in phase and col == 'sentence1':
                        example[col] = ""
                    else:
                        example[col] = row[i]

                if phase == "train" and example['gold_label'] == '':
                    print("empty")
                    continue
                else:
                    examples.append(example)
                    idx += 1

            except IndexError:
                if task_name == "qqp":
                    continue
        all_examples[phase] = examples

        if phase == "train":
            print(f"{phase} for {task_name} has {len(examples)} examples")

    return all_examples


def preprocess_all_data(
        input_base_path,
        output_base_path,
        keep_list_path,
        sub_idx,
):
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "configs"), exist_ok=True)
    for task_name in tqdm.tqdm(DATA_CONVERSION):
        task_data_path = os.path.join(output_base_path, "data", f"{task_name}sub{sub_idx}")
        os.makedirs(task_data_path, exist_ok=True)
        task_all_examples = get_full_examples(
            task_name=task_name,
            input_base_path=input_base_path,
            keep_list_path=keep_list_path,
            sub_idx=sub_idx,
        )
        config = {"task": task_name, "paths": {}, "name": task_name}
        for phase, phase_data in task_all_examples.items():
            phase_data_path = os.path.join(task_data_path, f"{phase}.jsonl")
            py_io.write_jsonl(
                data=phase_data, path=phase_data_path,
            )
            config["paths"][phase] = phase_data_path

        py_io.write_json(
            data=config, path=os.path.join(output_base_path, "configs", f"{task_name}sub{sub_idx}.json")
        )


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    input_base_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)
    keep_list_path = zconf.attr(type=str)
    sub_idx = zconf.attr(type=str)


def main():
    args = RunConfiguration.default_run_cli()
    preprocess_all_data(
        input_base_path=args.input_base_path,
        output_base_path=args.output_base_path,
        keep_list_path=args.keep_list_path,
        sub_idx=args.sub_idx,
    )


if __name__ == "__main__":
    main()
