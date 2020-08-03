import csv
import os
import tqdm

import jiant.utils.python.io as py_io
import jiant.utils.zconf as zconf


DATA_CONVERSION = {
    "mnli": {
        "data": {
            "train": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 11},
                "meta": {"skiprows": 1},
            },
            "val": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 15},
                "meta": {"filename": "dev_matched", "skiprows": 1},
            },
            "val_mismatched": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 15},
                "meta": {"filename": "dev_mismatched", "skiprows": 1},
            },
            "test": {
                "cols": {"premise": 8, "hypothesis": 9},
                "meta": {"filename": "test_matched", "skiprows": 1},
            },
            "test_mismatched": {
                "cols": {"premise": 8, "hypothesis": 9},
                "meta": {"filename": "test_mismatched", "skiprows": 1},
            },
        },
        "dir_name": "MNLI",
        "file_format": "tsv",
    },
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
                "meta": {"filename": "train", "skiprows": 1},
            },
            "val_hypothesis": {
                "cols": {"sentence1": 7, "sentence2": 8, "gold_label": 14},
                "meta": {"filename": "dev", "skiprows": 1},
            },
        },
        "dir_name": "SNLI",
        "file_format": "tsv",
    },
    "counterfactual_nli": {
        "data": {
            "train": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("all_combined", "train"), "skiprows": 1},
            },
            "val": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("all_combined", "dev"), "skiprows": 1},
            },
            "test": {
                "cols": {"premise": 0, "hypothesis": 1},
                "meta": {"filename": os.path.join("all_combined", "test"), "skiprows": 1},
            },
            "train_revised_combined": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_combined", "train"), "skiprows": 1},
            },
            "val_revised_combined": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_combined", "dev"), "skiprows": 1},
            },
            "test_revised_combined": {
                "cols": {"premise": 0, "hypothesis": 1},
                "meta": {"filename": os.path.join("revised_combined", "test"), "skiprows": 1},
            },
            "train_revised_hypothesis": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_hypothesis", "train"), "skiprows": 1},
            },
            "val_revised_hypothesis": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_hypothesis", "dev"), "skiprows": 1},
            },
            "test_revised_hypothesis": {
                "cols": {"premise": 0, "hypothesis": 1},
                "meta": {"filename": os.path.join("revised_hypothesis", "test"), "skiprows": 1},
            },
            "train_revised_premise": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_premise", "train"), "skiprows": 1},
            },
            "val_revised_premise": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("revised_premise", "dev"), "skiprows": 1},
            },
            "test_revised_premise": {
                "cols": {"premise": 0, "hypothesis": 1},
                "meta": {"filename": os.path.join("revised_premise", "test"), "skiprows": 1},
            },
            "val_snli": {
                "cols": {"premise": 7, "hypothesis": 8, "label": 14},
                "meta": {"filename": "dev_snli", "skiprows": 1},
            },
            "val_mnli": {
                "cols": {"premise": 8, "hypothesis": 9, "label": 15},
                "meta": {"filename": "dev_matched", "skiprows": 1},
            },
            "train_hypothesis": {
                "cols": {"premise": 0, "hypothesis": 1, "label": 2},
                "meta": {"filename": os.path.join("all_combined", "train"), "skiprows": 1},
            },
            "val_snli_hypothesis": {
                "cols": {"premise": 7, "hypothesis": 8, "label": 14},
                "meta": {"filename": "dev_snli", "skiprows": 1},
            },
        },
        "dir_name": "CounterfactualNLI",
        "file_format": "tsv",
    },
    "adversarial_nli": {
        "data": {
            "train": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R3", "train")},
            },
            "val": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R3", "dev")},
            },
            "test": {
                "cols": {"context": "context", "hypothesis": "hypothesis"},
                "meta": {"filename": os.path.join("R3", "test")},
            },
            "train_R2": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R2", "train")},
            },
            "val_R2": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R2", "dev")},
            },
            "test_R2": {
                "cols": {"context": "context", "hypothesis": "hypothesis"},
                "meta": {"filename": os.path.join("R1", "test")},
            },
            "train_R1": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R1", "train")},
            },
            "val_R1": {
                "cols": {"context": "context", "hypothesis": "hypothesis", "label": "label"},
                "meta": {"filename": os.path.join("R1", "dev")},
            },
            "test_R1": {
                "cols": {"context": "context", "hypothesis": "hypothesis"},
                "meta": {"filename": os.path.join("R1", "test")},
            },
        },
        "dir_name": "anli_v0.1",
        "file_format": "jsonl",
    },
}


def read_tsv(input_file, quotechar=None, skiprows=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        result = list(csv.reader(f, delimiter="\t", quotechar=quotechar))
    if skiprows:
        result = result[skiprows:]
    return result


def get_full_examples(task_name, input_base_path):
    task_metadata = DATA_CONVERSION[task_name]
    file_format = task_metadata["file_format"]
    all_examples = {}
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
        for row in rows:
            try:
                example = {}
                for col, i in phase_config["cols"].items():
                    if 'hypothesis' in phase and ((task_name == 'snli' and col == 'sentence1') or (task_name == 'counterfactual_nli' and col == 'premise')):
                        example[col] = ""
                    else:
                        example[col] = row[i]
                examples.append(example)

            except IndexError:
                if task_name == "qqp":
                    continue
        all_examples[phase] = examples

        if phase == "train":
            print(f"{phase} for {task_name} has {len(examples)} examples")

    return all_examples


def preprocess_all_data(input_base_path, output_base_path):
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_base_path, "configs"), exist_ok=True)
    for task_name in tqdm.tqdm(DATA_CONVERSION):
        task_data_path = os.path.join(output_base_path, "data", task_name)
        os.makedirs(task_data_path, exist_ok=True)
        task_all_examples = get_full_examples(task_name=task_name, input_base_path=input_base_path)
        config = {"task": task_name, "paths": {}, "name": task_name}
        for phase, phase_data in task_all_examples.items():
            phase_data_path = os.path.join(task_data_path, f"{phase}.jsonl")
            py_io.write_jsonl(
                data=phase_data, path=phase_data_path,
            )
            config["paths"][phase] = phase_data_path

        py_io.write_json(
            data=config, path=os.path.join(output_base_path, "configs", f"{task_name}.json")
        )


@zconf.run_config
class RunConfiguration(zconf.RunConfig):
    input_base_path = zconf.attr(type=str)
    output_base_path = zconf.attr(type=str)


def main():
    args = RunConfiguration.default_run_cli()
    preprocess_all_data(
        input_base_path=args.input_base_path, output_base_path=args.output_base_path,
    )


if __name__ == "__main__":
    main()
