import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from jiant.tasks.core import (
    BaseExample,
    BaseTokenizedExample,
    BaseDataRow,
    BatchMixin,
    Task,
    TaskTypes,
)
from jiant.tasks.lib.templates.shared import labels_to_bimap, double_sentence_featurize
from jiant.utils.python.io import read_json_lines


@dataclass
class Example(BaseExample):
    guid: str
    passage_text: str
    query_text: str
    entity_start_char_idx: int  # unused
    entity_end_char_idx: int  # unused
    entity_str: str
    passage_idx: int
    question_idx: int
    answers_dict: dict
    label: bool

    def tokenize(self, tokenizer):
        filled_query_text = self.query_text.replace("@placeholder", self.entity_str)

        return TokenizedExample(
            guid=self.guid,
            passage_tokens=tokenizer.tokenize(self.passage_text),
            query_tokens=tokenizer.tokenize(filled_query_text),
            label_id=ReCoRDTask.LABEL_TO_ID[self.label],
        )


@dataclass
class TokenizedExample(BaseTokenizedExample):
    guid: str
    passage_tokens: List
    query_tokens: List
    label_id: int

    def featurize(self, tokenizer, feat_spec):
        return double_sentence_featurize(
            guid=self.guid,
            input_tokens_a=self.passage_tokens,
            input_tokens_b=self.query_tokens,
            label_id=self.label_id,
            tokenizer=tokenizer,
            feat_spec=feat_spec,
            data_row_class=DataRow,
        )


@dataclass
class DataRow(BaseDataRow):
    guid: str
    input_ids: np.ndarray
    input_mask: np.ndarray
    segment_ids: np.ndarray
    label_id: int
    tokens: list


@dataclass
class Batch(BatchMixin):
    input_ids: torch.LongTensor
    input_mask: torch.LongTensor
    segment_ids: torch.LongTensor
    label_id: torch.LongTensor
    tokens: list


class ReCoRDTask(Task):
    Example = Example
    TokenizedExample = Example
    DataRow = DataRow
    Batch = Batch

    TASK_TYPE = TaskTypes.CLASSIFICATION
    LABELS = [False, True]
    LABEL_TO_ID, ID_TO_LABEL = labels_to_bimap(LABELS)

    def get_train_examples(self):
        return self._create_examples(lines=read_json_lines(self.train_path), set_type="train")

    def get_val_examples(self):
        return self._create_examples(lines=read_json_lines(self.val_path), set_type="val")

    def get_test_examples(self):
        return self._create_examples(lines=read_json_lines(self.test_path), set_type="test")

    @classmethod
    def _create_examples(cls, lines, set_type):
        examples = []
        for line in lines:
            passage_text = line["passage"]["text"]
            for qas in line["qas"]:
                answers_dict = {
                    (answer["start"], answer["end"]): answer["text"] for answer in qas["answers"]
                }
                for entity in line["passage"]["entities"]:
                    entity_span = (entity["start"], entity["end"])
                    if entity_span in answers_dict:
                        assert (
                            passage_text[entity_span[0] : entity_span[1] + 1]
                            == answers_dict[entity_span]
                        )
                        label = True
                    else:
                        label = False
                    examples.append(
                        Example(
                            guid="%s-%s" % (set_type, len(examples)),
                            passage_text=passage_text,
                            query_text=qas["query"],
                            entity_start_char_idx=entity_span[0],
                            entity_end_char_idx=entity_span[1] + 1,  # make exclusive
                            entity_str=passage_text[entity_span[0] : entity_span[1] + 1],
                            passage_idx=line["idx"],
                            question_idx=qas["idx"],
                            answers_dict=answers_dict,
                            label=label,
                        )
                    )
        return examples
