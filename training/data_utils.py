import pandas as pd
from typing import Dict, Optional, Sequence
import json
import transformers
from torch.utils.data import Dataset
import torch
from dataclasses import dataclass, field



class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.list_data_dict = pd.read_json(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            example['Instruction'] for example in list_data_dict
        ]
        targets = [f"{filter_notebook(example['Notebook'])}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


def filter_notebook(notebook: str, include_outputs=True) -> str:
    """Filters a notebook to only include the source code."""
    try:
        dic = json.loads(notebook)
    except Exception:
        print(notebook)
        return notebook

    cells = dic["cells"]
    returned_array = []
    for cell in cells:
        if include_outputs:
            try:
                returned_array.append(cell["source"] + cell["outputs"])
            except Exception:
                returned_array.append(cell["source"])
        else:
            returned_array.append(cell["source"])
    return " ".join(returned_array)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_dataset = dataset_cls(data_args.data_path, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_dataset = dataset_cls(data_args.eval_path, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

# class KaggleInstructionDataset(Dataset):
#     def __init__(self, raw_data, tokenizer, max_words=1500):
#         super(KaggleInstructionDataset, self).__init__()
#         self.ann = raw_data
#         self.max_words = max_words
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):
#         IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

#         ann = self.ann.iloc[index]
#         prompt = ann["Instruction"]
#         output = filter_notebook(ann["Notebook"])
#         example = prompt + '\n' + output  # This is the full example
#         prompt = torch.tensor(
#             self.tokenizer.encode(prompt), dtype=torch.int64
#         )
#         example = self.tokenizer.encode(example)
#         example.append(self.tokenizer.eos_token_id)
#         example = torch.tensor(
#             example, dtype=torch.int64
#         )
#         padding = self.max_words - example.shape[0]
#         if padding > 0:
#             example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
#         elif padding < 0:
#             example = example[: self.max_words]
#         labels = copy.deepcopy(example)
#         labels[: len(prompt)] = -1
#         example_mask = example.ge(0)
#         label_mask = labels.ge(0)
#         example[~example_mask] = 0
#         labels[~label_mask] = IGNORE_INDEX
#         example_mask = example_mask.float()
#         label_mask = label_mask.float()

#         return {
#             "input_ids": example,
#             "labels": labels,
#             "attention_mask": example_mask,
#         }
    
#     def get_notebook(self,idx):
#        ann  = self.ann.iloc[idx]

#        return filter_notebook(ann["Notebook"])