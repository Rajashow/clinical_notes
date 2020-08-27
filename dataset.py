from typing import Counter
import config
import torch
import numpy as np


def get_one_hot(targets, n_labels):
    return torch.from_numpy(np.eye(n_labels)[targets])


class EntityDataset:
    def __init__(self, texts, tags, verbose=False):
        # texts: [["hi", ",", "my", "name", "is", "bob"], ["hello".....]]
        # tags: [[1 2 3 4 1 5], [....].....]]
        self.texts = texts
        self.tags = tags
        self._verbose = verbose

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        target_tag = self.tags[item]

        # text -> bert tokens
        inputs = config.TOKENIZER.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            truncation=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        # pad the seq to max len
        padding_len = config.MAX_LEN - len(ids)
        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
