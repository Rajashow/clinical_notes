import config
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


class EntityModel(nn.Module):
    def __init__(
        self, num_tag,
    ):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)

    def forward(self, ids, mask, token_type_ids, **kwargs):
        _, pooled_out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids
        )
        bo_tag = self.bert_drop_1(pooled_out)
        output = self.out_tag(bo_tag)
        return output
