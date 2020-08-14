import config
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn


def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_tag, ):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = AutoModel.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)

    def forward(self, ids, mask, token_type_ids, target_tag):
        # double check?
        o1, _ = self.bert(ids, attention_mask=mask,
                          token_type_ids=token_type_ids)

        bo_tag = self.bert_drop_1(o1)

        tag = self.out_tag(bo_tag)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        return (tag,  loss_tag)
