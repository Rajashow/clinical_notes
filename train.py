import numpy as np

import joblib
import torch

from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel
from utils import process_data_class
import argparse


def get_args():

    parser = argparse.ArgumentParser("Train a bert model on data")
    parser.add_argument(
        "--SSWST",
        help="Skip sentences with the same tags/labels",
        action="store_true",
        default=False,
        required=False,
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"defined but unknown args {unknown}")
    return args


if __name__ == "__main__":
    args = get_args()
    # loading data

    (sentences, tag, enc_tag) = process_data_class(config.TRAINING_FILE,)
    (tsentences, ttag, _) = process_data_class(config.TESTING_FILE, enc_label=enc_tag)

    # saving encoder
    meta_data = {
        "enc_tag": enc_tag,
    }
    joblib.dump(meta_data, "meta.bin")

    num_tag = len(list(enc_tag.classes_))
    # spliting data into train and val
    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag,
    ) = model_selection.train_test_split(sentences, tag, random_state=42, test_size=0.1)
    # creating datasets and data loaders

    train_dataset = dataset.EntityDataset(texts=train_sentences, tags=train_tag)

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(texts=test_sentences, tags=test_tag)

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    test_dataset = dataset.EntityDataset(texts=tsentences, tags=ttag)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=4
    )
    # load model
    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag)
    model.to(device)

    # init parameters
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    # init optimizer
    num_train_steps = int(
        len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS
    )
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    # train model
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        # train loop
        train_loss, train_acc = engine.train_fn(
            train_data_loader, model, optimizer, device, scheduler
        )
        # validation loop
        test_loss, test_acc = engine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        print(f"Train acc = {train_acc} Valid acc = {test_acc}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss

    engine.eval_fn_with_report(test_data_loader, model, device, enc_tag)
    print(f"best loss =  {best_loss}")
