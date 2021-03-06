import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentence = input("Give me an sentence to try out:")
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)

    test_dataset = dataset.EntityDataset(texts=[sentence], tags=[[0] * len(sentence)])

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag,)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

        y_pred_softmax = torch.log_softmax(tag, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        print(enc_tag.inverse_transform(y_pred_tags.cpu()))
