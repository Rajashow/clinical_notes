from train import sentences
import numpy as np

import joblib
import torch

import config
import dataset
from model import EntityModel
from tqdm import tqdm
from utils import process_data
from sklearn.metrics import f1_score

if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]
    num_tag = len(list(enc_tag.classes_))
    sentences, tags, enc_label_ = process_data(config.TESTING_FILE, enc_tag)
    dataset = dataset.EntityDataset(texts=sentences, tags=tags)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag,)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    final_loss = 0
    model.eval()
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        tag, _loss = model(**data)
        y_pred = tag.argmax(2).cpu().numpy().reshape(-1)
        y_true = data["target_tag"]
        final_loss += f1_score(y_true, y_pred,)
    print(f"f1 score : { final_loss / len(data_loader):.5f}")
