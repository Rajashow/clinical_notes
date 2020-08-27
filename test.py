import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
import seaborn as sns

import config
import dataset
from model import EntityModel
from utils import process_data_class

if __name__ == "__main__":

    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]
    idx2class = list(enc_tag.classes_)
    num_tag = len(list(enc_tag.classes_))
    sentences, tags, enc_label_ = process_data_class(config.TESTING_FILE, enc_tag)
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
    y_pred_list = []

    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        tag, _ = model(**data)

        y_pred = torch.argmax(tag, dim=-1).cpu().numpy()
        y_true = data["target_tag"].cpu().numpy()
        y_pred_list.append(y_pred)
        enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[
            1 : len(y) - 1
        ]

    # print(f"f1 score : { final_loss / len(data_loader):.5%}")
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix_df = pd.DataFrame(confusion_matrix(tags, y_pred_list)).rename(
        columns=idx2class, index=idx2class
    )
    print(classification_report(tags, y_pred_list))
    sns.heatmap(confusion_matrix_df, annot=True)
