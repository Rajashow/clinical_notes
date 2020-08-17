from sklearn import preprocessing
import pandas as pd


def process_data(csv_path, enc_label=None):
    GROUPBY = ["filename", "sentence"]
    data = pd.read_csv(csv_path, keep_default_na=False)
    data.groupby(GROUPBY)["word"].apply(lambda x: " ".join(map(str, x)))
    if not enc_label:
        enc_label = preprocessing.LabelEncoder()

    data.loc[:, "label"] = enc_label.fit_transform(data["label"])
    sentences = data.groupby(GROUPBY)["word"].apply(list).values
    tag = data.groupby(GROUPBY)["label"].apply(list).values
    return (sentences, tag, enc_label)
