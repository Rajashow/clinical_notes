from sklearn import preprocessing
import pandas as pd


def process_data(csv_path, enc_label=None, remove_idetincal_chained_values=False):
    """
    process_data return process data

    read the csv and load data

    Parameters
    ----------
    csv_path : str/path
        path of the csv file
    enc_label : sklearn.labelEncoder, optional
        labelencoder to encode tags, by default None
    remove_idetincal_chained_values : bool, optional
        should sentences with the same labels be removed

    Returns
    -------
    tuple[list[str],list[int],labelEncoder]
        return (sentences,tags,label encod
        er)
    """

    GROUPBY = ["filename", "sentence"]
    data = pd.read_csv(csv_path, keep_default_na=False)
    if not enc_label:
        enc_label = preprocessing.LabelEncoder()
        enc_label.fit(("ANY_SEQ__", *data["label"].unique()))
    data["label"] = enc_label.transform(data["label"])
    data = data.groupby(GROUPBY)
    if remove_idetincal_chained_values:
        data = (
            data.apply(
                lambda group: group if len(group["label"].unique()) != 1 else None
            )
            .dropna()
            .reset_index(drop=True)
            .groupby(GROUPBY)
        )

    sentences = data["word"].apply(list).values
    tag = data["label"].apply(list).values
    return sentences, tag, enc_label
