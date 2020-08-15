# %%
import csv
from glob import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from tqdm import tqdm
import config

# %%
OUTDIR = "processed/ner"
TRAIN_DIR = "new_bert/train"
TEST_DIR = "new_bert/test"
TAGS = set(
    [
        "MEDICATION",
        "DIABETES",
        "OBESE",
        "SMOKER",
        "HYPERTENSION",
        "CAD",
        "PHI",
        "FAMILY_HIST",
        "HYPERLIPIDEMIA",
    ]
)
TEXT_TAG = "TEXT"
TAGS_TAG = "TAGS"
NUMBER_WORD_PER_SENT = config.MAX_LEN // 1.5
# %%


def maxDisjointIntervals(itr: iter):
    return_lst = []
    if itr:
        # Lambda function to sort the list
        # elements by second element of pairs
        itr.sort(key=lambda x: x[1])

        # First interval will always be
        # included in set
        return_lst.append(itr[0])

        # End point of first interval
        r1 = itr[0][1]

        for elem in itr:
            l1, r2, _, _ = elem
            # Check if given interval overlap with
            # previously included interval, if not
            # then include this interval and update
            # the end point of last added interval
            if l1 > r1:
                return_lst.append(elem)
                r1 = r2
    return return_lst


def has_valid_label_idx(tag):
    return tag.tag in TAGS and tag.attrib["start"] != "-1" and tag.attrib["end"] != "-1"


def is_doc_lvl_tag(tag):
    DOC_LVL_MODIFER = set(("not present", "unknown"))
    return get_modifer(tag, leading_hyphen=False) in DOC_LVL_MODIFER


def get_modifer(tag, leading_hyphen=True):
    modifer = tag.attrib.get(
        "status", tag.attrib.get("TYPE", tag.attrib.get("indicator", ""))
    ).replace(" ", "-")
    if modifer and leading_hyphen:
        modifer = f"-{modifer}"
    return modifer.upper()


def get_tag_data(tag):

    tag_name = f"{tag.tag}{get_modifer(tag)}"
    return (
        int(tag.attrib["start"]),
        int(tag.attrib["end"]),
        tag_name,
        tag.attrib["text"],
    )


def is_end_of_sentence(split_txt, idx, curr_sentence_size):
    # has a period and follows with a space or end of line is sentence
    return (
        (
            ((not (idx + 1 < len(split_txt))) or split_txt[idx + 1].isspace())
            and "." in split_txt[idx]
        )
        or (
            split_txt[idx + 1].isspace()
            and (curr_sentence_size >= int(NUMBER_WORD_PER_SENT * 0.9))
            and split_txt[idx:].index("\n") >= int(NUMBER_WORD_PER_SENT * 0.1)
        )
        or (
            (not split_txt[idx].isalnum())
            and split_txt[idx:].index(split_txt[idx])
            >= (int(NUMBER_WORD_PER_SENT * 0.9) - curr_sentence_size)
        )
    )


def process_xml(i, file):

    xml_parsed = ET.parse(file)

    clinical_note = xml_parsed.find(TEXT_TAG).text
    tag_containers = xml_parsed.findall(TAGS_TAG)
    ext_tags = [
        get_tag_data(tag)
        for tag in tag_containers[0]
        if has_valid_label_idx(tag) and not is_doc_lvl_tag(tag)
    ]
    disjoint_interval_idx = 0
    words = []
    labels = []
    hit = False
    char_idx = 0
    sentence_words = []
    sentence_labels = []
    disjoint_interval_tags = maxDisjointIntervals(ext_tags)
    split_txt = [word for word in re.split("(\W)", clinical_note,) if word]
    try:

        low, high, tag, _txt = disjoint_interval_tags[disjoint_interval_idx]

    except IndexError:
        for i, word in enumerate(split_txt):
            if not word.isspace():
                sentence_words.append(word)
                sentence_labels.append("O")
                if is_end_of_sentence(split_txt, i, len(sentence_words)):
                    words.append(sentence_words)
                    labels.append(sentence_labels)
                    sentence_labels = []
                    sentence_words = []
            char_idx += len(word)
    else:

        for i, word in enumerate(split_txt):
            if not word.isspace():
                label = "O"
                if low <= char_idx <= high:
                    hit = True
                    label = tag
                else:
                    if hit:
                        hit = False
                        try:
                            disjoint_interval_idx += 1
                            low, high, tag, _txt = disjoint_interval_tags[
                                disjoint_interval_idx
                            ]
                        except IndexError:
                            pass
                    try:
                        while high < char_idx:
                            disjoint_interval_idx += 1
                            low, high, tag, _txt = disjoint_interval_tags[
                                disjoint_interval_idx
                            ]
                    except IndexError:
                        pass

                sentence_labels.append(label)
                sentence_words.append(word)
                if not hit and is_end_of_sentence(split_txt, i, len(sentence_words)):
                    words.append(sentence_words)
                    labels.append(sentence_labels)
                    sentence_labels = []
                    sentence_words = []
            assert clinical_note[char_idx : char_idx + len(word)] == word
            char_idx += len(word)

    if sentence_labels and sentence_words:
        words.append(sentence_words)
        labels.append(sentence_labels)
    assert len(words) == len(labels), "words and labels count don't match"
    try:

        assert (
            len(disjoint_interval_tags) <= (disjoint_interval_idx+1)
        ), "didn't use all the tags"

        pass
    except StopIteration:
        pass
    return words, labels


def process_all_xml(folder=None, outdir="", out_modifer=""):
    xmls = glob(os.path.join(folder, "*.xml"))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, f"i2b2{out_modifer}.csv"), "w") as file:
        csv_file = csv.writer(file,)
        csv_file.writerow(["filename", "sentence", "number", "word", "label"])
        for i, file in tqdm(
            enumerate(xmls), total=len(xmls), desc=f"Processing files from {folder}: "
        ):
            lst_words, lst_labels = process_xml(i, file)
            csv_file.writerows(
                [
                    (os.path.basename(file), i, j, word, label,)
                    for i, (words, labels) in enumerate(zip(lst_words, lst_labels))
                    for j, (word, label) in enumerate(zip(words, labels))
                ]
            )
    pass


# %%
process_all_xml(TRAIN_DIR, outdir=OUTDIR, out_modifer="_train")
process_all_xml(TEST_DIR, outdir=OUTDIR, out_modifer="_test")

# %%
