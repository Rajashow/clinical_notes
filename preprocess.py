# %%
import csv
from glob import glob
import pandas as pd
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
TAGS = set(["MEDICATION", "OBSEE", "SMOKER", "HYPERTENSION", "PHI", "FAMILY_HIST"])
TEXT_TAG = "TEXT"
TAGS_TAG = "TAGS"
NUMBER_WORD_PER_SENT = config.MAX_LEN // 1.5
# %%


def maxDisjointIntervals(itr: iter):
    if itr:
        return_lst = []
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
    else:
        return [[-1, -1, "O", ""]]


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


def process_xml(i, file):
    xml_parsed = ET.parse(file)

    clinical_note = xml_parsed.find(TEXT_TAG).text
    tag_containers = xml_parsed.findall(TAGS_TAG)
    ext_tags = [
        get_tag_data(tag)
        for tag in tag_containers[0]
        if has_valid_label_idx(tag) and not is_doc_lvl_tag(tag)
    ]

    interval_sub_set_ = maxDisjointIntervals(ext_tags)
    interval_idx = 0
    idx = 0
    hit = False
    words = []
    labels = []
    low, high, tag, txt = interval_sub_set_[interval_idx]

    for word in re.split("(\W)", clinical_note):
        if len(word) and not word.isspace():
            if low <= idx <= high and word in txt:
                labels.append(tag)
                hit = True
            else:
                labels.append("O")
                # TODO: CLEAN THIS
                if hit:
                    hit = False
                    interval_idx += 1
                    if interval_idx < len(interval_sub_set_):
                        low, high, tag, txt = interval_sub_set_[interval_idx]
            words.append(word)
        idx += len(word)
    assert len(words) == len(labels), "words and labels count don't match"
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
            words, labels = process_xml(i, file)
            csv_file.writerows(
                [
                    (
                        os.path.basename(file),
                        i // NUMBER_WORD_PER_SENT,
                        i % NUMBER_WORD_PER_SENT,
                        word,
                        label,
                    )
                    for i, (word, label) in enumerate(zip(words, labels))
                ]
            )


# %%
process_all_xml(TRAIN_DIR, outdir=OUTDIR, out_modifer="_train")
process_all_xml(TEST_DIR, outdir=OUTDIR, out_modifer="_test")

# %%
