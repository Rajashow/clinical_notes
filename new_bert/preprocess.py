
# %%
import csv
from glob import glob
import pandas as pd
import os
import xml.etree.ElementTree as ET
import numpy as np
import re
from tqdm import tqdm
# %%
OUTDIR = "./processed"
TRAIN_DIRS = ['training-RiskFactors-Gold-Set1/', ]
TEST_DIRS = ['testing-RiskFactors-Gold']
# ['MEDICATION', 'OBSEE', 'SMOKER', 'HYPERTENSION', 'PHI', 'FAMILY_HIST']
TAGS = set(("SMOKER"))
TEXT_TAG = "TEXT"
TAGS_TAG = "TAGS"
START_CDATA = "<TEXT><![CDATA["
END_CDATA = "]]></TEXT>"

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


def valid_label(tag):
    return tag.tag in TAGS and tag.attrib["start"] != "-1" and tag.attrib["end"] != "-1" and ((tag.tag == "SMOKER" and tag.attrib["status"] != "unknown") or tag.tag != "SMOKER")


def get_tag_data(tag):
    if tag.tag != "SMOKER":
        return (int(tag.attrib['start']), int(tag.attrib['end']), tag.tag, tag.attrib['text'])
    else:
        return (int(tag.attrib['start']), int(tag.attrib['end']), f"{tag.tag}-{tag.attrib['status']}", tag.attrib['text'])


def process_xml(i, file):
    xml_parsed = ET.parse(file)

    clinical_note = xml_parsed.find(TEXT_TAG).text
    tag_containers = xml_parsed.findall(TAGS_TAG)
    ext_tags = [get_tag_data(tag)
                for tag in tag_containers[0] if valid_label(tag)]

    interval_sub_set_ = maxDisjointIntervals(ext_tags)
    interval_idx = 0
    idx = 0
    hit = False
    words = []
    labels = []
    low, high, tag, txt = interval_sub_set_[interval_idx]

    for word in re.split('(\W)', clinical_note):
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
    return words, labels


def process_all_xml(folder=None, outdir="", out_modifer=""):
    xmls = glob(os.path.join(folder, "*.xml"))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with \
            open(os.path.join(outdir, f"words{out_modifer}.csv"), "w") as words_file,\
            open(os.path.join(outdir, f"labels{out_modifer}.csv"), "w") as labels_file:
        words_csv = csv.writer(words_file,)
        labels_csv = csv.writer(labels_file,)
        words_csv.writerow(["filename", "words"])
        labels_csv.writerow(["filename", "labels"])
        for i, file in tqdm(enumerate(xmls)):
            words, labels = process_xml(i, file)
            words_csv.writerow([file]+words)
            labels_csv.writerow([file]+labels)


# %%
process_all_xml("new_bert/train", outdir="processed/ner", out_modifer="_train")
process_all_xml("new_bert/test", outdir="processed/ner", out_modifer="_test")

# %%
