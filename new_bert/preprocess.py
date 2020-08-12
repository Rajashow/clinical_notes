
from glob import glob
import pandas as pd
import os
import xml.etree.ElementTree as ET
import numpy as np

OUTDIR = "./processed"
TRAIN_DIRS = ['training-RiskFactors-Gold-Set1/', ]
TEST_DIRS = ['testing-RiskFactors-Gold']
"""
format::
Sentence #,Word,POS,Tag
Sentence: 1,Thousands,NNS,O
,of,IN,O
,demonstrators,NNS,O
,have,VBP,O
,marched,VBN,O
,through,IN,O
,London,NNP,B-geo
,to,TO,O
,protest,VB,O
,the,DT,O
,war,NN,O
,in,IN,O
,Iraq,NNP,B-geo
,and,CC,O
,demand,VB,O
,the,DT,O
,withdrawal,NN,O
,of,IN,O
,British,JJ,B-gpe
,troops,NNS,O
,from,IN,O
,that,DT,O
,country,NN,O
,.,.,O
"""
"Sentence #{i}"
TAGS = set(("SMOKERS", "DIABETES", "SMOKER"))
TEXT_TAG = "TEXT"

# %%
START_CDATA = "<TEXT><![CDATA["
END_CDATA = "]]></TEXT>"
error_files = ["256-01.xml"]


def valid_label(tag):
    return tag.tag in TAGS and tag.attrib["start"] != "-1" and tag.attrib["end"] != "-1"


def process_xml(i, file):
    if file in error_files:
        return None, None
    with open(file, mode='r') as f:
        lines = f.readlines()
        text, in_text = [], False
        for i, l in enumerate(lines):
            if START_CDATA in l:
                text.append(list(l[l.find(START_CDATA) + len(START_CDATA):]))
                in_text = True
            elif END_CDATA in l:
                text.append(list(l[:l.find(END_CDATA)]))
                break
            elif in_text:
                if file.endswith('180-03.xml') and '0808' in l and 'Effingham' in l:
                    print("Adjusting known error")
                    l = l[:9] + ' ' * 4 + l[9:]
                text.append(list(l))
    pos_transformer = {}
    linear_pos = 1
    for line, sentence in enumerate(text):
        for char_pos, _char in enumerate(sentence):
            pos_transformer[linear_pos] = (line, char_pos)
            linear_pos += 1
    xml_parsed = ET.parse(file)
    tag_containers = xml_parsed.findall("TAGS")
    ext_tags = [(tag.tag, tag)
                for tag in tag_containers[0] if valid_label(tag)]

    labels = [['O'] * len(sentence) for sentence in text]

    for label, _tag in ext_tags:
        # label = _tag.attrib['TYPE']
        start_pos, end_pos, _text = _tag.attrib['start'], _tag.attrib['end'], _tag.attrib['text']
        start_pos, end_pos = int(start_pos)+1, int(end_pos)
        _text = ' '.join(_text.split())

        if _text == 'Johnson and Johnson' and file.endswith('188-05.xml'):
            print("Adjusting known error")
            _text = 'Johnson & Johnson'

        (start_line, start_char), (end_line,
                                   end_char) = pos_transformer[start_pos], pos_transformer[end_pos]

        obs_text = []
        last_line = None
        last_text = None
        last_start_c = None
        last_end_c = None
        for line in range(start_line, end_line+1):
            t = text[line]
            s = start_char if line == start_line else 0
            e = end_char if line == end_line else len(t)
            obs_text.append(''.join(t[s:e+1]).strip())

            last_line = line
            last_text = t
            last_start_c = s
            last_end_c = e

        obs_text = ' '.join(obs_text)
        obs_text = ' '.join(obs_text.split())

        assert obs_text == _text, (
            (f"Texts don't match! {_text} v {obs_text}  \n||\n") + str((
                start_pos, end_pos, last_line, last_start_c, last_end_c, last_text, file
            ) + "\n||")
        )

        labels[end_line][end_char] = f'I-{label}'
        labels[start_line][start_char] = f'B-{label}'

        for line in range(start_line, end_line+1):
            t = text[line]
            s = start_char+1 if line == start_line else 0
            e = end_char-1 if line == end_line else len(t)-1
            for i in range(s, e+1):
                labels[line][i] = f'I-{label}'

    return text, labels


def merge_into_words(text_by_char, all_labels_by_char):
    assert len(text_by_char) == len(
        all_labels_by_char), "Incorrect # of sentences!"

    N = len(text_by_char)

    text_by_word, all_labels_by_word = [], []

    for sentence_num in range(N):
        sentence_by_char = text_by_char[sentence_num]
        labels_by_char = all_labels_by_char[sentence_num]

        assert len(sentence_by_char) == len(
            labels_by_char), "Incorrect # of chars in sentence!"
        S = len(sentence_by_char)

        if labels_by_char == (['O'] * len(sentence_by_char)):
            sentence_by_word = ''.join(sentence_by_char).split()
            labels_by_word = ['O'] * len(sentence_by_word)
        else:
            sentence_by_word, labels_by_word = [], []
            text_chunks, labels_chunks = [], []
            s = 0
            for i in range(S):
                if i == S-1:
                    text_chunks.append(sentence_by_char[s:])
                    labels_chunks.append(labels_by_char[s:])
                elif labels_by_char[i] == 'O':
                    continue
                else:
                    if i > 0 and labels_by_char[i-1] == 'O':
                        text_chunks.append(sentence_by_char[s:i])
                        labels_chunks.append(labels_by_char[s:i])
                        s = i
                    if labels_by_char[i+1] == 'O' or labels_by_char[i+1][2:] != labels_by_char[i][2:]:
                        text_chunks.append(sentence_by_char[s:i+1])
                        labels_chunks.append(labels_by_char[s:i+1])
                        s = i+1

            for text_chunk, labels_chunk in zip(text_chunks, labels_chunks):
                assert len(text_chunk) == len(
                    labels_chunk), "Bad Chunking (len)"
                assert len(text_chunk) > 0, "Bad chunking (len 0)" + \
                    str(text_chunks) + str(labels_chunks)

                labels_set = set(labels_chunk)
                assert labels_set == set(['O']) or (len(labels_set) <= 3 and 'O' not in labels_set), (
                    ("Bad chunking (contents) %s" % ', '.join(labels_set)) +
                    str(text_chunks) + str(labels_chunks)
                )

                text_chunk_by_word = ''.join(text_chunk).split()
                W = len(text_chunk_by_word)
                if W == 0:
                    #                     assert labels_set == set(['O']), "0-word chunking and non-0 label!" + str(
                    #                         text_chunks) + str(labels_chunks
                    #                     )
                    continue

                if labels_chunk[0] == 'O':
                    labels_chunk_by_word = ['O'] * W
                elif W == 1:
                    labels_chunk_by_word = [labels_chunk[0]]
                elif W == 2:
                    labels_chunk_by_word = [labels_chunk[0], labels_chunk[-1]]
                else:
                    labels_chunk_by_word = [
                        labels_chunk[0]
                    ] + [labels_chunk[1]] * (W - 2) + [
                        labels_chunk[-1]
                    ]

                sentence_by_word.extend(text_chunk_by_word)
                labels_by_word.extend(labels_chunk_by_word)

        assert len(sentence_by_word) == len(
            labels_by_word), "Incorrect # of words in sentence!"

        if len(sentence_by_word) == 0:
            continue

        text_by_word.append(sentence_by_word)
        all_labels_by_word.append(labels_by_word)
    return text_by_word, all_labels_by_word


def process_all_xml(folder=None, outfile="test.csv"):
    xmls = glob(os.path.join(folder, "*.xml"))

    df = pd.DataFrame()
    for i, file in enumerate(xmls):
        text, labels = process_xml(i, file)
        if text and labels:
            text_by_word, labels_by_word = merge_into_words(text, labels)
            print("a")
        # df = df.append(data)
    df.to_csv(outfile)


# %%
process_all_xml("bert/train")
