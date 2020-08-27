# %%
import csv
from glob import glob
import os
import xml.etree.ElementTree as ET
import re
from numpy.lib.function_base import extract, iterable
from tqdm import tqdm
from collections import Counter, OrderedDict
import spacy

import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

# Core models
import en_core_sci_sm
import en_core_sci_lg

# NER specific models
import en_ner_craft_md
import en_ner_bc5cdr_md
import en_ner_jnlpba_md
import en_ner_bionlp13cg_md
import en_core_med7_lg
from negspacy.negation import Negex

from spacy import displacy

# %%
TEXT_TAG = "TEXT"
MODELS = {
    "nlp_bc": (en_ner_bc5cdr_md.load(), "CHEMICAL"),
    "nlp_bi": (en_core_med7_lg.load(), "DRUG"),
    "med7": (en_ner_bionlp13cg_md.load(), "SIMPLE_CHEMICAL"),
}
# %%
def show_medical_abbreviation(model, document):
    """
    This function detects and resolves medical abbreviations in word entities

    Parameters:
         model(module): A pretrained biomedical model from ScispaCy(https://allenai.github.io/scispacy/)
         document(str): Document to be processed

    Returns: List of unique abbreviations and their resolution
     """
    nlp = model.load()
    abbreviation_pipe = AbbreviationDetector(nlp)
    nlp.add_pipe(abbreviation_pipe)
    doc = nlp(document)
    abbreviated = list(
        set([f"{abrv}  {abrv._.long_form}" for abrv in doc._.abbreviations])
    )  # list is set to ensure only unique values are returned
    return abbreviated


def unified_medical_language_entity_linker(model, document):
    """
    This function links named entities to the Unified Medical Language System UMLS (https://www.nlm.nih.gov/research/umls/)

    Parameters:
         model(module): A pretrained biomedical model from ScispaCy(https://allenai.github.io/scispacy/)
         document(str): Document to be processed

    Returns: Attributes of Named entities accessible in the Unified Medical Language System database
     """
    nlp = model.load()
    linker = UmlsEntityLinker(
        k=10, max_entities_per_mention=2
    )  # parameters are tunable
    nlp.add_pipe(linker)
    doc = nlp(document)
    entity = doc.ents
    entity = [
        str(item) for item in entity
    ]  # convert each entity tuple to list of strings
    entity = str(OrderedDict.fromkeys(entity))  # returns unique entities only
    entity = nlp(entity).ents  # convert unique entities back to '.ents' object
    for entity in entity:
        for umls_ent in entity._.umls_ents:
            print("Entity Name:", entity)
            Concept_Id, Score = umls_ent
            print("Concept_Id = {} Score = {}".format(Concept_Id, Score))
            print(linker.umls.cui_to_entity[umls_ent[0]])


def extract_drug_info(i: int, file: str, vote: int = 2, med7_only: bool = True):
    # TODO make model generation inst per args not per call
    xml_parsed = ET.parse(file)
    # get text and tags
    clinical_note = xml_parsed.find(TEXT_TAG).text
    if vote:

        drug_counter = Counter()
        for nlp, drug_tag in MODELS.items():
            entries = set()
            if not nlp.has_pipe("Negex"):
                negex = Negex(nlp, language="en_clinical_sensitive")
                nlp.add_pipe(negex, last=True)
            doc = nlp(clinical_note)
            for e in doc.ents:
                if e.ent_type_ == drug_tag:
                    entries.add(f"{e.text}&{e._.negex}")
            for entry in entries:
                drug_counter[entry] += 1

        return [drug for drug, number in drug_counter.items() if number >= vote]
    elif med7_only:
        med7, drug_tag = MODELS.get("med7")
        if not med7.has_pipe("Negex"):
            negex = Negex(med7, language="en_clinical_sensitive")
            med7.add_pipe(negex, last=True)
        return list(
            set(
                (
                    #
                    f"{e.text}&{ e._.negex}"
                    for e in med7(clinical_note)
                    if e.ent_type_ == drug_tag
                    and re.sub(r"[^\w\s]", "", e.text).strip()
                )
            )
        )
    else:
        raise NotImplementedError("UMLS LINKED NOT ADDED YET")


def get_row(arg, max_number: int = 0) -> tuple:
    return (*arg, *(None for _ in range(len(arg) - max_number)))


def process_all_xml(
    folder: str = None, outdir: str = "", out_modifer: str = ""
) -> None:
    """
    process_all_xml prcoess all xml 2 csv

    for a folder get all xmls and place them in a csv in the outdir with the out_modifer

    Parameters
    ----------
    folder : str/path, optional
        where to get xmls, by default None
    outdir : str, optional
        outdir, by default ""
    out_modifer : str, optional
        modifer to add the csv file, by default ""
    """
    xmls = glob(os.path.join(folder, "*.xml"))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(os.path.join(outdir, f"i2b2{out_modifer}.csv"), "w") as file:
        csv_file = csv.writer(file,)

        drugs_by_file = [
            (
                os.path.basename(file),
                *extract_drug_info(i, file, vote=None, med7_only=True),
            )
            for i, file in tqdm(
                enumerate(xmls),
                total=len(xmls),
                desc=f"Processing files from {folder}: ",
            )
        ]

        max_numb_of_drugs = max((len(elem) - 1 for elem in drugs_by_file))
        header = ["filename", *(f"drug#{i}" for i in range(1, max_numb_of_drugs + 1))]

        csv_file.writerow(header)
        csv_file.writerows(
            [get_row(elem, max_number=max_numb_of_drugs) for elem in drugs_by_file]
        )

        csv_file.writerows([])
    pass


# %%
OUTDIR = "processed/spacy"
TRAIN_DIR = "new_bert/train"

process_all_xml(TRAIN_DIR, outdir=OUTDIR, out_modifer="_drugs_train")

# %%
