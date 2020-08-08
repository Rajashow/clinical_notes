# BIO BERT

based on <https://github.com/EmilyAlsentzer/clinicalBERT>

## Download the bert weights

<https://github.com/naver/biobert-pretrained>

can use any weights i think but i used the BioBERT-Base v1.0 (+ PubMed 200K + PMC 270K)

## Install libs

```conda create --name bert_transformers --file 'requirements.txt'```

## get the i2b2 data from dropbox

.....

## Run preprocess.py

modify the values in the file

```python
OUTDIR = "./processed"
TRAIN_DIRS = ['training-RiskFactors-Gold-Set1/', ]
TEST_DIRS  =['testing-RiskFactors-Gold']
```

### Run it with python

>python preprocess.py

## Run the run i2b2 and get metrics

modify the values in the file run_i2b2.sh

``` bash
BERT_DIR=/biobert_large_v1.1_pubmed/biobert_large
NER_DIR=/processed/ner
```

### Eun the bash script

>./run_i2b2.sh

## Extra

based on <https://github.com/EmilyAlsentzer/clinicalBERT>

## Future

[] use the transformers version
