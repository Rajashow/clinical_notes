from transformers import AutoTokenizer

MAX_LEN = 4294
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "emilyalsentzer/Bio_ClinicalBERT"
TRAINING_FILE = "processed/ner/i2b2_train.csv"
TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
