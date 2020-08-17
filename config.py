from transformers import AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
MODEL_PATH = "Bio_ClinicalBERT.mdl"
TRAINING_FILE = "processed/ner/i2b2_train.csv"
TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
TESTING_FILE = "processed/ner/i2b2_test.csv"
