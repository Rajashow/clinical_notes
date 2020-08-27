from transformers import AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "Bio_ClinicalBERT.mdl"
TRAINING_FILE = "processed/class/i2b2_train.csv"
TOKENIZER = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
TESTING_FILE = "processed/class/i2b2_test.csv"
LEN_TO_SENTENCE_LOAD_FACTOR = 1.5
